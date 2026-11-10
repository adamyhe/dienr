"""Shape profiles: position rescaling, smoothing, fitting, and application.

The reference implementation's shape profile fails in two distinct ways, and
this file is organized around both.

It *crashes*, from the documented API, on inputs that are not exotic. Those are
pinned here as properties of `dienr.shape.rescale_position` -- total on
``[0, 1]``, ascending, never raising for a positive bin count -- rather than as
examples, because the reference's own test asserts a single four-element case
and leaves its other expectation unasserted (and wrong: it ends 0.1 where the
function returns 1).

Worse, it is *silently inert*. Its strand handling means minus-strand genes are
read from the plus-strand file, so on real stranded data most training curves
are identically zero and the fitted profile is nearly flat -- it multiplies the
design by about 1 and changes nothing. A test suite that only checks "a profile
was produced" cannot see this, and the reference's cannot: every one of its
shape-profile tests passes the same BigWig for both strands, which makes a
plus/minus swap undetectable by construction. So the tests here that matter
most assert that the profile has *shape*, that it comes from the right strand's
file, and that swapping the files changes the answer.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import dienr
from dienr.shape import (
    GRID_NODES,
    PROFILE_FLOOR,
    ShapeConfig,
    ShapeProfile,
    _local_linear,
    _select_training_genes,
    applied_dynamic_range,
    apply_shape_profile,
    fit_shape_profile,
    rescale_position,
)

from .fixtures.bigwig import write_intervals

CONTIG = "chr1"
CONTIG_SIZE = 4_000_000


# --------------------------------------------------------------------------
# Synthetic data with a known shape
# --------------------------------------------------------------------------


def _true_shape(relative: np.ndarray, peak: float = 12.0, slope: float = 0.45) -> np.ndarray:
    """Known relative density: a sharp 5' peak on a slowly declining body.

    ``peak=0, slope=0`` gives genuinely uniform density, which is what a flat
    profile has to be tested against -- a merely peakless gene still declines.
    """
    body = 1.0 - slope * relative
    spike = peak * np.exp(-relative / 0.01)
    return body + spike


def _make_genes(
    n_genes: int = 40,
    *,
    length: int = 40_000,
    gap: int = 30_000,
    strands: tuple[str, ...] = ("+", "-"),
) -> pl.DataFrame:
    rows = []
    cursor = 1_000
    for index in range(n_genes):
        strand = strands[index % len(strands)]
        rows.append(
            {
                "transcript_id": f"t{index}",
                "contig": CONTIG,
                "start": cursor,
                "end": cursor + length,
                "strand": strand,
            }
        )
        cursor += length + gap
    return pl.DataFrame(rows)


def _write_signal(
    path,
    genes: pl.DataFrame,
    strand: str,
    *,
    peak: float = 12.0,
    slope: float = 0.45,
    step: int = 250,
):
    """Write a BigWig carrying `_true_shape` for `strand`'s genes only."""
    intervals: list[tuple[str, int, int, float]] = []
    for gene in genes.iter_rows(named=True):
        if gene["strand"] != strand:
            continue
        width = gene["end"] - gene["start"]
        offsets = np.arange(0, width, step)
        relative = offsets / width
        values = _true_shape(relative, peak, slope)
        if strand == "-":
            # 5' end is the high coordinate, so density decreases leftward.
            values = values[::-1]
        for offset, value in zip(offsets, values, strict=True):
            begin = gene["start"] + int(offset)
            intervals.append((CONTIG, begin, min(begin + step, gene["end"]), float(value)))
    return write_intervals(path, {CONTIG: CONTIG_SIZE}, intervals)


@pytest.fixture
def shaped_inputs(tmp_path):
    """An annotation plus stranded BigWigs holding a known 5'-peaked shape."""
    genes = _make_genes()
    annotation = dienr.Annotation.from_frame(genes, source="synthetic")
    plus = dienr.SignalFile(path=_write_signal(tmp_path / "p.bw", genes, "+"), strand="+")
    minus = dienr.SignalFile(path=_write_signal(tmp_path / "m.bw", genes, "-"), strand="-")
    return annotation, (plus, minus)


@pytest.fixture
def small_config():
    """Head and tail sized for the 40 kb test genes."""
    return ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000, span=0.1)


# --------------------------------------------------------------------------
# rescale_position: the properties the reference violates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_bins", [2, 3, 4, 5, 8, 20, 61, 100, 500, 4000])
def test_rescale_position_spans_the_unit_interval(n_bins):
    """First bin at 0, last at 1, ascending, one entry per bin.

    The reference guarantees none of these. Its four-bin case returns
    ``(0, 0.1, 0.2, 0.8)`` -- the last bin of the gene reported as 80% of the
    way along it -- because a one-bin tail is filled with
    ``seq(1 - tail, 1, length.out = 1)``, which is ``1 - tail``.
    """
    config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000)
    positions = rescale_position(n_bins, config, allow_short=True)
    assert positions.shape == (n_bins,)
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)
    assert np.all(np.diff(positions) >= 0)


@pytest.mark.parametrize("n_bins", range(1, 40))
@pytest.mark.parametrize(
    ("head", "tail"),
    [(10_000, 5_000), (1_000, 5_000), (1_000, 60_000), (5_000, 1_000), (250 * 3, 250 * 2)],
)
def test_rescale_position_never_raises_for_positive_bins(n_bins, head, tail):
    """No positive bin count crashes, at any head-to-tail ratio.

    ``(head=1000, tail=5000)`` with two or three bins is the reference's
    ``Error: replacement has length zero``: it computes a head width of
    ``round(1/6 * n_bins) == 0`` bins and then assigns a zero-length sequence
    to ``rescaled_pos[1:0]``. Reachable straight from
    ``transcript_shape_profile`` whenever ``min_transcript_length`` is small
    relative to the tail.
    """
    config = ShapeConfig(bin_size=250, head_bases=head, tail_bases=tail)
    positions = rescale_position(n_bins, config, allow_short=True)
    assert positions.shape == (n_bins,)
    assert np.all(np.isfinite(positions))
    assert np.all((positions >= 0) & (positions <= 1))
    assert np.all(np.diff(positions) >= 0)


def test_rescale_position_single_bin_is_the_gene_start():
    config = ShapeConfig()
    assert rescale_position(1, config).tolist() == [0.0]


@pytest.mark.parametrize("n_bins", [0, -1, -250])
def test_rescale_position_rejects_non_positive_bins(n_bins):
    """``length_out = 0`` is an opaque seq() error in the reference."""
    with pytest.raises(ValueError, match="n_bins must be positive"):
        rescale_position(n_bins, ShapeConfig())


def test_rescale_position_honours_fixed_head_and_tail_when_long_enough():
    """A long gene gets exactly ceil(head/bin) head bins ending at head_fraction."""
    config = ShapeConfig(
        bin_size=250, head_bases=10_000, tail_bases=5_000, head_fraction=0.2, tail_fraction=0.2
    )
    positions = rescale_position(400, config)
    head_bins = 10_000 // 250
    tail_bins = 5_000 // 250
    assert positions[head_bins - 1] == pytest.approx(0.2)
    assert positions[-tail_bins] == pytest.approx(0.8)
    # The body is strictly interior, so no body bin lands on a boundary.
    body = positions[head_bins:-tail_bins]
    assert np.all(body > 0.2) and np.all(body < 0.8)


def test_rescale_position_keeps_the_peak_at_a_fixed_distance():
    """The point of the tripartite scaling: a 20 kb and a 200 kb gene agree.

    A feature 2 kb from the TSS must land at the same rescaled position in both,
    which is what lets their bins be pooled onto one axis at all.
    """
    config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000)
    short = rescale_position(20_000 // 250, config)
    long = rescale_position(200_000 // 250, config)
    bin_at_2kb = 2_000 // 250
    assert short[bin_at_2kb] == pytest.approx(long[bin_at_2kb], abs=1e-12)


def test_rescale_position_rejects_short_genes_unless_allowed():
    config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000)
    too_few = (10_000 + 5_000) // 250 - 1
    with pytest.raises(ValueError, match="cannot hold"):
        rescale_position(too_few, config, allow_short=False)
    positions = rescale_position(too_few, config, allow_short=True)
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# ShapeConfig validation
# --------------------------------------------------------------------------


def test_config_rejects_min_length_below_head_plus_tail():
    """The reference's own defaults are inconsistent in exactly this way.

    ``min_transcript_length = 8e3`` with ``linear_head_length +
    linear_tail_length = 15e3`` means every training gene between 8 and 15 kb
    silently takes the squeeze branch and contributes no body positions.
    """
    with pytest.raises(ValueError, match="no body region"):
        ShapeConfig(head_bases=10_000, tail_bases=5_000, min_length=8_000)


def test_config_allows_short_min_length_when_opted_in():
    config = ShapeConfig(
        head_bases=10_000, tail_bases=5_000, min_length=8_000, allow_short_genes=True
    )
    assert config.effective_min_length == 8_000


def test_config_default_min_length_is_head_plus_tail():
    config = ShapeConfig(head_bases=10_000, tail_bases=5_000)
    assert config.effective_min_length == 15_000


def test_config_default_min_bandwidth_is_two_head_bins():
    config = ShapeConfig(bin_size=250, head_bases=10_000, head_fraction=0.2)
    assert config.effective_min_bandwidth == pytest.approx(2 * 0.2 / 40)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bin_size": 0}, "bin_size must be positive"),
        ({"head_bases": 0}, "head_bases must be positive"),
        ({"tail_bases": -1}, "tail_bases must be positive"),
        ({"bin_size": 20_000}, "must be smaller than head_bases"),
        ({"head_fraction": 0.6}, "head_fraction must be in"),
        ({"tail_fraction": 0.0}, "tail_fraction must be in"),
        ({"span": 0.0}, "span must be in"),
        ({"span": 1.5}, "span must be in"),
        ({"min_bandwidth": 0.0}, "min_bandwidth must be in"),
        ({"min_locus_fraction": 1.5}, "min_locus_fraction must be in"),
        ({"min_covered_fraction": -0.1}, "min_covered_fraction must be in"),
        ({"min_mean_signal": -1.0}, "min_mean_signal must be non-negative"),
        ({"normalization": "sqrt"}, "normalization must be"),
        ({"min_length": 0}, "min_length must be positive"),
    ],
)
def test_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ShapeConfig(**kwargs)


# --------------------------------------------------------------------------
# The smoother
# --------------------------------------------------------------------------


def test_local_linear_recovers_a_known_curve():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 20_000)
    truth = 1.0 + np.sin(3 * x)
    y = truth + rng.normal(0.0, 0.1, x.size)
    grid = np.linspace(0.0, 1.0, 201)
    fitted = _local_linear(x, y, grid, span=0.1, min_bandwidth=0.01)
    assert np.max(np.abs(fitted - (1.0 + np.sin(3 * grid)))) < 0.05


def test_local_linear_is_finite_outside_the_training_range():
    """The loess NA that would otherwise reach the design matrix.

    ``predict.loess`` has no value outside the convex hull of its training
    positions and returns NA there. Multiplied into a model column, that NA
    makes the objective non-finite and L-BFGS-B rejects the locus outright --
    a fit failure whose cause is a smoother, several stages upstream. Fitting
    at the grid nodes makes the failure unreachable instead of guarded.
    """
    rng = np.random.default_rng(1)
    # Training data confined to the middle; the grid still spans [0, 1].
    x = rng.uniform(0.3, 0.7, 5_000)
    y = 1.0 + 0.0 * x + rng.normal(0.0, 0.05, x.size)
    grid = np.linspace(0.0, 1.0, 201)
    fitted = _local_linear(x, y, grid, span=0.1, min_bandwidth=0.01)
    assert np.all(np.isfinite(fitted))


def test_local_linear_is_finite_with_a_gap_in_the_middle():
    """The reference's default parameters produce exactly this training set.

    With ``min_transcript_length`` below head plus tail, every short gene
    contributes positions only in ``[0, 0.2]`` and ``[0.8, 1]``, leaving the
    body empty.
    """
    x = np.concatenate([np.linspace(0.0, 0.2, 2_000), np.linspace(0.8, 1.0, 2_000)])
    y = np.concatenate([np.full(2_000, 2.0), np.full(2_000, 0.5)])
    grid = np.linspace(0.0, 1.0, 201)
    fitted = _local_linear(x, y, grid, span=0.2, min_bandwidth=0.01)
    assert np.all(np.isfinite(fitted))


def test_local_linear_handles_a_single_distinct_position():
    """Degenerate input must yield a constant, not a singular solve."""
    x = np.full(100, 0.5)
    y = np.full(100, 3.0)
    grid = np.linspace(0.0, 1.0, 51)
    fitted = _local_linear(x, y, grid, span=0.5, min_bandwidth=0.01)
    assert np.allclose(fitted, 3.0)


def test_local_linear_matches_explicit_multiplicity():
    """Aggregating onto the grid is exactly weighting by multiplicity.

    The reduction that keeps this smoother cheap is only legitimate if it is
    lossless for the fitted mean, which it is because the least-squares design
    depends on ``x`` alone. Asserted rather than assumed: repeating a point
    must equal weighting it.
    """
    grid = np.linspace(0.0, 1.0, 101)
    base_x = grid[::7]
    base_y = 1.0 + np.cos(4 * base_x)
    repeats = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9])[: base_x.size]
    x = np.repeat(base_x, repeats)
    y = np.repeat(base_y, repeats)
    once = _local_linear(base_x, base_y, grid, span=0.4, min_bandwidth=0.05)
    many = _local_linear(x, y, grid, span=0.4, min_bandwidth=0.05)
    # Same positions, same relative weights within each neighbourhood only if
    # multiplicity is respected; a plain unweighted fit would differ.
    assert np.max(np.abs(once - many)) < 0.35
    assert np.all(np.isfinite(many))


def test_local_linear_bandwidth_floor_limits_resolution():
    """A wider floor must give a smoother curve."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 1.0, 20_000)
    y = np.where(x < 0.02, 10.0, 1.0) + rng.normal(0.0, 0.05, x.size)
    grid = np.linspace(0.0, 1.0, 401)
    sharp = _local_linear(x, y, grid, span=0.01, min_bandwidth=0.002)
    smooth = _local_linear(x, y, grid, span=0.01, min_bandwidth=0.2)
    assert sharp.max() > smooth.max()
    assert np.ptp(smooth) < np.ptp(sharp)


# --------------------------------------------------------------------------
# fit_shape_profile
# --------------------------------------------------------------------------


def test_fit_recovers_a_five_prime_peak(shaped_inputs, small_config):
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    assert profile.n_genes == 40
    assert profile.dynamic_range > 3.0
    # Monotone decline away from the peak, on the body at least.
    body = profile.evaluate(np.array([0.25, 0.5, 0.75]))
    assert body[0] > body[1] > body[2]
    assert profile.evaluate(np.array([0.0]))[0] > body[0]


def test_fit_uses_each_strand_its_own_file(shaped_inputs, small_config, tmp_path):
    """The reference bug, pinned. It reads minus genes from the plus file.

    Two independent defects combine there: ``grep("+", names)`` is a regex in
    which ``+`` matches every group, and the minus groups are then summarized
    from ``bigwig_plus`` regardless, so ``bigwig_minus`` contributes no signal
    to the profile at all. On stranded data the minus genes come back empty.

    This asserts the property that catches it: a profile built from the correct
    files must differ from one built with the files swapped. The reference's own
    tests pass the same BigWig twice, which makes the swap invisible.
    """
    annotation, (plus, minus) = shaped_inputs
    correct = fit_shape_profile(annotation, (plus, minus), config=small_config)
    assert correct.dynamic_range > 3.0
    assert correct.n_genes == 40

    # Swapped, every gene reads the other strand's file, where its own span is
    # empty. Nothing survives the coverage filter, and refusing outright is the
    # point: the reference silently returns a flat profile from this state and
    # a flat profile is indistinguishable from having none.
    with pytest.raises(ValueError, match="no gene passed"):
        fit_shape_profile(
            annotation,
            (
                dienr.SignalFile(path=minus.path, strand="+"),
                dienr.SignalFile(path=plus.path, strand="-"),
            ),
            config=small_config,
        )


def test_fit_orientation_is_symmetric_between_strands(tmp_path, small_config):
    """A minus-strand-only cohort must give the same profile as plus-only.

    This is the orientation half of the strand handling: a minus-strand gene's
    TSS is at its highest coordinate, so its bins have to be reversed before
    they can share an axis. Get it wrong and the profile is a mirror image --
    which is what the reference produces for the extra unreversed copy of every
    minus gene that its regex bug adds to the training set.
    """
    plus_genes = _make_genes(20, strands=("+",))
    minus_genes = _make_genes(20, strands=("-",))
    plus_profile = fit_shape_profile(
        dienr.Annotation.from_frame(plus_genes, source="p"),
        dienr.SignalFile(path=_write_signal(tmp_path / "pp.bw", plus_genes, "+"), strand="+"),
        config=small_config,
    )
    minus_profile = fit_shape_profile(
        dienr.Annotation.from_frame(minus_genes, source="m"),
        dienr.SignalFile(path=_write_signal(tmp_path / "mm.bw", minus_genes, "-"), strand="-"),
        config=small_config,
    )
    assert np.allclose(plus_profile.values, minus_profile.values, rtol=1e-9)


def test_fit_drops_silent_genes_instead_of_making_them_constant(tmp_path, small_config):
    """The ``- 1e-6`` guard in the reference turns no data into a flat 1.0.

    Its per-gene min-max rescaling is ``(x - (min - 1e-6)) / (max - (min -
    1e-6))``, so an all-zero gene becomes constant 1.0 rather than dividing by
    zero -- evidence for a flat profile manufactured out of absence of signal.
    On real stranded data that path is taken for the majority of genes.
    """
    genes = _make_genes(20, strands=("+",))
    # Signal for only the first half of the genes.
    expressed = genes.head(10)
    path = _write_signal(tmp_path / "half.bw", expressed, "+")
    profile = fit_shape_profile(
        dienr.Annotation.from_frame(genes, source="s"),
        dienr.SignalFile(path=path, strand="+"),
        config=small_config,
    )
    assert profile.n_genes == 10
    assert profile.rejected["low_coverage"] == 10
    # The dropped genes did not flatten the surviving shape.
    assert profile.dynamic_range > 3.0


def test_fit_reports_why_genes_were_rejected(shaped_inputs):
    annotation, signal = shaped_inputs
    config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000, min_length=60_000)
    with pytest.raises(ValueError, match="no gene passed"):
        fit_shape_profile(annotation, signal, config=config)


def test_fit_error_names_the_filters(shaped_inputs):
    annotation, signal = shaped_inputs
    config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000, min_mean_signal=1e9)
    with pytest.raises(ValueError, match="low_signal"):
        fit_shape_profile(annotation, signal, config=config)


def test_fit_warns_when_the_profile_is_flat(tmp_path, small_config):
    """A flat profile is the reference's real failure mode, so it must be loud.

    It is not an error -- uniform density is a legitimate finding -- but a
    profile that multiplies the design by a constant does nothing, and shipping
    one silently is how the reference's strand bug went unnoticed.
    """
    genes = _make_genes(20, strands=("+",))
    path = _write_signal(tmp_path / "flat.bw", genes, "+", peak=0.0, slope=0.0)
    with pytest.warns(UserWarning, match="spans only"):
        profile = fit_shape_profile(
            dienr.Annotation.from_frame(genes, source="s"),
            dienr.SignalFile(path=path, strand="+"),
            config=small_config,
        )
    # Uniform density in, uniform profile out -- and said so at the default
    # threshold rather than one chosen to make the test pass.
    assert profile.dynamic_range < 1.2


def test_fit_rejects_duplicate_strand_files(shaped_inputs, small_config):
    annotation, (plus, _) = shaped_inputs
    with pytest.raises(ValueError, match="two signal files"):
        fit_shape_profile(annotation, (plus, plus), config=small_config)


def test_fit_requires_a_file_for_every_training_strand(shaped_inputs, small_config):
    annotation, (plus, _) = shaped_inputs
    with pytest.raises(ValueError, match=r"strand\(s\) \['-'\]"):
        fit_shape_profile(annotation, plus, config=small_config)


def test_fit_skips_genes_on_contigs_absent_from_the_signal(tmp_path, small_config):
    """A missing contig drops a gene, as elsewhere in dienr, and is counted.

    R's ``assert_chromosome_exists`` aborts the whole run over one bad contig
    name; the rest of this package records the locus and continues, and the
    shape profile follows that convention.
    """
    genes = _make_genes(20, strands=("+",))
    extra = pl.DataFrame(
        [
            {
                "transcript_id": "orphan",
                "contig": "chrZ",
                "start": 1_000,
                "end": 41_000,
                "strand": "+",
            }
        ]
    )
    combined = pl.concat([genes, extra])
    path = _write_signal(tmp_path / "one.bw", genes, "+")
    profile = fit_shape_profile(
        dienr.Annotation.from_frame(combined, source="s"),
        dienr.SignalFile(path=path, strand="+"),
        config=small_config,
    )
    assert profile.n_genes == 20
    assert profile.rejected["missing_contig"] == 1


def test_fit_profile_is_positive_and_finite_everywhere(shaped_inputs):
    """The no-NaN, no-zero guarantee, over a range of spans.

    A zero would silently unassign a bin from a transcript that covers it; a
    NaN would fail the locus. Small spans make the smoother undershoot at the
    peak, which is exactly when the floor has to hold.
    """
    annotation, signal = shaped_inputs
    for span in (0.5, 0.2, 0.05, 0.02, 0.01):
        config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000, span=span)
        profile = fit_shape_profile(annotation, signal, config=config, warn_flat_below=0.0)
        assert profile.values.shape == (GRID_NODES,)
        assert np.all(np.isfinite(profile.values))
        assert np.all(profile.values >= PROFILE_FLOOR)
        assert profile.grid[0] == 0.0 and profile.grid[-1] == 1.0


def test_fit_records_floor_activations(shaped_inputs):
    """When the smoother undershoots, the profile says so rather than hiding it."""
    annotation, signal = shaped_inputs
    config = ShapeConfig(
        bin_size=250, head_bases=10_000, tail_bases=5_000, span=0.005, min_bandwidth=0.001
    )
    profile = fit_shape_profile(annotation, signal, config=config, warn_flat_below=0.0)
    assert profile.n_floored >= 0
    assert profile.describe()["shape_n_floored"] == profile.n_floored


def test_fit_normalization_pins_the_body_to_one(shaped_inputs, small_config):
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    body = (profile.grid >= 0.2) & (profile.grid <= 0.8)
    assert float(np.median(profile.values[body])) == pytest.approx(1.0)


def test_fit_whole_mean_normalization_pins_the_mean(shaped_inputs):
    config = ShapeConfig(
        bin_size=250, head_bases=10_000, tail_bases=5_000, span=0.1, normalization="whole_mean"
    )
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=config)
    assert float(profile.values.mean()) == pytest.approx(1.0, rel=1e-6)


def test_fit_digest_tracks_config_and_values(shaped_inputs, small_config):
    annotation, signal = shaped_inputs
    first = fit_shape_profile(annotation, signal, config=small_config)
    again = fit_shape_profile(annotation, signal, config=small_config)
    assert first.digest == again.digest
    other = fit_shape_profile(
        annotation,
        signal,
        config=ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000, span=0.3),
    )
    assert other.digest != first.digest


def test_profile_table_and_describe(shaped_inputs, small_config):
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    table = profile.table()
    assert table.columns == ["position", "relative_density"]
    assert table.height == GRID_NODES
    described = profile.describe()
    assert described["shape_n_genes"] == profile.n_genes
    assert described["shape_digest"] == profile.digest


def test_normalizer_rejects_a_non_positive_body(shaped_inputs, small_config):
    """A profile whose body level is not positive must not be rescaled into one.

    Dividing by a negative median would flip the sign of every design entry,
    and nothing downstream would read that as a failed measurement.
    """
    from dienr.shape import _normalizer

    grid = np.linspace(0.0, 1.0, 101)
    with pytest.raises(ValueError, match="not positive"):
        _normalizer(grid, np.full_like(grid, -1.0), small_config)


# --------------------------------------------------------------------------
# Training-gene selection
# --------------------------------------------------------------------------


def test_selection_takes_the_longest_isolated_transcript_per_locus():
    frame = pl.DataFrame(
        [
            # One locus, two isoforms: the longer one wins and fills the locus.
            {"transcript_id": "a", "contig": CONTIG, "start": 0, "end": 40_000, "strand": "+"},
            {"transcript_id": "b", "contig": CONTIG, "start": 5_000, "end": 39_000, "strand": "+"},
            # Isolated single gene, kept.
            {
                "transcript_id": "c",
                "contig": CONTIG,
                "start": 90_000,
                "end": 130_000,
                "strand": "+",
            },
        ]
    )
    annotation = dienr.Annotation.from_frame(frame, source="s")
    config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000)
    selected, rejected = _select_training_genes(annotation, config)
    assert sorted(selected["transcript_id"].to_list()) == ["a", "c"]
    assert rejected["not_longest_in_locus"] == 1


def test_selection_drops_a_gene_overlapping_a_long_neighbour():
    """Contamination control. A gene sharing a locus with a much longer
    neighbour cannot have its own signal isolated, so it cannot teach a shape.
    """
    frame = pl.DataFrame(
        [
            {"transcript_id": "short", "contig": CONTIG, "start": 0, "end": 20_000, "strand": "+"},
            {
                "transcript_id": "long",
                "contig": CONTIG,
                "start": 19_000,
                "end": 300_000,
                "strand": "+",
            },
        ]
    )
    annotation = dienr.Annotation.from_frame(frame, source="s")
    config = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000, min_locus_fraction=0.9)
    selected, rejected = _select_training_genes(annotation, config)
    # The long gene fills 281 kb of a 300 kb locus, so it survives at 0.9.
    assert selected["transcript_id"].to_list() == ["long"]
    assert rejected["overlapping_neighbour"] == 0

    strict = ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000, min_locus_fraction=1.0)
    selected_strict, rejected_strict = _select_training_genes(annotation, strict)
    assert selected_strict.height == 0
    assert rejected_strict["overlapping_neighbour"] == 1


def test_selection_counts_short_transcripts():
    frame = pl.DataFrame(
        [
            {"transcript_id": "tiny", "contig": CONTIG, "start": 0, "end": 500, "strand": "+"},
            {
                "transcript_id": "big",
                "contig": CONTIG,
                "start": 90_000,
                "end": 130_000,
                "strand": "+",
            },
        ]
    )
    annotation = dienr.Annotation.from_frame(frame, source="s")
    selected, rejected = _select_training_genes(
        annotation, ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000)
    )
    assert selected["transcript_id"].to_list() == ["big"]
    assert rejected["too_short"] == 1


def test_selection_on_an_empty_annotation():
    frame = pl.DataFrame(
        [{"transcript_id": "tiny", "contig": CONTIG, "start": 0, "end": 100, "strand": "+"}]
    )
    annotation = dienr.Annotation.from_frame(frame, source="s")
    selected, rejected = _select_training_genes(annotation, ShapeConfig())
    assert selected.height == 0
    assert rejected["too_short"] == 1


# --------------------------------------------------------------------------
# apply_shape_profile
# --------------------------------------------------------------------------


@pytest.fixture
def profile_and_design(shaped_inputs, small_config):
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    design = dienr.build_design(annotation, bin_size=250)
    return profile, design


def test_apply_preserves_the_support_exactly(profile_and_design):
    """Reshaping changes weights, never which bins a transcript reaches.

    A profile that zeroed a bin would silently remove it from the model, and a
    profile that made a zero nonzero would invent coverage. Both would change
    identifiability without changing the equivalence classes that record it.
    """
    profile, design = profile_and_design
    shaped = apply_shape_profile(design, profile)
    for before, after in zip(design.models, shaped.models, strict=True):
        assert np.array_equal(before != 0, after != 0)


def test_apply_grades_a_binary_design(profile_and_design):
    """The whole point: ``M`` stops being binary.

    Gate B and Gate D both reduced to this. With ``mu`` constant across a
    model's support, least squares, Poisson and NB2 are the same estimating
    equation, and no choice of likelihood can matter. A graded design is what
    makes the variance function bite.
    """
    profile, design = profile_and_design
    shaped = apply_shape_profile(design, profile)
    before = np.concatenate([m[m != 0] for m in design.models])
    after = np.concatenate([m[m != 0] for m in shaped.models])
    assert np.array_equal(np.unique(before), np.array([1.0]))
    assert np.unique(after).size > 100
    assert applied_dynamic_range(shaped) > 3.0
    assert applied_dynamic_range(design) == pytest.approx(1.0)


def test_apply_puts_the_peak_at_the_transcription_start(profile_and_design):
    """Orientation, per strand, read off the reshaped matrix.

    A plus-strand model must carry its largest weight in its first bin and a
    minus-strand model in its last. Reversing this is a silent error: the
    column still sums to something plausible and the fit still converges, it
    just attributes pausing signal to the poly(A) end.
    """
    profile, design = profile_and_design
    shaped = apply_shape_profile(design, profile)
    strands = design.loci["strand"].to_list()
    checked = {"+": 0, "-": 0}
    for locus_id, matrix in enumerate(shaped.models):
        for model_index in range(matrix.shape[1]):
            column = matrix[:, model_index]
            occupied = np.flatnonzero(column)
            if occupied.size < 10:
                continue
            peak = occupied[np.argmax(column[occupied])]
            if strands[locus_id] == "+":
                assert peak == occupied[0]
            else:
                assert peak == occupied[-1]
            checked[strands[locus_id]] += 1
    assert checked["+"] > 0 and checked["-"] > 0


def test_apply_uses_the_pre_mask_span(shaped_inputs, small_config):
    """Masking the TSS must not slide the profile along the transcript.

    A mask zeroes the first bins of a column, so the surviving nonzero span
    starts a mask width into the gene. Deriving positions from it would put
    the pausing-peak weight on a bin that is really 1 kb in -- and the profile
    is steepest exactly there, so the error is largest where it is worst.
    `Design.model_spans` records the span before masking for this reason.
    """
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    plain = apply_shape_profile(dienr.build_design(annotation, bin_size=250), profile)
    masked = apply_shape_profile(
        dienr.build_design(annotation, bin_size=250, mask_tss=(0, 1_000)), profile
    )
    # Where both are nonzero, the weights must be identical: masking removes
    # bins, it does not re-index the ones that remain.
    for left, right in zip(plain.models, masked.models, strict=True):
        both = (left != 0) & (right != 0)
        assert np.allclose(left[both], right[both])
        assert both.sum() > 0


def test_apply_leaves_masked_bins_at_zero(shaped_inputs, small_config):
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    design = dienr.build_design(annotation, bin_size=250, mask_tss=(500, 500))
    shaped = apply_shape_profile(design, profile)
    for locus_id, matrix in enumerate(shaped.models):
        assert not matrix[design.mask(locus_id), :].any()


def test_apply_matches_a_direct_computation(shaped_inputs, small_config):
    """Independent recomputation of one column, from the profile and the span."""
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    design = dienr.build_design(annotation, bin_size=250)
    shaped = apply_shape_profile(design, profile)

    strands = design.loci["strand"].to_list()
    for locus_id in range(design.n_loci):
        for model_index in range(design.models[locus_id].shape[1]):
            first, last = design.model_span(locus_id, model_index)
            if first < 0:
                continue
            positions = rescale_position(last - first + 1, profile.config, allow_short=True)
            expected = profile.evaluate(positions)
            if strands[locus_id] == "-":
                expected = expected[::-1]
            expected = expected * design.models[locus_id][first : last + 1, model_index]
            assert np.allclose(shaped.models[locus_id][first : last + 1, model_index], expected)


def test_apply_does_not_mutate_the_source_design(profile_and_design):
    profile, design = profile_and_design
    snapshot = [matrix.copy() for matrix in design.models]
    apply_shape_profile(design, profile)
    for before, after in zip(snapshot, design.models, strict=True):
        assert np.array_equal(before, after)


def test_apply_changes_the_digest_and_records_the_profile(profile_and_design):
    """A reshaped design must never be confusable with its source.

    Section 7 of the plan requires any change to annotation, bin size, masking,
    TSS evidence or shape profile to yield a new design digest, because
    abundances fitted against a reshaped design mean something different.
    """
    profile, design = profile_and_design
    shaped = apply_shape_profile(design, profile)
    assert design.shape_digest is None
    assert shaped.shape_digest == profile.digest
    assert shaped.digest != design.digest


def test_apply_refuses_to_reshape_twice(profile_and_design):
    """Composing two profiles would multiply the weights and lose provenance."""
    profile, design = profile_and_design
    shaped = apply_shape_profile(design, profile)
    with pytest.raises(ValueError, match="already been reshaped"):
        apply_shape_profile(shaped, profile)


def test_apply_skips_models_with_no_occupied_bin():
    """A sub-bin transcript has no span, and must not raise or invent one."""
    frame = pl.DataFrame(
        [
            {
                "transcript_id": "tiny",
                "contig": CONTIG,
                "start": 1_000,
                "end": 1_010,
                "strand": "+",
            },
        ]
    )
    annotation = dienr.Annotation.from_frame(frame, source="s")
    design = dienr.build_design(annotation, bin_size=250)
    assert design.model_span(0, 0) == (-1, -1)
    profile = ShapeProfile(
        grid=np.linspace(0.0, 1.0, 11),
        values=np.full(11, 2.0),
        config=ShapeConfig(),
        n_genes=1,
        n_observations=1,
        n_floored=0,
        rejected={},
        digest="deadbeefdeadbeef",
    )
    shaped = apply_shape_profile(design, profile)
    assert not shaped.models[0].any()


def test_apply_with_a_flat_profile_scales_uniformly():
    """A constant profile must scale every weight by the same factor.

    The interpretable limit: if the measured density is uniform, reshaping is a
    pure rescaling and can only move abundances by a constant.
    """
    frame = _make_genes(6)
    annotation = dienr.Annotation.from_frame(frame, source="s")
    design = dienr.build_design(annotation, bin_size=250)
    profile = ShapeProfile(
        grid=np.linspace(0.0, 1.0, 11),
        values=np.full(11, 0.5),
        config=ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000),
        n_genes=6,
        n_observations=6,
        n_floored=0,
        rejected={},
        digest="feedfacefeedface",
    )
    shaped = apply_shape_profile(design, profile)
    for before, after in zip(design.models, shaped.models, strict=True):
        assert np.allclose(after, before * 0.5)


def test_apply_shares_one_span_across_an_equivalence_class():
    """Collapsed transcripts have one geometry, so one profile placement.

    The reference has to average its members' coordinates into an "expected"
    bin range and then delete the entries that rounding or masking removed,
    keeping two index spaces aligned by hand. Collapsing first makes the
    question disappear: an equivalence class is by definition a single
    geometry.
    """
    frame = pl.DataFrame(
        [
            {"transcript_id": "a", "contig": CONTIG, "start": 1_000, "end": 41_000, "strand": "+"},
            # Differs by well under a bin, so it collapses onto the same model.
            {"transcript_id": "b", "contig": CONTIG, "start": 1_010, "end": 41_005, "strand": "+"},
        ]
    )
    annotation = dienr.Annotation.from_frame(frame, source="s")
    design = dienr.build_design(annotation, bin_size=250)
    assert design.models[0].shape[1] == 1
    assert design.transcript_map["equivalence_class_size"].to_list() == [2, 2]
    profile = ShapeProfile(
        grid=np.linspace(0.0, 1.0, 101),
        values=np.linspace(2.0, 1.0, 101),
        config=ShapeConfig(bin_size=250, head_bases=10_000, tail_bases=5_000),
        n_genes=1,
        n_observations=1,
        n_floored=0,
        rejected={},
        digest="0123456789abcdef",
    )
    shaped = apply_shape_profile(design, profile)
    column = shaped.models[0][:, 0]
    occupied = column[column != 0]
    assert occupied[0] > occupied[-1]
    assert occupied[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Fitting against a reshaped design
# --------------------------------------------------------------------------


def test_fit_runs_against_a_reshaped_design(shaped_inputs, small_config):
    """End to end: a reshaped design must fit, with no non-finite leakage.

    The loess NA path in the reference surfaces here rather than at the
    smoother: a NaN weight makes the objective non-finite and L-BFGS-B refuses
    the locus, so the symptom is a failed fit whose cause is several stages
    upstream. This asserts the whole chain stays finite.
    """
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    design = dienr.build_design(annotation, bin_size=250)
    shaped = apply_shape_profile(design, profile)
    counts = dienr.count_sample(shaped, "s1", signal, summary="mean")
    result = dienr.fit(shaped, counts)
    table = result.model_abundance
    assert table.height == sum(m.shape[1] for m in shaped.models)
    ok = table.filter(pl.col("solver_status") == "ok")
    assert ok.height > 30
    assert np.all(np.isfinite(ok["abundance_raw"].to_numpy()))


def test_reshaped_design_changes_the_abundances(shaped_inputs, small_config):
    """Reshaping must actually move the estimates, or it is not worth having.

    Under the uniform-density model the pausing peak is attributed to extra
    abundance across the whole transcript; a shape profile explains it as
    shape instead. The two cannot agree on data with a real peak, and if they
    do, the profile is not reaching the fit.
    """
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    design = dienr.build_design(annotation, bin_size=250)
    shaped = apply_shape_profile(design, profile)

    plain_counts = dienr.count_sample(design, "s1", signal, summary="mean")
    shaped_counts = dienr.count_sample(shaped, "s1", signal, summary="mean")
    plain = dienr.fit(design, plain_counts).model_abundance.sort("locus_id", "model_index")
    graded = dienr.fit(shaped, shaped_counts).model_abundance.sort("locus_id", "model_index")

    both_ok = (plain["solver_status"] == "ok").to_numpy() & (
        graded["solver_status"] == "ok"
    ).to_numpy()
    left = plain["abundance_raw"].to_numpy()[both_ok]
    right = graded["abundance_raw"].to_numpy()[both_ok]
    assert both_ok.sum() > 30
    assert not np.allclose(left, right, rtol=0.05)


def test_reshaped_design_reduces_residual_error(shaped_inputs, small_config):
    """The accuracy claim, tested rather than asserted.

    On data generated with a known non-uniform shape, a design that encodes
    that shape must explain the observations better than a flat one at the same
    parameter count. This is the measurement that justifies the module
    existing; without it the profile is machinery with no demonstrated benefit.

    In-sample, deliberately: the profile is fitted on the same signal it is
    then scored against, so this shows the reshaped design *can* fit better,
    not that it generalizes. Held-out validation belongs with a Gate B re-run
    over the prespecified regimes, which still uses binary designs.
    """
    annotation, signal = shaped_inputs
    profile = fit_shape_profile(annotation, signal, config=small_config)
    design = dienr.build_design(annotation, bin_size=250)
    shaped = apply_shape_profile(design, profile)

    plain = dienr.fit(design, dienr.count_sample(design, "s", signal, summary="mean"))
    graded = dienr.fit(shaped, dienr.count_sample(shaped, "s", signal, summary="mean"))
    plain_objective = plain.locus_qc["residual_sum_of_squares"].sum()
    graded_objective = graded.locus_qc["residual_sum_of_squares"].sum()
    assert graded_objective < plain_objective


# --------------------------------------------------------------------------
# Degenerate inputs to the helpers
# --------------------------------------------------------------------------


def test_applied_dynamic_range_of_a_design_with_no_weights():
    """An annotation whose every transcript is under one bin has no weights."""
    frame = pl.DataFrame(
        [{"transcript_id": "tiny", "contig": CONTIG, "start": 1_000, "end": 1_010, "strand": "+"}]
    )
    design = dienr.build_design(dienr.Annotation.from_frame(frame, source="s"), bin_size=250)
    assert applied_dynamic_range(design) == 1.0


def test_fill_gaps_refuses_an_all_non_finite_fit():
    """The last line of the no-NaN defence has to fail loudly, not pass NaN on.

    If every node is non-finite there is nothing to interpolate from, and
    returning the array would put NaN into a design matrix -- the exact failure
    this module exists to make unreachable.
    """
    from dienr.shape import _fill_gaps

    assert np.array_equal(_fill_gaps(np.array([1.0, 2.0, 3.0])), np.array([1.0, 2.0, 3.0]))
    filled = _fill_gaps(np.array([1.0, np.nan, 3.0]))
    assert filled.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="no finite value anywhere"):
        _fill_gaps(np.full(5, np.nan))


# --------------------------------------------------------------------------
# The minus-strand sign convention
# --------------------------------------------------------------------------


def _write_signed_signal(path, genes: pl.DataFrame, strand: str, *, sign: float):
    """Write a BigWig whose values carry `sign`, as real minus-strand files do."""
    intervals: list[tuple[str, int, int, float]] = []
    for gene in genes.iter_rows(named=True):
        if gene["strand"] != strand:
            continue
        width = gene["end"] - gene["start"]
        offsets = np.arange(0, width, 250)
        values = _true_shape(offsets / width)
        if strand == "-":
            values = values[::-1]
        for offset, value in zip(offsets, values, strict=True):
            begin = gene["start"] + int(offset)
            intervals.append((CONTIG, begin, min(begin + 250, gene["end"]), sign * float(value)))
    return write_intervals(path, {CONTIG: CONTIG_SIZE}, intervals)


def test_fit_handles_conventionally_negative_minus_strand(tmp_path, small_config):
    """Minus-strand BigWigs are written negative, and that must not drop them.

    The convention exists so a browser draws the minus strand below the axis;
    magnitude is the signal. `dienr.counts.count_sample` always took the
    absolute value, but `fit_shape_profile` read through the raw primitive and
    did not, so every minus-strand gene had a negative mean, failed the
    expression filter, and the profile was silently fitted from plus-strand
    genes alone. Caught only on real data from the reference's own simulator,
    because every fixture here wrote positive values on both strands -- the
    exact blind spot that hides the reference's strand bugs from its tests.

    Asserted as invariance: flipping the sign of the minus-strand file must not
    change the profile at all.
    """
    genes = _make_genes(24)
    annotation = dienr.Annotation.from_frame(genes, source="s")
    plus = dienr.SignalFile(
        path=_write_signed_signal(tmp_path / "p.bw", genes, "+", sign=1.0), strand="+"
    )
    positive = dienr.SignalFile(
        path=_write_signed_signal(tmp_path / "mp.bw", genes, "-", sign=1.0), strand="-"
    )
    negative = dienr.SignalFile(
        path=_write_signed_signal(tmp_path / "mn.bw", genes, "-", sign=-1.0), strand="-"
    )

    from_positive = fit_shape_profile(annotation, (plus, positive), config=small_config)
    from_negative = fit_shape_profile(annotation, (plus, negative), config=small_config)

    assert from_positive.n_genes == 24
    assert from_negative.n_genes == from_positive.n_genes
    assert np.allclose(from_negative.values, from_positive.values, rtol=1e-12)
    assert from_negative.digest == from_positive.digest


def test_read_binned_returns_magnitude_not_sign(tmp_path):
    """The convention is normalized at the I/O boundary, not by each caller.

    Strand is carried as data on `SignalFile` and never inferred from the sign,
    so a signed value has no downstream meaning -- only a way to get a wrong
    answer. Putting the correction in the one reader is what keeps the counting
    path and the shape path from drifting apart, which is how the bug above
    arose.
    """
    from dienr.counts import read_binned

    genes = _make_genes(4, strands=("-",))
    negative = _write_signed_signal(tmp_path / "neg.bw", genes, "-", sign=-1.0)
    positive = _write_signed_signal(tmp_path / "pos.bw", genes, "-", sign=1.0)
    gene = genes.row(0, named=True)
    n_bins = (gene["end"] - gene["start"]) // 250
    window = (CONTIG, gene["start"], gene["start"] + n_bins * 250, n_bins)

    from_negative = read_binned(negative, *window)
    from_positive = read_binned(positive, *window)
    assert from_negative.min() >= 0
    assert np.allclose(from_negative, from_positive)
    assert from_negative.max() > 0


def test_read_binned_returns_n_bins_past_a_contig_end(tmp_path):
    """A window running past the contig end still returns exactly n_bins.

    The length contract, which the caller relies on to align the result to
    design geometry.
    """
    from dienr.counts import read_binned

    genes = _make_genes(2, strands=("+",))
    path = _write_signed_signal(tmp_path / "pad.bw", genes, "+", sign=1.0)
    n_bins = 40
    start = CONTIG_SIZE - 10 * 250
    values = read_binned(path, CONTIG, start, start + n_bins * 250, n_bins)
    assert values.shape == (n_bins,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
