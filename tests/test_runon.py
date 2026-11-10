"""Run-on tails: estimation, application, and the properties that make it safe.

The tests that matter most assert what plan 18.10-18.14 rest on: the estimator
recovers a known tail from the right strand's file and orientation; the tail
lands past the PAS in the direction of transcription; it never changes which
bins a fit can use; single-model loci are untouched; and on data that carries
run-on, the run-on design recovers abundances better than the binary one.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import dienr
from dienr.runon import (
    MIN_TRAINING_GENES,
    RunonConfig,
    _select_training_genes,
    apply_runon_tail,
    fit_runon_tail,
)
from dienr.shape import ShapeConfig, apply_shape_profile, fit_shape_profile
from dienr.solvers import poisson
from dienr.solvers._glm import Exposure, usable_bins

from .fixtures.bigwig import write_intervals

CONTIG = "chr1"
CONTIG_SIZE = 6_000_000
BIN = 250
TAIL = np.array([1.5, 2.0, 1.6, 1.0, 0.6, 0.3, 0.15, 0.05])  # 2 kb, body-density units
CONFIG = RunonConfig(bin_size=BIN, tail_bases=TAIL.size * BIN, buffer_bases=2_000)


def _isolated_genes(n: int = 30, length: int = 20_000, gap: int = 30_000) -> pl.DataFrame:
    rows, cursor = [], 10_000
    for i in range(n):
        rows.append(
            {
                "transcript_id": f"t{i}",
                "contig": CONTIG,
                "start": cursor,
                "end": cursor + length,
                "strand": "+-"[i % 2],
            }
        )
        cursor += length + gap
    return pl.DataFrame(rows)


def _write(
    path, genes: pl.DataFrame, strand: str, *, density: float = 8.0, tail=TAIL, sign: float = 1.0
):
    """Flat body of `density` reads per bin plus `tail` past each PAS."""
    intervals = []
    for g in genes.filter(pl.col("strand") == strand).iter_rows(named=True):
        for b in range(g["start"], g["end"], BIN):
            intervals.append((CONTIG, b, b + BIN, sign * density / BIN))
        for d, r in enumerate(tail):
            b = g["end"] + d * BIN if strand == "+" else g["start"] - (d + 1) * BIN
            intervals.append((CONTIG, b, b + BIN, sign * r * density / BIN))
    return write_intervals(path, {CONTIG: CONTIG_SIZE}, intervals)


@pytest.fixture
def tail_inputs(tmp_path):
    genes = _isolated_genes()
    annotation = dienr.Annotation.from_frame(genes, source="synthetic")
    # Minus-strand file written negative, as real ones are.
    signal = (
        dienr.SignalFile(path=_write(tmp_path / "p.bw", genes, "+"), strand="+"),
        dienr.SignalFile(path=_write(tmp_path / "m.bw", genes, "-", sign=-1.0), strand="-"),
    )
    return annotation, signal


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------


def test_fit_recovers_a_known_tail(tail_inputs):
    annotation, signal = tail_inputs
    tail = fit_runon_tail(annotation, signal, config=CONFIG)
    assert tail.n_genes == 30
    np.testing.assert_allclose(tail.values, TAIL, rtol=1e-6)
    assert tail.mass == pytest.approx(TAIL.sum())


@pytest.mark.filterwarnings("ignore:run-on tail fitted from only")
def test_tail_is_oriented_from_the_pas_on_both_strands(tmp_path):
    """Plus-only and minus-only training must give the same tail, not a reversed one."""
    genes = _isolated_genes()
    plus = dienr.SignalFile(path=_write(tmp_path / "p.bw", genes, "+"), strand="+")
    minus = dienr.SignalFile(path=_write(tmp_path / "m.bw", genes, "-", sign=-1.0), strand="-")
    only_plus = dienr.Annotation.from_frame(genes.filter(pl.col("strand") == "+"), source="p")
    only_minus = dienr.Annotation.from_frame(genes.filter(pl.col("strand") == "-"), source="m")
    a = fit_runon_tail(only_plus, (plus, minus), config=CONFIG).values
    b = fit_runon_tail(only_minus, (plus, minus), config=CONFIG).values
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_each_gene_is_read_from_its_own_strand(tmp_path):
    """Swapping the files must break the fit, or strand handling is untested."""
    genes = _isolated_genes()
    annotation = dienr.Annotation.from_frame(genes, source="synthetic")
    p = _write(tmp_path / "p.bw", genes, "+")
    m = _write(tmp_path / "m.bw", genes, "-")
    swapped = (dienr.SignalFile(path=m, strand="+"), dienr.SignalFile(path=p, strand="-"))
    with pytest.raises(ValueError, match="no gene passed"):
        fit_runon_tail(annotation, swapped, config=CONFIG)


def test_ratio_of_sums_is_scale_free(tmp_path):
    """Doubling every gene's expression leaves r(d) unchanged."""
    genes = _isolated_genes()
    annotation = dienr.Annotation.from_frame(genes, source="synthetic")
    signal = (
        dienr.SignalFile(path=_write(tmp_path / "p.bw", genes, "+", density=16.0), strand="+"),
        dienr.SignalFile(path=_write(tmp_path / "m.bw", genes, "-", density=16.0), strand="-"),
    )
    np.testing.assert_allclose(fit_runon_tail(annotation, signal, config=CONFIG).values, TAIL)


def test_neighbours_and_isoforms_are_excluded():
    genes = _isolated_genes(4)
    extra = pl.DataFrame(
        [
            # A second isoform of t0: t0 is no longer single-isoform.
            {
                "transcript_id": "t0b",
                "contig": CONTIG,
                "start": genes["start"][0] + 5_000,
                "end": genes["end"][0],
                "strand": "+",
            },
            # A same-strand gene starting inside t2's tail window.
            {
                "transcript_id": "n2",
                "contig": CONTIG,
                "start": genes["end"][2] + 500,
                "end": genes["end"][2] + 1_000,
                "strand": "+",
            },
            # A plus-strand gene in minus-strand t1's tail window: other strand, ignored.
            {
                "transcript_id": "o1",
                "contig": CONTIG,
                "start": genes["start"][1] - 1_200,
                "end": genes["start"][1] - 600,
                "strand": "+",
            },
        ]
    )
    annotation = dienr.Annotation.from_frame(pl.concat([genes, extra]), source="s")
    selected, rejected = _select_training_genes(annotation, CONFIG)
    ids = set(selected["transcript_id"].to_list())
    assert "t0" not in ids and "t0b" not in ids
    assert "t2" not in ids
    assert "t1" in ids  # o1 is on the other strand
    assert "t3" in ids
    assert rejected["neighbour_within_buffer"] >= 3


def test_identical_spans_count_once():
    genes = _isolated_genes(2)
    dup = genes.head(1).with_columns(pl.lit("t0dup").alias("transcript_id"))
    annotation = dienr.Annotation.from_frame(pl.concat([genes, dup]), source="s")
    selected, rejected = _select_training_genes(annotation, CONFIG)
    assert len(selected) == 2
    assert rejected["duplicate_span"] == 1


def test_low_signal_genes_are_dropped_and_none_left_is_an_error(tmp_path):
    genes = _isolated_genes(6)
    annotation = dienr.Annotation.from_frame(genes, source="synthetic")
    signal = (
        dienr.SignalFile(path=_write(tmp_path / "p.bw", genes, "+", density=1.0), strand="+"),
        dienr.SignalFile(path=_write(tmp_path / "m.bw", genes, "-", density=1.0), strand="-"),
    )
    with pytest.raises(ValueError, match="low_signal"):
        fit_runon_tail(annotation, signal, config=CONFIG)


def test_few_training_genes_warn(tmp_path):
    genes = _isolated_genes(MIN_TRAINING_GENES - 4)
    annotation = dienr.Annotation.from_frame(genes, source="synthetic")
    signal = (
        dienr.SignalFile(path=_write(tmp_path / "p.bw", genes, "+"), strand="+"),
        dienr.SignalFile(path=_write(tmp_path / "m.bw", genes, "-"), strand="-"),
    )
    with pytest.warns(UserWarning, match="only"):
        fit_runon_tail(annotation, signal, config=CONFIG)


def test_missing_strand_file_is_an_error(tail_inputs):
    annotation, signal = tail_inputs
    with pytest.raises(ValueError, match="no signal file"):
        fit_runon_tail(annotation, signal[0], config=CONFIG)
    with pytest.raises(ValueError, match="two signal files"):
        fit_runon_tail(annotation, (signal[0], signal[0]), config=CONFIG)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tail_bases": 1_000, "bin_size": 300}, "multiple of bin_size"),
        ({"tail_bases": 0}, "multiple"),
        ({"bin_size": 0}, "bin_size"),
        ({"buffer_bases": -1}, "buffer_bases"),
        ({"min_length": 1_000}, "min_length"),
        ({"min_body_signal": -1.0}, "min_body_signal"),
    ],
)
def test_invalid_config_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RunonConfig(**kwargs)


def test_tail_table_and_provenance(tail_inputs):
    annotation, signal = tail_inputs
    tail = fit_runon_tail(annotation, signal, config=CONFIG)
    table = tail.table()
    assert table["distance_start"][0] == 0
    assert table["distance_end"][-1] == CONFIG.tail_bases
    described = tail.describe()
    assert described["runon_digest"] == tail.digest
    assert described["runon_n_genes"] == 30
    # Same inputs, same digest; a different config, a different one.
    again = fit_runon_tail(annotation, signal, config=CONFIG)
    assert again.digest == tail.digest
    other = fit_runon_tail(
        annotation,
        signal,
        config=RunonConfig(bin_size=BIN, tail_bases=TAIL.size * BIN, buffer_bases=2_500),
    )
    assert other.digest != tail.digest


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


def _two_isoform_design(strand: str) -> dienr.Design:
    """A long and a short isoform sharing a TSS, plus an unrelated single gene."""
    long_start, long_end = 100_000, 120_000
    short = (100000, 108000) if strand == "+" else (112000, 120000)
    frame = pl.DataFrame(
        [
            {
                "transcript_id": "long",
                "contig": CONTIG,
                "start": long_start,
                "end": long_end,
                "strand": strand,
            },
            {
                "transcript_id": "short",
                "contig": CONTIG,
                "start": short[0],
                "end": short[1],
                "strand": strand,
            },
            {
                "transcript_id": "solo",
                "contig": CONTIG,
                "start": 300_000,
                "end": 310_000,
                "strand": strand,
            },
        ]
    )
    return dienr.build_design(dienr.Annotation.from_frame(frame, source="s"), bin_size=BIN)


def _tail(values=TAIL):
    from dienr.runon import RunonTail, _tail_digest

    return RunonTail(
        values=np.asarray(values, float),
        config=CONFIG,
        n_genes=30,
        digest=_tail_digest(CONFIG, np.asarray(values, float)),
    )


def _locus_of(design: dienr.Design, transcript: str) -> tuple[int, int]:
    row = design.transcript_map.filter(pl.col("transcript_id") == transcript).row(0, named=True)
    return row["locus_id"], row["model_index"]


@pytest.mark.parametrize("strand", ["+", "-"])
def test_tail_lands_past_the_pas_in_the_direction_of_transcription(strand):
    design = _two_isoform_design(strand)
    out = apply_runon_tail(design, _tail())
    lid, k = _locus_of(design, "short")
    first, last = (int(x) for x in out.model_spans[lid][k])
    column = out.models[lid][:, k] - _padded(design, out, lid)[:, k]
    if strand == "+":
        expected = np.arange(last + 1, last + 1 + TAIL.size)
    else:
        expected = np.arange(first - 1, first - 1 - TAIL.size, -1)
    np.testing.assert_allclose(column[expected], TAIL)
    assert np.count_nonzero(column) == TAIL.size


def _padded(before: dienr.Design, after: dienr.Design, lid: int) -> np.ndarray:
    """`before`'s matrix for `lid`, zero-padded to `after`'s extended window."""
    matrix = before.models[lid]
    grow = after.bin_count(lid) - before.bin_count(lid)
    plus = before.loci["strand"][lid] == "+"
    return np.pad(matrix, ((0, grow) if plus else (grow, 0), (0, 0)))


@pytest.mark.parametrize("strand", ["+", "-"])
def test_the_longest_isoform_gets_its_whole_tail(strand):
    """Plan 18.25: the window used to end at its PAS, clipping its tail entirely,
    while shorter isoforms' tails landed on its 3' body -- a bias toward them."""
    design = _two_isoform_design(strand)
    out = apply_runon_tail(design, _tail())
    lid, k = _locus_of(design, "long")
    assert out.bin_count(lid) == design.bin_count(lid) + TAIL.size
    added = out.models[lid][:, k] - _padded(design, out, lid)[:, k]
    np.testing.assert_allclose(np.sort(added[added > 0]), np.sort(TAIL))
    # Original bins keep their geometry: same boundaries, same occupancy.
    np.testing.assert_array_equal(
        _padded(design, out, lid)[:, k] > 0, (out.models[lid][:, k] > 0) & (added == 0)
    )
    solo, _ = _locus_of(design, "solo")
    assert out.bin_count(solo) == design.bin_count(solo) + TAIL.size
    if strand == "+":
        assert out.window_start(lid) == design.window_start(lid)
    else:
        assert out.window_start(lid) == design.window_start(lid) - TAIL.size * BIN
        assert out.bins.filter(pl.col("locus_id") == lid)["start"].min() == out.window_start(lid)


def test_extension_stops_at_the_next_same_strand_locus():
    rows = [
        {"transcript_id": "a", "start": 100_000, "end": 110_000, "strand": "+"},
        {"transcript_id": "b", "start": 112_000, "end": 120_000, "strand": "+"},  # 2 kb on
        {"transcript_id": "c", "start": 110_500, "end": 111_000, "strand": "-"},  # other strand
    ]
    frame = pl.DataFrame([{**r, "contig": CONTIG} for r in rows])
    design = dienr.build_design(dienr.Annotation.from_frame(frame, source="s"), bin_size=BIN)
    out = apply_runon_tail(design, _tail())
    a, _ = _locus_of(design, "a")
    b, _ = _locus_of(design, "b")
    a_end = out.window_start(a) + out.bin_count(a) * BIN
    assert a_end <= out.window_start(b)  # never into b's window
    assert out.bin_count(a) - design.bin_count(a) == (112_000 - 110_000) // BIN
    assert out.bin_count(b) == design.bin_count(b) + TAIL.size  # nothing downstream of b


def test_extend_false_keeps_the_old_clipped_windows():
    design = _two_isoform_design("+")
    out = apply_runon_tail(design, _tail(), extend=False)
    solo, _ = _locus_of(design, "solo")
    np.testing.assert_array_equal(out.models[solo], design.models[solo])
    lid, k = _locus_of(design, "long")
    np.testing.assert_array_equal(out.models[lid][:, k], design.models[lid][:, k])
    assert out.digest != apply_runon_tail(design, _tail()).digest


def test_counting_reads_the_extended_window(tmp_path):
    """Signal past the longest PAS is now counted, on both strands."""
    rows = [
        {"transcript_id": "p", "start": 100_000, "end": 110_000, "strand": "+"},
        {"transcript_id": "m", "start": 300_000, "end": 310_000, "strand": "-"},
    ]
    frame = pl.DataFrame([{**r, "contig": CONTIG} for r in rows])
    design = dienr.build_design(dienr.Annotation.from_frame(frame, source="s"), bin_size=BIN)
    out = apply_runon_tail(design, _tail())
    # Read-through only: 2 kb past the + PAS and 2 kb past (below) the - PAS.
    plus = write_intervals(
        tmp_path / "p.bw", {CONTIG: CONTIG_SIZE}, [(CONTIG, 110_000, 112_000, 1.0)]
    )
    minus = write_intervals(
        tmp_path / "m.bw", {CONTIG: CONTIG_SIZE}, [(CONTIG, 298_000, 300_000, 1.0)]
    )
    signal = (dienr.SignalFile(plus, "+"), dienr.SignalFile(minus, "-"))
    before = dienr.count_sample(design, "s", signal, summary="sum")
    after = dienr.count_sample(out, "s", signal, summary="sum")
    for transcript in ("p", "m"):
        lid, _ = _locus_of(design, transcript)
        assert before.values[lid].sum() == 0
        assert after.values[lid].sum() == pytest.approx(2_000)


def test_tail_never_changes_which_bins_are_usable_without_extension():
    """Plan 18.12: a locus window is a union of spans, so an unextended tail adds no bin."""
    frame = pl.DataFrame(
        [
            {
                "transcript_id": f"t{i}",
                "contig": CONTIG,
                "start": 100_000 + 3_000 * i,
                "end": 104_000 + 7_000 * i,
                "strand": "+-"[i % 2],
            }
            for i in range(8)
        ]
    )
    design = dienr.build_design(
        dienr.Annotation.from_frame(frame, source="s"),
        bin_size=BIN,
        mask_start_bins=(1, 1),
        mask_end_bins=(4, 4),
    )
    out = apply_runon_tail(design, _tail(), extend=False)
    assert any(not np.array_equal(a, b) for a, b in zip(design.models, out.models, strict=True))
    for before, after in zip(design.models, out.models, strict=True):
        np.testing.assert_array_equal(
            usable_bins(before, Exposure()), usable_bins(after, Exposure())
        )


def test_masked_bins_stay_zero():
    design = dienr.build_design(
        dienr.Annotation.from_frame(
            pl.DataFrame(
                [
                    {
                        "transcript_id": "a",
                        "contig": CONTIG,
                        "start": 0 + 100_000,
                        "end": 120_000,
                        "strand": "+",
                    },
                    {
                        "transcript_id": "b",
                        "contig": CONTIG,
                        "start": 100_000,
                        "end": 106_000,
                        "strand": "+",
                    },
                ]
            ),
            source="s",
        ),
        bin_size=BIN,
        mask_end_bins=(4, 4),
    )
    out = apply_runon_tail(design, _tail())
    for lid, matrix in enumerate(out.models):
        assert matrix.shape[0] == out.masks[lid].size
        assert not matrix[out.masks[lid]].any()


def test_digest_and_guards():
    design = _two_isoform_design("+")
    out = apply_runon_tail(design, _tail())
    assert out.runon_digest == _tail().digest
    assert out.digest != design.digest
    assert design.runon_digest is None  # source unchanged
    with pytest.raises(ValueError, match="already carries"):
        apply_runon_tail(out, _tail())
    coarse = dienr.build_design(
        dienr.Annotation.from_frame(
            pl.DataFrame(
                [{"transcript_id": "x", "contig": CONTIG, "start": 0, "end": 5_000, "strand": "+"}]
            ),
            source="s",
        ),
        bin_size=500,
    )
    with pytest.raises(ValueError, match="bin size"):
        apply_runon_tail(coarse, _tail())


def test_commutes_with_a_shape_profile(tmp_path):
    """The profile reweights a model's own span; the tail lies outside it."""
    from .test_shape import _make_genes, _write_signal

    genes = _make_genes(12)
    annotation = dienr.Annotation.from_frame(genes, source="synthetic")
    signal = (
        dienr.SignalFile(path=_write_signal(tmp_path / "p.bw", genes, "+"), strand="+"),
        dienr.SignalFile(path=_write_signal(tmp_path / "m.bw", genes, "-"), strand="-"),
    )
    profile = fit_shape_profile(annotation, signal, config=ShapeConfig(span=0.1))
    design = _two_isoform_design("+")
    a = apply_runon_tail(apply_shape_profile(design, profile), _tail())
    b = apply_shape_profile(apply_runon_tail(design, _tail()), profile)
    for x, y in zip(a.models, b.models, strict=True):
        np.testing.assert_allclose(x, y)


# --------------------------------------------------------------------------
# Known truth: read-through confounds a binary design, the tail resolves it
# --------------------------------------------------------------------------


def test_runon_design_recovers_abundance_that_binary_misattributes():
    """Only the short isoform is expressed; its read-through lands in the long one.

    Under a binary design that read-through is evidence for the long isoform.
    With the true tail in the design it is explained, and the long isoform's
    estimate goes to (near) zero.
    """
    design = _two_isoform_design("+")
    out = apply_runon_tail(design, _tail())
    lid, short = _locus_of(design, "short")
    _, long_ = _locus_of(design, "long")
    truth = np.zeros(design.models[lid].shape[1])
    truth[short] = 40.0
    rng = np.random.default_rng(0)
    errors = {"binary": [], "runon": []}
    n_binary = design.bin_count(lid)  # + strand: the extension is at the end
    for _ in range(20):
        y = rng.poisson(out.models[lid] @ truth).astype(float)
        for name, d, obs in (("binary", design, y[:n_binary]), ("runon", out, y)):
            fit = poisson.fit_locus(d.models[lid], obs)
            errors[name].append(fit.abundance[long_])
    # Binary assigns the read-through to the long isoform; run-on does not.
    assert np.mean(errors["binary"]) > 5 * max(np.mean(errors["runon"]), 0.1)
    assert np.mean(errors["runon"]) < 1.0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_runon_cli_fits_saves_and_applies(tmp_path, capsys):
    from dienr.cli import DESIGN_FILENAME, RUNON_FILENAME, main

    from .conftest import write_gtf

    genes = _isolated_genes()
    gtf = write_gtf(
        tmp_path / "g.gtf",
        [
            {
                "transcript_id": r["transcript_id"],
                "gene_id": r["transcript_id"],
                "contig": r["contig"],
                "start": r["start"] + 1,
                "end": r["end"],
                "strand": r["strand"],
            }
            for r in genes.iter_rows(named=True)
        ],
    )
    plus = _write(tmp_path / "p.bw", genes, "+")
    minus = _write(tmp_path / "m.bw", genes, "-", sign=-1.0)
    tail_dir = tmp_path / "tail"
    code = main(
        [
            "runon-tail",
            "--gtf",
            str(gtf),
            "--bigwig-plus",
            str(plus),
            "--bigwig-minus",
            str(minus),
            "--output",
            str(tail_dir),
            "--tail-bases",
            str(TAIL.size * BIN),
            "--buffer-bases",
            "2000",
        ]
    )
    assert code == 0
    assert (tail_dir / RUNON_FILENAME).exists()
    table = pl.read_parquet(tail_dir / "runon_tail.parquet")
    np.testing.assert_allclose(table["relative_density"].to_numpy(), TAIL, rtol=1e-6)
    assert "30 training genes" in capsys.readouterr().out

    import pickle

    plain, with_tail = tmp_path / "plain", tmp_path / "runon"
    assert main(["build-design", "--gtf", str(gtf), "--output", str(plain)]) == 0
    assert (
        main(
            [
                "build-design",
                "--gtf",
                str(gtf),
                "--output",
                str(with_tail),
                "--runon-tail",
                str(tail_dir),
            ]
        )
        == 0
    )
    built = capsys.readouterr()
    assert "run-on tail" in built.out
    assert "plan 18.26" in built.err  # not recommended for isoform calls
    a = pickle.loads((plain / DESIGN_FILENAME).read_bytes())
    b = pickle.loads((with_tail / DESIGN_FILENAME).read_bytes())
    assert a.runon_digest is None and b.runon_digest is not None
    assert a.digest != b.digest

    # Fitting a run-on design with the legacy objective warns (plan 18.21).
    sheet = tmp_path / "samples.csv"
    sheet.write_text(f"sample_id,bigwig_plus,bigwig_minus\ns,{plus},{minus}\n")
    for design_dir, counts_dir in ((with_tail, tmp_path / "c1"), (plain, tmp_path / "c0")):
        count = ["count", "--design", str(design_dir), "--samples", str(sheet)]
        assert main([*count, "--output", str(counts_dir)]) == 0
    capsys.readouterr()
    fit = ["fit", "--counts", str(tmp_path / "c1"), "--design", str(with_tail)]
    main([*fit, "--output", str(tmp_path / "r1")])
    assert "plan 18.21" in capsys.readouterr().err
    fit = ["fit", "--counts", str(tmp_path / "c0"), "--design", str(plain)]
    main([*fit, "--output", str(tmp_path / "r0")])
    assert "plan 18.21" not in capsys.readouterr().err


def test_runon_cli_reports_errors(tmp_path):
    from dienr.cli import main

    from .conftest import write_gtf

    gtf = write_gtf(
        tmp_path / "g.gtf",
        [
            {
                "transcript_id": "a",
                "gene_id": "a",
                "contig": CONTIG,
                "start": 1,
                "end": 20_000,
                "strand": "+",
            }
        ],
    )
    with pytest.raises(SystemExit, match="no runon_tail.pkl"):
        main(
            [
                "build-design",
                "--gtf",
                str(gtf),
                "--output",
                str(tmp_path / "o"),
                "--runon-tail",
                str(tmp_path),
            ]
        )
    empty = write_intervals(tmp_path / "e.bw", {CONTIG: CONTIG_SIZE}, [])
    with pytest.raises(SystemExit, match="no gene passed"):
        main(
            [
                "runon-tail",
                "--gtf",
                str(gtf),
                "--bigwig-plus",
                str(empty),
                "--bigwig-minus",
                str(empty),
                "--output",
                str(tmp_path / "t"),
            ]
        )
