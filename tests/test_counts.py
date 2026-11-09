"""BigWig aggregation, exercised against the R reference package's fixtures."""

from __future__ import annotations

import typing

import numpy as np
import polars as pl
import pytest
from figwig import BigWigReader

from dienr import counts as counts_module
from dienr.annotation import Annotation
from dienr.counts import (
    SignalFile,
    count_sample,
    declared_contigs,
    read_binned,
    read_loci_binned,
)
from dienr.design import build_design
from dienr.solvers.legacy import fit_locus
from dienr.status import FitStatus

from .fixtures.bigwig import write_intervals


@pytest.fixture
def plus_bw(extdata):
    return extdata / "test_double_strand_plus.bw"


@pytest.fixture
def minus_bw(extdata):
    return extdata / "test_double_strand_minus.bw"


# --- exactness ---------------------------------------------------------------


def test_binned_read_is_exact_per_base_sum(plus_bw):
    """Bins are exact per-base sums, never a lossy zoom-level summary.

    ``[63, 68, 72, 77]`` is the true per-base sum of 250 bp bins over
    ``1:0-2000``; the zoom-level path `pybigtools` took without ``exact=True``
    returned ``[62.53, 69.53, 71.47, 76.76]`` with no error. figwig has no zoom
    path, but the numbers stay pinned so a reader that ever grows one cannot
    slip in silently.
    """
    per_base = BigWigReader(str(plus_bw)).read(["1"], [0], 2000)[0]
    truth = per_base.reshape(8, 250).sum(axis=1, dtype=np.float64)

    ours = read_binned(plus_bw, "1", 0, 2000, 8, summary="sum")

    np.testing.assert_array_equal(ours, truth)
    np.testing.assert_array_equal(ours[:4], [63, 68, 72, 77])


def test_uncovered_positions_read_as_zero_not_nan(extdata):
    """An absent interval means no polymerase, not a missing observation."""
    values = read_binned(extdata / "test_single_strand.bw", "1", 90_000, 100_000, 40)
    assert np.isfinite(values).all()


def test_reads_past_contig_end_are_zero(plus_bw):
    length = declared_contigs(plus_bw)["1"]
    values = read_binned(plus_bw, "1", length - 500, length + 500, 4)
    assert np.isfinite(values).all()


# --- header seqinfo ---------------------------------------------------------


def test_declared_contigs_come_from_the_header(extdata):
    """The header lists contigs whether or not they carry data in a window."""
    assert declared_contigs(extdata / "test_multichrom.bw") == {"1": 100_000, "Z": 100_000}


def test_absent_contig_raises_rather_than_returning_zeros(plus_bw):
    with pytest.raises(KeyError):
        read_binned(plus_bw, "chrNope", 0, 1000, 4)


# --- end-to-end -------------------------------------------------------------


@pytest.fixture
def two_locus_design():
    """Two loci on opposite strands, inside the fixtures' single contig."""
    frame = pl.DataFrame(
        {
            "transcript_id": ["tx_plus", "tx_minus"],
            "contig": ["1", "1"],
            "start": [1_000, 50_000],
            "end": [11_000, 60_000],
            "strand": ["+", "-"],
        }
    )
    return build_design(Annotation.from_frame(frame), bin_size=250)


def test_count_sample_aligns_to_design(two_locus_design, plus_bw, minus_bw):
    counts = count_sample(
        two_locus_design,
        "sample1",
        (SignalFile(plus_bw, "+"), SignalFile(minus_bw, "-")),
    )
    assert len(counts.values) == two_locus_design.n_loci
    for locus_id, values in enumerate(counts.values):
        assert values.size == two_locus_design.bin_count(locus_id)
    assert counts.design_digest == two_locus_design.digest
    # The settings that make uncovered bases distinguishable from zero-valued
    # intervals must be recorded, since every covered-base summary depends on
    # them (figwig's default `missing=0.0` would silently change each `mean`).
    assert counts.provenance["reader"] == "figwig"
    assert counts.provenance["reader_settings"]["missing"] == "nan"
    assert counts.provenance["reader_settings"]["oob"] == "nan"


def test_signal_is_nonnegative_on_both_strands(two_locus_design, plus_bw, minus_bw):
    """Minus-strand BigWigs are conventionally negative; counts are magnitudes."""
    counts = count_sample(
        two_locus_design,
        "sample1",
        (SignalFile(plus_bw, "+"), SignalFile(minus_bw, "-")),
    )
    for values in counts.values:
        assert (values >= 0).all()


def test_masked_bins_are_zeroed_in_the_signal(plus_bw):
    frame = pl.DataFrame(
        {
            "transcript_id": ["tx"],
            "contig": ["1"],
            "start": [1_000],
            "end": [11_000],
            "strand": ["+"],
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250, mask_start_bins=(3, 0))
    counts = count_sample(design, "s", SignalFile(plus_bw, "+"))
    assert counts.values[0][design.mask(0)].sum() == 0.0


def test_missing_contig_is_distinguished_from_all_zero(plus_bw):
    """Two loci, one on a contig the file does not declare."""
    frame = pl.DataFrame(
        {
            "transcript_id": ["real", "absent"],
            "contig": ["1", "chrNope"],
            "start": [1_000, 1_000],
            "end": [11_000, 11_000],
            "strand": ["+", "+"],
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250)
    counts = count_sample(design, "s", SignalFile(plus_bw, "+"))

    by_name = dict(zip(design.loci["locus_name"], counts.locus_status, strict=True))
    statuses = {name.split(":")[0]: status for name, status in by_name.items()}
    assert statuses["chrNope"] is FitStatus.MISSING_CONTIG
    assert statuses["1"] is not FitStatus.MISSING_CONTIG
    # A failure must never be reported as zero signal.
    absent_id = design.loci.filter(pl.col("contig") == "chrNope")["locus_id"][0]
    assert np.isnan(counts.values[absent_id]).all()


def test_missing_strand_file_is_rejected(two_locus_design, plus_bw):
    with pytest.raises(ValueError, match="no signal file"):
        count_sample(two_locus_design, "s", SignalFile(plus_bw, "+"))


def test_duplicate_strand_files_are_rejected(two_locus_design, plus_bw):
    with pytest.raises(ValueError, match="two signal files"):
        count_sample(
            two_locus_design,
            "s",
            (SignalFile(plus_bw, "+"), SignalFile(plus_bw, "+")),
        )


def test_normalized_signal_is_rejected_for_count_likelihoods(plus_bw):
    scaled = SignalFile(plus_bw, "+", value_kind="library_normalized")
    with pytest.raises(ValueError, match="count-equivalent"):
        scaled.require_count_equivalent("poisson")
    # A non-count likelihood is allowed to use it.
    scaled.require_count_equivalent("log")


def test_full_pipeline_fits_and_explains_signal(two_locus_design, plus_bw, minus_bw):
    """Annotation to abundance, on real fixture signal."""
    counts = count_sample(
        two_locus_design,
        "sample1",
        (SignalFile(plus_bw, "+"), SignalFile(minus_bw, "-")),
    )
    fits = [
        fit_locus(
            two_locus_design.models[i],
            counts.values[i],
            prior_status=counts.locus_status[i],
        )
        for i in range(two_locus_design.n_loci)
    ]
    assert all(f.status in {FitStatus.OK, FitStatus.ALL_ZERO} for f in fits)
    assert any(f.abundance.sum() > 0 for f in fits)
    for fit in fits:
        assert np.isfinite(fit.abundance).all()


# --- input validation and edge paths ---------------------------------------


def test_signal_file_rejects_a_bad_strand(plus_bw):
    with pytest.raises(ValueError, match="strand must be"):
        SignalFile(plus_bw, "?")


def test_signal_file_rejects_an_unknown_value_kind(plus_bw):
    with pytest.raises(ValueError, match="value_kind"):
        SignalFile(plus_bw, "+", value_kind="normalised")


def test_require_count_equivalent_rejects_a_mean_summary(plus_bw):
    """The per-bin summary is the second way input can fail to be a count.

    Coverage found this branch untested: only the transformed-file branch was
    exercised, even though the mean summary is the likelier mistake, being the
    default.
    """
    signal = SignalFile(plus_bw, "+")
    with pytest.raises(ValueError, match="count-preserving summary"):
        signal.require_count_equivalent("poisson", summary="mean")
    # The count-preserving summary passes, and a non-count likelihood is
    # unconstrained either way.
    signal.require_count_equivalent("poisson", summary="sum")
    signal.require_count_equivalent("legacy", summary="mean")


@pytest.mark.parametrize("likelihood", ["poisson", "quasi_poisson"])
def test_every_count_likelihood_is_guarded_by_the_public_check(likelihood):
    """`SignalFile.require_count_equivalent` covers every count likelihood.

    Its likelihood set once lived apart from `fit`'s and lacked
    ``"quasi_poisson"``, so a per-base mean passed the public check for it.
    `fit` was unaffected only because it carried a second, correct copy.
    The check reads metadata only, so no file is needed and this runs without
    the R reference fixtures.
    """
    path = "never-opened.bw"
    with pytest.raises(ValueError, match="count-preserving summary"):
        SignalFile(path, "+").require_count_equivalent(likelihood, summary="mean")
    with pytest.raises(ValueError, match="count-equivalent"):
        SignalFile(path, "+", value_kind="library_normalized").require_count_equivalent(likelihood)


def test_fit_and_signal_file_share_one_set_of_count_likelihoods():
    from dienr import pipeline

    assert set(typing.get_args(pipeline.Likelihood)) - {"legacy"} == counts_module.COUNT_LIKELIHOODS


def test_window_clipped_at_the_contig_end_is_zero_padded(plus_bw):
    """A locus running off the end of a contig must still align to the design.

    A vector shorter than the design's bin count would misalign every
    downstream index, so the clipped window must still yield every bin.
    """
    length = declared_contigs(plus_bw)["1"]
    frame = pl.DataFrame(
        {
            "transcript_id": ["tx_edge"],
            "contig": ["1"],
            # Deliberately overruns the declared contig length.
            "start": [length - 600],
            "end": [length + 4_000],
            "strand": ["+"],
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250)
    counts = count_sample(design, "s", SignalFile(plus_bw, "+"))
    assert counts.values[0].size == design.bin_count(0)
    assert np.isfinite(counts.values[0]).all()
    # The bins past the contig end carry no signal.
    assert counts.values[0][-1] == 0.0


def test_total_signal_sums_across_loci(two_locus_design, plus_bw, minus_bw):
    counts = count_sample(
        two_locus_design,
        "s",
        (SignalFile(plus_bw, "+"), SignalFile(minus_bw, "-")),
    )
    assert counts.total_signal() == pytest.approx(sum(v.sum() for v in counts.values))


# --- batched tile reads ------------------------------------------------------

_SUMMARIES = ("sum", "mean", "min", "max", "std")
_SIZES = {"c1": 50_003, "c2": 30_011}  # not multiples of any tile


@pytest.fixture
def random_signal(tmp_path):
    """Sparse, signed, partly zero-valued signal on two contigs."""
    rng = np.random.default_rng(7)
    intervals = []
    for contig, length in _SIZES.items():
        position = 0
        while True:
            position += int(rng.integers(0, 80))
            if position >= length:
                break
            end = min(position + int(rng.integers(1, 50)), length)
            value = float(rng.choice([0.0, 1.0, 3.0, -2.0, rng.normal() * 4]))
            intervals.append((contig, position, end, value))
            position = end
    return write_intervals(tmp_path / "random.bw", _SIZES, intervals)


def _random_windows(rng, bin_size, n=60):
    windows = []
    for _ in range(n):
        contig = str(rng.choice(list(_SIZES)))
        n_bins = int(rng.choice([1, 2, 15, 16, 17, 33, int(rng.integers(1, 120))]))
        start = int(rng.integers(0, _SIZES[contig] - n_bins * bin_size))
        windows.append((contig, start, n_bins))
    # Two windows ending exactly at the contig end, so their final tile runs
    # off it, and a duplicate.
    windows.append(("c1", _SIZES["c1"] - 7 * bin_size, 7))
    windows.append(("c2", _SIZES["c2"] - 20 * bin_size, 20))
    windows.append(windows[0])
    return windows


@pytest.mark.parametrize("summary", _SUMMARIES)
@pytest.mark.parametrize("max_bytes", [64 * 2**20, 1])
@pytest.mark.parametrize("bin_size", [1, 37, 250])
def test_batched_read_equals_one_window_at_a_time(
    random_signal, monkeypatch, summary, max_bytes, bin_size
):
    """`read_loci_binned` is `read_binned` per window, for every summary.

    ``max_bytes=1`` forces one window per figwig call, so chunk boundaries are
    exercised as hard as possible; the default puts every window in one call.
    """
    monkeypatch.setattr(counts_module, "MAX_CHUNK_BYTES", max_bytes)
    reader = BigWigReader(str(random_signal))
    windows = _random_windows(np.random.default_rng(bin_size), bin_size)

    batched = read_loci_binned(reader, windows, bin_size, summary=summary)

    for (contig, start, n_bins), got in zip(windows, batched, strict=True):
        expected = read_binned(
            random_signal, contig, start, start + n_bins * bin_size, n_bins, summary=summary
        )
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
        assert got.dtype == np.float64 and got.shape == (n_bins,)


def test_batched_results_do_not_share_memory(random_signal):
    """Each locus vector is masked in place downstream, so none may alias."""
    reader = BigWigReader(str(random_signal))
    first, second = read_loci_binned(reader, [("c1", 0, 4), ("c1", 1000, 4)], 250)
    assert not np.shares_memory(first, second)


@pytest.mark.parametrize("summary", ["sum", "mean"])
def test_count_sample_batched_equals_per_locus(random_signal, monkeypatch, summary):
    """`count_sample` through the batch matches a locus-by-locus read.

    The design includes a locus overrunning the contig end, which takes the
    separate clipped path, so both paths and their merge are covered.
    """
    frame = pl.DataFrame(
        {
            "transcript_id": ["a", "a2", "b", "c", "edge"],
            "contig": ["c1", "c1", "c1", "c2", "c2"],
            "start": [100, 900, 20_000, 5_000, _SIZES["c2"] - 600],
            "end": [6_000, 4_000, 31_000, 9_100, _SIZES["c2"] + 2_000],
            "strand": ["+", "+", "+", "+", "+"],
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250)
    signal = SignalFile(random_signal, "+")

    batched = count_sample(design, "s", signal, summary=summary)
    monkeypatch.setattr(counts_module, "MAX_CHUNK_BYTES", 1)
    one_per_call = count_sample(design, "s", signal, summary=summary, n_jobs=1)

    lengths = declared_contigs(random_signal)
    for locus in design.loci.iter_rows(named=True):
        locus_id = locus["locus_id"]
        n_bins = design.bin_count(locus_id)
        start = int(locus["start"])
        expected = read_binned(
            random_signal,
            locus["contig"],
            start,
            min(start + n_bins * 250, lengths[locus["contig"]]),
            n_bins,
            summary=summary,
        )
        assert batched.totals_all_bins[locus_id] == pytest.approx(expected.sum(), rel=1e-12)
        expected[design.mask(locus_id)] = 0.0
        np.testing.assert_allclose(batched.values[locus_id], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(batched.values[locus_id], one_per_call.values[locus_id])
    assert batched.locus_status == one_per_call.locus_status


def test_batched_read_rejects_a_nonpositive_thread_count(random_signal):
    reader = BigWigReader(str(random_signal))
    with pytest.raises(ValueError, match="n_jobs"):
        read_loci_binned(reader, [("c1", 0, 4)], 250, n_jobs=0)
