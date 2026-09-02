"""BigWig aggregation, exercised against the R reference package's fixtures."""

from __future__ import annotations

import numpy as np
import polars as pl
import pybigtools
import pytest

from dienr.annotation import Annotation
from dienr.counts import SignalFile, count_sample, declared_contigs, read_binned
from dienr.design import build_design
from dienr.solvers.legacy import fit_locus
from dienr.status import FitStatus


@pytest.fixture
def plus_bw(extdata):
    return extdata / "test_double_strand_plus.bw"


@pytest.fixture
def minus_bw(extdata):
    return extdata / "test_double_strand_minus.bw"


# --- the two mandatory reader flags ----------------------------------------


def test_binned_read_is_exact_not_zoom_level(plus_bw):
    """`exact=True` must be set; the zoom-level default silently disagrees.

    This is the defect the wrapper exists to prevent, so it is asserted
    directly rather than trusted.
    """
    handle = pybigtools.open(str(plus_bw))
    truth = np.asarray(handle.values("1", 0, 2000, fillna=0)).reshape(8, 250).sum(axis=1)
    approximate = np.asarray(handle.values("1", 0, 2000, bins=8, summary="sum", fillna=0))

    ours = read_binned(plus_bw, "1", 0, 2000, 8, summary="sum")

    np.testing.assert_allclose(ours, truth)
    # Guard against the wrapper regressing to the lossy path.
    assert not np.allclose(approximate, truth)


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
    assert counts.provenance["reader_settings"]["exact"] is True


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


def test_window_clipped_at_the_contig_end_is_zero_padded(plus_bw):
    """A locus running off the end of a contig must still align to the design.

    pybigtools returns fewer bins than requested when the window is clipped,
    so the vector is padded rather than left short, which would misalign every
    downstream index.
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
