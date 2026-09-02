"""Output schemas, round-tripping, and the status accounting."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from dienr.annotation import Annotation
from dienr.counts import SampleCounts
from dienr.design import build_design
from dienr.pipeline import all_statuses_accounted_for, fit
from dienr.results import (
    LOCUS_QC_SCHEMA,
    MODEL_ABUNDANCE_SCHEMA,
    SCHEMA_VERSION,
    FitResult,
    RunSummary,
    collect_results,
)
from dienr.solvers.legacy import fit_locus
from dienr.status import FitStatus


@pytest.fixture
def design():
    """Two loci: one with two distinguishable models, one with a single model."""
    frame = pl.DataFrame(
        {
            "transcript_id": ["a", "b", "c"],
            "gene_id": ["g1", "g1", "g2"],
            "contig": ["chr1", "chr1", "chr1"],
            "start": [0, 0, 500_000],
            "end": [2000, 4000, 502_000],
            "strand": ["+", "+", "+"],
        }
    )
    return build_design(Annotation.from_frame(frame), bin_size=250)


def synthetic_counts(design, sample_id="s1", scale=1.0):
    values = []
    for locus_id in range(design.n_loci):
        models = design.models[locus_id]
        truth = np.full(models.shape[1], 5.0 * scale)
        values.append(models @ truth)
    return SampleCounts(
        sample_id=sample_id,
        values=values,
        locus_status=[FitStatus.OK] * design.n_loci,
        design_digest=design.digest,
    )


# --- schema conformance -----------------------------------------------------


def test_output_tables_match_their_schemas(design):
    result = fit(design, synthetic_counts(design))
    assert dict(result.model_abundance.schema) == MODEL_ABUNDANCE_SCHEMA
    assert dict(result.locus_qc.schema) == LOCUS_QC_SCHEMA


def test_one_abundance_row_per_sample_and_model(design):
    result = fit(design, synthetic_counts(design))
    expected = sum(m.shape[1] for m in design.models)
    assert result.model_abundance.height == expected


def test_one_qc_row_per_sample_and_locus(design):
    result = fit(design, [synthetic_counts(design, "s1"), synthetic_counts(design, "s2")])
    assert result.locus_qc.height == 2 * design.n_loci
    assert result.model_abundance["sample_id"].n_unique() == 2


def test_provenance_carries_digests_and_version(design):
    result = fit(design, synthetic_counts(design))
    assert result.provenance["schema_version"] == SCHEMA_VERSION
    assert result.provenance["design_digest"] == design.digest
    assert result.provenance["annotation_digest"] == design.annotation_digest
    assert result.provenance["transform"] == "log"


def test_every_status_is_recognized(design):
    assert all_statuses_accounted_for(fit(design, synthetic_counts(design)))


# --- round trip -------------------------------------------------------------


def test_write_read_round_trip(design, tmp_path):
    result = fit(design, synthetic_counts(design))
    result.write(tmp_path / "out")
    back = FitResult.read(tmp_path / "out")

    assert back.model_abundance.equals(result.model_abundance)
    assert back.locus_qc.equals(result.locus_qc)
    assert back.provenance["design_digest"] == design.digest


def test_manifest_is_written_and_readable(design, tmp_path):
    fit(design, synthetic_counts(design)).write(tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert "summary" in manifest


def test_parquet_carries_metadata(design, tmp_path):
    """A single file must stay traceable when separated from the manifest."""
    fit(design, synthetic_counts(design)).write(tmp_path / "out")
    meta = pl.read_parquet_metadata(tmp_path / "out" / "model_abundance.parquet")
    assert meta["design_digest"] == design.digest
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["annotation_digest"] == design.annotation_digest


def test_reading_a_foreign_schema_version_is_rejected(design, tmp_path):
    out = tmp_path / "out"
    fit(design, synthetic_counts(design)).write(out)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["schema_version"] = "999"
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version"):
        FitResult.read(out)


# --- summary and failure accounting ----------------------------------------


def test_summary_counts_statuses(design):
    result = fit(design, synthetic_counts(design))
    assert result.summary.n_fits == design.n_loci
    assert result.summary.n_failures == 0
    assert result.summary.failure_fraction == 0.0


def test_failures_are_counted_and_not_zeros(design):
    counts = synthetic_counts(design)
    counts.locus_status[0] = FitStatus.MISSING_CONTIG
    counts.values[0] = np.full_like(counts.values[0], np.nan)

    result = fit(design, counts)
    assert result.summary.n_failures == 1

    failed = result.model_abundance.filter(pl.col("locus_id") == 0)
    assert failed["abundance_raw"].is_nan().all()
    assert (failed["solver_status"] == "missing_contig").all()


def test_exceeds_threshold_detects_a_mostly_failed_run():
    summary = RunSummary(by_status={"ok": 1, "missing_contig": 9}, n_fits=10)
    assert summary.n_failures == 9
    assert summary.exceeds(0.5)
    assert not summary.exceeds(0.95)


def test_empty_run_has_no_failure_fraction():
    assert RunSummary(by_status={}, n_fits=0).failure_fraction == 0.0


def test_rank_deficient_locus_is_reported_not_called_ok():
    """A converged fit on a degenerate design must not report plain 'ok'.

    Three distinct transcripts where the third spans exactly the union of the
    first two. Exact-duplicate collapsing cannot help — no two columns are
    equal — yet the design is rank 2 with 3 models, so the abundance split is
    not unique. This is a realistic annotation pattern, not a contrived one.
    """
    frame = pl.DataFrame(
        {
            "transcript_id": ["first_half", "second_half", "whole"],
            "contig": ["chr1"] * 3,
            "start": [0, 500, 0],
            "end": [500, 1000, 1000],
            "strand": ["+"] * 3,
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250)
    models = design.models[0]
    assert models.shape == (4, 3), "expected three distinct surviving models"
    assert np.linalg.matrix_rank(models) == 2, "expected a rank-deficient design"

    counts = SampleCounts(
        sample_id="s1",
        values=[models @ np.array([2.0, 3.0, 1.0])],
        locus_status=[FitStatus.OK],
        design_digest=design.digest,
    )
    fits = [fit_locus(models, counts.values[0])]
    result = collect_results(design, [(counts, fits)])

    assert result.locus_qc["solver_status"][0] == "rank_deficient"
    assert result.locus_qc["matrix_rank"][0] == 2
    assert result.locus_qc["n_models_after_collapse"][0] == 3
    # The prediction is still well determined even though the split is not.
    np.testing.assert_allclose(models @ fits[0].abundance, counts.values[0], atol=1e-2)


# --- mismatch guards --------------------------------------------------------


def test_counts_from_a_different_design_are_rejected(design):
    counts = synthetic_counts(design)
    object.__setattr__(counts, "design_digest", "not-the-same")
    with pytest.raises(ValueError, match="was counted against design"):
        fit(design, counts)


def test_wrong_number_of_fits_is_rejected(design):
    counts = synthetic_counts(design)
    with pytest.raises(ValueError, match="fits for"):
        collect_results(design, [(counts, [])])


# --- ML adapter and legacy export ------------------------------------------


def test_to_numpy_gives_a_sample_by_model_matrix(design):
    result = fit(design, [synthetic_counts(design, "s1"), synthetic_counts(design, "s2", 2.0)])
    matrix, samples, models = result.to_numpy()
    assert matrix.shape == (2, sum(m.shape[1] for m in design.models))
    assert samples == ["s1", "s2"]
    assert len(models) == matrix.shape[1]
    # s2 was generated at twice the abundance.
    assert matrix[1].sum() > matrix[0].sum()


def test_legacy_export_repeats_shared_models_with_caveat_columns():
    """The reference repeats a shared abundance per transcript; label it."""
    frame = pl.DataFrame(
        {
            "transcript_id": ["a", "b"],
            "contig": ["chr1", "chr1"],
            "start": [0, 0],
            "end": [1000, 1000],
            "strand": ["+", "+"],
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250)
    result = fit(design, synthetic_counts(design))

    legacy = result.legacy_transcript_abundance()
    assert legacy.height == 2
    assert legacy["abundance_raw"][0] == legacy["abundance_raw"][1]
    assert not legacy["exactly_identifiable"].any()
    assert (legacy["equivalence_class_size"] == 2).all()


# --- inactive transcripts ---------------------------------------------------


def test_inactive_transcript_pins_a_single_member_model(design):
    result = fit(design, synthetic_counts(design), inactive_transcripts={"c"})
    locus_c = design.transcript_map.filter(pl.col("transcript_id") == "c")["locus_id"][0]
    pinned = result.model_abundance.filter(pl.col("locus_id") == locus_c)
    assert (pinned["abundance_raw"] == 0.0).all()


def test_one_active_member_keeps_a_shared_model_active():
    """Section 5: an equivalence class is active if any member is active."""
    frame = pl.DataFrame(
        {
            "transcript_id": ["a", "b"],
            "contig": ["chr1", "chr1"],
            "start": [0, 0],
            "end": [1000, 1000],
            "strand": ["+", "+"],
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250)
    counts = synthetic_counts(design)

    only_one = fit(design, counts, inactive_transcripts={"a"})
    assert only_one.model_abundance["abundance_raw"][0] > 0.0

    both = fit(design, counts, inactive_transcripts={"a", "b"})
    assert both.model_abundance["abundance_raw"][0] == 0.0


# --- masking cost reporting -------------------------------------------------


def test_qc_reports_both_observed_totals_and_masked_fraction():
    """`total_observed` is fit-bin signal; coverage needs the pre-mask total.

    Masking zeroes bins in place, so without recording the pre-mask total a
    heavily masked locus would appear to have little signal rather than to have
    had most of it discarded.
    """
    frame = pl.DataFrame(
        {
            "transcript_id": ["tx"],
            "contig": ["chr1"],
            "start": [0],
            "end": [1000],
            "strand": ["+"],
        }
    )
    design = build_design(Annotation.from_frame(frame), bin_size=250, mask_tss=(0, 250))
    assert design.masked_fraction(0) == pytest.approx(0.5)

    observed = np.full(design.bin_count(0), 10.0)
    counts = SampleCounts(
        sample_id="s1",
        values=[observed.copy()],
        locus_status=[FitStatus.OK],
        design_digest=design.digest,
        totals_all_bins=[float(observed.sum())],
    )
    counts.values[0][design.mask(0)] = 0.0

    result = collect_results(design, [(counts, [fit_locus(design.models[0], counts.values[0])])])
    row = result.locus_qc.row(0, named=True)
    assert row["total_observed"] == pytest.approx(20.0)  # 2 unmasked bins
    assert row["total_observed_all_bins"] == pytest.approx(40.0)  # all 4
    assert row["masked_bin_fraction"] == pytest.approx(0.5)


def test_count_sample_records_pre_mask_totals(extdata):
    frame = pl.DataFrame(
        {
            "transcript_id": ["tx"],
            "contig": ["1"],
            "start": [1_000],
            "end": [11_000],
            "strand": ["+"],
        }
    )
    from dienr.counts import SignalFile, count_sample

    design = build_design(Annotation.from_frame(frame), bin_size=250, mask_tss=(0, 1000))
    counts = count_sample(design, "s", SignalFile(extdata / "test_double_strand_plus.bw", "+"))
    # The masked bins carried signal, so the pre-mask total must be larger.
    assert counts.totals_all_bins[0] > counts.values[0].sum()


# --- likelihood selection ---------------------------------------------------


def count_equivalent_counts(design, sample_id="s1", truth_scale=50.0, seed=0):
    """Counts recorded as if extracted with summary='sum'."""
    rng = np.random.default_rng(seed)
    values = [
        rng.poisson(design.models[i] @ np.full(design.models[i].shape[1], truth_scale)).astype(
            float
        )
        for i in range(design.n_loci)
    ]
    return SampleCounts(
        sample_id=sample_id,
        values=values,
        locus_status=[FitStatus.OK] * design.n_loci,
        design_digest=design.digest,
        totals_all_bins=[float(v.sum()) for v in values],
        provenance={
            "summary": "sum",
            "signal_files": [{"path": "x.bw", "strand": "+", "value_kind": "raw"}],
        },
    )


def test_poisson_likelihood_runs_end_to_end(design):
    from dienr.solvers import penalties

    counts = count_equivalent_counts(design)
    result = fit(design, counts, likelihood="poisson", penalty=penalties.ridge(0.01))
    assert result.summary.n_failures == 0
    assert result.provenance["likelihood"] == "poisson"
    assert result.provenance["penalty_type"] == "ridge"
    assert result.provenance["penalty_lambda2"] == 0.01


def test_negative_binomial_runs_end_to_end(design):
    counts = count_equivalent_counts(design, seed=1)
    result = fit(design, counts, likelihood="negative_binomial", dispersion=0.2)
    assert result.summary.n_failures == 0
    assert result.provenance["dispersion"] == 0.2


def test_mean_summary_is_rejected_for_count_likelihoods(design):
    """The default extraction is a per-base mean, for R parity. Guard it.

    Without this check the most likely way to misuse the Poisson solver is to
    build counts with the default settings and hand them straight over.
    """
    counts = count_equivalent_counts(design)
    counts.provenance["summary"] = "mean"
    with pytest.raises(ValueError, match="per-base density"):
        fit(design, counts, likelihood="poisson")


def test_transformed_signal_is_rejected_for_count_likelihoods(design):
    counts = count_equivalent_counts(design)
    counts.provenance["signal_files"] = [
        {"path": "x.bw", "strand": "+", "value_kind": "library_normalized"}
    ]
    with pytest.raises(ValueError, match="untransformed counts"):
        fit(design, counts, likelihood="poisson")


def test_nb_without_dispersion_is_rejected(design):
    counts = count_equivalent_counts(design)
    with pytest.raises(ValueError, match="requires an explicit dispersion"):
        fit(design, counts, likelihood="negative_binomial")


def test_penalty_on_the_legacy_solver_is_rejected(design):
    """The reference rejects a nonzero lambda too; do not silently ignore one."""
    from dienr.solvers import penalties

    with pytest.raises(ValueError, match="takes no penalty"):
        fit(design, synthetic_counts(design), penalty=penalties.ridge(0.1))


def test_unknown_likelihood_is_rejected(design):
    with pytest.raises(ValueError, match="likelihood must be"):
        fit(design, synthetic_counts(design), likelihood="gaussian")


def test_bin_exposure_is_legacy_incompatible(design):
    with pytest.raises(ValueError, match="only to the count likelihoods"):
        fit(design, synthetic_counts(design), use_bin_exposure=True)


def test_bin_exposure_is_passed_through(design):
    counts = count_equivalent_counts(design, seed=2)
    result = fit(design, counts, likelihood="poisson", use_bin_exposure=True)
    assert result.provenance["use_bin_exposure"] is True
    assert result.summary.n_failures == 0


def test_dispersion_estimated_from_a_poisson_fit_feeds_the_nb_fit(design):
    """The two-stage procedure section 4.3 prescribes, wired end to end."""
    from dienr.solvers import negative_binomial as nbmod

    counts = count_equivalent_counts(design, seed=3)
    poisson_result = fit(design, counts, likelihood="poisson")
    assert poisson_result.summary.n_failures == 0

    triples = []
    for locus_id in range(design.n_loci):
        abundance = (
            poisson_result.model_abundance.filter(pl.col("locus_id") == locus_id)
            .sort("model_index")["abundance_raw"]
            .to_numpy()
        )
        triples.append((design.models[locus_id], counts.values[locus_id], abundance))

    estimate = nbmod.estimate_dispersion(triples)
    # Poisson-simulated data: little or no overdispersion, so Poisson stands.
    assert estimate.dispersion < 0.05
