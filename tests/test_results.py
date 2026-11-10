"""Output schemas, round-tripping, and the status accounting."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from dienr.annotation import Annotation
from dienr.counts import SampleCounts
from dienr.design import DesignConfig, build_design
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


class _StubDesign:
    """A one-locus Design carrying a caller-supplied matrix.

    Needed because an ill-conditioned but full-rank design cannot arise from a
    real annotation: `bin_operation` rounds occupancies to integers, so two
    columns are either identical or differ by a whole bin. Constructing the
    matrix directly is the only way to reach the conditioning branch.
    """

    def __init__(self, models: np.ndarray) -> None:
        n_bins = models.shape[0]
        self.models = [models]
        self.masks = [np.zeros(n_bins, dtype=bool)]
        self.bin_lengths = [np.full(n_bins, 250.0)]
        self.digest = "stub-design"
        self.annotation_digest = "stub-annotation"
        self.n_loci = 1
        self.config = DesignConfig()
        self.loci = pl.DataFrame(
            {
                "locus_id": pl.Series([0], dtype=pl.UInt32),
                "locus_name": ["chr1:+:0-100"],
                "contig": ["chr1"],
                "strand": ["+"],
                "start": [0],
                "end": [100],
                "n_transcripts": pl.Series([models.shape[1]], dtype=pl.UInt32),
            }
        )
        self.model_table = pl.DataFrame(
            {
                "locus_id": pl.Series([0] * models.shape[1], dtype=pl.UInt32),
                "model_index": pl.Series(range(models.shape[1]), dtype=pl.UInt32),
                "model_id": [f"stub{i}" for i in range(models.shape[1])],
                "equivalence_class_size": pl.Series([1] * models.shape[1], dtype=pl.UInt32),
                "has_support": [True] * models.shape[1],
            }
        )
        self.transcript_map = self.model_table.select(
            pl.lit("tx").alias("transcript_id"),
            pl.lit("g").alias("gene_id"),
            "locus_id",
            "model_index",
            "model_id",
            "equivalence_class_size",
            pl.lit(True).alias("exactly_identifiable"),
            "has_support",
        )

    def bin_count(self, locus_id: int) -> int:
        return self.models[locus_id].shape[0]

    def mask(self, locus_id: int) -> np.ndarray:
        return self.masks[locus_id]

    def masked_fraction(self, locus_id: int) -> float:
        return 0.0


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


def test_mean_summary_is_rejected_for_count_likelihoods(design):
    """The default extraction is a per-base mean, for R parity. Guard it.

    Without this check the most likely way to misuse the Poisson solver is to
    build counts with the default settings and hand them straight over.
    """
    counts = count_equivalent_counts(design)
    counts.provenance["summary"] = "mean"
    for likelihood in ("poisson", "quasi_poisson"):
        with pytest.raises(ValueError, match="per-base density"):
            fit(design, counts, likelihood=likelihood)


def test_transformed_signal_is_rejected_for_count_likelihoods(design):
    counts = count_equivalent_counts(design)
    counts.provenance["signal_files"] = [
        {"path": "x.bw", "strand": "+", "value_kind": "library_normalized"}
    ]
    with pytest.raises(ValueError, match="untransformed counts"):
        fit(design, counts, likelihood="poisson")


def test_penalty_on_the_legacy_solver_is_rejected(design):
    """The reference rejects a nonzero lambda too; do not silently ignore one."""
    from dienr.solvers import penalties

    with pytest.raises(ValueError, match="takes no penalty"):
        fit(design, synthetic_counts(design), penalty=penalties.ridge(0.1))


@pytest.mark.parametrize("likelihood", ["gaussian", "negative_binomial"])
def test_unknown_likelihood_is_rejected(design, likelihood):
    """Including the removed NB solver: a stale config must fail, not fall through."""
    with pytest.raises(ValueError, match="likelihood must be"):
        fit(design, synthetic_counts(design), likelihood=likelihood)


def test_bin_exposure_is_legacy_incompatible(design):
    with pytest.raises(ValueError, match="only to the count likelihoods"):
        fit(design, synthetic_counts(design), use_bin_exposure=True)


def test_bin_exposure_is_passed_through(design):
    counts = count_equivalent_counts(design, seed=2)
    result = fit(design, counts, likelihood="poisson", use_bin_exposure=True)
    assert result.provenance["use_bin_exposure"] is True
    assert result.summary.n_failures == 0


def test_ill_conditioned_locus_is_escalated_from_ok():
    """The conditioning escalation for near-collinear, full-rank designs.

    Distinct from rank deficiency: the matrix has full numerical rank, so the
    solution is unique, but the allocation is unstable. Reporting plain 'ok'
    would overstate what was learned.
    """
    n_bins = 60
    models = np.ones((n_bins, 2))
    models[0, 1] = 1.0 + 1e-11  # full rank, appallingly conditioned
    design = _StubDesign(models)

    counts = SampleCounts(
        sample_id="s1",
        values=[models @ np.array([3.0, 4.0])],
        locus_status=[FitStatus.OK],
        design_digest=design.digest,
        totals_all_bins=[0.0],
    )
    fits = [fit_locus(models, counts.values[0], transform="identity")]
    result = collect_results(design, [(counts, fits)])
    assert result.locus_qc["solver_status"][0] == "ill_conditioned"
    assert result.locus_qc["matrix_rank"][0] == 2


def test_empty_result_has_an_empty_summary(design):
    """`FitResult.summary` on a run with no loci must not divide by zero."""
    empty = collect_results(design, [])
    assert empty.summary.n_fits == 0
    assert empty.summary.failure_fraction == 0.0
    assert empty.summary.by_status == {}


def test_extra_provenance_is_merged(design):
    result = fit(design, synthetic_counts(design), provenance={"run_label": "smoke"})
    assert result.provenance["run_label"] == "smoke"


# --- quasi-Poisson end to end ----------------------------------------------


def test_quasi_poisson_gives_the_same_abundances_as_poisson(design):
    """It is a variance model, not a different estimator.

    If the abundances ever diverge, quasi-Poisson has been implemented as a
    separate objective, which it must not be.
    """
    counts = count_equivalent_counts(design, seed=10)
    plain = fit(design, counts, likelihood="poisson")
    quasi = fit(design, counts, likelihood="quasi_poisson")
    np.testing.assert_allclose(
        plain.model_abundance["abundance_raw"].to_numpy(),
        quasi.model_abundance["abundance_raw"].to_numpy(),
        rtol=1e-12,
    )


def test_quasi_poisson_standard_errors_scale_by_root_dispersion(design):
    counts = count_equivalent_counts(design, seed=11)
    plain = fit(design, counts, likelihood="poisson")
    quasi = fit(design, counts, likelihood="quasi_poisson")

    phi = quasi.provenance["dispersion"]
    assert plain.provenance["dispersion"] == 1.0
    assert plain.provenance["dispersion_method"] == "poisson_assumed"
    assert quasi.provenance["dispersion_method"] == "quasi_poisson_pearson_pooled"

    ratio = (
        quasi.model_abundance["abundance_standard_error"].to_numpy()
        / plain.model_abundance["abundance_standard_error"].to_numpy()
    )
    finite = ratio[np.isfinite(ratio)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, np.sqrt(phi), rtol=1e-9)


def test_standard_errors_are_reported_in_the_output(design):
    counts = count_equivalent_counts(design, seed=12)
    result = fit(design, counts, likelihood="quasi_poisson")
    errors = result.model_abundance["abundance_standard_error"]
    assert errors.null_count() == 0
    assert np.isfinite(errors.to_numpy()).any()


def test_legacy_reports_no_standard_error(design):
    """A sum of squares implies no variance model, so inventing one would lie."""
    result = fit(design, synthetic_counts(design))
    assert result.model_abundance["abundance_standard_error"].null_count() == (
        result.model_abundance.height
    )


def test_failed_locus_has_no_standard_error(design):
    counts = count_equivalent_counts(design, seed=13)
    counts.locus_status[0] = FitStatus.MISSING_CONTIG
    counts.values[0] = np.full_like(counts.values[0], np.nan)
    result = fit(design, counts, likelihood="quasi_poisson")
    failed = result.model_abundance.filter(pl.col("locus_id") == 0)
    assert failed["abundance_standard_error"].is_null().all()


def test_dispersion_pools_across_samples(design):
    """The dispersion is one number for the run, not one per sample.

    Pooling across every locus of every sample is what makes it stable, and is
    why the fit needs a second pass.
    """
    samples = [count_equivalent_counts(design, f"s{i}", seed=20 + i) for i in range(3)]
    result = fit(design, samples, likelihood="quasi_poisson")
    assert result.provenance["dispersion_n_bins"] > 0
    # One dispersion, applied to every sample's errors.
    assert isinstance(result.provenance["dispersion"], float)
    assert result.model_abundance["sample_id"].n_unique() == 3


# --- BLAS thread cap (plan 14.4) ----------------------------------------------


def test_blas_cap_does_not_change_results(design):
    counts = synthetic_counts(design)
    capped = fit(design, counts, likelihood="poisson", blas_threads=1)
    free = fit(design, counts, likelihood="poisson", blas_threads=None)
    assert capped.model_abundance.equals(free.model_abundance)


def test_blas_cap_is_active_while_fitting(design, monkeypatch):
    from threadpoolctl import threadpool_info

    import dienr.pipeline as pipeline

    seen = []
    original = pipeline.poisson.fit_locus

    def spy(*args, **kwargs):
        seen.extend(i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline.poisson, "fit_locus", spy)
    fit(design, synthetic_counts(design), likelihood="poisson", blas_threads=1)
    assert seen and set(seen) == {1}


def test_blas_threads_must_be_positive(design):
    with pytest.raises(ValueError, match="blas_threads"):
        fit(design, synthetic_counts(design), blas_threads=0)


# --- parallel fitting ----------------------------------------------------------


@pytest.mark.parametrize("likelihood", ["legacy", "poisson", "quasi_poisson"])
def test_parallel_fit_is_identical_to_serial(design, likelihood):
    """Each locus is fitted by the same function in either case."""
    samples = [synthetic_counts(design, "a"), synthetic_counts(design, "b", scale=3.0)]
    if likelihood != "legacy":
        samples = [replace(s, provenance={**s.provenance, "summary": "sum"}) for s in samples]
    serial = fit(design, samples, likelihood=likelihood)
    parallel = fit(design, samples, likelihood=likelihood, n_jobs=3)
    assert parallel.model_abundance.equals(serial.model_abundance)
    assert parallel.locus_qc.equals(serial.locus_qc)


def test_n_jobs_must_be_positive(design):
    with pytest.raises(ValueError, match="n_jobs"):
        fit(design, synthetic_counts(design), n_jobs=0)


def test_parallel_fit_works_from_a_script_without_a_main_guard(tmp_path):
    """Most scripts don't guard ``__main__``; workers mustn't re-import it."""
    import subprocess
    import sys
    from pathlib import Path

    from .fixtures.bigwig import write_intervals

    # Count first, multithreaded, as in a real session; then idle pool threads
    # are alive when fit starts.
    rows = [
        {
            "transcript_id": f"t{i}",
            "contig": "c",
            "start": 30_000 * i,
            "end": 30_000 * i + 20_000 + 1_000 * (i % 3),
            "strand": "+",
        }
        for i in range(60)
    ]
    plus = write_intervals(tmp_path / "p.bw", {"c": 2_000_000}, [("c", 0, 1_900_000, 2.0)])
    minus = write_intervals(tmp_path / "m.bw", {"c": 2_000_000}, [("c", 0, 1_900_000, 0.1)])
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import threading, polars as pl, dienr\n"
        f"rows = {rows!r}\n"
        "d = dienr.build_design(dienr.Annotation.from_frame(pl.DataFrame(rows)), bin_size=250)\n"
        f"signal = (dienr.SignalFile({str(plus)!r}, '+'), dienr.SignalFile({str(minus)!r}, '-'))\n"
        "c = dienr.count_sample(d, 's', signal, summary='sum', n_jobs=4)\n"
        # figwig keeps a shared reader pool alive between reads; small reads
        # may not start it, so hold an idle executor open the same way.
        "from concurrent.futures import ThreadPoolExecutor\n"
        "idle = ThreadPoolExecutor(2); idle.submit(int).result(); idle.submit(int).result()\n"
        "assert threading.active_count() > 1\n"
        "print('threads after count:', threading.active_count())\n"
        "r = dienr.fit(d, c, likelihood='poisson', n_jobs=2)\n"
        "assert r.model_abundance.height > 0\n"
    )
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert done.stdout.count("threads after count") == 1  # the script ran once, not per worker


def test_log_offset_is_recorded_and_only_for_the_log_objective(design):
    result = fit(design, synthetic_counts(design), log_offset=0.01)
    assert result.provenance["log_offset"] == 0.01
    with pytest.raises(ValueError, match="log_offset"):
        fit(design, synthetic_counts(design), likelihood="poisson", log_offset=0.01)
