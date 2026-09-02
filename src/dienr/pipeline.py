"""Top-level orchestration: many samples against one reusable design.

The three stages stay separate — `build_design`, `count_sample`, `fit` — so
annotation geometry is computed once and reused across samples and cohorts, and
so a workflow engine can cache each stage independently (plan section 11).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

from .counts import SampleCounts
from .design import Design
from .results import FitResult, collect_results
from .solvers import negative_binomial, poisson
from .solvers._glm import Exposure
from .solvers.legacy import LocusFit, Transform, fit_locus
from .solvers.penalties import NO_PENALTY, Penalty
from .status import FitStatus

Likelihood = Literal["legacy", "poisson", "negative_binomial"]

#: Likelihoods that need raw counts rather than a per-base mean.
_COUNT_LIKELIHOODS = ("poisson", "negative_binomial")


def fit(
    design: Design,
    counts: SampleCounts | list[SampleCounts],
    *,
    likelihood: Likelihood = "legacy",
    transform: Transform = "log",
    penalty: Penalty = NO_PENALTY,
    dispersion: float | None = None,
    use_bin_exposure: bool = False,
    inactive_transcripts: set[str] | None = None,
    max_iterations: int = 15_000,
    min_unmasked_bins: int = 1,
    provenance: dict[str, Any] | None = None,
) -> FitResult:
    """Fit every locus of every sample and assemble the output tables.

    Args:
        design: Reusable locus geometry.
        counts: One or more samples counted against `design`.
        likelihood: ``"legacy"`` (default, and the only one with established R
            parity), ``"poisson"``, or ``"negative_binomial"``.
        transform: Legacy objective; ignored by the count likelihoods.
        penalty: Named penalty. Only the count likelihoods accept one; the
            legacy solver has no penalty support, matching the reference.
        dispersion: Shared ``phi``, required for ``negative_binomial`` and
            normally from `dienr.solvers.negative_binomial.estimate_dispersion`.
        use_bin_exposure: Pass each locus's effective bin lengths as exposure.
            Only meaningful for count likelihoods, where a partial terminal bin
            genuinely had less opportunity to accumulate reads.
        inactive_transcripts: Transcripts whose abundance is pinned to zero. A
            model is only pinned when **every** transcript mapped to it is
            listed, since one active member makes the shared model active.
        max_iterations: Optimizer iteration cap per locus.
        min_unmasked_bins: Minimum bins with model support to attempt a fit.
        provenance: Extra manifest fields.

    Returns:
        A `FitResult` covering all samples.

    Raises:
        ValueError: For an unknown likelihood, a penalty on the legacy solver,
            a missing dispersion for NB, or count-likelihood input that is not
            count-equivalent.
    """
    samples = [counts] if isinstance(counts, SampleCounts) else list(counts)
    if likelihood not in ("legacy", *_COUNT_LIKELIHOODS):
        raise ValueError(
            f"likelihood must be 'legacy', 'poisson' or 'negative_binomial', got {likelihood!r}"
        )
    if likelihood == "legacy":
        if penalty.is_active:
            raise ValueError(
                "the legacy least-squares solver takes no penalty; the reference "
                "implementation rejects a nonzero lambda too. Use likelihood="
                "'poisson' for a penalized fit."
            )
        if use_bin_exposure:
            raise ValueError(
                "use_bin_exposure applies only to the count likelihoods; the "
                "legacy objective has no exposure term"
            )
    if likelihood == "negative_binomial" and dispersion is None:
        raise ValueError(
            "negative_binomial requires an explicit dispersion. Estimate one "
            "across loci with dienr.solvers.negative_binomial.estimate_dispersion; "
            "section 4.3 forbids fitting it per locus."
        )
    if likelihood in _COUNT_LIKELIHOODS:
        for sample in samples:
            _require_count_equivalent(sample, likelihood)

    pinned = _pinned_models(design, inactive_transcripts)

    fitted: list[tuple[SampleCounts, list[LocusFit]]] = []
    for sample in samples:
        fits = []
        for locus_id in range(design.n_loci):
            models = design.models[locus_id]
            observed = sample.values[locus_id]
            inactive = pinned[locus_id]
            prior = sample.locus_status[locus_id]

            if likelihood == "legacy":
                fits.append(
                    fit_locus(
                        models,
                        observed,
                        transform=transform,
                        inactive=inactive,
                        max_iterations=max_iterations,
                        min_unmasked_bins=min_unmasked_bins,
                        prior_status=prior,
                    )
                )
                continue

            exposure = Exposure(
                sample_scale=sample.exposure,
                bin_exposure=design.bin_lengths[locus_id] if use_bin_exposure else None,
            )
            if likelihood == "poisson":
                fits.append(
                    poisson.fit_locus(
                        models,
                        observed,
                        exposure=exposure,
                        penalty=penalty,
                        inactive=inactive,
                        max_iterations=max_iterations,
                        min_unmasked_bins=min_unmasked_bins,
                        prior_status=prior,
                    )
                )
            else:
                fits.append(
                    negative_binomial.fit_locus(
                        models,
                        observed,
                        float(dispersion),  # type: ignore[arg-type]
                        exposure=exposure,
                        penalty=penalty,
                        inactive=inactive,
                        max_iterations=max_iterations,
                        min_unmasked_bins=min_unmasked_bins,
                        prior_status=prior,
                    )
                )
        fitted.append((sample, fits))

    merged: dict[str, Any] = {"likelihood": likelihood}
    if likelihood == "legacy":
        merged["transform"] = transform
    else:
        merged.update(penalty.describe())
        merged["use_bin_exposure"] = use_bin_exposure
    if likelihood == "negative_binomial":
        merged["dispersion"] = float(dispersion)  # type: ignore[arg-type]
    if inactive_transcripts:
        merged["n_inactive_transcripts"] = len(inactive_transcripts)
    if provenance:
        merged.update(provenance)
    return collect_results(design, fitted, provenance=merged)


def _require_count_equivalent(sample: SampleCounts, likelihood: str) -> None:
    """Reject input that is not count-equivalent for a count likelihood.

    Two ways it can fail, and the second is the one that actually bites: the
    reference's `add_data` hardcodes a per-base *mean* summary, so
    `count_sample` defaults to it for parity. Feeding that to a Poisson would
    hand a count likelihood a density divided by bin width.
    """
    summary = str(sample.provenance.get("summary", "sum"))
    files = sample.provenance.get("signal_files", [])
    kinds = {str(entry.get("value_kind", "raw")) for entry in files}
    if summary != "sum":
        raise ValueError(
            f"sample {sample.sample_id!r} was counted with summary={summary!r}, "
            f"which is a per-base density, but likelihood {likelihood!r} needs "
            "counts. Re-run count_sample with summary='sum'."
        )
    transformed = kinds - {"raw"}
    if transformed:
        raise ValueError(
            f"sample {sample.sample_id!r} has value_kind {sorted(transformed)}, "
            f"but likelihood {likelihood!r} requires untransformed counts."
        )


def _pinned_models(
    design: Design, inactive_transcripts: set[str] | None
) -> list[np.ndarray | None]:
    """Per-locus boolean arrays marking models to pin at zero.

    A model is pinned only when every transcript in its equivalence class is
    inactive. Pinning a model because one of several transcripts is inactive
    would suppress signal that another member explains.
    """
    if not inactive_transcripts:
        return [None] * design.n_loci

    flagged = design.transcript_map.with_columns(
        _inactive=pl.col("transcript_id").is_in(list(inactive_transcripts))
    )
    per_model = flagged.group_by("locus_id", "model_index").agg(
        all_inactive=pl.col("_inactive").all()
    )

    pinned: list[np.ndarray | None] = [
        np.zeros(design.models[i].shape[1], dtype=bool) for i in range(design.n_loci)
    ]
    for row in per_model.iter_rows(named=True):
        if row["all_inactive"]:
            pinned[row["locus_id"]][row["model_index"]] = True
    return pinned


def status_report(result: FitResult) -> pl.DataFrame:
    """Status tally as a table, for logging or a CLI summary."""
    summary = result.summary
    return pl.DataFrame(
        {
            "status": list(summary.by_status),
            "n": [summary.by_status[k] for k in summary.by_status],
        }
    ).sort("n", descending=True)


def all_statuses_accounted_for(result: FitResult) -> bool:
    """True when every QC row carries a recognized status.

    A guard against a status string leaking into the output that
    `FitStatus` does not define.
    """
    known = {str(s) for s in FitStatus}
    return set(result.locus_qc["solver_status"].unique().to_list()) <= known
