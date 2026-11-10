"""Top-level orchestration: many samples against one reusable design.

The three stages stay separate — `build_design`, `count_sample`, `fit` — so
annotation geometry is computed once and reused across samples and cohorts, and
so a workflow engine can cache each stage independently (plan section 11).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import polars as pl

from .counts import COUNT_LIKELIHOODS, SampleCounts, require_count_equivalent
from .design import Design
from .results import FitResult, collect_results
from .solvers import inference, poisson
from .solvers._glm import Exposure
from .solvers.legacy import LocusFit, Transform, fit_locus
from .solvers.penalties import NO_PENALTY, Penalty
from .status import FitStatus, is_failure

Likelihood = Literal["legacy", "poisson", "quasi_poisson"]

# Every count likelihood takes its point estimate from the Poisson solver.
# quasi-Poisson is not a different estimator -- it is the same fit with an
# honest variance -- so it must not be implemented as a separate objective.


def fit(
    design: Design,
    counts: SampleCounts | list[SampleCounts],
    *,
    likelihood: Likelihood = "legacy",
    transform: Transform = "log",
    penalty: Penalty = NO_PENALTY,
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
            parity), ``"poisson"`` or ``"quasi_poisson"``.

            ``"quasi_poisson"`` gives **the same abundances as** ``"poisson"``
            and differs only in the reported standard errors, which are scaled
            by ``sqrt(phi)`` for a dispersion pooled across loci. That is the
            recommended choice when the counts are overdispersed: a Poisson
            standard error understates the true spread by 23% at ``phi = 0.1``
            and 64% at ``phi = 1.0``, while quasi-Poisson tracks it to within a
            few percent and needs no second likelihood.

            There is no negative binomial: it was removed (plan Gate D). On a
            binary design it cannot move a point estimate at any dispersion,
            and quasi-Poisson matched its standard errors to within 1-4%.
        transform: Legacy objective; ignored by the count likelihoods.
        penalty: Named penalty. Only the count likelihoods accept one; the
            legacy solver has no penalty support, matching the reference.
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
            or count-likelihood input that is not count-equivalent.
    """
    samples = [counts] if isinstance(counts, SampleCounts) else list(counts)
    if likelihood != "legacy" and likelihood not in COUNT_LIKELIHOODS:
        raise ValueError(
            f"likelihood must be 'legacy', 'poisson' or 'quasi_poisson', got {likelihood!r}"
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
    if likelihood in COUNT_LIKELIHOODS:
        for sample in samples:
            # The reference's add_data hardcodes a per-base *mean* summary, so
            # count_sample defaults to it for parity; handing that to a count
            # likelihood is the likeliest misuse, and it is refused here.
            files = sample.provenance.get("signal_files", [])
            require_count_equivalent(
                likelihood,
                summary=str(sample.provenance.get("summary", "sum")),
                value_kinds=[str(entry.get("value_kind", "raw")) for entry in files],
                source=f"sample {sample.sample_id!r}",
            )

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

            fits.append(
                poisson.fit_locus(
                    models,
                    observed,
                    exposure=_sample_exposure(design, sample, locus_id, use_bin_exposure),
                    penalty=penalty,
                    inactive=inactive,
                    max_iterations=max_iterations,
                    min_unmasked_bins=min_unmasked_bins,
                    prior_status=prior,
                )
            )
        fitted.append((sample, fits))

    dispersion_record: dict[str, Any] = {}
    if likelihood in COUNT_LIKELIHOODS:
        # Quasi-Poisson needs a second pass: the dispersion is pooled over
        # every locus of every sample, which cannot be known while fitting one
        # locus. Section 4.3 forbids per-locus dispersion.
        estimate = (
            inference.POISSON_DISPERSION
            if likelihood == "poisson"
            else inference.estimate_quasi_poisson_dispersion(
                _dispersion_inputs(design, fitted, use_bin_exposure)
            )
        )
        dispersion_record = dict(estimate.describe())
        if likelihood == "poisson":
            dispersion_record["dispersion_method"] = "poisson_assumed"
        _attach_standard_errors(design, fitted, estimate.dispersion, use_bin_exposure)

    merged: dict[str, Any] = {"likelihood": likelihood}
    merged.update(dispersion_record)
    if likelihood == "legacy":
        merged["transform"] = transform
    else:
        merged.update(penalty.describe())
        merged["use_bin_exposure"] = use_bin_exposure
    if inactive_transcripts:
        merged["n_inactive_transcripts"] = len(inactive_transcripts)
    if provenance:
        merged.update(provenance)
    return collect_results(design, fitted, provenance=merged)


def _sample_exposure(
    design: Design, sample: SampleCounts, locus_id: int, use_bin_exposure: bool
) -> Exposure:
    """The exposure a given locus of a given sample was fitted under."""
    return Exposure(
        sample_scale=sample.exposure,
        bin_exposure=design.bin_lengths[locus_id] if use_bin_exposure else None,
    )


def _dispersion_inputs(
    design: Design,
    fitted: list[tuple[SampleCounts, list[LocusFit]]],
    use_bin_exposure: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Completed fits eligible to contribute to a pooled dispersion.

    Failed loci are excluded: their abundance is NaN, and an all-zero locus
    contributes no residual information about overdispersion either.
    """
    triples = []
    for sample, fits in fitted:
        for locus_id, fit in enumerate(fits):
            if is_failure(fit.status) or fit.status is FitStatus.ALL_ZERO:
                continue
            triples.append((design.models[locus_id], sample.values[locus_id], fit.abundance))
    return triples


def _attach_standard_errors(
    design: Design,
    fitted: list[tuple[SampleCounts, list[LocusFit]]],
    dispersion: float,
    use_bin_exposure: bool,
) -> None:
    """Fill in each fit's `standard_error` at the pooled dispersion.

    `LocusFit` is frozen, so this replaces entries in place rather than
    mutating them.
    """
    for sample, fits in fitted:
        for locus_id, fit in enumerate(fits):
            if is_failure(fit.status):
                continue
            errors = inference.standard_errors(
                design.models[locus_id],
                fit.abundance,
                exposure=_sample_exposure(design, sample, locus_id, use_bin_exposure),
                dispersion=dispersion,
            )
            fits[locus_id] = replace(fit, standard_error=errors)


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
