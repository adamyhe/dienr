"""Top-level orchestration: many samples against one reusable design.

The three stages stay separate — `build_design`, `count_sample`, `fit` — so
annotation geometry is computed once and reused across samples and cohorts, and
so a workflow engine can cache each stage independently (plan section 11).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .counts import SampleCounts
from .design import Design
from .results import FitResult, collect_results
from .solvers.legacy import LocusFit, Transform, fit_locus
from .status import FitStatus


def fit(
    design: Design,
    counts: SampleCounts | list[SampleCounts],
    *,
    transform: Transform = "log",
    inactive_transcripts: set[str] | None = None,
    max_iterations: int = 15_000,
    min_unmasked_bins: int = 1,
    provenance: dict[str, Any] | None = None,
) -> FitResult:
    """Fit every locus of every sample and assemble the output tables.

    Args:
        design: Reusable locus geometry.
        counts: One or more samples counted against `design`.
        transform: Legacy objective to use.
        inactive_transcripts: Transcripts whose abundance is pinned to zero. A
            model is only pinned when **every** transcript mapped to it is
            listed, since one active member makes the shared model active.
        max_iterations: Optimizer iteration cap per locus.
        min_unmasked_bins: Minimum bins with model support to attempt a fit.
        provenance: Extra manifest fields.

    Returns:
        A `FitResult` covering all samples.
    """
    samples = [counts] if isinstance(counts, SampleCounts) else list(counts)
    pinned = _pinned_models(design, inactive_transcripts)

    fitted: list[tuple[SampleCounts, list[LocusFit]]] = []
    for sample in samples:
        fits = [
            fit_locus(
                design.models[locus_id],
                sample.values[locus_id],
                transform=transform,
                inactive=pinned[locus_id],
                max_iterations=max_iterations,
                min_unmasked_bins=min_unmasked_bins,
                prior_status=sample.locus_status[locus_id],
            )
            for locus_id in range(design.n_loci)
        ]
        fitted.append((sample, fits))

    merged: dict[str, Any] = {"transform": transform, "likelihood": "legacy_least_squares"}
    if inactive_transcripts:
        merged["n_inactive_transcripts"] = len(inactive_transcripts)
    if provenance:
        merged.update(provenance)
    return collect_results(design, fitted, provenance=merged)


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
