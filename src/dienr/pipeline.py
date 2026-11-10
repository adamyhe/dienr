"""Top-level orchestration: many samples against one reusable design.

The three stages stay separate — `build_design`, `count_sample`, `fit` — so
annotation geometry is computed once and reused across samples and cohorts, and
so a workflow engine can cache each stage independently (plan section 11).
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from itertools import repeat
from typing import Any, Literal

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

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
    blas_threads: int | None = 1,
    n_jobs: int = 1,
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
        blas_threads: Threads BLAS may use while fitting; None leaves the
            library's own setting alone. Defaults to 1 because every locus is
            a tiny independent problem: on a whole-genome run, multithreaded
            BLAS made fitting 1.26-1.46x *slower* (plan 14.4). Results do not
            depend on it.
        n_jobs: Worker processes fitting loci in parallel. Every locus is an
            independent problem fitted by the same function either way, so the
            results are identical for any value; 1 fits in this process. If
            other Python threads are running when `fit` is called, workers
            start by `forkserver` and a calling script needs an
            ``if __name__ == "__main__":`` guard (see `_start_context`).

    Returns:
        A `FitResult` covering all samples.

    Raises:
        ValueError: For an unknown likelihood, a penalty on the legacy solver,
            or count-likelihood input that is not count-equivalent.
    """
    if blas_threads is not None and blas_threads < 1:
        raise ValueError(f"blas_threads must be >= 1 or None, got {blas_threads}")
    limit = (
        threadpool_limits(limits=blas_threads, user_api="blas")
        if blas_threads is not None
        else nullcontext()
    )
    with limit:
        return _fit(
            design,
            counts,
            likelihood=likelihood,
            transform=transform,
            penalty=penalty,
            use_bin_exposure=use_bin_exposure,
            inactive_transcripts=inactive_transcripts,
            max_iterations=max_iterations,
            min_unmasked_bins=min_unmasked_bins,
            provenance=provenance,
            blas_threads=blas_threads,
            n_jobs=n_jobs,
        )


def _fit(
    design: Design,
    counts: SampleCounts | list[SampleCounts],
    *,
    likelihood: Likelihood,
    transform: Transform,
    penalty: Penalty,
    use_bin_exposure: bool,
    inactive_transcripts: set[str] | None,
    max_iterations: int,
    min_unmasked_bins: int,
    provenance: dict[str, Any] | None,
    blas_threads: int | None,
    n_jobs: int,
) -> FitResult:
    """`fit`'s body, run inside its BLAS limit; workers apply the same limit."""
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
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

    options = _FitOptions(
        likelihood=likelihood,
        transform=transform,
        penalty=penalty,
        max_iterations=max_iterations,
        min_unmasked_bins=min_unmasked_bins,
    )
    tasks = [
        _LocusTask(
            models=design.models[locus_id],
            observed=sample.values[locus_id],
            inactive=pinned[locus_id],
            prior=sample.locus_status[locus_id],
            exposure=(
                _sample_exposure(design, sample, locus_id, use_bin_exposure)
                if likelihood != "legacy"
                else None
            ),
        )
        for sample in samples
        for locus_id in range(design.n_loci)
    ]
    results = _run_tasks(tasks, options, n_jobs=n_jobs, blas_threads=blas_threads)
    fitted: list[tuple[SampleCounts, list[LocusFit]]] = [
        (sample, results[index * design.n_loci : (index + 1) * design.n_loci])
        for index, sample in enumerate(samples)
    ]

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


@dataclass(frozen=True)
class _FitOptions:
    """Everything a locus fit needs besides its own data. Picklable."""

    likelihood: str
    transform: Transform
    penalty: Penalty
    max_iterations: int
    min_unmasked_bins: int


@dataclass(frozen=True)
class _LocusTask:
    models: np.ndarray
    observed: np.ndarray
    inactive: np.ndarray | None
    prior: FitStatus
    exposure: Exposure | None


def _fit_task(task: _LocusTask, options: _FitOptions) -> LocusFit:
    if options.likelihood == "legacy":
        return fit_locus(
            task.models,
            task.observed,
            transform=options.transform,
            inactive=task.inactive,
            max_iterations=options.max_iterations,
            min_unmasked_bins=options.min_unmasked_bins,
            prior_status=task.prior,
        )
    return poisson.fit_locus(
        task.models,
        task.observed,
        exposure=task.exposure,
        penalty=options.penalty,
        inactive=task.inactive,
        max_iterations=options.max_iterations,
        min_unmasked_bins=options.min_unmasked_bins,
        prior_status=task.prior,
    )


def _fit_chunk(
    tasks: list[_LocusTask], options: _FitOptions, blas_threads: int | None
) -> list[LocusFit]:
    """Worker entry point: fit a chunk of loci under the same BLAS limit."""
    if blas_threads is None:
        return [_fit_task(task, options) for task in tasks]
    with threadpool_limits(limits=blas_threads, user_api="blas"):
        return [_fit_task(task, options) for task in tasks]


#: Chunks per worker. More than one lets fast workers take up the slack of
#: slow ones (a few loci with thousands of bins and dozens of models dominate
#: the cost); many more would spend the time pickling instead.
_CHUNKS_PER_WORKER = 8


def _run_tasks(
    tasks: list[_LocusTask], options: _FitOptions, *, n_jobs: int, blas_threads: int | None
) -> list[LocusFit]:
    """Fit every task, in order, in this process or across `n_jobs` workers.

    Each locus's data is sent to exactly one worker, once; the design is never
    copied whole. Chunks are contiguous runs of tasks balanced by an estimate
    of their cost (bins x models^2), and results come back in task order.
    """
    if n_jobs == 1 or len(tasks) < 2:
        return [_fit_task(task, options) for task in tasks]
    cost = np.array([t.models.shape[0] * max(t.models.shape[1], 1) ** 2 for t in tasks], float)
    n_chunks = min(len(tasks), n_jobs * _CHUNKS_PER_WORKER)
    edges = np.searchsorted(np.cumsum(cost), np.linspace(0, cost.sum(), n_chunks + 1)[1:-1])
    bounds = [0, *sorted(set(int(e) + 1 for e in edges)), len(tasks)]
    chunks = [tasks[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]
    with warnings.catch_warnings():
        # Python 3.12+ warns before any fork in a process with more than one OS
        # thread, and NumPy's OpenBLAS starts idle worker threads at import, so
        # it would fire on every call. `_start_context` only forks when no
        # other *Python* thread exists; the native pools left are OpenBLAS's,
        # which reinitializes itself in the child via pthread_atfork, and the
        # workers never call numba or figwig.
        warnings.filterwarnings(
            "ignore",
            message=r".*use of fork\(\) may lead to deadlocks",
            category=DeprecationWarning,
        )
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_start_context()) as pool:
            parts = list(pool.map(_fit_chunk, chunks, repeat(options), repeat(blas_threads)))
    return [fit for part in parts for fit in part]


def _start_context() -> mp.context.BaseContext:
    """How to start fitting workers.

    `fork` when it is safe, because it is the only method that doesn't
    re-import the caller's ``__main__``: with `forkserver` or `spawn`, a script
    calling ``fit(n_jobs=4)`` without an ``if __name__ == "__main__":`` guard
    crashes as soon as the pool starts. Forking while another thread is mid-work
    can deadlock the child (Python 3.14 stopped defaulting to it for that
    reason), so it is used only when every other Python thread is a
    `concurrent.futures` pool worker. In practice those are figwig's: it keeps
    a shared reader pool between reads, so any session that counted before
    fitting still has it. Its workers are idle once `count` returns, the
    fitting workers never call figwig, and figwig rebuilds its pool in a new
    process (it is keyed by PID). Otherwise `forkserver` (or `spawn`), and the
    caller needs the guard. The `dienr fit` command never loads figwig at all.
    """
    methods = mp.get_all_start_methods()
    others = [t for t in threading.enumerate() if t is not threading.current_thread()]
    if "fork" in methods and all(t.name.startswith("ThreadPoolExecutor-") for t in others):
        return mp.get_context("fork")
    return mp.get_context("forkserver" if "forkserver" in methods else "spawn")


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
