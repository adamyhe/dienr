"""Legacy least-squares solvers, for parity with the published DENR method.

Two objectives over a nonnegative abundance vector ``a``:

.. math::
    L_\\text{identity}(a) = \\sum_b [y_b - (Ma)_b]^2

.. math::
    L_\\text{log}(a) = \\sum_b [\\log(y_b + \\epsilon) - \\log((Ma)_b + \\epsilon)]^2

with :math:`\\epsilon = 10^{-3}`. ``log`` is the reference implementation's
default because residuals on that scale are closer to Gaussian, which is what a
sum-of-squares objective assumes.

Unlike the reference, every fit returns its convergence status, iteration count,
final gradient norm, objective value and optimizer message. Discarding those is
what allows a numerical failure to be mistaken for a biological zero.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize

from ..status import FitStatus, is_failure
from ._lbfgsb import minimize_bounded

Transform = Literal["identity", "log"]

#: Offset inside the log transform, matching the published method.
LOG_OFFSET = 1e-3

#: Upper bound standing in for "free". Large but finite, so L-BFGS-B stays in a
#: box and cannot walk off to values that overflow the objective.
_FREE_UPPER = 1e9

#: Upper bound for a model pinned inactive. Exactly zero makes the log
#: objective's gradient undefined at the bound, so a tiny positive value is used
#: and the estimate is snapped to zero afterwards.
_INACTIVE_UPPER = 1e-100


@dataclass(frozen=True)
class LocusFit:
    """Result of fitting one locus for one sample.

    Attributes:
        abundance: Estimated abundance per identifiable model. All-NaN when
            `status` is a failure.
        status: Outcome of the fit.
        objective: Final objective value.
        gradient_norm: Infinity norm of the final gradient.
        iterations: Optimizer iterations used.
        converged: Whether the optimizer met its convergence criteria.
        message: Optimizer message, verbatim.
        failure_reason: Populated when `status` is a failure.
    """

    abundance: np.ndarray
    status: FitStatus
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    message: str
    failure_reason: str | None = None


def predict(models: np.ndarray, abundance: np.ndarray) -> np.ndarray:
    """Predicted signal ``M @ a`` for one locus."""
    return models @ abundance


def objective(
    abundance: np.ndarray,
    models: np.ndarray,
    observed: np.ndarray,
    transform: Transform,
) -> float:
    """Legacy sum-of-squares objective."""
    predicted = models @ abundance
    if transform == "identity":
        residual = observed - predicted
    elif transform == "log":
        residual = np.log(observed + LOG_OFFSET) - np.log(predicted + LOG_OFFSET)
    else:
        raise ValueError(f"transform must be 'identity' or 'log', got {transform!r}")
    return float(residual @ residual)


def gradient(
    abundance: np.ndarray,
    models: np.ndarray,
    observed: np.ndarray,
    transform: Transform,
) -> np.ndarray:
    """Analytic gradient of `objective` with respect to `abundance`."""
    predicted = models @ abundance
    if transform == "identity":
        return -2.0 * (models.T @ (observed - predicted))
    if transform == "log":
        residual = np.log(observed + LOG_OFFSET) - np.log(predicted + LOG_OFFSET)
        return -2.0 * (models.T @ (residual / (predicted + LOG_OFFSET)))
    raise ValueError(f"transform must be 'identity' or 'log', got {transform!r}")


def _value_and_grad(
    models: np.ndarray, observed: np.ndarray, transform: Transform
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Build a fused objective-and-gradient closure for one locus.

    Two profiled savings over calling `objective` and `gradient` separately,
    which is what the optimizer would otherwise do at every iteration:

    - ``M @ a`` is computed once per evaluation instead of twice. It is 16-45%
      of a single objective call depending on locus size.
    - ``log(observed + eps)`` is hoisted out of the loop. It does not depend on
      the abundance, yet accounted for roughly a quarter of every call.

    The standalone `objective` and `gradient` remain the readable reference
    definitions, and the equivalence is asserted in the tests.
    """
    if transform == "identity":

        def fused(abundance: np.ndarray) -> tuple[float, np.ndarray]:
            residual = observed - models @ abundance
            return float(residual @ residual), -2.0 * (models.T @ residual)

    elif transform == "log":
        log_observed = np.log(observed + LOG_OFFSET)

        def fused(abundance: np.ndarray) -> tuple[float, np.ndarray]:
            shifted = models @ abundance + LOG_OFFSET
            residual = log_observed - np.log(shifted)
            return float(residual @ residual), -2.0 * (models.T @ (residual / shifted))

    else:
        raise ValueError(f"transform must be 'identity' or 'log', got {transform!r}")

    return fused


def _fit_single_model(
    models: np.ndarray, observed: np.ndarray, transform: Transform
) -> tuple[np.ndarray, float, int, bool, str]:
    """Fit a locus that has exactly one model, without the general optimizer.

    Profiling showed that on a median realistic locus (64 bins, 1 model) about
    91% of the runtime was L-BFGS-B scaffolding rather than arithmetic, and
    single-model loci are the common case in a real annotation. This path
    avoids that overhead entirely.

    For the identity transform the minimizer is available in closed form: the
    objective is a nonnegative-constrained one-dimensional least squares, so
    ``a = max(0, m . y / m . m)``. For the log transform the objective is
    one-dimensional and convex but has no closed form, so a bounded scalar
    search is used, which is still far cheaper than L-BFGS-B.

    Returns:
        ``(abundance, objective, iterations, converged, message)``.
    """
    column = models[:, 0]
    denominator = float(column @ column)
    if denominator <= 0:  # pragma: no cover - caller checks model support first
        return np.zeros(1), float("nan"), 0, False, "model has no support"

    if transform == "identity":
        value = max(0.0, float(column @ observed) / denominator)
        estimate = np.array([value])
        return (
            estimate,
            objective(estimate, models, observed, transform),
            1,
            True,
            "closed-form nonnegative least squares",
        )

    # Bracket the search using the identity-transform solution, which is the
    # right order of magnitude, then widen generously.
    scale = max(1.0, float(column @ observed) / denominator)
    fused = _value_and_grad(models, observed, transform)
    result = optimize.minimize_scalar(
        lambda a: fused(np.array([a]))[0],
        bounds=(0.0, scale * 1e3),
        method="bounded",
        options={"xatol": 1e-10},
    )
    estimate = np.array([max(0.0, float(result.x))])
    return (
        estimate,
        float(result.fun),
        int(result.nit),
        bool(result.success),
        str(result.message),
    )


def fit_locus(
    models: np.ndarray,
    observed: np.ndarray,
    *,
    transform: Transform = "log",
    inactive: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    max_iterations: int = 15_000,
    min_unmasked_bins: int = 1,
    prior_status: FitStatus | None = None,
) -> LocusFit:
    """Fit one locus by nonnegative least squares.

    Args:
        models: ``(n_bins, n_models)`` design matrix, masked bins already zeroed.
        observed: Length ``n_bins`` signal vector, masked bins already zeroed.
        transform: ``"log"`` (default, as published) or ``"identity"``.
        inactive: Boolean array over models to pin at zero, for example from
            external TSS evidence.
        initial: Starting abundance. Defaults to zeros, matching the reference.
        max_iterations: Optimizer iteration cap.
        min_unmasked_bins: Fewer usable bins than this yields
            `FitStatus.INSUFFICIENT_BINS` rather than an unconstrained fit.
        prior_status: A status already determined upstream, such as
            `FitStatus.MISSING_CONTIG` from signal extraction. Failure statuses
            are returned unchanged without attempting a fit.
            `FitStatus.ALL_ZERO` is deliberately *not* short-circuited: it is a
            valid observation, and falls through to return zeros.

    Returns:
        A `LocusFit`. On failure the abundance is all-NaN, never zeros.
    """
    n_bins, n_models = models.shape

    # Only genuine failures skip the fit. An upstream ALL_ZERO must still
    # produce zeros rather than NaN, or a quiet locus would be reported as a
    # numerical failure (plan section 10).
    if prior_status is not None and is_failure(prior_status):
        return _short_circuit(n_models, prior_status)

    if n_models == 0:
        return _short_circuit(0, FitStatus.INSUFFICIENT_BINS, "locus has no models")

    if not np.all(np.isfinite(observed)):
        return _short_circuit(
            n_models, FitStatus.NUMERICAL_ERROR, "observed signal contains non-finite values"
        )

    # A bin can only constrain the fit if some model reaches it.
    usable = int(np.count_nonzero(models.any(axis=1)))
    if usable < max(min_unmasked_bins, 1):
        return _short_circuit(
            n_models,
            FitStatus.INSUFFICIENT_BINS,
            f"{usable} bin(s) with model support, need {min_unmasked_bins}",
        )

    pinned = (
        np.zeros(n_models, dtype=bool) if inactive is None else np.asarray(inactive, dtype=bool)
    )
    if pinned.shape != (n_models,):
        raise ValueError(f"inactive must have shape ({n_models},), got {pinned.shape}")

    if not observed.any():
        # A genuine zero, distinguishable from a failure. Report it as such
        # rather than letting the optimizer return zeros for an unclear reason.
        return LocusFit(
            abundance=np.zeros(n_models),
            status=FitStatus.ALL_ZERO,
            objective=objective(np.zeros(n_models), models, observed, transform),
            gradient_norm=0.0,
            iterations=0,
            converged=True,
            message="no observed signal at locus",
        )

    if n_models == 1 and not pinned[0] and initial is None:
        estimate, value, iterations, converged, message = _fit_single_model(
            models, observed, transform
        )
        gradient_norm = float(np.abs(gradient(estimate, models, observed, transform)).max())
        return LocusFit(
            abundance=estimate,
            status=FitStatus.OK if converged else FitStatus.NONCONVERGED,
            objective=value,
            gradient_norm=gradient_norm,
            iterations=iterations,
            converged=converged,
            message=message,
            failure_reason=None if converged else message,
        )

    start = np.zeros(n_models) if initial is None else np.array(initial, dtype=np.float64)
    start[pinned] = 0.0
    upper = np.where(pinned, _INACTIVE_UPPER, _FREE_UPPER)

    try:
        # The lean driver, not scipy.optimize.minimize: same Fortran routine
        # and same defaults, without the per-call Python scaffolding that made
        # us slower than R on many small loci (plan section 14.2). Verified
        # bit-identical in test_lean_driver_matches_scipy_minimize.
        abundance, value, final_gradient, iterations, converged, message = minimize_bounded(
            _value_and_grad(models, observed, transform),
            start,
            np.zeros(n_models),
            upper,
            maxiter=max_iterations,
        )
    except (ValueError, FloatingPointError) as error:  # pragma: no cover - defensive
        return _short_circuit(n_models, FitStatus.NUMERICAL_ERROR, str(error))

    # Pinned models may drift by the tiny bound; snap them back.
    abundance[pinned] = 0.0
    if pinned.any():
        final_gradient = gradient(abundance, models, observed, transform)
    gradient_norm = float(np.abs(final_gradient).max()) if n_models else 0.0

    if not np.all(np.isfinite(abundance)):
        return _short_circuit(
            n_models, FitStatus.INVALID_MEAN, "optimizer returned non-finite abundance"
        )

    return LocusFit(
        abundance=abundance,
        status=FitStatus.OK if converged else FitStatus.NONCONVERGED,
        objective=value,
        gradient_norm=gradient_norm,
        iterations=iterations,
        converged=converged,
        message=message,
        failure_reason=None if converged else message,
    )


def _short_circuit(n_models: int, status: FitStatus, reason: str | None = None) -> LocusFit:
    """A fit that never ran: NaN abundance plus a reason, never zeros."""
    return LocusFit(
        abundance=np.full(n_models, np.nan),
        status=status,
        objective=float("nan"),
        gradient_norm=float("nan"),
        iterations=0,
        converged=False,
        message=reason or status.value,
        failure_reason=reason or status.value,
    )
