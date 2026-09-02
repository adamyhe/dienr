"""Nonnegative Poisson fitting for raw bin counts.

Section 4.2 of the plan. For counts ``y_b ~ Poisson(mu_b)`` the negative
log-likelihood, dropping the ``log y!`` term that does not depend on the
parameters, is

    L(a) = sum_b [ mu_b - y_b log mu_b ],    mu_b = c e_b (M a)_b + beta_b

with gradient

    dL/da_k = sum_b M_bk c e_b (1 - y_b / mu_b).

With a fixed nonnegative design, fixed exposure and fixed background, this is
convex in ``a``, and stays convex under nonnegativity and L1/L2 penalties. That
does not make every parameter identifiable: rank deficiency is a property of
``M``, so collapsing exact duplicates and reporting conditioning remain
necessary (plan section 5).

Zero counts are handled exactly. `scipy.special.xlogy` gives ``0`` when
``y == 0`` whatever the mean is, so empty bins contribute only their ``mu``
term and no pseudocount is introduced anywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.special import xlogy

from ..status import FitStatus, is_failure
from ._glm import Exposure, mean, moment_initialization, usable_bins
from ._lbfgsb import minimize_bounded
from .legacy import _FREE_UPPER, _INACTIVE_UPPER, LocusFit, _short_circuit
from .penalties import NO_PENALTY, Penalty

#: Threshold for declaring the fitted mean collapsed. An order of magnitude
#: above `_glm.MU_FLOOR`, so hitting it means the optimizer genuinely drove the
#: mean to nothing rather than merely brushing the numerical guard.
MU_FLOOR_CHECK = 1e-11


def negative_log_likelihood(
    abundance: np.ndarray,
    models: np.ndarray,
    observed: np.ndarray,
    *,
    exposure: Exposure | None = None,
    penalty: Penalty = NO_PENALTY,
) -> float:
    """Poisson negative log-likelihood, plus any penalty.

    The ``log y!`` normalizing term is omitted: it is constant in `abundance`,
    so it shifts the objective without changing the optimum. Deviance
    comparisons that need it should use `deviance`.
    """
    exposure = exposure or Exposure()
    n_bins = models.shape[0]
    scale = exposure.scale(n_bins)
    offset = exposure.offset(n_bins)
    usable = usable_bins(models, exposure)

    mu = mean(abundance, models, scale, offset)
    value = float(np.sum(mu[usable] - xlogy(observed[usable], mu[usable])))
    return value + penalty.value(abundance)


def gradient(
    abundance: np.ndarray,
    models: np.ndarray,
    observed: np.ndarray,
    *,
    exposure: Exposure | None = None,
    penalty: Penalty = NO_PENALTY,
) -> np.ndarray:
    """Analytic gradient of `negative_log_likelihood`."""
    exposure = exposure or Exposure()
    n_bins = models.shape[0]
    scale = exposure.scale(n_bins)
    offset = exposure.offset(n_bins)
    usable = usable_bins(models, exposure)

    mu = mean(abundance, models, scale, offset)
    weight = np.zeros(n_bins)
    weight[usable] = scale[usable] * (1.0 - observed[usable] / mu[usable])
    return models.T @ weight + penalty.gradient(abundance)


def deviance(
    abundance: np.ndarray,
    models: np.ndarray,
    observed: np.ndarray,
    *,
    exposure: Exposure | None = None,
) -> float:
    """Poisson deviance, ``2 * sum[y log(y/mu) - (y - mu)]``.

    Zero at a saturated fit, and the quantity to compare across models. Unlike
    `negative_log_likelihood` this is on an absolute scale, so it is what the
    plan's held-out comparisons should use.
    """
    exposure = exposure or Exposure()
    n_bins = models.shape[0]
    usable = usable_bins(models, exposure)
    mu = mean(abundance, models, exposure.scale(n_bins), exposure.offset(n_bins))
    y = observed[usable]
    m = mu[usable]
    # xlogy handles y == 0, where the limit of y log(y/mu) is 0.
    return float(2.0 * np.sum(xlogy(y, y) - xlogy(y, m) - (y - m)))


def _value_and_grad(
    models: np.ndarray,
    observed: np.ndarray,
    exposure: Exposure,
    penalty: Penalty,
    pinned: np.ndarray | None = None,
):
    """Fused objective and gradient, sharing one ``M a`` product per call.

    Same reasoning as the legacy solver: the optimizer needs both at the same
    point, and computing them separately doubles the matrix-vector work.
    """
    n_bins = models.shape[0]
    scale = exposure.scale(n_bins)
    offset = exposure.offset(n_bins)
    usable = usable_bins(models, exposure, pinned)
    y_usable = observed[usable]
    scale_usable = scale[usable]
    weight = np.zeros(n_bins)

    def fused(abundance: np.ndarray) -> tuple[float, np.ndarray]:
        mu = mean(abundance, models, scale, offset)
        mu_usable = mu[usable]
        value = float(np.sum(mu_usable - xlogy(y_usable, mu_usable)))
        weight[:] = 0.0
        weight[usable] = scale_usable * (1.0 - y_usable / mu_usable)
        grad = models.T @ weight
        if penalty.is_active:
            value += penalty.value(abundance)
            grad = grad + penalty.gradient(abundance)
        return value, grad

    return fused


def fit_locus(
    models: np.ndarray,
    observed: np.ndarray,
    *,
    exposure: Exposure | None = None,
    penalty: Penalty = NO_PENALTY,
    inactive: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    max_iterations: int = 15_000,
    min_unmasked_bins: int = 1,
    prior_status: FitStatus | None = None,
) -> LocusFit:
    """Fit one locus by nonnegative Poisson maximum likelihood.

    Args:
        models: ``(n_bins, n_models)`` design matrix, masked bins zeroed.
        observed: Raw counts per bin. Must be count-equivalent: see
            `dienr.counts.SignalFile.require_count_equivalent`, which rejects a
            per-base mean summary here.
        exposure: Known scale and background. Defaults to unit scale.
        penalty: Named penalty; unpenalized by default.
        inactive: Models to pin at zero.
        initial: Starting abundance. Defaults to a positive moment estimate.
        max_iterations: Optimizer iteration cap.
        min_unmasked_bins: Minimum usable bins to attempt a fit.
        prior_status: A failure status decided upstream, returned unchanged.

    Returns:
        A `LocusFit`. Failures carry NaN abundance and a reason, never zeros.
    """
    exposure = exposure or Exposure()
    n_bins, n_models = models.shape

    if prior_status is not None and is_failure(prior_status):
        return _short_circuit(n_models, prior_status)
    if n_models == 0:
        return _short_circuit(0, FitStatus.INSUFFICIENT_BINS, "locus has no models")
    if not np.all(np.isfinite(observed)):
        return _short_circuit(
            n_models, FitStatus.NUMERICAL_ERROR, "observed signal contains non-finite values"
        )
    if np.any(observed < 0):
        return _short_circuit(
            n_models,
            FitStatus.NUMERICAL_ERROR,
            "Poisson fitting requires non-negative counts; got negative values",
        )

    invalid = exposure.validate(n_bins)
    if invalid is not None:
        return _short_circuit(
            n_models, invalid, "exposure must be finite, with positive bin exposure"
        )

    pinned = (
        np.zeros(n_models, dtype=bool) if inactive is None else np.asarray(inactive, dtype=bool)
    )
    if pinned.shape != (n_models,):
        raise ValueError(f"inactive must have shape ({n_models},), got {pinned.shape}")

    usable = usable_bins(models, exposure, pinned)
    n_usable = int(usable.sum())
    if n_usable < max(min_unmasked_bins, 1):
        return _short_circuit(
            n_models,
            FitStatus.INSUFFICIENT_BINS,
            f"{n_usable} free-model bin(s) with support, need {min_unmasked_bins}",
        )

    if not observed[usable].any():
        # A true zero, and the Poisson MLE is exactly zero abundance. Reported
        # as ALL_ZERO rather than OK so it is not confused with a fit that
        # happened to land on zero.
        zeros = np.zeros(n_models)
        return LocusFit(
            abundance=zeros,
            status=FitStatus.ALL_ZERO,
            objective=negative_log_likelihood(
                zeros, models, observed, exposure=exposure, penalty=penalty
            ),
            gradient_norm=0.0,
            iterations=0,
            converged=True,
            message="no observed counts at locus",
        )

    scale = exposure.scale(n_bins)
    offset = exposure.offset(n_bins)
    if initial is None:
        start = moment_initialization(models, observed, scale, offset)
    else:
        start = np.array(initial, dtype=np.float64)
    start[pinned] = 0.0
    upper = np.where(pinned, _INACTIVE_UPPER, _FREE_UPPER)

    fused = _value_and_grad(models, observed, exposure, penalty, pinned)
    try:
        abundance, value, final_gradient, iterations, converged, message = minimize_bounded(
            fused, start, np.zeros(n_models), upper, maxiter=max_iterations
        )
    except (ValueError, FloatingPointError) as error:  # pragma: no cover - defensive
        return _short_circuit(n_models, FitStatus.NUMERICAL_ERROR, str(error))

    abundance[pinned] = 0.0
    if pinned.any():
        final_gradient = gradient(abundance, models, observed, exposure=exposure, penalty=penalty)

    if not np.all(np.isfinite(abundance)):
        return _short_circuit(
            n_models, FitStatus.INVALID_MEAN, "optimizer returned non-finite abundance"
        )

    mu = mean(abundance, models, scale, offset)
    if np.any(observed[usable] > 0) and np.any(mu[usable & (observed > 0)] <= MU_FLOOR_CHECK):
        # The mean collapsed where counts exist; the likelihood is unbounded
        # there and the estimate is meaningless.
        return _short_circuit(
            n_models,
            FitStatus.INVALID_MEAN,
            "fitted mean vanished in a bin with nonzero counts",
        )

    return LocusFit(
        abundance=abundance,
        status=FitStatus.OK if converged else FitStatus.NONCONVERGED,
        objective=value,
        gradient_norm=float(np.abs(final_gradient).max()) if n_models else 0.0,
        iterations=iterations,
        converged=converged,
        message=message,
        failure_reason=None if converged else message,
    )
