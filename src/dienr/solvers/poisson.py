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
from ._glm import MU_FLOOR, Exposure, mean, moment_initialization, usable_bins
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

    # The usable mask is folded into precomputed vectors rather than applied by
    # boolean indexing per call. Fancy indexing allocates, and this closure is
    # the single largest cost in a Poisson fit.
    keep = usable_bins(models, exposure, pinned).astype(np.float64)
    scale_keep = scale * keep
    observed_keep = observed * keep
    observed_scaled = observed_keep * scale

    mu = np.empty(n_bins)
    weight = np.empty(n_bins)
    scratch = np.empty(n_bins)

    def fused(abundance: np.ndarray) -> tuple[float, np.ndarray]:
        np.matmul(models, abundance, out=mu)
        np.multiply(mu, scale, out=mu)
        np.add(mu, offset, out=mu)
        np.maximum(mu, MU_FLOOR, out=mu)

        # sum over usable bins of (mu - y log mu); the log is finite everywhere
        # thanks to the floor, and observed_keep is zero outside the mask, so
        # excluded bins contribute nothing to either term.
        np.log(mu, out=scratch)
        value = float(keep @ mu - observed_keep @ scratch)

        np.divide(observed_scaled, mu, out=weight)
        np.subtract(scale_keep, weight, out=weight)
        grad = models.T @ weight
        if penalty.is_active:
            value += penalty.value(abundance)
            grad += penalty.gradient(abundance)
        return value, grad

    return fused


def _fit_single_model(
    models: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    usable: np.ndarray,
) -> LocusFit | None:
    """Closed-form fit for one free model with no penalty or background.

    The Poisson score for ``mu_b = a s_b m_b`` is ``sum_b (y_b / a - s_b m_b)``
    over usable bins, so the MLE is ``a = sum y / sum s m`` -- for *any*
    nonnegative column, graded or run-on tailed, not only a 0/1 one. It is
    exactly `moment_initialization`'s start, so the general path would begin at
    the answer and spend its L-BFGS-B setup confirming it; single-model loci are
    the common case, which is why that cost is worth skipping (section 14.4).

    Returns None, sending the caller to the general optimizer, where the closed
    form isn't the optimum of the objective actually minimized: if the
    `MU_FLOOR` guard would bind in some usable bin, or the estimate exceeds
    the upper bound.
    """
    column = models[:, 0]
    exposure_mass = float(scale[usable] @ column[usable])
    if exposure_mass <= 0:  # pragma: no cover - usable bins have support
        return None
    estimate = float(observed[usable].sum()) / exposure_mass
    if not np.isfinite(estimate) or estimate > _FREE_UPPER:
        return None
    if float(np.min(estimate * scale[usable] * column[usable])) <= MU_FLOOR:
        return None
    # The fused objective's value and gradient, restricted to the usable bins it
    # sums over; the floor doesn't bind here, so mu is exactly a * s * m.
    exposure_col = scale[usable] * column[usable]
    y = observed[usable]
    mu = estimate * exposure_col
    value = float(mu.sum() - y @ np.log(mu))
    grad = float(exposure_mass - (y * exposure_col) @ (1.0 / mu))
    return LocusFit(
        abundance=np.array([estimate]),
        status=FitStatus.OK,
        objective=value,
        gradient_norm=abs(grad),
        iterations=1,
        converged=True,
        message="closed-form single-model Poisson MLE",
    )


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
    if (
        n_models == 1
        and initial is None
        and not penalty.is_active
        and exposure.background is None
        and not pinned[0]
    ):
        single = _fit_single_model(models, observed, scale, usable)
        if single is not None:
            return single

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
