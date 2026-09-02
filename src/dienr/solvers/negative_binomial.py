"""NB2 fitting with a shared dispersion, plus dispersion estimation.

**Deprioritized. Prefer `dienr.solvers.inference` quasi-Poisson.** Gate D
established two things that between them remove most of NB2's reason to exist
on the designs this package actually builds:

1. On a rounded binary design NB2 **cannot change the point estimate at all.**
   The identity-link score weights residuals by ``1 / V(mu)``, so a constant
   ``mu`` across a model's support makes that weight constant and the
   estimating equation reduces to the sample mean whatever the variance
   function is. Least squares, Poisson and NB2 agree to ``0.000e+00`` even at
   dispersion 3.0.
2. What NB2 legitimately offers is calibrated uncertainty, and **quasi-Poisson
   delivers that to within a few percent** using one scalar from a Poisson fit
   — no second likelihood, no per-fit fixed dispersion, no non-convex
   objective, and no mean-collapse guard, all of which this module needs.

Keep it for the case it is genuinely required: a design where the fitted mean
varies enough across a locus that the full variance function matters for the
estimate too. That means non-binary occupancy, which in practice means the
shape profile — currently a v1 non-goal. Until that exists, selecting
``likelihood="negative_binomial"`` buys nothing over ``"quasi_poisson"``.

Section 4.3 of the plan. NB2 means ``Var(y) = mu + phi mu^2``. Writing
``r = 1 / phi`` for the size parameter, the negative log-likelihood is

    L(a) = sum_b [ lgamma(y+1) + lgamma(r) - lgamma(y+r)
                   - r log r + (y+r) log(r+mu) - y log mu ]

Written that way it loses precision badly at small dispersion, where ``r`` is
huge and ``(y+r) log(r+mu)`` cancels against ``r log r`` to leave a small
difference between two large numbers. The algebraically identical form used
here is stable:

    L(a) = sum_b [ lgamma(y+1) + lgamma(r) - lgamma(y+r)
                   + r log1p(mu/r) + y log1p(r/mu) ]

Only the last two terms depend on ``a``. As ``phi -> 0`` the first of them
tends to ``mu`` and the NB2 optimum tends to the Poisson one, which
`test_nb_approaches_poisson_as_dispersion_vanishes` checks; the naive form
fails that check outright.

The gradient is unaffected by the rewrite and reduces to

    dL/da_k = sum_b M_bk c e_b * r (mu_b - y_b) / (mu_b (r + mu_b)).

**Dispersion is not estimated per locus.** The plan is explicit: estimate it
across many loci or borrow it across replicates, never unconstrained on a small
locus, where a handful of bins cannot separate overdispersion from a poor mean
fit. `estimate_dispersion` therefore pools Pearson residuals from a completed
Poisson fit over every locus and returns one shared value, which is then held
fixed. That is the two-stage procedure the section describes.

**The NB2 negative log-likelihood is not globally convex here.** With the
identity link used throughout this package, the per-bin second derivative is

    f''(mu) = r (-mu^2 + 2 mu y + r y) / (mu^2 + r mu)^2

so a bin is convex only while ``mu < y + sqrt(y^2 + r y)``, and a zero-count bin
is *always* concave: ``f''(mu) = -r / (mu + r)^2 < 0``. The Poisson objective
loses nothing to this (``r -> inf`` makes the condition vacuous), which is why
section 4.2 can claim convexity and this section cannot. In practice the
moment initialization lands in the convex region and no local optima have
turned up in testing, but the guarantee is absent, so a pathological start can
terminate with a collapsed mean and `fit_locus` checks for exactly that.

NB is **not** the default and should not become one without the evidence Gate D
asks for: reproducible overdispersion, stable estimation, and a material
held-out improvement over Poisson. `compare_dispersion_models` produces the
numbers for that comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, xlogy

from ..status import FitStatus, is_failure
from ._glm import (
    MU_FLOOR,
    Exposure,
    mean,
    moment_initialization,
    pearson_residuals,
    usable_bins,
)
from ._lbfgsb import minimize_bounded
from .legacy import _FREE_UPPER, _INACTIVE_UPPER, LocusFit, _short_circuit
from .penalties import NO_PENALTY, Penalty
from .poisson import MU_FLOOR_CHECK

#: Smallest dispersion treated as overdispersion at all. Below this the NB is
#: numerically indistinguishable from a Poisson and `fit_locus` says so rather
#: than inverting a near-infinite size parameter.
MIN_DISPERSION = 1e-8

#: Ceiling on a pooled dispersion estimate. A larger value almost always means
#: the mean model is wrong rather than that the counts are that overdispersed,
#: so it is clipped and reported rather than trusted.
MAX_DISPERSION = 1e3


@dataclass(frozen=True)
class DispersionEstimate:
    """A pooled dispersion estimate and the evidence behind it.

    Attributes:
        dispersion: ``phi`` in ``Var = mu + phi mu^2``.
        n_bins: Usable bins pooled over.
        n_loci: Loci contributing.
        pearson_chi2_per_df: Poisson Pearson statistic over residual degrees of
            freedom. This is the quasi-Poisson dispersion: 1.0 means Poisson
            variance fits, and larger means overdispersed.
        clipped: True when the estimate hit `MAX_DISPERSION`.
    """

    dispersion: float
    n_bins: int
    n_loci: int
    pearson_chi2_per_df: float
    clipped: bool

    @property
    def is_overdispersed(self) -> bool:
        """True when the pooled evidence exceeds Poisson variance."""
        return self.dispersion > MIN_DISPERSION


def estimate_dispersion(
    loci: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    exposures: list[Exposure] | None = None,
) -> DispersionEstimate:
    """Estimate one shared NB2 dispersion by pooling across loci.

    Uses the moment relation for NB2: with ``Var = mu + phi mu^2``, the expected
    squared Pearson residual is ``1 + phi mu``, so regressing
    ``(y - mu)^2 - mu`` on ``mu^2`` through the origin gives ``phi``. Pooling
    over all loci is what makes it stable; the same calculation on one small
    locus is dominated by noise.

    Args:
        loci: One ``(models, observed, abundance)`` triple per locus, where
            `abundance` comes from a completed Poisson fit. Loci whose fit
            failed should be omitted.
        exposures: Matching exposures, or None for unit exposure throughout.

    Returns:
        A `DispersionEstimate`. Zero dispersion when the data show no
        overdispersion, which is the correct answer and means "use Poisson".
    """
    numerator = 0.0
    denominator = 0.0
    chi2 = 0.0
    total_bins = 0
    total_params = 0
    contributing = 0

    for index, (models, observed, abundance) in enumerate(loci):
        exposure = exposures[index] if exposures is not None else Exposure()
        n_bins = models.shape[0]
        usable = usable_bins(models, exposure)
        if not usable.any():
            continue
        mu = mean(abundance, models, exposure.scale(n_bins), exposure.offset(n_bins))
        y = observed[usable]
        m = mu[usable]

        numerator += float(np.sum((y - m) ** 2 - m))
        denominator += float(np.sum(m**2))
        residuals = pearson_residuals(observed, mu, usable)
        chi2 += float(residuals @ residuals)
        total_bins += int(usable.sum())
        total_params += int(models.shape[1])
        contributing += 1

    if denominator <= 0:
        return DispersionEstimate(0.0, total_bins, contributing, float("nan"), False)

    raw = numerator / denominator
    clipped = raw > MAX_DISPERSION
    dispersion = float(min(max(raw, 0.0), MAX_DISPERSION))
    degrees_of_freedom = max(total_bins - total_params, 1)
    return DispersionEstimate(
        dispersion=dispersion,
        n_bins=total_bins,
        n_loci=contributing,
        pearson_chi2_per_df=chi2 / degrees_of_freedom,
        clipped=clipped,
    )


def negative_log_likelihood(
    abundance: np.ndarray,
    models: np.ndarray,
    observed: np.ndarray,
    dispersion: float,
    *,
    exposure: Exposure | None = None,
    penalty: Penalty = NO_PENALTY,
) -> float:
    """NB2 negative log-likelihood at fixed `dispersion`, plus any penalty."""
    exposure = exposure or Exposure()
    n_bins = models.shape[0]
    usable = usable_bins(models, exposure)
    mu = mean(abundance, models, exposure.scale(n_bins), exposure.offset(n_bins))
    y = observed[usable]
    m = mu[usable]
    size = 1.0 / max(dispersion, MIN_DISPERSION)

    value = float(
        np.sum(
            gammaln(y + 1.0)
            + gammaln(size)
            - gammaln(y + size)
            + size * np.log1p(m / size)
            + xlogy(y, 1.0 + size / m)
        )
    )
    return value + penalty.value(abundance)


def gradient(
    abundance: np.ndarray,
    models: np.ndarray,
    observed: np.ndarray,
    dispersion: float,
    *,
    exposure: Exposure | None = None,
    penalty: Penalty = NO_PENALTY,
) -> np.ndarray:
    """Analytic gradient of `negative_log_likelihood`."""
    exposure = exposure or Exposure()
    n_bins = models.shape[0]
    scale = exposure.scale(n_bins)
    usable = usable_bins(models, exposure)
    mu = mean(abundance, models, scale, exposure.offset(n_bins))
    size = 1.0 / max(dispersion, MIN_DISPERSION)

    weight = np.zeros(n_bins)
    y = observed[usable]
    m = mu[usable]
    # r (mu - y) / (mu (r + mu)), the stable reduction of
    # (y + r)/(r + mu) - y/mu.
    weight[usable] = scale[usable] * size * (m - y) / (m * (size + m))
    return models.T @ weight + penalty.gradient(abundance)


def _value_and_grad(
    models: np.ndarray,
    observed: np.ndarray,
    dispersion: float,
    exposure: Exposure,
    penalty: Penalty,
    pinned: np.ndarray | None = None,
):
    """Fused objective and gradient, one ``M a`` product per evaluation."""
    n_bins = models.shape[0]
    scale = exposure.scale(n_bins)
    offset = exposure.offset(n_bins)
    size = 1.0 / max(dispersion, MIN_DISPERSION)

    # As in the Poisson closure: fold the usable mask into precomputed vectors
    # and preallocate, so no per-call indexing or temporaries. Terms with no
    # dependence on the abundance are dropped from the value the optimizer
    # sees; `negative_log_likelihood` reports the complete figure.
    keep = usable_bins(models, exposure, pinned).astype(np.float64)
    observed_keep = observed * keep
    weight_scale = scale * keep * size

    mu = np.empty(n_bins)
    weight = np.empty(n_bins)
    scratch = np.empty(n_bins)

    def fused(abundance: np.ndarray) -> tuple[float, np.ndarray]:
        np.matmul(models, abundance, out=mu)
        np.multiply(mu, scale, out=mu)
        np.add(mu, offset, out=mu)
        np.maximum(mu, MU_FLOOR, out=mu)

        # r log1p(mu/r)
        np.divide(mu, size, out=scratch)
        np.log1p(scratch, out=scratch)
        value = size * float(keep @ scratch)
        # + y log1p(r/mu)
        np.divide(size, mu, out=scratch)
        np.log1p(scratch, out=scratch)
        value += float(observed_keep @ scratch)

        # weight = scale * keep * r (mu - y) / (mu (r + mu))
        np.add(mu, size, out=scratch)
        np.multiply(scratch, mu, out=scratch)
        np.subtract(mu, observed, out=weight)
        np.multiply(weight, weight_scale, out=weight)
        np.divide(weight, scratch, out=weight)

        grad = models.T @ weight
        if penalty.is_active:
            value += penalty.value(abundance)
            grad += penalty.gradient(abundance)
        return value, grad

    return fused


def fit_locus(
    models: np.ndarray,
    observed: np.ndarray,
    dispersion: float,
    *,
    exposure: Exposure | None = None,
    penalty: Penalty = NO_PENALTY,
    inactive: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    max_iterations: int = 15_000,
    min_unmasked_bins: int = 1,
    prior_status: FitStatus | None = None,
) -> LocusFit:
    """Fit one locus by nonnegative NB2 maximum likelihood at fixed dispersion.

    Args:
        models: ``(n_bins, n_models)`` design matrix, masked bins zeroed.
        observed: Raw counts per bin.
        dispersion: Shared ``phi``, from `estimate_dispersion`. Values at or
            below `MIN_DISPERSION` are rejected rather than silently treated as
            Poisson, so the caller chooses the likelihood knowingly.
        exposure: Known scale and background.
        penalty: Named penalty.
        inactive: Models to pin at zero.
        initial: Starting abundance; defaults to a positive moment estimate.
        max_iterations: Optimizer iteration cap.
        min_unmasked_bins: Minimum usable bins to attempt a fit.
        prior_status: A failure status decided upstream.

    Returns:
        A `LocusFit`. `objective` is the complete NB2 negative log-likelihood,
        including the terms constant in `abundance`, so it is comparable across
        fits at the same dispersion.

    Raises:
        ValueError: If `dispersion` is not usable, or `inactive` is misshaped.
    """
    exposure = exposure or Exposure()
    n_bins, n_models = models.shape

    if not np.isfinite(dispersion) or dispersion <= MIN_DISPERSION:
        raise ValueError(
            f"dispersion must exceed {MIN_DISPERSION} to justify NB over Poisson, "
            f"got {dispersion!r}. Use dienr.solvers.poisson for equidispersed counts."
        )

    if prior_status is not None and is_failure(prior_status):
        return _short_circuit(n_models, prior_status)
    if n_models == 0:
        return _short_circuit(0, FitStatus.INSUFFICIENT_BINS, "locus has no models")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0):
        return _short_circuit(
            n_models,
            FitStatus.NUMERICAL_ERROR,
            "NB fitting requires finite non-negative counts",
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

    # Support is judged on the free columns; a pinned model contributes nothing.
    usable = usable_bins(models, exposure, pinned)
    n_usable = int(usable.sum())
    if n_usable < max(min_unmasked_bins, 1):
        return _short_circuit(
            n_models,
            FitStatus.INSUFFICIENT_BINS,
            f"{n_usable} free-model bin(s) with support, need {min_unmasked_bins}",
        )

    if not observed[usable].any():
        zeros = np.zeros(n_models)
        return LocusFit(
            abundance=zeros,
            status=FitStatus.ALL_ZERO,
            objective=negative_log_likelihood(
                zeros, models, observed, dispersion, exposure=exposure, penalty=penalty
            ),
            gradient_norm=0.0,
            iterations=0,
            converged=True,
            message="no observed counts at locus",
        )

    scale = exposure.scale(n_bins)
    offset = exposure.offset(n_bins)
    start = (
        moment_initialization(models, observed, scale, offset)
        if initial is None
        else np.array(initial, dtype=np.float64)
    )
    start[pinned] = 0.0
    upper = np.where(pinned, _INACTIVE_UPPER, _FREE_UPPER)

    fused = _value_and_grad(models, observed, dispersion, exposure, penalty, pinned)
    try:
        abundance, _, final_gradient, iterations, converged, message = minimize_bounded(
            fused, start, np.zeros(n_models), upper, maxiter=max_iterations
        )
    except (ValueError, FloatingPointError) as error:  # pragma: no cover - defensive
        return _short_circuit(n_models, FitStatus.NUMERICAL_ERROR, str(error))

    abundance[pinned] = 0.0
    if pinned.any():
        final_gradient = gradient(
            abundance, models, observed, dispersion, exposure=exposure, penalty=penalty
        )
    if not np.all(np.isfinite(abundance)):
        return _short_circuit(
            n_models, FitStatus.INVALID_MEAN, "optimizer returned non-finite abundance"
        )

    # The same guard the Poisson solver applies, and NB needs it more: unlike
    # the Poisson negative log-likelihood, the NB2 one with an identity link is
    # not globally convex -- every zero-count bin contributes a concave term,
    # ``f''(mu) = -r / (mu + r)^2`` -- so a poor starting point can terminate
    # somewhere the mean has collapsed. Without this check that came back as an
    # apparently converged fit of exactly zero with a badly wrong objective.
    mu = mean(abundance, models, scale, offset)
    collapsed = usable & (observed > 0)
    if collapsed.any() and np.any(mu[collapsed] <= MU_FLOOR_CHECK):
        return _short_circuit(
            n_models,
            FitStatus.INVALID_MEAN,
            "fitted mean vanished in a bin with nonzero counts",
        )

    return LocusFit(
        abundance=abundance,
        status=FitStatus.OK if converged else FitStatus.NONCONVERGED,
        # Report the complete likelihood, not the reduced form the optimizer saw.
        objective=negative_log_likelihood(
            abundance, models, observed, dispersion, exposure=exposure, penalty=penalty
        ),
        gradient_norm=float(np.abs(final_gradient).max()) if n_models else 0.0,
        iterations=iterations,
        converged=converged,
        message=message,
        failure_reason=None if converged else message,
    )


def compare_dispersion_models(
    loci: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    exposures: list[Exposure] | None = None,
) -> dict[str, float]:
    """Evidence for or against needing NB over Poisson.

    Produces the numbers Gate D asks for. NB should not be adopted on the
    strength of a dispersion estimate alone: `pearson_chi2_per_df` near 1
    means Poisson variance already fits, and a held-out deviance comparison
    should decide it either way.

    Args:
        loci: ``(models, observed, abundance)`` per locus from a Poisson fit.
        exposures: Matching exposures, or None.

    Returns:
        Mapping with the pooled dispersion, the quasi-Poisson dispersion, the
        pooled bin and locus counts, and a recommendation flag.
    """
    estimate = estimate_dispersion(loci, exposures=exposures)
    return {
        "nb2_dispersion": estimate.dispersion,
        "quasi_poisson_dispersion": estimate.pearson_chi2_per_df,
        "n_bins": float(estimate.n_bins),
        "n_loci": float(estimate.n_loci),
        "overdispersion_detected": float(estimate.is_overdispersed),
        "dispersion_clipped": float(estimate.clipped),
    }
