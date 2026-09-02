"""Standard errors for the count likelihoods, and quasi-Poisson dispersion.

Gate D established that on a rounded binary design the variance function cannot
affect the point estimate at all: least squares, Poisson and NB2 agree exactly.
What differs is the *uncertainty*, and there the gap is large — a Poisson
standard error understates the true spread of estimates by 23% at dispersion
0.1 and 64% at 1.0. Quasi-Poisson closes that with a single scalar taken from a
Poisson fit, matching NB2 to within a few percent and needing no second
likelihood.

For an identity link with exposure, ``d mu_b / d a_k = c e_b M_bk``, so the
Fisher information is

    I_jk = sum_b (c e_b)^2 M_bj M_bk / V(mu_b)

with ``V(mu) = mu`` for Poisson and ``V(mu) = phi mu`` for quasi-Poisson. The
quasi-Poisson standard error is therefore exactly ``sqrt(phi)`` times the
Poisson one, which is why it costs nothing beyond estimating ``phi``.

**Two honesty constraints on any interval built from these.** A Wald standard
error describes a parameter in the interior of the feasible region; a model
sitting at the nonnegativity boundary has a one-sided sampling distribution and
no meaningful symmetric error, so `standard_errors` returns NaN there rather
than a number that would invite a symmetric interval. And where the information
matrix is singular — rank-deficient design, which `dienr.diagnostics` already
reports — the standard error is undefined in the affected directions and is
likewise NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._glm import MU_FLOOR, Exposure, mean, pearson_residuals, usable_bins

#: Abundance at or below which a model counts as sitting on the boundary.
#: Scaled against the largest estimate in the locus, so it adapts to the count
#: scale rather than assuming one.
BOUNDARY_RELATIVE_TOLERANCE = 1e-8

#: Condition number above which the information matrix is treated as singular.
MAX_INFORMATION_CONDITION = 1e12


@dataclass(frozen=True)
class DispersionEstimate:
    """A pooled quasi-Poisson dispersion and the evidence for it.

    Attributes:
        dispersion: ``phi``, the Pearson statistic over residual degrees of
            freedom. 1.0 means Poisson variance fits the data.
        n_bins: Usable bins pooled over.
        n_loci: Loci contributing.
        degrees_of_freedom: Pooled bins minus pooled free parameters.
    """

    dispersion: float
    n_bins: int
    n_loci: int
    degrees_of_freedom: int

    @property
    def standard_error(self) -> float:
        """Approximate sampling error of the dispersion estimate.

        Under the Poisson null the Pearson statistic over its degrees of
        freedom has mean 1 and variance ``2 / df``.
        """
        if self.degrees_of_freedom <= 0:
            return float("inf")
        return float(np.sqrt(2.0 / self.degrees_of_freedom))

    def is_overdispersed(self, z: float = 2.0) -> bool:
        """True when the dispersion exceeds 1 by more than sampling noise.

        Deliberately not a bare ``dispersion > 1`` test. That statistic is
        noisy — its standard error is ``sqrt(2 / df)`` under the Poisson null —
        so a knife-edge comparison flags about half of perfectly equidispersed
        datasets. Requiring `z` standard errors of headroom is what makes the
        answer mean something.

        Args:
            z: Standard errors of headroom required. The default of 2 is about
               a 2.5% false-positive rate on Poisson data.
        """
        return self.dispersion > 1.0 + z * self.standard_error

    def describe(self) -> dict[str, float | str]:
        """Provenance record, so a reported error is never unattributed."""
        return {
            "dispersion": self.dispersion,
            "dispersion_method": "quasi_poisson_pearson_pooled",
            "dispersion_n_bins": float(self.n_bins),
            "dispersion_degrees_of_freedom": float(self.degrees_of_freedom),
            "dispersion_standard_error": self.standard_error,
            "overdispersion_detected": float(self.is_overdispersed()),
        }


#: The Poisson assumption: variance equals the mean.
POISSON_DISPERSION = DispersionEstimate(dispersion=1.0, n_bins=0, n_loci=0, degrees_of_freedom=0)


def estimate_quasi_poisson_dispersion(
    loci: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    exposures: list[Exposure] | None = None,
) -> DispersionEstimate:
    """Estimate one pooled quasi-Poisson dispersion from completed Poisson fits.

    ``phi = sum(Pearson residual^2) / sum(degrees of freedom)``, pooled over
    every locus. Pooling is not optional: the same statistic on one small locus
    is dominated by noise, which is the reason section 4.3 forbids per-locus
    dispersion. This is the quasi-Poisson counterpart of that rule.

    Args:
        loci: One ``(models, observed, abundance)`` triple per locus, the
            abundance from a completed Poisson fit. Omit failed loci.
        exposures: Matching exposures, or None for unit exposure.

    Returns:
        A `DispersionEstimate`. Returns `POISSON_DISPERSION` when there are no
        usable degrees of freedom, which is the correct answer: no evidence of
        overdispersion means no adjustment.
    """
    chi_squared = 0.0
    total_bins = 0
    total_parameters = 0
    contributing = 0

    for index, (models, observed, abundance) in enumerate(loci):
        exposure = exposures[index] if exposures is not None else Exposure()
        n_bins = models.shape[0]
        usable = usable_bins(models, exposure)
        if not usable.any():
            continue
        fitted = mean(abundance, models, exposure.scale(n_bins), exposure.offset(n_bins))
        residuals = pearson_residuals(observed, fitted, usable)
        chi_squared += float(residuals @ residuals)
        total_bins += int(usable.sum())
        # Free parameters, not columns: a model pinned at zero consumed no
        # degree of freedom.
        total_parameters += int(np.count_nonzero(abundance))
        contributing += 1

    degrees_of_freedom = total_bins - total_parameters
    if degrees_of_freedom <= 0:
        return POISSON_DISPERSION
    return DispersionEstimate(
        dispersion=chi_squared / degrees_of_freedom,
        n_bins=total_bins,
        n_loci=contributing,
        degrees_of_freedom=degrees_of_freedom,
    )


def fisher_information(
    models: np.ndarray,
    abundance: np.ndarray,
    *,
    exposure: Exposure | None = None,
    dispersion: float = 1.0,
) -> np.ndarray:
    """Fisher information ``I_jk = sum_b (c e_b)^2 M_bj M_bk / V(mu_b)``.

    Args:
        models: ``(n_bins, n_models)`` design matrix.
        abundance: Fitted abundance.
        exposure: Known scale and background.
        dispersion: ``phi``; 1.0 gives the Poisson information, larger values
            the quasi-Poisson one.

    Returns:
        An ``(n_models, n_models)`` matrix.
    """
    exposure = exposure or Exposure()
    n_bins = models.shape[0]
    scale = exposure.scale(n_bins)
    usable = usable_bins(models, exposure)

    fitted = mean(abundance, models, scale, exposure.offset(n_bins))
    variance = np.maximum(dispersion * fitted, MU_FLOOR)
    weight = np.where(usable, scale**2 / variance, 0.0)
    return models.T @ (models * weight[:, None])


def standard_errors(
    models: np.ndarray,
    abundance: np.ndarray,
    *,
    exposure: Exposure | None = None,
    dispersion: float = 1.0,
) -> np.ndarray:
    """Wald standard errors, NaN where they are not defined.

    Args:
        models: ``(n_bins, n_models)`` design matrix.
        abundance: Fitted abundance.
        exposure: Known scale and background.
        dispersion: ``phi``. Multiplying it by ``k`` scales every error by
            ``sqrt(k)``, which is the whole of the quasi-Poisson correction.

    Returns:
        Length ``n_models`` array. Entries are NaN for models at the
        nonnegativity boundary, where the sampling distribution is one-sided
        and a symmetric error would mislead, and for every model when the
        information matrix is singular, where the errors are undefined rather
        than large.
    """
    abundance = np.asarray(abundance, dtype=np.float64)
    n_models = abundance.size
    errors = np.full(n_models, np.nan)
    if n_models == 0 or not np.all(np.isfinite(abundance)):
        return errors

    largest = float(np.max(abundance)) if n_models else 0.0
    interior = abundance > max(largest, 1.0) * BOUNDARY_RELATIVE_TOLERANCE
    if not interior.any():
        return errors

    information = fisher_information(models, abundance, exposure=exposure, dispersion=dispersion)
    # Restrict to the interior block: a boundary parameter contributes no
    # information about the others' uncertainty beyond what conditioning on it
    # already implies.
    block = information[np.ix_(interior, interior)]
    if block.size == 0:
        return errors

    singular_values = np.linalg.svd(block, compute_uv=False)
    if singular_values[-1] <= 0 or singular_values[0] / singular_values[-1] > (
        MAX_INFORMATION_CONDITION
    ):
        # Rank deficient or worse: the errors are undefined, not merely large.
        return errors

    covariance = np.linalg.inv(block)
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0):  # pragma: no cover - defensive
        return errors
    errors[interior] = np.sqrt(diagonal)
    return errors
