"""Shared pieces for the count likelihoods: exposure, the mean, initialization.

The mean follows section 4 of the plan,

    mu_b = c * e_b * (M a)_b + beta_b

with ``c`` a known sample scale factor (spike-in or library size), ``e_b`` the
usable exposure of bin ``b`` (effective length, mappability), and ``beta_b`` an
optional fixed background. All three are *known*, not estimated, which is what
keeps the Poisson objective convex in ``a`` under nonnegativity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..status import FitStatus

#: Lower bound applied to the fitted mean during evaluation.
#:
#: This is a numerical guard, not a pseudocount. Nothing is added to the
#: observed counts. Its only role is to keep ``log mu`` and ``y / mu`` finite at
#: interior points the optimizer probes, notably ``a = 0``, which L-BFGS-B
#: reaches exactly by projecting onto its lower bound.
#:
#: It cannot move an *interior* optimum: wherever ``y_b > 0`` the negative log
#: likelihood diverges as ``mu_b -> 0``, so no minimizer has ``mu_b`` anywhere
#: near this value. It does, however, bound a boundary gradient that is truly
#: infinite. At ``a = 0`` with no background the exact gradient is ``-inf``;
#: here it is about ``-y / MU_FLOOR``, around ``-1e13``. The consequence is
#: real and worth knowing: an L1 penalty cannot drive a model to exactly zero
#: unless a background makes ``mu > 0`` without it, because otherwise no finite
#: penalty outweighs a divergent likelihood. That is correct behavior for a
#: Poisson likelihood, but the specific weight at which a penalty starts to win
#: in the no-background case is an artifact of this constant, not a modeling
#: statement.
MU_FLOOR = 1e-12


@dataclass(frozen=True)
class Exposure:
    """Known multiplicative scale on the fitted mean.

    Attributes:
        sample_scale: ``c``, one factor for the whole sample.
        bin_exposure: ``e_b``, per bin. `dienr.design.Design.bin_lengths` is the
            natural choice for count data: a partial terminal bin has had less
            opportunity to accumulate reads than a full one, and ignoring that
            would inflate its implied rate.
        background: ``beta_b``, a fixed per-bin offset. Also the way to let bins
            with no model support contribute, since without it such bins carry
            ``mu = 0`` and are excluded.
    """

    sample_scale: float = 1.0
    bin_exposure: np.ndarray | None = None
    background: np.ndarray | None = None

    def validate(self, n_bins: int) -> FitStatus | None:
        """Return a failure status if this exposure cannot be used.

        Returns:
            `FitStatus.INVALID_EXPOSURE` with the reason implied, or None.
        """
        if not np.isfinite(self.sample_scale) or self.sample_scale <= 0:
            return FitStatus.INVALID_EXPOSURE
        for array, must_be_positive in (
            (self.bin_exposure, True),
            (self.background, False),
        ):
            if array is None:
                continue
            if array.shape != (n_bins,):
                raise ValueError(f"exposure array has shape {array.shape}, expected ({n_bins},)")
            if not np.all(np.isfinite(array)):
                return FitStatus.INVALID_EXPOSURE
            if must_be_positive and np.any(array <= 0):
                return FitStatus.INVALID_EXPOSURE
            if not must_be_positive and np.any(array < 0):
                return FitStatus.INVALID_EXPOSURE
        return None

    def scale(self, n_bins: int) -> np.ndarray:
        """The combined multiplicative factor ``c * e_b``, per bin."""
        if self.bin_exposure is None:
            return np.full(n_bins, float(self.sample_scale))
        return self.sample_scale * self.bin_exposure

    def offset(self, n_bins: int) -> np.ndarray:
        """The additive background ``beta_b``, per bin."""
        if self.background is None:
            return np.zeros(n_bins)
        return self.background


def usable_bins(
    models: np.ndarray,
    exposure: Exposure,
    pinned: np.ndarray | None = None,
) -> np.ndarray:
    """Bins that can contribute to a count likelihood.

    A bin with no model support and no background has ``mu = 0`` however the
    abundance is chosen. If it also has ``y > 0`` the likelihood is ``-inf``
    there for every parameter value, which is not a fit failure but a
    misspecification: nothing in the model can explain that signal. Section 4.2
    of the plan therefore excludes such bins unless a background covers them.

    Args:
        models: The design matrix.
        exposure: Known scale and background.
        pinned: Models held at zero. Support is judged on the *free* columns
            only, because a pinned model contributes nothing to the mean. Left
            in, a bin reachable only by a pinned model would drive ``mu`` to
            zero against nonzero counts and abort the whole locus -- which is
            what happened before this argument existed, making `inactive`
            unusable with a count likelihood.
    """
    columns = models if pinned is None else models[:, ~pinned]
    support = columns.any(axis=1) if columns.size else np.zeros(models.shape[0], dtype=bool)
    if exposure.background is not None:
        support = support | (exposure.background > 0)
    return support


def mean(
    abundance: np.ndarray,
    models: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """The fitted mean ``mu = c * e * (M a) + beta``, floored for evaluation."""
    return np.maximum(scale * (models @ abundance) + offset, MU_FLOOR)


def moment_initialization(
    models: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """A strictly positive starting abundance, by method of moments.

    Per model, total observed signal it touches divided by the total exposure it
    contributes, which is the abundance that would reproduce that signal if the
    model acted alone. A strictly positive start matters: at ``a = 0`` the mean
    collapses to the background and the Poisson gradient is dominated by the
    ``y / mu`` term, so starting there wastes the first several iterations
    climbing away from a degenerate point.

    That abundance is ``sum(y over the support) / sum(scale * M)``, the Poisson
    MLE of a model fitted alone. The numerator counts each supported bin's
    signal once, *not* weighted by ``M``: the weighted form ``M'y / M'scale``
    agrees only for a 0/1 column, and for a column of scale ``s`` it starts
    ``~s^2`` too high. On graded designs with large entries (an exact-density
    column sums expected reads, ~10^2 per bin) that start put ``mu`` ~500x above
    the data, and L-BFGS-B collapsed every abundance onto the zero bound -- 493
    of 910 loci on one nascentRNASim seed (plan 18.10).
    """
    residual = np.maximum(observed - offset, 0.0)
    numerator = (models > 0).T.astype(np.float64) @ residual
    denominator = models.T @ scale
    with np.errstate(divide="ignore", invalid="ignore"):
        start = np.where(denominator > 0, numerator / denominator, 0.0)
    # A floor relative to the data keeps every mu strictly positive without
    # biasing the answer; the optimizer moves off it immediately.
    typical = float(np.mean(residual)) if residual.size else 0.0
    return np.maximum(start, max(typical, 1.0) * 1e-6)


def pearson_residuals(observed: np.ndarray, fitted: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Poisson Pearson residuals ``(y - mu) / sqrt(mu)`` over usable bins."""
    y = observed[usable]
    mu = fitted[usable]
    return (y - mu) / np.sqrt(np.maximum(mu, MU_FLOOR))
