"""Per-locus conditioning and residual diagnostics.

Identifiability is a reported output, not an implementation detail (plan
section 5). Exact duplicate columns are already collapsed in `dienr.design`;
what remains here is detecting *near* collinearity, which regularization can
stabilize but never resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocusConditioning:
    """Numerical conditioning of one locus design matrix.

    Attributes:
        n_bins: Total bins, masked included.
        n_unmasked_bins: Bins that can constrain the fit.
        n_models: Columns after exact-equivalence collapsing.
        matrix_rank: Numerical rank over unmasked rows.
        condition_number: Ratio of largest to smallest singular value, or
            infinity when the smallest is zero.
        smallest_singular_value: Smallest singular value.
        rank_deficient: True when `matrix_rank` is below `n_models`.
        ill_conditioned: True when `condition_number` exceeds the threshold.
    """

    n_bins: int
    n_unmasked_bins: int
    n_models: int
    matrix_rank: int
    condition_number: float
    smallest_singular_value: float
    rank_deficient: bool
    ill_conditioned: bool


def assess(
    models: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    condition_threshold: float = 1e8,
) -> LocusConditioning:
    """Assess conditioning of one locus design matrix.

    Args:
        models: ``(n_bins, n_models)`` design matrix.
        mask: Boolean array over bins; True means masked out. Masked rows are
            excluded, since zeroed rows constrain nothing.
        condition_threshold: Condition number above which the locus is flagged
            ill-conditioned.

    Returns:
        A `LocusConditioning` for the locus.
    """
    n_bins, n_models = models.shape
    usable = models if mask is None else models[~mask, :]

    if usable.size == 0 or n_models == 0:
        return LocusConditioning(
            n_bins=n_bins,
            n_unmasked_bins=int(usable.shape[0]),
            n_models=n_models,
            matrix_rank=0,
            condition_number=float("inf"),
            smallest_singular_value=0.0,
            rank_deficient=n_models > 0,
            ill_conditioned=True,
        )

    singular = np.linalg.svd(usable, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    smallest = float(singular[-1]) if singular.size else 0.0
    # Same relative tolerance numpy.linalg.matrix_rank uses.
    tolerance = largest * max(usable.shape) * np.finfo(float).eps
    rank = int((singular > tolerance).sum())
    condition = float("inf") if smallest <= 0 else largest / smallest

    return LocusConditioning(
        n_bins=n_bins,
        n_unmasked_bins=int(usable.shape[0]),
        n_models=n_models,
        matrix_rank=rank,
        condition_number=condition,
        smallest_singular_value=smallest,
        rank_deficient=rank < n_models,
        ill_conditioned=condition > condition_threshold,
    )


def residual_sum_of_squares(
    models: np.ndarray, abundance: np.ndarray, observed: np.ndarray
) -> float:
    """Residual sum of squares of the fitted signal against `observed`."""
    residual = observed - models @ abundance
    return float(residual @ residual)


def variance_explained(models: np.ndarray, abundance: np.ndarray, observed: np.ndarray) -> float:
    """Fraction of variance in `observed` explained by the fitted signal.

    Returns NaN when `observed` has no variance to explain, rather than the
    misleading 0.0 that the ratio would otherwise give.
    """
    centered = observed - observed.mean()
    total = float(centered @ centered)
    if total <= 0:
        return float("nan")
    return 1.0 - residual_sum_of_squares(models, abundance, observed) / total
