"""Locus fit statuses.

Section 10 of the plan: a failed or invalid locus must never be reported as a
biological zero. Every sample/locus fit carries exactly one of these.
"""

from __future__ import annotations

from enum import StrEnum


class FitStatus(StrEnum):
    """Outcome of a single sample x locus fit."""

    OK = "ok"
    """Converged normally."""

    ALL_ZERO = "all_zero"
    """Valid locus with no observed signal. Abundance is a true zero."""

    INSUFFICIENT_BINS = "insufficient_bins"
    """Too few unmasked bins to constrain the models."""

    MISSING_CONTIG = "missing_contig"
    """The locus contig is absent from the signal file's header."""

    INVALID_EXPOSURE = "invalid_exposure"
    """Non-finite or non-positive exposure where a positive value is required."""

    RANK_DEFICIENT = "rank_deficient"
    """Design is rank deficient after collapsing exactly duplicate columns."""

    ILL_CONDITIONED = "ill_conditioned"
    """Condition number exceeds the configured threshold."""

    NONCONVERGED = "nonconverged"
    """The optimizer stopped without meeting its convergence criteria."""

    INVALID_MEAN = "invalid_mean"
    """A fitted mean was non-finite or invalid for the chosen likelihood."""

    NUMERICAL_ERROR = "numerical_error"
    """An unexpected numerical failure. Carries a reason."""


#: Statuses for which abundance must be reported as NaN rather than zero.
FAILURE_STATUSES = frozenset(
    {
        FitStatus.INSUFFICIENT_BINS,
        FitStatus.MISSING_CONTIG,
        FitStatus.INVALID_EXPOSURE,
        FitStatus.INVALID_MEAN,
        FitStatus.NUMERICAL_ERROR,
    }
)

#: Statuses that still carry a usable estimate, but flag a caveat on it.
QUALIFIED_STATUSES = frozenset(
    {
        FitStatus.RANK_DEFICIENT,
        FitStatus.ILL_CONDITIONED,
        FitStatus.NONCONVERGED,
    }
)


def is_failure(status: FitStatus) -> bool:
    """True if `status` means no estimate exists, so abundance must be NaN."""
    return status in FAILURE_STATUSES
