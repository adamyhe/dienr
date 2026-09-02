"""Conditioning and identifiability diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from dienr.diagnostics import assess, residual_sum_of_squares, variance_explained


def test_independent_models_are_full_rank():
    models = np.eye(4)
    result = assess(models)
    assert result.matrix_rank == 4
    assert not result.rank_deficient
    assert not result.ill_conditioned


def test_duplicate_columns_are_rank_deficient():
    """Exact duplicates should already be collapsed; if not, say so."""
    models = np.tile(np.array([[1.0], [1.0], [0.0]]), (1, 2))
    result = assess(models)
    assert result.rank_deficient
    assert result.matrix_rank == 1
    assert result.n_models == 2


def test_near_collinear_columns_are_ill_conditioned():
    """Near-duplicates survive collapsing but cannot be told apart stably."""
    models = np.zeros((100, 2))
    models[:, 0] = 1.0
    models[:, 1] = 1.0
    models[0, 1] = 1.0 + 1e-12
    result = assess(models, condition_threshold=1e8)
    assert result.ill_conditioned


def test_masked_rows_are_excluded():
    """Zeroed rows constrain nothing and must not count as usable bins."""
    models = np.zeros((4, 1))
    models[2:, 0] = 1.0
    mask = np.array([True, True, False, False])
    result = assess(models, mask=mask)
    assert result.n_bins == 4
    assert result.n_unmasked_bins == 2


def test_all_masked_locus_is_flagged_not_crashed():
    models = np.ones((2, 1))
    result = assess(models, mask=np.array([True, True]))
    assert result.rank_deficient
    assert result.ill_conditioned
    assert result.n_unmasked_bins == 0


def test_zero_column_is_rank_deficient():
    """The all-zero model produced by a sub-bin transcript."""
    models = np.zeros((4, 2))
    models[:, 0] = 1.0
    assert assess(models).rank_deficient


def test_residual_sum_of_squares_is_zero_for_perfect_fit():
    models = np.eye(3)
    observed = np.array([1.0, 2.0, 3.0])
    assert residual_sum_of_squares(models, observed, observed) == pytest.approx(0.0)


def test_variance_explained_is_one_for_perfect_fit():
    models = np.eye(3)
    observed = np.array([1.0, 5.0, 9.0])
    assert variance_explained(models, observed, observed) == pytest.approx(1.0)


def test_variance_explained_is_nan_when_there_is_none():
    """A flat signal has no variance to explain; 0.0 would be misleading."""
    models = np.eye(3)
    observed = np.full(3, 4.0)
    assert np.isnan(variance_explained(models, observed, observed))
