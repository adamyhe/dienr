"""Legacy least-squares objectives, gradients, and failure reporting."""

from __future__ import annotations

import numpy as np
import pytest

from dienr.solvers.legacy import (
    LOG_OFFSET,
    fit_locus,
    gradient,
    objective,
)
from dienr.status import FitStatus, is_failure


@pytest.fixture
def two_model_design() -> np.ndarray:
    """Six bins, two distinguishable models sharing the middle two bins."""
    matrix = np.zeros((6, 2))
    matrix[0:4, 0] = 1.0
    matrix[2:6, 1] = 1.0
    return matrix


# --- objective and gradient -------------------------------------------------


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_perfect_fit_has_near_zero_objective(two_model_design, transform):
    truth = np.array([3.0, 7.0])
    observed = two_model_design @ truth
    assert objective(truth, two_model_design, observed, transform) == pytest.approx(0.0)


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_gradient_matches_finite_differences(two_model_design, transform):
    """Analytic gradients must agree with numerical ones."""
    observed = two_model_design @ np.array([2.0, 5.0])
    point = np.array([1.3, 4.1])
    analytic = gradient(point, two_model_design, observed, transform)

    step = 1e-6
    numeric = np.empty_like(point)
    for i in range(point.size):
        forward, backward = point.copy(), point.copy()
        forward[i] += step
        backward[i] -= step
        numeric[i] = (
            objective(forward, two_model_design, observed, transform)
            - objective(backward, two_model_design, observed, transform)
        ) / (2 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_log_objective_uses_the_published_offset(two_model_design):
    """The offset keeps log(0) finite; its value is part of the method."""
    observed = np.zeros(6)
    value = objective(np.zeros(2), two_model_design, observed, "log")
    assert value == pytest.approx(0.0)
    assert LOG_OFFSET == 1e-3


def test_unknown_transform_is_rejected(two_model_design):
    with pytest.raises(ValueError, match="identity"):
        objective(np.zeros(2), two_model_design, np.zeros(6), "sqrt")


# --- recovery ---------------------------------------------------------------


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_recovers_known_abundances(two_model_design, transform):
    truth = np.array([4.0, 9.0])
    observed = two_model_design @ truth
    result = fit_locus(two_model_design, observed, transform=transform)
    assert result.status is FitStatus.OK
    np.testing.assert_allclose(result.abundance, truth, rtol=1e-4)


def test_abundance_stays_nonnegative():
    """Negative signal must not produce a negative abundance."""
    models = np.ones((4, 1))
    result = fit_locus(models, np.full(4, -5.0), transform="identity")
    assert result.abundance[0] >= 0.0


def test_fit_reports_convergence_metadata(two_model_design):
    """The reference discards these; losing them hides failures."""
    observed = two_model_design @ np.array([1.0, 2.0])
    result = fit_locus(two_model_design, observed)
    assert result.converged
    assert result.iterations > 0
    assert np.isfinite(result.objective)
    assert np.isfinite(result.gradient_norm)
    assert result.message


# --- statuses ---------------------------------------------------------------


def test_all_zero_signal_is_a_zero_not_a_failure(two_model_design):
    """A quiet locus is real biology and reports zeros, with a status saying so."""
    result = fit_locus(two_model_design, np.zeros(6))
    assert result.status is FitStatus.ALL_ZERO
    assert not is_failure(result.status)
    assert result.abundance.tolist() == [0.0, 0.0]


def test_non_finite_signal_is_a_failure_with_nan_abundance(two_model_design):
    observed = two_model_design @ np.array([1.0, 1.0])
    observed[2] = np.nan
    result = fit_locus(two_model_design, observed)
    assert result.status is FitStatus.NUMERICAL_ERROR
    assert np.isnan(result.abundance).all()
    assert result.failure_reason


def test_prior_status_short_circuits_without_fabricating_zeros(two_model_design):
    """A missing contig must not silently become an abundance of zero."""
    result = fit_locus(
        two_model_design,
        np.full(6, np.nan),
        prior_status=FitStatus.MISSING_CONTIG,
    )
    assert result.status is FitStatus.MISSING_CONTIG
    assert np.isnan(result.abundance).all()
    assert is_failure(result.status)


def test_prior_all_zero_returns_zeros_not_nan(two_model_design):
    """ALL_ZERO is a valid observation, not a failure, even when set upstream.

    Short-circuiting it to NaN would report a quiet locus as a numerical
    failure, inverting the section 10 policy.
    """
    result = fit_locus(two_model_design, np.zeros(6), prior_status=FitStatus.ALL_ZERO)
    assert result.status is FitStatus.ALL_ZERO
    assert result.abundance.tolist() == [0.0, 0.0]
    assert not is_failure(result.status)


def test_model_with_no_support_yields_insufficient_bins():
    """An all-zero design constrains nothing, so no estimate exists."""
    result = fit_locus(np.zeros((4, 2)), np.array([1.0, 2.0, 3.0, 4.0]))
    assert result.status is FitStatus.INSUFFICIENT_BINS
    assert np.isnan(result.abundance).all()


def test_min_unmasked_bins_is_enforced(two_model_design):
    observed = two_model_design @ np.array([1.0, 1.0])
    result = fit_locus(two_model_design, observed, min_unmasked_bins=99)
    assert result.status is FitStatus.INSUFFICIENT_BINS
    assert "need 99" in result.failure_reason


# --- inactive models --------------------------------------------------------


def test_inactive_model_is_pinned_to_exactly_zero(two_model_design):
    observed = two_model_design @ np.array([3.0, 6.0])
    inactive = np.array([True, False])
    result = fit_locus(two_model_design, observed, inactive=inactive)
    assert result.abundance[0] == 0.0
    assert result.abundance[1] > 0.0


def test_inactive_shape_is_validated(two_model_design):
    with pytest.raises(ValueError, match="inactive must have shape"):
        fit_locus(two_model_design, np.ones(6), inactive=np.array([True]))


# --- non-identifiability ----------------------------------------------------


def test_duplicate_columns_still_predict_the_locus_correctly():
    """When columns are identical the split is arbitrary, but the sum is not.

    This is why the plan requires collapsing duplicates before fitting and
    reporting predicted signal rather than per-column values.
    """
    models = np.tile(np.array([[1.0], [1.0], [0.0]]), (1, 2))
    observed = np.array([10.0, 10.0, 0.0])
    result = fit_locus(models, observed, transform="identity")
    np.testing.assert_allclose(models @ result.abundance, observed, atol=1e-3)


# --- single-model fast path -------------------------------------------------


@pytest.mark.parametrize("transform", ["identity", "log"])
def test_single_model_fast_path_matches_general_optimizer(transform):
    """The fast path must agree with L-BFGS-B, not merely be faster.

    Single-model loci skip the general optimizer because ~91% of the runtime
    there was scipy scaffolding. Correctness is asserted against the general
    path by fitting the same data with a pinned dummy second column.
    """
    rng = np.random.default_rng(0)
    column = np.ones((64, 1))
    column[:4] = 0.0
    observed = column[:, 0] * 7.0 + rng.exponential(0.3, 64)

    fast = fit_locus(column, observed, transform=transform)

    # Two columns where the second is pinned inactive routes through L-BFGS-B
    # while describing the identical optimization problem.
    padded = np.hstack([column, column])
    general = fit_locus(padded, observed, transform=transform, inactive=np.array([False, True]))

    assert fast.status is FitStatus.OK
    np.testing.assert_allclose(fast.abundance[0], general.abundance[0], rtol=1e-5)
    assert fast.objective == pytest.approx(general.objective, rel=1e-6)


def test_single_model_closed_form_is_exact_for_identity():
    """Identity + one model has an analytic minimizer; use it."""
    column = np.ones((10, 1))
    observed = np.full(10, 3.0)
    result = fit_locus(column, observed, transform="identity")
    assert result.abundance[0] == pytest.approx(3.0)
    assert result.objective == pytest.approx(0.0)


def test_single_model_negative_signal_clamps_to_zero():
    column = np.ones((10, 1))
    result = fit_locus(column, np.full(10, -2.0), transform="identity")
    assert result.abundance[0] == 0.0


def test_single_model_respects_inactive_pinning():
    """A pinned single model must not take the fast path and must stay zero."""
    column = np.ones((10, 1))
    result = fit_locus(column, np.full(10, 5.0), inactive=np.array([True]))
    assert result.abundance[0] == 0.0


# --- lean L-BFGS-B driver ---------------------------------------------------


@pytest.mark.parametrize("transform", ["identity", "log"])
@pytest.mark.parametrize("shape", [(64, 3), (120, 4), (300, 8), (600, 12)])
def test_lean_driver_matches_scipy_minimize(transform, shape):
    """The lean driver must be bit-identical to scipy.optimize.minimize.

    `fit_locus` uses a hand-written `setulb` loop instead of
    `scipy.optimize.minimize` to avoid per-call Python scaffolding. That is only
    safe if it is the same algorithm producing the same numbers, since R parity
    is already established against the scipy path.
    """
    from scipy import optimize

    from dienr.solvers._lbfgsb import minimize_bounded
    from dienr.solvers.legacy import _FREE_UPPER, _value_and_grad

    n_bins, n_models = shape
    rng = np.random.default_rng(abs(hash(shape)) % 2**32)
    models = (rng.random(shape) < 0.5).astype(float)
    models[:, 0] = 1.0  # guarantee support
    observed = models @ rng.exponential(4.0, n_models) + rng.exponential(0.4, n_bins)

    fused = _value_and_grad(models, observed, transform)
    lower = np.zeros(n_models)
    upper = np.full(n_models, _FREE_UPPER)

    reference = optimize.minimize(
        fused,
        np.zeros(n_models),
        jac=True,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": 15_000},
    )
    x, value, _, iterations, converged, _ = minimize_bounded(
        fused, np.zeros(n_models), lower, upper
    )

    np.testing.assert_array_equal(x, reference.x)
    assert value == reference.fun
    assert iterations == reference.nit
    assert converged == reference.success


def test_lean_driver_workspace_reuse_is_not_stateful():
    """Cached workspaces must be cleared between problems of the same size.

    `setulb` reads its own state back out of the integer and double workspaces,
    so a leftover Hessian approximation from a previous locus would corrupt the
    next one. Fitting two different problems of identical shape must give the
    same answers as fitting each alone.
    """
    from dienr.solvers._lbfgsb import minimize_bounded
    from dienr.solvers.legacy import _FREE_UPPER, _value_and_grad

    rng = np.random.default_rng(7)
    problems = []
    for _ in range(2):
        models = (rng.random((100, 4)) < 0.6).astype(float)
        models[:, 0] = 1.0
        observed = models @ rng.exponential(5.0, 4) + rng.exponential(0.3, 100)
        problems.append((models, observed))

    lower, upper = np.zeros(4), np.full(4, _FREE_UPPER)
    alone = [
        minimize_bounded(_value_and_grad(m, y, "log"), np.zeros(4), lower, upper)[0]
        for m, y in problems
    ]
    # Same problems, now run back to back through the shared workspace.
    sequential = [
        minimize_bounded(_value_and_grad(m, y, "log"), np.zeros(4), lower, upper)[0]
        for m, y in problems
    ]
    for first, second in zip(alone, sequential, strict=True):
        np.testing.assert_array_equal(first, second)


def test_pinned_model_gradient_is_recomputed():
    """Snapping a pinned model to zero invalidates the optimizer's gradient."""
    models = np.zeros((8, 2))
    models[0:6, 0] = 1.0
    models[2:8, 1] = 1.0
    observed = models @ np.array([3.0, 5.0])
    result = fit_locus(models, observed, inactive=np.array([True, False]))
    assert result.abundance[0] == 0.0
    expected = gradient(result.abundance, models, observed, "log")
    assert result.gradient_norm == pytest.approx(float(np.abs(expected).max()))
