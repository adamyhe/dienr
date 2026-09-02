"""Poisson likelihood: gradients, recovery, exposure, and failure handling."""

from __future__ import annotations

import numpy as np
import pytest

from dienr.solvers import penalties, poisson
from dienr.solvers._glm import MU_FLOOR, Exposure, usable_bins
from dienr.status import FitStatus, is_failure


@pytest.fixture
def design() -> np.ndarray:
    """Eight bins, two distinguishable models overlapping in the middle."""
    matrix = np.zeros((8, 2))
    matrix[0:5, 0] = 1.0
    matrix[3:8, 1] = 1.0
    return matrix


def simulate(matrix: np.ndarray, truth: np.ndarray, seed: int, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    return rng.poisson(scale * (matrix @ truth)).astype(np.float64)


# --- gradient correctness ---------------------------------------------------


@pytest.mark.parametrize(
    "penalty",
    [
        penalties.NO_PENALTY,
        penalties.ridge(0.05),
        penalties.lasso(0.2),
        penalties.elastic_net(0.1, 0.05),
    ],
)
def test_gradient_matches_finite_differences(design, penalty):
    """Required by plan section 13: analytic gradients checked numerically."""
    observed = simulate(design, np.array([30.0, 70.0]), seed=0)
    point = np.array([22.0, 61.0])

    analytic = poisson.gradient(point, design, observed, penalty=penalty)
    step = 1e-6
    numeric = np.empty_like(point)
    for i in range(point.size):
        up, down = point.copy(), point.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (
            poisson.negative_log_likelihood(up, design, observed, penalty=penalty)
            - poisson.negative_log_likelihood(down, design, observed, penalty=penalty)
        ) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_gradient_matches_finite_differences_with_exposure(design):
    """The exposure enters the gradient through the chain rule; check it."""
    exposure = Exposure(
        sample_scale=2.5,
        bin_exposure=np.linspace(0.5, 1.0, 8),
        background=np.full(8, 0.3),
    )
    observed = simulate(design, np.array([12.0, 25.0]), seed=1, scale=2.0)
    point = np.array([9.0, 21.0])

    analytic = poisson.gradient(point, design, observed, exposure=exposure)
    step = 1e-6
    numeric = np.empty_like(point)
    for i in range(point.size):
        up, down = point.copy(), point.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (
            poisson.negative_log_likelihood(up, design, observed, exposure=exposure)
            - poisson.negative_log_likelihood(down, design, observed, exposure=exposure)
        ) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_fused_objective_matches_the_reference_definitions(design):
    """The optimizer's fused closure must agree with the readable functions."""
    observed = simulate(design, np.array([40.0, 15.0]), seed=2)
    exposure = Exposure(sample_scale=1.3)
    penalty = penalties.ridge(0.01)
    fused = poisson._value_and_grad(design, observed, exposure, penalty)

    point = np.array([37.0, 14.0])
    value, grad = fused(point)
    assert value == pytest.approx(
        poisson.negative_log_likelihood(point, design, observed, exposure=exposure, penalty=penalty)
    )
    np.testing.assert_allclose(
        grad,
        poisson.gradient(point, design, observed, exposure=exposure, penalty=penalty),
    )


# --- recovery ---------------------------------------------------------------


def test_recovers_known_abundances_at_high_counts(design):
    """Gate B's first requirement: recover the truth in Poisson simulation."""
    truth = np.array([500.0, 900.0])
    observed = simulate(design, truth, seed=3)
    result = poisson.fit_locus(design, observed)
    assert result.status is FitStatus.OK
    np.testing.assert_allclose(result.abundance, truth, rtol=0.15)


def test_recovery_improves_with_depth(design):
    """More counts, less error. A sanity check on the estimator itself."""
    truth = np.array([20.0, 40.0])
    errors = []
    for scale in (1.0, 100.0):
        observed = simulate(design, truth, seed=4, scale=scale)
        fit = poisson.fit_locus(design, observed, exposure=Exposure(sample_scale=scale))
        errors.append(float(np.abs(fit.abundance - truth).mean()))
    assert errors[1] < errors[0]


def test_exposure_rescales_abundance_not_signal(design):
    """A known sample scale must divide out of the estimate.

    This is what keeps `abundance_raw` a polymerase density rather than a
    library-size artefact, per plan section 15.
    """
    truth = np.array([50.0, 80.0])
    scale = 8.0
    observed = simulate(design, truth, seed=5, scale=scale)

    with_exposure = poisson.fit_locus(design, observed, exposure=Exposure(sample_scale=scale))
    without = poisson.fit_locus(design, observed)

    np.testing.assert_allclose(with_exposure.abundance, truth, rtol=0.15)
    # Ignoring the exposure inflates the estimate by roughly that factor.
    np.testing.assert_allclose(without.abundance / scale, with_exposure.abundance, rtol=0.05)


def test_bin_exposure_accounts_for_partial_terminal_bins(design):
    """A half-length bin has had half the opportunity to accumulate reads."""
    lengths = np.ones(8)
    lengths[-1] = 0.5
    truth = np.array([100.0, 100.0])
    rng = np.random.default_rng(6)
    observed = rng.poisson(lengths * (design @ truth)).astype(np.float64)

    aware = poisson.fit_locus(design, observed, exposure=Exposure(bin_exposure=lengths))
    naive = poisson.fit_locus(design, observed)
    # The model covering the short bin is underestimated when exposure is ignored.
    assert abs(aware.abundance[1] - 100.0) < abs(naive.abundance[1] - 100.0)


# --- zeros and edge cases ---------------------------------------------------


def test_zero_counts_need_no_pseudocount(design):
    """Empty bins must contribute their mu term and nothing else."""
    observed = np.zeros(8)
    observed[0] = 5.0
    # Finite objective and gradient despite seven zero bins.
    point = np.array([1.0, 1.0])
    assert np.isfinite(poisson.negative_log_likelihood(point, design, observed))
    assert np.all(np.isfinite(poisson.gradient(point, design, observed)))


def test_objective_is_finite_at_zero_abundance(design):
    """L-BFGS-B projects onto its lower bound, so a = 0 is always evaluated."""
    observed = simulate(design, np.array([10.0, 10.0]), seed=7)
    value = poisson.negative_log_likelihood(np.zeros(2), design, observed)
    assert np.isfinite(value)


def test_mu_floor_does_not_shift_the_optimum(design):
    """The floor is a numerical guard, not a pseudocount.

    Its value must be far below anything the optimum reaches, so scaling it
    by orders of magnitude cannot move the estimate.
    """
    observed = simulate(design, np.array([200.0, 300.0]), seed=8)
    baseline = poisson.fit_locus(design, observed)
    assert MU_FLOOR < 1e-9
    assert np.all(design @ baseline.abundance > 1e6 * MU_FLOOR)


def test_all_zero_counts_give_zero_not_a_failure(design):
    result = poisson.fit_locus(design, np.zeros(8))
    assert result.status is FitStatus.ALL_ZERO
    assert not is_failure(result.status)
    assert result.abundance.tolist() == [0.0, 0.0]


def test_bins_without_model_support_are_excluded(design):
    """Section 4.2: a bin no model reaches cannot constrain the fit.

    Signal there is unexplainable by construction, so including it would make
    the likelihood -inf for every parameter value rather than indicate a bad
    fit.
    """
    padded = np.vstack([design, np.zeros((2, 2))])
    observed = np.concatenate([simulate(design, np.array([30.0, 30.0]), seed=9), [7.0, 3.0]])
    assert usable_bins(padded, Exposure()).tolist() == [True] * 8 + [False, False]

    result = poisson.fit_locus(padded, observed)
    assert result.status is FitStatus.OK
    # The unexplainable bins neither break the fit nor bias it.
    reference = poisson.fit_locus(design, observed[:8])
    np.testing.assert_allclose(result.abundance, reference.abundance, rtol=1e-6)


def test_background_lets_unsupported_bins_contribute(design):
    """With a background, a bin no model reaches is no longer excluded."""
    padded = np.vstack([design, np.zeros((1, 2))])
    background = np.zeros(9)
    background[-1] = 2.0
    assert usable_bins(padded, Exposure(background=background))[-1]


def test_negative_counts_are_rejected(design):
    result = poisson.fit_locus(design, np.full(8, -1.0))
    assert result.status is FitStatus.NUMERICAL_ERROR
    assert np.isnan(result.abundance).all()


def test_non_finite_counts_are_rejected(design):
    observed = simulate(design, np.array([5.0, 5.0]), seed=10)
    observed[2] = np.nan
    assert poisson.fit_locus(design, observed).status is FitStatus.NUMERICAL_ERROR


@pytest.mark.parametrize(
    "exposure",
    [
        Exposure(sample_scale=0.0),
        Exposure(sample_scale=-1.0),
        Exposure(sample_scale=np.nan),
        Exposure(bin_exposure=np.zeros(8)),
        Exposure(background=np.full(8, -1.0)),
    ],
)
def test_invalid_exposure_is_reported(design, exposure):
    observed = simulate(design, np.array([10.0, 10.0]), seed=11)
    result = poisson.fit_locus(design, observed, exposure=exposure)
    assert result.status is FitStatus.INVALID_EXPOSURE
    assert np.isnan(result.abundance).all()


def test_prior_failure_short_circuits(design):
    result = poisson.fit_locus(design, np.full(8, np.nan), prior_status=FitStatus.MISSING_CONTIG)
    assert result.status is FitStatus.MISSING_CONTIG
    assert np.isnan(result.abundance).all()


def test_inactive_models_are_pinned(design):
    observed = simulate(design, np.array([40.0, 60.0]), seed=12)
    result = poisson.fit_locus(design, observed, inactive=np.array([True, False]))
    assert result.abundance[0] == 0.0
    assert result.abundance[1] > 0.0


# --- deviance ---------------------------------------------------------------


def test_deviance_is_zero_for_a_saturated_fit():
    """One model per bin can reproduce the counts exactly."""
    matrix = np.eye(5)
    observed = np.array([3.0, 0.0, 7.0, 2.0, 11.0])
    assert poisson.deviance(observed, matrix, observed) == pytest.approx(0.0, abs=1e-9)


def test_deviance_increases_with_misfit(design):
    observed = simulate(design, np.array([100.0, 100.0]), seed=13)
    good = poisson.fit_locus(design, observed)
    assert poisson.deviance(good.abundance, design, observed) < poisson.deviance(
        good.abundance * 3.0, design, observed
    )


# --- penalties --------------------------------------------------------------


def test_ridge_shrinks_toward_zero(design):
    observed = simulate(design, np.array([100.0, 100.0]), seed=14)
    unpenalized = poisson.fit_locus(design, observed)
    shrunk = poisson.fit_locus(design, observed, penalty=penalties.ridge(1.0))
    assert shrunk.abundance.sum() < unpenalized.abundance.sum()


def test_lasso_needs_no_proximal_operator(design):
    """Under a >= 0 the L1 term is linear, so the smooth driver suffices."""
    observed = simulate(design, np.array([50.0, 50.0]), seed=15)
    result = poisson.fit_locus(design, observed, penalty=penalties.lasso(5.0))
    assert result.status is FitStatus.OK
    assert np.all(result.abundance >= 0.0)


def test_lasso_cannot_zero_a_model_without_a_background(design):
    """A real limit of L1 under a Poisson likelihood, not a solver defect.

    A bin with nonzero counts reachable only by model k makes the negative log
    likelihood diverge as ``a_k -> 0``, so no finite penalty outweighs it and
    the L1 optimum is strictly positive. The exact gradient at ``a = 0`` is
    ``-inf``; the floor bounds it near ``-1e13``, which is why an absurd
    penalty eventually appears to win.
    """
    observed = simulate(design, np.array([2.0, 2.0]), seed=16)
    result = poisson.fit_locus(design, observed, penalty=penalties.lasso(1e4))
    assert result.abundance.sum() > 0.0
    # Still shrunk hard relative to the unpenalized fit.
    unpenalized = poisson.fit_locus(design, observed)
    assert result.abundance.sum() < 0.01 * unpenalized.abundance.sum()


def test_lasso_zeroes_a_model_when_a_background_explains_the_counts(design):
    """With a background the mean stays positive, so L1 can win.

    This is the configuration to use when sparsity is the goal.
    """
    observed = simulate(design, np.array([2.0, 2.0]), seed=16)
    exposure = Exposure(background=np.full(8, 1.0))
    result = poisson.fit_locus(design, observed, exposure=exposure, penalty=penalties.lasso(100.0))
    assert result.abundance.tolist() == [0.0, 0.0]


def test_ridge_stabilizes_near_collinear_allocation():
    """Gate C's motivation, checked directly: less replicate-to-replicate swing.

    Two nearly identical models cannot be separated, so the unpenalized split
    swings between replicates while their sum stays put. A ridge penalty damps
    the swing without being asked to invent resolution.
    """
    matrix = np.ones((60, 2))
    matrix[0, 1] = 0.0  # a single bin distinguishes them
    truth = np.array([50.0, 50.0])

    spreads = {}
    for penalty in (penalties.NO_PENALTY, penalties.ridge(1.0)):
        splits = []
        for seed in range(8):
            observed = simulate(matrix, truth, seed=100 + seed)
            fit = poisson.fit_locus(matrix, observed, penalty=penalty)
            splits.append(fit.abundance[0] - fit.abundance[1])
        spreads[penalty.kind] = float(np.std(splits))

    assert spreads["ridge"] < spreads["none"]


def test_penalty_is_recorded_for_provenance():
    described = penalties.elastic_net(0.3, 0.7).describe()
    assert described == {
        "penalty_type": "elastic_net",
        "penalty_lambda1": 0.3,
        "penalty_lambda2": 0.7,
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "ridge"}, "ridge penalty requires"),
        ({"kind": "lasso"}, "lasso penalty requires"),
        ({"kind": "elastic_net", "lambda1": 1.0}, "elastic_net penalty requires"),
        ({"kind": "none", "lambda2": 1.0}, "name the penalty explicitly"),
        ({"kind": "ridge", "lambda2": -1.0}, "must be non-negative"),
    ],
)
def test_invalid_penalties_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        penalties.Penalty(**kwargs)


def test_duplicate_columns_stay_unresolved_under_penalty():
    """Section 4.4: a penalty must not manufacture resolution.

    Exact duplicates are collapsed in `dienr.design`, but if a pair reaches a
    solver anyway, ridge splits them symmetrically rather than picking one.
    That is the honest answer: nothing in the data prefers either.
    """
    matrix = np.tile(np.ones((20, 1)), (1, 2))
    observed = simulate(matrix, np.array([25.0, 25.0]), seed=17)
    result = poisson.fit_locus(matrix, observed, penalty=penalties.ridge(0.5))
    assert result.abundance[0] == pytest.approx(result.abundance[1], rel=1e-4)


def test_pinned_model_does_not_abort_the_locus(design):
    """Regression: pinning must not make a locus unfittable.

    Bins 0-2 are reachable only by model 0. Pinning it leaves them with no
    free model, so they must be excluded from the likelihood rather than
    driving the mean to zero against nonzero counts and failing the locus.
    """
    observed = simulate(design, np.array([40.0, 60.0]), seed=12)
    result = poisson.fit_locus(design, observed, inactive=np.array([True, False]))
    assert result.status is FitStatus.OK
    assert result.abundance[0] == 0.0
    assert result.abundance[1] > 0.0
    # The surviving model is estimated from the bins it actually covers.
    assert result.abundance[1] == pytest.approx(60.0, rel=0.25)


def test_pinning_every_model_leaves_nothing_to_fit(design):
    observed = simulate(design, np.array([10.0, 10.0]), seed=18)
    result = poisson.fit_locus(design, observed, inactive=np.array([True, True]))
    assert result.status is FitStatus.INSUFFICIENT_BINS
    assert np.isnan(result.abundance).all()
