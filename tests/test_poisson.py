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


# --- Gate B findings, pinned as regressions ---------------------------------


def test_legacy_log_is_misspecified_at_low_counts():
    """The dominant Gate B finding, kept as a test so it cannot be forgotten.

    A zero-count bin contributes ``[log(1e-3) - log(mu + 1e-3)]^2`` to the
    legacy log objective, roughly 38 at ``mu = 0.5``. Empty bins therefore
    dominate and drag the fit toward zero, which is why the reference's default
    transform is the worst option on sparse data.
    """
    from dienr.solvers import legacy

    matrix = np.ones((40, 1))
    truth = np.array([0.5])
    rng = np.random.default_rng(0)

    log_errors, identity_errors = [], []
    for _ in range(30):
        observed = rng.poisson(matrix @ truth).astype(np.float64)
        log_errors.append(
            abs(legacy.fit_locus(matrix, observed, transform="log").abundance[0] - 0.5)
        )
        identity_errors.append(
            abs(legacy.fit_locus(matrix, observed, transform="identity").abundance[0] - 0.5)
        )

    # The log transform is not merely worse, it is biased far low.
    assert np.median(log_errors) > 2 * np.median(identity_errors)


def mean_ridge_shrinkage(n_bins, truth, weight, replicates=20, seed=1):
    """Mean shrinkage of a ridge fit relative to the unpenalized fit.

    Measuring against the unpenalized fit on the *same* counts cancels the
    sampling noise the two share, which makes the penalty's effect visible at
    bin counts where comparing against the truth would not resolve it.
    """
    matrix = np.ones((n_bins, 1))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        observed = rng.poisson(matrix @ np.array([truth])).astype(np.float64)
        unpenalized = poisson.fit_locus(matrix, observed).abundance[0]
        penalized = poisson.fit_locus(matrix, observed, penalty=penalties.ridge(weight)).abundance[
            0
        ]
        values.append((unpenalized - penalized) / unpenalized)
    return float(np.mean(values))


def test_ridge_shrinkage_follows_two_lambda_a_over_n_bins():
    """The scaling law that makes a fixed penalty weight untransferable.

    The ridge gradient is ``2 lambda a`` while the Poisson score accumulates
    over bins, so relative shrinkage goes as ``2 lambda a / n_bins``: it grows
    with the abundance and shrinks with the amount of data. Predicted against
    measured, which is what makes the next two tests interpretable rather than
    magic numbers.
    """
    for n_bins, truth in [(40, 1.0), (40, 200.0), (400, 200.0)]:
        predicted = 2 * 0.01 * truth / n_bins
        measured = mean_ridge_shrinkage(n_bins, truth, 0.01)
        assert measured == pytest.approx(predicted, rel=0.25), (n_bins, truth)


def test_a_fixed_ridge_weight_biases_more_as_counts_grow():
    """Why a penalty weight must never be a shipped default.

    Same weight, same design, two count scales: negligible at one, an 8% bias
    at the other. Gate B saw the extreme version of this, a held-out deviance
    of 68.7 against 1.06 unpenalized.
    """
    low = mean_ridge_shrinkage(40, 1.0, 0.01)
    high = mean_ridge_shrinkage(40, 200.0, 0.01)
    assert low < 0.005
    assert high > 0.05
    assert high > 10 * low


def test_scaling_the_ridge_weight_restores_transfer():
    """The fix: ``lambda`` proportional to ``1 / a`` cancels the a-dependence.

    A moment estimate of the abundance suffices, and the resulting shrinkage is
    comparable at both ends of the count range.
    """
    shrinkage = {}
    for truth in (1.0, 200.0):
        # lambda = c / a leaves shrinkage at 2c / n_bins, independent of a.
        shrinkage[truth] = mean_ridge_shrinkage(40, truth, 0.01 / truth)
    assert max(shrinkage.values()) < 0.005
    assert shrinkage[200.0] == pytest.approx(shrinkage[1.0], rel=0.5)


def test_poisson_rejects_a_design_with_no_models():
    result = poisson.fit_locus(np.zeros((10, 0)), np.ones(10))
    assert result.status is FitStatus.INSUFFICIENT_BINS
    assert result.abundance.size == 0


def test_poisson_accepts_an_explicit_initial_point(design):
    """An explicit start must be honoured, and reach the same optimum."""
    truth = np.array([60.0, 90.0])
    observed = simulate(design, truth, seed=70)
    default = poisson.fit_locus(design, observed)
    explicit = poisson.fit_locus(design, observed, initial=np.array([50.0, 80.0]))
    assert explicit.status is FitStatus.OK
    np.testing.assert_allclose(explicit.abundance, default.abundance, rtol=1e-4)


def test_iteration_cap_reports_nonconvergence(design):
    """The maxiter path, which sets the NONCONVERGED status.

    A converged-looking result from a truncated run would be the worst kind of
    silent failure, so this asserts the status rather than the numbers.
    """
    observed = simulate(design, np.array([500.0, 900.0]), seed=71)
    result = poisson.fit_locus(design, observed, max_iterations=1)
    assert result.status is FitStatus.NONCONVERGED
    assert not result.converged
    assert "EXCEEDS LIMIT" in result.message
    # A truncated estimate is still returned, not NaN: it is a qualified
    # result, not a failure.
    assert np.all(np.isfinite(result.abundance))


@pytest.mark.parametrize(
    "exposure",
    [
        Exposure(bin_exposure=np.ones(3)),  # wrong length
        Exposure(background=np.ones(99)),
    ],
)
def test_misshaped_exposure_arrays_raise(design, exposure):
    """A shape mismatch is a programming error, not a data condition."""
    observed = simulate(design, np.array([5.0, 5.0]), seed=72)
    with pytest.raises(ValueError, match="expected"):
        poisson.fit_locus(design, observed, exposure=exposure)


def test_non_finite_bin_exposure_is_reported(design):
    observed = simulate(design, np.array([5.0, 5.0]), seed=73)
    bad = np.ones(8)
    bad[3] = np.inf
    result = poisson.fit_locus(design, observed, exposure=Exposure(bin_exposure=bad))
    assert result.status is FitStatus.INVALID_EXPOSURE


# --- why Poisson barely beats least squares here ---------------------------


def test_poisson_and_least_squares_coincide_on_a_binary_design():
    """On a 0/1 design column the two are the *same* estimator, not merely close.

    Both estimating equations differ only in the weight ``1 / mu_b``:

        Poisson:  sum_b M_bk (y_b - mu_b) / mu_b = 0
        MSE:      sum_b M_bk (y_b - mu_b)       = 0

    so when ``mu`` is constant across a model's support the weight factors out
    and the equations are identical. For one indicator column that is exactly
    the case: both reduce to ``a = sum(y) / n_support``.

    This is the mechanism behind the Gate B result. `bin_operation` rounds
    fractional occupancy to 0/1, which removes precisely the heterogeneity the
    Poisson weighting would exploit.
    """
    from dienr.solvers import legacy

    matrix = np.zeros((60, 1))
    matrix[10:50, 0] = 1.0
    rng = np.random.default_rng(0)

    for _ in range(50):
        observed = rng.poisson(matrix[:, 0] * 7.0).astype(np.float64)
        poisson_estimate = poisson.fit_locus(matrix, observed).abundance[0]
        mse_estimate = legacy.fit_locus(matrix, observed, transform="identity").abundance[0]
        assert poisson_estimate == pytest.approx(mse_estimate, abs=1e-9)
        # And both equal the closed form.
        assert poisson_estimate == pytest.approx(observed.sum() / 40.0, rel=1e-6)


def test_poisson_efficiency_gain_needs_a_graded_design():
    """The gain is bounded by ``sum(M^3) sum(M) / sum(M^2)^2``, tight in practice.

    Cauchy-Schwarz puts that ratio at or above 1, with equality iff ``M`` is
    constant. Even a fiftyfold range in occupancy gives only about 1.12, i.e.
    6% in standard error -- which is why the measured win rate sits near 55%
    rather than anywhere decisive.
    """

    def variance_ratio(column: np.ndarray) -> float:
        return float((column**3).sum() * column.sum() / ((column**2).sum() ** 2))

    flat = np.ones(200)
    graded = np.linspace(0.02, 1.0, 200)
    assert variance_ratio(flat) == pytest.approx(1.0)
    assert 1.05 < variance_ratio(graded) < 1.2


# --- starting point ----------------------------------------------------------


def _graded_design(entry_scale: float) -> np.ndarray:
    """Two overlapping, front-loaded columns: an exact-density style design."""
    rng = np.random.default_rng(3)
    matrix = np.zeros((60, 2))
    matrix[0:40, 0] = rng.gamma(2.0, 1.0, 40) + np.r_[np.full(4, 8.0), np.zeros(36)]
    matrix[15:60, 1] = rng.gamma(2.0, 1.0, 45) + np.r_[np.full(4, 8.0), np.zeros(41)]
    return matrix * entry_scale


def test_moment_start_is_the_single_model_poisson_mle():
    """Alone, a model's start is sum(y on its support) / sum(scale * M)."""
    from dienr.solvers._glm import moment_initialization

    matrix = _graded_design(1.0)
    observed = simulate(matrix, np.array([3.0, 0.5]), seed=1)
    scale = np.full(matrix.shape[0], 2.0)
    start = moment_initialization(matrix, observed, scale, np.zeros(matrix.shape[0]))
    for k in range(2):
        support = matrix[:, k] > 0
        assert start[k] == pytest.approx(observed[support].sum() / (2.0 * matrix[:, k].sum()))


@pytest.mark.parametrize("entry_scale", [1.0, 500.0])
def test_fit_is_invariant_to_column_scale(entry_scale):
    """Regression (plan 18.10): large design entries collapsed every abundance to 0.

    The old start, ``M'y / M'1``, was the M-weighted mean of y -- ~entry_scale^2
    too high -- and from there L-BFGS-B ran onto the zero bound and reported
    ``invalid_mean`` at 493/910 nascentRNASim loci with exact-density columns.
    Scaling a column by ``s`` must scale its abundance by exactly ``1/s``.
    """
    unit = _graded_design(1.0)
    observed = simulate(unit, np.array([0.8, 0.3]), seed=2)
    reference = poisson.fit_locus(unit, observed)
    scaled = poisson.fit_locus(unit * entry_scale, observed)
    assert reference.status is FitStatus.OK
    assert scaled.status is FitStatus.OK
    np.testing.assert_allclose(scaled.abundance * entry_scale, reference.abundance, rtol=1e-4)


# --------------------------------------------------------------------------
# Single-model fast path (plan 14.4)
# --------------------------------------------------------------------------

FAST_MESSAGE = "closed-form single-model Poisson MLE"


def _single_columns():
    binary = np.ones((64, 1))
    binary[:4] = 0.0
    graded = binary.copy()
    graded[40:, 0] = np.linspace(1.4, 0.15, 24)  # a run-on-like tail
    return {"binary": binary, "graded": graded}


@pytest.mark.parametrize("kind", ["binary", "graded"])
@pytest.mark.parametrize("scale", [None, "bins"])
def test_single_model_fast_path_matches_general_optimizer(kind, scale):
    """Same answer as L-BFGS-B, which is routed to by a pinned dummy column."""
    column = _single_columns()[kind]
    rng = np.random.default_rng(1)
    observed = rng.poisson(6.0 * column[:, 0]).astype(float)
    exposure = Exposure() if scale is None else Exposure(bin_exposure=rng.uniform(0.5, 1.5, 64))

    fast = poisson.fit_locus(column, observed, exposure=exposure)
    general = poisson.fit_locus(
        np.hstack([column, column]),
        observed,
        exposure=exposure,
        inactive=np.array([False, True]),
    )
    assert fast.message == FAST_MESSAGE
    assert general.message != FAST_MESSAGE
    assert fast.status is FitStatus.OK
    np.testing.assert_allclose(fast.abundance[0], general.abundance[0], rtol=1e-9)
    assert fast.objective == pytest.approx(general.objective, rel=1e-12)
    assert fast.gradient_norm < 1e-8 * observed.sum()


def test_single_model_fast_path_is_the_closed_form():
    column = _single_columns()["graded"]
    observed = np.random.default_rng(2).poisson(4.0 * column[:, 0]).astype(float)
    fast = poisson.fit_locus(column, observed)
    keep = column[:, 0] > 0
    assert fast.abundance[0] == pytest.approx(observed[keep].sum() / column[keep, 0].sum())


def test_single_model_fast_path_is_skipped_when_it_would_be_wrong():
    """Penalty, background, a supplied start, or a binding mu floor: general path."""
    column = _single_columns()["binary"]
    observed = np.random.default_rng(3).poisson(5.0 * column[:, 0]).astype(float)
    cases = [
        poisson.fit_locus(column, observed, penalty=penalties.ridge(1.0)),
        poisson.fit_locus(column, observed, exposure=Exposure(background=np.full(64, 0.1))),
        poisson.fit_locus(column, observed, initial=np.array([1.0])),
    ]
    tiny = column.copy()
    tiny[10, 0] = 1e-14  # a*m there sits below MU_FLOOR, so the floor binds
    sparse = observed.copy()
    sparse[10] = 0.0  # with counts there, the general path rightly says invalid_mean
    cases.append(poisson.fit_locus(tiny, sparse))
    for fit in cases:
        assert fit.message != FAST_MESSAGE
        assert fit.status is FitStatus.OK


def test_single_model_fast_path_leaves_short_circuits_alone():
    column = _single_columns()["binary"]
    assert poisson.fit_locus(column, np.zeros(64)).status is FitStatus.ALL_ZERO
    pinned = poisson.fit_locus(column, np.ones(64), inactive=np.array([True]))
    assert pinned.status is FitStatus.INSUFFICIENT_BINS
