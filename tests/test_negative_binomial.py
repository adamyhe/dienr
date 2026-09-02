"""NB2 fitting and shared dispersion estimation."""

from __future__ import annotations

import numpy as np
import pytest

from dienr.solvers import negative_binomial as nb
from dienr.solvers import penalties, poisson
from dienr.solvers._glm import Exposure
from dienr.status import FitStatus


@pytest.fixture
def design() -> np.ndarray:
    matrix = np.zeros((40, 2))
    matrix[0:25, 0] = 1.0
    matrix[15:40, 1] = 1.0
    return matrix


def simulate_poisson(matrix, truth, seed):
    return np.random.default_rng(seed).poisson(matrix @ truth).astype(np.float64)


def simulate_nb(matrix, truth, dispersion, seed):
    """Draw NB2 counts with Var = mu + phi mu^2, via a gamma-Poisson mixture."""
    rng = np.random.default_rng(seed)
    mu = matrix @ truth
    size = 1.0 / dispersion
    rate = rng.gamma(shape=size, scale=mu / size)
    return rng.poisson(rate).astype(np.float64)


# --- gradient correctness ---------------------------------------------------


@pytest.mark.parametrize("dispersion", [0.05, 0.5, 2.0])
def test_gradient_matches_finite_differences(design, dispersion):
    observed = simulate_nb(design, np.array([30.0, 50.0]), dispersion, seed=0)
    point = np.array([26.0, 44.0])

    analytic = nb.gradient(point, design, observed, dispersion)
    step = 1e-6
    numeric = np.empty_like(point)
    for i in range(point.size):
        up, down = point.copy(), point.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (
            nb.negative_log_likelihood(up, design, observed, dispersion)
            - nb.negative_log_likelihood(down, design, observed, dispersion)
        ) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_gradient_matches_finite_differences_with_penalty(design):
    observed = simulate_nb(design, np.array([20.0, 20.0]), 0.3, seed=1)
    penalty = penalties.elastic_net(0.5, 0.2)
    point = np.array([18.0, 19.0])

    analytic = nb.gradient(point, design, observed, 0.3, penalty=penalty)
    step = 1e-6
    numeric = np.empty_like(point)
    for i in range(point.size):
        up, down = point.copy(), point.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (
            nb.negative_log_likelihood(up, design, observed, 0.3, penalty=penalty)
            - nb.negative_log_likelihood(down, design, observed, 0.3, penalty=penalty)
        ) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_nb_approaches_poisson_as_dispersion_vanishes(design):
    """A sanity check on the parameterization: phi -> 0 is the Poisson limit."""
    observed = simulate_poisson(design, np.array([30.0, 30.0]), seed=2)
    tiny = nb.fit_locus(design, observed, dispersion=1e-6)
    reference = poisson.fit_locus(design, observed)
    np.testing.assert_allclose(tiny.abundance, reference.abundance, rtol=1e-3)


# --- recovery ---------------------------------------------------------------


def test_recovers_known_abundances(design):
    truth = np.array([100.0, 200.0])
    observed = simulate_nb(design, truth, 0.1, seed=3)
    result = nb.fit_locus(design, observed, dispersion=0.1)
    assert result.status is FitStatus.OK
    np.testing.assert_allclose(result.abundance, truth, rtol=0.35)


def test_nb_is_not_an_outlier_robust_estimator(design):
    """NB2 does not down-weight a single spike, and should not be sold as doing so.

    The NB2 score weights bin ``b`` by ``r / (mu_b (r + mu_b))``, which depends
    on the *fitted* mean, not on how far the observation sits from it. Bins
    sharing a fitted mean therefore share a weight, so one large excursion pulls
    an NB2 fit essentially as hard as it pulls a Poisson fit. What NB2 changes
    is the variance function, and hence inference; it is not a robust estimator.

    This matters for Gate D: if NB is adopted, it should be for calibrated
    uncertainty, not for point estimates, which barely move.
    """
    truth = np.array([50.0, 50.0])
    observed = simulate_poisson(design, truth, seed=4)
    observed[5] += 500.0

    poisson_fit = poisson.fit_locus(design, observed)
    nb_fit = nb.fit_locus(design, observed, dispersion=1.0)

    # Both are pulled well away from the truth, and by a similar amount.
    assert poisson_fit.abundance[0] > 70.0
    assert nb_fit.abundance[0] > 70.0
    assert nb_fit.abundance[0] == pytest.approx(poisson_fit.abundance[0], rel=0.15)


def test_nb_reweights_bins_by_fitted_mean(design):
    """What NB2 does change: relative weight across bins of unequal mean.

    With ``Var = mu + phi mu^2``, a high-mean bin carries proportionally more
    variance than Poisson assumes and so contributes less to the score. Where
    fitted means differ across a locus, NB2 and Poisson therefore disagree.
    """
    truth = np.array([200.0, 5.0])  # the two models differ by 40x
    observed = simulate_nb(design, truth, 0.4, seed=40)

    poisson_fit = poisson.fit_locus(design, observed)
    nb_fit = nb.fit_locus(design, observed, dispersion=0.4)
    assert poisson_fit.status is FitStatus.OK
    assert nb_fit.status is FitStatus.OK
    # Same data, same design, materially different allocation.
    assert not np.allclose(nb_fit.abundance, poisson_fit.abundance, rtol=0.02)


def test_nb_and_poisson_agree_closely_at_equal_means(design):
    """The counterpart: with a flat mean there is little for NB to change."""
    observed = simulate_nb(design, np.array([60.0, 60.0]), 0.2, seed=41)
    poisson_fit = poisson.fit_locus(design, observed)
    nb_fit = nb.fit_locus(design, observed, dispersion=0.2)
    np.testing.assert_allclose(nb_fit.abundance, poisson_fit.abundance, rtol=0.05)


# --- dispersion estimation --------------------------------------------------


def test_dispersion_is_near_zero_for_poisson_data(design):
    """Equidispersed counts must not be given spurious overdispersion."""
    loci = []
    for seed in range(12):
        observed = simulate_poisson(design, np.array([40.0, 60.0]), seed=200 + seed)
        fit = poisson.fit_locus(design, observed)
        loci.append((design, observed, fit.abundance))

    estimate = nb.estimate_dispersion(loci)
    assert estimate.dispersion < 0.01
    assert estimate.pearson_chi2_per_df == pytest.approx(1.0, abs=0.3)
    assert estimate.n_loci == 12


def test_dispersion_is_recovered_for_overdispersed_data(design):
    truth = np.array([60.0, 60.0])
    for true_phi in (0.1, 0.5):
        loci = []
        for seed in range(40):
            observed = simulate_nb(design, truth, true_phi, seed=int(true_phi * 1000) + seed)
            fit = poisson.fit_locus(design, observed)
            if fit.status is FitStatus.OK:
                loci.append((design, observed, fit.abundance))
        estimate = nb.estimate_dispersion(loci)
        assert estimate.is_overdispersed
        # Moment estimators of dispersion are noisy; a factor of two is enough
        # to establish it is measuring the right quantity.
        assert 0.5 * true_phi < estimate.dispersion < 2.0 * true_phi


def test_dispersion_pools_across_loci_not_within_one(design):
    """Section 4.3 forbids per-locus dispersion; pooling is what stabilizes it.

    One locus gives a wildly variable estimate; forty give a stable one.
    """
    truth = np.array([60.0, 60.0])
    singles = []
    for seed in range(20):
        observed = simulate_nb(design, truth, 0.3, seed=500 + seed)
        fit = poisson.fit_locus(design, observed)
        singles.append(nb.estimate_dispersion([(design, observed, fit.abundance)]).dispersion)

    pooled = []
    for block in range(5):
        loci = []
        for seed in range(40):
            observed = simulate_nb(design, truth, 0.3, seed=700 + block * 40 + seed)
            fit = poisson.fit_locus(design, observed)
            loci.append((design, observed, fit.abundance))
        pooled.append(nb.estimate_dispersion(loci).dispersion)

    assert np.std(pooled) < np.std(singles)


def test_dispersion_estimate_is_clipped_and_flagged():
    """An absurd estimate usually means the mean model is wrong."""
    matrix = np.ones((30, 1))
    observed = np.zeros(30)
    observed[0] = 1e6  # nothing like a mean-model fit
    estimate = nb.estimate_dispersion([(matrix, observed, np.array([1.0]))])
    assert estimate.clipped
    assert estimate.dispersion == nb.MAX_DISPERSION


def test_empty_dispersion_input_is_handled():
    estimate = nb.estimate_dispersion([])
    assert estimate.dispersion == 0.0
    assert estimate.n_loci == 0


def test_compare_dispersion_models_reports_the_gate_d_numbers(design):
    loci = []
    for seed in range(12):
        observed = simulate_poisson(design, np.array([40.0, 40.0]), seed=900 + seed)
        fit = poisson.fit_locus(design, observed)
        loci.append((design, observed, fit.abundance))
    report = nb.compare_dispersion_models(loci)
    assert set(report) == {
        "nb2_dispersion",
        "quasi_poisson_dispersion",
        "n_bins",
        "n_loci",
        "overdispersion_detected",
        "dispersion_clipped",
    }
    assert report["overdispersion_detected"] == 0.0


# --- guards -----------------------------------------------------------------


@pytest.mark.parametrize("dispersion", [0.0, -1.0, 1e-12, np.nan])
def test_unusable_dispersion_is_rejected_loudly(design, dispersion):
    """NB at zero dispersion is Poisson; make the caller say which they mean."""
    observed = simulate_poisson(design, np.array([10.0, 10.0]), seed=5)
    with pytest.raises(ValueError, match="dispersion must exceed"):
        nb.fit_locus(design, observed, dispersion=dispersion)


def test_all_zero_counts_give_zero_not_a_failure(design):
    result = nb.fit_locus(design, np.zeros(40), dispersion=0.2)
    assert result.status is FitStatus.ALL_ZERO
    assert result.abundance.tolist() == [0.0, 0.0]


def test_prior_failure_short_circuits(design):
    result = nb.fit_locus(
        design, np.full(40, np.nan), dispersion=0.2, prior_status=FitStatus.MISSING_CONTIG
    )
    assert result.status is FitStatus.MISSING_CONTIG
    assert np.isnan(result.abundance).all()


def test_invalid_exposure_is_reported(design):
    observed = simulate_poisson(design, np.array([10.0, 10.0]), seed=6)
    result = nb.fit_locus(design, observed, dispersion=0.2, exposure=Exposure(sample_scale=-1.0))
    assert result.status is FitStatus.INVALID_EXPOSURE


def test_pinned_model_does_not_abort_the_locus(design):
    observed = simulate_nb(design, np.array([40.0, 60.0]), 0.2, seed=7)
    result = nb.fit_locus(design, observed, dispersion=0.2, inactive=np.array([True, False]))
    assert result.status is FitStatus.OK
    assert result.abundance[0] == 0.0
    assert result.abundance[1] > 0.0


def test_objective_includes_the_constant_terms(design):
    """`objective` must be the full likelihood, not the optimizer's reduced form.

    The reduced form drops terms constant in the abundance, which is fine for
    optimization but would make the reported value incomparable to anything.
    """
    observed = simulate_nb(design, np.array([30.0, 30.0]), 0.25, seed=8)
    result = nb.fit_locus(design, observed, dispersion=0.25)
    assert result.objective == pytest.approx(
        nb.negative_log_likelihood(result.abundance, design, observed, 0.25)
    )


# --- score structure and convexity -----------------------------------------


def test_score_is_weight_times_residual(design):
    """NB2's score uses the residual linearly, weighted by 1 / V(mu).

    This is the standard GLM form, ``sum_b M_bk (mu_b - y_b) / V(mu_b)`` with
    ``V(mu) = mu + phi mu^2``. The residual drives the fit; only the *weight*
    is a function of the fitted mean.
    """
    dispersion = 0.4
    observed = simulate_nb(design, np.array([30.0, 30.0]), dispersion, seed=50)
    point = np.array([28.0, 31.0])

    mu = design @ point
    variance = mu + dispersion * mu**2
    glm_form = design.T @ ((mu - observed) / variance)
    np.testing.assert_allclose(
        nb.gradient(point, design, observed, dispersion), glm_form, rtol=1e-12
    )


def test_influence_of_an_observation_is_unbounded():
    """Why NB2 is not robust: the score grows without bound in y.

    The weight depends only on the fitted mean, so a larger residual
    contributes proportionally more, with no cap. Bounding it would require an
    estimating equation whose weights depend on y, which is M-estimation and a
    different estimator, not NB2 maximum likelihood.
    """
    matrix = np.ones((6, 1))
    dispersion = 0.4
    contributions = []
    for spike in (5.0, 40.0, 400.0):
        observed = np.full(6, 5.0)
        observed[2] = spike
        contributions.append(abs(nb.gradient(np.array([5.0]), matrix, observed, dispersion)[0]))
    # Strictly increasing and roughly linear in the spike size.
    assert contributions[0] < contributions[1] < contributions[2]
    assert contributions[2] / contributions[1] > 5.0


@pytest.mark.parametrize(
    ("observed_value", "mu", "dispersion", "expect_convex"),
    [
        (5.0, 5.0, 0.4, True),
        (5.0, 30.0, 0.4, False),
        (0.0, 10.0, 0.4, False),  # a zero-count bin is always concave
        (1.0, 20.0, 2.0, False),
    ],
)
def test_nb_identity_link_is_not_globally_convex(observed_value, mu, dispersion, expect_convex):
    """Unlike Poisson, NB2 with an identity link can be locally concave.

    Checked against a numerical second derivative so the analytic condition in
    the module docstring cannot drift from the implementation.
    """
    matrix = np.ones((1, 1))
    step = 1e-4

    def value(m):
        return nb.negative_log_likelihood(
            np.array([m]), matrix, np.array([observed_value]), dispersion
        )

    second = (value(mu + step) - 2 * value(mu) + value(mu - step)) / step**2
    size = 1.0 / dispersion
    analytic = (
        size
        * (-(mu**2) + 2 * mu * observed_value + size * observed_value)
        / (mu**2 + size * mu) ** 2
    )
    assert second == pytest.approx(analytic, rel=1e-3)
    assert (second > 0) is expect_convex


@pytest.mark.parametrize("dispersion", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("mu", [0.5, 5.0, 50.0])
def test_zero_count_bins_are_always_concave(dispersion, mu):
    """The clearest source of the non-convexity, measured on the real objective."""
    matrix = np.ones((1, 1))
    step = 1e-4

    def value(m):
        return nb.negative_log_likelihood(np.array([m]), matrix, np.array([0.0]), dispersion)

    second = (value(mu + step) - 2 * value(mu) + value(mu - step)) / step**2
    assert second < 0


def test_collapsed_mean_is_flagged_not_returned_as_zero(design):
    """Regression: NB lacked the Poisson solver's mean-collapse guard.

    A pathological start terminated with abundance exactly zero and an
    objective off by orders of magnitude, reported as a converged fit. Because
    the NB2 identity-link objective is not globally convex, this cannot be
    ruled out by appealing to convexity the way it can for Poisson.
    """
    observed = simulate_poisson(design, np.array([1.5, 1.5]), seed=51)
    result = nb.fit_locus(design, observed, dispersion=1e-6, initial=np.full(2, 1e4))
    assert result.status is FitStatus.INVALID_MEAN
    assert np.isnan(result.abundance).all()
    assert "vanished" in result.failure_reason


def test_default_initialization_avoids_the_collapse(design):
    """The moment start lands in the convex region, which is why it is default."""
    observed = simulate_poisson(design, np.array([1.5, 1.5]), seed=51)
    result = nb.fit_locus(design, observed, dispersion=1e-6)
    assert result.status is FitStatus.OK
    assert np.all(result.abundance > 0)


def test_penalized_nb_fit_runs_through_the_fused_path(design):
    """Coverage found this: the penalty branch of the NB fused closure was dead.

    `nb.gradient` was exercised with a penalty, but no test ever reached
    `fit_locus` with an active one, so the fused closure's penalty arm never
    ran. Ridge must shrink here exactly as it does for Poisson.
    """
    observed = simulate_nb(design, np.array([100.0, 100.0]), 0.2, seed=60)
    unpenalized = nb.fit_locus(design, observed, dispersion=0.2)
    shrunk = nb.fit_locus(design, observed, dispersion=0.2, penalty=penalties.ridge(1.0))
    assert shrunk.status is FitStatus.OK
    assert shrunk.abundance.sum() < unpenalized.abundance.sum()
    # And the reported objective includes the penalty.
    assert shrunk.objective > nb.negative_log_likelihood(shrunk.abundance, design, observed, 0.2)


def test_nb_rejects_a_design_with_no_models(design):
    result = nb.fit_locus(np.zeros((10, 0)), np.ones(10), dispersion=0.2)
    assert result.status is FitStatus.INSUFFICIENT_BINS
    assert result.abundance.size == 0


def test_nb_validates_the_inactive_shape(design):
    observed = simulate_poisson(design, np.array([5.0, 5.0]), seed=61)
    with pytest.raises(ValueError, match="inactive must have shape"):
        nb.fit_locus(design, observed, dispersion=0.2, inactive=np.array([True]))


# --- what NB2 does and does not buy ----------------------------------------


@pytest.mark.parametrize("dispersion", [0.3, 3.0])
def test_nb_poisson_and_least_squares_coincide_on_a_binary_design(dispersion):
    """NB2 cannot improve the point estimate on a 0/1 design, at any dispersion.

    For an identity link the score weights each residual by ``1 / V(mu)``. When
    ``mu`` is constant across a model's support that weight factors out, so the
    estimating equation reduces to the sample mean *whatever the variance
    function is*. Poisson and NB2 therefore give bit-identical answers to plain
    least squares here, even under severe overdispersion.

    This is the sharper form of the Gate B result: the limitation is the binary
    design, not the choice of likelihood.
    """
    from dienr.solvers import legacy, poisson

    matrix = np.zeros((60, 1))
    matrix[10:50, 0] = 1.0
    rng = np.random.default_rng(0)

    for _ in range(25):
        observed = simulate_nb(matrix, np.array([7.0]), dispersion, seed=int(rng.integers(1e6)))
        mse = legacy.fit_locus(matrix, observed, transform="identity").abundance[0]
        pois = poisson.fit_locus(matrix, observed).abundance[0]
        nb2 = nb.fit_locus(matrix, observed, dispersion=dispersion).abundance[0]
        assert pois == pytest.approx(mse, abs=1e-9)
        assert nb2 == pytest.approx(mse, abs=1e-9)
        assert nb2 == pytest.approx(observed.sum() / 40.0, rel=1e-6)


def test_poisson_standard_error_understates_overdispersed_spread():
    """What NB2 *is* for, and the reason a Poisson-only pipeline misleads.

    The estimates agree, but the uncertainty does not. With identity link and
    constant mu the model-based standard error is ``sqrt(V(mu) / n)``, so
    Poisson reports ``sqrt(a/n)`` while the truth is ``sqrt(a(1 + phi a)/n)``.
    At phi = 1 and a = 7 that is an understatement of about 60%.
    """
    n_support, truth = 40, 7.0
    matrix = np.zeros((60, 1))
    matrix[10:50, 0] = 1.0

    for dispersion, min_understatement in ((0.3, 0.30), (1.0, 0.50)):
        estimates = [
            simulate_nb(matrix, np.array([truth]), dispersion, seed=s).sum() / n_support
            for s in range(600)
        ]
        true_sd = float(np.std(estimates))
        mean_estimate = float(np.mean(estimates))
        poisson_se = np.sqrt(mean_estimate / n_support)
        nb_se = np.sqrt(mean_estimate * (1 + dispersion * mean_estimate) / n_support)

        assert 1 - poisson_se / true_sd > min_understatement
        assert nb_se == pytest.approx(true_sd, rel=0.15)
