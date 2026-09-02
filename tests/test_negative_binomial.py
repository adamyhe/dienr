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
