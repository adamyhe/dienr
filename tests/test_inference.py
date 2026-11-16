"""Standard errors and quasi-Poisson dispersion."""

from __future__ import annotations

import numpy as np
import pytest

from dienr.solvers import inference, poisson
from dienr.solvers._glm import Exposure


@pytest.fixture
def indicator() -> np.ndarray:
    """One 0/1 column over 40 of 60 bins, the shape a real design produces."""
    matrix = np.zeros((60, 1))
    matrix[10:50, 0] = 1.0
    return matrix


def simulate_nb(matrix, truth, dispersion, seed):
    """NB2 counts with Var = mu + phi mu^2, via a gamma-Poisson mixture."""
    rng = np.random.default_rng(seed)
    mu = np.maximum(matrix @ truth, 1e-12)
    size = 1.0 / dispersion
    return rng.poisson(rng.gamma(shape=size, scale=mu / size)).astype(np.float64)


# --- the closed form -------------------------------------------------------


def test_poisson_standard_error_matches_the_closed_form(indicator):
    """For one indicator column the Poisson error is ``sqrt(a / n_support)``."""
    abundance = np.array([7.0])
    errors = inference.standard_errors(indicator, abundance)
    assert errors[0] == pytest.approx(np.sqrt(7.0 / 40.0), rel=1e-9)


def test_dispersion_scales_every_error_by_its_square_root(indicator):
    """The whole of the quasi-Poisson correction, asserted directly."""
    abundance = np.array([7.0])
    base = inference.standard_errors(indicator, abundance)
    for phi in (0.5, 2.0, 9.0):
        scaled = inference.standard_errors(indicator, abundance, dispersion=phi)
        np.testing.assert_allclose(scaled, np.sqrt(phi) * base, rtol=1e-12)


def test_exposure_enters_the_information(indicator):
    """Doubling the sample scale halves the standard error of the abundance.

    The abundance is a rate per unit exposure, so more exposure means a
    better-determined rate.
    """
    abundance = np.array([7.0])
    single = inference.standard_errors(indicator, abundance)
    doubled = inference.standard_errors(indicator, abundance, exposure=Exposure(sample_scale=4.0))
    assert doubled[0] == pytest.approx(single[0] / 2.0, rel=1e-9)


# --- calibration, which is the point of the feature ------------------------


@pytest.mark.parametrize(
    ("dispersion", "min_poisson_understatement"),
    [(0.1, 0.15), (0.3, 0.30), (1.0, 0.50)],
)
def test_quasi_poisson_errors_are_calibrated_where_poisson_errors_are_not(
    indicator, dispersion, min_poisson_understatement
):
    """The reason quasi-Poisson exists: honest uncertainty at the same estimate.

    Compares each model-based standard error against the *observed* spread of
    estimates over many replicates. Poisson understates it once the counts are
    overdispersed; the pooled quasi-Poisson correction recovers it.
    """
    truth = np.array([7.0])
    estimates, quasi_errors = [], []

    for seed in range(400):
        observed = simulate_nb(indicator, truth, dispersion, seed)
        fit = poisson.fit_locus(indicator, observed)
        estimates.append(fit.abundance[0])
        estimate = inference.estimate_quasi_poisson_dispersion(
            [(indicator, observed, fit.abundance)]
        )
        quasi_errors.append(
            inference.standard_errors(indicator, fit.abundance, dispersion=estimate.dispersion)[0]
        )

    true_spread = float(np.std(estimates))
    poisson_error = float(
        np.mean([inference.standard_errors(indicator, np.array([e]))[0] for e in estimates])
    )
    quasi_error = float(np.mean(quasi_errors))

    assert 1 - poisson_error / true_spread > min_poisson_understatement
    assert quasi_error == pytest.approx(true_spread, rel=0.15)


def test_dispersion_is_one_for_poisson_data(indicator):
    """No overdispersion must mean no correction."""
    rng = np.random.default_rng(3)
    triples = []
    for _ in range(40):
        observed = rng.poisson(indicator[:, 0] * 12.0).astype(np.float64)
        fit = poisson.fit_locus(indicator, observed)
        triples.append((indicator, observed, fit.abundance))

    estimate = inference.estimate_quasi_poisson_dispersion(triples)
    assert estimate.dispersion == pytest.approx(1.0, abs=0.15)
    assert not estimate.is_overdispersed()
    assert estimate.n_loci == 40


def test_overdispersion_test_accounts_for_its_own_sampling_noise():
    """A bare `dispersion > 1` test would fire on half of Poisson datasets.

    The Pearson statistic over df has standard error ``sqrt(2 / df)`` under the
    null, so the threshold has to carry that headroom. This caught a real
    knife-edge bug: pure Poisson data gave 1.0215 and was reported
    overdispersed.
    """
    # 1560 df gives a standard error of about 0.036, so 1.02 is well inside
    # the null and 1.5 is far outside.
    borderline = inference.DispersionEstimate(1.0215, 1600, 40, 1560)
    assert borderline.standard_error == pytest.approx(np.sqrt(2 / 1560), rel=1e-9)
    assert not borderline.is_overdispersed()

    clear = inference.DispersionEstimate(1.5, 1600, 40, 1560)
    assert clear.is_overdispersed()

    # With no degrees of freedom nothing can be concluded.
    assert not inference.DispersionEstimate(9.0, 0, 0, 0).is_overdispersed()


def test_dispersion_rises_with_overdispersion(indicator):
    previous = 0.0
    for dispersion in (0.0, 0.2, 1.0):
        triples = []
        for seed in range(40):
            observed = (
                np.random.default_rng(seed).poisson(indicator[:, 0] * 10.0).astype(float)
                if dispersion == 0
                else simulate_nb(indicator, np.array([10.0]), dispersion, seed)
            )
            fit = poisson.fit_locus(indicator, observed)
            triples.append((indicator, observed, fit.abundance))
        estimate = inference.estimate_quasi_poisson_dispersion(triples)
        assert estimate.dispersion > previous
        previous = estimate.dispersion


def test_empty_dispersion_input_falls_back_to_the_poisson_assumption():
    estimate = inference.estimate_quasi_poisson_dispersion([])
    assert estimate is inference.POISSON_DISPERSION
    assert estimate.dispersion == 1.0


def test_dispersion_provenance_is_recorded(indicator):
    record = inference.estimate_quasi_poisson_dispersion(
        [(indicator, indicator[:, 0] * 5.0, np.array([5.0]))]
    ).describe()
    assert record["dispersion_method"] == "quasi_poisson_pearson_pooled"
    assert record["dispersion_degrees_of_freedom"] > 0


# --- where a Wald error is not defined -------------------------------------


def test_boundary_models_get_nan_not_a_number():
    """A model at zero has a one-sided distribution; a symmetric error misleads."""
    matrix = np.zeros((40, 2))
    matrix[0:25, 0] = 1.0
    matrix[15:40, 1] = 1.0
    errors = inference.standard_errors(matrix, np.array([5.0, 0.0]))
    assert np.isfinite(errors[0])
    assert np.isnan(errors[1])


def test_all_boundary_gives_all_nan():
    matrix = np.ones((20, 2))
    assert np.isnan(inference.standard_errors(matrix, np.zeros(2))).all()


def test_singular_information_gives_nan_not_a_huge_number():
    """Rank deficiency makes the error undefined, not merely large.

    Reporting a large finite number would invite a reader to treat it as a wide
    interval, when in fact the parameter is not identified at all.
    """
    duplicated = np.tile(np.ones((30, 1)), (1, 2))
    errors = inference.standard_errors(duplicated, np.array([3.0, 3.0]))
    assert np.isnan(errors).all()


def test_non_finite_abundance_gives_nan():
    matrix = np.ones((10, 1))
    assert np.isnan(inference.standard_errors(matrix, np.array([np.nan]))).all()


def test_empty_model_set_is_handled():
    assert inference.standard_errors(np.zeros((10, 0)), np.zeros(0)).size == 0


def _graded_locus(seed: int, n_bins: int = 60, n_models: int = 3):
    rng = np.random.default_rng(seed)
    models = rng.uniform(0.2, 3.0, size=(n_bins, n_models))
    models *= rng.uniform(size=(n_bins, n_models)) > 0.35
    models[:5, 0] = 1.0  # every column keeps some support
    models[-5:, -1] = 1.0
    truth = rng.uniform(1.0, 8.0, size=n_models)
    return models, rng, truth


EXPOSURES = {
    "unit": lambda n, rng: Exposure(),
    "sample_scale": lambda n, rng: Exposure(sample_scale=3.0),
    "bin_exposure": lambda n, rng: Exposure(bin_exposure=np.full(n, 250.0)),
    "background": lambda n, rng: Exposure(sample_scale=2.0, background=rng.uniform(0.1, 1.0, n)),
}


@pytest.mark.parametrize("exposure_name", sorted(EXPOSURES))
@pytest.mark.parametrize("pin", [False, True])
def test_fit_locus_with_mean_hands_on_exactly_what_would_be_recomputed(exposure_name, pin):
    """The kept mean must be bit-identical to rebuilding it (plan 14.7)."""
    from dienr.solvers._glm import mean, usable_bins

    models, rng, truth = _graded_locus(11)
    n_bins, n_models = models.shape
    exposure = EXPOSURES[exposure_name](n_bins, rng)
    observed = rng.poisson(exposure.scale(n_bins) * (models @ truth) + exposure.offset(n_bins))
    observed = observed.astype(np.float64)
    inactive = np.array([False, True, False]) if pin else None

    fit, kept = poisson.fit_locus_with_mean(models, observed, exposure=exposure, inactive=inactive)
    assert kept is not None and fit.status.value == "ok"
    scale, offset = exposure.scale(n_bins), exposure.offset(n_bins)
    np.testing.assert_array_equal(kept.raw, scale * (models @ fit.abundance) + offset)
    np.testing.assert_array_equal(kept.floored, mean(fit.abundance, models, scale, offset))
    np.testing.assert_array_equal(kept.usable, usable_bins(models, exposure, inactive))

    plain = poisson.fit_locus(models, observed, exposure=exposure, inactive=inactive)
    np.testing.assert_array_equal(plain.abundance, fit.abundance)
    assert plain.fitted_total == fit.fitted_total
    assert plain.residual_sum_of_squares == fit.residual_sum_of_squares
    assert plain.penalty_value == fit.penalty_value
    assert plain.status is fit.status and plain.objective == fit.objective
    assert plain.fitted_total == float((scale * (models @ fit.abundance) + offset).sum())

    recomputed = inference.dispersion_term(
        models, observed, fit.abundance, exposure=exposure, pinned=inactive
    )
    reused = inference.dispersion_term(
        models,
        observed,
        fit.abundance,
        exposure=exposure,
        pinned=inactive,
        fitted=kept.floored,
        usable=kept.usable,
    )
    assert reused == recomputed


def test_fit_locus_with_mean_keeps_nothing_on_shortcuts():
    """Shortcut and failure paths carry no mean; callers recompute."""
    models, rng, _ = _graded_locus(5)
    single = models[:, :1]
    _, kept = poisson.fit_locus_with_mean(single, rng.poisson(4.0, size=single.shape[0]) * 1.0)
    assert kept is None  # single-model closed form
    _, kept = poisson.fit_locus_with_mean(models, np.zeros(models.shape[0]))
    assert kept is None  # all-zero locus
    unsupported = models.copy()
    unsupported[:, 1] = 0.0
    _, kept = poisson.fit_locus_with_mean(unsupported, rng.poisson(4.0, size=models.shape[0]) * 1.0)
    assert kept is None  # a model with no support goes through fit_supported_models
