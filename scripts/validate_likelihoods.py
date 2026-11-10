"""Gate B: does the Poisson likelihood actually beat the legacy objectives?

Plan section 12, Phase 3 exit gate. Poisson must

1. recover known abundances in Poisson simulation,
2. improve held-out deviance or abundance error over legacy fitting in
   prespecified low-count and heterogeneous-depth regimes, and
3. not regress materially on high-count simulations.

Method notes that matter for the verdict being meaningful:

- **Held-out blocks are contiguous, never random bins.** Section 12 says so for
  hyperparameter selection and the same reasoning applies here: neighbouring
  bins of one transcript are near-duplicates, so a random split leaks the
  answer across the fold boundary and every method looks good.
- **All methods see identical data.** Counts are simulated once per replicate
  and handed to each objective unchanged. The legacy objectives were designed
  for the reference's mean-coverage input rather than raw counts, which is
  itself part of what is being tested: legacy-log's fixed ``1e-3`` offset is
  not scale-invariant, so it behaves differently on counts than on densities.
- **Held-out deviance is the primary metric,** since it needs no access to the
  truth and so is the one available on real data. Abundance error and rank
  correlation are reported alongside because they answer the question users
  actually ask.
- **Regimes are declared here, before running,** rather than chosen after
  seeing which favours Poisson.
- **Comparisons are paired.** Every method sees the identical counts and the
  identical fold, so the meaningful statistic is the per-fold difference and
  the fraction of folds each method wins. Comparing medians of independently
  aggregated columns would let a 2% gap look decisive when it is noise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from dienr.solvers import legacy, poisson
from dienr.solvers._glm import Exposure
from dienr.solvers.penalties import ridge
from dienr.status import FitStatus


@dataclass(frozen=True)
class Regime:
    """One prespecified simulation regime.

    Attributes:
        name: Identifier.
        purpose: What it is meant to discriminate.
        mean_counts: Approximate expected counts per covered bin.
        depth_factors: Per-sample exposure multipliers. More than one entry
            makes the regime heterogeneous in depth.
        n_replicates: Independent count draws per configuration.
    """

    name: str
    purpose: str
    mean_counts: float
    depth_factors: tuple[float, ...]
    n_replicates: int = 24


REGIMES = (
    Regime(
        name="low_count",
        purpose="counts near zero, where a log-transformed sum of squares is "
        "most badly misspecified",
        mean_counts=0.5,
        depth_factors=(1.0,),
    ),
    Regime(
        name="moderate_count",
        purpose="the middle of the range, where neither objective is obviously wrong",
        mean_counts=8.0,
        depth_factors=(1.0,),
    ),
    Regime(
        name="high_count",
        purpose="counts where the Poisson is nearly Gaussian; Poisson must not regress here",
        mean_counts=200.0,
        depth_factors=(1.0,),
    ),
    Regime(
        name="heterogeneous_depth",
        purpose="a tenfold depth spread; only the Poisson has an exposure term, "
        "so only it can return a depth-independent abundance",
        mean_counts=5.0,
        depth_factors=(0.2, 0.6, 1.0, 2.0),
    ),
)


def make_design(n_models: int, bins_per_model: int, overlap: int) -> np.ndarray:
    """A staircase of overlapping models, as isoforms sharing a locus produce."""
    stride = bins_per_model - overlap
    n_bins = stride * (n_models - 1) + bins_per_model
    matrix = np.zeros((n_bins, n_models))
    for k in range(n_models):
        matrix[k * stride : k * stride + bins_per_model, k] = 1.0
    return matrix


def contiguous_folds(n_bins: int, n_folds: int) -> list[np.ndarray]:
    """Boolean held-out masks over contiguous blocks of bins."""
    edges = np.linspace(0, n_bins, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        mask = np.zeros(n_bins, dtype=bool)
        mask[edges[i] : edges[i + 1]] = True
        folds.append(mask)
    return folds


def poisson_deviance_on(
    models: np.ndarray,
    abundance: np.ndarray,
    observed: np.ndarray,
    where: np.ndarray,
    exposure_scale: float,
) -> float:
    """Poisson deviance restricted to `where`, per bin.

    Used to score every method, including the legacy ones. That is the point:
    the held-out data are counts however they were fitted, so the count
    likelihood is the honest yardstick for predicting them.
    """
    mu = np.maximum(exposure_scale * (models[where] @ abundance), 1e-12)
    y = observed[where]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)) / max(where.sum(), 1))


def fit_one(
    method: str,
    models: np.ndarray,
    observed: np.ndarray,
    train: np.ndarray,
    exposure_scale: float,
):
    """Fit `method` using only the `train` bins.

    Held-out bins are zeroed in both the design and the counts, exactly as
    masking does, so they cannot influence the estimate.
    """
    fit_models = models.copy()
    fit_models[~train, :] = 0.0
    fit_observed = observed.copy()
    fit_observed[~train] = 0.0

    if method == "legacy_log":
        return legacy.fit_locus(fit_models, fit_observed, transform="log")
    if method == "legacy_identity":
        return legacy.fit_locus(fit_models, fit_observed, transform="identity")
    if method == "poisson":
        return poisson.fit_locus(
            fit_models, fit_observed, exposure=Exposure(sample_scale=exposure_scale)
        )
    if method == "poisson_ridge":
        return poisson.fit_locus(
            fit_models,
            fit_observed,
            exposure=Exposure(sample_scale=exposure_scale),
            penalty=ridge(FIXED_RIDGE),
        )
    if method == "poisson_ridge_scaled":
        # A fixed lambda cannot transfer across count scales: the ridge term
        # grows as a^2 while the Poisson log-likelihood grows about as a, so
        # the same lambda that is negligible at a ~ 1 dominates at a ~ 200.
        # Scaling by the moment estimate of the abundance keeps the penalty a
        # constant fraction of the likelihood.
        support = fit_models.sum(axis=0)
        typical = float(fit_observed.sum() / max(exposure_scale * support.sum(), 1e-9))
        return poisson.fit_locus(
            fit_models,
            fit_observed,
            exposure=Exposure(sample_scale=exposure_scale),
            penalty=ridge(FIXED_RIDGE / max(typical, 1e-6)),
        )
    raise ValueError(f"unknown method {method!r}")


#: Ridge weight for the unscaled variant, deliberately left fixed so the
#: scale-transfer failure it causes is visible in the report.
FIXED_RIDGE = 0.01


METHODS = (
    "legacy_log",
    "legacy_identity",
    "poisson",
    "poisson_ridge",
    "poisson_ridge_scaled",
)

#: DENR's default, and therefore the baseline a user is actually switching from.
BASELINE = "legacy_log"

#: The strongest legacy alternative, and the harder comparison to win.
STRONGEST_LEGACY = "legacy_identity"


def run_regime(regime: Regime, rng: np.random.Generator, n_folds: int = 4) -> dict:
    """Score every method on one regime, returning aggregate metrics."""
    models = make_design(n_models=3, bins_per_model=24, overlap=8)
    n_bins, n_models = models.shape
    folds = contiguous_folds(n_bins, n_folds)

    records: dict[str, dict[str, list[float]]] = {
        method: {"deviance": [], "rmse": [], "spearman": [], "estimates": []} for method in METHODS
    }
    failures = dict.fromkeys(METHODS, 0)
    # Paired per-fold values, so differences are computed on identical data.
    paired: list[dict[str, tuple[float, float]]] = []

    for _ in range(regime.n_replicates):
        # A fresh truth each replicate, so results are not an artefact of one
        # abundance vector. Spread over an order of magnitude.
        truth = regime.mean_counts * rng.uniform(0.3, 3.0, n_models)

        for depth in regime.depth_factors:
            expected = depth * (models @ truth)
            observed = rng.poisson(expected).astype(np.float64)

            for held_out in folds:
                train = ~held_out
                fold_values: dict[str, tuple[float, float]] = {}
                for method in METHODS:
                    result = fit_one(method, models, observed, train, depth)
                    if result.status not in (FitStatus.OK, FitStatus.ALL_ZERO):
                        failures[method] += 1
                        continue
                    abundance = result.abundance
                    # Legacy has no exposure term, so its estimate absorbs the
                    # depth; divide it out before comparing to the truth.
                    scaled = abundance / depth if method.startswith("legacy") else abundance

                    deviance = poisson_deviance_on(
                        models,
                        abundance,
                        observed,
                        held_out,
                        depth if method.startswith("poisson") else 1.0,
                    )
                    nrmse = float(np.sqrt(np.mean((scaled - truth) ** 2)) / truth.mean())
                    records[method]["deviance"].append(deviance)
                    records[method]["rmse"].append(nrmse)
                    records[method]["spearman"].append(_spearman(scaled, truth))
                    records[method]["estimates"].append(float(scaled[0] / truth[0]))
                    fold_values[method] = (deviance, nrmse)
                paired.append(fold_values)

    summary = {}
    for method in METHODS:
        entry = records[method]
        summary[method] = {
            "held_out_deviance": _median(entry["deviance"]),
            "abundance_nrmse": _median(entry["rmse"]),
            "spearman": _median(entry["spearman"]),
            # Spread of the estimate for one model across replicates and
            # depths, relative to truth: the depth-invariance check.
            "estimate_cv": (
                float(np.std(entry["estimates"])) if entry["estimates"] else float("nan")
            ),
            "n_failures": failures[method],
            "n_fits": len(entry["deviance"]),
        }
        for reference in (BASELINE, STRONGEST_LEGACY):
            if method == reference:
                continue
            summary[method][f"vs_{reference}"] = _paired(paired, method, reference)
    return summary


def _paired(
    paired: list[dict[str, tuple[float, float]]], method: str, reference: str
) -> dict[str, float]:
    """Paired comparison of `method` against `reference` over identical folds.

    Reports the fraction of folds `method` wins on each metric, plus the median
    relative difference. A win fraction near 0.5 means the two are
    indistinguishable however the medians happen to fall.
    """
    deviance_wins, nrmse_wins, deviance_rel, nrmse_rel = [], [], [], []
    for fold in paired:
        if method not in fold or reference not in fold:
            continue
        (dm, nm), (dr, nr) = fold[method], fold[reference]
        if np.isfinite(dm) and np.isfinite(dr):
            deviance_wins.append(dm < dr)
            if dr != 0:
                deviance_rel.append((dm - dr) / abs(dr))
        if np.isfinite(nm) and np.isfinite(nr):
            nrmse_wins.append(nm < nr)
            if nr != 0:
                nrmse_rel.append((nm - nr) / abs(nr))
    return {
        "deviance_win_fraction": float(np.mean(deviance_wins)) if deviance_wins else float("nan"),
        "deviance_median_relative": _median(deviance_rel),
        "nrmse_win_fraction": float(np.mean(nrmse_wins)) if nrmse_wins else float("nan"),
        "nrmse_median_relative": _median(nrmse_rel),
        "n_paired": len(deviance_wins),
    }


def _fmt(value: float | None) -> str:
    """Win fractions as percentages, with a dash for the self-comparison."""
    return "-" if value is None or not np.isfinite(value) else f"{100 * value:.0f}%"


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks for ties.

    Ties are common here -- zero estimates form a large block, and nascentRNASim
    draws abundances from a discrete kernel grid -- and the old
    ``argsort(argsort())`` ranking broke them by row order, which polars does not
    fix after a join or group_by, so the statistic varied between identical runs.
    """
    if a.size < 2:
        return float("nan")
    ra = rankdata(a)
    rb = rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denominator) if denominator > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = {}
    for regime in REGIMES:
        rng = np.random.default_rng(args.seed + abs(hash(regime.name)) % 10_000)
        result = run_regime(regime, rng)
        report[regime.name] = {"purpose": regime.purpose, "methods": result}

        print(f"\n=== {regime.name} ===  {regime.purpose}")
        print(
            f"{'method':>22} {'dev':>8} {'nRMSE':>8} {'spread':>7} "
            f"{'win vs log':>11} {'win vs ident':>13}"
        )
        for method in METHODS:
            row = result[method]
            against_log = row.get(f"vs_{BASELINE}", {})
            against_identity = row.get(f"vs_{STRONGEST_LEGACY}", {})
            print(
                f"{method:>22} {row['held_out_deviance']:8.4f} "
                f"{row['abundance_nrmse']:8.4f} {row['estimate_cv']:7.3f} "
                f"{_fmt(against_log.get('deviance_win_fraction')):>11} "
                f"{_fmt(against_identity.get('deviance_win_fraction')):>13}"
            )

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
