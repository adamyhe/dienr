"""Does shape modeling improve accuracy, and does it make Poisson beat MSE?

Both questions answered against known truth on the DENR authors' own simulator
(`nascentRNASim`), which is what the paper used for every accuracy claim it
makes -- section 3.2 states plainly that they lacked a real-data gold standard.
Generate the input with ``scripts/simulate_nascentrnasim.R``.

The simulator resamples read-counts from 62 manually curated PRO-seq archetypes
(real signal, Dukler et al. 2017) onto simulated loci whose structure is drawn
from Ensembl GRCh37, with abundances from a GTEx skeletal-muscle kernel. So the
shape is real and per-isoform truth is known, which is exactly what neither a
hand-rolled simulation nor real data can offer at once.

Design of the comparison:

- **The profile is fitted on the same dataset it is evaluated on, and that is
  the realistic setting**, not a leak: a user fits the profile from the sample
  they are quantifying. Training genes are selected by geometry and expression,
  so most evaluation loci contribute nothing to the profile; the ``--holdout``
  flag additionally restricts profile training to a random half of the contigs
  and evaluates on the rest, which bounds any optimism.
- **Every arm sees identical counts.** Counts are extracted once per summary
  type and shared, so a difference between arms cannot come from the data.
- **Truth is per transcript; estimates are per model.** Equivalence classes are
  collapsed before fitting, so a model's truth is the *sum* of the abundances of
  the transcripts mapped to it -- that is the only quantity the data can
  identify, and scoring against a single member instead would penalize every
  method for the annotation's ambiguity rather than for its own error.
- **Gene level and isoform level are both reported**, as the paper does. Gene
  level sums models within a locus; isoform level scores each identifiable
  model. Dominant-isoform accuracy is reported for loci with two or more
  models, which is the claim the method exists to support.
- **Metrics are scale-free where they can be.** Spearman and dominant-isoform
  accuracy need no calibration. For error we report the median absolute log2
  ratio of estimate to truth over expressed models, which does not let a few
  high-abundance loci dominate the way RMSE does.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

import dienr
from dienr.shape import ShapeConfig, applied_dynamic_range, apply_shape_profile, fit_shape_profile

#: Below this true abundance a transcript is treated as unexpressed. The
#: simulator draws a point mass at zero, and a log-ratio error is undefined
#: there, so the expressed set is what error metrics are computed on.
EXPRESSED = 1e-6

#: The masking the paper settled on after testing all 72 feature combinations:
#: "masking of one bin around the TSS and four bins around the PAS". Notably
#: this is a *one*-bin TSS mask, which is what leaves a shape profile anything
#: to say -- see docs/shape-profiles.md on the two features competing.
PAPER_MASK = {"mask_start_bins": (1, 1), "mask_end_bins": (4, 4)}


@dataclass(frozen=True)
class Arm:
    """One configuration to score.

    Attributes:
        name: Identifier used in the report.
        shape: Whether to reweight the design by a fitted shape profile.
        likelihood: Passed to `dienr.fit`.
        transform: Legacy objective transform; ignored for count likelihoods.
    """

    name: str
    shape: bool
    likelihood: str
    transform: str = "log"

    @property
    def summary(self) -> str:
        """Count likelihoods need summed counts; legacy wants the mean."""
        return "sum" if self.likelihood in ("poisson", "quasi_poisson") else "mean"


ARMS = (
    # The reference's own default scheme, and therefore the baseline.
    Arm("binary/legacy_log", shape=False, likelihood="legacy", transform="log"),
    Arm("binary/legacy_identity", shape=False, likelihood="legacy", transform="identity"),
    Arm("binary/poisson", shape=False, likelihood="poisson"),
    Arm("shape/legacy_log", shape=True, likelihood="legacy", transform="log"),
    Arm("shape/legacy_identity", shape=True, likelihood="legacy", transform="identity"),
    Arm("shape/poisson", shape=True, likelihood="poisson"),
)


def read_truth(bed: Path) -> pl.DataFrame:
    """Read the simulator's BED, whose score column is the true abundance."""
    rows = []
    with gzip.open(bed, "rt") as handle:
        for line in handle:
            contig, start, end, _dot, score, strand, gene_id, transcript_id = line.rstrip(
                "\n"
            ).split("\t")
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "gene_id": gene_id,
                    "contig": contig,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "truth": float(score),
                }
            )
    return pl.DataFrame(rows)


def model_truth(design: dienr.Design, truth: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-transcript truth onto identifiable models.

    A model's truth is the sum over its equivalence class. Two transcripts that
    a 250 bp design cannot distinguish contribute one identifiable quantity,
    and their total is it; blaming a method for not splitting them would be
    scoring the annotation, not the estimator.
    """
    return (
        design.transcript_map.join(
            truth.select("transcript_id", "truth"), on="transcript_id", how="left"
        )
        .group_by("locus_id", "model_index")
        .agg(
            truth=pl.col("truth").sum(),
            equivalence_class_size=pl.col("equivalence_class_size").first(),
        )
    )


def score(estimates: pl.DataFrame, truth_by_model: pl.DataFrame) -> dict[str, float]:
    """Isoform-level, gene-level and dominant-isoform metrics."""
    joined = estimates.join(truth_by_model, on=["locus_id", "model_index"], how="inner")
    ok = joined.filter((pl.col("solver_status") == "ok") & pl.col("abundance_raw").is_not_null())
    if not ok.height:
        return {}

    # Isoform level, on expressed models only.
    expressed = ok.filter(pl.col("truth") > EXPRESSED)
    est = expressed["abundance_raw"].to_numpy()
    tru = expressed["truth"].to_numpy()
    positive = est > 0
    # One global calibration constant, removed before any error is computed.
    # Abundance units are arbitrary and differ between objectives -- the
    # simulator's truth is on a GTEx-derived scale, legacy abundance is mean
    # per-base coverage, and Poisson abundance carries an exposure term -- so
    # an uncalibrated log ratio measures the unit mismatch, not the estimate.
    # Removing exactly one degree of freedom is what a user does anyway when
    # they convert to TPM, and it is applied identically to every arm.
    ratio = est[positive] / tru[positive]
    calibration = float(np.median(ratio)) if ratio.size else 1.0
    log_error = np.abs(np.log2(ratio / calibration))

    # Gene level: total abundance per locus.
    gene = ok.group_by("locus_id").agg(est=pl.col("abundance_raw").sum(), tru=pl.col("truth").sum())
    gene = gene.filter((pl.col("tru") > EXPRESSED) & (pl.col("est") > 0))
    gene_ratio = gene["est"].to_numpy() / gene["tru"].to_numpy()
    gene_calibration = float(np.median(gene_ratio)) if gene_ratio.size else 1.0
    gene_log_error = np.abs(np.log2(gene_ratio / gene_calibration))

    # Dominant isoform, on loci with a choice to get wrong and a real winner.
    multi = (
        ok.filter(pl.col("truth") > EXPRESSED)
        .group_by("locus_id")
        .agg(
            n=pl.len(),
            # Correct if the top estimate is *a* truly dominant model; truth
            # ties are common and were broken by row order (see `_spearman`).
            hit=pl.col("truth")
            .sort_by(["abundance_raw", "model_index"], descending=[True, False])
            .first()
            == pl.col("truth").max(),
        )
    )
    multi = multi.filter(pl.col("n") >= 2)
    dominant = float(multi["hit"].mean()) if multi.height else float("nan")

    return {
        "n_models_scored": float(expressed.height),
        "calibration": calibration,
        "isoform_median_abs_log2": float(np.median(log_error)) if log_error.size else float("nan"),
        "isoform_spearman": _spearman(est, tru),
        "gene_median_abs_log2": (
            float(np.median(gene_log_error)) if gene_log_error.size else float("nan")
        ),
        "gene_spearman": _spearman(gene["est"].to_numpy(), gene["tru"].to_numpy()),
        "dominant_isoform_accuracy": dominant,
        "n_multi_model_loci": float(multi.height),
        "zero_estimate_fraction": float(np.mean(~positive)) if positive.size else float("nan"),
    }


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
    parser.add_argument("--sim", type=Path, default=Path("build/sim/sim1500"))
    parser.add_argument("--bin-size", type=int, default=250)
    parser.add_argument("--span", type=float, default=ShapeConfig().span)
    parser.add_argument(
        "--reference-span",
        action="store_true",
        help="use the reference's span of 0.4 instead of the dienr default",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="skip the paper's 1-bin TSS / 4-bin PAS masking",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="fit the profile on half the contigs and score on the other half",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    bed = args.sim.with_suffix(".bed.gz")
    plus = dienr.SignalFile(path=f"{args.sim}_plus.bw", strand="+")
    minus = dienr.SignalFile(path=f"{args.sim}_minus.bw", strand="-")
    signal = (plus, minus)

    truth = read_truth(bed)
    annotation = dienr.Annotation.from_frame(
        truth.select("transcript_id", "gene_id", "contig", "start", "end", "strand"),
        source=str(bed),
    )
    print(
        f"simulation: {len(annotation)} transcripts, {truth['truth'].gt(EXPRESSED).sum()} expressed"
    )

    profile_annotation = annotation
    evaluation = annotation
    if args.holdout:
        contigs = sorted(truth["contig"].unique().to_list())
        rng = np.random.default_rng(0)
        train_contigs = set(rng.permutation(contigs)[: len(contigs) // 2])
        profile_annotation = dienr.Annotation.from_frame(
            truth.filter(pl.col("contig").is_in(list(train_contigs))).select(
                "transcript_id", "gene_id", "contig", "start", "end", "strand"
            ),
            source="profile-train",
        )
        evaluation = dienr.Annotation.from_frame(
            truth.filter(~pl.col("contig").is_in(list(train_contigs))).select(
                "transcript_id", "gene_id", "contig", "start", "end", "strand"
            ),
            source="evaluate",
        )
        print(
            f"holdout: profile from {len(profile_annotation)} transcripts on "
            f"{len(train_contigs)} contigs, scored on {len(evaluation)}"
        )

    span = 0.4 if args.reference_span else args.span
    config = ShapeConfig(bin_size=args.bin_size, span=span)
    profile = fit_shape_profile(profile_annotation, signal, config=config)
    print(
        f"shape profile: {profile.n_genes} training genes, "
        f"dynamic range {profile.dynamic_range:.2f}x, span {span}"
    )
    grid = np.array([0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print("  position " + " ".join(f"{p:5.2f}" for p in grid))
    print("  density  " + " ".join(f"{v:5.2f}" for v in profile.evaluate(grid)))

    mask = {} if args.no_mask else PAPER_MASK
    base = dienr.build_design(evaluation, bin_size=args.bin_size, **mask)
    shaped = apply_shape_profile(base, profile)
    print(
        f"design: {base.n_loci} loci, {base.model_table.height} models, "
        f"{100 * float(base.bins['masked'].mean()):.1f}% bins masked; "
        f"applied shape range {applied_dynamic_range(shaped):.2f}x"
    )

    designs = {False: base, True: shaped}
    # Counts are extracted per design even though reshaping leaves the bins
    # untouched: `SampleCounts` records the design digest it was built against
    # and `collect_results` refuses a mismatch, which is the check that stops a
    # reshaped design being scored with counts from an unshaped one. Paying for
    # the extra read is cheaper than weakening that guard.
    counts = {
        (shape, summary): dienr.count_sample(design, "sim", signal, summary=summary)
        for shape, design in designs.items()
        for summary in ("mean", "sum")
    }

    truth_by_model = {key: model_truth(design, truth) for key, design in designs.items()}

    report: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        design = designs[arm.shape]
        kwargs = {"likelihood": arm.likelihood}
        if arm.likelihood == "legacy":
            kwargs["transform"] = arm.transform
        result = dienr.fit(design, counts[(arm.shape, arm.summary)], **kwargs)
        report[arm.name] = score(result.model_abundance, truth_by_model[arm.shape])
        print(f"  fitted {arm.name}")

    header = (
        f"{'arm':>24} {'iso |log2|':>11} {'iso rho':>8} "
        f"{'gene |log2|':>12} {'gene rho':>9} {'dominant':>9} {'zeros':>7}"
    )
    print(f"\n=== accuracy against known truth ===\n{header}")
    for arm in ARMS:
        row = report[arm.name]
        if not row:
            print(f"{arm.name:>24}  no successful fits")
            continue
        print(
            f"{arm.name:>24} {row['isoform_median_abs_log2']:11.4f} "
            f"{row['isoform_spearman']:8.4f} {row['gene_median_abs_log2']:12.4f} "
            f"{row['gene_spearman']:9.4f} {row['dominant_isoform_accuracy']:9.3f} "
            f"{row['zero_estimate_fraction']:7.3f}"
        )

    print("\n=== the two questions ===")
    _answer(
        "does shape modeling improve things?",
        report,
        "binary/legacy_identity",
        "shape/legacy_identity",
    )
    _answer(
        "  (same, against the reference's default objective)",
        report,
        "binary/legacy_log",
        "shape/legacy_log",
    )
    _answer(
        "does Poisson beat MSE on a BINARY design?",
        report,
        "binary/legacy_identity",
        "binary/poisson",
    )
    _answer(
        "does Poisson beat MSE on a SHAPED design?",
        report,
        "shape/legacy_identity",
        "shape/poisson",
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "arms": report,
                    "profile": profile.describe(),
                    "applied_dynamic_range": applied_dynamic_range(shaped),
                    "config": {
                        "bin_size": args.bin_size,
                        "span": span,
                        "mask": {k: list(v) for k, v in mask.items()},
                        "holdout": args.holdout,
                        "sim": str(args.sim),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


def _answer(question: str, report: dict, baseline: str, candidate: str) -> None:
    """Print the isoform-level and dominant-isoform change for one contrast."""
    left, right = report.get(baseline), report.get(candidate)
    if not left or not right:
        print(f"{question:<52} insufficient data")
        return
    error_change = (right["isoform_median_abs_log2"] - left["isoform_median_abs_log2"]) / left[
        "isoform_median_abs_log2"
    ]
    dominant_change = right["dominant_isoform_accuracy"] - left["dominant_isoform_accuracy"]
    verdict = "better" if error_change < 0 else "worse"
    print(
        f"{question:<52} {verdict}: isoform error {100 * error_change:+.1f}%, "
        f"dominant-isoform {100 * dominant_change:+.1f} pts"
    )


if __name__ == "__main__":
    raise SystemExit(main())
