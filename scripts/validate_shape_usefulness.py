"""Is a shape profile worth applying at all, and where?

The narrow question left open by `benchmark_shape_accuracy.py`: that script
showed shape modeling improves isoform ranking and degrades magnitude, which
does not by itself say whether to turn it on. This one is built to answer that,
and to be capable of returning "no".

Method choices that make the answer mean something:

- **Multi-model loci only, by default.** A shape profile is pure loss where
  there is nothing to apportion -- 4.5x worse magnitude under least squares at
  single-model loci, with no ranking gain. Including those loci would let a
  known, avoidable penalty swamp whatever real effect exists, and no one who
  read the stratified result would deploy it that way.
- **Calibration within the stratum being scored.** Abundance units differ
  between objectives and between designs; one global constant per arm per
  stratum removes exactly that and nothing else.
- **Paired per locus.** Every arm sees the same simulated counts at the same
  locus, so the statistic is the per-locus difference and the fraction of loci
  each arm wins. Medians of independently aggregated columns would let noise
  look decisive, which is a live risk here because the effects are small.
- **Several seeds, reported individually as well as pooled.** The
  dominant-isoform metric moved +9.8 and -2.4 points on the first two seeds.
  Anything that swings that far needs its spread shown, not averaged away.
- **The comparison that matters is against the best available alternative**,
  which is a binary design fitted with the same likelihood -- not against
  DENR's default, which loses for unrelated reasons.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

import dienr
from dienr.shape import ShapeConfig, apply_shape_profile, fit_shape_profile

EXPRESSED = 1e-6
PAPER_MASK = {"mask_start_bins": (1, 1), "mask_end_bins": (4, 4)}

#: (name, shape, likelihood, transform, summary)
ARMS = (
    ("binary/legacy_log", False, "legacy", "log", "mean"),
    ("binary/poisson", False, "poisson", None, "sum"),
    ("shape/legacy_log", True, "legacy", "log", "mean"),
    ("shape/poisson", True, "poisson", None, "sum"),
)


def read_truth(bed: Path) -> pl.DataFrame:
    rows = []
    with gzip.open(bed, "rt") as handle:
        for line in handle:
            contig, start, end, _dot, score, strand, gene, transcript = line.rstrip("\n").split(
                "\t"
            )
            rows.append(
                {
                    "transcript_id": transcript,
                    "gene_id": gene,
                    "contig": contig,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "truth": float(score),
                }
            )
    return pl.DataFrame(rows)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denominator) if denominator > 0 else float("nan")


def run_seed(sim: Path, *, min_models: int, span: float) -> dict:
    """Score every arm on one simulation, returning per-arm and paired metrics."""
    truth = read_truth(sim.with_suffix(".bed.gz"))
    annotation = dienr.Annotation.from_frame(
        truth.select("transcript_id", "gene_id", "contig", "start", "end", "strand"),
        source=str(sim),
    )
    signal = (
        dienr.SignalFile(path=f"{sim}_plus.bw", strand="+"),
        dienr.SignalFile(path=f"{sim}_minus.bw", strand="-"),
    )
    profile = fit_shape_profile(annotation, signal, config=ShapeConfig(span=span))
    base = dienr.build_design(annotation, bin_size=250, **PAPER_MASK)
    shaped = apply_shape_profile(base, profile)
    designs = {False: base, True: shaped}

    truth_by_model = (
        base.transcript_map.join(
            truth.select("transcript_id", "truth"), on="transcript_id", how="left"
        )
        .group_by("locus_id", "model_index")
        .agg(truth=pl.col("truth").sum())
    )
    n_models = base.model_table.group_by("locus_id").agg(n_models=pl.len())

    per_arm: dict[str, dict[str, float]] = {}
    # locus -> arm -> (correct_dominant, locus_abs_log2_error)
    per_locus: dict[int, dict[str, tuple[bool, float]]] = {}

    for name, shape, likelihood, transform, summary in ARMS:
        design = designs[shape]
        counts = dienr.count_sample(design, "s", signal, summary=summary)
        kwargs: dict = {"likelihood": likelihood}
        if transform:
            kwargs["transform"] = transform
        result = dienr.fit(design, counts, **kwargs)

        table = (
            result.model_abundance.join(truth_by_model, on=["locus_id", "model_index"])
            .join(n_models, on="locus_id")
            .filter((pl.col("n_models") >= min_models) & (pl.col("truth") > EXPRESSED))
        )
        succeeded = table.filter(pl.col("solver_status") == "ok")
        estimate = succeeded["abundance_raw"].to_numpy()
        actual = succeeded["truth"].to_numpy()
        positive = estimate > 0
        ratio = estimate[positive] / actual[positive]
        calibration = float(np.median(ratio)) if ratio.size else 1.0

        dominant = (
            succeeded.filter(pl.col("abundance_raw") > 0)
            .group_by("locus_id")
            .agg(
                n=pl.len(),
                estimated=pl.col("model_index").sort_by("abundance_raw", descending=True).first(),
                truthful=pl.col("model_index").sort_by("truth", descending=True).first(),
            )
            .filter(pl.col("n") >= 2)
        )
        hits = (dominant["estimated"] == dominant["truthful"]).to_numpy()

        # Per-locus error, for pairing: median |log2| over the locus's models.
        locus_error = (
            succeeded.filter(pl.col("abundance_raw") > 0)
            .with_columns(
                err=(pl.col("abundance_raw") / pl.col("truth") / calibration).log(2).abs()
            )
            .group_by("locus_id")
            .agg(err=pl.col("err").median())
        )
        errors = dict(
            zip(
                locus_error["locus_id"].to_list(),
                locus_error["err"].to_list(),
                strict=True,
            )
        )
        for locus, hit in zip(dominant["locus_id"].to_list(), hits, strict=True):
            per_locus.setdefault(locus, {})[name] = (bool(hit), errors.get(locus, float("nan")))

        per_arm[name] = {
            "n_models": float(succeeded.height),
            "n_loci_with_choice": float(dominant.height),
            "dominant_accuracy": float(hits.mean()) if hits.size else float("nan"),
            "isoform_spearman": spearman(estimate, actual),
            "isoform_median_abs_log2": (
                float(np.median(np.abs(np.log2(ratio / calibration))))
                if ratio.size
                else float("nan")
            ),
            "zero_fraction": float(np.mean(~positive)) if positive.size else float("nan"),
        }

    per_arm["_paired"] = _pair(per_locus)
    per_arm["_profile"] = {
        "dynamic_range": profile.dynamic_range,
        "n_genes": float(profile.n_genes),
    }
    return per_arm


def _pair(per_locus: dict[int, dict[str, tuple[bool, float]]]) -> dict[str, float]:
    """Paired shape-versus-binary comparison, same likelihood, same locus."""
    out: dict[str, float] = {}
    for likelihood in ("poisson", "legacy_log"):
        binary_name, shape_name = f"binary/{likelihood}", f"shape/{likelihood}"
        gained = lost = 0
        error_wins: list[bool] = []
        for arms in per_locus.values():
            if binary_name not in arms or shape_name not in arms:
                continue
            binary_hit, binary_error = arms[binary_name]
            shape_hit, shape_error = arms[shape_name]
            if shape_hit and not binary_hit:
                gained += 1
            elif binary_hit and not shape_hit:
                lost += 1
            if np.isfinite(binary_error) and np.isfinite(shape_error):
                error_wins.append(shape_error < binary_error)
        decided = gained + lost
        out[f"{likelihood}_dominant_gained"] = float(gained)
        out[f"{likelihood}_dominant_lost"] = float(lost)
        # Of the loci where the two arms disagree about the dominant isoform,
        # the share where shape is the one that got it right. 0.5 is a wash
        # however the aggregate accuracies happen to fall.
        out[f"{likelihood}_dominant_win_share"] = (
            float(gained) / decided if decided else float("nan")
        )
        out[f"{likelihood}_error_win_fraction"] = (
            float(np.mean(error_wins)) if error_wins else float("nan")
        )
        out[f"{likelihood}_n_discordant"] = float(decided)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--min-models",
        type=int,
        default=2,
        help="score only loci with at least this many models (default 2)",
    )
    parser.add_argument("--span", type=float, default=ShapeConfig().span)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = {}
    for sim in args.sims:
        report[sim.name] = run_seed(sim, min_models=args.min_models, span=args.span)
        print(f"scored {sim.name}")

    names = [arm[0] for arm in ARMS]
    print(f"\n=== loci with >= {args.min_models} models, per simulation ===")
    print(f"{'metric':>26} " + " ".join(f"{s:>12}" for s in report))
    for metric in (
        "dominant_accuracy",
        "isoform_spearman",
        "isoform_median_abs_log2",
        "zero_fraction",
    ):
        print(f"  {metric}")
        for name in names:
            values = [report[s][name][metric] for s in report]
            print(
                f"{name:>26} "
                + " ".join(f"{v:12.4f}" for v in values)
                + f"   mean {np.mean(values):.4f}"
            )

    print("\n=== paired, shape vs binary at the same locus and likelihood ===")
    for likelihood in ("poisson", "legacy_log"):
        print(f"  {likelihood}")
        for key, label in (
            ("dominant_win_share", "share of discordant loci shape wins"),
            ("n_discordant", "  (n discordant loci)"),
            ("error_win_fraction", "fraction of loci shape has lower error"),
        ):
            values = [report[s]["_paired"][f"{likelihood}_{key}"] for s in report]
            print(
                f"{label:>42} "
                + " ".join(f"{v:12.4f}" for v in values)
                + f"   mean {np.mean(values):.4f}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
