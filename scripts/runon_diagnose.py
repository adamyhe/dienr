"""Why does the fitted run-on tail fail the per-locus error criterion?

`runon_trial.py` (plan 18.10) found the fitted shared tail wins 82% of
discordant dominant-isoform calls against binary/poisson but lowers per-locus
error at only ~52% of loci (exact tail: ~65%). This splits that error-win rate
by locus properties to say where the loss is, and whether it has a fix.

Hypotheses, each with the split that tests it:

1. **Metric artifact.** A locus's error is the median over expressed models with
   a *positive* estimate. The tail rescues models from zero (18.7% -> 13.6%), so
   the two arms can be scored on different model sets. Split: error over
   models positive in both arms only.
2. **Tail lands on nothing.** If no model's tail overlaps another model's
   support, the tail cannot resolve confusion; it only moves the fit. Split:
   loci with vs without tail-on-body overlap.
3. **Gene-specific termination.** The shared tail's mass differs from each
   isoform's true tail mass. Split: by locus misfit, the expression-weighted
   |log2(exact tail mass / fitted tail mass)| over the same in-window bins.
4. **Length.** A fixed 6 kb tail is a larger fraction of a short isoform.
   Split: by shortest expressed isoform.

Diagnostic only: no decision rule, nothing here changes 18.10's verdict.

    uv run python scripts/runon_diagnose.py build/sim/sim1500_s{5,17,29,41,53}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_shape_ceiling import (  # noqa: E402
    BIN_SIZE,
    EXPRESSED,
    PAPER_MASK,
    _calibration,
    load_archetypes,
    read_truth,
    transcript_columns,
)
from runon_trial import _tail_bins, fit_tail, runon_designs  # noqa: E402

import dienr  # noqa: E402

ARMS = (
    ("binary", "binary"),
    ("runon", "runon"),
    ("runon_oracle", "runon_oracle"),
)


def model_features(
    base: dienr.Design, columns: dict[str, np.ndarray], tail: np.ndarray
) -> pl.DataFrame:
    """Per model: length, tail exposure, and how wrong the shared tail is for it."""
    members = {
        (r["locus_id"], r["model_index"]): r["transcript_id"]
        for r in base.transcript_map.group_by("locus_id", "model_index")
        .agg(pl.col("transcript_id"))
        .iter_rows(named=True)
    }
    strands = base.loci["strand"].to_list()
    rows = []
    for locus_id, matrix in enumerate(base.models):
        if matrix.shape[1] < 2:
            continue
        unmasked = ~base.mask(locus_id)
        plus = strands[locus_id] == "+"
        supports = (matrix != 0) & unmasked[:, None]
        for k in range(matrix.shape[1]):
            first, last = (int(x) for x in base.model_spans[locus_id][k])
            bins = _tail_bins(matrix.shape[0], first, last, plus)
            fitted = tail[: bins.size] * unmasked[bins]
            others = np.delete(supports, k, axis=1).any(axis=1)
            exact = []
            for tx in members[(locus_id, k)]:
                full = columns[tx]
                level = full[supports[:, k]].mean() if supports[:, k].any() else 0.0
                exact.append(full[bins] / level if level > 0 else np.zeros(bins.size))
            exact_mass = (
                float((np.mean(exact, axis=0) * unmasked[bins]).sum()) if bins.size else 0.0
            )
            rows.append(
                {
                    "locus_id": locus_id,
                    "model_index": k,
                    "length_kb": (last - first + 1) * BIN_SIZE / 1000,
                    "tail_bins_in_window": int(bins.size),
                    "fitted_tail_mass": float(fitted.sum()),
                    "exact_tail_mass": exact_mass,
                    "tail_on_other_body": float(fitted[others[bins]].sum()),
                }
            )
    return pl.DataFrame(rows)


def run_seed(sim: Path, archetypes: Path) -> pl.DataFrame:
    truth = read_truth(sim.with_suffix(".bed.gz"))
    annotation = dienr.Annotation.from_frame(
        truth.select("transcript_id", "gene_id", "contig", "start", "end", "strand"),
        source=str(sim),
    )
    signal = (
        dienr.SignalFile(path=f"{sim}_plus.bw", strand="+"),
        dienr.SignalFile(path=f"{sim}_minus.bw", strand="-"),
    )
    base = dienr.build_design(annotation, bin_size=BIN_SIZE, **PAPER_MASK)
    counts = dienr.count_sample(base, "s", signal, summary="sum")
    vectors, flank = load_archetypes(archetypes)
    columns = transcript_columns(base, truth, vectors, flank)
    tail, _ = fit_tail(base, counts, truth, signal, flank)
    designs = {"binary": base, **runon_designs(base, columns, tail)}

    table = (
        base.transcript_map.join(truth.select("transcript_id", "truth"), on="transcript_id")
        .group_by("locus_id", "model_index")
        .agg(truth=pl.col("truth").sum())
        .join(model_features(base, columns, tail), on=["locus_id", "model_index"])
    )
    for name, variant in ARMS:
        design = designs[variant]
        fit = dienr.fit(
            design, dienr.count_sample(design, "s", signal, summary="sum"), likelihood="poisson"
        )
        est = fit.model_abundance.select(
            "locus_id",
            "model_index",
            pl.col("abundance_raw").alias(name),
            pl.col("solver_status").alias(f"{name}_status"),
        )
        table = table.join(est, on=["locus_id", "model_index"])
    # One calibration per arm over all multi-model loci, as the harness does.
    for name, _ in ARMS:
        frame = table.select(
            "truth",
            pl.col(name).alias("abundance_raw"),
            pl.col(f"{name}_status").alias("solver_status"),
        )
        table = table.with_columns((pl.col(name) / _calibration(frame)).alias(f"{name}_cal"))
    return table.with_columns(seed=pl.lit(sim.name))


def locus_table(models: pl.DataFrame) -> pl.DataFrame:
    """One row per scorable locus with both arms' errors and the split features."""
    rows = []
    status_cols = [f"{n}_status" for n, _ in ARMS]
    for (seed, locus_id), frame in models.group_by("seed", "locus_id"):
        if any((frame[c] != "ok").any() for c in status_cols):
            continue
        truth = frame["truth"].to_numpy()
        if truth.max() <= EXPRESSED:
            continue
        expressed = truth > EXPRESSED

        def err(
            arm: str, keep: np.ndarray, frame: pl.DataFrame = frame, truth: np.ndarray = truth
        ) -> float:
            est = frame[f"{arm}_cal"].to_numpy()
            k = keep & (est > 0)
            return float(np.median(np.abs(np.log2(est[k] / truth[k])))) if k.any() else np.nan

        pos_b = frame["binary"].to_numpy() > 0
        pos_r = frame["runon"].to_numpy() > 0
        pos_o = frame["runon_oracle"].to_numpy() > 0
        fitted = frame["fitted_tail_mass"].to_numpy()
        exact = frame["exact_tail_mass"].to_numpy()
        w = truth * expressed * (fitted > 0)
        logratio = np.log2(np.maximum(exact, 1e-3) / np.maximum(fitted, 1e-3))
        misfit = float(np.sum(w * np.abs(logratio)) / w.sum()) if w.sum() > 0 else np.nan
        signed = float(np.sum(w * logratio) / w.sum()) if w.sum() > 0 else np.nan
        rows.append(
            {
                "seed": seed,
                "locus_id": locus_id,
                "n_models": frame.height,
                "err_binary": err("binary", expressed),
                "err_runon": err("runon", expressed),
                "err_oracle_tail": err("runon_oracle", expressed),
                "err_binary_common": err("binary", expressed & pos_b & pos_r),
                "err_runon_common": err("runon", expressed & pos_b & pos_r),
                "err_binary_common_o": err("binary", expressed & pos_b & pos_o),
                "err_oracle_common_o": err("runon_oracle", expressed & pos_b & pos_o),
                "rescued": int((expressed & ~pos_b & pos_r).sum()),
                "tail_overlap": float(frame["tail_on_other_body"].sum()) > 0,
                "misfit": misfit,
                "signed_misfit": signed,
                "min_length_kb": float(
                    frame.filter(pl.col("truth") > EXPRESSED)["length_kb"].min()
                ),
            }
        )
    return pl.DataFrame(rows)


def win(frame: pl.DataFrame, a: str, b: str) -> tuple[float, int]:
    sub = frame.filter(pl.col(a).is_not_nan() & pl.col(b).is_not_nan())
    if sub.height == 0:
        return float("nan"), 0
    return float((sub[a] < sub[b]).mean()), sub.height


def report(loci: pl.DataFrame) -> None:
    def line(label: str, frame: pl.DataFrame) -> None:
        f, n = win(frame, "err_runon", "err_binary")
        fc, nc = win(frame, "err_runon_common", "err_binary_common")
        o, _ = win(frame, "err_oracle_tail", "err_binary")
        oc, _ = win(frame, "err_oracle_common_o", "err_binary_common_o")
        print(
            f"{label:>38}  n={n:4d}  fitted {f:.3f}  common {fc:.3f} (n={nc:4d})"
            f"   exact {o:.3f}  common {oc:.3f}"
        )

    print("error-win fraction vs binary/poisson (common = models positive in both arms)\n")
    print("== per seed ==")
    for (seed,), frame in sorted(loci.group_by("seed"), key=lambda x: x[0]):
        line(seed, frame)
    line("pooled", loci)

    print("\n== H1: does rescuing zeros change the scored set? ==")
    line("no model rescued from zero", loci.filter(pl.col("rescued") == 0))
    line(">= 1 model rescued from zero", loci.filter(pl.col("rescued") > 0))

    print("\n== H2: does a tail land on another model's body? ==")
    line("tail overlaps another body", loci.filter(pl.col("tail_overlap")))
    line("no overlap (tail lands on nothing)", loci.filter(~pl.col("tail_overlap")))

    print("\n== H3: shared-tail misfit, |log2 exact/fitted| (expr-weighted) ==")
    q = loci["misfit"].drop_nans().quantile(0.5)
    print(
        f"   median misfit {q:.2f};  signed (exact>fitted is +): "
        f"median {loci['signed_misfit'].drop_nans().median():+.2f}"
    )
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)):
        line(
            f"misfit in [{lo}, {hi})",
            loci.filter((pl.col("misfit") >= lo) & (pl.col("misfit") < hi)),
        )
    for lo, hi, lab in (
        (-99, -1, "fitted >2x too big"),
        (-1, 1, "within 2x"),
        (1, 99, "fitted >2x too small"),
    ):
        line(lab, loci.filter((pl.col("signed_misfit") >= lo) & (pl.col("signed_misfit") < hi)))

    print("\n== H4: shortest expressed isoform ==")
    for lo, hi in ((0, 10), (10, 30), (30, 80), (80, 10_000)):
        line(
            f"min length [{lo}, {hi}) kb",
            loci.filter((pl.col("min_length_kb") >= lo) & (pl.col("min_length_kb") < hi)),
        )

    print("\n== median |log2| error by arm (pooled loci) ==")
    for c in ("err_binary", "err_runon", "err_oracle_tail"):
        print(f"   {c:>16} {loci[c].drop_nans().median():.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sims", nargs="+", type=Path)
    parser.add_argument("--archetypes", type=Path, default=Path("build/oracle"))
    parser.add_argument("--output", type=Path, default=Path("build/runon/diagnose_models.parquet"))
    args = parser.parse_args(argv)
    models = pl.concat([run_seed(s, args.archetypes) for s in args.sims])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    models.write_parquet(args.output)
    report(locus_table(models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
