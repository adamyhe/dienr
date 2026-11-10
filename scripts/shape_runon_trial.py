"""Trial a shape profile combined with the fitted run-on tail (plan 18.13).

18.10's fitted run-on tail is the current candidate (dominant accuracy 0.697 ->
0.833 vs binary/poisson). 18.8/18.11 found a shared shape profile alone close to
a wash (0.744 -> 0.754 there; 0.698 -> 0.715 on 18.9's loci). The two act on
different bins -- the profile inside each model's span, the tail past its PAS
-- so their gains could add. The 18.9 ladder bounds what is available: body
shape is worth ~0.18 of dominant accuracy *if known per gene*, but a shared
profile captured ~0.02 of it. This asks whether that small gain survives on
top of the tail, where the remaining confusion is different.

**Design.** `shape_runon` = `apply_shape_profile(binary, profile)` (dienr
defaults, `span=0.05`), then the same shared tail `r(d)` as 18.10 added past
each model's PAS. The profile is mean-normalized per gene and `r(d)` is in
body-mean units, so the two share a scale. Profile and tail are both fitted
from the data alone; no truth is used.

Arms, scored on the same multi-model loci (tie-aware, order-independent, 18.11;
error wins exclude exact ties, 18.12):

    binary/legacy_log        DENR default
    binary/poisson           baseline
    shape/poisson            profile only (18.8)
    runon/poisson            tail only -- the incumbent candidate (18.10)
    shape_runon/poisson      both
    shape_runon/legacy_log   both, under DENR's objective (descriptive)

**Decision rule, fixed before running.** `shape_runon/poisson` replaces
`runon/poisson` as the candidate only if, paired against `runon/poisson`:

1. discordant-dominant win share >= 0.60 pooled and > 0.50 in every seed, AND
2. per-locus error wins > 0.50 in every seed (exact ties excluded).

This is the 18.10 bar, applied to the incumbent rather than to binary. Also
reported, not part of the rule: the same against binary/poisson and DENR's
default, and single-model-locus error -- 18.6 found a profile is pure loss
there (0.078 -> 0.185), so a passing combination would be applied at
multi-model loci only, which this harness cannot score jointly (two abundance
scales under one calibration; 18.6).

    uv run python scripts/shape_runon_trial.py --sim build/sim/sim1500_s5 \\
        --output build/shape_runon/sim1500_s5.json
    uv run python scripts/shape_runon_trial.py --aggregate build/shape_runon/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_shape_ceiling import (  # noqa: E402
    BIN_SIZE,
    PAPER_MASK,
    _arm_metrics,
    _locus_calls,
    _per_base_mean,
    load_archetypes,
    read_truth,
    transcript_columns,
)
from runon_trial import fit_tail, runon_designs  # noqa: E402

import dienr  # noqa: E402
from dienr.shape import ShapeConfig, apply_shape_profile, fit_shape_profile  # noqa: E402

ARMS = (
    ("binary/legacy_log", "binary", "legacy", "log", "mean"),
    ("binary/poisson", "binary", "poisson", None, "sum"),
    ("shape/poisson", "shape", "poisson", None, "sum"),
    ("runon/poisson", "runon", "poisson", None, "sum"),
    ("shape_runon/poisson", "shape_runon", "poisson", None, "sum"),
    ("shape_runon/legacy_log", "shape_runon", "legacy", "log", "mean"),
)
CANDIDATE = "shape_runon/poisson"
INCUMBENT = "runon/poisson"
PAIRS = (
    (CANDIDATE, INCUMBENT),
    (CANDIDATE, "binary/poisson"),
    (CANDIDATE, "binary/legacy_log"),
    ("shape/poisson", "binary/poisson"),
    (INCUMBENT, "binary/poisson"),
)


def run_seed(sim: Path, archetypes: Path) -> dict:
    started = time.perf_counter()
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
    base_counts = dienr.count_sample(base, "s", signal, summary="sum")
    vectors, flank = load_archetypes(archetypes)
    columns = transcript_columns(base, truth, vectors, flank)
    tail, n_training = fit_tail(base, base_counts, truth, signal, flank)
    profile = fit_shape_profile(annotation, signal, config=ShapeConfig())
    shaped = apply_shape_profile(base, profile)
    designs = {
        "binary": base,
        "shape": shaped,
        "runon": runon_designs(base, columns, tail)["runon"],
        # Same tail, added to the reshaped columns.
        "shape_runon": runon_designs(shaped, columns, tail)["runon"],
    }

    truth_by_model = (
        base.transcript_map.join(truth.select("transcript_id", "truth"), on="transcript_id")
        .group_by("locus_id", "model_index")
        .agg(truth=pl.col("truth").sum())
    )
    n_models = base.model_table.group_by("locus_id").agg(n_models=pl.len())
    tables: dict[str, pl.DataFrame] = {}
    for name, variant, likelihood, transform, summary in ARMS:
        design = designs[variant]
        if summary == "mean" and variant != "binary":
            design = _per_base_mean(design)
        counts = dienr.count_sample(design, "s", signal, summary=summary)
        kwargs: dict = {"likelihood": likelihood}
        if transform:
            kwargs["transform"] = transform
        result = dienr.fit(design, counts, **kwargs)
        tables[name] = (
            result.model_abundance.select(
                "locus_id", "model_index", "abundance_raw", "solver_status"
            )
            .join(truth_by_model, on=["locus_id", "model_index"])
            .join(n_models, on="locus_id")
        )
        print(f"  {sim.name}: fitted {name}", flush=True)

    calls = {name: _locus_calls(table) for name, table in tables.items()}
    common = set.intersection(*(set(c) for c in calls.values()))
    report: dict = {name: _arm_metrics(table) for name, table in tables.items()}
    report["_accuracy"] = {
        name: float(np.mean([calls[name][locus][0] for locus in common])) for name in calls
    }
    report["_pairs"] = {f"{a} vs {b}": _pair(calls[a], calls[b], common) for a, b in PAIRS}
    report["_meta"] = {
        "n_common_loci": float(len(common)),
        "tail_training_loci": float(n_training),
        "profile_dynamic_range": float(profile.dynamic_range),
        "seconds": time.perf_counter() - started,
    }
    return report


def _pair(arm: dict, reference: dict, loci: set) -> dict[str, float]:
    gained = lost = ties = 0
    wins: list[bool] = []
    for locus in loci:
        hit, _, err = arm[locus]
        ref_hit, _, ref_err = reference[locus]
        gained += int(hit and not ref_hit)
        lost += int(ref_hit and not hit)
        if np.isfinite(err) and np.isfinite(ref_err):
            if err == ref_err:
                ties += 1
            else:
                wins.append(err < ref_err)
    return {
        "gained": float(gained),
        "lost": float(lost),
        "win_share": gained / (gained + lost) if gained + lost else float("nan"),
        "error_win_fraction": float(np.mean(wins)) if wins else float("nan"),
        "error_ties": float(ties),
    }


def aggregate(paths: list[Path]) -> int:
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(paths)}
    seeds = list(reports)
    names = [arm[0] for arm in ARMS]

    def row(label: str, values: list[float], fmt: str = "{:9.3f}") -> None:
        cells = " ".join(fmt.format(v) for v in values)
        print(f"{label:>34} {cells}   mean {np.nanmean(values):.3f}")

    print(f"{'seeds:':>34} " + " ".join(f"{s.split('_')[-1]:>9}" for s in seeds))
    row("loci scored", [reports[s]["_meta"]["n_common_loci"] for s in seeds], "{:9.0f}")
    row("profile dynamic range", [reports[s]["_meta"]["profile_dynamic_range"] for s in seeds])
    print("\n== dominant-isoform accuracy, multi-model loci (same loci for every arm) ==")
    for name in names:
        row(name, [reports[s]["_accuracy"][name] for s in seeds])
    for metric, label in (
        ("multi_median_abs_log2", "median |log2 error|, multi-model loci"),
        ("multi_zero_fraction", "expressed models estimated at zero, multi-model"),
        ("single_median_abs_log2", "median |log2 error|, single-model loci"),
    ):
        print(f"\n== {label} ==")
        for name in names:
            row(name, [reports[s][name][metric] for s in seeds])

    print("\n== paired (error wins exclude exact ties) ==")
    for key in reports[seeds[0]]["_pairs"]:
        per = [reports[s]["_pairs"][key] for s in seeds]
        g, lo = sum(p["gained"] for p in per), sum(p["lost"] for p in per)
        row(f"{key}  win", [p["win_share"] for p in per])
        row("err wins", [p["error_win_fraction"] for p in per])
        print(f"{'':>34} pooled dominant {int(g)}/{int(lo)} = {g / (g + lo):.3f}")

    per = [reports[s]["_pairs"][f"{CANDIDATE} vs {INCUMBENT}"] for s in seeds]
    g, lo = sum(p["gained"] for p in per), sum(p["lost"] for p in per)
    pooled = g / (g + lo) if g + lo else float("nan")
    c1 = pooled >= 0.60 and all(p["win_share"] > 0.5 for p in per)
    c2 = all(p["error_win_fraction"] > 0.5 for p in per)
    print(f"\n== DECISION ({CANDIDATE} vs {INCUMBENT}) ==")
    print(
        f"  1. pooled win share {pooled:.3f} [{int(g)}/{int(lo)}], per seed "
        f"{[round(p['win_share'], 3) for p in per]} => {c1}"
    )
    print(f"  2. error wins per seed {[round(p['error_win_fraction'], 3) for p in per]} => {c2}")
    print(f"  => shape + run-on replaces run-on alone: {c1 and c2}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sim", type=Path)
    parser.add_argument("--archetypes", type=Path, default=Path("build/oracle"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aggregate", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.aggregate:
        return aggregate(args.aggregate)
    if not (args.sim and args.output):
        parser.error("--sim and --output are required unless --aggregate is given")
    report = run_seed(args.sim, args.archetypes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {args.output} in {report['_meta']['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
