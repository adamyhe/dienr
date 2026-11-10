"""Oracle ceiling: could *any* density model beat a binary design here?

Plan 18.8 found a shared shape profile mostly not worth applying, and the DENR
paper (Discussion) suggests a richer per-gene density model instead. Before
building one, this asks the question that bounds it: if the design knew every
simulated isoform's *true* expected density, how much better would isoform
deconvolution get? No density model estimated from data can beat that, so a
ceiling that fails means stop.

It is exact, not approximate. nascentRNASim gives each simulated transcript the
width of exactly one archetype (archetype widths are unique) and generates its
sense signal by resampling that archetype's counts, so

    E[count at base p] = score * L * sense_arch[p] / abundance_estimate_arch

for one library-wide constant L. `scripts/export_archetypes.R` exports those
vectors; this script turns them into design columns.

Arms, all on the paper's masking (1 bin at the TSS, 4 at the PAS), scored on
multi-model loci with one calibration constant per arm:

    binary/legacy_log      DENR's default
    binary/poisson         the comparison that matters
    shape/poisson          shared profile (plan 18)
    oracle_flat/poisson    each isoform's true *scale*, no shape -- isolates the
                           units part of any oracle gain: truth is defined
                           against a masked-mean abundance estimate, so a flat
                           design is off by a per-archetype factor (CV 0.175)
    oracle_2level/poisson  true head (first 2 kb) and body levels, one step per
                           isoform: the simplest per-gene density model, with
                           its parameter known
    oracle_smooth/poisson  true shape, 5-bin running mean: the most a smooth
                           per-gene density model could hope to estimate
    oracle/poisson         exact expected density on the annotated support
    oracle_flank/poisson   ...plus run-on and upstream signal inside the locus
    oracle/legacy_log      the exact oracle under DENR's objective

**Decision rule, fixed before any result was seen.** Primary comparison is
``oracle/poisson`` against ``binary/poisson``, paired at the same locus. The
ceiling *exists* only if, at loci where the two pick different dominant
isoforms, the oracle is right in **>= 60% pooled and > 50% in every seed**, AND
the mean within-locus Spearman gain is **> 0 in every seed**. Otherwise no
density model can help on this benchmark -- the one most favorable to shape
modeling, since every simulated isoform *is* an archetype -- and the idea
should be dropped. If it passes, the ladder (flat -> 2level -> smooth -> exact)
says which part of the knowledge carries the gain.

Paired scoring, unlike 18.8, fixes each locus's eligibility independently of
the arm: a locus counts if it has >= 2 models, at least one expressed, and every
arm fitted it; an arm whose largest estimate is zero has made no call and is
scored wrong. 18.8 dropped loci per arm where fewer than two estimates were
positive, which lets arms be scored on different loci.

Usage, one seed per process (see build/slurm/oracle.sh), then aggregate:

    uv run python scripts/oracle_shape_ceiling.py --sim build/sim/sim1500_s5 \\
        --archetypes build/oracle --output build/oracle/sim1500_s5.json
    uv run python scripts/oracle_shape_ceiling.py --aggregate build/oracle/sim1500_s*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_shape_usefulness import EXPRESSED, PAPER_MASK, read_truth, spearman  # noqa: E402

import dienr  # noqa: E402
from dienr.shape import ShapeConfig, apply_shape_profile, fit_shape_profile  # noqa: E402

BIN_SIZE = 250
HEAD_BASES = 2_000
SMOOTH_BINS = 5

#: (arm, design variant, likelihood, transform, summary)
ARMS = (
    ("binary/legacy_log", "binary", "legacy", "log", "mean"),
    ("binary/poisson", "binary", "poisson", None, "sum"),
    ("shape/poisson", "shape", "poisson", None, "sum"),
    ("oracle_flat/poisson", "oracle_flat", "poisson", None, "sum"),
    ("oracle_2level/poisson", "oracle_2level", "poisson", None, "sum"),
    ("oracle_smooth/poisson", "oracle_smooth", "poisson", None, "sum"),
    ("oracle/poisson", "oracle", "poisson", None, "sum"),
    ("oracle_flank/poisson", "oracle_flank", "poisson", None, "sum"),
    ("oracle/legacy_log", "oracle", "legacy", "log", "mean"),
)
REFERENCES = ("binary/poisson", "binary/legacy_log")
PRIMARY = ("oracle/poisson", "binary/poisson")
WIN_SHARE_POOLED, WIN_SHARE_EACH = 0.60, 0.50


# --------------------------------------------------------------------------
# Oracle designs
# --------------------------------------------------------------------------


def load_archetypes(directory: Path) -> tuple[dict[int, np.ndarray], int]:
    """Expected per-base sense density per archetype, keyed by width."""
    meta = pl.read_csv(directory / "archetypes_meta.tsv", separator="\t")
    entries = pl.read_csv(directory / "archetypes_sense.tsv.gz", separator="\t")
    flanks = set(meta["flank"].to_list())
    if len(flanks) != 1:
        raise ValueError(f"archetypes disagree on flank: {flanks}")
    vectors: dict[int, np.ndarray] = {}
    for row in meta.iter_rows(named=True):
        sub = entries.filter(pl.col("archetype") == row["archetype"])
        vector = np.zeros(row["vector_length"])
        vector[sub["index0"].to_numpy()] = sub["value"].to_numpy()
        vectors[int(row["width"])] = vector / row["abundance_estimate"]
    return vectors, flanks.pop()


def transcript_columns(
    design: dienr.Design, truth: pl.DataFrame, vectors: dict[int, np.ndarray], flank: int
) -> dict[str, np.ndarray]:
    """Each transcript's expected signal summed into its locus's bins.

    A vector covers ``[start - flank, end + flank)`` in the direction of
    transcription; for a minus-strand transcript it is reversed onto the genome,
    after which it starts at ``start - flank`` either way.
    """
    loci = design.loci
    locus_start = dict(zip(loci["locus_id"].to_list(), loci["start"].to_list(), strict=True))
    by_tx = {row["transcript_id"]: row for row in truth.iter_rows(named=True)}
    columns: dict[str, np.ndarray] = {}
    for row in design.transcript_map.select("transcript_id", "locus_id").iter_rows(named=True):
        tx = by_tx[row["transcript_id"]]
        width = tx["end"] - tx["start"]
        if width not in vectors:
            raise ValueError(f"{row['transcript_id']}: width {width} matches no archetype")
        genomic = vectors[width] if tx["strand"] == "+" else vectors[width][::-1]
        origin = tx["start"] - flank
        locus_id = row["locus_id"]
        n_bins = design.bin_count(locus_id)
        window_start = int(locus_start[locus_id])
        window = np.zeros(n_bins * BIN_SIZE)
        lo = max(origin, window_start)
        hi = min(origin + genomic.size, window_start + window.size)
        if hi > lo:
            window[lo - window_start : hi - window_start] = genomic[lo - origin : hi - origin]
        columns[row["transcript_id"]] = window.reshape(n_bins, BIN_SIZE).sum(axis=1)
    return columns


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width)
    return np.convolve(values, kernel, "same") / np.convolve(np.ones_like(values), kernel, "same")


def oracle_designs(
    base: dienr.Design, columns: dict[str, np.ndarray]
) -> tuple[dict[str, dienr.Design], dict[str, float]]:
    """Every oracle variant of `base`, sharing its bins, masks and models."""
    members = (
        base.transcript_map.group_by("locus_id", "model_index")
        .agg(pl.col("transcript_id"))
        .iter_rows(named=True)
    )
    by_model = {(r["locus_id"], r["model_index"]): r["transcript_id"] for r in members}
    strands = base.loci["strand"].to_list()
    head_bins = HEAD_BASES // BIN_SIZE
    variants = ("oracle", "oracle_flank", "oracle_smooth", "oracle_flat", "oracle_2level")
    out: dict[str, list[np.ndarray]] = {v: [] for v in variants}
    n_mixed = n_models = 0

    for locus_id, matrix in enumerate(base.models):
        unmasked = ~base.mask(locus_id)
        built = {v: np.zeros_like(matrix, dtype=float) for v in variants}
        for k in range(matrix.shape[1]):
            n_models += 1
            txs = by_model[(locus_id, k)]
            stack = np.array([columns[t] for t in txs])
            # Members of one equivalence class share bins, not necessarily an
            # archetype; equal weights is the only abundance-free choice.
            n_mixed += int(len(txs) > 1 and not np.allclose(stack, stack[0]))
            full = stack.mean(axis=0)
            support = matrix[:, k] != 0
            built["oracle"][:, k] = full * support
            built["oracle_flank"][:, k] = full * unmasked
            built["oracle_smooth"][:, k] = _smooth(full, SMOOTH_BINS) * support
            if support.any():
                built["oracle_flat"][:, k] = full[support].mean() * support
                first, last = (int(x) for x in base.model_spans[locus_id][k])
                head = np.zeros(matrix.shape[0], dtype=bool)
                if strands[locus_id] == "+":
                    head[first : min(first + head_bins, last + 1)] = True
                else:
                    head[max(last - head_bins + 1, first) : last + 1] = True
                body = support & ~head
                head &= support
                head_level = full[head].mean() if head.any() else full[body].mean()
                body_level = full[body].mean() if body.any() else head_level
                built["oracle_2level"][:, k] = np.where(head, head_level, body_level) * support
        for v in variants:
            out[v].append(np.ascontiguousarray(built[v]))

    designs = {
        v: replace(base, models=out[v], digest=f"{base.digest}+{v}", shape_digest=v)
        for v in variants
    }
    return designs, {"n_models": float(n_models), "n_mixed_equivalence_classes": float(n_mixed)}


def _per_base_mean(design: dienr.Design) -> dienr.Design:
    """The same design in `summary="mean"` units: every bin reads 250 bases."""
    return replace(
        design,
        models=[m / BIN_SIZE for m in design.models],
        digest=f"{design.digest}+mean",
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


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
    profile = fit_shape_profile(annotation, signal, config=ShapeConfig())
    vectors, flank = load_archetypes(archetypes)
    oracles, oracle_stats = oracle_designs(base, transcript_columns(base, truth, vectors, flank))
    designs = {"binary": base, "shape": apply_shape_profile(base, profile), **oracles}

    truth_by_model = (
        base.transcript_map.join(truth.select("transcript_id", "truth"), on="transcript_id")
        .group_by("locus_id", "model_index")
        .agg(truth=pl.col("truth").sum())
    )
    n_models = base.model_table.group_by("locus_id").agg(n_models=pl.len())

    tables: dict[str, pl.DataFrame] = {}
    for name, variant, likelihood, transform, summary in ARMS:
        design = (
            designs[variant]
            if summary == "sum" or variant == "binary"
            else _per_base_mean(designs[variant])
        )
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

    report = {name: _arm_metrics(table) for name, table in tables.items()}
    report["_paired"] = _paired(tables)
    report["_meta"] = {
        **oracle_stats,
        "profile_dynamic_range": profile.dynamic_range,
        "seconds": time.perf_counter() - started,
    }
    return report


def _calibration(table: pl.DataFrame) -> float:
    ok = table.filter((pl.col("solver_status") == "ok") & (pl.col("truth") > EXPRESSED))
    ratio = (ok["abundance_raw"] / ok["truth"]).to_numpy()
    ratio = ratio[ratio > 0]
    return float(np.median(ratio)) if ratio.size else float("nan")


def _arm_metrics(table: pl.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for stratum, frame in (
        ("multi", table.filter(pl.col("n_models") >= 2)),
        ("single", table.filter(pl.col("n_models") == 1)),
    ):
        calibration = _calibration(frame)
        expressed = frame.filter((pl.col("solver_status") == "ok") & (pl.col("truth") > EXPRESSED))
        estimate, actual = expressed["abundance_raw"].to_numpy(), expressed["truth"].to_numpy()
        positive = estimate > 0
        log_err = np.abs(np.log2(estimate[positive] / actual[positive] / calibration))
        out[f"{stratum}_median_abs_log2"] = float(np.median(log_err)) if log_err.size else np.nan
        out[f"{stratum}_zero_fraction"] = float(np.mean(~positive)) if positive.size else np.nan
        out[f"{stratum}_spearman_pooled"] = spearman(estimate, actual)
    return out


def _locus_calls(table: pl.DataFrame) -> dict[int, tuple[bool, float, float]]:
    """locus -> (dominant correct, within-locus rho or nan, median abs log2 error)."""
    calibration = _calibration(table.filter(pl.col("n_models") >= 2))
    multi = table.filter(pl.col("n_models") >= 2)
    failed = set(multi.filter(pl.col("solver_status") != "ok")["locus_id"].to_list())
    calls: dict[int, tuple[bool, float, float]] = {}
    for (locus_id,), frame in multi.group_by("locus_id"):
        if locus_id in failed:
            continue
        frame = frame.sort("model_index")  # group_by row order is not fixed
        estimate, actual = frame["abundance_raw"].to_numpy(), frame["truth"].to_numpy()
        if actual.max() <= EXPRESSED:
            continue
        # Correct if the top estimate is *a* truly dominant model: truth ties
        # are common (discrete abundance kernel) and argmax over them is arbitrary.
        correct = bool(estimate.max() > 0 and actual[np.argmax(estimate)] == actual.max())
        expressed = actual > EXPRESSED
        rho = spearman(estimate[expressed], actual[expressed]) if expressed.sum() >= 3 else np.nan
        keep = expressed & (estimate > 0)
        err = (
            float(np.median(np.abs(np.log2(estimate[keep] / actual[keep] / calibration))))
            if keep.any()
            else np.nan
        )
        calls[int(locus_id)] = (correct, rho, err)
    return calls


def _paired(
    tables: dict[str, pl.DataFrame], references: tuple[str, ...] = REFERENCES
) -> dict[str, dict[str, float]]:
    calls = {name: _locus_calls(table) for name, table in tables.items()}
    common = set.intersection(*(set(c) for c in calls.values()))
    out: dict[str, dict[str, float]] = {}
    for reference in references:
        for name in calls:
            if name == reference:
                continue
            gained = lost = 0
            rho_gain: list[float] = []
            err_wins: list[bool] = []
            for locus in common:
                hit, rho, err = calls[name][locus]
                ref_hit, ref_rho, ref_err = calls[reference][locus]
                gained += int(hit and not ref_hit)
                lost += int(ref_hit and not hit)
                if np.isfinite(rho) and np.isfinite(ref_rho):
                    rho_gain.append(rho - ref_rho)
                if np.isfinite(err) and np.isfinite(ref_err):
                    err_wins.append(err < ref_err)
            out[f"{name} vs {reference}"] = {
                "gained": float(gained),
                "lost": float(lost),
                "win_share": gained / (gained + lost) if gained + lost else float("nan"),
                "rho_gain_mean": float(np.mean(rho_gain)) if rho_gain else float("nan"),
                "n_rho_loci": float(len(rho_gain)),
                "error_win_fraction": float(np.mean(err_wins)) if err_wins else float("nan"),
            }
    out["_accuracy"] = {
        name: float(np.mean([calls[name][locus][0] for locus in common])) for name in calls
    }
    out["_n_common_loci"] = {"n": float(len(common))}
    return out


# --------------------------------------------------------------------------
# Aggregation and the decision
# --------------------------------------------------------------------------


def aggregate(paths: list[Path]) -> int:
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(paths)}
    seeds = list(reports)
    names = [arm[0] for arm in ARMS]

    def row(label: str, values: list[float], fmt: str = "{:9.3f}") -> None:
        cells = " ".join(fmt.format(v) for v in values)
        print(f"{label:>34} {cells}   mean {np.nanmean(values):.3f}")

    print(f"{'seeds:':>34} " + " ".join(f"{s.split('_')[-1]:>9}" for s in seeds))
    print("\n== dominant-isoform accuracy, multi-model loci (same loci for every arm) ==")
    for name in names:
        row(name, [reports[s]["_paired"]["_accuracy"][name] for s in seeds])
    for metric, label in (
        ("multi_median_abs_log2", "median |log2 error|, multi-model loci"),
        ("single_median_abs_log2", "median |log2 error|, single-model loci (oracle sanity)"),
        ("multi_zero_fraction", "expressed models estimated at zero, multi-model"),
    ):
        print(f"\n== {label} ==")
        for name in names:
            row(name, [reports[s][name][metric] for s in seeds])

    for reference in REFERENCES:
        print(f"\n== paired against {reference} ==")
        print(f"{'':>34} win share of discordant loci | within-locus rho gain | error wins")
        for name in names:
            if name == reference:
                continue
            key = f"{name} vs {reference}"
            row(f"{name}  win share", [reports[s]["_paired"][key]["win_share"] for s in seeds])
            row(
                "rho gain", [reports[s]["_paired"][key]["rho_gain_mean"] for s in seeds], "{:+9.3f}"
            )
            row("error wins", [reports[s]["_paired"][key]["error_win_fraction"] for s in seeds])

    key = f"{PRIMARY[0]} vs {PRIMARY[1]}"
    per_seed = [reports[s]["_paired"][key] for s in seeds]
    gained = sum(p["gained"] for p in per_seed)
    lost = sum(p["lost"] for p in per_seed)
    pooled = gained / (gained + lost) if gained + lost else float("nan")
    each_share = all(p["win_share"] > WIN_SHARE_EACH for p in per_seed)
    each_rho = all(p["rho_gain_mean"] > 0 for p in per_seed)
    verdict = pooled >= WIN_SHARE_POOLED and each_share and each_rho
    print(f"\n== DECISION ({key}) ==")
    print(
        f"  pooled win share {pooled:.3f} (needs >= {WIN_SHARE_POOLED})  "
        f"[{int(gained)} gained / {int(lost)} lost]"
    )
    print(f"  win share > {WIN_SHARE_EACH} in every seed: {each_share}")
    print(f"  within-locus rho gain > 0 in every seed: {each_rho}")
    print(f"  => ceiling {'EXISTS: a density model could help' if verdict else 'ABSENT: stop'}")
    meta = [reports[s]["_meta"] for s in seeds]
    print(
        f"\n  mixed equivalence classes: {[int(m['n_mixed_equivalence_classes']) for m in meta]} "
        f"of {[int(m['n_models']) for m in meta]} models"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sim", type=Path, help="simulation prefix, e.g. build/sim/sim1500_s5")
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
