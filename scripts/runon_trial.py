"""Trial a *fitted* run-on (PAS read-through) term on the oracle harness.

Plan 18.9 found most of the oracle ceiling in run-on: giving each isoform its
true expected signal past the PAS moved dominant-isoform accuracy from 0.912 to
0.978. Read-through from a shorter isoform lands in a longer isoform's body and
looks like it, which the 4-bin PAS mask only partly removes (bins PAS+1..+4).
This asks whether that gain survives *estimation*.

**The fitted term.** Each binary model column gets a shared tail ``r(d)`` in the
``d``-th bin past its PAS, in the direction of transcription, inside the locus
window (masked bins stay zero). ``r`` is estimated from the data alone as a
ratio of sums over isolated single-isoform loci,

    r(d) = sum_genes tail_signal(d) / sum_genes body_density,

which weights genes by expression and so is stable under Poisson noise. A
training locus must have no other transcript, on either strand, within its
tail window widened by the simulator's flank -- another gene's signal would
read as run-on. Training loci are single-isoform and scoring loci are
multi-isoform, so they are disjoint; no truth is used in fitting.

Arms (paper masking; scored on the same multi-model loci, as in 18.9):

    binary/legacy_log          DENR default
    binary/poisson             the baseline the term is added to
    runon_oracle/poisson       binary body + each isoform's *exact* tail, in
                               body units: the ceiling of this component alone
    runon/poisson              binary body + fitted shared tail
    runon/legacy_log           the same design under DENR's objective
    oracle/poisson             18.9 references: exact body, no tail ...
    oracle_flank/poisson       ... and exact body plus exact flanks

**Decision rule, fixed before running.** Primary: ``runon/poisson`` vs
``binary/poisson``, paired. Adopt as a package candidate only if it wins
**>= 60% of discordant-dominant loci pooled and > 50% in every seed**, AND has
lower per-locus error at **> 50% of loci in every seed**. (18.9's within-locus
Spearman criterion is replaced by the error criterion because it covered only
9-16 loci per seed; the replacement is made from that lesson, before any run-on
result.) To be *recommended* it must also beat DENR's default on pooled win
share. Also reported: the fraction of the exact tail's gain the fitted tail
realizes.

Caveat built into the simulator: archetype vectors end 6 kb past the PAS, so
simulated run-on is truncated there. Real read-through is often longer and
varies by gene, so a fitted shared tail is easier here than in biology.

    uv run python scripts/runon_trial.py --sim build/sim/sim1500_s5 \\
        --archetypes build/oracle --output build/runon/sim1500_s5.json
    uv run python scripts/runon_trial.py --aggregate build/runon/sim1500_s*.json
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
from figwig import BigWigReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_shape_ceiling import (  # noqa: E402
    BIN_SIZE,
    EXPRESSED,
    PAPER_MASK,
    _arm_metrics,
    _paired,
    _per_base_mean,
    load_archetypes,
    oracle_designs,
    read_truth,
    transcript_columns,
)

import dienr  # noqa: E402
from dienr.counts import read_loci_binned  # noqa: E402

#: Tail length in bins: the simulator's 6 kb flank at 250 bp.
TAIL_BINS = 24
#: A training locus needs at least this much body signal per bin (sum summary),
#: judged from the data, so the ratio is not dominated by near-silent genes.
MIN_BODY_PER_BIN = 2.0

ARMS = (
    ("binary/legacy_log", "binary", "legacy", "log", "mean"),
    ("binary/poisson", "binary", "poisson", None, "sum"),
    ("runon_oracle/poisson", "runon_oracle", "poisson", None, "sum"),
    ("runon/poisson", "runon", "poisson", None, "sum"),
    ("runon/legacy_log", "runon", "legacy", "log", "mean"),
    ("oracle/poisson", "oracle", "poisson", None, "sum"),
    ("oracle_flank/poisson", "oracle_flank", "poisson", None, "sum"),
)
REFERENCES = ("binary/poisson", "binary/legacy_log")
#: Arms that fail at most loci; excluded from the set the decision is paired on.
ORACLE_ARMS = ("oracle/poisson", "oracle_flank/poisson")
PRIMARY = ("runon/poisson", "binary/poisson")
WIN_SHARE_POOLED, WIN_SHARE_EACH, ERROR_WINS_EACH = 0.60, 0.50, 0.50


# --------------------------------------------------------------------------
# Fitting the shared tail
# --------------------------------------------------------------------------


def fit_tail(
    base: dienr.Design,
    counts: dienr.SampleCounts,
    truth: pl.DataFrame,
    signal: tuple[dienr.SignalFile, dienr.SignalFile],
    flank: int,
) -> tuple[np.ndarray, int]:
    """Shared run-on tail ``r(d)``, d = 1..TAIL_BINS, in body-density units."""
    span = TAIL_BINS * BIN_SIZE
    loci = base.loci
    readers = {s.strand: BigWigReader(str(s.path)) for s in signal}
    by_contig = {
        contig: frame.select("start", "end").to_numpy()
        for (contig,), frame in truth.group_by("contig")
    }
    windows: dict[str, list[tuple[str, int, int]]] = {"+": [], "-": []}
    bodies: dict[str, list[float]] = {"+": [], "-": []}
    for row in loci.filter(pl.col("n_transcripts") == 1).iter_rows(named=True):
        locus_id = row["locus_id"]
        support = (base.models[locus_id][:, 0] != 0) & ~base.mask(locus_id)
        if not support.any():
            continue
        body = float(counts.values[locus_id][support].mean())
        if body < MIN_BODY_PER_BIN:
            continue
        # Anchor on the model's own span, exactly as `_tail_bins` does in the
        # design: the window's last bin can be rounded out of the support, and
        # anchoring on the window edge would shift the fit by a bin.
        first, last = (int(x) for x in base.model_spans[locus_id][0])
        origin = int(row["start"])
        if row["strand"] == "+":
            tail_start = origin + (last + 1) * BIN_SIZE
        else:
            tail_start = origin + first * BIN_SIZE - span
        if tail_start < 0:
            continue
        # Isolation: no other transcript's signal (annotation +/- flank) may
        # reach the tail window, on either strand.
        spans = by_contig[row["contig"]]
        reach = (spans[:, 0] - flank < tail_start + span) & (spans[:, 1] + flank > tail_start)
        own = (spans[:, 0] == row["start"]) & (spans[:, 1] == row["end"])
        if (reach & ~own).any():
            continue
        windows[row["strand"]].append((row["contig"], tail_start, TAIL_BINS))
        bodies[row["strand"]].append(body)

    numerator = np.zeros(TAIL_BINS)
    denominator = 0.0
    n_genes = 0
    for strand in ("+", "-"):
        if not windows[strand]:
            continue
        tails = read_loci_binned(readers[strand], windows[strand], BIN_SIZE, summary="sum")
        for tail, body in zip(tails, bodies[strand], strict=True):
            # Orient so d = 1 is the bin adjacent to the PAS.
            numerator += tail if strand == "+" else tail[::-1]
            denominator += body
            n_genes += 1
    if n_genes == 0:
        raise ValueError("no isolated, expressed single-isoform locus to fit a tail from")
    return numerator / denominator, n_genes


# --------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------


def _tail_bins(n_bins: int, first: int, last: int, plus: bool) -> np.ndarray:
    """Bin indices d = 1.. past the PAS, clipped to the locus window."""
    if plus:
        return np.arange(last + 1, min(last + 1 + TAIL_BINS, n_bins))
    return np.arange(first - 1, max(first - 1 - TAIL_BINS, -1), -1)


def runon_designs(
    base: dienr.Design, columns: dict[str, np.ndarray], tail: np.ndarray
) -> dict[str, dienr.Design]:
    """Binary body plus a fitted shared tail, and plus each isoform's exact tail."""
    members = {
        (r["locus_id"], r["model_index"]): r["transcript_id"]
        for r in base.transcript_map.group_by("locus_id", "model_index")
        .agg(pl.col("transcript_id"))
        .iter_rows(named=True)
    }
    strands = base.loci["strand"].to_list()
    fitted: list[np.ndarray] = []
    exact: list[np.ndarray] = []
    for locus_id, matrix in enumerate(base.models):
        unmasked = ~base.mask(locus_id)
        plus = strands[locus_id] == "+"
        with_fit = matrix.astype(float).copy()
        with_exact = matrix.astype(float).copy()
        for k in range(matrix.shape[1]):
            first, last = (int(x) for x in base.model_spans[locus_id][k])
            if first < 0:
                continue
            bins = _tail_bins(matrix.shape[0], first, last, plus)
            support = (matrix[:, k] != 0) & unmasked
            with_fit[bins, k] += tail[: bins.size]
            # Exact tail in body units, averaged over the class's members.
            normalized = []
            for tx in members[(locus_id, k)]:
                full = columns[tx]
                level = full[support].mean() if support.any() else 0.0
                normalized.append(full[bins] / level if level > 0 else np.zeros(bins.size))
            with_exact[bins, k] += np.mean(normalized, axis=0)
        fitted.append(np.ascontiguousarray(with_fit * unmasked[:, None]))
        exact.append(np.ascontiguousarray(with_exact * unmasked[:, None]))
    return {
        "runon": replace(base, models=fitted, digest=f"{base.digest}+runon", shape_digest="runon"),
        "runon_oracle": replace(
            base, models=exact, digest=f"{base.digest}+runon_oracle", shape_digest="runon_oracle"
        ),
    }


# --------------------------------------------------------------------------
# One seed
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
    base_counts = dienr.count_sample(base, "s", signal, summary="sum")
    vectors, flank = load_archetypes(archetypes)
    columns = transcript_columns(base, truth, vectors, flank)
    tail, n_training = fit_tail(base, base_counts, truth, signal, flank)

    # The exact mean tail over the same kind of locus, to compare the fit with.
    oracles, oracle_stats = oracle_designs(base, columns)
    designs = {"binary": base, **oracles, **runon_designs(base, columns, tail)}

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

    report = {name: _arm_metrics(table) for name, table in tables.items()}
    # The exact-body oracle arms fail (`invalid_mean`) at most loci: their
    # columns vanish in bins that still hold counts. Pairing over *every* arm
    # therefore scored the decision on the ~25% of loci where those unrelated
    # arms happen to fit. The decision is paired over the non-oracle arms; the
    # all-arm subset is kept as a separate readout.
    decision_tables = {n: t for n, t in tables.items() if n not in ORACLE_ARMS}
    report["_paired"] = _paired(decision_tables, REFERENCES)
    report["_paired_oracle_subset"] = _paired(tables, REFERENCES)
    report["_meta"] = {
        **oracle_stats,
        "tail": [float(x) for x in tail],
        "tail_training_loci": float(n_training),
        "tail_mass": float(tail.sum()),
        "seconds": time.perf_counter() - started,
        "expressed_threshold": EXPRESSED,
    }
    return report


# --------------------------------------------------------------------------
# Aggregation and the decision
# --------------------------------------------------------------------------


def aggregate(paths: list[Path]) -> int:
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(paths)}
    seeds = list(reports)
    names = [arm[0] for arm in ARMS]

    def row(label: str, values: list[float], fmt: str = "{:9.3f}") -> None:
        print(
            f"{label:>30} "
            + " ".join(fmt.format(v) for v in values)
            + f"   mean {np.nanmean(values):.3f}"
        )

    print(f"{'seeds:':>30} " + " ".join(f"{s.split('_')[-1]:>9}" for s in seeds))
    meta = [reports[s]["_meta"] for s in seeds]
    print("\n== fitted tail r(d), body-density units (mean over seeds) ==")
    tails = np.array([m["tail"] for m in meta])
    print(
        "  d (kb past PAS): "
        + " ".join(f"{(d + 1) * BIN_SIZE / 1000:5.2f}" for d in range(0, TAIL_BINS, 2))
    )
    print("  r(d):            " + " ".join(f"{v:5.2f}" for v in tails.mean(axis=0)[::2]))
    row("training loci", [m["tail_training_loci"] for m in meta], "{:9.0f}")
    row("tail mass (sum r)", [m["tail_mass"] for m in meta])

    for metric, label in (
        ("multi_median_abs_log2", "median |log2 error|, multi-model loci (all fitted loci)"),
        ("multi_zero_fraction", "expressed models estimated at zero, multi-model"),
        ("single_median_abs_log2", "median |log2 error|, single-model loci"),
    ):
        print(f"\n== {label} ==")
        for name in names:
            row(name, [reports[s][name][metric] for s in seeds])

    decide(reports, seeds, [n for n in names if n not in ORACLE_ARMS], "_paired", row)
    print("\n\n######## as first scored: paired over every arm, oracle arms included ########")
    print("(restricted to the loci where the exact-body oracle arms fit; not the decision)")
    decide(reports, seeds, names, "_paired_oracle_subset", row)
    return 0


def decide(reports: dict, seeds: list[str], names: list[str], section: str, row) -> None:
    """Paired accuracy, win shares, and the pre-registered decision on one locus set."""
    n_loci = [reports[s][section]["_n_common_loci"]["n"] for s in seeds]
    print(f"\n== dominant-isoform accuracy, paired loci ({section}) ==")
    row("loci scored", n_loci, "{:9.0f}")
    for name in names:
        row(name, [reports[s][section]["_accuracy"][name] for s in seeds])

    for reference in REFERENCES:
        print(f"\n== paired against {reference}: win share of discordant loci / error wins ==")
        for name in names:
            if name == reference:
                continue
            key = f"{name} vs {reference}"
            row(f"{name}  win", [reports[s][section][key]["win_share"] for s in seeds])
            row("err wins", [reports[s][section][key]["error_win_fraction"] for s in seeds])

    key = f"{PRIMARY[0]} vs {PRIMARY[1]}"
    per_seed = [reports[s][section][key] for s in seeds]
    gained, lost = sum(p["gained"] for p in per_seed), sum(p["lost"] for p in per_seed)
    pooled = gained / (gained + lost) if gained + lost else float("nan")
    each_share = all(p["win_share"] > WIN_SHARE_EACH for p in per_seed)
    each_error = all(p["error_win_fraction"] > ERROR_WINS_EACH for p in per_seed)
    candidate = pooled >= WIN_SHARE_POOLED and each_share and each_error

    vs_default = [reports[s][section][f"{PRIMARY[0]} vs binary/legacy_log"] for s in seeds]
    g2, l2 = sum(p["gained"] for p in vs_default), sum(p["lost"] for p in vs_default)
    pooled_default = g2 / (g2 + l2) if g2 + l2 else float("nan")
    each_default = all(p["win_share"] > WIN_SHARE_EACH for p in vs_default)

    acc = {n: np.mean([reports[s][section]["_accuracy"][n] for s in seeds]) for n in names}
    room = acc["runon_oracle/poisson"] - acc["binary/poisson"]
    realized = (acc["runon/poisson"] - acc["binary/poisson"]) / room if room > 0 else float("nan")

    print(f"\n== DECISION ({key}; {section}) ==")
    print(
        f"  pooled win share {pooled:.3f} (needs >= {WIN_SHARE_POOLED})  "
        f"[{int(gained)} / {int(lost)}]"
    )
    print(f"  win share > {WIN_SHARE_EACH} in every seed: {each_share}")
    print(f"  error wins > {ERROR_WINS_EACH} in every seed: {each_error}")
    print(f"  => package candidate: {candidate}")
    print(
        f"  vs DENR default, pooled win share {pooled_default:.3f} [{int(g2)} / {int(l2)}],"
        f" > 0.5 every seed: {each_default}"
        f" => recommendable: {candidate and pooled_default > 0.5}"
    )
    print(f"  share of the exact tail's accuracy gain realized by the fitted tail: {realized:.2f}")


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
