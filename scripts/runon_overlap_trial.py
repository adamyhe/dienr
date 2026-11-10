"""Trial an overlap-only run-on tail against the full tail (plan 18.12).

18.10's fitted shared tail passes its rule, but `runon_diagnose.py` found its
per-locus error loss concentrated where a model's tail lands on no other
model's body (error wins 0.465 vs binary there). The mechanism: a bin no binary
column covers is unusable (`_glm.usable_bins`), so the full tail *adds* bins to
the fit -- past-PAS bins explained only by the shared tail, whose gene-specific
misfit then leaks into that model's abundance.

**The variant.** `runon_overlap` keeps each model's fitted tail only in bins
where some *other* model has unmasked support. Tail entries elsewhere are
dropped, so the usable bin set is exactly the binary design's: the tail only
re-attributes signal in bins already being fitted, which is the confusion it
exists to resolve. At loci where no tail overlaps another body the design *is*
the binary design. A per-model gate (full tail for a model iff it overlaps
anything) is reported as a secondary, descriptive arm, not part of the rule.

Same tail estimate `r(d)` as 18.10, same seeds, masking and paired scoring
(tie-aware, order-independent, 18.11).

**Decision rule, fixed before running.** `runon_overlap/poisson` replaces
`runon/poisson` as the candidate only if all hold:

1. It still passes 18.10's rule against `binary/poisson`: discordant-dominant
   win share >= 0.60 pooled and > 0.50 in every seed, and per-locus error wins
   > 0.50 in every seed.
2. Against `runon/poisson`, per-locus error wins > 0.50 in every seed.
3. Against `runon/poisson`, the dominant call is not worse: pooled discordant
   win share >= 0.50.

Error wins here are counted **over loci where the two arms' errors differ**;
exact ties are excluded and their count reported. This matters only for this
trial: where no tail overlaps another body, `runon_overlap` *is* the binary
design and ties it exactly, and 18.10's scoring would count every tie as a
loss for the challenger.

Otherwise the full tail stays the candidate.

**Result (18.12): the three designs are bit-identical.** A locus window is the
union of its transcripts' spans, so every in-window tail bin already overlaps
another model; the full tail never adds a bin. Kept as the record of that.

Also reported: both comparisons split by whether the locus has any
tail-on-other-body overlap.

    uv run python scripts/runon_overlap_trial.py --sim build/sim/sim1500_s5 \\
        --output build/runon_overlap/sim1500_s5.json
    uv run python scripts/runon_overlap_trial.py --aggregate build/runon_overlap/*.json
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
from oracle_shape_ceiling import (  # noqa: E402
    BIN_SIZE,
    PAPER_MASK,
    _locus_calls,
    _paired,
    _per_base_mean,
    load_archetypes,
    read_truth,
    transcript_columns,
)
from runon_trial import _tail_bins, fit_tail, runon_designs  # noqa: E402

import dienr  # noqa: E402

ARMS = (
    ("binary/legacy_log", "binary", "legacy", "log", "mean"),
    ("binary/poisson", "binary", "poisson", None, "sum"),
    ("runon/poisson", "runon", "poisson", None, "sum"),
    ("runon_overlap/poisson", "runon_overlap", "poisson", None, "sum"),
    ("runon_gated/poisson", "runon_gated", "poisson", None, "sum"),
    ("runon_overlap/legacy_log", "runon_overlap", "legacy", "log", "mean"),
)
REFERENCES = ("binary/poisson", "binary/legacy_log", "runon/poisson")
CANDIDATE, BASELINE, INCUMBENT = "runon_overlap/poisson", "binary/poisson", "runon/poisson"


def overlap_designs(base: dienr.Design, tail: np.ndarray) -> tuple[dict[str, dienr.Design], set]:
    """Overlap-only (bin-level) and gated (model-level) tails; loci with any overlap."""
    strands = base.loci["strand"].to_list()
    overlap: list[np.ndarray] = []
    gated: list[np.ndarray] = []
    overlapping_loci: set[int] = set()
    for locus_id, matrix in enumerate(base.models):
        unmasked = ~base.mask(locus_id)
        plus = strands[locus_id] == "+"
        supports = (matrix != 0) & unmasked[:, None]
        with_overlap = matrix.astype(float).copy()
        with_gate = matrix.astype(float).copy()
        for k in range(matrix.shape[1]):
            first, last = (int(x) for x in base.model_spans[locus_id][k])
            if first < 0 or matrix.shape[1] < 2:
                continue
            bins = _tail_bins(matrix.shape[0], first, last, plus)
            others = np.delete(supports, k, axis=1).any(axis=1)[bins]
            values = tail[: bins.size]
            with_overlap[bins, k] += values * others
            if (others & unmasked[bins]).any():
                with_gate[bins, k] += values
                overlapping_loci.add(locus_id)
        overlap.append(np.ascontiguousarray(with_overlap * unmasked[:, None]))
        gated.append(np.ascontiguousarray(with_gate * unmasked[:, None]))
    return {
        "runon_overlap": replace(
            base, models=overlap, digest=f"{base.digest}+runon_overlap", shape_digest="ro"
        ),
        "runon_gated": replace(
            base, models=gated, digest=f"{base.digest}+runon_gated", shape_digest="rg"
        ),
    }, overlapping_loci


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
    extra, overlapping = overlap_designs(base, tail)
    designs = {"binary": base, **runon_designs(base, columns, tail), **extra}

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

    report: dict = {"_paired": _paired(tables, REFERENCES)}
    # The split: same pairing, restricted to loci with / without any overlap.
    calls = {name: _locus_calls(table) for name, table in tables.items()}
    common = set.intersection(*(set(c) for c in calls.values()))
    for label, keep in (
        ("all", common),
        ("overlap", common & overlapping),
        ("no_overlap", common - overlapping),
    ):
        report[f"_split_{label}"] = _split(calls, keep)
    report["_meta"] = {
        "tail_training_loci": float(n_training),
        "n_overlap_loci": float(len(common & overlapping)),
        "n_common_loci": float(len(common)),
        "seconds": time.perf_counter() - started,
    }
    return report


def _split(calls: dict, loci: set) -> dict[str, dict[str, float]]:
    out = {}
    pairs = ((CANDIDATE, BASELINE), (CANDIDATE, INCUMBENT), (INCUMBENT, BASELINE))
    for name, reference in pairs:
        gained = lost = ties = 0
        wins: list[bool] = []
        for locus in loci:
            hit, _, err = calls[name][locus]
            ref_hit, _, ref_err = calls[reference][locus]
            gained += int(hit and not ref_hit)
            lost += int(ref_hit and not hit)
            if np.isfinite(err) and np.isfinite(ref_err):
                if err == ref_err:
                    ties += 1
                else:
                    wins.append(err < ref_err)
        out[f"{name} vs {reference}"] = {
            "n": float(len(loci)),
            "gained": float(gained),
            "lost": float(lost),
            "win_share": gained / (gained + lost) if gained + lost else float("nan"),
            "error_win_fraction": float(np.mean(wins)) if wins else float("nan"),
            "error_ties": float(ties),
        }
    return out


def aggregate(paths: list[Path]) -> int:
    reports = {p.stem: json.loads(p.read_text()) for p in sorted(paths)}
    seeds = list(reports)
    names = [arm[0] for arm in ARMS]

    def row(label: str, values: list[float], fmt: str = "{:9.3f}") -> None:
        cells = " ".join(fmt.format(v) for v in values)
        print(f"{label:>34} {cells}   mean {np.nanmean(values):.3f}")

    print(f"{'seeds:':>34} " + " ".join(f"{s.split('_')[-1]:>9}" for s in seeds))
    row("loci scored", [reports[s]["_meta"]["n_common_loci"] for s in seeds], "{:9.0f}")
    row(
        "  with tail-on-body overlap",
        [reports[s]["_meta"]["n_overlap_loci"] for s in seeds],
        "{:9.0f}",
    )
    print("\n== dominant-isoform accuracy (same loci for every arm) ==")
    for name in names:
        row(name, [reports[s]["_paired"]["_accuracy"][name] for s in seeds])
    for reference in REFERENCES:
        print(f"\n== paired against {reference}: win share / error wins ==")
        for name in names:
            if name == reference:
                continue
            key = f"{name} vs {reference}"
            row(f"{name}  win", [reports[s]["_paired"][key]["win_share"] for s in seeds])
            row("err wins", [reports[s]["_paired"][key]["error_win_fraction"] for s in seeds])

    for label in ("all", "overlap", "no_overlap"):
        print(f"\n== tie-excluded error wins, {label} loci ==")
        for key in reports[seeds[0]][f"_split_{label}"]:
            per = [reports[s][f"_split_{label}"][key] for s in seeds]
            g, lo = sum(p["gained"] for p in per), sum(p["lost"] for p in per)
            share = g / (g + lo) if g + lo else float("nan")
            row(f"{key}  err wins", [p["error_win_fraction"] for p in per])
            print(
                f"{'':>34} dominant gained/lost {int(g)}/{int(lo)}  pooled share {share:.3f}"
                f"  error ties {int(sum(p['error_ties'] for p in per))}"
            )

    def pooled(key: str) -> tuple[float, list[dict]]:
        per = [reports[s]["_split_all"][key] for s in seeds]
        g, lo = sum(p["gained"] for p in per), sum(p["lost"] for p in per)
        return (g / (g + lo) if g + lo else float("nan")), per

    share_b, per_b = pooled(f"{CANDIDATE} vs {BASELINE}")
    share_i, per_i = pooled(f"{CANDIDATE} vs {INCUMBENT}")
    c1 = (
        share_b >= 0.60
        and all(p["win_share"] > 0.5 for p in per_b)
        and all(p["error_win_fraction"] > 0.5 for p in per_b)
    )
    c2 = all(p["error_win_fraction"] > 0.5 for p in per_i)
    c3 = share_i >= 0.5
    print("\n== DECISION ==")
    print(
        f"  1. vs {BASELINE}: pooled win share {share_b:.3f}, error wins "
        f"{[round(p['error_win_fraction'], 3) for p in per_b]} => {c1}"
    )
    print(
        f"  2. vs {INCUMBENT}: error wins "
        f"{[round(p['error_win_fraction'], 3) for p in per_i]} => {c2}"
    )
    print(f"  3. vs {INCUMBENT}: pooled dominant win share {share_i:.3f} (>= 0.5) => {c3}")
    print(f"  => overlap-only replaces the full tail: {c1 and c2 and c3}")
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
