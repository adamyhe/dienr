"""Tune the log objective's offset and the bin size, without forking paths (plan 18.23).

18.22 made the built-in defaults -- DENR's: log least squares on
``log(y + 0.001)`` with a ``mean`` summary, 250 bp bins, no masks -- the
dominant-isoform recommendation. Two of those numbers were never tuned. The
offset ``c`` sets how much empty bins weigh, which is exactly why the log
objective ranks isoforms well and misfits counts. The bin size trades
resolution against counts per bin.

**Grid** (all: legacy log, ``mean``, no masks, no tail): bin size
{100, 250, 500} x ``c`` {1e-4, 1e-3, 1e-2, 1e-1, 1}, 15 arms. Locus grouping
is fixed at ``distance=250`` for every arm (the default at 250 bp bins), so
every arm sees the same loci and only the binning changes. ``b250_c0.001`` is
the built-in default.

**Truth and scoring** are 18.20's (ENCODE K562 PacBio long reads, Ensembl ID
matching, groups every arm resolves, decisive >= 2x loci).

**Selection, then confirmation, on disjoint data.** Choosing the best of 15
arms on the data that then judges it would overstate the winner. So:

- **select** on PRO-seq **rep1**, loci on **odd** chromosomes (1, 3, ..., 21):
  the arm with the highest dominant accuracy (ties broken toward the default,
  then toward 250 bp, then toward ``c`` nearest 1e-3 on a log scale);
- **confirm** on PRO-seq **rep2**, loci on **even** chromosomes and X.

**Decision rule, fixed before running.** If the selected arm is not the
default, it replaces the default's setting in the dominant-isoform
recommendation only if, on the confirmation set, it has higher accuracy than
``b250_c0.001`` AND wins > 50% of the loci where exactly one of the two is
correct, with a two-sided sign test p < 0.05. The code defaults stay DENR's
either way (R parity); a passing setting becomes a recommended flag. The full
grid on every locus of both replicates is reported, not used.

    # per replicate (Slurm):
    uv run python scripts/k562_longread_sweep.py run \\
        --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --longread build/k562/longread --score rep1 --tail-from rep2 \\
        --output build/k562/sweep_rep1.json --per-locus build/k562/sweep_rep1.parquet
    # then:
    uv run python scripts/k562_longread_sweep.py decide \\
        --select build/k562/sweep_rep1.parquet --confirm build/k562/sweep_rep2.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

BINS = (100, 250, 500)
OFFSETS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DEFAULT = "b250_c0.001"
DISTANCE = 250
SELECT_CONTIGS = {f"chr{i}" for i in range(1, 23, 2)}
CONFIRM_CONTIGS = {f"chr{i}" for i in range(2, 23, 2)} | {"chrX"}


def arm_name(bin_size: int, offset: float) -> str:
    return f"b{bin_size}_c{offset:g}"


ARMS: dict[str, dict] = {
    arm_name(b, c): {
        "bin": b,
        "distance": DISTANCE,
        "tss": (0, 0),
        "pas": (0, 0),
        "tail": False,
        "likelihood": "legacy",
        "log_offset": c,
    }
    for b in BINS
    for c in OFFSETS
}


def _tiebreak(name: str) -> tuple:
    arm = ARMS[name]
    return (
        name != DEFAULT,
        arm["bin"] != 250,
        abs(math.log10(arm["log_offset"]) - math.log10(1e-3)),
    )


def decide(select_path: Path, confirm_path: Path) -> dict:
    select = pl.read_parquet(select_path).filter(pl.col("contig").is_in(SELECT_CONTIGS))
    confirm = pl.read_parquet(confirm_path).filter(pl.col("contig").is_in(CONFIRM_CONTIGS))
    accuracy = {name: float(select[name].mean()) for name in ARMS}
    best = max(accuracy.values())
    chosen = min((n for n, a in accuracy.items() if a == best), key=_tiebreak)
    out: dict = {
        "selection_loci": select.height,
        "confirmation_loci": confirm.height,
        "selection_accuracy": accuracy,
        "selected": chosen,
    }
    if chosen == DEFAULT:
        out["decision"] = "the default was selected; nothing to confirm"
        return out
    a = confirm[chosen].to_numpy()
    b = confirm[DEFAULT].to_numpy()
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    n = only_a + only_b
    p = float(binomtest(only_a, n).pvalue) if n else float("nan")
    passes = bool(a.mean() > b.mean() and n and only_a / n > 0.5 and p < 0.05)
    out["confirmation"] = {
        "accuracy_selected": float(a.mean()),
        "accuracy_default": float(b.mean()),
        "only_selected": only_a,
        "only_default": only_b,
        "win_share": only_a / n if n else float("nan"),
        "p_two_sided": p,
    }
    out["decision"] = (
        f"{chosen} passes: recommend it for dominant isoforms"
        if passes
        else f"{chosen} does not confirm: keep the defaults"
    )
    return out


def grid(path: Path) -> str:
    frame = pl.read_parquet(path)
    lines = [
        f"{path.name}: {frame.height} loci",
        "bin \\ c  " + "".join(f"{c:>9g}" for c in OFFSETS),
    ]
    for b in BINS:
        lines.append(
            f"{b:>7d}  " + "".join(f"{frame[arm_name(b, c)].mean():9.3f}" for c in OFFSETS)
        )
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        from k562_longread_accuracy import main as run

        return run(sys.argv[2:], arms=ARMS, candidates={}, extra_pairs=(), description=__doc__)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--select", type=Path, required=True)
    d.add_argument("--confirm", type=Path, required=True)
    d.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = decide(args.select, args.confirm)
    print(grid(args.select))
    print(grid(args.confirm))
    print(json.dumps(result, indent=1))
    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
