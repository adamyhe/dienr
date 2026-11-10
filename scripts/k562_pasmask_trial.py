"""Does the run-on tail make the PAS mask unnecessary? (plan 18.18)

The recommended configuration masks 4 bins on each side of every PAS
(``mask_end_bins=(4, 4)``, the paper's setting). The mask hid signal near the
PAS that a binary design can't explain. The run-on tail now explains much of
it: on K562 the fitted tail peaks at ~1.4x body density ~1.25 kb past the PAS
(18.14), inside the outer half of that mask. So the mask may now be discarding
informative bins.

Setup is 18.17's: K562 PRO-seq (GSE96869, 0 min), Ensembl GRCh37.87
protein-coding, 250 bp bins, TSS mask ``(1, 1)`` throughout, Poisson, the
packaged tail fitted on the *other* replicate, deviance with
``mu = max(M a, 0.01)``.

**Arms** (PAS mask ``(inside, outside)`` bins):

- ``binary_4_4`` -- reference: the paper's design, no tail.
- ``runon_4_4`` -- the current recommendation.
- ``runon_4_0`` -- unmask past the PAS only (where the tail lives).
- ``runon_1_1``, ``runon_0_0`` -- shrink or drop the PAS mask.
- ``binary_0_0`` -- no PAS mask, no tail (does unmasking help on its own?).

**Comparable scoring.** Arms with smaller masks fit more bins, so each arm's
own held-out deviance would be over different bins. Instead every arm is
scored on the *same* test bins: those usable under the ``(4, 4)`` mask, split
into 5 contiguous blocks. For a block spanning bin indices ``[lo, hi]``, each
arm trains on its own usable bins outside ``[lo, hi]`` -- so a smaller mask
only ever adds *training* bins, and the question is whether they improve the
prediction of bins every arm can see. Counts are read once with no PAS mask;
the bins an arm masks are never used by it. Loci and bins are identical across
arms (windows are unions of transcript spans; masking doesn't change them),
which the script asserts; model counts can differ (less masking can split an
equivalence class).

A locus is scored if it has >= 2 models in ``binary_4_4``, >= 10 test bins
and >= 20 reads in them, and every arm's fit succeeds on every fold.

**Decision rule, fixed before running.** Against ``runon_4_4``, a smaller
mask qualifies if, in **both** replicates (tail from the other one):

1. lower total held-out deviance, AND
2. lower per-locus held-out deviance at > 50% of the loci where they differ.

If several qualify, the new default is the one with the lowest total deviance
averaged over the two replicates' relative changes. ``binary_0_0`` vs
``binary_4_4`` is reported, not part of the rule.

    uv run python scripts/k562_pasmask_trial.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --score rep1 --tail-from rep2 --output build/k562/pasmask_rep1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_runon_check import (  # noqa: E402
    BIN_SIZE,
    MIN_READS,
    MIN_USABLE_BINS,
    N_FOLDS,
    deviance,
    read_protein_coding,
    signal_files,
)

import dienr  # noqa: E402
from dienr.solvers import poisson  # noqa: E402

TSS_MASK = (1, 1)
REFERENCE = "runon_4_4"
ARMS: dict[str, tuple[tuple[int, int], bool]] = {
    "binary_4_4": ((4, 4), False),
    "runon_4_4": ((4, 4), True),
    "runon_4_0": ((4, 0), True),
    "runon_1_1": ((1, 1), True),
    "runon_0_0": ((0, 0), True),
    "binary_0_0": ((0, 0), False),
}
CANDIDATES = ["runon_4_0", "runon_1_1", "runon_0_0"]


def usable(matrix: np.ndarray) -> np.ndarray:
    return np.flatnonzero((matrix != 0).any(axis=1))


def score_locus(
    matrices: dict[str, np.ndarray], test_bins: np.ndarray, y: np.ndarray
) -> dict[str, float] | None:
    folds = np.array_split(test_bins, N_FOLDS)
    out = {}
    for name, m in matrices.items():
        own = usable(m)
        total = 0.0
        for test in folds:
            lo, hi = test[0], test[-1]
            train = own[(own < lo) | (own > hi)]
            fit = poisson.fit_locus(m[train], y[train])
            if fit.status.value not in ("ok", "all_zero"):
                return None
            total += deviance(y[test], m[test] @ fit.abundance)
        out[name] = total
    return out


def masked_bins(design: dienr.Design) -> float:
    """Fraction of all bins masked, over every locus."""
    masked = sum(int(design.mask(i).sum()) for i in range(design.n_loci))
    return masked / sum(design.bin_count(i) for i in range(design.n_loci))


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    differ = a != b
    return {
        "relative_change": float(a.sum() / b.sum() - 1.0),
        "locus_win_fraction": float(np.mean(a[differ] < b[differ])) if differ.any() else np.nan,
        "n_differ": float(differ.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--bigwigs", type=Path, required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--tail-from", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    annotation = read_protein_coding(args.gtf)
    tail = dienr.fit_runon_tail(annotation, signal_files(args.bigwigs, args.tail_from))
    designs = {}
    for name, (pas, with_tail) in ARMS.items():
        d = dienr.build_design(
            annotation, bin_size=BIN_SIZE, mask_start_bins=TSS_MASK, mask_end_bins=pas
        )
        designs[name] = dienr.apply_runon_tail(d, tail) if with_tail else d
    ref = designs["binary_4_4"]
    for name, d in designs.items():
        assert d.n_loci == ref.n_loci, name
        assert all(d.bin_count(i) == ref.bin_count(i) for i in range(ref.n_loci)), name
    print(f"tail: {tail.n_genes} genes, mass {tail.mass:.1f}; {ref.n_loci} loci", flush=True)
    for name, d in designs.items():
        print(f"  {name}: masked fraction {masked_bins(d):.3f}", flush=True)

    # Counts with no PAS mask: only the TSS bins are zeroed, and no arm uses those.
    y_all = dienr.count_sample(
        designs["binary_0_0"], "score", signal_files(args.bigwigs, args.score), summary="sum"
    ).values

    scores: dict[str, dict[int, float]] = {name: {} for name in ARMS}
    n_models: dict[str, list[int]] = {name: [] for name in ARMS}
    for lid in range(ref.n_loci):
        if ref.models[lid].shape[1] < 2:
            continue
        y = y_all[lid]
        test_bins = usable(ref.models[lid])
        if test_bins.size < MIN_USABLE_BINS or y[test_bins].sum() < MIN_READS:
            continue
        result = score_locus({n: d.models[lid] for n, d in designs.items()}, test_bins, y)
        if result is None:
            continue
        for name, v in result.items():
            scores[name][lid] = v
            n_models[name].append(designs[name].models[lid].shape[1])

    loci = sorted(scores[REFERENCE])
    arr = {name: np.array([scores[name][i] for i in loci]) for name in ARMS}
    report: dict = {
        "score": args.score,
        "tail_from": args.tail_from,
        "n_loci": float(len(loci)),
        "tail_training_genes": float(tail.n_genes),
        "masked_fraction": {n: masked_bins(d) for n, d in designs.items()},
        "mean_models_per_scored_locus": {n: float(np.mean(v)) for n, v in n_models.items()},
        "total_deviance": {n: float(v.sum()) for n, v in arr.items()},
    }
    for name in ARMS:
        if name != REFERENCE:
            report[f"{name} vs {REFERENCE}"] = compare(arr[name], arr[REFERENCE])
    report["runon_4_4 vs binary_4_4"] = compare(arr["runon_4_4"], arr["binary_4_4"])
    report["binary_0_0 vs binary_4_4"] = compare(arr["binary_0_0"], arr["binary_4_4"])
    report["seconds"] = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    for key, value in report.items():
        if isinstance(value, dict) and "relative_change" in value:
            print(key, {k: round(v, 4) for k, v in value.items()})
    print(f"wrote {args.output} in {report['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
