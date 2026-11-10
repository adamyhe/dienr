"""Is the TSS mask still the right size? (plan 18.19)

18.18 found the paper's 4-bin PAS mask throws away informative bins. The TSS
mask -- ``mask_start_bins=(1, 1)``, one 250 bp bin either side of every TSS --
has the opposite rationale: the promoter-proximal pause peak there is far above
gene-body density, a 0/1 column can't explain it, and fitting it would inflate
the abundance of whichever models start there. So a smaller mask could hurt
and a larger one could help. This measures which.

Setup is 18.18's: K562 PRO-seq (GSE96869, 0 min), Ensembl GRCh37.87
protein-coding, 250 bp bins, Poisson, packaged run-on tail fitted on the
*other* replicate and applied to every arm, deviance with
``mu = max(M a, 0.01)``. The PAS mask is fixed at the one 18.18's rule
selected (``--pas-mask``).

**Arms** (TSS mask ``(inside, outside)`` bins; all with the tail):
``tss_0_0``, ``tss_1_0``, ``tss_1_1`` (the current default, reference),
``tss_2_2``, ``tss_4_4``.

**Comparable scoring** as in 18.18: every arm is scored on the same test bins,
those usable under the *largest* mask (``tss_4_4``), in 5 contiguous blocks;
each arm trains on its own usable bins outside the block's span. A smaller mask
only adds training bins. Counts are read once with no masking at all.

A locus is scored if it has >= 2 models in ``tss_4_4``, >= 10 test bins and
>= 20 reads in them, and every arm's fit succeeds on every fold.

**Decision rule, fixed before running.** Against ``tss_1_1``, another mask
qualifies if, in **both** replicates (tail from the other one):

1. lower total held-out deviance, AND
2. lower per-locus held-out deviance at > 50% of the loci where they differ.

If several qualify, the new default is the one with the lowest total deviance
averaged over the two replicates' relative changes. Otherwise ``(1, 1)``
stays.

**Caveat.** Held-out deviance scores signal prediction, not isoform calls. The
pause peak is also where alternative TSSs differ most, so a mask that fits
better here could still change which isoform looks dominant; that needs a
truth set (plan 20, item 1).

    uv run python scripts/k562_tssmask_trial.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --score rep1 --tail-from rep2 --pas-mask 0 0 \\
        --output build/k562/tssmask_rep1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_pasmask_trial import compare, masked_bins, score_locus, usable  # noqa: E402
from k562_runon_check import (  # noqa: E402
    BIN_SIZE,
    MIN_READS,
    MIN_USABLE_BINS,
    read_protein_coding,
    signal_files,
)

import dienr  # noqa: E402

REFERENCE = "tss_1_1"
LARGEST = "tss_4_4"
ARMS: dict[str, tuple[int, int]] = {
    "tss_0_0": (0, 0),
    "tss_1_0": (1, 0),
    "tss_1_1": (1, 1),
    "tss_2_2": (2, 2),
    "tss_4_4": (4, 4),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--bigwigs", type=Path, required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--tail-from", required=True)
    parser.add_argument("--pas-mask", type=int, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    pas = tuple(args.pas_mask)

    annotation = read_protein_coding(args.gtf)
    tail = dienr.fit_runon_tail(annotation, signal_files(args.bigwigs, args.tail_from))
    designs = {
        name: dienr.apply_runon_tail(
            dienr.build_design(
                annotation, bin_size=BIN_SIZE, mask_start_bins=tss, mask_end_bins=pas
            ),
            tail,
        )
        for name, tss in ARMS.items()
    }
    ref = designs[LARGEST]
    for name, d in designs.items():
        assert d.n_loci == ref.n_loci, name
        assert all(d.bin_count(i) == ref.bin_count(i) for i in range(ref.n_loci)), name
    print(f"tail: {tail.n_genes} genes; {ref.n_loci} loci; PAS mask {pas}", flush=True)
    for name, d in designs.items():
        print(f"  {name}: masked fraction {masked_bins(d):.3f}", flush=True)

    unmasked = dienr.build_design(annotation, bin_size=BIN_SIZE)
    y_all = dienr.count_sample(
        unmasked, "score", signal_files(args.bigwigs, args.score), summary="sum"
    ).values

    scores: dict[str, dict[int, float]] = {name: {} for name in ARMS}
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

    loci = sorted(scores[REFERENCE])
    arr = {name: np.array([scores[name][i] for i in loci]) for name in ARMS}
    report: dict = {
        "score": args.score,
        "tail_from": args.tail_from,
        "pas_mask": list(pas),
        "n_loci": float(len(loci)),
        "masked_fraction": {n: masked_bins(d) for n, d in designs.items()},
        "total_deviance": {n: float(v.sum()) for n, v in arr.items()},
    }
    for name in ARMS:
        if name != REFERENCE:
            report[f"{name} vs {REFERENCE}"] = compare(arr[name], arr[REFERENCE])
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
