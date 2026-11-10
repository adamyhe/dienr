"""Per-locus termination scale and a data-quality gate for the run-on tail (plan 18.17).

18.14/18.15 left two residuals. Termination varies by gene, and the shared tail
is wrong where a gene's own tail is much larger or smaller (18.10: error wins
0.486 where the shared tail is > 2x an isoform's own). And on K562 the tail
helps where data are rich (top decile of loci: -12% held-out deviance, 66% of
loci) but slightly hurts the low-signal half (+1-2%, 52%). This trial asks
whether either can be fixed without fitting the test data.

Setup is 18.14's with the packaged estimator (18.15): K562 PRO-seq (GSE96869,
0 min), Ensembl GRCh37.87 protein-coding, paper masking, 250 bp bins, Poisson,
five contiguous held-out blocks per multi-model locus, tail fitted on the
*other* replicate, deviance with ``mu = max(M a, 0.01)``. Same locus filter
(>= 2 models, >= 10 usable bins, >= 20 reads).

**Arms.**

- ``binary``, ``runon`` -- as in 18.15.
- ``runon_scaled`` -- per-locus tail scale ``s``. In each fold, fit ``runon``
  on the training bins; predict the signal its tails put in a 15 kb window
  *downstream of the locus* (outside it, so never scored), ``E``; read the
  observed signal there, ``O``; set ``s = clip((O + 20) / (E + 20), 0.25, 4)``
  (20 reads of shrinkage toward 1); refit with every tail in the locus scaled
  by ``s``; score the held-out block. ``s = 1`` where any other same-strand
  transcript comes within 5 kb of that window or the window leaves the contig.
- ``runon_gated`` -- the tail only at loci whose depth (reads in usable bins)
  in the *training* replicate is at least ``T``; elsewhere the binary column.
  ``T`` is the decile of training-replicate depth (0%, 10%, ..., 90%) that
  minimizes total held-out deviance **on the training replicate**; it is then
  applied to the scored replicate. No scored data chooses ``T``.

**Decision rules, fixed before running.** Against ``runon``, in **both**
replicates:

- ``runon_scaled`` replaces it if total held-out deviance is lower AND
  per-locus held-out deviance is lower at > 50% of the loci where they differ.
- ``runon_gated`` replaces it if total held-out deviance is lower.

Also reported: the distribution of ``s`` and how many loci get one, the chosen
``T``, and both arms against ``binary``.

    uv run python scripts/k562_pergene_trial.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --score rep1 --tail-from rep2 --output build/k562/pergene_rep1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from figwig import BigWigReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_runon_check import (  # noqa: E402
    BIN_SIZE,
    MIN_READS,
    MIN_USABLE_BINS,
    N_FOLDS,
    PAPER_MASK,
    deviance,
    read_protein_coding,
    signal_files,
)

import dienr  # noqa: E402
from dienr.counts import read_loci_binned  # noqa: E402
from dienr.solvers import poisson  # noqa: E402

DOWNSTREAM_BASES = 15_000
BUFFER = 5_000
SHRINK = 20.0
S_MIN, S_MAX = 0.25, 4.0
DECILES = np.arange(0, 100, 10)


def scored_loci(base: dienr.Design, y: list[np.ndarray]) -> list[int]:
    out = []
    for lid, matrix in enumerate(base.models):
        if matrix.shape[1] < 2:
            continue
        usable = (matrix != 0).any(axis=1)
        if usable.sum() >= MIN_USABLE_BINS and y[lid][usable].sum() >= MIN_READS:
            out.append(lid)
    return out


def downstream_windows(
    base: dienr.Design, annotation: dienr.Annotation, loci: list[int], sizes: dict
) -> dict[int, tuple[str, int, int]]:
    """Isolated downstream windows ``(contig, start, n_bins)`` per locus."""
    spans = {
        key: frame.select("start", "end").to_numpy()
        for key, frame in annotation.transcripts.group_by("contig", "strand")
    }
    n = DOWNSTREAM_BASES // BIN_SIZE
    rows = base.loci
    out = {}
    for lid in loci:
        row = rows.row(lid, named=True)
        contig, strand = row["contig"], row["strand"]
        w0 = int(row["start"])
        w1 = w0 + base.bin_count(lid) * BIN_SIZE
        start = w1 if strand == "+" else w0 - DOWNSTREAM_BASES
        if start < 0 or start + DOWNSTREAM_BASES > sizes[strand].get(contig, -1):
            continue
        s = spans[(contig, strand)]
        lo, hi = start - BUFFER, start + DOWNSTREAM_BASES + BUFFER
        # Exclude the locus's own transcripts: they end inside the window.
        reach = (s[:, 0] < hi) & (s[:, 1] > lo) & ((s[:, 1] <= w0) | (s[:, 0] >= w1))
        if reach.any():
            continue
        out[lid] = (contig, start, n)
    return out


def downstream_tail_matrix(base: dienr.Design, lid: int, tail: np.ndarray) -> np.ndarray:
    """``(n_down, n_models)``: each model's tail falling in the downstream window.

    Rows run away from the locus (row 0 is adjacent to the window edge).
    """
    n_bins = base.bin_count(lid)
    n_down = DOWNSTREAM_BASES // BIN_SIZE
    plus = base.loci["strand"][lid] == "+"
    spans = base.model_spans[lid]
    out = np.zeros((n_down, spans.shape[0]))
    for k in range(spans.shape[0]):
        first, last = int(spans[k, 0]), int(spans[k, 1])
        if first < 0:
            continue
        for row in range(n_down):
            # Distance (in bins past the PAS, 0-based) of downstream row `row`.
            d = (n_bins + row) - last - 1 if plus else first - 1 - (-1 - row)
            if 0 <= d < tail.size:
                out[row, k] = tail[d]
    return out


def fold_scores(
    binary: np.ndarray,
    tail_part: np.ndarray,
    y: np.ndarray,
    down_matrix: np.ndarray | None,
    observed_down: float | None,
) -> tuple[float, float, float, list[float]]:
    """Held-out deviance for binary, runon and runon_scaled at one locus; and the s used."""
    runon = binary + tail_part
    usable = np.flatnonzero((binary != 0).any(axis=1))
    folds = np.array_split(usable, N_FOLDS)
    totals = [0.0, 0.0, 0.0]
    scales: list[float] = []
    for test in folds:
        train = np.setdiff1d(usable, test)
        fits = []
        for m in (binary, runon):
            fit = poisson.fit_locus(m[train], y[train])
            if fit.status.value not in ("ok", "all_zero"):
                return np.nan, np.nan, np.nan, []
            fits.append(fit)
        s = 1.0
        if down_matrix is not None and observed_down is not None:
            expected = float((down_matrix @ fits[1].abundance).sum())
            s = float(np.clip((observed_down + SHRINK) / (expected + SHRINK), S_MIN, S_MAX))
        scales.append(s)
        scaled = binary + s * tail_part
        fit_s = poisson.fit_locus(scaled[train], y[train])
        if fit_s.status.value not in ("ok", "all_zero"):
            return np.nan, np.nan, np.nan, []
        totals[0] += deviance(y[test], binary[test] @ fits[0].abundance)
        totals[1] += deviance(y[test], runon[test] @ fits[1].abundance)
        totals[2] += deviance(y[test], scaled[test] @ fit_s.abundance)
    return totals[0], totals[1], totals[2], scales


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
    base = dienr.build_design(annotation, bin_size=BIN_SIZE, **PAPER_MASK)
    train_signal = signal_files(args.bigwigs, args.tail_from)
    score_signal = signal_files(args.bigwigs, args.score)
    tail = dienr.fit_runon_tail(annotation, train_signal)
    runon = dienr.apply_runon_tail(base, tail)
    tail_parts = [r - b for r, b in zip(runon.models, base.models, strict=True)]

    y_train = dienr.count_sample(base, "train", train_signal, summary="sum").values
    y_score = dienr.count_sample(base, "score", score_signal, summary="sum").values

    readers = {s.strand: BigWigReader(str(s.path)) for s in score_signal}
    sizes = {k: dict(r.chrom_sizes) for k, r in readers.items()}

    def run(y: list[np.ndarray], down_obs: dict[int, float] | None, loci: list[int]) -> dict:
        res: dict[str, dict[int, float]] = {"binary": {}, "runon": {}, "runon_scaled": {}}
        scales: dict[int, list[float]] = {}
        for lid in loci:
            dm = (
                downstream_tail_matrix(base, lid, tail.values)
                if down_obs and lid in down_obs
                else None
            )
            b, r, sc, s = fold_scores(
                base.models[lid],
                tail_parts[lid],
                y[lid],
                dm,
                down_obs.get(lid) if down_obs else None,
            )
            if np.isfinite(b) and np.isfinite(r) and np.isfinite(sc):
                res["binary"][lid], res["runon"][lid], res["runon_scaled"][lid] = b, r, sc
                scales[lid] = s
        return {"scores": res, "scales": scales}

    # --- choose the gate threshold on the TRAINING replicate (binary vs runon only) ---
    train_loci = scored_loci(base, y_train)
    train_run = run(y_train, None, train_loci)["scores"]
    t_loci = sorted(train_run["binary"])
    depth_train = {
        lid: float(y_train[lid][(base.models[lid] != 0).any(axis=1)].sum())
        for lid in range(base.n_loci)
        if base.models[lid].shape[1] >= 2
    }
    d = np.array([depth_train[i] for i in t_loci])
    tb = np.array([train_run["binary"][i] for i in t_loci])
    tr = np.array([train_run["runon"][i] for i in t_loci])
    totals = {}
    for q in DECILES:
        thr = float(np.percentile(d, q)) if q else 0.0
        totals[int(q)] = float(np.where(d >= thr, tr, tb).sum())
    best_q = min(totals, key=totals.get)
    threshold = float(np.percentile(d, best_q)) if best_q else 0.0
    print(
        f"gate: decile {best_q} (depth >= {threshold:.0f} reads) chosen on {args.tail_from}",
        flush=True,
    )

    # --- score ---
    loci = scored_loci(base, y_score)
    windows = downstream_windows(base, annotation, loci, sizes)
    down_obs: dict[int, float] = {}
    for strand in ("+", "-"):
        items = [(lid, w) for lid, w in windows.items() if base.loci["strand"][lid] == strand]
        if not items:
            continue
        vals = read_loci_binned(readers[strand], [w for _, w in items], BIN_SIZE, summary="sum")
        down_obs.update({lid: float(v.sum()) for (lid, _), v in zip(items, vals, strict=True)})
    scored = run(y_score, down_obs, loci)
    res = scored["scores"]
    keep = sorted(res["binary"])
    b = np.array([res["binary"][i] for i in keep])
    r = np.array([res["runon"][i] for i in keep])
    sc = np.array([res["runon_scaled"][i] for i in keep])
    gate = np.array([depth_train.get(i, 0.0) >= threshold for i in keep])
    g = np.where(gate, r, b)
    with_window = np.array([i in down_obs for i in keep])
    s_mean = np.array([np.mean(scored["scales"][i]) for i in keep])

    report = {
        "score": args.score,
        "tail_from": args.tail_from,
        "n_loci": float(len(keep)),
        "tail_training_genes": float(tail.n_genes),
        "gate_decile": float(best_q),
        "gate_threshold_reads": threshold,
        "gate_training_totals": totals,
        "gate_fraction_applied": float(gate.mean()),
        "n_loci_with_downstream_window": float(with_window.sum()),
        "scale_quantiles": {
            str(q): float(np.percentile(s_mean[with_window], q)) for q in (5, 25, 50, 75, 95)
        }
        if with_window.any()
        else {},
        "runon vs binary": compare(r, b),
        "runon_scaled vs runon": compare(sc, r),
        "runon_scaled vs binary": compare(sc, b),
        "runon_gated vs runon": compare(g, r),
        "runon_gated vs binary": compare(g, b),
        "runon_scaled vs runon, loci with a window": compare(sc[with_window], r[with_window]),
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    for key, value in report.items():
        if isinstance(value, dict) and "relative_change" in value:
            print(key, {k: round(v, 4) for k, v in value.items()})
    print("scale quantiles", report["scale_quantiles"], "windows", int(with_window.sum()))
    print(f"wrote {args.output} in {report['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
