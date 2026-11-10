"""EXPLORATORY: why the run-on tail and the TSS mask hurt Poisson together (plan 18.24).

18.24's decomposition: on 3,460 decisive long-read loci, Poisson + tail and
Poisson + TSS mask (1, 1) each beat plain Poisson, but Poisson + tail + TSS
mask lost 5-10 points to the mask alone. This looks at the loci behind that:
those where ``tss`` (mask, no tail) calls the dominant group right and
``runon`` (mask + tail) calls it wrong, against all other evaluable loci.

Per locus it records the geometry of the truth's top group and of the group
``runon`` picks instead -- shared TSS or PAS, which one ends further
downstream -- and, in the ``runon`` design, how many tail bins of each model
were zeroed by the mask, and how many bins two models' columns differ in
before and after the tail. Same truth, matching and groups as 18.20 (groups
from the five arms below, matching 18.24's locus set).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_longread_accuracy import (  # noqa: E402
    DECISIVE,
    MIN_READS,
    UnionFind,
    fit_arm,
    read_annotation_versioned,
    read_truth,
)
from k562_runon_check import BIN_SIZE, signal_files  # noqa: E402

import dienr  # noqa: E402

ARMS = {
    "plain": ((0, 0), False, "poisson"),
    "tss": ((1, 1), False, "poisson"),
    "tail": ((0, 0), True, "poisson"),
    "runon": ((1, 1), True, "poisson"),
    "tuned_log": ((0, 0), False, "legacy"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--bigwigs", type=Path, required=True)
    parser.add_argument("--longread", type=Path, required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--tail-from", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotation = read_annotation_versioned(args.gtf)
    reads_exact, v29_exact, _ = read_truth(args.longread)
    reads = {}
    for t, n in reads_exact.items():
        reads[t.split(".")[0]] = reads.get(t.split(".")[0], 0.0) + n
    v29 = {t.split(".")[0] for t in v29_exact}
    tail = dienr.fit_runon_tail(annotation, signal_files(args.bigwigs, args.tail_from))
    signal = signal_files(args.bigwigs, args.score)

    designs, fits = {}, {}
    for name, (tss, with_tail, lik) in ARMS.items():
        d = dienr.build_design(annotation, bin_size=BIN_SIZE, mask_start_bins=tss)
        designs[name] = dienr.apply_runon_tail(d, tail) if with_tail else d
        summary = "mean" if lik == "legacy" else "sum"
        y = dienr.count_sample(designs[name], "s", signal, summary=summary).values
        fits[name] = fit_arm(designs[name], y, lik, 1.0)
        if name == "runon":
            y_runon = y
    plain_tss = dienr.build_design(annotation, bin_size=BIN_SIZE, mask_start_bins=(1, 1))

    uf = UnionFind()
    model_of = {}
    for name, d in designs.items():
        tm = d.transcript_map.select("transcript_id", "locus_id", "model_index")
        model_of[name] = dict(zip(tm["transcript_id"], tm["model_index"], strict=True))
        for _, grp in tm.group_by("locus_id", "model_index"):
            ids = grp["transcript_id"].to_list()
            for other in ids[1:]:
                uf.union(ids[0], other)
    tx = annotation.transcripts
    span = {
        t: (s, e, st)
        for t, s, e, st in tx.select("transcript_id", "start", "end", "strand").iter_rows()
    }
    tm = designs["runon"].transcript_map
    by_locus: dict[int, list[str]] = {}
    for t, lid in zip(tm["transcript_id"], tm["locus_id"], strict=True):
        by_locus.setdefault(lid, []).append(t)

    def ends(group: list[str]) -> tuple[int, int, str]:
        """(min TSS, max PAS) in transcription order, as genomic coords, and strand."""
        s, e, st = zip(*(span[t] for t in group), strict=True)
        if st[0] == "+":
            return min(s), max(e), "+"
        return max(e), min(s), "-"

    records = []
    for lid, tids in by_locus.items():
        groups: dict[str, list[str]] = {}
        for t in tids:
            groups.setdefault(uf.find(t), []).append(t)
        if len(groups) < 2 or not all(t.split(".")[0] in v29 for t in tids):
            continue
        keys = sorted(groups)
        truth = np.array([sum(reads.get(t.split(".")[0], 0.0) for t in groups[k]) for k in keys])
        if truth.sum() < MIN_READS or not all(lid in fits[n] for n in ARMS):
            continue
        order = np.sort(truth)[::-1]
        if not (order[0] >= DECISIVE * order[1] and order[0] > 0):
            continue
        top = int(np.argmax(truth))
        rec = {"locus_id": int(lid), "n_groups": len(keys), "strand": span[tids[0]][2]}
        picks = {}
        for name in ARMS:
            a = fits[name][lid]
            est = np.array(
                [sum(a[i] for i in {model_of[name][t] for t in groups[k]}) for k in keys]
            )
            winners = np.flatnonzero(est == est.max())
            rec[name] = bool(winners.size == 1 and winners[0] == top and est.max() > 0)
            picks[name] = int(np.argmax(est))
        # Geometry of the truth's top group vs the group runon picks.
        tss_t, pas_t, strand = ends(groups[keys[top]])
        tss_r, pas_r, _ = ends(groups[keys[picks["runon"]]])
        sign = 1 if strand == "+" else -1
        rec["pick_same_tss"] = tss_t == tss_r
        rec["pick_same_pas"] = pas_t == pas_r
        rec["pick_pas_downstream_kb"] = sign * (pas_r - pas_t) / 1000.0
        rec["pick_tss_downstream_kb"] = sign * (tss_r - tss_t) / 1000.0
        # Does the window clip the truth's tail? The window ends at the locus's
        # most downstream PAS, so that group's tail is cut off entirely.
        loc = designs["runon"].loci.row(lid, named=True)
        window_end = loc["end"] if strand == "+" else loc["start"]
        rec["truth_pas_at_window_end"] = pas_t == window_end
        rec["pick_pas_at_window_end"] = pas_r == window_end
        rec["truth_pas_to_window_end_kb"] = sign * (window_end - pas_t) / 1000.0
        rt = designs["runon"].models[lid]
        rb = plain_tss.models[lid]
        tm_r = model_of["runon"]
        rec["truth_tail_bins"] = int(
            ((rt - rb) > 0)[:, sorted({tm_r[t] for t in groups[keys[top]]})].sum()
        )
        rec["pick_tail_bins"] = int(
            ((rt - rb) > 0)[:, sorted({tm_r[t] for t in groups[keys[picks["runon"]]]})].sum()
        )
        # Tail bins lost to the mask, and how distinct the columns stay.
        m_tail = designs["runon"].models[lid]
        m_bin = plain_tss.models[lid]
        mask = designs["runon"].masks[lid]
        added = (m_tail - m_bin) > 0
        rec["tail_bins"] = int(added.sum())
        rec["masked_bins"] = int(mask.sum())
        rec["n_bins"] = int(mask.size)
        if m_tail.shape[1] >= 2:
            corr = np.corrcoef(m_tail.T)
            corr_b = np.corrcoef(m_bin.T)
            iu = np.triu_indices(m_tail.shape[1], 1)
            rec["max_col_corr_tail"] = float(np.nanmax(corr[iu]))
            rec["max_col_corr_binary"] = float(np.nanmax(corr_b[iu]))
            rec["cond_tail"] = float(np.linalg.cond(m_tail))
            rec["cond_binary"] = float(np.linalg.cond(m_bin))
        y = y_runon[lid]
        rec["reads"] = float(y.sum())
        records.append(rec)

    frame = pl.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.output)
    bad = frame.filter(pl.col("tss") & ~pl.col("runon"))
    rest = frame.filter(~(pl.col("tss") & ~pl.col("runon")))
    good = frame.filter(~pl.col("tss") & pl.col("runon"))

    def describe(f: pl.DataFrame) -> dict:
        if f.height == 0:
            return {"n": 0}
        return {
            "n": f.height,
            "pick_same_tss": float(f["pick_same_tss"].mean()),
            "pick_same_pas": float(f["pick_same_pas"].mean()),
            "pick_pas_downstream": float((f["pick_pas_downstream_kb"] > 0).mean()),
            "pick_pas_upstream": float((f["pick_pas_downstream_kb"] < 0).mean()),
            "median_pick_pas_shift_kb": float(f["pick_pas_downstream_kb"].median()),
            "median_max_col_corr_tail": float(f["max_col_corr_tail"].median()),
            "median_max_col_corr_binary": float(f["max_col_corr_binary"].median()),
            "median_log10_cond_tail": float(np.log10(f["cond_tail"]).median()),
            "median_log10_cond_binary": float(np.log10(f["cond_binary"]).median()),
            "median_groups": float(f["n_groups"].median()),
            "truth_pas_at_window_end": float(f["truth_pas_at_window_end"].mean()),
            "pick_pas_at_window_end": float(f["pick_pas_at_window_end"].mean()),
            "median_truth_tail_bins": float(f["truth_tail_bins"].median()),
            "median_pick_tail_bins": float(f["pick_tail_bins"].median()),
            "plain_right": float(f["plain"].mean()),
            "tail_right": float(f["tail"].mean()),
            "tuned_log_right": float(f["tuned_log"].mean()),
        }

    report = {
        "loci": frame.height,
        "accuracy": {n: float(frame[n].mean()) for n in ARMS},
        "tss right, runon wrong": describe(bad),
        "tss wrong, runon right": describe(good),
        "all others": describe(rest),
    }
    print(json.dumps(report, indent=1))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
