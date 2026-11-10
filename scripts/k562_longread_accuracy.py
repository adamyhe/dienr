"""Dominant-isoform accuracy on real data, against K562 long reads (plan 18.20).

Every design decision since 18.14 -- the run-on tail (18.14/18.15), dropping
the PAS mask (18.18), keeping the TSS mask (18.19) -- was judged by held-out
deviance, which scores how well a design *predicts signal*, not whether it
calls the right isoform. This asks the second question of real data.

**Truth.** ENCODE K562 PacBio Sequel II long-read RNA-seq, ENCSR983KDL
(Mortazavi lab), TALON transcript quantifications against GENCODE v29
(ENCFF302ZXZ rep1, ENCFF448BDX rep2), pooled. Only reads assigned to
*Known* transcripts are used. They are matched to our Ensembl GRCh37.87
protein-coding transcripts by **Ensembl transcript ID** (stable across
assemblies, so no liftover). A transcript annotated in v29 with no long reads
has truth zero; a GRCh37 transcript whose ID is absent from v29 has unknown
truth, and its locus is dropped. Matching on ID *and version* (identical
structure) would be stricter, but only 21k of 81.6k GRCh37.87 transcripts keep
their version in v29 (versions bump for UTR edits too), against 74k by ID,
which leaves too few loci; the exact-version subset is reported as a
sensitivity analysis. (Decided from match coverage alone, before any fit.)

**What is compared.** DENR estimates pre-RNA *span* models, and transcripts
that a design can't tell apart share one model -- which ones depends on the
masks. So scoring is on **groups**: transcripts linked by sharing a model in
*any* arm (union-find), which every arm resolves. An arm's group abundance is
the sum of its distinct models' abundances in the group; the truth is the sum
of its transcripts' long reads.

**Arms** (all with K562 PRO-seq GSE96869 0 min, 250 bp bins, TSS mask
``(1, 1)``; tail fitted on the other PRO-seq replicate):

- ``denr_default`` -- PAS mask ``(4, 4)``, no tail, legacy log least squares,
  ``mean`` summary: R DENR with the paper's masking. (Not DENR's function
  defaults, which mask nothing; see 18.22.)
- ``binary_poisson`` -- PAS ``(4, 4)``, no tail, Poisson, ``sum``.
- ``runon_4_4`` -- PAS ``(4, 4)``, tail, Poisson.
- ``binary_0_0`` -- no PAS mask, no tail, Poisson.
- ``recommended`` -- no PAS mask, tail, Poisson (18.19's configuration).

**Evaluable loci**: >= 2 groups; every transcript's ID.version in v29;
>= 20 long reads in the locus; the top group has >= 2x the reads of the
second (a decisive truth); every arm's fit has status ``ok``. A call is
correct if the arm's largest group abundance is unique and belongs to the
truth's top group.

**Decision rule, fixed before running.** ``recommended`` is supported on
isoform accuracy if, on **both** PRO-seq replicates, against **each** of
``denr_default`` and ``binary_poisson``:

1. higher dominant-group accuracy, AND
2. it wins > 50% of the loci where exactly one of the two is correct, with a
   two-sided sign test p < 0.05.

If it fails, the held-out-deviance recommendations stand but are flagged as
unconfirmed on accuracy. Also reported, not part of the rule: ``runon_4_4``
and ``binary_0_0`` (which change decomposes the effect), within-locus
Spearman between group abundance and long reads (average ranks), the
1/n_groups chance level, the rule's comparisons at 1.5x and 3x
decisiveness, and the primary comparison restricted to loci whose transcripts
all match v29 by ID *and* version.

**Caveat.** Long reads measure steady-state mature RNA; nascent transcription
differs from it by isoform stability. That bias is shared by every arm, so
paired comparisons rank designs fairly, but absolute accuracies are agreement
with steady state, not accuracy against nascent truth.

    uv run python scripts/k562_longread_accuracy.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --longread build/k562/longread --score rep1 --tail-from rep2 \\
        --output build/k562/longread_rep1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_runon_check import BIN_SIZE, CONTIGS, signal_files  # noqa: E402

import dienr  # noqa: E402
from dienr.solvers import legacy, poisson  # noqa: E402

LONGREAD_FILES = ("ENCFF302ZXZ.tsv", "ENCFF448BDX.tsv")
TSS_MASK = (1, 1)
ARMS: dict[str, dict] = {
    "denr_default": {"pas": (4, 4), "tail": False, "likelihood": "legacy"},
    "binary_poisson": {"pas": (4, 4), "tail": False, "likelihood": "poisson"},
    "runon_4_4": {"pas": (4, 4), "tail": True, "likelihood": "poisson"},
    "binary_0_0": {"pas": (0, 0), "tail": False, "likelihood": "poisson"},
    "recommended": {"pas": (0, 0), "tail": True, "likelihood": "poisson"},
}
RULE_REFERENCES = ("denr_default", "binary_poisson")
CANDIDATES = {"recommended": RULE_REFERENCES}
EXTRA_PAIRS = (("runon_4_4", "binary_poisson"), ("binary_0_0", "binary_poisson"))
MIN_READS = 20
DECISIVE = 2.0
SENSITIVITY = (1.5, 3.0)


def read_annotation_versioned(gtf: Path) -> dienr.Annotation:
    """GRCh37.87 protein-coding transcripts, ID.version, contigs N -> chrN."""
    frame = (
        pl.scan_csv(
            gtf,
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=["contig", "src", "feature", "start", "end", "s", "strand", "f", "attr"],
            schema_overrides={"contig": pl.String, "start": pl.Int64, "end": pl.Int64},
        )
        .filter(
            (pl.col("feature") == "transcript")
            & pl.col("contig").is_in(CONTIGS)
            & pl.col("attr").str.contains('transcript_biotype "protein_coding"')
        )
        .select(
            (
                pl.col("attr").str.extract(r'transcript_id "([^"]+)"')
                + "."
                + pl.col("attr").str.extract(r'transcript_version "([^"]+)"')
            ).alias("transcript_id"),
            pl.col("attr").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
            ("chr" + pl.col("contig")).alias("contig"),
            (pl.col("start") - 1).alias("start"),
            pl.col("end"),
            pl.col("strand"),
        )
        .collect()
    )
    return dienr.Annotation.from_frame(frame, source=f"{gtf} (protein_coding, ID.version)")


def read_truth(directory: Path) -> tuple[dict[str, float], set[str], dict]:
    reads: dict[str, float] = {}
    info = {}
    for name in LONGREAD_FILES:
        frame = pl.read_csv(directory / name, separator="\t", infer_schema_length=0)
        column = frame.columns[-1]
        known = frame.filter(pl.col("transcript_novelty") == "Known").select(
            "annot_transcript_id", pl.col(column).cast(pl.Float64)
        )
        info[name] = {"known_reads": float(known[column].sum()), "column": column}
        for tid, n in known.iter_rows():
            reads[tid] = reads.get(tid, 0.0) + n
    v29 = set((directory / "v29_transcripts.txt").read_text().split())
    return reads, v29, info


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def fit_arm(
    design: dienr.Design,
    y: list[np.ndarray],
    likelihood: str,
    log_offset: float = legacy.LOG_OFFSET,
    transform: str = "log",
) -> dict[int, np.ndarray]:
    out = {}
    for lid, m in enumerate(design.models):
        if m.shape[1] < 2:
            continue
        if likelihood == "legacy":
            fit = legacy.fit_locus(m, y[lid], transform=transform, log_offset=log_offset)
        else:
            fit = poisson.fit_locus(m, y[lid])
        if fit.status.value == "ok":
            out[lid] = fit.abundance
    return out


def sign(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    n = only_a + only_b
    return {
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
        "only_a": float(only_a),
        "only_b": float(only_b),
        "win_share_a": only_a / n if n else float("nan"),
        "p_two_sided": float(binomtest(only_a, n).pvalue) if n else float("nan"),
    }


def main(
    argv: list[str] | None = None,
    *,
    arms: dict[str, dict] = ARMS,
    candidates: dict[str, tuple[str, ...]] = CANDIDATES,
    extra_pairs: tuple[tuple[str, str], ...] = EXTRA_PAIRS,
    alpha: float = 0.05,
    description: str | None = None,
) -> int:
    """Run the comparison. Other trials (18.21) reuse it with their own arms and rule."""
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--bigwigs", type=Path, required=True)
    parser.add_argument("--longread", type=Path, required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--tail-from", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-locus",
        type=Path,
        default=None,
        help="also write each decisive (2x) locus's contig and per-arm correctness",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()

    annotation = read_annotation_versioned(args.gtf)
    reads_exact, v29_exact, lr_info = read_truth(args.longread)
    reads_id: dict[str, float] = {}
    for tid, n in reads_exact.items():
        reads_id[tid.split(".")[0]] = reads_id.get(tid.split(".")[0], 0.0) + n
    v29_id = {t.split(".")[0] for t in v29_exact}
    tail = dienr.fit_runon_tail(annotation, signal_files(args.bigwigs, args.tail_from))
    signal = signal_files(args.bigwigs, args.score)

    designs, fits = {}, {}
    for name, arm in arms.items():
        d = dienr.build_design(
            annotation,
            bin_size=arm.get("bin", BIN_SIZE),
            distance=arm.get("distance"),
            mask_start_bins=arm.get("tss", TSS_MASK),
            mask_end_bins=arm["pas"],
        )
        designs[name] = (
            dienr.apply_runon_tail(d, tail, extend=arm.get("extend", True)) if arm["tail"] else d
        )
    # Locus ids index every arm's fits, so every arm must group loci the same way
    # (bin size alone doesn't change grouping; `distance` does).
    first = designs[next(iter(arms))].loci
    for name, d in designs.items():
        assert d.loci.select("locus_id", "start", "end").equals(
            first.select("locus_id", "start", "end")
        ), f"{name} groups loci differently; set the same 'distance' on every arm"
    for name, arm in arms.items():
        summary = "mean" if arm["likelihood"] == "legacy" else "sum"
        y = dienr.count_sample(designs[name], "s", signal, summary=summary).values
        fits[name] = fit_arm(
            designs[name],
            y,
            arm["likelihood"],
            arm.get("log_offset", legacy.LOG_OFFSET),
            arm.get("transform", "log"),
        )
        print(f"{name}: {len(fits[name])} multi-model loci fitted ok", flush=True)

    # Groups: transcripts sharing a model in any arm.
    uf = UnionFind()
    maps = {}
    for name, d in designs.items():
        tm = d.transcript_map.select("transcript_id", "locus_id", "model_index")
        maps[name] = tm
        for (_, _), grp in tm.group_by("locus_id", "model_index"):
            ids = grp["transcript_id"].to_list()
            for other in ids[1:]:
                uf.union(ids[0], other)
    base = maps[next(iter(arms))]  # every design holds the same transcripts
    locus_of = dict(zip(base["transcript_id"], base["locus_id"], strict=True))
    by_locus: dict[int, list[str]] = {}
    for tid, lid in locus_of.items():
        by_locus.setdefault(lid, []).append(tid)
    model_of = {
        name: dict(zip(tm["transcript_id"], tm["model_index"], strict=True))
        for name, tm in maps.items()
    }

    def build_rows(exact: bool) -> tuple[list, dict]:
        """Per evaluable locus: truth per group, and each arm's estimate per group."""
        known = v29_exact if exact else v29_id
        reads = reads_exact if exact else reads_id

        def key(t: str) -> str:
            return t if exact else t.split(".")[0]

        funnel = {"loci": len(by_locus), ">=2 groups": 0, "all in v29": 0, "reads": 0, "fitted": 0}
        rows = []
        for lid, tids in by_locus.items():
            groups: dict[str, list[str]] = {}
            for t in tids:
                groups.setdefault(uf.find(t), []).append(t)
            if len(groups) < 2:
                continue
            funnel[">=2 groups"] += 1
            if not all(key(t) in known for t in tids):
                continue
            funnel["all in v29"] += 1
            keys = sorted(groups)
            truth = np.array([sum(reads.get(key(t), 0.0) for t in groups[k]) for k in keys])
            if truth.sum() < MIN_READS:
                continue
            funnel["reads"] += 1
            if not all(lid in fits[name] for name in arms):
                continue
            funnel["fitted"] += 1
            est = {}
            for name in arms:
                a = fits[name][lid]
                est[name] = np.array(
                    [sum(a[i] for i in {model_of[name][t] for t in groups[k]}) for k in keys]
                )
            rows.append((lid, truth, est))
        return rows, funnel

    rows, funnel = build_rows(exact=False)
    rows_exact, funnel_exact = build_rows(exact=True)
    print("funnel", funnel, "exact-version funnel", funnel_exact, flush=True)

    def evaluate(decisive: float, rows: list = rows) -> dict:
        keep = []
        for lid, truth, est in rows:
            order = np.sort(truth)[::-1]
            if order[0] >= decisive * order[1] and order[0] > 0:
                keep.append((lid, truth, est))
        correct = {name: [] for name in arms}
        rho = {name: [] for name in arms}
        chance = []
        for _, truth, est in keep:
            top = int(np.argmax(truth))
            chance.append(1.0 / truth.size)
            for name in arms:
                e = est[name]
                winners = np.flatnonzero(e == e.max())
                correct[name].append(winners.size == 1 and winners[0] == top and e.max() > 0)
                if np.ptp(e) > 0 and np.ptp(truth) > 0:
                    rho[name].append(spearmanr(e, truth).statistic)
        c = {name: np.array(v, dtype=bool) for name, v in correct.items()}
        out = {
            "n_loci": float(len(keep)),
            "chance": float(np.mean(chance)) if chance else float("nan"),
            "accuracy": {name: float(v.mean()) for name, v in c.items()},
            "mean_spearman": {name: float(np.nanmean(v)) for name, v in rho.items()},
        }
        for focal in candidates:
            for other in arms:
                if other != focal:
                    out[f"{focal} vs {other}"] = sign(c[focal], c[other])
        for a, b in extra_pairs:
            out[f"{a} vs {b}"] = sign(c[a], c[b])
        return out

    report = {
        "score": args.score,
        "tail_from": args.tail_from,
        "longread": lr_info,
        "funnel": funnel,
        "funnel_exact_version": funnel_exact,
        "primary": evaluate(DECISIVE),
        "sensitivity": {str(d): evaluate(d) for d in SENSITIVITY},
        "exact_version": evaluate(DECISIVE, rows_exact),
    }
    primary = report["primary"]
    passes = {
        focal: all(
            primary[f"{focal} vs {ref}"]["accuracy_a"] > primary[f"{focal} vs {ref}"]["accuracy_b"]
            and primary[f"{focal} vs {ref}"]["win_share_a"] > 0.5
            and primary[f"{focal} vs {ref}"]["p_two_sided"] < alpha
            for ref in refs
        )
        for focal, refs in candidates.items()
    }
    report["rule_alpha"] = alpha
    report["rule_passes_by_candidate"] = passes
    # A trial with no candidates (18.23) decides elsewhere, from --per-locus.
    report["rule_passes_this_replicate"] = all(passes.values()) if passes else None
    if args.per_locus is not None:
        loci_table = designs[next(iter(arms))].loci
        contig_of = dict(zip(loci_table["locus_id"], loci_table["contig"], strict=True))
        records = []
        for lid, truth, est in rows:
            order = np.sort(truth)[::-1]
            if not (order[0] >= DECISIVE * order[1] and order[0] > 0):
                continue
            top = int(np.argmax(truth))
            record = {"locus_id": int(lid), "contig": contig_of[lid], "n_groups": truth.size}
            for name in arms:
                e = est[name]
                winners = np.flatnonzero(e == e.max())
                record[name] = bool(winners.size == 1 and winners[0] == top and e.max() > 0)
            records.append(record)
        pl.DataFrame(records).write_parquet(args.per_locus)
    report["seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({k: report[k] for k in ("funnel", "rule_passes_by_candidate")}))
    print("accuracy", {k: round(v, 4) for k, v in primary["accuracy"].items()})
    print("chance", round(primary["chance"], 4), "loci", primary["n_loci"])
    for key, value in primary.items():
        if isinstance(value, dict) and "win_share_a" in value:
            print(key, {k: round(v, 4) for k, v in value.items()})
    print(f"wrote {args.output} in {report['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
