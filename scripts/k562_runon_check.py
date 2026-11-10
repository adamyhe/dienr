"""Real-data check of the run-on tail on K562 PRO-seq (plan 18.14).

Everything in 18.9-18.13 is simulator-only, and nascentRNASim truncates run-on
at 6 kb. This asks two things of real data, with no ground truth needed:

**(a) What does the real tail look like?** The shared tail ``r(d)`` (18.10's
ratio-of-sums estimator) fitted out to 15 kb past the PAS from isolated,
expressed single-isoform genes: its shape, and how much of its mass lies past
the simulator's 6 kb.

**(b) Does the run-on design predict held-out real signal better?** At
multi-model loci, contiguous blocks of bins are held out (5 folds, as in Gate
B); each design is fitted with Poisson on the rest and scored by Poisson
deviance on the held-out block. The comparison is fair by construction: the
run-on and binary designs fit exactly the same bins (18.12) and the tail adds
no per-locus parameter -- it is fixed before fitting. The tail and the shape
profile are fitted on one replicate and scored on the *other*, so no data is
reused.

Data: GEO GSE96869 (Dukler et al. 2017; the data nascentRNASim's archetypes
were built from), K562 0 min, replicates 1 and 2, `primary` 3'-end bigWigs,
hg19. Annotation: Ensembl GRCh37.87 protein-coding transcripts on chr1-22 and
X, contigs renamed explicitly ``N -> chrN``. Paper masking (1 TSS bin, 4 PAS
bins), 250 bp bins, `summary="sum"`.

Arms: ``binary``, ``runon`` (real tail, 15 kb, other replicate), ``runon6``
(the same tail cut at 6 kb, as simulated), ``shape`` (dienr profile, other
replicate), all Poisson.

A locus is scored if it has >= 2 models, >= 10 usable bins and >= 20 reads in
them, and every arm's fit succeeds on every fold. Held-out deviance uses
``mu = max(M a, 0.01)`` for every arm alike, so a bin no training fit reaches
is finite and penalized identically.

**Decision rule, fixed before running.** Run-on is supported on real data only
if, in **both** replicates (tail from the other one), ``runon`` vs ``binary``:

1. lower total held-out deviance summed over scored loci, AND
2. lower per-locus held-out deviance at > 50% of loci (exact ties excluded).

Also reported, not part of the rule: ``runon6`` and ``shape`` against
``binary``; and the tail's shape and its mass beyond 6 kb.

    uv run python scripts/k562_runon_check.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --score rep1 --tail-from rep2 --output build/k562/score_rep1.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from figwig import BigWigReader

import dienr
from dienr.counts import read_loci_binned
from dienr.shape import ShapeConfig, apply_shape_profile, fit_shape_profile
from dienr.solvers import poisson

BIN_SIZE = 250
PAPER_MASK = {"mask_start_bins": (1, 1), "mask_end_bins": (4, 4)}
TAIL_BINS = 60  # 15 kb
SIM_TAIL_BINS = 24  # nascentRNASim's 6 kb
BUFFER = 5_000  # isolation margin beyond the tail window
MIN_BODY_PER_BIN = 2.0
N_FOLDS = 5
MIN_USABLE_BINS, MIN_READS = 10, 20.0
MU_FLOOR = 0.01
CONTIGS = [str(i) for i in range(1, 23)] + ["X"]


def read_protein_coding(gtf: Path) -> dienr.Annotation:
    frame = (
        pl.scan_csv(
            gtf,
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=[
                "contig",
                "src",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attr",
            ],
            schema_overrides={"contig": pl.String, "start": pl.Int64, "end": pl.Int64},
        )
        .filter(
            (pl.col("feature") == "transcript")
            & pl.col("contig").is_in(CONTIGS)
            & pl.col("attr").str.contains('transcript_biotype "protein_coding"')
        )
        .select(
            pl.col("attr").str.extract(r'transcript_id "([^"]+)"').alias("transcript_id"),
            pl.col("attr").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
            # Explicit, recorded rename: Ensembl "N" -> UCSC "chrN".
            ("chr" + pl.col("contig")).alias("contig"),
            (pl.col("start") - 1).alias("start"),
            pl.col("end"),
            pl.col("strand"),
        )
        .collect()
    )
    return dienr.Annotation.from_frame(frame, source=f"{gtf} (protein_coding, N->chrN)")


def signal_files(directory: Path, rep: str) -> tuple[dienr.SignalFile, dienr.SignalFile]:
    return (
        dienr.SignalFile(path=str(directory / f"k562_0min_{rep}_plus.bw"), strand="+"),
        dienr.SignalFile(path=str(directory / f"k562_0min_{rep}_minus.bw"), strand="-"),
    )


def fit_real_tail(
    base: dienr.Design,
    counts: dienr.SampleCounts,
    annotation: dienr.Annotation,
    signal: tuple[dienr.SignalFile, dienr.SignalFile],
) -> tuple[np.ndarray, int]:
    """18.10's ratio-of-sums tail, to TAIL_BINS, from isolated single-isoform genes.

    Isolation is same-strand only: PRO-seq is strand-specific, so an
    opposite-strand gene cannot put reads in the tail window.
    """
    span = TAIL_BINS * BIN_SIZE
    spans = {
        key: frame.select("start", "end").to_numpy()
        for key, frame in annotation.transcripts.group_by("contig", "strand")
    }
    readers = {s.strand: BigWigReader(str(s.path)) for s in signal}
    sizes = {s.strand: readers[s.strand].chrom_sizes for s in signal}
    windows: dict[str, list] = {"+": [], "-": []}
    bodies: dict[str, list] = {"+": [], "-": []}
    for row in base.loci.filter(pl.col("n_transcripts") == 1).iter_rows(named=True):
        lid = row["locus_id"]
        support = (base.models[lid][:, 0] != 0) & ~base.mask(lid)
        if not support.any():
            continue
        body = float(counts.values[lid][support].mean())
        if body < MIN_BODY_PER_BIN:
            continue
        first, last = (int(x) for x in base.model_spans[lid][0])
        origin, strand, contig = int(row["start"]), row["strand"], row["contig"]
        tail_start = (
            origin + (last + 1) * BIN_SIZE if strand == "+" else origin + first * BIN_SIZE - span
        )
        if tail_start < 0 or tail_start + span > sizes[strand].get(contig, 0):
            continue
        s = spans[(contig, strand)]
        reach = (s[:, 0] - BUFFER < tail_start + span) & (s[:, 1] + BUFFER > tail_start)
        own = (s[:, 0] == row["start"]) & (s[:, 1] == row["end"])
        if (reach & ~own).any():
            continue
        windows[strand].append((contig, tail_start, TAIL_BINS))
        bodies[strand].append(body)
    numerator, denominator, n = np.zeros(TAIL_BINS), 0.0, 0
    for strand in ("+", "-"):
        if not windows[strand]:
            continue
        tails = read_loci_binned(readers[strand], windows[strand], BIN_SIZE, summary="sum")
        for tail, body in zip(tails, bodies[strand], strict=True):
            tail = np.abs(tail)  # minus-strand bigWigs are negative
            numerator += tail if strand == "+" else tail[::-1]
            denominator += body
            n += 1
    if n == 0:
        raise ValueError("no isolated, expressed single-isoform gene to fit a tail from")
    return numerator / denominator, n


def with_tail(base: dienr.Design, tail: np.ndarray, label: str) -> dienr.Design:
    strands = base.loci["strand"].to_list()
    models = []
    for lid, matrix in enumerate(base.models):
        unmasked = ~base.mask(lid)
        plus = strands[lid] == "+"
        out = matrix.astype(float).copy()
        for k in range(matrix.shape[1]):
            first, last = (int(x) for x in base.model_spans[lid][k])
            if first < 0:
                continue
            if plus:
                bins = np.arange(last + 1, min(last + 1 + tail.size, matrix.shape[0]))
            else:
                bins = np.arange(first - 1, max(first - 1 - tail.size, -1), -1)
            out[bins, k] += tail[: bins.size]
        models.append(np.ascontiguousarray(out * unmasked[:, None]))
    return replace(base, models=models, digest=f"{base.digest}+{label}", shape_digest=label)


def deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, MU_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def held_out(designs: dict[str, dienr.Design], y_by_locus: list[np.ndarray]) -> dict:
    """Per-locus summed held-out deviance for every arm, on loci every arm scores."""
    base = designs["binary"]
    out: dict[str, dict[int, float]] = {name: {} for name in designs}
    for lid, matrix in enumerate(base.models):
        if matrix.shape[1] < 2:
            continue
        y = y_by_locus[lid]
        usable = np.flatnonzero((matrix != 0).any(axis=1))
        if usable.size < MIN_USABLE_BINS or y[usable].sum() < MIN_READS:
            continue
        folds = np.array_split(usable, N_FOLDS)
        scores = {}
        for name, design in designs.items():
            m = design.models[lid]
            total = 0.0
            for test in folds:
                train = np.setdiff1d(usable, test)
                fit = poisson.fit_locus(m[train], y[train])
                if fit.status.value not in ("ok", "all_zero"):
                    total = np.nan
                    break
                total += deviance(y[test], m[test] @ fit.abundance)
            scores[name] = total
        if all(np.isfinite(v) for v in scores.values()):
            for name, v in scores.items():
                out[name][lid] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--bigwigs", type=Path, required=True)
    parser.add_argument("--score", required=True, help="replicate to score, e.g. rep1")
    parser.add_argument("--tail-from", required=True, help="replicate to fit tail/profile on")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    annotation = read_protein_coding(args.gtf)
    base = dienr.build_design(annotation, bin_size=BIN_SIZE, **PAPER_MASK)
    print(f"annotation {len(annotation)} transcripts; {base.loci.height} loci", flush=True)

    train_signal = signal_files(args.bigwigs, args.tail_from)
    train_counts = dienr.count_sample(base, "train", train_signal, summary="sum")
    tail, n_training = fit_real_tail(base, train_counts, annotation, train_signal)
    profile = fit_shape_profile(annotation, train_signal, config=ShapeConfig())
    print(f"tail from {n_training} isolated genes; profile from {profile.n_genes}", flush=True)

    designs = {
        "binary": base,
        "runon": with_tail(base, tail, "runon15"),
        "runon6": with_tail(base, tail[:SIM_TAIL_BINS], "runon6"),
        "shape": apply_shape_profile(base, profile),
    }
    score_counts = dienr.count_sample(
        base, "score", signal_files(args.bigwigs, args.score), summary="sum"
    )
    scores = held_out(designs, score_counts.values)
    loci = sorted(scores["binary"])

    def compare(arm: str, ref: str = "binary") -> dict[str, float]:
        a = np.array([scores[arm][i] for i in loci])
        b = np.array([scores[ref][i] for i in loci])
        differ = a != b
        return {
            "total_deviance": float(a.sum()),
            "reference_total_deviance": float(b.sum()),
            "relative_change": float(a.sum() / b.sum() - 1.0),
            "locus_win_fraction": float(np.mean(a[differ] < b[differ])) if differ.any() else np.nan,
            "n_ties": float((~differ).sum()),
        }

    report = {
        "score": args.score,
        "tail_from": args.tail_from,
        "n_loci_scored": float(len(loci)),
        "tail": [float(x) for x in tail],
        "tail_training_genes": float(n_training),
        "tail_mass": float(tail.sum()),
        "tail_mass_beyond_6kb": float(tail[SIM_TAIL_BINS:].sum()),
        "profile_dynamic_range": float(profile.dynamic_range),
        "runon vs binary": compare("runon"),
        "runon6 vs binary": compare("runon6"),
        "shape vs binary": compare("shape"),
        "runon vs runon6": compare("runon", "runon6"),
        "per_locus": {name: [scores[name][i] for i in loci] for name in scores},
        "loci": loci,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    for key in ("runon vs binary", "runon6 vs binary", "shape vs binary", "runon vs runon6"):
        print(key, {k: round(v, 4) for k, v in report[key].items()})
    print(f"wrote {args.output} in {report['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
