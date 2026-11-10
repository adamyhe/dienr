"""Profile one genome-scale run on K562, stage by stage (plan 14.4).

The 14.2 benchmarks are 124 and 955 loci. Before any numba work (14.1, 14.3
declined it twice), this measures where time goes on a real genome-scale
annotation: Ensembl GRCh37.87, contigs renamed N -> chrN, either the
protein-coding transcripts the K562 trials use or every transcript (denser
loci). Stages are timed separately and each is run under cProfile; the top
functions by own time are printed and the .prof files written for snakeviz.

Stages: parse, runon-tail, build-design, apply-tail, count (sum), strand
check, fit legacy (the R-parity default), fit poisson (recommended), write.

    uv run python scripts/profile_k562.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --rep rep1 --biotype protein_coding --output build/profile/pc
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import resource
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_runon_check import CONTIGS, signal_files  # noqa: E402

import dienr  # noqa: E402


def read_annotation(gtf: Path, biotype: str) -> dienr.Annotation:
    keep = pl.col("feature") == "transcript"
    keep &= pl.col("contig").is_in(CONTIGS)
    if biotype != "all":
        keep &= pl.col("attr").str.contains(f'transcript_biotype "{biotype}"')
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
        .filter(keep)
        .select(
            pl.col("attr").str.extract(r'transcript_id "([^"]+)"').alias("transcript_id"),
            pl.col("attr").str.extract(r'gene_id "([^"]+)"').alias("gene_id"),
            ("chr" + pl.col("contig")).alias("contig"),
            (pl.col("start") - 1).alias("start"),
            pl.col("end"),
            pl.col("strand"),
        )
        .collect()
    )
    return dienr.Annotation.from_frame(frame, source=f"{gtf} ({biotype}, N->chrN)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--bigwigs", type=Path, required=True)
    parser.add_argument("--rep", default="rep1")
    parser.add_argument("--biotype", default="protein_coding")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    signal = signal_files(args.bigwigs, args.rep)
    timings: dict[str, float] = {}
    tops: dict[str, str] = {}
    state: dict = {}

    def stage(name, fn):
        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        profiler.enable()
        out = fn()
        profiler.disable()
        timings[name] = time.perf_counter() - t0
        profiler.dump_stats(args.output / f"{name}.prof")
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats("tottime").print_stats(12)
        tops[name] = buf.getvalue()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"{name:14s} {timings[name]:8.2f} s   peak RSS so far {rss:7.0f} MB", flush=True)
        return out

    # runon_tail is the first BigWig read and includes figwig's ~0.25 s numba warm-up.
    annotation = stage("parse", lambda: read_annotation(args.gtf, args.biotype))
    tail = stage(
        "runon_tail", lambda: dienr.fit_runon_tail(annotation, signal, n_jobs=args.threads)
    )
    base = stage(
        "build_design",
        lambda: dienr.build_design(
            annotation, bin_size=250, mask_start_bins=(1, 1), mask_end_bins=(4, 4)
        ),
    )
    design = stage("apply_tail", lambda: dienr.apply_runon_tail(base, tail))
    counts = stage(
        "count",
        lambda: dienr.count_sample(design, args.rep, signal, summary="sum", n_jobs=args.threads),
    )
    stage(
        "strand_check", lambda: dienr.check_strand_orientation(design, signal, n_jobs=args.threads)
    )
    counts_mean = dienr.count_sample(base, args.rep, signal, n_jobs=args.threads)
    stage("fit_legacy", lambda: dienr.fit(base, counts_mean))
    result = stage("fit_poisson", lambda: dienr.fit(design, counts, likelihood="poisson"))
    stage("write", lambda: result.write(args.output / "results"))

    sizes = [m.shape for m in design.models]
    state = {
        "biotype": args.biotype,
        "rep": args.rep,
        "n_transcripts": len(annotation),
        "n_loci": design.n_loci,
        "n_models": int(sum(s[1] for s in sizes)),
        "multi_model_loci": int(sum(s[1] > 1 for s in sizes)),
        "max_models": int(max(s[1] for s in sizes)),
        "max_bins": int(max(s[0] for s in sizes)),
        "timings_s": timings,
        "total_s": sum(timings.values()),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (args.output / "profile.json").write_text(json.dumps(state, indent=1) + "\n")
    print(json.dumps({k: v for k, v in state.items() if k != "timings_s"}))
    total = state["total_s"]
    for name, sec in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {name:14s} {sec:8.2f} s  {100 * sec / total:5.1f}%")
    for name in sorted(timings, key=lambda k: -timings[k])[:3]:
        print(f"\n===== {name}: top functions by own time =====")
        print(tops[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
