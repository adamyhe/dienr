"""Time the Python pipeline on the shared benchmark dataset.

Stages are reported separately because they have different reuse properties:
the design is built once and reused across samples, while counting and fitting
are per sample.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from dienr.annotation import read_annotation
from dienr.counts import SignalFile, count_sample
from dienr.design import build_design
from dienr.pipeline import fit


def peak_rss_mb() -> float:
    """Peak resident set size in MB (macOS reports bytes, Linux kilobytes)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if raw > 1e7 else raw / 1e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    spec = json.loads((args.inputs / "spec.json").read_text())
    timings: dict[str, float] = {}

    start = time.perf_counter()
    annotation = read_annotation(spec["gtf"])
    timings["read_annotation"] = time.perf_counter() - start

    start = time.perf_counter()
    design = build_design(
        annotation,
        bin_size=spec["bin_size"],
        mask_start_bins=tuple(spec["mask_start_bins"]),
        mask_end_bins=tuple(spec["mask_end_bins"]),
        bin_operation=spec["bin_operation"],
    )
    timings["build_design"] = time.perf_counter() - start

    counts = []
    start = time.perf_counter()
    for sample in spec["samples"]:
        counts.append(
            count_sample(
                design,
                sample["sample_id"],
                (SignalFile(sample["plus"], "+"), SignalFile(sample["minus"], "-")),
                summary=spec["summary"],
            )
        )
    timings["count_all_samples"] = time.perf_counter() - start

    start = time.perf_counter()
    result = fit(design, counts, transform=spec["transform"])
    timings["fit_all_samples"] = time.perf_counter() - start

    timings["total"] = sum(
        timings[k]
        for k in (
            "read_annotation",
            "build_design",
            "count_all_samples",
            "fit_all_samples",
        )
    )

    report = {
        "implementation": "dienr",
        "n_transcripts": len(annotation),
        "n_loci": design.n_loci,
        "n_models": design.model_table.height,
        "n_samples": len(spec["samples"]),
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "total_abundance": round(float(result.model_abundance["abundance_raw"].sum()), 6),
        "status_counts": result.summary.by_status,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
