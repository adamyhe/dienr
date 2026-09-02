"""Print a side-by-side comparison of the two benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = (
    ("read_annotation", "annotation parse"),
    ("build_design", "design construction"),
    ("count_all_samples", "signal aggregation"),
    ("fit_all_samples", "fitting"),
    ("total", "TOTAL"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dienr", required=True, type=Path)
    parser.add_argument("--denr", required=True, type=Path)
    args = parser.parse_args(argv)

    ours = json.loads(args.dienr.read_text())
    theirs = json.loads(args.denr.read_text())

    print(
        f"{theirs['n_transcripts']} transcripts, {theirs['n_loci']} loci, "
        f"{theirs['n_models']} models, {theirs['n_samples']} sample(s)"
    )
    print(f"reference: DENR {theirs['denr_version']} on R {theirs['r_version']}\n")
    print(f"{'stage':<22} {'DENR (s)':>10} {'dienr (s)':>11} {'speedup':>9}")
    print("-" * 55)
    for key, label in STAGES:
        r = theirs["timings_seconds"].get(key)
        p = ours["timings_seconds"].get(key)
        if r is None or p is None:
            continue
        ratio = f"{r / p:.1f}x" if p > 0 else "n/a"
        print(f"{label:<22} {r:>10.3f} {p:>11.3f} {ratio:>9}")
    print("-" * 55)
    if "peak_rss_mb" in theirs:
        print(f"{'peak RSS (MB)':<22} {theirs['peak_rss_mb']:>10.1f} {ours['peak_rss_mb']:>11.1f}")
    else:
        # R has no reliable in-process RSS figure; gc() reports heap cells.
        print(f"{'peak RSS (MB)':<22} {'external':>10} {ours['peak_rss_mb']:>11.1f}")
        print("  measure R with:  /usr/bin/time -l Rscript scripts/benchmark_denr.R ...")
    for field in ("n_loci", "n_models"):
        if theirs.get(field) != ours.get(field):
            print(
                f"\nWARNING: {field} differs (DENR {theirs.get(field)} vs "
                f"dienr {ours.get(field)}). Investigate before trusting the ratio."
            )
    if "caveat" in theirs:
        print(f"\nCaveat: {theirs['caveat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
