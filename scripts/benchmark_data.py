"""Generate a benchmark dataset shared by the R and Python timing scripts.

Both implementations must see byte-identical inputs, so the dataset is written
once here rather than generated twice.

Scale is chosen so the R side finishes in minutes rather than hours: its fitting
loop is per-locus `stats::optim` driven from R, and the point of the benchmark
is a like-for-like ratio, not a whole-genome run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pybigtools

#: Contig length. One contig keeps the comparison about loci and bins rather
#: than about per-contig bookkeeping differences.
CONTIG = "chr1"


def build(
    output: Path,
    *,
    n_transcripts: int,
    contig_size: int,
    n_samples: int,
    signal_step: int,
    seed: int,
) -> dict:
    """Write a GTF, stranded BigWigs per sample, and a spec file."""
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Cluster transcripts so multi-isoform loci actually occur: pick locus
    # anchors, then place a few isoforms around each with shared or shifted
    # ends. A uniform scatter would make almost every locus single-isoform and
    # understate the work both implementations do.
    n_anchors = max(1, n_transcripts // 3)
    anchors = np.sort(rng.integers(1_000, contig_size - 200_000, n_anchors))

    records = []
    for index in range(n_transcripts):
        anchor = int(anchors[index % n_anchors])
        strand = "+" if index % 2 == 0 else "-"
        start = anchor + int(rng.integers(0, 3_000))
        length = int(rng.integers(2_000, 100_000))
        end = min(start + length, contig_size - 1)
        if end - start < 1_000:
            continue
        records.append(
            {
                "transcript_id": f"tx{index:06d}",
                "gene_id": f"gene{index // 3:06d}",
                "start": start + 1,  # GTF is 1-based inclusive
                "end": end,
                "strand": strand,
            }
        )

    gtf = output / "benchmark.gtf"
    lines = ["##dienr benchmark"]
    for record in records:
        attributes = f'gene_id "{record["gene_id"]}"; transcript_id "{record["transcript_id"]}";'
        for feature in ("transcript", "exon"):
            lines.append(
                "\t".join(
                    [
                        CONTIG,
                        "dienr",
                        feature,
                        str(record["start"]),
                        str(record["end"]),
                        ".",
                        record["strand"],
                        ".",
                        attributes,
                    ]
                )
            )
    gtf.write_text("\n".join(lines) + "\n")

    samples = []
    for sample in range(n_samples):
        paths = {}
        for key, sign in (("plus", 1.0), ("minus", -1.0)):
            path = output / f"sample{sample}_{key}.bw"
            positions = np.arange(0, contig_size - signal_step, signal_step)
            values = sign * rng.poisson(3.0, positions.size).astype(np.float64)
            writer = pybigtools.open(str(path), "w")
            writer.write(
                {CONTIG: contig_size},
                iter(
                    (CONTIG, int(p), int(p) + signal_step, float(v))
                    for p, v in zip(positions, values, strict=True)
                ),
            )
            writer.close()
            paths[key] = str(path)
        samples.append({"sample_id": f"sample{sample}", **paths})

    spec = {
        "gtf": str(gtf),
        "contig": CONTIG,
        "contig_size": contig_size,
        "n_transcripts": len(records),
        "n_intervals_per_bigwig": int((contig_size - signal_step) // signal_step),
        "signal_step": signal_step,
        "samples": samples,
        "bin_size": 250,
        "mask_start_bins": [4, 4],
        "mask_end_bins": [4, 4],
        "bin_operation": "round",
        "transform": "log",
        "summary": "mean",
    }
    (output / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transcripts", type=int, default=3_000)
    parser.add_argument("--contig-size", type=int, default=20_000_000)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--signal-step", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec = build(
        args.output,
        n_transcripts=args.transcripts,
        contig_size=args.contig_size,
        n_samples=args.samples,
        signal_step=args.signal_step,
        seed=args.seed,
    )
    print(
        f"wrote {spec['n_transcripts']} transcripts, {len(spec['samples'])} sample(s), "
        f"{spec['n_intervals_per_bigwig']:,} intervals per BigWig, to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
