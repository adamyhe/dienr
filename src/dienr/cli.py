"""Workflow-oriented command-line interface.

Build, count and fit are separate subcommands so annotation geometry is
computed once and cached by a workflow engine, rather than recomputed per
cohort (plan section 11).

The process exits nonzero when the fraction of failed loci exceeds
``--max-failure-fraction``, so a run that quietly failed most of its loci
cannot pass for a successful one.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import polars as pl

from . import __version__
from .annotation import read_annotation
from .counts import VALUE_KINDS, SignalFile, count_sample
from .design import Design, build_design
from .pipeline import fit, status_report
from .results import FitResult

#: Filename for a serialized `Design` inside a design directory.
DESIGN_FILENAME = "design.pkl"

#: Filename for serialized `SampleCounts` inside a counts directory.
COUNTS_FILENAME = "counts.pkl"


def _add_design_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bin-size", type=int, default=250, help="bin width in bases")
    parser.add_argument(
        "--distance",
        type=int,
        default=None,
        help="locus-grouping gap threshold (default: bin size)",
    )
    parser.add_argument(
        "--mask-start-bins",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("INSIDE", "OUTSIDE"),
        help="bins masked at each transcript's 5' end",
    )
    parser.add_argument(
        "--mask-end-bins",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("INSIDE", "OUTSIDE"),
        help="bins masked at each transcript's 3' end",
    )
    parser.add_argument(
        "--mask-tss",
        type=int,
        nargs=2,
        default=None,
        metavar=("UPSTREAM", "DOWNSTREAM"),
        help="bases masked around each TSS, along the direction of "
        "transcription (preferred over --mask-start-bins)",
    )
    parser.add_argument(
        "--mask-pas",
        type=int,
        nargs=2,
        default=None,
        metavar=("UPSTREAM", "DOWNSTREAM"),
        help="bases masked around each PAS",
    )
    parser.add_argument(
        "--bin-operation",
        choices=("round", "floor", "ceiling"),
        default="round",
        help="rounding applied to fractional bin occupancy",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="dienr",
        description="Deconvolution of Isoform Expression from Nascent RNAs",
    )
    parser.add_argument("--version", action="version", version=f"dienr {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser(
        "build-design", help="build reusable locus geometry from an annotation"
    )
    design.add_argument("--gtf", required=True, type=Path, help="GTF/GFF annotation")
    design.add_argument("--output", required=True, type=Path, help="output directory")
    _add_design_options(design)

    count = subparsers.add_parser("count", help="aggregate stranded BigWig signal into bins")
    count.add_argument("--design", required=True, type=Path, help="design directory")
    count.add_argument(
        "--samples",
        required=True,
        type=Path,
        help="CSV with columns sample_id, bigwig_plus, bigwig_minus "
        "and optionally value_kind and exposure",
    )
    count.add_argument("--output", required=True, type=Path)
    count.add_argument("--summary", choices=("sum", "mean"), default="sum")

    fit_parser = subparsers.add_parser("fit", help="estimate abundances")
    fit_parser.add_argument("--design", required=True, type=Path)
    fit_parser.add_argument("--counts", required=True, type=Path)
    fit_parser.add_argument("--output", required=True, type=Path)
    fit_parser.add_argument("--transform", choices=("log", "identity"), default="log")
    fit_parser.add_argument(
        "--inactive-transcripts",
        type=Path,
        default=None,
        help="file with one transcript id per line to pin at zero",
    )
    fit_parser.add_argument(
        "--max-failure-fraction",
        type=float,
        default=1.0,
        help="exit nonzero if more than this fraction of fits failed",
    )

    validate = subparsers.add_parser("validate", help="check a results directory")
    validate.add_argument("--results", required=True, type=Path)
    validate.add_argument("--max-failure-fraction", type=float, default=1.0)

    return parser


def _load_design(directory: Path) -> Design:
    path = directory / DESIGN_FILENAME
    if not path.exists():
        raise SystemExit(f"{directory}: no {DESIGN_FILENAME}; run 'dienr build-design' first")
    with path.open("rb") as handle:
        return pickle.load(handle)


def cmd_build_design(args: argparse.Namespace) -> int:
    annotation = read_annotation(args.gtf)
    design = build_design(
        annotation,
        bin_size=args.bin_size,
        distance=args.distance,
        mask_start_bins=tuple(args.mask_start_bins),
        mask_end_bins=tuple(args.mask_end_bins),
        mask_tss=tuple(args.mask_tss) if args.mask_tss else None,
        mask_pas=tuple(args.mask_pas) if args.mask_pas else None,
        bin_operation=args.bin_operation,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / DESIGN_FILENAME).open("wb") as handle:
        pickle.dump(design, handle)
    design.bins.write_parquet(args.output / "bins.parquet")
    design.loci.write_parquet(args.output / "loci.parquet")
    design.model_table.write_parquet(args.output / "models.parquet")

    print(
        f"design {design.digest}: {len(annotation)} transcripts, {design.n_loci} loci, "
        f"{design.model_table.height} models"
    )
    heavy = design.heavily_masked_loci(threshold=0.5)
    if heavy.height:
        worst = heavy.row(0, named=True)
        print(
            f"warning: {heavy.height} locus/loci lost more than half their bins "
            f"to masking (worst: {worst['locus_name']} at "
            f"{100 * worst['masked_fraction']:.0f}%). Masking is a union over "
            "every transcript in a locus, so narrow loci with several isoforms "
            "are fitted on much less data than their bin count suggests.",
            file=sys.stderr,
        )
    unsupported = design.unsupported_models.height
    if unsupported:
        print(
            f"warning: {unsupported} model(s) have no bin support at bin size "
            f"{design.config.bin_size} and cannot be estimated; "
            "lower --bin-size or exclude those transcripts",
            file=sys.stderr,
        )
    return 0


def cmd_count(args: argparse.Namespace) -> int:
    design = _load_design(args.design)
    table = pl.read_csv(args.samples)
    required = {"sample_id", "bigwig_plus", "bigwig_minus"}
    missing = required - set(table.columns)
    if missing:
        raise SystemExit(f"{args.samples}: missing column(s) {sorted(missing)}")

    collected = []
    for row in table.iter_rows(named=True):
        kind = str(row.get("value_kind") or "raw")
        if kind not in VALUE_KINDS:
            raise SystemExit(
                f"{args.samples}: sample {row['sample_id']!r} has value_kind "
                f"{kind!r}, expected one of {list(VALUE_KINDS)}"
            )
        signal = (
            SignalFile(row["bigwig_plus"], "+", value_kind=kind),  # type: ignore[arg-type]
            SignalFile(row["bigwig_minus"], "-", value_kind=kind),  # type: ignore[arg-type]
        )
        collected.append(
            count_sample(
                design,
                row["sample_id"],
                signal,
                summary=args.summary,
                exposure=float(row.get("exposure") or 1.0),
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / COUNTS_FILENAME).open("wb") as handle:
        pickle.dump(collected, handle)
    print(f"counted {len(collected)} sample(s) against design {design.digest}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    design = _load_design(args.design)
    counts_path = args.counts / COUNTS_FILENAME
    if not counts_path.exists():
        raise SystemExit(f"{args.counts}: no {COUNTS_FILENAME}; run 'dienr count' first")
    with counts_path.open("rb") as handle:
        samples = pickle.load(handle)

    inactive = None
    if args.inactive_transcripts:
        inactive = {
            line.strip()
            for line in args.inactive_transcripts.read_text().splitlines()
            if line.strip()
        }

    result = fit(design, samples, transform=args.transform, inactive_transcripts=inactive)
    result.write(args.output)
    return _report(result, args.max_failure_fraction)


def cmd_validate(args: argparse.Namespace) -> int:
    result = FitResult.read(args.results)
    print(f"schema_version {result.provenance.get('schema_version')}")
    print(f"design_digest  {result.provenance.get('design_digest')}")
    return _report(result, args.max_failure_fraction)


def _report(result: FitResult, threshold: float) -> int:
    """Print the status tally and decide the exit code."""
    summary = result.summary
    print(status_report(result))
    print(
        f"{summary.n_fits} fits, {summary.n_failures} failed "
        f"({100 * summary.failure_fraction:.1f}%)"
    )
    if summary.exceeds(threshold):
        print(
            f"error: failure fraction {summary.failure_fraction:.3f} exceeds "
            f"threshold {threshold:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    handlers = {
        "build-design": cmd_build_design,
        "count": cmd_count,
        "fit": cmd_fit,
        "validate": cmd_validate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
