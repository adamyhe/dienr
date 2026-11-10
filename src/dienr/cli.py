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
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

from . import __version__
from .annotation import Annotation, read_annotation
from .contigs import ContigMap, check_contigs, rename_contigs
from .counts import VALUE_KINDS, SignalFile, count_sample, declared_contigs
from .design import Design, build_design
from .pipeline import fit, status_report
from .results import FitResult
from .shape import ShapeConfig, ShapeProfile, applied_dynamic_range, apply_shape_profile
from .shape import fit_shape_profile as _fit_shape_profile
from .solvers.penalties import Penalty
from .strand import check_strand_orientation

#: Filename for a serialized `Design` inside a design directory.
DESIGN_FILENAME = "design.pkl"

#: Filename for serialized `SampleCounts` inside a counts directory.
COUNTS_FILENAME = "counts.pkl"

#: Filename for a serialized `ShapeProfile` inside a profile directory.
SHAPE_FILENAME = "shape_profile.pkl"


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
    _add_contig_map_argument(design)
    design.add_argument("--output", required=True, type=Path, help="output directory")
    design.add_argument(
        "--shape-profile",
        type=Path,
        default=None,
        help=(
            "directory from 'dienr shape-profile'. Reweights every model "
            "column by measured relative density, which yields a new design "
            "digest because an abundance then means something different"
        ),
    )
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
    count.add_argument(
        "--summary",
        choices=("sum", "mean"),
        default="mean",
        help="per-bin statistic. 'mean' (default) matches the R reference, "
        "whose add_data hardcodes it, so the default pipeline reproduces it. "
        "Count likelihoods need 'sum'; 'dienr fit' rejects the mismatch.",
    )
    count.add_argument(
        "--threads",
        type=_positive_int,
        default=8,
        help="threads for reading and binning each BigWig (default 8). Samples "
        "are counted one after another; this parallelizes within a sample.",
    )
    count.add_argument(
        "--allow-contig-mismatch",
        action="store_true",
        help=(
            "count even if no locus's contig is in the BigWigs (every locus is then "
            "missing_contig). By default that is refused, since it almost always means "
            "the annotation and BigWigs use different naming; see --contig-map"
        ),
    )
    count.add_argument(
        "--allow-strand-mismatch",
        action="store_true",
        help=(
            "count even if the plus/minus files look swapped. By default each sample's "
            "gene-body signal is checked for being mostly in the declared-sense file, "
            "and a sample that fails is refused"
        ),
    )

    fit_parser = subparsers.add_parser("fit", help="estimate abundances")
    fit_parser.add_argument("--design", required=True, type=Path)
    fit_parser.add_argument("--counts", required=True, type=Path)
    fit_parser.add_argument("--output", required=True, type=Path)
    fit_parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=_default_jobs(),
        help=(
            "worker processes fitting loci in parallel (default: 4, or fewer if "
            "fewer CPUs are available to this process). Results are identical for "
            "any value; beyond 4 the single-threaded remainder dominates (plan 14.5)"
        ),
    )
    fit_parser.add_argument(
        "--blas-threads",
        type=_blas_threads,
        default=1,
        metavar="N|auto",
        help=(
            "threads BLAS may use while fitting (default 1: each locus is a tiny "
            "problem, and more threads made fitting slower, plan 14.4); 'auto' "
            "leaves the BLAS library's own setting"
        ),
    )
    fit_parser.add_argument(
        "--likelihood",
        choices=("legacy", "poisson", "quasi_poisson"),
        default="legacy",
        help="legacy least squares (default, the only one with R parity); "
        "poisson; or quasi_poisson, which gives identical abundances to poisson "
        "with standard errors scaled by a pooled dispersion and is the "
        "recommended choice for overdispersed counts",
    )
    fit_parser.add_argument("--transform", choices=("log", "identity"), default="log")
    fit_parser.add_argument(
        "--penalty",
        choices=("none", "ridge", "lasso", "elastic_net"),
        default="none",
        help="count likelihoods only; recorded in the output",
    )
    fit_parser.add_argument("--lambda1", type=float, default=0.0, help="L1 weight")
    fit_parser.add_argument("--lambda2", type=float, default=0.0, help="L2 weight")
    fit_parser.add_argument(
        "--use-bin-exposure",
        action="store_true",
        help="treat effective bin length as exposure (count likelihoods only)",
    )
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

    shape = subparsers.add_parser(
        "shape-profile",
        help="fit an empirical relative-density profile from isolated genes",
    )
    shape.add_argument("--gtf", required=True, type=Path, help="GTF/GFF annotation")
    _add_contig_map_argument(shape)
    shape.add_argument("--bigwig-plus", required=True, type=Path)
    shape.add_argument(
        "--bigwig-minus",
        required=True,
        type=Path,
        help=(
            "minus-strand BigWig. Required, and genuinely used: the R "
            "reference reads minus-strand genes from the plus file, which "
            "leaves most training curves empty and the profile flat"
        ),
    )
    shape.add_argument("--output", required=True, type=Path, help="output directory")
    shape.add_argument("--bin-size", type=int, default=250)
    shape.add_argument("--head-bases", type=int, default=10_000)
    shape.add_argument("--tail-bases", type=int, default=5_000)
    shape.add_argument(
        "--span",
        type=float,
        default=ShapeConfig().span,
        help=(
            "fraction of observations per smoothing neighbourhood. The "
            "reference's 0.4 is thirty times wider than a pausing peak and "
            "flattens it; the default here is smaller on purpose"
        ),
    )
    shape.add_argument(
        "--min-bandwidth",
        type=float,
        default=None,
        help=(
            "floor on the smoothing half-width in rescaled-position units; "
            "defaults to two head bins, so the profile is never fitted more "
            "finely than the bins it multiplies"
        ),
    )
    shape.add_argument(
        "--min-length",
        type=int,
        default=None,
        help="shortest training gene; defaults to head + tail",
    )
    shape.add_argument("--min-locus-fraction", type=float, default=0.9)
    shape.add_argument("--min-covered-fraction", type=float, default=0.5)
    shape.add_argument("--min-mean-signal", type=float, default=0.1)
    shape.add_argument(
        "--normalization", choices=("body_median", "whole_mean"), default="body_median"
    )
    shape.add_argument(
        "--allow-short-genes",
        action="store_true",
        help="accept genes shorter than head + tail, which contribute no body positions",
    )

    validate = subparsers.add_parser("validate", help="check a results directory")
    validate.add_argument("--results", required=True, type=Path)
    validate.add_argument("--max-failure-fraction", type=float, default=1.0)

    return parser


def _default_jobs() -> int:
    """4 fitting workers, capped by the CPUs this process may use (Slurm-aware)."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not on Linux
        available = os.cpu_count() or 1
    return max(1, min(4, available))


def _blas_threads(value: str) -> int | None:
    if value == "auto":
        return None
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("--blas-threads must be >= 1 or 'auto'")
    return number


def _load_design(directory: Path) -> Design:
    path = directory / DESIGN_FILENAME
    if not path.exists():
        raise SystemExit(f"{directory}: no {DESIGN_FILENAME}; run 'dienr build-design' first")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _add_contig_map_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--contig-map",
        default=None,
        metavar="FILE|PRESET",
        help=(
            "rename annotation contigs to match the BigWigs: a two-column "
            "old<TAB>new file, or a preset (ensembl-to-ucsc, ucsc-to-ensembl; "
            "primary chromosomes only). Recorded in the design and results"
        ),
    )


def _read_annotation(args: argparse.Namespace) -> Annotation:
    annotation = read_annotation(args.gtf)
    if args.contig_map is None:
        return annotation
    try:
        contig_map = ContigMap.load(args.contig_map)
        annotation, report = rename_contigs(annotation, contig_map)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    print(f"contig map {contig_map.source} ({contig_map.digest}): {report.message()}")
    return annotation


def _load_shape_profile(directory: Path) -> ShapeProfile:
    path = directory / SHAPE_FILENAME
    if not path.exists():
        raise SystemExit(f"{directory}: no {SHAPE_FILENAME}; run 'dienr shape-profile' first")
    with path.open("rb") as handle:
        profile: ShapeProfile = pickle.load(handle)
    return profile


def cmd_shape_profile(args: argparse.Namespace) -> int:
    annotation = _read_annotation(args)
    config = ShapeConfig(
        bin_size=args.bin_size,
        head_bases=args.head_bases,
        tail_bases=args.tail_bases,
        span=args.span,
        min_bandwidth=args.min_bandwidth,
        min_length=args.min_length,
        min_locus_fraction=args.min_locus_fraction,
        min_covered_fraction=args.min_covered_fraction,
        min_mean_signal=args.min_mean_signal,
        normalization=args.normalization,
        allow_short_genes=args.allow_short_genes,
    )
    signal = (
        SignalFile(path=args.bigwig_plus, strand="+"),
        SignalFile(path=args.bigwig_minus, strand="-"),
    )
    try:
        profile = _fit_shape_profile(annotation, signal, config=config)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / SHAPE_FILENAME).open("wb") as handle:
        pickle.dump(profile, handle)
    profile.table().write_parquet(args.output / "shape_profile.parquet")

    print(
        f"shape profile {profile.digest}: {profile.n_genes} training genes, "
        f"{profile.n_observations} bins, dynamic range {profile.dynamic_range:.2f}x"
    )
    dropped = ", ".join(f"{reason}={count}" for reason, count in profile.rejected.items() if count)
    if dropped:
        print(f"  genes dropped: {dropped}")
    # Print the curve so a flat or inverted profile is visible without plotting.
    sample = profile.evaluate(np.array([0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]))
    print("  position  0.00  0.02  0.05  0.10  0.25  0.50  0.75  1.00")
    print("  density  " + " ".join(f"{value:5.2f}" for value in sample))
    if profile.n_floored:
        print(
            f"warning: {profile.n_floored} of {len(profile.grid)} grid nodes were "
            "clamped to the profile floor, meaning the smoother undershot below "
            "zero there. Raise --span",
            file=sys.stderr,
        )
    if profile.dynamic_range < 1.2:
        print(
            f"warning: the profile spans only {profile.dynamic_range:.3f}x, so "
            "reshaping a design with it will barely change the fit. Check that "
            "--bigwig-plus and --bigwig-minus are the right way round and that "
            "the training genes are expressed",
            file=sys.stderr,
        )
    return 0


def cmd_build_design(args: argparse.Namespace) -> int:
    annotation = _read_annotation(args)
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
    if args.shape_profile is not None:
        profile = _load_shape_profile(args.shape_profile)
        design = apply_shape_profile(design, profile)

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
    if design.shape_digest is not None:
        applied = applied_dynamic_range(design)
        print(f"  reshaped by profile {design.shape_digest}; applied range {applied:.2f}x")
        if applied < 1.2:
            print(
                f"warning: only {applied:.3f}x of the profile's grading survives in "
                "this design. A shape profile and TSS masking compete for the same "
                "bins -- the pausing peak is the biggest thing the profile measures "
                "and the first thing a mask removes. Reduce --mask-tss/"
                "--mask-start-bins if the profile is meant to do work",
                file=sys.stderr,
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


def _positive_int(text: str) -> int:
    """argparse type for a count that must be at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


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
        contigs = check_contigs(
            design.loci["contig"].to_list(),
            set().union(*(declared_contigs(f.path) for f in signal)),
        )
        if contigs.none_match and not args.allow_contig_mismatch:
            raise SystemExit(
                f"error: sample {row['sample_id']!r}: {contigs.message()}. "
                "Pass --allow-contig-mismatch to count anyway."
            )
        if contigs.missing:
            print(
                f"warning: sample {row['sample_id']!r}: {contigs.message()}; those loci "
                "will be reported as missing_contig",
                file=sys.stderr,
            )
        check = check_strand_orientation(design, signal, n_jobs=args.threads)
        print(f"{row['sample_id']}: {check.message()}")
        if check.verdict == "swapped" and not args.allow_strand_mismatch:
            raise SystemExit(
                f"error: sample {row['sample_id']!r}: most gene-body signal is in the "
                "opposite-strand file, so bigwig_plus and bigwig_minus look swapped. "
                "Single-end PRO-seq reads are the reverse complement of the RNA; check how "
                "the BigWigs were made. Pass --allow-strand-mismatch to count anyway."
            )
        if check.verdict in ("low", "swapped", "insufficient"):
            print(
                f"warning: sample {row['sample_id']!r}: strand orientation is "
                f"{check.verdict}; check the plus/minus assignment",
                file=sys.stderr,
            )
        counts = count_sample(
            design,
            row["sample_id"],
            signal,
            summary=args.summary,
            exposure=float(row.get("exposure") or 1.0),
            n_jobs=args.threads,
        )
        counts.provenance.update(check.describe())
        collected.append(counts)

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

    result = fit(
        design,
        samples,
        likelihood=args.likelihood,
        transform=args.transform,
        penalty=Penalty(kind=args.penalty, lambda1=args.lambda1, lambda2=args.lambda2),
        use_bin_exposure=args.use_bin_exposure,
        inactive_transcripts=inactive,
        blas_threads=args.blas_threads,
        n_jobs=args.jobs,
    )
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
        "shape-profile": cmd_shape_profile,
        "validate": cmd_validate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
