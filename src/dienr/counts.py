"""BigWig aggregation into per-bin signal, with exposure and provenance.

All BigWig access in this package goes through `read_binned`, built on
`figwig`. figwig reads exact per-base values and has no zoom-level path, so the
binning -- and therefore every semantic choice about it -- happens here, in
plain NumPy, where it is visible and tested:

Uncovered bases
    A base no interval covers is read as ``NaN`` (``missing=nan``) so it can be
    told apart from an interval whose value is ``0``. It contributes nothing to
    any summary, and a bin with no covered base summarizes to ``0``: in a
    nascent-RNA BigWig an absent interval means zero observed polymerase, and a
    leaked ``NaN`` would surface as a spurious numerical failure rather than a
    legitimate zero.

Covered-base statistics
    ``mean``, ``min``, ``max`` and ``std`` are taken over *covered* bases only,
    a zero-valued interval counting as covered; ``std`` is the sample standard
    deviation, ``0`` for fewer than two bases. These are the semantics the
    package had under `pybigtools` (``values(..., exact=True, fillna=0)``),
    measured base by base, and R parity was established against them.

Bin edges
    Bin ``i`` of a window ``[start, end)`` is
    ``[start + floor(i * w / n), start + floor((i + 1) * w / n))`` with
    ``w = end - start``, so a window that is not a multiple of ``n`` gives the
    remainder to later bins rather than reading fractional bases.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from figwig import BigWigReader

from .design import Design
from .status import FitStatus

Summary = Literal["sum", "mean", "min", "max", "std"]

#: How values in a signal file should be interpreted, per plan section 8.
ValueKind = Literal["raw", "spike_in_scaled", "library_normalized", "other_transformed"]

#: Every permitted `ValueKind`, for validating untyped external input.
VALUE_KINDS: tuple[str, ...] = (
    "raw",
    "spike_in_scaled",
    "library_normalized",
    "other_transformed",
)

#: Likelihoods that require count-equivalent input. The single definition:
#: `fit` and `SignalFile.require_count_equivalent` both go through
#: `require_count_equivalent`. There used to be two copies, and this one lacked
#: ``"quasi_poisson"``, so the public check let a per-base mean through for it.
COUNT_LIKELIHOODS = frozenset({"poisson", "quasi_poisson"})

#: Per-bin summaries that preserve counts. ``mean`` divides by the number of
#: covered bases, so it is a density and cannot be fed to a count likelihood.
_COUNT_SUMMARIES = frozenset({"sum"})


@dataclass(frozen=True)
class SignalFile:
    """One stranded BigWig plus the metadata needed to validate its use.

    Attributes:
        path: Path to the BigWig.
        strand: Which strand's signal this file holds.
        value_kind: Whether values are raw counts or have been transformed.
            Recorded rather than guessed; `require_count_equivalent` enforces it.
    """

    path: str | Path
    strand: Literal["+", "-"]
    value_kind: ValueKind = "raw"

    def __post_init__(self) -> None:
        # External input (a samples CSV) is untyped, and a typo here would
        # silently defeat the count-equivalence check below rather than
        # failing, so validate rather than trusting the annotation.
        if self.value_kind not in VALUE_KINDS:
            raise ValueError(
                f"{self.path}: value_kind {self.value_kind!r} is not one of {list(VALUE_KINDS)}"
            )
        if self.strand not in ("+", "-"):
            raise ValueError(f"{self.path}: strand must be '+' or '-', got {self.strand!r}")

    def require_count_equivalent(self, likelihood: str, summary: str = "sum") -> None:
        """Raise if `likelihood` needs counts but the input is not count-like.

        Two distinct ways the input can fail to be a count: the file itself was
        transformed, or the per-bin summary divided by bin width.
        """
        require_count_equivalent(
            likelihood, summary=summary, value_kinds=(self.value_kind,), source=str(self.path)
        )


def require_count_equivalent(
    likelihood: str,
    *,
    summary: str,
    value_kinds: Iterable[str],
    source: str,
) -> None:
    """Raise if `likelihood` needs counts and the input described is not counts.

    Args:
        likelihood: Likelihood name; anything outside `COUNT_LIKELIHOODS` passes.
        summary: Per-bin summary the signal was aggregated with.
        value_kinds: `ValueKind` of every signal file involved.
        source: What to name in the error -- a path or a sample.

    Raises:
        ValueError: If any file is transformed, or the summary is a density.
    """
    if likelihood not in COUNT_LIKELIHOODS:
        return
    transformed = sorted(set(value_kinds) - {"raw"})
    if transformed:
        raise ValueError(
            f"{source}: likelihood {likelihood!r} requires count-equivalent, "
            f"untransformed counts, but value_kind is {transformed}. Use an "
            "untransformed BigWig or select a non-count likelihood explicitly."
        )
    if summary not in _COUNT_SUMMARIES:
        raise ValueError(
            f"{source}: likelihood {likelihood!r} requires a count-preserving summary, "
            f"but summary is {summary!r}, which is a per-base density. Re-count with "
            f"one of {sorted(_COUNT_SUMMARIES)}."
        )


@dataclass
class SampleCounts:
    """Per-bin signal for one sample, aligned to a `Design`.

    Attributes:
        sample_id: Identifier for the sample.
        values: Per-locus bin vectors, indexed by ``locus_id``. Masked bins are
            already zeroed.
        locus_status: Per-locus status assigned during extraction. Only
            `FitStatus.MISSING_CONTIG` and `FitStatus.ALL_ZERO` originate here;
            everything else is decided by the solver.
        totals_all_bins: Per-locus signal summed over *all* bins, before
            masking zeroed any of them. Kept because `values` is masked in
            place, so the unmasked total is otherwise unrecoverable, and
            reporting only the masked total would understate observed coverage
            at loci where masking removed a lot.
        design_digest: Digest of the `Design` these counts were built against.
        provenance: Free-form record of inputs and reader settings.
        exposure: Optional known scaling factor for the sample, such as a
            spike-in or library size. Never applied silently; solvers take it
            explicitly.
    """

    sample_id: str
    values: list[np.ndarray]
    locus_status: list[FitStatus]
    design_digest: str
    totals_all_bins: list[float] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    exposure: float = 1.0

    def total_signal(self) -> float:
        """Sum of all unmasked signal across loci."""
        return float(sum(v.sum() for v in self.values))


def read_binned(
    path: str | Path,
    contig: str,
    start: int,
    end: int,
    n_bins: int,
    *,
    summary: Summary = "sum",
    reader: BigWigReader | None = None,
) -> np.ndarray:
    """Read `n_bins` exact, non-negative binned summaries from a BigWig.

    The single BigWig read path for the whole package. Every normalization a
    caller needs is applied *here*, at the I/O boundary, rather than by each
    caller; the module docstring gives the binning semantics.

    - **Magnitude.** Minus-strand BigWigs are conventionally written with
      negative values so a browser draws them below the axis; the signal is the
      magnitude. Strand is carried as data on `SignalFile`, never inferred from
      the sign, so a signed value has no meaning downstream and is only a way
      to get a wrong answer.
    - **Length.** The result always has exactly `n_bins` values, aligned to the
      design geometry, including for a window clipped at or past a contig end
      (bases past the end are uncovered).

    Both used to live in `count_sample`, so when `dienr.shape` was added it read
    through this function and got neither. On a conventionally signed
    minus-strand file that made every minus-strand gene's mean negative, so all
    of them failed the shape profile's expression filter and the profile was
    silently fitted from plus-strand genes alone. No test caught it because
    every fixture wrote positive values on both strands -- the same blind spot
    that hides the reference implementation's strand bugs from its own tests.
    The fix is structural: one reader, and the corrections are not optional.

    Args:
        path: BigWig path. Ignored when `reader` is supplied.
        contig: Contig name, which must appear in the file header.
        start: 0-based inclusive window start.
        end: Half-open window end.
        n_bins: Number of bins spanning the window.
        summary: Per-bin statistic. ``"sum"`` is the count-preserving choice
            and the default *here*, since this is the raw aggregation
            primitive. Note `count_sample` defaults to ``"mean"`` instead,
            because that is what the reference implementation uses; the two
            defaults differ deliberately and each is documented where it sits.
        reader: An open reader to reuse across calls.

    Returns:
        A non-negative float64 array of exactly `n_bins` values, with
        uncovered positions and positions past a contig end counted as zero.

    Raises:
        KeyError: If `contig` is absent from the file header. figwig itself
            would only warn and return zeros, which is the silent conflation of
            `FitStatus.MISSING_CONTIG` with `FitStatus.ALL_ZERO` that section 10
            of the plan forbids.
        ValueError: If the window is empty or `n_bins` is not positive.
    """
    width = end - start
    if width <= 0 or n_bins <= 0:
        raise ValueError(f"empty window {contig}:{start}-{end} or n_bins={n_bins}")
    handle = reader if reader is not None else BigWigReader(str(path))
    if contig not in handle.chrom_sizes:
        raise KeyError(contig)

    per_base = handle.read(
        [contig],
        [start],
        width,
        n_jobs=1,  # one window per call: threads cannot help and cost startup
        missing=np.nan,  # uncovered must stay distinguishable from a 0 interval
        oob=np.nan,  # past the contig end is uncovered, not zero
    )[0]
    return np.abs(_summarize_bins(per_base, n_bins, summary))


def _summarize_bins(per_base: np.ndarray, n_bins: int, summary: Summary) -> np.ndarray:
    """Summarize per-base values (``NaN`` = uncovered) into `n_bins` bins.

    `per_base` is figwig's float32 output and is overwritten: uncovered bases
    are zeroed in place, and every reduction accumulates in float64 without
    materializing a float64 copy of the window. On the dense benchmark the
    copies were as expensive as decompressing the BigWig.
    """
    width = per_base.size
    if width % n_bins == 0:
        # The common case -- every design window is a whole number of bins --
        # is a reshape, with no index arithmetic to get wrong.
        return _summarize_rows(per_base.reshape(n_bins, width // n_bins), summary)

    uncovered = np.isnan(per_base)
    per_base[uncovered] = 0.0
    edges = (np.arange(n_bins + 1, dtype=np.int64) * width) // n_bins
    lengths = np.diff(edges)
    nonempty = lengths > 0
    starts = edges[:-1][nonempty]

    def reduce_bins(ufunc: np.ufunc, array: np.ndarray, empty: float) -> np.ndarray:
        out = np.full(n_bins, empty)
        out[nonempty] = ufunc.reduceat(array, starts)
        return out

    n = lengths - reduce_bins(np.add, uncovered.astype(np.int64), 0)
    sums = reduce_bins(np.add, per_base.astype(np.float64), 0.0)
    if summary == "sum":
        return sums
    mean = np.divide(sums, n, out=np.zeros(n_bins), where=n > 0)
    if summary == "mean":
        return mean
    if summary in ("min", "max"):
        fill = np.inf if summary == "min" else -np.inf
        ufunc = np.minimum if summary == "min" else np.maximum
        out = reduce_bins(ufunc, np.where(uncovered, fill, per_base.astype(np.float64)), fill)
        return np.where(n > 0, out, 0.0)
    deviations = per_base - np.repeat(mean, lengths)
    deviations[uncovered] = 0.0
    squares = reduce_bins(np.add, np.square(deviations), 0.0)
    return np.sqrt(np.divide(squares, n - 1, out=np.zeros(n_bins), where=n > 1))


def _summarize_rows(rows: np.ndarray, summary: Summary) -> np.ndarray:
    """Summarize each row of a 2-D per-base array (``NaN`` = uncovered).

    One row is one bin. Used both for a single whole-bin window and for every
    bin of a batched tile read at once. `rows` is overwritten.
    """
    n_rows, width = rows.shape
    uncovered = np.isnan(rows)
    np.copyto(rows, 0.0, where=uncovered)  # unlike rows[mask] = 0, releases the GIL
    sums = rows.sum(axis=1, dtype=np.float64)
    if summary == "sum":
        return sums
    n = width - uncovered.sum(axis=1)
    mean = np.divide(sums, n, out=np.zeros(n_rows), where=n > 0)
    if summary == "mean":
        return mean
    if summary in ("min", "max"):
        fill = np.inf if summary == "min" else -np.inf
        reduce = np.min if summary == "min" else np.max
        masked = np.where(uncovered, fill, rows)
        return np.where(n > 0, reduce(masked, axis=1).astype(np.float64), 0.0)
    deviations = rows - mean[:, None]
    deviations[uncovered] = 0.0
    squares = np.square(deviations).sum(axis=1)
    return np.sqrt(np.divide(squares, n - 1, out=np.zeros(n_rows), where=n > 1))


#: Bins per tile in `read_loci_binned`. A tile is the unit figwig reads; it is a
#: whole number of bins so no bin straddles two tiles. 16 bins (4 kb at the
#: default 250 bp) wastes at most one tile per locus and gives a thread pool
#: thousands of equal windows to share.
TILE_BINS = 16

#: Upper bound on one figwig call's per-base output. Loci are read in chunks
#: under this cap, so memory stays flat however much of the genome the design
#: spans (a whole genome read per base is gigabytes). A locus larger than the
#: cap is read alone.
MAX_CHUNK_BYTES = 64 * 2**20


def read_loci_binned(
    reader: BigWigReader,
    windows: list[tuple[str, int, int]],
    bin_size: int,
    *,
    summary: Summary = "sum",
    n_jobs: int = 8,
) -> list[np.ndarray]:
    """Read many whole-bin windows at once, as `read_binned` would one by one.

    figwig is fast when given many equal-width windows in one multithreaded
    call, and no faster than one call per window otherwise -- one call per
    locus left its thread pool idle and was slower than the previous reader.
    Each window ``(contig, start, n_bins)`` is cut into tiles of
    ``TILE_BINS * bin_size`` bases; tiles are read in calls capped at
    `MAX_CHUNK_BYTES`, every bin of a chunk is summarized in one vectorized
    pass, and each window takes its contiguous run of bins.

    The result is identical to ``read_binned(..., start, start + n_bins *
    bin_size, n_bins)`` for each window, which a test asserts for every
    summary. Tile bases past a window's end are read and discarded; bases past
    a contig end are uncovered, exactly as in `read_binned`.

    Args:
        reader: Open reader; every window's contig must be in its header.
        windows: ``(contig, start, n_bins)`` per window, in any order.
        bin_size: Bin width in bases.
        summary: Per-bin statistic, as in `read_binned`.
        n_jobs: Threads per call, for figwig's read and for binning.

    Raises:
        ValueError: If `n_jobs` is below 1. (figwig alone would read ``-1`` as
            every CPU; the binning pool has no such convention, so it is
            refused rather than half-honored.)

    Returns:
        One non-negative float64 array of `n_bins` values per window, in input
        order.
    """
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1, got {n_jobs}")
    tile = TILE_BINS * bin_size
    max_tiles = max(1, MAX_CHUNK_BYTES // (4 * tile))
    out: list[np.ndarray] = [np.empty(0)] * len(windows)
    # Genomic order keeps each chunk's tiles close together in the file.
    order = sorted(range(len(windows)), key=lambda i: (windows[i][0], windows[i][1]))

    chunk: list[int] = []
    n_tiles = 0
    pool = ThreadPoolExecutor(n_jobs) if n_jobs > 1 else None

    def flush() -> None:
        tiles_per = [-(-windows[i][2] // TILE_BINS) for i in chunk]
        contigs = np.repeat([windows[i][0] for i in chunk], tiles_per)
        starts = np.concatenate(
            [
                windows[i][1] + tile * np.arange(k, dtype=np.int64)
                for i, k in zip(chunk, tiles_per, strict=True)
            ]
        )
        per_base = reader.read(contigs, starts, tile, n_jobs=n_jobs, missing=np.nan, oob=np.nan)
        rows = per_base.reshape(-1, bin_size)
        # Binning a chunk costs about as much as reading it, single-threaded;
        # NumPy releases the GIL in these reductions, so split the rows across
        # the same number of threads figwig used.
        blocks = np.array_split(rows, min(n_jobs, len(rows)))
        if pool is None:
            parts = [_summarize_rows(block, summary) for block in blocks]
        else:
            parts = list(pool.map(lambda block: _summarize_rows(block, summary), blocks))
        binned = np.abs(np.concatenate(parts))
        first = 0
        for i, k in zip(chunk, tiles_per, strict=True):
            out[i] = binned[first : first + windows[i][2]].copy()
            first += k * TILE_BINS

    try:
        for i in order:
            k = -(-windows[i][2] // TILE_BINS)
            if chunk and n_tiles + k > max_tiles:
                flush()
                chunk, n_tiles = [], 0
            chunk.append(i)
            n_tiles += k
        if chunk:
            flush()
    finally:
        if pool is not None:
            pool.shutdown()
    return out


def declared_contigs(path: str | Path) -> dict[str, int]:
    """Contig names and lengths declared in a BigWig header.

    The header is the only way to tell a contig that is absent from the file
    (`FitStatus.MISSING_CONTIG`) from one that is present but carries no signal
    in the window of interest (`FitStatus.ALL_ZERO`). Inferring the contig list
    from observed data conflates the two, which section 10 of the plan forbids.
    """
    return dict(BigWigReader(str(path)).chrom_sizes)


def count_sample(
    design: Design,
    sample_id: str,
    signal: SignalFile | tuple[SignalFile, ...],
    *,
    summary: Summary = "mean",
    exposure: float = 1.0,
    n_jobs: int = 8,
) -> SampleCounts:
    """Aggregate stranded BigWig signal into `design`'s bins.

    Each locus is read from the file matching its strand. Masked bins are zeroed
    so that model and data agree, as the objective requires.

    Args:
        design: Geometry to align to.
        sample_id: Identifier for this sample.
        signal: One `SignalFile` per strand. A single file is accepted only when
            every locus lies on its strand.
        summary: Per-bin statistic. Defaults to ``"mean"`` because that is what
            the reference implementation uses -- ``add_data`` hardcodes
            ``summary_operation <- "mean"`` -- so its abundances are in units of
            mean per-base polymerase density, not counts. Verified against R:
            our ``"sum"`` was exactly ``bin_size`` times its values. A count
            likelihood needs ``"sum"`` instead, and must not be combined with
            ``"mean"``.
        exposure: Known sample scaling factor, recorded but not applied here.
        n_jobs: Threads for the batched BigWig reads (`read_loci_binned`).

    Returns:
        A `SampleCounts` aligned to `design`.

    Raises:
        ValueError: If a required strand has no signal file, or if two files
            claim the same strand.
    """
    files = (signal,) if isinstance(signal, SignalFile) else tuple(signal)
    by_strand: dict[str, SignalFile] = {}
    for item in files:
        if item.strand in by_strand:
            raise ValueError(f"two signal files given for strand {item.strand!r}")
        by_strand[item.strand] = item

    needed = set(design.loci["strand"].unique().to_list())
    missing = needed - set(by_strand)
    if missing:
        raise ValueError(f"design contains loci on strand(s) {sorted(missing)} with no signal file")

    readers = {s: BigWigReader(str(f.path)) for s, f in by_strand.items()}
    headers = {s: dict(r.chrom_sizes) for s, r in readers.items()}

    values: list[np.ndarray] = [np.empty(0) for _ in range(design.n_loci)]
    statuses: list[FitStatus] = [FitStatus.OK] * design.n_loci
    totals: list[float] = [float("nan")] * design.n_loci

    # Loci whose window lies inside the contig are batched per strand; that is
    # nearly all of them. A window running past the contig end is read alone
    # through `read_binned`, clipped at the end, exactly as before batching.
    batched: dict[str, list[tuple[int, tuple[str, int, int]]]] = {s: [] for s in by_strand}
    raws: dict[int, np.ndarray] = {}
    bin_size = design.config.bin_size
    for locus in design.loci.iter_rows(named=True):
        locus_id = locus["locus_id"]
        strand = locus["strand"]
        n_bins = design.bin_count(locus_id)
        contig = locus["contig"]

        if contig not in headers[strand]:
            # A real failure, distinct from a locus with no signal.
            values[locus_id] = np.full(n_bins, np.nan)
            statuses[locus_id] = FitStatus.MISSING_CONTIG
            continue

        # Bins are anchored at the locus start, so no lookup is needed. Taking
        # this from the bins frame would also mean an unguarded `.min()` that
        # returns None for an empty selection.
        window_start = int(locus["start"])
        window_end = window_start + n_bins * bin_size
        if window_end <= headers[strand][contig]:
            batched[strand].append((locus_id, (contig, window_start, n_bins)))
            continue
        raws[locus_id] = read_binned(
            by_strand[strand].path,
            contig,
            window_start,
            headers[strand][contig],
            n_bins,
            summary=summary,
            reader=readers[strand],
        )

    for strand, items in batched.items():
        if not items:
            continue
        results = read_loci_binned(
            readers[strand],
            [window for _, window in items],
            bin_size,
            summary=summary,
            n_jobs=n_jobs,
        )
        raws.update(zip((locus_id for locus_id, _ in items), results, strict=True))

    for locus_id, raw in raws.items():
        # Record the total before masking; after this the masked bins are gone.
        totals[locus_id] = float(raw.sum())
        raw[design.mask(locus_id)] = 0.0
        values[locus_id] = raw
        if not raw.any():
            statuses[locus_id] = FitStatus.ALL_ZERO

    return SampleCounts(
        sample_id=sample_id,
        values=values,
        locus_status=statuses,
        design_digest=design.digest,
        totals_all_bins=totals,
        provenance={
            "signal_files": [
                {"path": str(f.path), "strand": f.strand, "value_kind": f.value_kind} for f in files
            ],
            "summary": summary,
            "reader": "figwig",
            "tile_bases": TILE_BINS * bin_size,
            "reader_settings": {
                "missing": "nan",
                "oob": "nan",
                "uncovered": "excluded from every summary; empty bin = 0",
            },
        },
        exposure=exposure,
    )
