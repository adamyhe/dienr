"""Bins, masks, transcript-model matrices, and exact-equivalence collapsing.

A `Design` is the reusable, annotation-only half of the pipeline: it depends on
the annotation, bin size, masking and bin operation, but not on any sample. One
`Design` serves many samples (plan section 7).

Geometry, per locus:

- Bins are anchored at the locus start, each of full ``bin_size`` width, with
  ``ceil(locus_width / bin_size)`` of them. The final bin therefore overhangs
  the locus end; its true usable extent is recorded as ``effective_length``.
- ``M[b, t]`` is transcript ``t``'s fractional occupancy of bin ``b``, that is
  overlap width divided by ``bin_size``, so a fully covered bin holds 1.0.
- `bin_operation` then rounds those fractions to integers, which is what makes
  "exactly identical model" a meaningful test rather than a float coincidence.
- Masked bins are zeroed in ``M`` and must be zeroed in the count vector too.
- Finally, identical columns collapse into one identifiable model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import polars as pl

from .annotation import Annotation
from .loci import assign_loci, locus_table

BinOperation = Literal["round", "floor", "ceiling"]

_BIN_OPERATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "round": np.round,
    "floor": np.floor,
    "ceiling": np.ceil,
}


@dataclass(frozen=True)
class DesignConfig:
    """Design-construction parameters.

    Attributes:
        bin_size: Bin width in bases.
        distance: Locus-grouping gap threshold. Defaults to `bin_size` when
            None, matching the reference implementation.
        mask_start_bins: ``(inside, outside)`` bin counts masked at each
            transcript's 5' end. "Inside" bins start at the transcript's first
            occupied bin and run into the transcript; "outside" bins run away
            from it.
        mask_end_bins: The same at the 3' end.
        mask_tss: ``(upstream, downstream)`` bases masked around each
            transcript's TSS, measured along the direction of transcription. A
            bin is masked when it overlaps that genomic window, which includes
            the anchor base, so ``(n, n)`` spans ``2n + 1`` bases. ``(0, 0)``
            disables masking at that end rather than masking the single anchor
            base. Preferred over
            `mask_start_bins`: the masked span is then a fixed number of bases
            rather than a count of bins, so it does not change meaning when
            `bin_size` changes and does not jitter by up to a bin depending on
            where the TSS happens to fall inside one.
        mask_pas: The same around the PAS.
        bin_operation: How fractional bin occupancies are rounded before
            equivalence testing.
    """

    bin_size: int = 250
    distance: int | None = None
    mask_start_bins: tuple[int, int] = (0, 0)
    mask_end_bins: tuple[int, int] = (0, 0)
    mask_tss: tuple[int, int] | None = None
    mask_pas: tuple[int, int] | None = None
    bin_operation: BinOperation = "round"

    def __post_init__(self) -> None:
        if self.bin_size <= 0:
            raise ValueError(f"bin_size must be positive, got {self.bin_size}")
        if self.bin_operation not in _BIN_OPERATIONS:
            raise ValueError(
                f"bin_operation must be one of {sorted(_BIN_OPERATIONS)}, "
                f"got {self.bin_operation!r}"
            )
        for name in ("mask_start_bins", "mask_end_bins"):
            value = getattr(self, name)
            if len(value) != 2 or any(v < 0 for v in value):
                raise ValueError(f"{name} must be two non-negative integers, got {value!r}")
        for name in ("mask_tss", "mask_pas"):
            value = getattr(self, name)
            if value is None:
                continue
            if len(value) != 2 or any(v < 0 for v in value):
                raise ValueError(f"{name} must be two non-negative integers, got {value!r}")
        if self.uses_base_windows and self.uses_bin_counts:
            raise ValueError(
                "mask_tss/mask_pas (bases) and mask_start_bins/mask_end_bins "
                "(bin counts) are alternative ways to specify the same masking "
                "and cannot be combined. Use the base form unless you need "
                "parity with the R reference."
            )
        if self.distance is not None and self.distance < 0:
            raise ValueError(f"distance must be non-negative, got {self.distance}")

    @property
    def uses_bin_counts(self) -> bool:
        """True when masking is specified as bin counts, the legacy form."""
        return any(self.mask_start_bins) or any(self.mask_end_bins)

    @property
    def uses_base_windows(self) -> bool:
        """True when masking is specified as base windows."""
        return self.mask_tss is not None or self.mask_pas is not None

    @property
    def effective_distance(self) -> int:
        """Locus-grouping distance, defaulting to `bin_size`."""
        return self.bin_size if self.distance is None else self.distance


@dataclass(frozen=True)
class Design:
    """Immutable, sample-independent locus geometry.

    Attributes:
        bins: One row per bin: ``locus_id``, ``bin_index``, ``contig``,
            ``start``, ``end``, ``effective_length``, ``masked``.
        models: Per-locus reduced design matrices, each ``(n_bins, n_models)``
            with masked bins zeroed. Indexed by ``locus_id``.
        masks: Per-locus boolean bin masks, indexed by ``locus_id``. Held
            alongside `bins` so per-locus access is a list index rather than a
            DataFrame filter, which would otherwise be paid once per locus per
            sample.
        bin_lengths: Per-locus usable bin widths, indexed by ``locus_id``.
        model_spans: Per-locus ``(n_models, 2)`` arrays of the first and last
            bin each model occupies *before* masking, or ``(-1, -1)`` for a
            model with no occupied bin at all. Recorded at build time because
            masking destroys it: a TSS mask zeroes the first bins of a column,
            so the surviving nonzero span starts a mask-width into the
            transcript and no longer says where the transcript began. Anything
            that needs a position along the transcript -- the shape profile --
            needs the pre-mask span, and it cannot be recovered afterwards.
        transcript_map: Many-to-one transcript-to-model mapping, carrying
            ``equivalence_class_size`` and ``exactly_identifiable``.
        loci: One row per locus, from `dienr.loci.locus_table`.
        model_table: One row per identifiable model: ``locus_id``,
            ``model_index``, ``model_id``, ``equivalence_class_size``,
            ``has_support``.
        config: The `DesignConfig` used.
        annotation_digest: Digest of the source annotation.
        shape_digest: Digest of the `dienr.shape.ShapeProfile` applied to
            `models`, or None while they are plain occupancies. Reshaping
            changes what an abundance means, so it must be visible on the
            object and in `digest` rather than inferred from whether the
            matrices happen to hold non-integer values.
        runon_digest: Digest of the `dienr.runon.RunonTail` added past each
            model's PAS, or None. Like `shape_digest`, it changes what a model
            column predicts, so it is recorded on the object and in `digest`.
        digest: Digest over annotation, config and resulting geometry. A change
            to any input yields a new value.
        window_starts: Genomic start of each locus's bin window, or None when
            every window starts at its locus's transcript span (`loci`
            ``start``), as built. `dienr.runon.apply_runon_tail` extends
            minus-strand windows downstream -- to lower coordinates -- so
            their windows start before their spans. Read it through
            `window_start`, never from `loci`.
        contig_map: The ``(old, new)`` contig renaming applied to the
            annotation, or None. Already reflected in `annotation_digest` (the
            contig names changed); kept here so results can record it.
    """

    bins: pl.DataFrame
    models: list[np.ndarray]
    masks: list[np.ndarray]
    bin_lengths: list[np.ndarray]
    model_spans: list[np.ndarray]
    transcript_map: pl.DataFrame
    loci: pl.DataFrame
    model_table: pl.DataFrame
    config: DesignConfig
    annotation_digest: str
    digest: str
    shape_digest: str | None = None
    runon_digest: str | None = None
    contig_map: tuple[tuple[str, str], ...] | None = None
    window_starts: list[int] | None = None

    @property
    def n_loci(self) -> int:
        return len(self.models)

    def bin_count(self, locus_id: int) -> int:
        """Number of bins in `locus_id`, masked bins included."""
        return self.models[locus_id].shape[0]

    def mask(self, locus_id: int) -> np.ndarray:
        """Boolean mask over the bins of `locus_id`; True means masked out.

        A list index, not a filter over `bins`. This is called once per locus
        per sample, so filtering here cost 5.57 s of a 5.92 s fitting stage on
        the 955-locus benchmark -- more than the optimizer itself.
        """
        return self.masks[locus_id]

    def masked_fraction(self, locus_id: int) -> float:
        """Fraction of `locus_id`'s bins removed by masking.

        Worth watching: masking is a union across every transcript in the
        locus, so a short locus with several isoforms can lose most of its
        bins. On the benchmark annotation, the reference's recommended
        ±4 bins discarded 22.7% of all bins and 46% at the worst locus.
        `Design.heavily_masked_loci` lists the offenders.
        """
        mask = self.masks[locus_id]
        return float(mask.sum()) / mask.size if mask.size else 0.0

    def heavily_masked_loci(self, threshold: float = 0.5) -> pl.DataFrame:
        """Loci where masking removed more than `threshold` of the bins.

        Such loci are still fitted, but on much less data than their bin count
        suggests, so the estimate leans harder on the model. Reported rather
        than silently accepted.
        """
        fractions = [self.masked_fraction(i) for i in range(self.n_loci)]
        return (
            self.loci.with_columns(masked_fraction=pl.Series(fractions, dtype=pl.Float64))
            .filter(pl.col("masked_fraction") > threshold)
            .sort("masked_fraction", descending=True)
        )

    @property
    def unsupported_models(self) -> pl.DataFrame:
        """Models with no nonzero bin, which no data can constrain.

        A transcript shorter than roughly half a bin rounds away to an all-zero
        column under ``bin_operation="round"``, and any abundance fits it
        equally well. Such models are reported rather than dropped: silently
        discarding them would hide a transcript, and silently keeping them
        would emit a meaningless estimate. Callers should either lower
        `bin_size` or exclude these transcripts deliberately.
        """
        return self.model_table.filter(~pl.col("has_support"))

    def window_start(self, locus_id: int) -> int:
        """Genomic start of `locus_id`'s first bin. Bins are `bin_size` wide from here.

        The locus's transcript span start unless a run-on tail extended the
        window (see `window_starts`). An index, not a scan.
        """
        starts = getattr(self, "window_starts", None)  # absent on old pickles
        if starts is not None:
            return starts[locus_id]
        return int(self._span_starts[locus_id])

    @cached_property
    def _span_starts(self) -> np.ndarray:
        return self.loci["start"].to_numpy()

    def model_span(self, locus_id: int, model_index: int) -> tuple[int, int]:
        """Pre-mask ``(first_bin, last_bin)`` of one model, or ``(-1, -1)``.

        A list index, never a scan; see `mask`.
        """
        first, last = self.model_spans[locus_id][model_index]
        return int(first), int(last)

    def effective_lengths(self, locus_id: int) -> np.ndarray:
        """Usable width of each bin in `locus_id`, in bases."""
        return self.bin_lengths[locus_id]


def build_design(
    annotation: Annotation,
    *,
    bin_size: int = 250,
    distance: int | None = None,
    mask_start_bins: tuple[int, int] = (0, 0),
    mask_end_bins: tuple[int, int] = (0, 0),
    mask_tss: tuple[int, int] | None = None,
    mask_pas: tuple[int, int] | None = None,
    bin_operation: BinOperation = "round",
) -> Design:
    """Build a reusable `Design` from an `Annotation`.

    Args:
        annotation: Source transcripts, 0-based half-open.
        bin_size: Bin width in bases.
        distance: Locus-grouping gap threshold; defaults to `bin_size`.
        mask_start_bins: ``(inside, outside)`` bins masked at the 5' end.
        mask_end_bins: ``(inside, outside)`` bins masked at the 3' end.
        mask_tss: ``(upstream, downstream)`` **bases** masked around each TSS.
            Preferred over `mask_start_bins`; mutually exclusive with it.
        mask_pas: ``(upstream, downstream)`` bases masked around each PAS.
        bin_operation: Rounding applied to fractional bin occupancies.

    Returns:
        A `Design` whose per-locus matrices are masked and collapsed.
    """
    config = DesignConfig(
        bin_size=bin_size,
        distance=distance,
        mask_start_bins=mask_start_bins,
        mask_end_bins=mask_end_bins,
        mask_tss=mask_tss,
        mask_pas=mask_pas,
        bin_operation=bin_operation,
    )
    return _build(annotation, config)


def _build(annotation: Annotation, config: DesignConfig) -> Design:
    width = config.bin_size
    grouped = assign_loci(annotation.transcripts, distance=config.effective_distance)
    loci = locus_table(grouped)
    n_loci = loci.height

    models: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    bin_lengths: list[np.ndarray] = []
    model_spans: list[np.ndarray] = []
    map_rows: list[dict] = []
    model_rows: list[dict] = []

    # Per-locus transcript slices from one stable sort: within a locus,
    # transcripts keep their input order, exactly as `partition_by(...,
    # maintain_order=True)` gave them. Plain arrays, because building a
    # DataFrame per locus was a third of the whole stage at genome scale
    # (plan 14.4).
    order = np.argsort(grouped["locus_id"].to_numpy(), kind="stable")
    tx_locus = grouped["locus_id"].to_numpy()[order]
    tx_start = grouped["start"].to_numpy()[order]
    tx_end = grouped["end"].to_numpy()[order]
    tx_plus = grouped["strand"].to_numpy()[order] == "+"
    tx_id = grouped["transcript_id"].to_numpy()[order]
    tx_gene = grouped["gene_id"].to_numpy()[order]
    locus_ids = loci["locus_id"].to_numpy()
    first = np.searchsorted(tx_locus, locus_ids, side="left")
    last = np.searchsorted(tx_locus, locus_ids, side="right")
    locus_lo = loci["start"].to_numpy()
    locus_hi = loci["end"].to_numpy()
    locus_names = loci["locus_name"].to_list()
    n_bins_all = -(-(locus_hi - locus_lo) // width)  # ceil division

    for row in range(n_loci):
        locus_id = int(locus_ids[row])
        a, b = first[row], last[row]
        starts, ends, plus = tx_start[a:b], tx_end[a:b], tx_plus[a:b]
        lo, hi = int(locus_lo[row]), int(locus_hi[row])
        n_bins = int(n_bins_all[row])

        bin_starts = lo + np.arange(n_bins, dtype=np.int64) * width
        bin_ends = bin_starts + width
        # The last bin overhangs the locus; its usable extent is shorter.
        effective = np.minimum(bin_ends, hi) - bin_starts

        matrix = _occupancy_matrix(starts, ends, bin_starts, bin_ends, width)

        # Masks come from the *unrounded* occupancy, and rounding happens after.
        # The order matters and is not interchangeable: a transcript whose first
        # bin is less than half covered rounds to zero there, so deriving the
        # mask from the rounded matrix would put its 5' anchor one bin too far
        # in and shift every masked bin with it. That silently changed which
        # models collapsed together on 12 of 124 loci in the benchmark, in both
        # directions, while every small fixture still passed.
        if config.uses_base_windows:
            masked = _mask_bins_by_bases(starts, ends, plus, bin_starts, bin_ends, config)
        else:
            masked = _mask_bins(matrix, plus, config)
        rounded = _BIN_OPERATIONS[config.bin_operation](matrix)
        # Read the occupied span off the rounded matrix before masking zeroes
        # its ends, then keep only the representative columns' spans.
        spans = _occupied_spans(rounded)
        rounded[masked, :] = 0.0

        representatives, class_members = _collapse(rounded)
        reduced = rounded[:, representatives]
        models.append(np.ascontiguousarray(reduced))
        masks.append(masked)
        bin_lengths.append(effective.astype(np.float64))
        model_spans.append(spans[representatives, :])

        transcript_ids = tx_id[a:b]
        gene_ids = tx_gene[a:b]
        for model_index, member_columns in enumerate(class_members):
            size = len(member_columns)
            model_id = _model_id(locus_names[row], reduced[:, model_index])
            has_support = bool(reduced[:, model_index].any())
            model_rows.append(
                {
                    "locus_id": locus_id,
                    "model_index": model_index,
                    "model_id": model_id,
                    "equivalence_class_size": size,
                    "has_support": has_support,
                }
            )
            for column in member_columns:
                map_rows.append(
                    {
                        "transcript_id": transcript_ids[column],
                        "gene_id": gene_ids[column],
                        "locus_id": locus_id,
                        "model_index": model_index,
                        "model_id": model_id,
                        "equivalence_class_size": size,
                        "exactly_identifiable": size == 1,
                        "has_support": has_support,
                    }
                )

    bins = _bins_table(loci, n_bins_all, width, masks)

    transcript_map = pl.DataFrame(
        map_rows,
        schema={
            "transcript_id": pl.String,
            "gene_id": pl.String,
            "locus_id": pl.UInt32,
            "model_index": pl.UInt32,
            "model_id": pl.String,
            "equivalence_class_size": pl.UInt32,
            "exactly_identifiable": pl.Boolean,
            "has_support": pl.Boolean,
        },
    ).sort("locus_id", "model_index", "transcript_id")

    model_table = pl.DataFrame(
        model_rows,
        schema={
            "locus_id": pl.UInt32,
            "model_index": pl.UInt32,
            "model_id": pl.String,
            "equivalence_class_size": pl.UInt32,
            "has_support": pl.Boolean,
        },
    ).sort("locus_id", "model_index")

    return Design(
        bins=bins,
        models=models,
        masks=masks,
        bin_lengths=bin_lengths,
        model_spans=model_spans,
        transcript_map=transcript_map,
        loci=loci,
        model_table=model_table,
        config=config,
        annotation_digest=annotation.digest,
        digest=_design_digest(annotation.digest, config, model_table),
        contig_map=annotation.contig_map,
    )


def _occupied_spans(matrix: np.ndarray) -> np.ndarray:
    """First and last occupied bin per column, as an ``(n_cols, 2)`` int array.

    Columns with no occupied bin -- a transcript shorter than the rounding
    threshold -- get ``(-1, -1)`` rather than a silently wrong range.
    """
    occupied = matrix != 0
    any_occupied = occupied.any(axis=0)
    first = np.argmax(occupied, axis=0)
    # argmax on the reversed rows gives the distance from the end.
    last = occupied.shape[0] - 1 - np.argmax(occupied[::-1, :], axis=0)
    spans = np.stack([first, last], axis=1).astype(np.int64)
    spans[~any_occupied, :] = -1
    return spans


def _bins_table(
    loci: pl.DataFrame, n_bins: np.ndarray, width: int, masks: list[np.ndarray]
) -> pl.DataFrame:
    """The design's bin table, built once for every locus."""
    total = int(n_bins.sum())
    locus_index = np.repeat(np.arange(loci.height), n_bins)
    offsets = np.concatenate([[0], np.cumsum(n_bins)[:-1]]).astype(np.int64)
    bin_index = np.arange(total, dtype=np.int64) - np.repeat(offsets, n_bins)
    lo = loci["start"].to_numpy().astype(np.int64)[locus_index]
    hi = loci["end"].to_numpy().astype(np.int64)[locus_index]
    starts = lo + bin_index * width
    ends = starts + width
    return pl.DataFrame(
        {
            "locus_id": loci["locus_id"].to_numpy()[locus_index].astype(np.uint32),
            "bin_index": bin_index.astype(np.uint32),
            "contig": loci["contig"].gather(pl.Series(locus_index, dtype=pl.UInt32)),
            "start": starts,
            "end": ends,
            "effective_length": np.minimum(ends, hi) - starts,
            "masked": np.concatenate(masks) if masks else np.zeros(0, dtype=bool),
        },
        schema={
            "locus_id": pl.UInt32,
            "bin_index": pl.UInt32,
            "contig": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "effective_length": pl.Int64,
            "masked": pl.Boolean,
        },
    )


def _occupancy_matrix(
    tx_starts: np.ndarray,
    tx_ends: np.ndarray,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    width: int,
) -> np.ndarray:
    """Fractional occupancy of each bin by each transcript.

    Returns a ``(n_bins, n_transcripts)`` array of ``overlap / width``. Dividing
    by the nominal width rather than the effective length is deliberate: it
    keeps a fully covered bin at exactly 1.0 and matches the published method.
    """
    starts = tx_starts[None, :]
    ends = tx_ends[None, :]
    overlap = np.minimum(bin_ends[:, None], ends) - np.maximum(bin_starts[:, None], starts)
    np.maximum(overlap, 0, out=overlap)
    return overlap.astype(np.float64) / width


def _mask_bins(
    matrix: np.ndarray,
    plus: np.ndarray,
    config: DesignConfig,
) -> np.ndarray:
    """Union of per-transcript masked bins for one locus.

    For each transcript the first and last occupied bins are located, then bins
    are masked inward and outward from each end. Which end counts as 5' depends
    on strand, so a minus-strand transcript takes `mask_start_bins` at its
    highest-coordinate bin.

    Vectorized over transcripts: each of the four marks is one broadcast of
    ``origin + step * arange(count)``. Marking one transcript at a time was
    326k calls at genome scale (plan 14.4).

    Returns a boolean array over bins; True means masked.
    """
    n_bins = matrix.shape[0]
    masked = np.zeros(n_bins, dtype=bool)
    start_in, start_out = config.mask_start_bins
    end_in, end_out = config.mask_end_bins
    if start_in == start_out == end_in == end_out == 0:
        return masked

    occupied = matrix != 0
    present = occupied.any(axis=0)
    low = np.argmax(occupied, axis=0)[present]
    high = (n_bins - 1 - np.argmax(occupied[::-1, :], axis=0))[present]
    plus = np.asarray(plus, dtype=bool)[present]

    # On the minus strand the 5' end sits at the higher coordinate, so
    # "inward" runs toward lower bin indices.
    five_prime = np.where(plus, low, high)
    three_prime = np.where(plus, high, low)
    inward = np.where(plus, 1, -1)
    outward = -inward

    _mark(masked, five_prime, start_in, inward)
    _mark(masked, five_prime + outward, start_out, outward)
    _mark(masked, three_prime, end_in, -inward)
    _mark(masked, three_prime - outward, end_out, -outward)
    return masked


def _mask_bins_by_bases(
    tx_starts: np.ndarray,
    tx_ends: np.ndarray,
    plus: np.ndarray,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    config: DesignConfig,
) -> np.ndarray:
    """Union of masked bins from base-specified TSS and PAS windows.

    A bin is masked when it overlaps the genomic window around any
    transcript's TSS or PAS. Windows are given in the direction of
    transcription, so ``mask_tss=(1000, 1000)`` masks 1 kb either side of the
    TSS on both strands, with the genomic orientation flipped for minus-strand
    transcripts.

    Windows are clipped to the locus, because bins exist only inside it. An
    isolated transcript's locus starts at its own TSS, so an upstream window
    has nothing to act on there; it bites only where another transcript
    extends the locus further back. Nothing is silently lost -- there is no
    data outside the locus to discard -- but a caller comparing masked spans
    between loci should expect that asymmetry.

    The union across transcripts is deliberate and not a limitation. Masking
    encodes "the observed signal in this bin is untrustworthy", which is a
    property of the bin, not of a model: an initiation spike from one
    transcript's TSS contaminates that bin for every transcript running
    through it. Masking per model while leaving the shared count vector intact
    would ask the surviving models to explain signal they did not generate.
    The cost is real -- see `Design.masked_fraction` -- but it is the correct
    treatment, and the way to reduce it is to mask a narrower window or to
    exclude transcripts known to be inactive, not to narrow the union.

    Returns a boolean array over bins; True means masked.
    """
    masked = np.zeros(bin_starts.size, dtype=bool)
    plus = np.asarray(plus, dtype=bool)
    # Half-open spans: the first transcribed base is `start`, the last is
    # `end - 1`. Which of those is the TSS depends on strand.
    tss = np.where(plus, tx_starts, tx_ends - 1)
    pas = np.where(plus, tx_ends - 1, tx_starts)

    for window, anchor in ((config.mask_tss, tss), (config.mask_pas, pas)):
        # (0, 0) disables that end, matching how the legacy bin-count form
        # treats it. Without this it would still mask the single base at
        # the anchor, and hence its whole bin, which is not what a caller
        # writing (0, 0) means.
        if window is None or window == (0, 0):
            continue
        upstream, downstream = window
        # Upstream is the higher coordinate on the minus strand.
        lo = np.where(plus, anchor - upstream, anchor - downstream)
        # Half-open window centred so that (u, d) is symmetric when u == d.
        hi = np.where(plus, anchor + downstream, anchor + upstream) + 1
        keep = hi > lo
        lo, hi = lo[keep], hi[keep]
        hits = (bin_starts[:, None] < hi[None, :]) & (bin_ends[:, None] > lo[None, :])
        masked |= hits.any(axis=1)

    return masked


def _mark(masked: np.ndarray, origin: np.ndarray, count: int, step: np.ndarray) -> None:
    """Mask `count` bins from each `origin` in its direction `step`, clipped."""
    if count <= 0 or origin.size == 0:
        return
    indices = (origin[:, None] + step[:, None] * np.arange(count)[None, :]).ravel()
    indices = indices[(indices >= 0) & (indices < masked.size)]
    masked[indices] = True


def _collapse(matrix: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """Group exactly identical columns of `matrix`.

    Returns the representative column index per equivalence class, and the
    member columns of each class. Classes appear in order of first occurrence,
    and the representative is the lowest-indexed member, so the result is
    deterministic.

    Hashing each column's bytes makes this linear in the number of columns,
    where the reference implementation is quadratic.
    """
    classes: dict[bytes, list[int]] = {}
    for column in range(matrix.shape[1]):
        key = np.ascontiguousarray(matrix[:, column]).tobytes()
        classes.setdefault(key, []).append(column)
    members = list(classes.values())
    return [group[0] for group in members], members


def _model_id(locus_name: str, column: np.ndarray) -> str:
    """Stable content-derived model identifier.

    Derived from the locus span and the model's own values, so an unchanged
    model keeps its identifier across runs and across annotations that happen
    to share the locus.
    """
    hasher = hashlib.sha256()
    hasher.update(b"dienr.model.v1\x00")
    hasher.update(locus_name.encode())
    hasher.update(b"\x00")
    hasher.update(np.ascontiguousarray(column, dtype=np.float64).tobytes())
    return hasher.hexdigest()[:16]


def _design_digest(annotation_digest: str, config: DesignConfig, model_table: pl.DataFrame) -> str:
    """Digest over annotation, configuration and resulting model identities."""
    hasher = hashlib.sha256()
    hasher.update(b"dienr.design.v1\x00")
    hasher.update(annotation_digest.encode())
    hasher.update(
        repr(
            (
                config.bin_size,
                config.effective_distance,
                config.mask_start_bins,
                config.mask_end_bins,
                config.mask_tss,
                config.mask_pas,
                config.bin_operation,
            )
        ).encode()
    )
    hasher.update(b"\x00")
    hasher.update("\x00".join(model_table["model_id"].to_list()).encode())
    return hasher.hexdigest()[:16]
