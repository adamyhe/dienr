"""Run-on (read-through) tails: polymerase signal past each isoform's PAS.

Polymerase does not stop at the annotated polyadenylation site. It slows,
accumulates in a termination zone, and is released over several kilobases, so
PRO-seq signal continues well past each transcript's 3' end. A binary design
says that signal belongs to nothing -- or, inside a locus, to whichever longer
isoform's body it happens to land in. Read-through from a short isoform then
looks like evidence for a longer one, which is the confusion the reference's
PAS mask (4 bins, 1 kb at 250 bp) only partly papers over: the termination
zone extends well beyond it.

This module estimates one shared run-on tail ``r(d)`` -- signal in the
``d``-th bin past the PAS, in units of the gene's own body density -- from
isolated, expressed single-isoform genes, and adds it to every model column of
a `Design`, past that model's PAS and inside its locus window.

Evidence (plan 18.10, 18.14): on the DENR authors' simulator the fitted tail
raised dominant-isoform accuracy from 0.70 to 0.83 against a binary Poisson
design and lowered per-locus error at ~60% of loci; on real K562 PRO-seq, with
the tail fitted on one replicate and scored on the other, it lowered held-out
Poisson deviance by 5.5% and won at 57% of loci. The real tail peaks around
1.4x body density ~1.25 kb past the PAS and is still ~0.15x at 15 kb, so the
default length is 15 kb.

Three properties make the operation safe to apply everywhere:

- **It adds no parameter.** The tail is fixed before fitting; each model keeps
  one abundance.
- **It never changes which bins are fitted.** A locus window is the union of
  its transcripts' spans, so a tail bin inside it is always covered by some
  other model already (plan 18.12). The tail re-attributes signal in bins the
  binary design was fitting anyway.
- **Single-model loci are untouched**: with no other model, every tail bin
  falls outside the window.

Use it with a count likelihood (`poisson`). It was not validated under the
legacy least-squares objectives, and combining it with a shape profile was
tested and found not worth it (plan 18.13).
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from .annotation import Annotation
from .counts import SignalFile, open_bigwig, read_loci_binned
from .design import (
    Design,
    LocusTranscripts,
    build_locus,
    model_table_frame,
    transcript_map_frame,
)

#: Fewer training genes than this triggers a warning: the tail is a ratio of
#: sums, so a handful of genes lets one gene's termination set it.
MIN_TRAINING_GENES = 20


@dataclass(frozen=True)
class RunonConfig:
    """Run-on tail parameters.

    Attributes:
        bin_size: Bin width in bases. Must equal the design's: the tail is a
            per-bin vector added to bins.
        tail_bases: Length of the tail past the PAS, a multiple of `bin_size`.
            15 kb by default because the real K562 tail still carried ~0.15x
            body density there and 27% of its mass lay beyond the 6 kb the
            DENR authors' simulator models; truncating at 6 kb kept only ~58%
            of the held-out-deviance gain (plan 18.14).
        buffer_bases: Isolation margin. A training gene is rejected if any
            other same-strand transcript comes within this distance of its
            span or its tail window, since that transcript's signal would be
            read as run-on. Opposite-strand transcripts are ignored: stranded
            nascent RNA data cannot put their reads on this strand.
        body_skip_5p: Bases after the TSS left out of the body density, so the
            promoter-proximal pause does not inflate the denominator.
        body_skip_3p: Bases before the PAS left out of the body density,
            matching the usual PAS mask.
        min_length: Shortest training gene, in bases. Must leave a body after
            the two skips.
        min_body_signal: Smallest mean *summed* signal per body bin (reads per
            bin for count data) for a gene to train the tail, so near-silent
            genes do not add noise to the denominator. The tail itself is a
            ratio and is unaffected by the summary's scale.
    """

    bin_size: int = 250
    tail_bases: int = 15_000
    buffer_bases: int = 5_000
    body_skip_5p: int = 250
    body_skip_3p: int = 1_000
    min_length: int = 5_000
    min_body_signal: float = 2.0

    def __post_init__(self) -> None:
        if self.bin_size <= 0:
            raise ValueError(f"bin_size must be positive, got {self.bin_size}")
        if self.tail_bases <= 0 or self.tail_bases % self.bin_size:
            raise ValueError(
                f"tail_bases must be a positive multiple of bin_size ({self.bin_size}), "
                f"got {self.tail_bases}"
            )
        for name in ("buffer_bases", "body_skip_5p", "body_skip_3p"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.min_length < self.body_skip_5p + self.body_skip_3p + self.bin_size:
            raise ValueError(
                f"min_length ({self.min_length}) must leave at least one body bin after "
                f"body_skip_5p + body_skip_3p ({self.body_skip_5p + self.body_skip_3p})"
            )
        if self.min_body_signal < 0:
            raise ValueError(f"min_body_signal must be non-negative, got {self.min_body_signal}")

    @property
    def n_bins(self) -> int:
        """Tail length in bins."""
        return self.tail_bases // self.bin_size


@dataclass(frozen=True)
class RunonTail:
    """A fitted run-on tail.

    Attributes:
        values: ``r(d)`` for ``d = 1 .. config.n_bins``: mean signal in the
            ``d``-th bin past the PAS, as a multiple of body density.
        config: Parameters it was fitted with.
        n_genes: Training genes used.
        rejected: Candidate genes dropped, per reason.
        digest: Digest of config and values; enters the design digest.
    """

    values: np.ndarray
    config: RunonConfig
    n_genes: int
    rejected: dict[str, int] = field(default_factory=dict)
    digest: str = ""

    @property
    def mass(self) -> float:
        """Total tail signal, in bins of body density."""
        return float(self.values.sum())

    def table(self) -> pl.DataFrame:
        """One row per tail bin, distances in bases past the PAS."""
        d = np.arange(self.values.size)
        return pl.DataFrame(
            {
                "bin": d + 1,
                "distance_start": d * self.config.bin_size,
                "distance_end": (d + 1) * self.config.bin_size,
                "relative_density": self.values,
            }
        )

    def describe(self) -> dict[str, float | str | int]:
        """Provenance record for the manifest."""
        return {
            "runon_digest": self.digest,
            "runon_n_genes": self.n_genes,
            "runon_tail_bases": self.config.tail_bases,
            "runon_mass": self.mass,
            "runon_peak": float(self.values.max()),
        }


def fit_runon_tail(
    annotation: Annotation,
    signal: SignalFile | tuple[SignalFile, ...],
    *,
    config: RunonConfig | None = None,
    n_jobs: int = 8,
) -> RunonTail:
    """Estimate the shared run-on tail from isolated, expressed genes.

    ``r(d) = sum_genes signal(d) / sum_genes body_density`` -- a ratio of sums,
    which weights genes by expression and so stays stable under Poisson noise
    where a mean of per-gene ratios would be dominated by weak genes.

    A training gene is a transcript with no other same-strand transcript
    within `RunonConfig.buffer_bases` of its span or of its tail window (so
    multi-isoform genes are excluded), at least `RunonConfig.min_length`
    long, with its whole tail window on the contig, and with body signal of at
    least `RunonConfig.min_body_signal`. Each gene is read from the file for
    its own strand; signs are normalized by the reader.

    Args:
        annotation: Candidate training transcripts -- normally the same
            annotation the design is built from.
        signal: One `SignalFile` per strand.
        config: Parameters; defaults to `RunonConfig()`.
        n_jobs: Reader threads.

    Returns:
        A `RunonTail`.

    Raises:
        ValueError: If a strand with candidates has no signal file, two files
            claim one strand, or no gene survives the filters. The last is an
            error rather than a zero tail, which would apply silently and
            change nothing.
    """
    config = config or RunonConfig()
    files = (signal,) if isinstance(signal, SignalFile) else tuple(signal)
    by_strand: dict[str, SignalFile] = {}
    for item in files:
        if item.strand in by_strand:
            raise ValueError(f"two signal files given for strand {item.strand!r}")
        by_strand[item.strand] = item

    candidates, rejected = _select_training_genes(annotation, config)
    needed = set(candidates["strand"].unique().to_list()) if len(candidates) else set()
    missing = needed - set(by_strand)
    if missing:
        raise ValueError(f"training genes lie on strand(s) {sorted(missing)} with no signal file")

    span = config.tail_bases
    size = config.bin_size
    numerator = np.zeros(config.n_bins)
    denominator = 0.0
    n_genes = 0
    for strand, item in by_strand.items():
        genes = candidates.filter(pl.col("strand") == strand)
        if not len(genes):
            continue
        reader = open_bigwig(item.path)
        sizes = reader.chrom_sizes
        tails: list[tuple[str, int, int]] = []
        bodies: list[tuple[str, int, int]] = []
        for gene in genes.iter_rows(named=True):
            contig, start, end = gene["contig"], int(gene["start"]), int(gene["end"])
            if contig not in sizes:
                rejected["missing_contig"] = rejected.get("missing_contig", 0) + 1
                continue
            if strand == "+":
                tail_start = end
                body_start = start + config.body_skip_5p
                body_end = end - config.body_skip_3p
            else:
                tail_start = start - span
                body_start = start + config.body_skip_3p
                body_end = end - config.body_skip_5p
            if tail_start < 0 or tail_start + span > sizes[contig]:
                rejected["tail_off_contig"] = rejected.get("tail_off_contig", 0) + 1
                continue
            tails.append((contig, tail_start, config.n_bins))
            bodies.append((contig, body_start, max((body_end - body_start) // size, 1)))
        if not tails:
            continue
        tail_values = read_loci_binned(reader, tails, size, summary="sum", n_jobs=n_jobs)
        body_values = read_loci_binned(reader, bodies, size, summary="sum", n_jobs=n_jobs)
        for tail, body in zip(tail_values, body_values, strict=True):
            density = float(body.mean())
            if density < config.min_body_signal:
                rejected["low_signal"] = rejected.get("low_signal", 0) + 1
                continue
            # Orient so d = 1 is the bin adjacent to the PAS.
            numerator += tail if strand == "+" else tail[::-1]
            denominator += density
            n_genes += 1

    if n_genes == 0:
        reasons = ", ".join(f"{k}: {v}" for k, v in sorted(rejected.items()) if v)
        raise ValueError(
            f"no gene passed the run-on filters ({reasons or 'no candidates'}). Lower "
            "min_body_signal or buffer_bases, or check that the signal files match the "
            "annotation's contigs and strands."
        )
    if n_genes < MIN_TRAINING_GENES:
        warnings.warn(
            f"run-on tail fitted from only {n_genes} genes; one gene's termination can dominate it",
            stacklevel=2,
        )
    values = numerator / denominator
    return RunonTail(
        values=values,
        config=config,
        n_genes=n_genes,
        rejected=dict(sorted(rejected.items())),
        digest=_tail_digest(config, values),
    )


def apply_runon_tail(design: Design, tail: RunonTail, *, extend: bool = True) -> Design:
    """Add `tail` past every model's PAS, returning a new `Design`.

    For each transcript the tail starts at the first bin past its own
    pre-mask span in the direction of transcription; masks are drawn over the
    whole extended window and masked bins stay zero; and equivalence classes
    are formed afterwards, so transcripts whose PAS bins differ get separate
    models even if masking made their bodies identical. Each locus is rebuilt
    from `Design.transcripts` by the same `dienr.design.build_locus` that
    built it, so `model_table` and `transcript_map` (model ids, class sizes,
    `has_support`) describe the tailed columns. The design digest changes and
    `Design.runon_digest` records the tail, so a run-on design can never be
    mistaken for its source.

    **Each locus window is first extended downstream by the tail's length**
    (`extend`, default True), in whole bins, stopping short of the next
    same-strand locus's window and of coordinate 0. Without it the window
    ends at the most downstream PAS, so that isoform's tail falls outside it
    while every shorter isoform's tail lands on the longer isoform's 3' body
    -- a bias toward shorter isoforms that decided the wrong dominant isoform
    at loci with nearby alternative PAS sites (plan 18.25). With it, every
    model's tail sits on observed signal. Bin boundaries are unchanged: plus
    windows gain bins at the end, minus windows at the start (so their
    `Design.window_start` moves and their spans shift by the bins added).
    ``extend=False`` reproduces the earlier, biased behaviour.

    Apply it before `dienr.shape.apply_shape_profile`, never after: the
    profile reweights only a model's own span, which the tail lies outside,
    so that order gives what either order used to, but rebuilding a reshaped
    design would discard its reshaping. The combination is not recommended
    (plan 18.13).

    Raises:
        ValueError: If the bin sizes differ, `design` already carries a tail or
            a shape profile, or it does not record its transcripts.
    """
    if getattr(design, "runon_digest", None) is not None:
        raise ValueError(
            f"design already carries run-on tail {design.runon_digest}; rebuild it from "
            "the annotation to apply a different one"
        )
    if design.shape_digest is not None:
        raise ValueError(
            "apply the run-on tail before the shape profile: the tail rebuilds each "
            "locus from its transcripts, which would discard the reshaping"
        )
    width = design.config.bin_size
    if width != tail.config.bin_size:
        raise ValueError(f"tail was fitted at bin size {tail.config.bin_size}, design uses {width}")
    n_tail = tail.values.size
    grow = _downstream_room(design, n_tail) if extend else np.zeros(design.n_loci, np.int64)

    if design.transcripts is None:
        raise ValueError(
            "design does not record its transcripts' coordinates (built by an older "
            "dienr); rebuild it with build-design to apply a run-on tail"
        )

    # Each locus is rebuilt from its transcripts over the extended window,
    # rather than by padding the collapsed design. Padding left the new bins
    # unmasked, so a downstream PAS mask missed exactly the isoform the
    # extension exists for; and it placed one tail per equivalence class at
    # the representative's PAS, though masking can hide a difference in PAS
    # between members whose tails then differ (plan 14.6).
    transcripts = design.transcripts
    tx_locus = transcripts["locus_id"].to_numpy()
    tx_start = transcripts["start"].to_numpy()
    tx_end = transcripts["end"].to_numpy()
    tx_plus = transcripts["strand"].to_numpy() == "+"
    tx_id = transcripts["transcript_id"].to_numpy()
    tx_gene = transcripts["gene_id"].to_numpy()
    locus_ids = design.loci["locus_id"].to_numpy()
    first = np.searchsorted(tx_locus, locus_ids, side="left")
    last = np.searchsorted(tx_locus, locus_ids, side="right")
    strands = design.loci["strand"].to_list()
    locus_ends = design.loci["end"].to_numpy()
    names = design.loci["locus_name"].to_list()

    models: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    bin_lengths: list[np.ndarray] = []
    model_spans: list[np.ndarray] = []
    window_starts: list[int] = []
    model_rows: list[dict] = []
    map_rows: list[dict] = []
    for row, locus_id in enumerate(locus_ids):
        lid = int(locus_id)
        plus = strands[lid] == "+"
        k = int(grow[lid])
        n_bins = design.bin_count(lid) + k
        start = design.window_start(lid) - (0 if plus else k * width)
        # Plus windows grow past the locus end, so the overhanging last bin
        # gains a real continuation; minus windows grow at the low end.
        clip_end = start + n_bins * width if plus and k else int(locus_ends[lid])
        a, b = first[row], last[row]
        built = build_locus(
            LocusTranscripts(
                starts=tx_start[a:b],
                ends=tx_end[a:b],
                plus=tx_plus[a:b],
                transcript_ids=tx_id[a:b],
                gene_ids=tx_gene[a:b],
            ),
            locus_id=lid,
            locus_name=names[lid],
            window_start=start,
            n_bins=n_bins,
            clip_end=clip_end,
            config=design.config,
            tail=tail.values,
        )
        models.append(built.model)
        masks.append(built.mask)
        bin_lengths.append(built.bin_lengths)
        model_spans.append(built.spans)
        window_starts.append(int(start))
        model_rows.extend(built.model_rows)
        map_rows.extend(built.map_rows)
    table = _bins_frame(design, models, masks, bin_lengths, window_starts)
    return replace(
        design,
        models=models,
        masks=masks,
        bin_lengths=bin_lengths,
        model_spans=model_spans,
        window_starts=window_starts,
        bins=table,
        model_table=model_table_frame(model_rows),
        transcript_map=transcript_map_frame(map_rows),
        runon_digest=tail.digest,
        digest=_combined_digest(design.digest, tail.digest + ("|extended" if extend else "")),
    )


def _downstream_room(design: Design, n_tail: int) -> np.ndarray:
    """Bins each locus window may grow downstream: up to `n_tail`, in whole bins.

    Stops short of the nearest same-strand locus window downstream (so no read
    is counted twice) and, on the minus strand, of coordinate 0. A window may
    still run past the contig end; counting clips there, as it already does.
    """
    width = design.config.bin_size
    loci = design.loci
    ids = loci["locus_id"].to_numpy()
    contigs = loci["contig"].to_numpy()
    strands = loci["strand"].to_numpy()
    starts = np.array([design.window_start(int(i)) for i in ids], dtype=np.int64)
    n_bins = np.array([design.bin_count(int(i)) for i in ids], dtype=np.int64)
    ends = starts + n_bins * width
    room = np.full(design.n_loci, n_tail, dtype=np.int64)
    for contig in np.unique(contigs):
        for strand in ("+", "-"):
            sel = np.flatnonzero((contigs == contig) & (strands == strand))
            if not sel.size:
                continue
            order = sel[np.argsort(starts[sel], kind="stable")]
            if strand == "+":
                # Downstream is higher coordinates: the gap to the next window.
                gap = np.append(starts[order][1:] - ends[order][:-1], np.iinfo(np.int64).max)
            else:
                # Downstream is lower coordinates: the gap to the previous
                # window, or to coordinate 0 for the first.
                gap = starts[order] - np.append(0, ends[order][:-1])
            room[ids[order]] = np.minimum(n_tail, np.maximum(gap, 0) // width)
    return room


def _bins_frame(
    design: Design,
    models: list[np.ndarray],
    masks: list[np.ndarray],
    bin_lengths: list[np.ndarray],
    window_starts: list[int],
) -> pl.DataFrame:
    """The bin table for windows that may no longer start at their loci."""
    width = design.config.bin_size
    n_bins = np.array([m.shape[0] for m in models], dtype=np.int64)
    locus_index = np.repeat(np.arange(design.n_loci), n_bins)
    offsets = np.concatenate([[0], np.cumsum(n_bins)[:-1]]).astype(np.int64)
    bin_index = np.arange(int(n_bins.sum()), dtype=np.int64) - np.repeat(offsets, n_bins)
    starts = np.asarray(window_starts, dtype=np.int64)[locus_index] + bin_index * width
    return pl.DataFrame(
        {
            "locus_id": design.loci["locus_id"].to_numpy()[locus_index].astype(np.uint32),
            "bin_index": bin_index.astype(np.uint32),
            "contig": design.loci["contig"].gather(pl.Series(locus_index, dtype=pl.UInt32)),
            "start": starts,
            "end": starts + width,
            "effective_length": np.concatenate(bin_lengths).astype(np.int64)
            if bin_lengths
            else np.zeros(0, np.int64),
            "masked": np.concatenate(masks) if masks else np.zeros(0, dtype=bool),
        },
        schema=design.bins.schema,
    )


def _select_training_genes(
    annotation: Annotation, config: RunonConfig
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Isolated, long-enough transcripts; returns them and a rejection tally."""
    transcripts = annotation.transcripts.select("transcript_id", "contig", "start", "end", "strand")
    # Identical spans are one pre-RNA; keep one so it does not count as its own neighbour.
    unique = transcripts.unique(subset=["contig", "strand", "start", "end"], keep="first")
    rejected = {"duplicate_span": len(transcripts) - len(unique)}
    long_enough = unique.filter((pl.col("end") - pl.col("start")) >= config.min_length)
    rejected["too_short"] = len(unique) - len(long_enough)

    keep: list[pl.DataFrame] = []
    n_crowded = 0
    for (contig, strand), group in unique.group_by("contig", "strand"):
        starts = np.sort(group["start"].to_numpy())
        ends = np.sort(group["end"].to_numpy())
        cand = long_enough.filter((pl.col("contig") == contig) & (pl.col("strand") == strand))
        if not len(cand):
            continue
        s, e = cand["start"].to_numpy(), cand["end"].to_numpy()
        if strand == "+":
            lo, hi = s - config.buffer_bases, e + config.tail_bases + config.buffer_bases
        else:
            lo, hi = s - config.tail_bases - config.buffer_bases, e + config.buffer_bases
        # Transcripts overlapping [lo, hi): starts before hi minus those ending by lo.
        overlapping = np.searchsorted(starts, hi, "left") - np.searchsorted(ends, lo, "right")
        isolated = overlapping == 1  # only itself
        n_crowded += int((~isolated).sum())
        keep.append(cand.filter(pl.Series(isolated)))
    rejected["neighbour_within_buffer"] = n_crowded
    selected = pl.concat(keep) if keep else long_enough.head(0)
    return selected.sort("contig", "start", "transcript_id"), rejected


def _tail_digest(config: RunonConfig, values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for name in sorted(config.__dataclass_fields__):
        hasher.update(f"{name}={getattr(config, name)}".encode())
    hasher.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return hasher.hexdigest()[:16]


def _combined_digest(design_digest: str, tail_digest: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(design_digest.encode())
    hasher.update(b"|runon|")
    hasher.update(tail_digest.encode())
    return hasher.hexdigest()[:16]
