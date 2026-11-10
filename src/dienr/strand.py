"""Strand-orientation check: are the plus and minus files the right way round?

`dienr` takes each file's strand from what the user declares and never infers
it from the sign of the values. That is the right policy -- a sign is only a
convention -- but it means a swapped pair is undetectable from the files
alone. A swap is also the easiest conversion mistake to make: single-end
PRO-seq reads are the reverse complement of the RNA, so a converter that
records each read on its own alignment strand writes minus-strand transcription
into the "plus" file, and both files still look like perfectly good nascent RNA
signal.

The annotation can tell. Over gene bodies, stranded nascent RNA is dominated by
sense transcription; antisense signal comes mostly from divergent transcription
near promoters and from overlapping opposite-strand genes. So read each
locus's window from *both* files and compare: correctly oriented data has most
of its signal in the declared-sense file, and a swapped pair has most of it in
the other one.

Loci that overlap an opposite-strand locus are left out, since there the
"antisense" file legitimately carries another gene's sense signal. Measured
(plan 18.16): K562 PRO-seq (GSE96869), 0.876 / 0.879 for correctly labelled
files and 0.124 / 0.121 swapped; nascentRNASim, 0.958-0.963 and 0.037-0.042.
The thresholds below sit well inside that gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .counts import SignalFile, open_bigwig, read_loci_binned
from .design import Design

Verdict = Literal["ok", "low", "swapped", "insufficient"]

#: Below this sense fraction the files are taken to be swapped.
REFUSE_BELOW = 0.5
#: Below this (but above `REFUSE_BELOW`) the orientation is suspicious.
WARN_BELOW = 0.7
#: Fewer informative loci than this and no verdict is given.
MIN_LOCI = 20


@dataclass(frozen=True)
class StrandCheck:
    """Outcome of `check_strand_orientation`.

    Attributes:
        sense: Total signal read from each locus's declared-sense file.
        antisense: Total signal from the opposite file over the same windows.
        n_loci: Loci with any signal in either file that entered the totals.
        locus_agreement: Fraction of those loci where sense exceeds antisense,
            a check that does not let a few very deep loci decide.
        verdict: ``"ok"``, ``"low"`` (between the thresholds), ``"swapped"``,
            or ``"insufficient"`` (too few informative loci to judge).
    """

    sense: float
    antisense: float
    n_loci: int
    locus_agreement: float
    verdict: Verdict

    @property
    def sense_fraction(self) -> float:
        total = self.sense + self.antisense
        return self.sense / total if total > 0 else float("nan")

    def describe(self) -> dict[str, float | int | str]:
        """Provenance record."""
        return {
            "strand_check_sense_fraction": self.sense_fraction,
            "strand_check_locus_agreement": self.locus_agreement,
            "strand_check_n_loci": self.n_loci,
            "strand_check_verdict": self.verdict,
        }

    def message(self) -> str:
        """One line for a log or an error."""
        return (
            f"strand check: {self.sense_fraction:.3f} of gene-body signal is in the declared-"
            f"sense file, sense > antisense at {self.locus_agreement:.2f} of {self.n_loci} loci "
            f"-> {self.verdict}"
        )


def check_strand_orientation(
    design: Design,
    signal: tuple[SignalFile, ...],
    *,
    max_loci: int = 2_000,
    warn_below: float = WARN_BELOW,
    refuse_below: float = REFUSE_BELOW,
    n_jobs: int = 8,
) -> StrandCheck:
    """Compare declared-sense and opposite-strand signal over gene bodies.

    Args:
        design: Supplies the locus windows (whole windows; masks are ignored,
            since a mask says a bin is untrustworthy for fitting, not that it
            is on the wrong strand).
        signal: Exactly one `SignalFile` per strand.
        max_loci: Cap on loci read, taken evenly across the design, so the
            check costs a fraction of counting.
        warn_below: Sense fraction below which the verdict is ``"low"``.
        refuse_below: Sense fraction below which the verdict is ``"swapped"``.
        n_jobs: Reader threads.

    Returns:
        A `StrandCheck`. It never raises on the verdict; callers decide.

    Raises:
        ValueError: Unless there is exactly one file per strand, or if the
            thresholds are out of order.
    """
    if not 0.0 <= refuse_below <= warn_below <= 1.0:
        raise ValueError(
            f"need 0 <= refuse_below ({refuse_below}) <= warn_below ({warn_below}) <= 1"
        )
    by_strand: dict[str, SignalFile] = {item.strand: item for item in signal}
    if len(signal) != 2 or set(by_strand) != {"+", "-"}:
        raise ValueError("the strand check needs exactly one plus and one minus signal file")
    readers = {strand: open_bigwig(item.path) for strand, item in by_strand.items()}
    headers = {strand: dict(reader.chrom_sizes) for strand, reader in readers.items()}

    candidates = _single_strand_loci(design)
    if len(candidates) > max_loci:
        picks = np.linspace(0, len(candidates) - 1, max_loci).astype(int)
        candidates = [candidates[i] for i in picks]

    bin_size = design.config.bin_size
    windows: dict[str, list[tuple[str, int, int]]] = {"+": [], "-": []}
    for strand, contig, start, n_bins in candidates:
        end = start + n_bins * bin_size
        if all(end <= headers[s].get(contig, -1) for s in ("+", "-")):
            windows[strand].append((contig, start, n_bins))

    sense_totals: list[float] = []
    anti_totals: list[float] = []
    for strand, items in windows.items():
        if not items:
            continue
        other = "-" if strand == "+" else "+"
        sense = read_loci_binned(readers[strand], items, bin_size, summary="sum", n_jobs=n_jobs)
        anti = read_loci_binned(readers[other], items, bin_size, summary="sum", n_jobs=n_jobs)
        sense_totals.extend(float(v.sum()) for v in sense)
        anti_totals.extend(float(v.sum()) for v in anti)

    s = np.asarray(sense_totals)
    a = np.asarray(anti_totals)
    informative = (s + a) > 0
    s, a = s[informative], a[informative]
    n = int(informative.sum())
    total_sense, total_anti = float(s.sum()), float(a.sum())
    agreement = float(np.mean(s > a)) if n else float("nan")
    fraction = total_sense / (total_sense + total_anti) if n else float("nan")

    verdict: Verdict
    if n < MIN_LOCI:
        verdict = "insufficient"
    elif fraction < refuse_below:
        verdict = "swapped"
    elif fraction < warn_below:
        verdict = "low"
    else:
        verdict = "ok"
    return StrandCheck(total_sense, total_anti, n, agreement, verdict)


def _single_strand_loci(design: Design) -> list[tuple[str, str, int, int]]:
    """``(strand, contig, start, n_bins)`` of loci no opposite-strand locus overlaps."""
    loci = design.loci.sort("contig", "start")
    out: list[tuple[str, str, int, int]] = []
    for contig in loci["contig"].unique(maintain_order=True).to_list():
        rows = loci.filter(loci["contig"] == contig)
        ids = rows["locus_id"].to_numpy()
        starts = rows["start"].to_numpy()
        ends = rows["end"].to_numpy()
        strands = rows["strand"].to_numpy()
        for strand in ("+", "-"):
            mine = strands == strand
            other = ~mine
            o_start = np.sort(starts[other])
            o_end = np.sort(ends[other])
            s, e = starts[mine], ends[mine]
            # Opposite-strand loci overlapping [s, e): started before e, not ended by s.
            overlaps = np.searchsorted(o_start, e, "left") - np.searchsorted(o_end, s, "right")
            for lid, start, keep in zip(ids[mine], s, overlaps == 0, strict=True):
                if keep:
                    out.append((strand, contig, int(start), design.bin_count(int(lid))))
    out.sort(key=lambda row: (row[1], row[2]))
    return out
