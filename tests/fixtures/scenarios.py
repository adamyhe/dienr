"""Declarative parity scenarios, shared by the generator and the parity tests.

These specifications are the source of truth. Input files (GTF and BigWig) are
**generated on demand** from them rather than committed, and deliberately are
not copied from the R reference package: its ``LICENSE.md`` (MIT) and
``DESCRIPTION`` (GPL >=2) disagree, so redistributing its ``inst/extdata``
fixtures is exactly what plan Phase 0 says to avoid until that is resolved.
Generating equivalent inputs ourselves sidesteps the question entirely.

Golden values produced by running R DENR over these inputs *are* committed,
under ``tests/fixtures/parity/``. Program output is not a derivative work of
the program, and committing it keeps the parity tests hermetic and offline.

Each scenario covers a specific behavior from plan section 13, chosen so a
coordinate or masking error cannot pass unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Transcript:
    """A transcript in 1-based inclusive GTF coordinates, as written to file."""

    transcript_id: str
    gene_id: str
    contig: str
    start: int
    end: int
    strand: str


@dataclass(frozen=True)
class Scenario:
    """One parity case: an annotation, a signal, and design parameters.

    Attributes:
        name: Identifier; also the golden filename stem.
        purpose: What this case is meant to catch, for failure messages.
        transcripts: Annotation records.
        contig_sizes: Contig lengths declared in the generated BigWig headers.
        signal: Per-strand list of ``(contig, start0, end0, value)`` intervals
            in 0-based half-open coordinates.
        bin_size: Design bin width.
        mask_start_bins: ``(inside, outside)`` at the 5' end.
        mask_end_bins: ``(inside, outside)`` at the 3' end.
        distance: Locus-grouping threshold; None means bin_size.
        bin_operation: Rounding of fractional occupancy.
        transform: Legacy objective.
    """

    name: str
    purpose: str
    transcripts: tuple[Transcript, ...]
    contig_sizes: dict[str, int]
    signal: dict[str, tuple[tuple[str, int, int, float], ...]]
    bin_size: int = 250
    mask_start_bins: tuple[int, int] = (0, 0)
    mask_end_bins: tuple[int, int] = (0, 0)
    distance: int | None = None
    bin_operation: str = "round"
    transform: str = "log"
    #: Per-bin summary. "mean" matches the reference, whose add_data hardcodes
    #: it; a scenario using anything else cannot be compared against R.
    summary: str = "mean"
    #: Set when R cannot process the scenario at all, with the reason. Such a
    #: scenario is an intended divergence, not missing coverage.
    r_incompatible: str | None = None
    tolerances: dict[str, float] = field(default_factory=dict)

    def gtf_lines(self) -> list[str]:
        """Render the annotation as GTF text.

        Each transcript gets both a ``transcript`` record and a single
        full-span ``exon``. The exon is required, not decorative:
        ``GenomicFeatures::makeTxDbFromGFF`` builds a TxDb from subfeatures and
        yields an empty database from ``transcript`` lines alone, so the R side
        would silently have nothing to quantify.

        A single exon spanning the whole transcript is also the correct pre-RNA
        representation — nascent transcription covers introns — and our own
        parser takes the min start and max end over all records, so both sides
        see the same span.
        """
        lines = [f"##dienr parity scenario: {self.name}"]
        for tx in self.transcripts:
            attributes = f'gene_id "{tx.gene_id}"; transcript_id "{tx.transcript_id}";'
            for feature in ("transcript", "exon"):
                lines.append(
                    "\t".join(
                        [
                            tx.contig,
                            "dienr",
                            feature,
                            str(tx.start),
                            str(tx.end),
                            ".",
                            tx.strand,
                            ".",
                            attributes,
                        ]
                    )
                )
        return lines


def _ramp(contig: str, start: int, end: int, step: int, base: float) -> tuple:
    """A staircase signal, so a coordinate shift changes the binned result.

    Uniform signal would hide off-by-one errors: every bin would agree
    regardless of where the boundaries fell.
    """
    return tuple(
        (contig, position, min(position + step, end), base + (index % 7))
        for index, position in enumerate(range(start, end, step))
    )


_PLUS = "+"
_MINUS = "-"

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="two_distinguishable_isoforms",
        purpose="baseline: two isoforms sharing a TSS with different 3' ends",
        transcripts=(
            Transcript("tx_short", "gene_a", "chr1", 1_001, 3_000, _PLUS),
            Transcript("tx_long", "gene_a", "chr1", 1_001, 6_000, _PLUS),
        ),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 8_000, 50, 1.0), _MINUS: ()},
    ),
    Scenario(
        name="duplicate_models_collapse",
        purpose="identical spans must collapse to one identifiable model",
        transcripts=(
            Transcript("tx_a", "gene_a", "chr1", 1_001, 5_000, _PLUS),
            Transcript("tx_b", "gene_a", "chr1", 1_001, 5_000, _PLUS),
            Transcript("tx_c", "gene_b", "chr1", 1_001, 5_000, _PLUS),
        ),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 7_000, 50, 2.0), _MINUS: ()},
    ),
    Scenario(
        name="rank_deficient_union",
        purpose="third isoform spans the union of the first two: rank 2, 3 models",
        transcripts=(
            Transcript("tx_first", "gene_a", "chr1", 1_001, 3_000, _PLUS),
            Transcript("tx_second", "gene_a", "chr1", 3_001, 5_000, _PLUS),
            Transcript("tx_whole", "gene_a", "chr1", 1_001, 5_000, _PLUS),
        ),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 7_000, 50, 1.5), _MINUS: ()},
    ),
    Scenario(
        name="minus_strand_orientation",
        purpose="minus-strand 5' end is the high coordinate; masks must mirror",
        transcripts=(
            Transcript("tx_m_short", "gene_c", "chr1", 3_001, 6_000, _MINUS),
            Transcript("tx_m_long", "gene_c", "chr1", 1_001, 6_000, _MINUS),
        ),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: (), _MINUS: _ramp("chr1", 500, 8_000, 50, 1.0)},
        mask_start_bins=(2, 1),
        mask_end_bins=(1, 0),
    ),
    Scenario(
        name="masked_ends",
        purpose="TSS/PAS masking on the plus strand, inside and outside",
        transcripts=(
            Transcript("tx_short", "gene_a", "chr1", 1_001, 4_000, _PLUS),
            Transcript("tx_long", "gene_a", "chr1", 2_001, 8_000, _PLUS),
        ),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 10_000, 50, 1.0), _MINUS: ()},
        mask_start_bins=(2, 2),
        mask_end_bins=(2, 1),
    ),
    Scenario(
        name="terminal_partial_bin",
        purpose="locus width not a multiple of bin size: terminal bin overhangs",
        transcripts=(Transcript("tx_odd", "gene_d", "chr1", 1_001, 2_137, _PLUS),),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 4_000, 37, 1.0), _MINUS: ()},
        bin_size=250,
    ),
    Scenario(
        name="both_strands_separate_loci",
        purpose="overlapping opposite-strand transcripts must not share a locus",
        transcripts=(
            Transcript("tx_p", "gene_e", "chr1", 1_001, 5_000, _PLUS),
            Transcript("tx_m", "gene_f", "chr1", 1_001, 5_000, _MINUS),
        ),
        contig_sizes={"chr1": 20_000},
        signal={
            _PLUS: _ramp("chr1", 500, 7_000, 50, 1.0),
            _MINUS: _ramp("chr1", 500, 7_000, 50, 3.0),
        },
    ),
    Scenario(
        name="quiet_locus",
        purpose="a locus with no signal is all_zero, never a failure",
        transcripts=(Transcript("tx_quiet", "gene_g", "chr1", 12_001, 15_000, _PLUS),),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 3_000, 50, 1.0), _MINUS: ()},
    ),
    Scenario(
        name="separate_loci_by_distance",
        purpose="gap exactly at the grouping threshold must not connect loci",
        transcripts=(
            Transcript("tx_left", "gene_h", "chr1", 1_001, 3_000, _PLUS),
            # 0-based gap from 3000 to 3250 is exactly 250 == distance.
            Transcript("tx_right", "gene_i", "chr1", 3_251, 5_000, _PLUS),
        ),
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 7_000, 50, 1.0), _MINUS: ()},
        distance=250,
    ),
    Scenario(
        name="multi_contig",
        purpose="a declared contig with no signal is all_zero, not missing_contig",
        transcripts=(
            Transcript("tx_c1", "gene_j", "chr1", 1_001, 4_000, _PLUS),
            Transcript("tx_c2", "gene_k", "chr2", 1_001, 4_000, _PLUS),
        ),
        contig_sizes={"chr1": 20_000, "chr2": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 6_000, 50, 1.0), _MINUS: ()},
    ),
    Scenario(
        name="absent_contig",
        purpose="a locus on a contig the signal file omits is a failure, not a zero",
        transcripts=(
            Transcript("tx_here", "gene_l", "chr1", 1_001, 4_000, _PLUS),
            Transcript("tx_nowhere", "gene_m", "chrGhost", 1_001, 4_000, _PLUS),
        ),
        # chrGhost is deliberately absent from the BigWig headers.
        contig_sizes={"chr1": 20_000},
        signal={_PLUS: _ramp("chr1", 500, 6_000, 50, 1.0), _MINUS: ()},
        r_incompatible=(
            "R's add_data calls assert_chromosome_exists and aborts the entire "
            "run with 'chromosome chrGhost does not exist in ...'. It cannot "
            "produce a golden here, which is the divergence this scenario "
            "documents: we record missing_contig per locus and keep going."
        ),
    ),
)

SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def write_inputs(scenario: Scenario, directory: Path) -> dict[str, Path]:
    """Write a scenario's GTF and stranded BigWigs into `directory`.

    Returns:
        Mapping with keys ``gtf``, ``plus`` and ``minus``.

    Raises:
        ImportError: If `pybigtools` is unavailable. It is the only library in
            the stack that can *write* BigWig; `polars-bio` has no
            ``sink_bigwig``.
    """
    import pybigtools

    directory.mkdir(parents=True, exist_ok=True)
    gtf = directory / f"{scenario.name}.gtf"
    gtf.write_text("\n".join(scenario.gtf_lines()) + "\n")

    paths = {"gtf": gtf}
    for strand, key in ((_PLUS, "plus"), (_MINUS, "minus")):
        path = directory / f"{scenario.name}_{key}.bw"
        writer = pybigtools.open(str(path), "w")
        intervals = [(c, s, e, float(v)) for c, s, e, v in scenario.signal.get(strand, ())]
        covered = {interval[0] for interval in intervals}

        # Two writer behaviors have to be worked around here, both verified:
        #
        # 1. A BigWig written with zero intervals is not a readable file.
        #    pybigtools reports "Input bedGraph is empty" on write and then
        #    fails to open it with BBIReadError.
        # 2. Contigs passed in the size map but carrying no interval are
        #    *dropped from the header*. Asking for {chr1, chr2} and writing only
        #    to chr1 yields a file declaring chr1 alone.
        #
        # Since the header is what distinguishes MISSING_CONTIG from ALL_ZERO,
        # a fixture cannot express "declared but quiet" without an explicit
        # zero-valued placeholder base on every declared contig.
        placeholders = [
            (contig, 0, 1, 0.0) for contig in scenario.contig_sizes if contig not in covered
        ]
        writer.write(dict(scenario.contig_sizes), iter(sorted(placeholders + intervals)))
        writer.close()
        paths[key] = path
    return paths
