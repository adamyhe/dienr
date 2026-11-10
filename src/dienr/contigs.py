"""Explicit contig renaming (plan section 20, item 2).

Ensembl GTFs name contigs ``1``, ``X``, ``MT``; UCSC-style BigWigs name them
``chr1``, ``chrX``, ``chrM``. Out of the box nothing matches and every locus
gets ``missing_contig``. The plan's policy is that renaming is explicit and
recorded, never guessed: a `ContigMap` comes from a file the user wrote, or
from a named preset the user chose, and the map itself travels with the
`Design` into every results manifest.

Renaming applies to the annotation, before the design is built, so a renamed
design has a new annotation digest and so a new design digest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl

from .annotation import Annotation

_ENSEMBL_PRIMARY = [str(i) for i in range(1, 23)] + ["X", "Y"]

#: Named maps for the one mismatch nearly everyone hits: primary human
#: chromosomes between Ensembl/NCBI and UCSC names. Scaffolds and patches are
#: deliberately absent -- their names differ in ways a prefix rule gets wrong
#: (Ensembl ``GL000192.1`` is UCSC ``chr1_gl000192_random``), so they are left
#: unmapped and reported rather than guessed.
PRESETS: dict[str, dict[str, str]] = {
    "ensembl-to-ucsc": {**{c: f"chr{c}" for c in _ENSEMBL_PRIMARY}, "MT": "chrM"},
    "ucsc-to-ensembl": {**{f"chr{c}": c for c in _ENSEMBL_PRIMARY}, "chrM": "MT"},
}


@dataclass(frozen=True)
class ContigMap:
    """A one-to-one renaming of annotation contigs.

    Attributes:
        mapping: ``old name -> new name``. Contigs not listed keep their name.
        source: Where the map came from: a preset name or a file path.
    """

    mapping: dict[str, str]
    source: str

    def __post_init__(self) -> None:
        if not self.mapping:
            raise ValueError(f"contig map {self.source!r} is empty")
        for old, new in self.mapping.items():
            if not old or not new:
                raise ValueError(f"contig map {self.source!r} has an empty name")
        seen: set[str] = set()
        for dup in self.mapping.values():
            if dup not in seen:
                seen.add(dup)
                continue
            raise ValueError(
                f"contig map {self.source!r} sends two contigs to {dup!r}; merging "
                "contigs would silently join their loci"
            )

    @classmethod
    def load(cls, spec: str | Path) -> ContigMap:
        """A preset name (see `PRESETS`) or a path to a two-column file.

        The file has one ``old<TAB>new`` (or comma-separated) pair per line;
        blank lines and lines starting with ``#`` are ignored. An empty
        ``new`` means "no equivalent": the contig is left unmapped. That is
        how the ChromosomeMappings files (github.com/dpryan79/ChromosomeMappings)
        mark patches and scaffolds, so those files load as they are.
        """
        if isinstance(spec, str) and spec in PRESETS:
            return cls(mapping=dict(PRESETS[spec]), source=spec)
        path = Path(spec)
        if not path.exists():
            raise ValueError(
                f"contig map {str(spec)!r} is neither a file nor a preset "
                f"({', '.join(sorted(PRESETS))})"
            )
        mapping: dict[str, str] = {}
        for number, raw in enumerate(path.read_text().splitlines(), start=1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            fields = [f.strip() for f in (raw.split("\t") if "\t" in raw else raw.split(","))]
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f"{path}:{number}: expected 'old<TAB>new', got {raw!r}")
            old, new = fields
            if not new:
                continue
            if old in mapping and mapping[old] != new:
                raise ValueError(f"{path}:{number}: {old!r} is mapped twice")
            mapping[old] = new
        return cls(mapping=mapping, source=str(path))

    @property
    def digest(self) -> str:
        hasher = hashlib.sha256(b"dienr.contigmap.v1\x00")
        for old, new in sorted(self.mapping.items()):
            hasher.update(f"{old}\x00{new}\x01".encode())
        return hasher.hexdigest()[:16]

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.mapping.items()))


@dataclass(frozen=True)
class RenameReport:
    """What `rename_contigs` did, for printing and provenance."""

    renamed: dict[str, str]
    unmapped: dict[str, int]
    transcripts_renamed: int

    def message(self) -> str:
        text = (
            f"renamed {len(self.renamed)} contig(s) covering "
            f"{self.transcripts_renamed} transcript(s)"
        )
        if self.unmapped:
            n = sum(self.unmapped.values())
            shown = ", ".join(sorted(self.unmapped)[:5])
            more = "" if len(self.unmapped) <= 5 else f", ... ({len(self.unmapped)} total)"
            text += f"; {n} transcript(s) on unmapped contigs kept their names ({shown}{more})"
        return text


def rename_contigs(
    annotation: Annotation, contig_map: ContigMap
) -> tuple[Annotation, RenameReport]:
    """Return `annotation` with contigs renamed, and a report.

    Contigs absent from the map keep their names, and are reported. Fails if
    no contig of the annotation is in the map (the map is for a different
    naming scheme) or if renaming would collide with a contig already present.
    """
    table = annotation.transcripts
    counts = dict(table.group_by("contig").len().iter_rows())
    present = set(counts)
    renamed = {old: new for old, new in contig_map.mapping.items() if old in present}
    if not renamed:
        example = ", ".join(sorted(present)[:3])
        raise ValueError(
            f"contig map {contig_map.source!r} matches none of the annotation's contigs "
            f"(e.g. {example}); is it the right direction?"
        )
    collisions = sorted(set(renamed.values()) & (present - set(renamed)))
    if collisions:
        raise ValueError(
            f"renaming would merge into contig(s) already in the annotation: {collisions[:5]}"
        )
    unmapped = {c: n for c, n in counts.items() if c not in renamed}
    out = table.with_columns(pl.col("contig").replace(renamed))
    source = f"{annotation.source} [contigs renamed by {contig_map.source}, {contig_map.digest}]"
    renamed_annotation = replace(
        Annotation.from_frame(out, source=source), contig_map=contig_map.pairs()
    )
    report = RenameReport(
        renamed=renamed,
        unmapped=unmapped,
        transcripts_renamed=sum(counts[c] for c in renamed),
    )
    return renamed_annotation, report


@dataclass(frozen=True)
class ContigCheck:
    """How many of a design's loci sit on contigs the BigWigs lack."""

    missing: dict[str, int]
    n_loci: int
    hint: str | None

    @property
    def missing_loci(self) -> int:
        return sum(self.missing.values())

    @property
    def none_match(self) -> bool:
        return self.n_loci > 0 and self.missing_loci == self.n_loci

    def message(self) -> str:
        shown = ", ".join(sorted(self.missing)[:5])
        text = (
            f"{self.missing_loci} of {self.n_loci} loci are on contigs absent from the "
            f"BigWigs ({shown}{', ...' if len(self.missing) > 5 else ''})"
        )
        return text + (f"; {self.hint}" if self.hint else "")


def check_contigs(design_contigs: list[str], bigwig_contigs: set[str]) -> ContigCheck:
    """Compare a design's per-locus contigs with the BigWigs' contigs.

    The hint names a preset only when applying it would match contigs that
    currently don't -- a suggestion for the user to act on, never applied.
    """
    missing: dict[str, int] = {}
    for contig in design_contigs:
        if contig not in bigwig_contigs:
            missing[contig] = missing.get(contig, 0) + 1
    hint = None
    if missing:
        for name, mapping in PRESETS.items():
            if any(mapping.get(c) in bigwig_contigs for c in missing):
                hint = f"rebuild the design with --contig-map {name} (or your own map)"
                break
    return ContigCheck(missing=missing, n_loci=len(design_contigs), hint=hint)
