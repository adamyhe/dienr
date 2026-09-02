"""GTF/GFF parsing and pre-RNA span construction.

Coordinate policy (plan section 6): coordinates are 0-based half-open
everywhere inside this package. GTF/GFF is 1-based inclusive, and the single
``start - 1`` conversion happens here, at the parser boundary, and nowhere else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl

#: The nine fixed GTF/GFF columns, in file order.
_GXF_COLUMNS = [
    "contig",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attributes",
]

#: Schema of a normalized transcript table. Order is part of the contract.
TRANSCRIPT_SCHEMA = {
    "transcript_id": pl.String,
    "gene_id": pl.String,
    "contig": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "strand": pl.String,
    "coord_source": pl.String,
}

_VALID_STRANDS = ("+", "-")


def _attribute(column: str, key: str) -> pl.Expr:
    """Extract `key` from a GTF or GFF3 attribute string.

    GTF writes ``key "value";`` and GFF3 writes ``key=value;``. Both forms are
    tried so a single parser handles either dialect.
    """
    gtf = pl.col(column).str.extract(rf'{key}\s+"([^"]*)"', 1)
    gff3 = pl.col(column).str.extract(rf"{key}=([^;]*)", 1)
    return pl.coalesce(gtf, gff3)


@dataclass(frozen=True)
class Annotation:
    """Validated transcript records defining continuous pre-RNA spans.

    Attributes:
        transcripts: One row per transcript, conforming to `TRANSCRIPT_SCHEMA`.
            ``start``/``end`` are 0-based half-open.
        digest: Content-derived identifier. Any change to the transcript table
            produces a new digest, which in turn invalidates a `Design`.
        source: Provenance string, normally the input path.
    """

    transcripts: pl.DataFrame
    digest: str
    source: str

    def __post_init__(self) -> None:
        got = self.transcripts.schema
        for name, dtype in TRANSCRIPT_SCHEMA.items():
            if name not in got:
                raise ValueError(f"transcript table is missing required column {name!r}")
            if got[name] != dtype:
                raise ValueError(f"column {name!r} has dtype {got[name]}, expected {dtype}")

    def __len__(self) -> int:
        return self.transcripts.height

    @property
    def contigs(self) -> list[str]:
        """Sorted contig names referenced by the annotation."""
        return sorted(self.transcripts["contig"].unique().to_list())

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, *, source: str = "in-memory") -> Annotation:
        """Build an `Annotation` from an already 0-based half-open table.

        Intended for tests and programmatic construction. Input must carry
        ``transcript_id``, ``contig``, ``start``, ``end`` and ``strand``;
        ``gene_id`` and ``coord_source`` are filled with nulls and
        ``"provided"`` respectively when absent.
        """
        frame = frame.with_columns(
            gene_id=pl.col("gene_id") if "gene_id" in frame.columns else pl.lit(None, pl.String),
            coord_source=(
                pl.col("coord_source")
                if "coord_source" in frame.columns
                else pl.lit("provided", pl.String)
            ),
        )
        table = _finalize(frame.lazy())
        return cls(transcripts=table, digest=_digest(table, source), source=source)


def read_annotation(
    path: str | Path,
    *,
    transcript_key: str = "transcript_id",
    gene_key: str = "gene_id",
    span_features: tuple[str, ...] = ("transcript", "exon"),
) -> Annotation:
    """Read a GTF/GFF file into pre-RNA transcript spans.

    A pre-RNA model is the continuous span from a transcript's annotated TSS to
    its PAS, so each transcript's span is the minimum start and maximum end over
    all of its records. Exonic structure is deliberately discarded: nascent
    transcription covers introns.

    Args:
        path: Path to a GTF or GFF file. ``.gz`` is handled by Polars.
        transcript_key: Attribute key holding the transcript identifier.
        gene_key: Attribute key holding the gene identifier.
        span_features: Feature types contributing to a span. ``exon`` is
            included so annotations lacking explicit ``transcript`` rows still
            yield correct spans.

    Returns:
        An `Annotation` with 0-based half-open coordinates.

    Raises:
        ValueError: If no usable records are found, or if any transcript has an
            unusable strand or spans more than one contig or strand.
    """
    path = Path(path)
    lazy = (
        pl.scan_csv(
            path,
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=_GXF_COLUMNS,
            schema_overrides={"start": pl.Int64, "end": pl.Int64},
            truncate_ragged_lines=False,
        )
        .filter(pl.col("feature").is_in(span_features))
        .select(
            pl.col("contig").cast(pl.String),
            # The one and only 1-based inclusive -> 0-based half-open conversion.
            (pl.col("start") - 1).alias("start"),
            pl.col("end"),
            pl.col("strand").cast(pl.String),
            pl.col("feature"),
            _attribute("attributes", transcript_key).alias("transcript_id"),
            _attribute("attributes", gene_key).alias("gene_id"),
        )
        .filter(pl.col("transcript_id").is_not_null())
    )

    spans = lazy.group_by("transcript_id").agg(
        contig=pl.col("contig").unique().sort(),
        strand=pl.col("strand").unique().sort(),
        start=pl.col("start").min(),
        end=pl.col("end").max(),
        gene_id=pl.col("gene_id").drop_nulls().first(),
        # Record whether the span came from a real transcript feature or was
        # reconstructed from subfeatures (plan section 8).
        has_transcript_feature=(pl.col("feature") == "transcript").any(),
    )

    frame = spans.collect()
    if frame.height == 0:
        raise ValueError(
            f"{path}: no records with attribute {transcript_key!r} among features {span_features}"
        )

    _reject_split_transcripts(frame)

    frame = frame.with_columns(
        contig=pl.col("contig").list.first(),
        strand=pl.col("strand").list.first(),
        coord_source=pl.when(pl.col("has_transcript_feature"))
        .then(pl.lit("transcript_feature"))
        .otherwise(pl.lit("exon_reconstructed")),
    )

    table = _finalize(frame.lazy())
    return Annotation(transcripts=table, digest=_digest(table, str(path)), source=str(path))


def _reject_split_transcripts(frame: pl.DataFrame) -> None:
    """Fail loudly if a transcript_id appears on multiple contigs or strands.

    Silently taking the first would fabricate a span across unrelated loci.
    """
    for column in ("contig", "strand"):
        bad = frame.filter(pl.col(column).list.len() > 1)
        if bad.height:
            offenders = bad["transcript_id"].head(5).to_list()
            raise ValueError(
                f"{bad.height} transcript(s) span more than one {column}, "
                f"e.g. {offenders}. Transcript identifiers must be unique per "
                f"{column}."
            )


def _finalize(lazy: pl.LazyFrame) -> pl.DataFrame:
    """Validate, order and sort a transcript table into `TRANSCRIPT_SCHEMA`."""
    table = (
        lazy.select(
            pl.col("transcript_id").cast(pl.String),
            pl.col("gene_id").cast(pl.String),
            pl.col("contig").cast(pl.String),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            pl.col("strand").cast(pl.String),
            pl.col("coord_source").cast(pl.String),
        )
        .sort("contig", "strand", "start", "end", "transcript_id")
        .collect()
    )

    bad_strand = table.filter(~pl.col("strand").is_in(_VALID_STRANDS))
    if bad_strand.height:
        seen = sorted(set(bad_strand["strand"].to_list()))
        raise ValueError(
            f"{bad_strand.height} transcript(s) have unusable strand {seen}. "
            "Quantification requires '+' or '-'."
        )

    empty = table.filter(pl.col("end") <= pl.col("start"))
    if empty.height:
        offenders = empty["transcript_id"].head(5).to_list()
        raise ValueError(f"{empty.height} transcript(s) have empty spans, e.g. {offenders}")

    duplicated = table["transcript_id"].is_duplicated()
    if duplicated.any():
        offenders = table.filter(duplicated)["transcript_id"].head(5).to_list()
        raise ValueError(f"transcript identifiers must be unique, e.g. {offenders}")

    return table


def _digest(table: pl.DataFrame, source: str) -> str:
    """Content-derived digest of a transcript table.

    Depends only on the normalized coordinates and identifiers, so two runs over
    the same annotation agree regardless of input path or row order.
    """
    hasher = hashlib.sha256()
    hasher.update(b"dienr.annotation.v1\x00")
    for column in ("transcript_id", "gene_id", "contig", "start", "end", "strand"):
        hasher.update(column.encode())
        hasher.update(b"\x00")
        series = table[column]
        if series.dtype == pl.String:
            hasher.update("\x00".join("" if v is None else v for v in series).encode())
        else:
            hasher.update(series.to_numpy().tobytes())
        hasher.update(b"\x01")
    return hasher.hexdigest()[:16]
