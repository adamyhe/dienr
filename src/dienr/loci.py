"""Strand-specific locus grouping.

A locus is a connected component of transcripts on one contig and one strand,
where two transcripts are connected when the gap between them is smaller than
``distance``. Transcripts that merely touch are not connected.

This needs no interval tree: sorting by start and sweeping a running maximum of
``end`` yields connected components directly (plan section 7).
"""

from __future__ import annotations

import polars as pl

#: Columns added by `assign_loci`.
LOCUS_COLUMNS = ("locus_id", "locus_name")


def assign_loci(transcripts: pl.DataFrame, *, distance: int) -> pl.DataFrame:
    """Assign each transcript to a strand-specific locus.

    Args:
        transcripts: Transcript table with ``contig``, ``strand``, ``start`` and
            ``end`` columns, 0-based half-open.
        distance: Two transcripts join the same locus when the gap between them
            is strictly less than this many bases. ``0`` therefore separates
            abutting transcripts, matching the reference implementation's
            treatment of perfectly adjoining ranges.

    Returns:
        The input table with `LOCUS_COLUMNS` appended, sorted by
        ``(contig, strand, start, end, transcript_id)``. ``locus_id`` is a
        contiguous 0-based index in that order; ``locus_name`` is the readable
        ``contig:strand:start-end`` span of the locus.

    Raises:
        ValueError: If `distance` is negative.
    """
    if distance < 0:
        raise ValueError(f"distance must be non-negative, got {distance}")

    ordered = transcripts.sort("contig", "strand", "start", "end", "transcript_id")

    # Running maximum of end over preceding transcripts in the same
    # contig/strand. A new component starts when the gap to everything seen so
    # far reaches `distance`.
    grouped = ordered.with_columns(
        _reach=pl.col("end").cum_max().shift(1).over("contig", "strand")
    ).with_columns(
        _starts_locus=pl.col("_reach").is_null()
        | ((pl.col("start") - pl.col("_reach")) >= distance)
    )

    # cum_sum of the boundary flag numbers components within contig/strand;
    # combining with the group keys and re-ranking gives a global id that
    # respects the (contig, strand, start) ordering.
    grouped = grouped.with_columns(
        _local=pl.col("_starts_locus").cum_sum().over("contig", "strand")
    )

    spans = grouped.group_by("contig", "strand", "_local").agg(
        _lo=pl.col("start").min(), _hi=pl.col("end").max()
    )
    spans = spans.sort("contig", "strand", "_lo", "_hi").with_row_index("locus_id")
    spans = spans.with_columns(
        locus_name=pl.concat_str(
            [
                pl.col("contig"),
                pl.lit(":"),
                pl.col("strand"),
                pl.lit(":"),
                pl.col("_lo").cast(pl.String),
                pl.lit("-"),
                pl.col("_hi").cast(pl.String),
            ]
        )
    )

    return (
        grouped.join(spans, on=["contig", "strand", "_local"], how="left")
        .drop("_reach", "_starts_locus", "_local", "_lo", "_hi")
        .with_columns(pl.col("locus_id").cast(pl.UInt32))
        .sort("locus_id", "start", "end", "transcript_id")
    )


def locus_table(transcripts_with_loci: pl.DataFrame) -> pl.DataFrame:
    """Summarize loci from a table already carrying `LOCUS_COLUMNS`.

    Returns one row per locus with its contig, strand, span and transcript
    count, ordered by ``locus_id``.
    """
    return (
        transcripts_with_loci.group_by("locus_id")
        .agg(
            locus_name=pl.col("locus_name").first(),
            contig=pl.col("contig").first(),
            strand=pl.col("strand").first(),
            start=pl.col("start").min(),
            end=pl.col("end").max(),
            n_transcripts=pl.len().cast(pl.UInt32),
        )
        .sort("locus_id")
    )
