"""Locus grouping: connected components per contig and strand."""

from __future__ import annotations

import polars as pl
import pytest

from dienr.loci import assign_loci, locus_table


def frame(rows: list[tuple[str, str, int, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["transcript_id", "contig", "start", "end", "strand"], orient="row"
    )


def test_overlapping_transcripts_share_a_locus():
    tx = frame([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 500, 1500, "+")])
    assert assign_loci(tx, distance=0)["locus_id"].n_unique() == 1


def test_opposite_strands_never_share_a_locus():
    """Same coordinates, different strands: grouping is strand-specific."""
    tx = frame([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 0, 1000, "-")])
    assert assign_loci(tx, distance=0)["locus_id"].n_unique() == 2


def test_different_contigs_never_share_a_locus():
    tx = frame([("a", "chr1", 0, 1000, "+"), ("b", "chr2", 0, 1000, "+")])
    assert assign_loci(tx, distance=0)["locus_id"].n_unique() == 2


def test_abutting_transcripts_are_separate_at_distance_zero():
    """Touching but not overlapping stays separate, as in the reference."""
    tx = frame([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 1000, 2000, "+")])
    assert assign_loci(tx, distance=0)["locus_id"].n_unique() == 2


@pytest.mark.parametrize(
    ("gap", "distance", "expected"),
    [
        (250, 250, 2),  # gap == distance: not connected
        (249, 250, 1),  # gap < distance: connected
        (0, 1, 1),  # abutting, distance 1: connected
        (1000, 250, 2),
    ],
)
def test_distance_threshold_is_gap_strictly_less_than_distance(gap, distance, expected):
    tx = frame([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 1000 + gap, 2000 + gap, "+")])
    assert assign_loci(tx, distance=distance)["locus_id"].n_unique() == expected


def test_transitive_chaining_forms_one_locus():
    """A and C do not overlap, but both overlap B, so all three join."""
    tx = frame(
        [
            ("a", "chr1", 0, 1000, "+"),
            ("b", "chr1", 900, 2000, "+"),
            ("c", "chr1", 1900, 3000, "+"),
        ]
    )
    assigned = assign_loci(tx, distance=0)
    assert assigned["locus_id"].n_unique() == 1


def test_nested_transcript_does_not_split_locus():
    """A short transcript inside a long one must not end the component."""
    tx = frame(
        [
            ("long", "chr1", 0, 10_000, "+"),
            ("short", "chr1", 100, 200, "+"),
            ("tail", "chr1", 9_000, 11_000, "+"),
        ]
    )
    assert assign_loci(tx, distance=0)["locus_id"].n_unique() == 1


def test_locus_ids_are_contiguous_and_ordered():
    tx = frame(
        [
            ("c", "chr2", 0, 100, "+"),
            ("a", "chr1", 0, 100, "+"),
            ("b", "chr1", 5000, 5100, "+"),
        ]
    )
    assigned = assign_loci(tx, distance=0)
    ids = sorted(assigned["locus_id"].unique().to_list())
    assert ids == list(range(len(ids)))
    table = locus_table(assigned)
    # Ordering follows (contig, strand, start).
    assert table["contig"].to_list() == ["chr1", "chr1", "chr2"]


def test_locus_name_reports_the_span():
    tx = frame([("a", "chr1", 100, 1000, "+"), ("b", "chr1", 500, 2500, "+")])
    table = locus_table(assign_loci(tx, distance=0))
    assert table["locus_name"][0] == "chr1:+:100-2500"


def test_locus_table_counts_members():
    tx = frame(
        [
            ("a", "chr1", 0, 1000, "+"),
            ("b", "chr1", 100, 1100, "+"),
            ("c", "chr1", 90_000, 91_000, "+"),
        ]
    )
    table = locus_table(assign_loci(tx, distance=250))
    assert table["n_transcripts"].to_list() == [2, 1]


def test_negative_distance_rejected():
    tx = frame([("a", "chr1", 0, 100, "+")])
    with pytest.raises(ValueError, match="non-negative"):
        assign_loci(tx, distance=-1)
