"""Annotation parsing and the coordinate boundary."""

from __future__ import annotations

import polars as pl
import pytest

from dienr.annotation import Annotation, read_annotation

from .conftest import write_gtf


def test_gtf_start_converts_once_to_zero_based(tmp_path):
    """GTF is 1-based inclusive; internal representation is 0-based half-open."""
    gtf = write_gtf(
        tmp_path / "a.gtf",
        [
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 1001,
                "end": 2000,
                "strand": "+",
            }
        ],
    )
    annotation = read_annotation(gtf)
    row = annotation.transcripts.row(0, named=True)
    assert row["start"] == 1000
    assert row["end"] == 2000
    # Width is preserved by the conversion: 2000 - 1001 + 1 == 2000 - 1000.
    assert row["end"] - row["start"] == 1000


def test_single_base_transcript(tmp_path):
    """A one-base GTF feature becomes a one-base half-open span."""
    gtf = write_gtf(
        tmp_path / "b.gtf",
        [
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 500,
                "end": 500,
                "strand": "+",
            }
        ],
    )
    row = read_annotation(gtf).transcripts.row(0, named=True)
    assert (row["start"], row["end"]) == (499, 500)


def test_span_is_min_start_to_max_end_over_records(tmp_path):
    """Pre-RNA spans cover introns, so exons collapse to their outer bounds."""
    gtf = write_gtf(
        tmp_path / "c.gtf",
        [
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 101,
                "end": 200,
                "strand": "+",
                "feature": "exon",
            },
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 901,
                "end": 1000,
                "strand": "+",
                "feature": "exon",
            },
        ],
    )
    annotation = read_annotation(gtf)
    row = annotation.transcripts.row(0, named=True)
    assert (row["start"], row["end"]) == (100, 1000)
    assert row["coord_source"] == "exon_reconstructed"


def test_coord_source_records_transcript_feature(tmp_path):
    gtf = write_gtf(
        tmp_path / "d.gtf",
        [
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 101,
                "end": 1000,
                "strand": "+",
                "feature": "transcript",
            },
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 101,
                "end": 200,
                "strand": "+",
                "feature": "exon",
            },
        ],
    )
    assert read_annotation(gtf).transcripts.row(0, named=True)["coord_source"] == (
        "transcript_feature"
    )


def test_gff3_attribute_syntax(tmp_path):
    """The same parser handles ``key=value`` attributes."""
    path = tmp_path / "e.gff3"
    path.write_text(
        "##gff-version 3\nchr1\ttest\ttranscript\t101\t500\t.\t-\t.\tgene_id=g1;transcript_id=t1\n"
    )
    row = read_annotation(path).transcripts.row(0, named=True)
    assert row["transcript_id"] == "t1"
    assert row["gene_id"] == "g1"
    assert (row["start"], row["end"], row["strand"]) == (100, 500, "-")


def test_comment_lines_are_skipped(tmp_path):
    path = tmp_path / "f.gtf"
    path.write_text(
        "#!genome-build test\n"
        "##provider: test\n"
        'chr1\tt\ttranscript\t1\t100\t.\t+\t.\tgene_id "g"; transcript_id "t";\n'
    )
    assert len(read_annotation(path)) == 1


def test_unstranded_transcript_is_rejected(tmp_path):
    """Quantification is strand-specific; '.' cannot be silently accepted."""
    gtf = write_gtf(
        tmp_path / "g.gtf",
        [
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 1,
                "end": 100,
                "strand": ".",
            }
        ],
    )
    with pytest.raises(ValueError, match="unusable strand"):
        read_annotation(gtf)


def test_transcript_split_across_contigs_is_rejected(tmp_path):
    """Taking the first contig would fabricate a span across unrelated loci."""
    gtf = write_gtf(
        tmp_path / "h.gtf",
        [
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr1",
                "start": 1,
                "end": 100,
                "strand": "+",
                "feature": "exon",
            },
            {
                "transcript_id": "t1",
                "gene_id": "g1",
                "contig": "chr2",
                "start": 1,
                "end": 100,
                "strand": "+",
                "feature": "exon",
            },
        ],
    )
    with pytest.raises(ValueError, match="more than one contig"):
        read_annotation(gtf)


def test_missing_transcript_key_raises(tmp_path):
    path = tmp_path / "i.gtf"
    path.write_text('chr1\tt\ttranscript\t1\t100\t.\t+\t.\tgene_id "g";\n')
    with pytest.raises(ValueError, match="no records with attribute"):
        read_annotation(path)


def test_digest_depends_on_coordinates(simple_transcripts):
    a = Annotation.from_frame(simple_transcripts)
    shifted = simple_transcripts.with_columns(start=pl.col("start") + 1)
    b = Annotation.from_frame(shifted)
    assert a.digest != b.digest


def test_digest_is_row_order_invariant(simple_transcripts):
    a = Annotation.from_frame(simple_transcripts)
    b = Annotation.from_frame(simple_transcripts.reverse())
    assert a.digest == b.digest


def test_from_frame_rejects_duplicate_transcript_ids(simple_transcripts):
    doubled = pl.concat([simple_transcripts, simple_transcripts.head(1)])
    with pytest.raises(ValueError, match="unique"):
        Annotation.from_frame(doubled)


def test_from_frame_rejects_empty_span():
    frame = pl.DataFrame(
        {
            "transcript_id": ["t1"],
            "contig": ["chr1"],
            "start": [100],
            "end": [100],
            "strand": ["+"],
        }
    )
    with pytest.raises(ValueError, match="empty spans"):
        Annotation.from_frame(frame)


def test_ensembl_numeric_contig_names_stay_strings(tmp_path):
    """Regression: Ensembl GTFs name contigs "1", "2", ..., "X".

    Polars inferred the contig column as an integer from the leading numeric
    names and then failed on "X", so a stock Ensembl GTF could not be read.
    (Found running `dienr runon-tail` on GRCh37.87; plan 18.15.)
    """
    rows = [
        {
            "transcript_id": f"t{i}",
            "gene_id": f"g{i}",
            "contig": str(1 + i % 22),
            "start": 1001 + 10 * i,
            "end": 5000 + 10 * i,
            "strand": "+",
        }
        for i in range(2_000)
    ] + [
        {
            "transcript_id": "tx",
            "gene_id": "gx",
            "contig": "X",
            "start": 1001,
            "end": 5000,
            "strand": "-",
        }
    ]
    annotation = read_annotation(write_gtf(tmp_path / "ens.gtf", rows))
    assert annotation.transcripts["contig"].dtype == pl.String
    assert set(annotation.contigs) >= {"1", "22", "X"}
