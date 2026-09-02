"""Shared fixtures.

The R reference package's synthetic data lives in an untracked ``_reference/``
clone, so tests that need it skip cleanly when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

REFERENCE_EXTDATA = Path(__file__).resolve().parents[1] / "_reference" / "DENR" / "inst" / "extdata"


@pytest.fixture(scope="session")
def extdata() -> Path:
    """Path to the R reference package's fixture directory."""
    if not REFERENCE_EXTDATA.is_dir():
        pytest.skip(f"reference fixtures not present at {REFERENCE_EXTDATA}")
    return REFERENCE_EXTDATA


@pytest.fixture
def simple_transcripts() -> pl.DataFrame:
    """Three transcripts: two overlapping on '+', one separate on '-'."""
    return pl.DataFrame(
        {
            "transcript_id": ["tx_a", "tx_b", "tx_c"],
            "gene_id": ["gene_1", "gene_1", "gene_2"],
            "contig": ["chr1", "chr1", "chr1"],
            "start": [0, 500, 100_000],
            "end": [2000, 2500, 101_000],
            "strand": ["+", "+", "-"],
        }
    )


def write_gtf(path: Path, rows: list[dict]) -> Path:
    """Write a minimal GTF. ``start``/``end`` are 1-based inclusive, as in GTF."""
    lines = ["##description: dienr test fixture"]
    for row in rows:
        attributes = f'gene_id "{row["gene_id"]}"; transcript_id "{row["transcript_id"]}";'
        lines.append(
            "\t".join(
                [
                    row["contig"],
                    "test",
                    row.get("feature", "transcript"),
                    str(row["start"]),
                    str(row["end"]),
                    ".",
                    row["strand"],
                    ".",
                    attributes,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def built(tmp_path):
    """Build a design and counts for a parity scenario.

    Lives here so both test_parity and test_parity_harness can use it;
    the harness calls the parity test functions directly.
    """
    from dienr.annotation import read_annotation
    from dienr.counts import SignalFile, count_sample
    from dienr.design import build_design

    from .fixtures.scenarios import write_inputs

    def _build(scenario):
        paths = write_inputs(scenario, tmp_path / scenario.name)
        annotation = read_annotation(paths["gtf"])
        design = build_design(
            annotation,
            bin_size=scenario.bin_size,
            distance=scenario.distance,
            mask_start_bins=scenario.mask_start_bins,
            mask_end_bins=scenario.mask_end_bins,
            bin_operation=scenario.bin_operation,
        )
        counts = count_sample(
            design,
            "parity",
            (SignalFile(paths["plus"], "+"), SignalFile(paths["minus"], "-")),
            summary=scenario.summary,
        )
        return design, counts

    return _build
