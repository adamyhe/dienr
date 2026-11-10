"""Strand-orientation check: a swapped plus/minus pair must be caught.

`dienr` never infers strand from sign, so without this check a swapped pair
fits without complaint -- and every fixture that writes the same file for both
strands, as the R reference's shape-profile tests do, would hide it.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import pytest

import dienr
from dienr.strand import MIN_LOCI, StrandCheck, _single_strand_loci, check_strand_orientation

from .conftest import write_gtf
from .fixtures.bigwig import write_intervals

CONTIG = "chr1"
SIZE = 5_000_000
BIN = 250


def _genes(n: int = 40) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "transcript_id": f"t{i}",
                "contig": CONTIG,
                "start": 10_000 + 40_000 * i,
                "end": 30_000 + 40_000 * i,
                "strand": "+-"[i % 2],
            }
            for i in range(n)
        ]
    )


def _write(path, genes: pl.DataFrame, strand: str, *, sense: float, antisense: float):
    """Per-base value `sense` over this strand's genes, `antisense` over the others'."""
    rows = []
    for g in genes.iter_rows(named=True):
        value = sense if g["strand"] == strand else antisense
        if value:
            rows.append((CONTIG, g["start"], g["end"], value))
    return write_intervals(path, {CONTIG: SIZE}, rows)


@pytest.fixture
def stranded(tmp_path):
    genes = _genes()
    design = dienr.build_design(dienr.Annotation.from_frame(genes, source="s"), bin_size=BIN)
    plus = _write(tmp_path / "p.bw", genes, "+", sense=1.0, antisense=0.1)
    minus = _write(tmp_path / "m.bw", genes, "-", sense=1.0, antisense=0.1)
    return design, plus, minus


def _signal(plus, minus):
    return (dienr.SignalFile(path=plus, strand="+"), dienr.SignalFile(path=minus, strand="-"))


def test_correct_files_pass(stranded):
    design, plus, minus = stranded
    check = check_strand_orientation(design, _signal(plus, minus))
    assert check.verdict == "ok"
    assert check.sense_fraction == pytest.approx(1 / 1.1)
    assert check.locus_agreement == 1.0
    assert check.n_loci == 40


def test_swapped_files_are_detected(stranded):
    design, plus, minus = stranded
    check = check_strand_orientation(design, _signal(minus, plus))
    assert check.verdict == "swapped"
    assert check.sense_fraction == pytest.approx(0.1 / 1.1)
    assert check.locus_agreement == 0.0


def test_negative_minus_files_are_read_by_magnitude(tmp_path):
    genes = _genes()
    design = dienr.build_design(dienr.Annotation.from_frame(genes, source="s"), bin_size=BIN)
    plus = _write(tmp_path / "p.bw", genes, "+", sense=1.0, antisense=0.1)
    minus = _write(tmp_path / "m.bw", genes, "-", sense=-1.0, antisense=-0.1)
    assert check_strand_orientation(design, _signal(plus, minus)).verdict == "ok"


def test_weakly_stranded_data_is_low(tmp_path):
    genes = _genes()
    design = dienr.build_design(dienr.Annotation.from_frame(genes, source="s"), bin_size=BIN)
    plus = _write(tmp_path / "p.bw", genes, "+", sense=1.0, antisense=0.7)
    minus = _write(tmp_path / "m.bw", genes, "-", sense=1.0, antisense=0.7)
    check = check_strand_orientation(design, _signal(plus, minus))
    assert check.verdict == "low"  # 1 / 1.7 = 0.59


def test_too_few_informative_loci_gives_no_verdict(tmp_path):
    genes = _genes(MIN_LOCI - 2)
    design = dienr.build_design(dienr.Annotation.from_frame(genes, source="s"), bin_size=BIN)
    plus = _write(tmp_path / "p.bw", genes, "+", sense=1.0, antisense=0.0)
    minus = _write(tmp_path / "m.bw", genes, "-", sense=1.0, antisense=0.0)
    assert check_strand_orientation(design, _signal(minus, plus)).verdict == "insufficient"


def test_opposite_strand_overlaps_are_excluded():
    """There the 'antisense' file carries a real gene's sense signal."""
    frame = pl.DataFrame(
        [
            {"transcript_id": "a", "contig": CONTIG, "start": 0, "end": 20_000, "strand": "+"},
            {"transcript_id": "b", "contig": CONTIG, "start": 15_000, "end": 40_000, "strand": "-"},
            {
                "transcript_id": "c",
                "contig": CONTIG,
                "start": 100_000,
                "end": 120_000,
                "strand": "+",
            },
        ]
    )
    design = dienr.build_design(dienr.Annotation.from_frame(frame, source="s"), bin_size=BIN)
    kept = _single_strand_loci(design)
    assert [(row[0], row[2]) for row in kept] == [("+", 100_000)]


def test_max_loci_caps_the_reads(stranded):
    design, plus, minus = stranded
    check = check_strand_orientation(design, _signal(plus, minus), max_loci=MIN_LOCI)
    assert check.n_loci == MIN_LOCI
    assert check.verdict == "ok"


def test_arguments_are_validated(stranded):
    design, plus, minus = stranded
    with pytest.raises(ValueError, match="exactly one plus and one minus"):
        check_strand_orientation(design, (dienr.SignalFile(path=plus, strand="+"),))
    with pytest.raises(ValueError, match="refuse_below"):
        check_strand_orientation(design, _signal(plus, minus), warn_below=0.4, refuse_below=0.6)


def test_describe_and_message():
    check = StrandCheck(sense=9.0, antisense=1.0, n_loci=50, locus_agreement=0.8, verdict="ok")
    assert check.sense_fraction == pytest.approx(0.9)
    assert check.describe()["strand_check_verdict"] == "ok"
    assert "0.900" in check.message()
    assert np.isnan(StrandCheck(0.0, 0.0, 0, float("nan"), "insufficient").sense_fraction)


# --------------------------------------------------------------------------
# dienr count
# --------------------------------------------------------------------------


@pytest.fixture
def cli_inputs(tmp_path):
    genes = _genes()
    gtf = write_gtf(
        tmp_path / "g.gtf",
        [
            {
                "transcript_id": r["transcript_id"],
                "gene_id": r["transcript_id"],
                "contig": r["contig"],
                "start": r["start"] + 1,
                "end": r["end"],
                "strand": r["strand"],
            }
            for r in genes.iter_rows(named=True)
        ],
    )
    plus = _write(tmp_path / "p.bw", genes, "+", sense=1.0, antisense=0.1)
    minus = _write(tmp_path / "m.bw", genes, "-", sense=1.0, antisense=0.1)
    from dienr.cli import main

    assert main(["build-design", "--gtf", str(gtf), "--output", str(tmp_path / "d")]) == 0
    return tmp_path, plus, minus


def _sheet(directory, plus, minus):
    path = directory / "samples.csv"
    path.write_text(f"sample_id,bigwig_plus,bigwig_minus\ns,{plus},{minus}\n")
    return path


def test_count_records_a_passing_check(cli_inputs, capsys):
    from dienr.cli import COUNTS_FILENAME, main

    tmp, plus, minus = cli_inputs
    args = [
        "count",
        "--design",
        str(tmp / "d"),
        "--samples",
        str(_sheet(tmp, plus, minus)),
        "--output",
        str(tmp / "c"),
        "--summary",
        "sum",
    ]
    assert main(args) == 0
    assert "-> ok" in capsys.readouterr().out
    (counts,) = pickle.loads((tmp / "c" / COUNTS_FILENAME).read_bytes())
    assert counts.provenance["strand_check_verdict"] == "ok"
    assert counts.provenance["strand_check_sense_fraction"] > 0.9


def test_count_refuses_swapped_files_unless_overridden(cli_inputs, capsys):
    from dienr.cli import main

    tmp, plus, minus = cli_inputs
    args = [
        "count",
        "--design",
        str(tmp / "d"),
        "--samples",
        str(_sheet(tmp, minus, plus)),
        "--output",
        str(tmp / "c"),
        "--summary",
        "sum",
    ]
    with pytest.raises(SystemExit, match="look swapped"):
        main(args)
    assert main([*args, "--allow-strand-mismatch"]) == 0
    assert "strand orientation is swapped" in capsys.readouterr().err
