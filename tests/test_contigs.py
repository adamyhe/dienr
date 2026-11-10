"""Explicit contig renaming (plan section 20, item 2).

Ensembl GTFs say ``1``; UCSC BigWigs say ``chr1``. Without a map every locus
is ``missing_contig``. Renaming must be explicit, recorded, and never merge
two contigs.
"""

from __future__ import annotations

import json
import pickle

import polars as pl
import pytest

import dienr
from dienr.contigs import PRESETS, ContigMap, check_contigs, rename_contigs

from .conftest import write_gtf
from .fixtures.bigwig import write_intervals


def _annotation(contigs=("1", "2", "MT", "GL000192.1")) -> dienr.Annotation:
    rows = [
        {
            "transcript_id": f"t{i}",
            "contig": c,
            "start": 10_000,
            "end": 30_000,
            "strand": "+",
        }
        for i, c in enumerate(contigs)
    ]
    return dienr.Annotation.from_frame(pl.DataFrame(rows), source="ensembl.gtf")


# --------------------------------------------------------------------------
# Loading maps
# --------------------------------------------------------------------------


def test_presets_cover_primary_chromosomes_both_ways():
    forward = ContigMap.load("ensembl-to-ucsc")
    assert forward.mapping["1"] == "chr1"
    assert forward.mapping["MT"] == "chrM"
    assert len(forward.mapping) == 25
    back = PRESETS["ucsc-to-ensembl"]
    assert all(back[new] == old for old, new in forward.mapping.items())


def test_file_formats(tmp_path):
    tab = tmp_path / "m.tsv"
    tab.write_text("# comment\n1\tchr1\n\nMT\tchrM\nHG1007_PATCH\t\n")
    loaded = ContigMap.load(tab)
    # An empty target means "no equivalent", as in the ChromosomeMappings files.
    assert loaded.mapping == {"1": "chr1", "MT": "chrM"}
    assert loaded.source == str(tab)
    comma = tmp_path / "m.csv"
    comma.write_text("1,chr1\nMT,chrM\n")
    assert ContigMap.load(comma).digest == loaded.digest


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("1\tchr1\n2\tchr1\n", "sends two contigs"),
        ("1\tchr1\n1\tchr9\n", "mapped twice"),
        ("1\tchr1\textra\n", "expected 'old<TAB>new'"),
        ("# only a comment\n", "is empty"),
    ],
)
def test_bad_files_are_rejected(tmp_path, text, match):
    path = tmp_path / "m.tsv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        ContigMap.load(path)


def test_unknown_spec_names_the_presets(tmp_path):
    with pytest.raises(ValueError, match="ensembl-to-ucsc"):
        ContigMap.load(str(tmp_path / "absent.tsv"))


# --------------------------------------------------------------------------
# Renaming
# --------------------------------------------------------------------------


def test_rename_reports_and_records():
    annotation = _annotation()
    renamed, report = rename_contigs(annotation, ContigMap.load("ensembl-to-ucsc"))
    assert renamed.contigs == sorted(["chr1", "chr2", "chrM", "GL000192.1"])
    assert report.renamed == {"1": "chr1", "2": "chr2", "MT": "chrM"}
    assert report.unmapped == {"GL000192.1": 1}
    assert "GL000192.1" in report.message()
    assert renamed.digest != annotation.digest
    assert "ensembl-to-ucsc" in renamed.source
    assert dict(renamed.contig_map)["MT"] == "chrM"


def test_map_in_the_wrong_direction_is_an_error():
    with pytest.raises(ValueError, match="right direction"):
        rename_contigs(_annotation(), ContigMap.load("ucsc-to-ensembl"))


def test_renaming_never_merges_into_an_existing_contig():
    annotation = _annotation(contigs=("1", "chr1"))
    with pytest.raises(ValueError, match="merge"):
        rename_contigs(annotation, ContigMap.load("ensembl-to-ucsc"))


def test_contig_map_reaches_the_design_and_the_manifest(tmp_path):
    renamed, _ = rename_contigs(_annotation(), ContigMap.load("ensembl-to-ucsc"))
    design = dienr.build_design(renamed, bin_size=250)
    assert dict(design.contig_map)["1"] == "chr1"
    plain = dienr.build_design(_annotation(), bin_size=250)
    assert plain.contig_map is None
    assert design.digest != plain.digest


def test_check_contigs_hints_only_when_a_preset_would_help():
    check = check_contigs(["1", "1", "2"], {"chr1", "chr2"})
    assert check.none_match
    assert check.missing == {"1": 2, "2": 1}
    assert "--contig-map ensembl-to-ucsc" in check.message()
    partial = check_contigs(["chr1", "chrUn_x"], {"chr1"})
    assert not partial.none_match
    assert partial.hint is None
    assert not check_contigs(["chr1"], {"chr1"}).missing


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def ensembl_inputs(tmp_path):
    rows = [
        {
            "transcript_id": f"t{i}",
            "gene_id": f"g{i}",
            "contig": "1",
            "start": 10_001 + 40_000 * i,
            "end": 30_000 + 40_000 * i,
            "strand": "+-"[i % 2],
        }
        for i in range(30)
    ]
    gtf = write_gtf(tmp_path / "ensembl.gtf", rows)
    sizes = {"chr1": 2_000_000}
    plus = write_intervals(
        tmp_path / "p.bw",
        sizes,
        [("chr1", r["start"] - 1, r["end"], 1.0) for r in rows if r["strand"] == "+"],
    )
    minus = write_intervals(
        tmp_path / "m.bw",
        sizes,
        [("chr1", r["start"] - 1, r["end"], 1.0) for r in rows if r["strand"] == "-"],
    )
    sheet = tmp_path / "samples.csv"
    sheet.write_text(f"sample_id,bigwig_plus,bigwig_minus\ns,{plus},{minus}\n")
    return tmp_path, gtf, sheet


def _count(tmp, design_dir, sheet, *extra):
    from dienr.cli import main

    return main(
        [
            "count",
            "--design",
            str(design_dir),
            "--samples",
            str(sheet),
            "--output",
            str(tmp / "c"),
            "--summary",
            "sum",
            *extra,
        ]
    )


def test_count_refuses_when_no_contig_matches_and_names_the_fix(ensembl_inputs):
    from dienr.cli import main

    tmp, gtf, sheet = ensembl_inputs
    assert main(["build-design", "--gtf", str(gtf), "--output", str(tmp / "d")]) == 0
    with pytest.raises(SystemExit, match="--contig-map ensembl-to-ucsc"):
        _count(tmp, tmp / "d", sheet)
    assert _count(tmp, tmp / "d", sheet, "--allow-contig-mismatch") == 0


def test_build_design_with_contig_map_counts_and_records_it(ensembl_inputs, capsys):
    from dienr.cli import COUNTS_FILENAME, main

    tmp, gtf, sheet = ensembl_inputs
    args = ["build-design", "--gtf", str(gtf), "--output", str(tmp / "d")]
    assert main([*args, "--contig-map", "ensembl-to-ucsc"]) == 0
    assert "renamed 1 contig(s) covering 30 transcript(s)" in capsys.readouterr().out
    assert _count(tmp, tmp / "d", sheet) == 0
    (counts,) = pickle.loads((tmp / "c" / COUNTS_FILENAME).read_bytes())
    assert counts.values[0].sum() > 0
    assert (
        main(
            [
                "fit",
                "--design",
                str(tmp / "d"),
                "--counts",
                str(tmp / "c"),
                "--output",
                str(tmp / "r"),
                "--likelihood",
                "poisson",
            ]
        )
        == 0
    )
    manifest = json.loads((tmp / "r" / "manifest.json").read_text())
    assert manifest["contig_map"]["1"] == "chr1"


def test_bad_contig_map_is_a_clean_cli_error(ensembl_inputs):
    from dienr.cli import main

    tmp, gtf, _ = ensembl_inputs
    with pytest.raises(SystemExit, match="right direction"):
        main(
            [
                "build-design",
                "--gtf",
                str(gtf),
                "--output",
                str(tmp / "d"),
                "--contig-map",
                "ucsc-to-ensembl",
            ]
        )
