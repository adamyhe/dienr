"""CLI stage separation, exit codes, and input validation."""

from __future__ import annotations

import polars as pl
import pytest

from dienr.cli import COUNTS_FILENAME, DESIGN_FILENAME, main
from dienr.results import FitResult

from .conftest import write_gtf


@pytest.fixture
def annotation_gtf(tmp_path):
    """Two plus-strand and one minus-strand transcript inside fixture bounds."""
    return write_gtf(
        tmp_path / "ann.gtf",
        [
            {
                "transcript_id": "tx1",
                "gene_id": "g1",
                "contig": "1",
                "start": 1001,
                "end": 11_000,
                "strand": "+",
            },
            {
                "transcript_id": "tx2",
                "gene_id": "g1",
                "contig": "1",
                "start": 1001,
                "end": 6_000,
                "strand": "+",
            },
            {
                "transcript_id": "tx3",
                "gene_id": "g2",
                "contig": "1",
                "start": 50_001,
                "end": 60_000,
                "strand": "-",
            },
        ],
    )


@pytest.fixture
def samples_csv(tmp_path, extdata):
    path = tmp_path / "samples.csv"
    pl.DataFrame(
        {
            "sample_id": ["s1"],
            "bigwig_plus": [str(extdata / "test_double_strand_plus.bw")],
            "bigwig_minus": [str(extdata / "test_double_strand_minus.bw")],
        }
    ).write_csv(path)
    return path


def test_full_cli_pipeline(annotation_gtf, samples_csv, tmp_path, capsys):
    """build-design, count, fit and validate, as separate cached stages."""
    design_dir = tmp_path / "design"
    counts_dir = tmp_path / "counts"
    results_dir = tmp_path / "results"

    assert (
        main(
            [
                "build-design",
                "--gtf",
                str(annotation_gtf),
                "--output",
                str(design_dir),
                "--bin-size",
                "250",
                "--mask-start-bins",
                "2",
                "0",
            ]
        )
        == 0
    )
    assert (design_dir / DESIGN_FILENAME).exists()
    # Design artifacts are readable without unpickling.
    assert pl.read_parquet(design_dir / "loci.parquet").height == 2

    assert (
        main(
            [
                "count",
                "--design",
                str(design_dir),
                "--samples",
                str(samples_csv),
                "--output",
                str(counts_dir),
            ]
        )
        == 0
    )
    assert (counts_dir / COUNTS_FILENAME).exists()

    assert (
        main(
            [
                "fit",
                "--design",
                str(design_dir),
                "--counts",
                str(counts_dir),
                "--output",
                str(results_dir),
            ]
        )
        == 0
    )
    result = FitResult.read(results_dir)
    assert result.model_abundance.height > 0
    assert result.summary.n_failures == 0

    assert main(["validate", "--results", str(results_dir)]) == 0
    assert "schema_version" in capsys.readouterr().out


def test_design_is_reusable_across_count_runs(annotation_gtf, samples_csv, tmp_path):
    """The whole point of separate stages: geometry is built once."""
    design_dir = tmp_path / "design"
    main(["build-design", "--gtf", str(annotation_gtf), "--output", str(design_dir)])

    for name in ("counts_a", "counts_b"):
        assert (
            main(
                [
                    "count",
                    "--design",
                    str(design_dir),
                    "--samples",
                    str(samples_csv),
                    "--output",
                    str(tmp_path / name),
                ]
            )
            == 0
        )


def test_fit_exits_nonzero_when_failures_exceed_threshold(annotation_gtf, tmp_path, extdata):
    """A run that failed most loci must not look successful."""
    # Point the annotation at a contig the BigWigs do not declare.
    gtf = write_gtf(
        tmp_path / "absent.gtf",
        [
            {
                "transcript_id": "tx1",
                "gene_id": "g1",
                "contig": "chrAbsent",
                "start": 1001,
                "end": 11_000,
                "strand": "+",
            }
        ],
    )
    samples = tmp_path / "s.csv"
    pl.DataFrame(
        {
            "sample_id": ["s1"],
            "bigwig_plus": [str(extdata / "test_double_strand_plus.bw")],
            "bigwig_minus": [str(extdata / "test_double_strand_minus.bw")],
        }
    ).write_csv(samples)

    design_dir, counts_dir, results_dir = (tmp_path / "d", tmp_path / "c", tmp_path / "r")
    main(["build-design", "--gtf", str(gtf), "--output", str(design_dir)])
    main(
        [
            "count",
            "--design",
            str(design_dir),
            "--samples",
            str(samples),
            "--output",
            str(counts_dir),
        ]
    )

    code = main(
        [
            "fit",
            "--design",
            str(design_dir),
            "--counts",
            str(counts_dir),
            "--output",
            str(results_dir),
            "--max-failure-fraction",
            "0.5",
        ]
    )
    assert code == 1
    # The failure is recorded, not hidden.
    assert FitResult.read(results_dir).summary.n_failures == 1


def test_count_rejects_unknown_value_kind(annotation_gtf, tmp_path, extdata):
    """A typo must fail loudly rather than defeat the count-equivalence check."""
    design_dir = tmp_path / "design"
    main(["build-design", "--gtf", str(annotation_gtf), "--output", str(design_dir)])

    samples = tmp_path / "bad.csv"
    pl.DataFrame(
        {
            "sample_id": ["s1"],
            "bigwig_plus": [str(extdata / "test_double_strand_plus.bw")],
            "bigwig_minus": [str(extdata / "test_double_strand_minus.bw")],
            "value_kind": ["libary_normalised"],
        }
    ).write_csv(samples)

    with pytest.raises(SystemExit, match="value_kind"):
        main(
            [
                "count",
                "--design",
                str(design_dir),
                "--samples",
                str(samples),
                "--output",
                str(tmp_path / "c"),
            ]
        )


def test_count_rejects_missing_columns(annotation_gtf, tmp_path):
    design_dir = tmp_path / "design"
    main(["build-design", "--gtf", str(annotation_gtf), "--output", str(design_dir)])
    samples = tmp_path / "thin.csv"
    pl.DataFrame({"sample_id": ["s1"]}).write_csv(samples)
    with pytest.raises(SystemExit, match="missing column"):
        main(
            [
                "count",
                "--design",
                str(design_dir),
                "--samples",
                str(samples),
                "--output",
                str(tmp_path / "c"),
            ]
        )


def test_count_without_a_design_fails_clearly(tmp_path, samples_csv):
    with pytest.raises(SystemExit, match="build-design"):
        main(
            [
                "count",
                "--design",
                str(tmp_path / "nope"),
                "--samples",
                str(samples_csv),
                "--output",
                str(tmp_path / "c"),
            ]
        )


def test_fit_without_counts_fails_clearly(annotation_gtf, tmp_path):
    design_dir = tmp_path / "design"
    main(["build-design", "--gtf", str(annotation_gtf), "--output", str(design_dir)])
    with pytest.raises(SystemExit, match="dienr count"):
        main(
            [
                "fit",
                "--design",
                str(design_dir),
                "--counts",
                str(tmp_path / "nope"),
                "--output",
                str(tmp_path / "r"),
            ]
        )


def test_unsupported_models_are_warned_about(tmp_path, capsys):
    """A transcript that rounds away must not vanish silently."""
    gtf = write_gtf(
        tmp_path / "tiny.gtf",
        [
            {
                "transcript_id": "tiny",
                "gene_id": "g",
                "contig": "1",
                "start": 1000,
                "end": 1010,
                "strand": "+",
            }
        ],
    )
    assert main(["build-design", "--gtf", str(gtf), "--output", str(tmp_path / "d")]) == 0
    assert "no bin support" in capsys.readouterr().err


def test_inactive_transcript_file_is_applied(annotation_gtf, samples_csv, tmp_path):
    design_dir, counts_dir, results_dir = (tmp_path / "d", tmp_path / "c", tmp_path / "r")
    main(["build-design", "--gtf", str(annotation_gtf), "--output", str(design_dir)])
    main(
        [
            "count",
            "--design",
            str(design_dir),
            "--samples",
            str(samples_csv),
            "--output",
            str(counts_dir),
        ]
    )

    inactive = tmp_path / "inactive.txt"
    inactive.write_text("tx3\n")
    assert (
        main(
            [
                "fit",
                "--design",
                str(design_dir),
                "--counts",
                str(counts_dir),
                "--output",
                str(results_dir),
                "--inactive-transcripts",
                str(inactive),
            ]
        )
        == 0
    )

    result = FitResult.read(results_dir)
    assert result.provenance["n_inactive_transcripts"] == 1
    minus_locus = result.transcript_model_map.filter(pl.col("transcript_id") == "tx3")["locus_id"][
        0
    ]
    pinned = result.model_abundance.filter(pl.col("locus_id") == minus_locus)
    assert (pinned["abundance_raw"] == 0.0).all()


def test_heavily_masked_loci_are_warned_about(tmp_path, capsys):
    """A locus fitted on a fraction of its bins must say so at build time."""
    gtf = write_gtf(
        tmp_path / "narrow.gtf",
        [
            {
                "transcript_id": "tx",
                "gene_id": "g",
                "contig": "1",
                "start": 1_001,
                "end": 2_000,
                "strand": "+",
            }
        ],
    )
    assert (
        main(
            [
                "build-design",
                "--gtf",
                str(gtf),
                "--output",
                str(tmp_path / "d"),
                "--mask-tss",
                "0",
                "600",
            ]
        )
        == 0
    )
    assert "lost more than half their bins" in capsys.readouterr().err


def test_poisson_via_the_cli(annotation_gtf, samples_csv, tmp_path):
    """The count likelihoods must be reachable end to end from the CLI.

    Counting has to use --summary sum, since the parity default of mean is
    rejected for a count likelihood.
    """
    design_dir, counts_dir, results_dir = (tmp_path / "d", tmp_path / "c", tmp_path / "r")
    main(["build-design", "--gtf", str(annotation_gtf), "--output", str(design_dir)])
    main(
        [
            "count",
            "--design",
            str(design_dir),
            "--samples",
            str(samples_csv),
            "--output",
            str(counts_dir),
            "--summary",
            "sum",
        ]
    )
    assert (
        main(
            [
                "fit",
                "--design",
                str(design_dir),
                "--counts",
                str(counts_dir),
                "--output",
                str(results_dir),
                "--likelihood",
                "poisson",
                "--penalty",
                "ridge",
                "--lambda2",
                "0.01",
            ]
        )
        == 0
    )
    result = FitResult.read(results_dir)
    assert result.provenance["likelihood"] == "poisson"
    assert result.provenance["penalty_type"] == "ridge"


def test_mean_summary_is_rejected_for_poisson_via_the_cli(annotation_gtf, samples_csv, tmp_path):
    """The default `dienr count` is a per-base mean, for R parity.

    So the default count output cannot feed a count likelihood, and the guard
    has to fire rather than silently fitting a density. This also pins the CLI
    default to `mean`: it was `sum`, which meant the all-defaults pipeline
    quietly diverged from the reference it is supposed to reproduce.
    """
    design_dir, counts_dir = tmp_path / "d", tmp_path / "c"
    main(["build-design", "--gtf", str(annotation_gtf), "--output", str(design_dir)])
    main(
        [
            "count",
            "--design",
            str(design_dir),
            "--samples",
            str(samples_csv),
            "--output",
            str(counts_dir),
        ]
    )
    with pytest.raises(ValueError, match="per-base density"):
        main(
            [
                "fit",
                "--design",
                str(design_dir),
                "--counts",
                str(counts_dir),
                "--output",
                str(tmp_path / "r"),
                "--likelihood",
                "poisson",
            ]
        )
