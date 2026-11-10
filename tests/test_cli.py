"""CLI stage separation, exit codes, and input validation."""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import pytest

from dienr.cli import COUNTS_FILENAME, DESIGN_FILENAME, SHAPE_FILENAME, main
from dienr.results import FitResult

from .conftest import write_gtf
from .fixtures.scenarios import SCENARIOS_BY_NAME, write_inputs


def _read_design(directory):
    """Load the pickled `Design` a build-design run wrote."""
    import pickle

    with (directory / DESIGN_FILENAME).open("rb") as handle:
        return pickle.load(handle)


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
            "--allow-contig-mismatch",
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


# --------------------------------------------------------------------------
# shape-profile
# --------------------------------------------------------------------------


@pytest.fixture
def shape_cli_inputs(tmp_path):
    """A GTF plus stranded BigWigs carrying a known 5'-peaked density."""
    from .test_shape import _make_genes, _write_signal

    genes = _make_genes(30)
    gtf = write_gtf(
        tmp_path / "shape.gtf",
        [
            {
                "transcript_id": row["transcript_id"],
                "gene_id": row["transcript_id"],
                "contig": row["contig"],
                "start": row["start"] + 1,
                "end": row["end"],
                "strand": row["strand"],
            }
            for row in genes.iter_rows(named=True)
        ],
    )
    plus = _write_signal(tmp_path / "sp.bw", genes, "+")
    minus = _write_signal(tmp_path / "sm.bw", genes, "-")
    return gtf, plus, minus


def test_shape_profile_command_writes_a_reusable_profile(shape_cli_inputs, tmp_path, capsys):
    gtf, plus, minus = shape_cli_inputs
    out = tmp_path / "prof"
    code = main(
        [
            "shape-profile",
            "--gtf",
            str(gtf),
            "--bigwig-plus",
            str(plus),
            "--bigwig-minus",
            str(minus),
            "--output",
            str(out),
            "--span",
            "0.1",
        ]
    )
    assert code == 0
    assert (out / SHAPE_FILENAME).exists()
    curve = pl.read_parquet(out / "shape_profile.parquet")
    assert curve.columns == ["position", "relative_density"]
    printed = capsys.readouterr().out
    assert "training genes" in printed
    assert "dynamic range" in printed


def test_shape_profile_command_reports_a_flat_profile_on_stderr(tmp_path, capsys):
    """The reference's silent failure, made loud at the command line."""
    from .test_shape import _make_genes, _write_signal

    genes = _make_genes(20, strands=("+",))
    gtf = write_gtf(
        tmp_path / "flat.gtf",
        [
            {
                "transcript_id": row["transcript_id"],
                "gene_id": row["transcript_id"],
                "contig": row["contig"],
                "start": row["start"] + 1,
                "end": row["end"],
                "strand": row["strand"],
            }
            for row in genes.iter_rows(named=True)
        ],
    )
    flat = _write_signal(tmp_path / "flat.bw", genes, "+", peak=0.0, slope=0.0)
    empty = _write_signal(tmp_path / "empty.bw", genes, "-")
    with pytest.warns(UserWarning):
        code = main(
            [
                "shape-profile",
                "--gtf",
                str(gtf),
                "--bigwig-plus",
                str(flat),
                "--bigwig-minus",
                str(empty),
                "--output",
                str(tmp_path / "p"),
                "--span",
                "0.1",
            ]
        )
    assert code == 0
    assert "spans only" in capsys.readouterr().err


def test_shape_profile_command_fails_cleanly_when_no_gene_qualifies(shape_cli_inputs, tmp_path):
    gtf, plus, minus = shape_cli_inputs
    with pytest.raises(SystemExit, match="no gene passed"):
        main(
            [
                "shape-profile",
                "--gtf",
                str(gtf),
                "--bigwig-plus",
                str(plus),
                "--bigwig-minus",
                str(minus),
                "--output",
                str(tmp_path / "p"),
                "--min-length",
                "500000",
            ]
        )


def test_build_design_applies_a_shape_profile(shape_cli_inputs, tmp_path, capsys):
    gtf, plus, minus = shape_cli_inputs
    profile_dir = tmp_path / "prof"
    main(
        [
            "shape-profile",
            "--gtf",
            str(gtf),
            "--bigwig-plus",
            str(plus),
            "--bigwig-minus",
            str(minus),
            "--output",
            str(profile_dir),
            "--span",
            "0.1",
        ]
    )
    capsys.readouterr()

    plain_dir = tmp_path / "plain"
    shaped_dir = tmp_path / "shaped"
    assert main(["build-design", "--gtf", str(gtf), "--output", str(plain_dir)]) == 0
    plain_out = capsys.readouterr().out
    assert (
        main(
            [
                "build-design",
                "--gtf",
                str(gtf),
                "--output",
                str(shaped_dir),
                "--shape-profile",
                str(profile_dir),
            ]
        )
        == 0
    )
    shaped_out = capsys.readouterr().out
    assert "reshaped by profile" in shaped_out
    assert "reshaped by profile" not in plain_out

    plain = _read_design(plain_dir)
    shaped = _read_design(shaped_dir)
    assert plain.shape_digest is None
    assert shaped.shape_digest is not None
    assert shaped.digest != plain.digest


def test_build_design_warns_when_masking_cancels_the_profile(shape_cli_inputs, tmp_path, capsys):
    """The two features compete, and a run that neutralizes one must say so.

    Masking calls the pausing peak untrustworthy; the profile calls it
    predictable. Applying both leaves the design barely graded, which looks
    exactly like a working shape profile from the outside.
    """
    gtf, plus, minus = shape_cli_inputs
    profile_dir = tmp_path / "prof"
    main(
        [
            "shape-profile",
            "--gtf",
            str(gtf),
            "--bigwig-plus",
            str(plus),
            "--bigwig-minus",
            str(minus),
            "--output",
            str(profile_dir),
            "--span",
            "0.1",
        ]
    )
    capsys.readouterr()
    main(
        [
            "build-design",
            "--gtf",
            str(gtf),
            "--output",
            str(tmp_path / "masked"),
            "--shape-profile",
            str(profile_dir),
            "--mask-tss",
            "0",
            "20000",
            "--mask-pas",
            "10000",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert "compete for the same" in captured.err


def test_build_design_rejects_a_missing_profile_directory(annotation_gtf, tmp_path):
    with pytest.raises(SystemExit, match="no shape_profile.pkl"):
        main(
            [
                "build-design",
                "--gtf",
                str(annotation_gtf),
                "--output",
                str(tmp_path / "d"),
                "--shape-profile",
                str(tmp_path / "absent"),
            ]
        )


def test_shape_profile_command_reports_dropped_genes(tmp_path, capsys):
    """A summary that reports only surviving genes hides how selective it was."""
    from .test_shape import _make_genes, _write_signal

    genes = pl.concat(
        [
            _make_genes(12, strands=("+",)),
            pl.DataFrame(
                [
                    {
                        "transcript_id": "short",
                        "contig": "chr1",
                        "start": 3_000_000,
                        "end": 3_004_000,
                        "strand": "+",
                    }
                ]
            ),
        ]
    )
    gtf = write_gtf(
        tmp_path / "mixed.gtf",
        [
            {
                "transcript_id": row["transcript_id"],
                "gene_id": row["transcript_id"],
                "contig": row["contig"],
                "start": row["start"] + 1,
                "end": row["end"],
                "strand": row["strand"],
            }
            for row in genes.iter_rows(named=True)
        ],
    )
    plus = _write_signal(tmp_path / "mp.bw", genes, "+")
    minus = _write_signal(tmp_path / "mm.bw", genes, "-")
    code = main(
        [
            "shape-profile",
            "--gtf",
            str(gtf),
            "--bigwig-plus",
            str(plus),
            "--bigwig-minus",
            str(minus),
            "--output",
            str(tmp_path / "prof"),
            "--span",
            "0.005",
            "--min-covered-fraction",
            "0.2",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "genes dropped: too_short=1" in captured.out


def test_shape_profile_command_warns_when_the_fit_was_floored(
    annotation_gtf, monkeypatch, tmp_path, capsys
):
    """A floored node is a value the smoother did not produce, so it is flagged.

    Driven through a stub rather than data: local linear regression undershoots
    below zero only on noisy input with a sub-bin peak, and `test_shape`
    already establishes that the smoother can do it and that the floor catches
    it. What is asserted here is that the command says so.
    """
    import numpy as np

    from dienr import cli
    from dienr.shape import ShapeConfig, ShapeProfile

    stub = ShapeProfile(
        grid=np.linspace(0.0, 1.0, 5),
        values=np.array([1e-3, 1e-3, 1.0, 1.0, 1.0]),
        config=ShapeConfig(),
        n_genes=3,
        n_observations=90,
        n_floored=2,
        rejected={},
        digest="abcdef0123456789",
    )
    monkeypatch.setattr(cli, "_fit_shape_profile", lambda *a, **k: stub)
    code = main(
        [
            "shape-profile",
            "--gtf",
            str(annotation_gtf),
            "--bigwig-plus",
            str(tmp_path / "p.bw"),
            "--bigwig-minus",
            str(tmp_path / "m.bw"),
            "--output",
            str(tmp_path / "prof"),
        ]
    )
    assert code == 0
    assert "clamped to the profile floor" in capsys.readouterr().err


@pytest.mark.parametrize("scenario", ["multi_contig", "both_strands_separate_loci"])
def test_count_threads_do_not_change_the_counts(tmp_path, scenario):
    """--threads parallelizes the read; the counts must not depend on it.

    Inputs come from the parity scenarios rather than the R package's extdata,
    so this runs whether or not `_reference/` is present.
    """
    inputs = write_inputs(SCENARIOS_BY_NAME[scenario], tmp_path / "inputs")
    samples = tmp_path / "samples.csv"
    pl.DataFrame(
        {
            "sample_id": ["s1"],
            "bigwig_plus": [str(inputs["plus"])],
            "bigwig_minus": [str(inputs["minus"])],
        }
    ).write_csv(samples)
    design_dir = tmp_path / "design"
    main(["build-design", "--gtf", str(inputs["gtf"]), "--output", str(design_dir)])

    outputs = {}
    for threads in ("1", "3", "8"):
        out = tmp_path / f"t{threads}"
        base = ["count", "--design", str(design_dir), "--samples", str(samples)]
        assert main([*base, "--output", str(out), "--threads", threads]) == 0
        with (out / COUNTS_FILENAME).open("rb") as handle:
            outputs[threads] = pickle.load(handle)[0]

    reference = outputs["1"]
    assert reference.total_signal() > 0  # a vacuous all-zero comparison proves nothing
    for counts in outputs.values():
        assert counts.locus_status == reference.locus_status
        for x, y in zip(counts.values, reference.values, strict=True):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("bad", ["0", "-1", "two"])
def test_count_rejects_a_bad_thread_count(tmp_path, bad, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "count",
                "--design",
                str(tmp_path / "d"),
                "--samples",
                str(tmp_path / "s.csv"),
                "--output",
                str(tmp_path / "c"),
                "--threads",
                bad,
            ]
        )
    assert excinfo.value.code == 2  # argparse usage error, before any work
    assert "--threads" in capsys.readouterr().err
