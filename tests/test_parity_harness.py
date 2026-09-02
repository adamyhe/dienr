"""Tests for the parity comparison machinery itself.

The parity suite is entirely skipped until Phase 0 produces R golden files, so
without these tests its plumbing — column-major reshaping, locus matching by
geometry, 0-based mask indices, the sign convention on minus-strand signal —
would be unexercised until someone finally ran R and hit the bugs.

These build a golden file **from our own implementation**, in exactly the shape
`scripts/export_r_golden.R` emits, and confirm the comparisons pass on it.

That is a self-consistency check, not parity. It proves the harness works; it
proves nothing about agreement with R. Nothing written here is ever placed in
``tests/fixtures/parity/``, and `test_synthetic_goldens_are_never_committed`
enforces that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dienr.annotation import read_annotation
from dienr.counts import SignalFile, count_sample
from dienr.design import build_design
from dienr.solvers.legacy import fit_locus, objective

from . import test_parity
from .fixtures.scenarios import SCENARIOS_BY_NAME, write_inputs

#: A scenario per interesting geometry, kept small so this stays fast.
HARNESS_SCENARIOS = [
    "two_distinguishable_isoforms",
    "duplicate_models_collapse",
    "rank_deficient_union",
    "minus_strand_orientation",
    "masked_ends",
    "terminal_partial_bin",
    "both_strands_separate_loci",
    "multi_contig",
    "absent_contig",
]


def emit_golden_from_our_pipeline(scenario, directory) -> dict:
    """Produce a golden dict in `export_r_golden.R`'s output format.

    Deliberately mirrors the R script's conventions rather than our internals:
    column-major matrix flattening, a locus dict keyed by an opaque id, and
    mask indices as a sorted 0-based list.
    """
    paths = write_inputs(scenario, directory)
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
        "harness",
        (SignalFile(paths["plus"], "+"), SignalFile(paths["minus"], "-")),
    )

    loci = {}
    for row in design.loci.iter_rows(named=True):
        locus_id = row["locus_id"]
        models = design.models[locus_id]
        bins = design.bins.filter(design.bins["locus_id"] == locus_id).sort("bin_index")
        fit = fit_locus(
            models,
            counts.values[locus_id],
            transform=scenario.transform,
            prior_status=counts.locus_status[locus_id],
        )
        # R's group ids are opaque to us, so use a deliberately unrelated key
        # to prove matching is by geometry rather than by index.
        loci[f"rgroup_{97 - locus_id}"] = {
            "locus_key": f"rgroup_{97 - locus_id}",
            "bin_starts": bins["start"].to_list(),
            "bin_ends": bins["end"].to_list(),
            "contig": row["contig"],
            "strand": row["strand"],
            "n_bins": models.shape[0],
            "n_models": models.shape[1],
            "models": models.flatten(order="F").tolist(),
            "counts": counts.values[locus_id].tolist(),
            "masked_bins_0based": np.flatnonzero(design.mask(locus_id)).tolist(),
            "abundance": fit.abundance.tolist(),
            "objective": objective(
                fit.abundance, models, counts.values[locus_id], scenario.transform
            ),
            "predicted": (models @ fit.abundance).tolist(),
        }

    grouped = design.transcript_map.group_by("locus_id", "model_index").agg(
        tx=__import__("polars").col("transcript_id")
    )
    tx_name, group, model = [], [], []
    for record in grouped.iter_rows(named=True):
        for name in record["tx"]:
            tx_name.append(name)
            group.append(int(record["locus_id"]) + 1)
            model.append(int(record["model_index"]) + 1)

    return {
        "scenario": scenario.name,
        "reference_commit": "SYNTHETIC-NOT-R",
        "denr_version": "synthetic",
        "transcript_model_key": {"tx_name": tx_name, "group": group, "model": model},
        "loci": loci,
    }


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Install synthetic goldens into a temporary golden directory."""
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    monkeypatch.setattr(test_parity, "GOLDEN_DIR", golden_dir)

    def _install(name: str):
        scenario = SCENARIOS_BY_NAME[name]
        golden = emit_golden_from_our_pipeline(scenario, tmp_path / "inputs" / name)
        (golden_dir / f"{name}.json").write_text(json.dumps(golden))
        return scenario

    return _install


@pytest.mark.parametrize("name", HARNESS_SCENARIOS)
def test_all_comparisons_pass_on_a_self_consistent_golden(name, harness, built):
    """Every comparison in the parity suite must run and pass on our own output."""
    harness(name)
    test_parity.test_locus_membership_matches(name, built)
    test_parity.test_bin_boundaries_match(name, built)
    test_parity.test_mask_indices_match(name, built)
    test_parity.test_reduced_design_matrix_matches(name, built)
    test_parity.test_binned_signal_matches(name, built)
    test_parity.test_equivalence_classes_match(name, built)
    test_parity.test_predicted_signal_and_objective_match(name, built)


@pytest.mark.parametrize(
    "name", ["two_distinguishable_isoforms", "terminal_partial_bin", "multi_contig"]
)
def test_abundance_comparison_runs_on_full_rank_loci(name, harness, built):
    harness(name)
    test_parity.test_abundance_matches_where_identifiable(name, built)


# --- the comparisons must actually be able to fail --------------------------


def test_shifted_bins_are_detected(harness, built, tmp_path):
    """A one-base coordinate shift must fail, or the test proves nothing."""
    name = "two_distinguishable_isoforms"
    harness(name)
    path = test_parity.GOLDEN_DIR / f"{name}.json"
    golden = json.loads(path.read_text())
    for locus in golden["loci"].values():
        locus["bin_starts"] = [value + 1 for value in locus["bin_starts"]]
    path.write_text(json.dumps(golden))

    with pytest.raises(AssertionError):
        test_parity.test_bin_boundaries_match(name, built)


def test_wrong_mask_indices_are_detected(harness, built):
    name = "masked_ends"
    harness(name)
    path = test_parity.GOLDEN_DIR / f"{name}.json"
    golden = json.loads(path.read_text())
    for locus in golden["loci"].values():
        locus["masked_bins_0based"] = []
    path.write_text(json.dumps(golden))

    with pytest.raises(AssertionError):
        test_parity.test_mask_indices_match(name, built)


def test_transposed_matrix_is_detected(harness, built):
    """Guards the column-major reshape, an easy thing to get backwards."""
    name = "rank_deficient_union"
    harness(name)
    path = test_parity.GOLDEN_DIR / f"{name}.json"
    golden = json.loads(path.read_text())
    for locus in golden["loci"].values():
        matrix = np.asarray(locus["models"]).reshape(
            (locus["n_bins"], locus["n_models"]), order="F"
        )
        # Row-major flattening of the same matrix: a plausible mistake.
        locus["models"] = matrix.flatten(order="C").tolist()
    path.write_text(json.dumps(golden))

    with pytest.raises(AssertionError):
        test_parity.test_reduced_design_matrix_matches(name, built)


def test_wrong_equivalence_classes_are_detected(harness, built):
    name = "duplicate_models_collapse"
    harness(name)
    path = test_parity.GOLDEN_DIR / f"{name}.json"
    golden = json.loads(path.read_text())
    # Split the collapsed class into singletons.
    key = golden["transcript_model_key"]
    key["model"] = list(range(1, len(key["tx_name"]) + 1))
    path.write_text(json.dumps(golden))

    with pytest.raises(AssertionError, match="equivalence classes differ"):
        test_parity.test_equivalence_classes_match(name, built)


def test_missing_locus_is_detected(harness, built):
    name = "both_strands_separate_loci"
    harness(name)
    path = test_parity.GOLDEN_DIR / f"{name}.json"
    golden = json.loads(path.read_text())
    # Move one locus to a contig we do not have.
    locus = next(iter(golden["loci"].values()))
    locus["contig"] = "chrGhost"
    path.write_text(json.dumps(golden))

    with pytest.raises(AssertionError, match="no counterpart"):
        test_parity.test_locus_membership_matches(name, built)


def test_missing_golden_skips_rather_than_passes(harness, built):
    """An absent golden must skip loudly, never silently succeed."""
    with pytest.raises(pytest.skip.Exception, match="no R golden values"):
        test_parity.load_golden("two_distinguishable_isoforms")


# --- guard against a synthetic golden being mistaken for parity -------------


def test_synthetic_goldens_are_never_committed():
    """No committed golden may carry the synthetic marker.

    If one ever did, the parity suite would report verified agreement with R
    while actually comparing our implementation to itself.
    """
    directory = test_parity.GOLDEN_DIR
    if not directory.exists():
        pytest.skip("no golden directory yet")
    for path in directory.glob("*.json"):
        golden = json.loads(path.read_text())
        assert golden.get("reference_commit") != "SYNTHETIC-NOT-R", (
            f"{path} is a self-consistency fixture, not an R golden. "
            "Delete it; parity must come from scripts/export_r_golden.R."
        )
        assert golden.get("reference_commit"), (
            f"{path} has no reference_commit. Re-export with "
            "--commit <denr sha> so a failure can name the revision."
        )
