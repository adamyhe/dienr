"""Parity against golden values frozen from the R DENR reference.

How this works
--------------
Inputs are generated from `tests.fixtures.scenarios`, never copied from the R
package. Golden values live in ``tests/fixtures/parity/<scenario>.json``,
produced by ``scripts/export_r_golden.R`` inside the reference environment.

Every test here **skips with an actionable message when its golden file is
absent**, so the suite stays green before Phase 0 is done. A skip is not a
pass: `test_parity_coverage_is_reported` prints how many scenarios are still
unverified, and `test_no_golden_file_is_silently_ignored` fails if a golden
file exists for a scenario name nobody defined.

What is compared, per plan section 13: locus membership, bin boundaries, mask
indices, the reduced design matrix, the transcript-to-model mapping, binned
signal, fitted abundance where the solution is unique, and predicted signal and
objective everywhere — including where it is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dienr.solvers.legacy import fit_locus, objective

from .fixtures.scenarios import SCENARIOS, SCENARIOS_BY_NAME

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "parity"

#: Absolute tolerance for geometry, which must agree exactly up to float noise.
GEOMETRY_ATOL = 1e-12

#: Relative tolerance for optimizer-dependent quantities. Two different
#: implementations of L-BFGS-B will not agree to machine precision.
FIT_RTOL = 1e-4


def as_array(value) -> np.ndarray:
    """Coerce a golden field to a 1-D array.

    `jsonlite::toJSON(auto_unbox = TRUE)` renders a length-1 R vector as a bare
    JSON scalar rather than a one-element list, so any field that is normally a
    list arrives as a scalar in single-transcript or single-bin scenarios.
    """
    return np.atleast_1d(np.asarray(value))


def as_list(value) -> list:
    """`as_array` for string fields, preserving Python types."""
    return [value] if isinstance(value, str) else list(value)


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def load_golden(name: str) -> dict:
    """Load a golden file, skipping the test when it has not been produced."""
    scenario = SCENARIOS_BY_NAME[name]
    if scenario.r_incompatible:
        pytest.skip(f"{name}: no R counterpart by design. {scenario.r_incompatible}")
    path = golden_path(name)
    if not path.exists():
        pytest.skip(
            f"no R golden values for {name!r}. Produce them with:\n"
            f"  uv run python -m tests.fixtures.generate --output build/parity-inputs\n"
            f"  Rscript scripts/export_r_golden.R --inputs build/parity-inputs "
            f"--output {GOLDEN_DIR}\n"
            f"See scripts/export_r_golden.R for the reference container."
        )
    return json.loads(path.read_text())


def match_loci(design, golden: dict) -> dict[int, dict]:
    """Pair our loci with R's by (contig, strand, span).

    R keys loci by an internal group id whose numbering we deliberately do not
    reproduce, so matching is by geometry. A locus that fails to match is
    itself a parity failure and is reported as one.
    """
    ours = {
        (row["contig"], row["strand"], int(row["start"])): row["locus_id"]
        for row in design.loci.iter_rows(named=True)
    }
    paired: dict[int, dict] = {}
    unmatched = []
    for locus in golden["loci"].values():
        contig = locus["contig"] if isinstance(locus["contig"], str) else locus["contig"][0]
        strand = locus["strand"] if isinstance(locus["strand"], str) else locus["strand"][0]
        key = (contig, strand, int(as_array(locus["bin_starts"]).min()))
        if key in ours:
            paired[ours[key]] = locus
        else:
            unmatched.append(key)
    assert not unmatched, (
        f"R loci with no counterpart in our design: {unmatched}. Ours: {sorted(ours)}"
    )
    return paired


def golden_matrix(locus: dict) -> np.ndarray:
    """Rebuild R's model matrix from its column-major flattening."""
    return (
        as_array(locus["models"])
        .astype(np.float64)
        .reshape((locus["n_bins"], locus["n_models"]), order="F")
    )


ALL_NAMES = [scenario.name for scenario in SCENARIOS]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_locus_membership_matches(name, built):
    golden = load_golden(name)
    design, _ = built(SCENARIOS_BY_NAME[name])
    paired = match_loci(design, golden)
    assert len(paired) == design.n_loci == len(golden["loci"]), (
        f"{name}: {design.n_loci} loci here vs {len(golden['loci'])} in R "
        f"({SCENARIOS_BY_NAME[name].purpose})"
    )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_bin_boundaries_match(name, built):
    """Bin starts and ends must agree exactly; this is where off-by-ones show."""
    golden = load_golden(name)
    design, _ = built(SCENARIOS_BY_NAME[name])
    for locus_id, locus in match_loci(design, golden).items():
        bins = design.bins.filter(design.bins["locus_id"] == locus_id).sort("bin_index")
        np.testing.assert_array_equal(
            bins["start"].to_numpy(), as_array(locus["bin_starts"]), err_msg=f"{name} starts"
        )
        np.testing.assert_array_equal(
            bins["end"].to_numpy(), as_array(locus["bin_ends"]), err_msg=f"{name} ends"
        )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_mask_indices_match(name, built):
    golden = load_golden(name)
    design, _ = built(SCENARIOS_BY_NAME[name])
    for locus_id, locus in match_loci(design, golden).items():
        ours = np.flatnonzero(design.mask(locus_id))
        theirs = as_array(locus["masked_bins_0based"]).astype(int)
        np.testing.assert_array_equal(ours, theirs, err_msg=f"{name} locus {locus_id} mask")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_reduced_design_matrix_matches(name, built):
    """Column order is not part of the contract, so compare column multisets."""
    golden = load_golden(name)
    design, _ = built(SCENARIOS_BY_NAME[name])
    for locus_id, locus in match_loci(design, golden).items():
        ours = design.models[locus_id]
        theirs = golden_matrix(locus)
        assert ours.shape == theirs.shape, (
            f"{name} locus {locus_id}: shape {ours.shape} vs R {theirs.shape}"
        )
        ours_columns = sorted(tuple(ours[:, i]) for i in range(ours.shape[1]))
        their_columns = sorted(tuple(theirs[:, i]) for i in range(theirs.shape[1]))
        np.testing.assert_allclose(
            np.asarray(ours_columns),
            np.asarray(their_columns),
            atol=GEOMETRY_ATOL,
            err_msg=f"{name} locus {locus_id} design matrix",
        )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_binned_signal_matches(name, built):
    """Our BigWig aggregation must agree with rtracklayer's."""
    golden = load_golden(name)
    design, counts = built(SCENARIOS_BY_NAME[name])
    for locus_id, locus in match_loci(design, golden).items():
        theirs = np.abs(as_array(locus["counts"]).astype(np.float64))
        ours = counts.values[locus_id]
        # R stores unmasked counts; mask both sides before comparing.
        mask = design.mask(locus_id)
        theirs = theirs.copy()
        theirs[mask] = 0.0
        np.testing.assert_allclose(
            ours, theirs, rtol=1e-9, atol=1e-9, err_msg=f"{name} locus {locus_id} signal"
        )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_equivalence_classes_match(name, built):
    """Transcripts sharing a model in R must share one here, and vice versa."""
    golden = load_golden(name)
    design, _ = built(SCENARIOS_BY_NAME[name])

    key = golden["transcript_model_key"]
    their_groups: dict[tuple[int, int], set[str]] = {}
    for tx, group, model in zip(
        as_list(key["tx_name"]),
        as_array(key["group"]).tolist(),
        as_array(key["model"]).tolist(),
        strict=True,
    ):
        their_groups.setdefault((group, model), set()).add(tx)
    their_partition = {frozenset(members) for members in their_groups.values()}

    our_partition = {
        frozenset(
            design.transcript_map.filter(
                (design.transcript_map["locus_id"] == locus_id)
                & (design.transcript_map["model_index"] == model_index)
            )["transcript_id"].to_list()
        )
        for locus_id, model_index in zip(
            design.model_table["locus_id"], design.model_table["model_index"], strict=True
        )
    }
    assert our_partition == their_partition, (
        f"{name}: equivalence classes differ.\n  ours: {sorted(map(sorted, our_partition))}"
        f"\n  R:    {sorted(map(sorted, their_partition))}"
    )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_predicted_signal_and_objective_match(name, built):
    """Holds even where the parameter split is not unique.

    On a rank-deficient locus the abundance vector is not identifiable, but the
    fitted signal and objective value are. Comparing those is the meaningful
    check; comparing abundances there would be comparing arbitrary choices.
    """
    golden = load_golden(name)
    scenario = SCENARIOS_BY_NAME[name]
    design, counts = built(scenario)

    for locus_id, locus in match_loci(design, golden).items():
        models = design.models[locus_id]
        fit = fit_locus(
            models,
            counts.values[locus_id],
            transform=scenario.transform,  # type: ignore[arg-type]
            prior_status=counts.locus_status[locus_id],
        )
        theirs_predicted = as_array(locus["predicted"]).astype(np.float64)
        ours_predicted = models @ fit.abundance
        np.testing.assert_allclose(
            ours_predicted,
            theirs_predicted,
            rtol=FIT_RTOL,
            atol=1e-6,
            err_msg=f"{name} locus {locus_id} predicted signal",
        )
        their_objective = float(np.ravel(locus["objective"])[0])
        # Both NaN means both implementations declined to fit the locus, which
        # is agreement, not a mismatch.
        our_objective = objective(
            fit.abundance,
            models,
            counts.values[locus_id],
            scenario.transform,  # type: ignore[arg-type]
        )
        assert our_objective == pytest.approx(
            their_objective, rel=FIT_RTOL, abs=1e-6, nan_ok=True
        ), f"{name} locus {locus_id} objective"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_abundance_matches_where_identifiable(name, built):
    """Compare abundances only on full-rank loci, where they are unique."""
    golden = load_golden(name)
    scenario = SCENARIOS_BY_NAME[name]
    design, counts = built(scenario)

    compared = 0
    for locus_id, locus in match_loci(design, golden).items():
        models = design.models[locus_id]
        if np.linalg.matrix_rank(models) < models.shape[1]:
            continue
        fit = fit_locus(
            models,
            counts.values[locus_id],
            transform=scenario.transform,  # type: ignore[arg-type]
            prior_status=counts.locus_status[locus_id],
        )
        theirs = as_array(locus["abundance"]).astype(np.float64)
        # Column order is not part of the contract; compare sorted values.
        np.testing.assert_allclose(
            np.sort(fit.abundance),
            np.sort(theirs),
            rtol=FIT_RTOL,
            atol=1e-6,
            err_msg=f"{name} locus {locus_id} abundance",
        )
        compared += 1

    if compared == 0:
        pytest.skip(f"{name}: no full-rank loci to compare abundances on")


# --- meta-tests: a skip must not look like a pass ---------------------------


def test_no_golden_file_is_silently_ignored():
    """A golden file for an unknown scenario means the specs drifted."""
    if not GOLDEN_DIR.exists():
        pytest.skip(f"{GOLDEN_DIR} does not exist yet")
    orphans = sorted(
        path.stem for path in GOLDEN_DIR.glob("*.json") if path.stem not in SCENARIOS_BY_NAME
    )
    assert not orphans, (
        f"golden files with no matching scenario: {orphans}. Either restore the "
        f"scenario in tests/fixtures/scenarios.py or delete the stale golden."
    )


def test_parity_coverage_is_reported(capsys):
    """Report how much of the parity matrix is actually verified.

    This deliberately does not fail on missing goldens — Phase 0 is a separate
    task — but it makes the gap visible instead of letting a fully skipped
    parity suite read as success.
    """
    comparable = [name for name in ALL_NAMES if not SCENARIOS_BY_NAME[name].r_incompatible]
    excluded = [name for name in ALL_NAMES if SCENARIOS_BY_NAME[name].r_incompatible]
    present = sorted(name for name in comparable if golden_path(name).exists())
    missing = sorted(name for name in comparable if not golden_path(name).exists())
    with capsys.disabled():
        print(
            f"\nR parity coverage: {len(present)}/{len(comparable)} comparable "
            "scenarios verified."
            + (f" Missing: {', '.join(missing)}." if missing else " Complete.")
            + (f" Excluded as R-incompatible by design: {', '.join(excluded)}." if excluded else "")
        )
    assert len(present) + len(missing) == len(comparable)
