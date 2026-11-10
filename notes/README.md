# notes/

Maintainer records. **User documentation is in [`docs/`](../docs)**; nothing
here is needed to use `dienr`.

## What lives where

| | `docs/` | `notes/` |
|---|---|---|
| audience | users | maintainers and reviewers |
| answers | how do I run it, what should I choose, why | why is the code like this; what was tested, how, and what it showed |
| style | current recommendations only | append-only history, corrections kept visible |
| cites | plan sections for evidence | code, scripts, seeds and commits |

## `denr-python-implementation-plan.md`

The **binding design spec** (sections 1-16: architecture, conventions, input
and output contracts, failure policy, phases) and the **validation record**
(section 17, decision gates; section 18, shape profiles, density models and
run-on, 18.1-18.14). Read the relevant section before changing behaviour.

It is deliberately one file with stable section numbers. "Plan §18.10" is
cited from code docstrings, tests, analysis scripts, commit messages and pull
requests. Splitting or renumbering it would orphan those references, so don't.
Add new material as new subsections.

### Conventions for the validation record

These came from mistakes in this record. Each one is documented in the section
named.

- **Fix the decision rule before running**, in the analysis script's
  docstring, and report the verdict by that rule even when it is
  inconvenient. A rule changed after seeing data is recorded as post hoc
  (18.9, 18.10).
- **Corrections are added, not overwritten.** Mark the superseded text
  "Corrected by 18.x" and keep it, with the old numbers where they matter
  (18.6, 18.9, 18.10).
- **Record every simulation seed you report.** Three of 18.8's seeds were never
  recorded and its table can't be reproduced (18.11).
- **Pair arms on a common locus set, and check how many loci each arm drops.**
  An arm failing at 60% of loci silently restricted 18.9 and 18.10 to the
  easy quarter (18.10).
- **Scoring must not depend on row order.** Ties broken by Polars row order
  made identical runs disagree (18.11).
- **A per-arm calibration constant makes identical fits score differently.**
  Count those loci as ties (18.12).
- **Stratify by something measured independently of the outcome.** Binning
  loci by the comparator arm's own held-out deviance invented a "harm at
  low-signal loci" by regression to the mean (18.17).
- **Heavy jobs run on the cluster.** Analysis scripts take paths and write
  JSON; aggregation is a separate `--aggregate` pass.

## Analysis scripts

They live in [`scripts/`](../scripts), one per experiment. The docstring holds
the question, the arms and the pre-registered rule; the plan section holds the
results.

| script | plan |
|---|---|
| `validate_likelihoods.py` | 17 (Gate B) |
| `benchmark_shape_accuracy.py` | 18.6 |
| `validate_shape_usefulness.py` | 18.8, 18.11 |
| `oracle_shape_ceiling.py`, `export_archetypes.R` | 18.9 |
| `runon_trial.py`, `runon_diagnose.py` | 18.10 |
| `runon_overlap_trial.py` | 18.12 |
| `shape_runon_trial.py` | 18.13 |
| `k562_runon_check.py` | 18.14 |
| `k562_pergene_trial.py` | 18.17 |
| `k562_pasmask_trial.py` | 18.18 |
| `k562_tssmask_trial.py` | 18.19 |
| `k562_longread_accuracy.py` (shared long-read harness) | 18.20 |
| `k562_longread_logmask.py` | 18.21 |
| `k562_longread_defaults.py` | 18.22 |
| `k562_longread_sweep.py` | 18.23 |
| `k562_longread_headtohead.py`, `k562_longread_explore.py` | 18.24 |
| `k562_tail_mask_diagnose.py` | 18.25 |
| `k562_longread_tailfix.py` | 18.26 |
| `profile_k562.py` | 14.4 |
| `simulate_nascentrnasim.R` | simulation inputs for 18.6-18.13 |
| `probe_r_shape_profile.R` | 18.1-18.7 (R reference defects) |
| `benchmark_*` | 14 (performance) |
| `export_r_golden.R` | R parity goldens (`tests/fixtures/parity/README.md`) |
