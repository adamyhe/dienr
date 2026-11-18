# Changelog

Notable changes to `dienr`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: a minor version may change
output). Section numbers (§) refer to the
[implementation plan](notes/denr-python-implementation-plan.md), which has the
measurements behind each entry.

## [Unreleased]

## [0.2.0] — 2026-11-16

A correctness release. Several fixes change output; the affected cases are
noted under each entry. Fits with the defaults (legacy objective, unit
exposure, no pinned models, no run-on tail, windows inside their contigs)
produce the same abundances as 0.1.0, except for models with no unmasked
bins, which are now `NaN`.

### Changed

- **Models with no unmasked bins are reported as `NaN`, not as an estimate.**
  An all-zero column fits any abundance equally well; it used to come back as
  `0.0` (legacy) or the Poisson starting floor with status `ok`, and made every
  other model at the locus lose its standard error. Both solvers now fit only
  the supported columns.
- **`model_abundance.solver_status` carries the qualified status** from
  `locus_qc` (`rank_deficient`, `ill_conditioned`). It said `ok` where QC
  said otherwise. Scripts filtering `solver_status == "ok"` should keep every
  converged status (§18.29).
- **`transcript_model_map` `start`/`end` are each transcript's own span**,
  not the locus span repeated per row. Column names and types are unchanged.
- **The manifest is strict JSON.** Non-finite floats (e.g. the standard error
  of an assumed Poisson dispersion, formerly `Infinity`) are written as `null`.
- **`n_inactive_transcripts` counts only identifiers present in the design**;
  the rest are recorded as `n_inactive_transcripts_not_in_design`, with a
  warning.
- **A run-on tail must be applied before a shape profile.**
  `apply_runon_tail` now rebuilds each locus from its transcripts and refuses
  a reshaped design; `build-design` applies them in that order.

### Fixed

- **Quasi-Poisson standard errors with an exposure.** The pooled dispersion
  ignored the sample exposure and `--use-bin-exposure`, inflating φ (e.g.
  ~3×10⁵ instead of ~1 with 250 bp bin exposure) and every standard error by
  √φ. Point estimates were unaffected, as were runs at unit exposure (§14.6).
- **Quasi-Poisson dispersion with pinned models.** Bins reached only by a
  pinned (inactive) model entered the dispersion at the μ floor; one pinned
  model could push φ from 0.91 to 1.5×10¹⁴ for the whole run.
- **Poisson fits at large exposure.** At a library-size-scale exposure the
  optimizer stopped after one iteration at a wrong estimate while reporting
  `ok`, or failed as `invalid_mean`. The fit is now invariant to the exposure
  scale; unit-exposure fits are bit-identical.
- **Windows running past a contig end** were read at the contig length and
  stretched across the design's bins, moving signal between bins (e.g. into
  run-on tail bins of the last locus on a contig). They are now read at full
  width with off-contig bases uncovered.
- **`min`/`max` summaries on signed minus-strand BigWigs** were swapped; the
  magnitude is now taken per base.
- **`locus_qc` fitted signal** (`total_fitted`, `residual_sum_of_squares`) now
  uses the mean the fit used, `c·e·M·a + b`, for count likelihoods. It was
  off by exactly the exposure. Legacy output is unchanged.
- **`gradient_norm`** is taken over the free models only; pinned models at
  their bound made converged fits report e.g. 1.6×10¹⁴.
- **`model_abundance.penalty_value`** reports the penalty at the estimate; it
  was hard-coded to `0.0`.
- **`dienr count`: an explicit exposure of `0`** in the sample sheet reaches
  the fit as `invalid_exposure` instead of becoming `1.0`. Non-numeric values
  are refused.
- **`dienr count`: contigs are checked per strand.** A contig present only in
  the minus BigWig let plus-strand loci silently come back `missing_contig`.
- **Run-on tails** (experimental): new tail bins were unmasked, so a PAS mask
  missed the most downstream isoform; transcripts in one equivalence class
  shared a tail offset from the representative's PAS; and the model table and
  transcript map were left stale. Designs without a PAS mask are unchanged.
- **Shape profiles** (experimental): head/tail positions were mapped in the
  profile's bin size rather than the design's, so a 10 kb head became 3.9 kb
  or 24.75 kb when they differed; and training genes were not isolated from
  short same-strand neighbours. Re-running the shape-profile validation under
  the fix left both verdicts unchanged (§18.28–18.29).

### Performance

- Per-locus work after a parallel fit (standard errors, dispersion terms,
  conditioning) runs on the worker pool instead of serially in the main
  process: 1.2–1.5× faster at 4–8 jobs, bit-identical (§14.6).
- On K562, quasi-Poisson is 1.24× / 1.47× faster than 0.1.0 at 4 / 8 jobs.
  At one job (the library default) the correctness fixes' bookkeeping leaves
  legacy ~6% and quasi-Poisson ~11% slower than 0.1.0, after recovering part
  of it (§14.7).

## [0.1.0] — 2026-11-10

First release on PyPI.

### Added

- The DENR pipeline end to end: GTF/GFF annotation → loci → binned design
  with masks and collapsed equivalence classes → stranded BigWig aggregation
  → legacy least-squares fit (identity or log) → versioned Parquet results
  with a manifest.
- CLI: `build-design`, `count`, `fit`, `validate`, plus `shape-profile` and
  `runon-tail`.
- **R parity** with DENR 1.0.0 (commit `09d6cde`) on 12/12 comparable
  scenarios: locus membership, bins, masks, design matrix, equivalence
  classes, binned signal, predicted signal, objectives and abundances.
- An explicit status for every sample × locus fit; failed fits are `NaN`,
  never zero.
- Input checks: strand-orientation check on every sample, contig-name
  checks, and explicit contig renaming (`--contig-map FILE|PRESET`).
- Masking in bases (`mask_tss`/`mask_pas`) alongside DENR's bin counts.
- Experimental options, tested but not recommended (§18.27): Poisson and
  quasi-Poisson likelihoods, `--log-offset`, run-on tails, shape profiles,
  ridge/lasso/elastic-net penalties.
- Fixes for DENR's shape-profile defects (strand dispatch, negative
  minus-strand values, the unsatisfiable `min_percent_transcribed` default,
  two crashes) — see [differences from DENR](docs/differences-from-denr.md).
- Parallel fitting (`--jobs`), batched BigWig reads via figwig, and a BLAS
  thread cap. About 14.5× faster than R DENR with 8–9× less memory (§14.2).
- User guide in `docs/`, `CONTRIBUTING.md`, `CITATION.cff`.

[Unreleased]: https://github.com/adamyhe/dienr/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/adamyhe/dienr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/adamyhe/dienr/releases/tag/v0.1.0
