# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`dienr` — "Deconvolution of Isoform Expression from Nascent RNAs" — is a **greenfield Python reimplementation** of the R/Bioconductor package [DENR](https://github.com/CshlSiepelLab/DENR) (Zhao & Dukler, *Bioinformatics* 2021). The method deconvolves nascent RNA sequencing signal (PRO-seq, GRO-seq, etc., processed so each read is a single 3'-end count) into abundances of annotated pre-RNA isoform models.

**Current state: no source code exists yet.** The tracked repo is `README.md`, `LICENSE`, `.gitattributes` only. Everything substantive lives in two untracked directories:

- `notes/denr-python-implementation-plan.md` — the 600-line design document. **Read this before writing any code**; it is the binding spec for architecture, schemas, phases, and policy.
- `_reference/DENR/` — a full clone of the R reference implementation, pinned at commit `09d6cde` (`master`, 2025-04-18). This is the behavioral reference for parity testing. It is a nested git repo; never commit into it or stage it.

## Commands

Dependencies are managed with **uv**. The package lives in `src/dienr/`, tests in `tests/`.

```bash
uv sync --group dev              # create/refresh .venv from uv.lock
uv run pytest -q                 # full suite
uv run pytest tests/test_design.py::test_bins_are_anchored_at_locus_start   # one test
uv run ruff check . && uv run ruff format .
uv run mypy                      # config in pyproject; files list is set there
```

Core dependencies are NumPy/SciPy, Polars, `pybigtools` (all BigWig access) and pydantic — deliberately no genomics interval/annotation library. `pysam` (bam), `polars-bio` (intervals), PyTorch/AnnData (ml) are optional extras; pandas is not a dependency at all.

Tests that need the R reference's synthetic BigWigs skip cleanly when `_reference/` is absent (see `tests/conftest.py`).

The R reference package can be exercised for fixture generation and parity checks (`R` and `Rscript` are on PATH via miniforge; the Bioconductor dependency stack may not be installed — check before assuming):

```bash
# From _reference/DENR/
Rscript -e 'devtools::test()'                                   # full testthat suite
Rscript -e 'devtools::test(filter = "model_fitting")'           # one test file (tests/testthat/test-<filter>.R)
Rscript -e 'devtools::load_all(); ...'                          # interactive use of the reference API
```

Test fixtures live in `_reference/DENR/inst/extdata/`: synthetic stranded BigWigs plus matching `.txdb` annotations (single-strand, double-strand, multichrom, transcript-filter, UPR cases). `tests/testthat/setup-model_fitting.R` shows the canonical end-to-end R invocation and is the model for what Phase 0 must export as golden fixtures.

## Reference pipeline (what is being reimplemented)

The R package's flow, and the source files to consult for each stage:

1. **Annotation → loci → design matrices** (`R/transcript_models_build_helpers.R`, `R/transcript_quantifier-class.R`). Transcripts are grouped into strand-specific connected components (`group_transcripts`), binned at `bin_size` (default 250bp), converted to a per-locus matrix of fractional bin overlap (`create_transcript_models`), TSS/PAS-adjacent bins masked (`create_model_masks`, typically ±1kb), and then indistinguishable transcripts are collapsed to shared models (`reduce_transcript_models`). The result is the S4 `transcript_quantifier` ("tq") object: per-locus `models` matrices, `masks`, `counts`, `model_abundance`, and a `transcript_model_key` mapping transcript → locus group → model column.
2. **Signal extraction** (`R/data_handling.R`). `add_data`/`summarize_bigwig` aggregate stranded BigWigs into per-bin count vectors.
3. **Optional shape profile** (`R/transcript_shape_profile-class.R`). A loess fit of relative polymerase density vs. rescaled gene-body position, applied to reweight model matrices away from the uniform-density assumption.
4. **Optional inactive-TSS filtering** (`R/tss_predictor.R`, `R/upstream_density_ratio.R`). A bundled Keras CNN predicts inactive TSSs; two heuristics (upstream polymerase ratio ≥ log2(5), and a goodness-of-fit rescue at all-inactive loci) can override the classifier. Inactive models are pinned to ~zero via optimizer upper bounds.
5. **Fitting** (`R/model_fitting.R:110`). Per locus, `stats::optim(method = "L-BFGS-B")` minimizes sum-of-squares between observed and `models %*% abundance`, under either `transform = "identity"` or the default `transform = "log"` (log of value + 1e-3). Only `opt_result$par` is retained.
6. **Reporting** (`R/abundance_tables.R`, `R/model_fit_statistics.R`). `transcript_abundance` repeats a shared model's abundance across every transcript mapped to it.

## Implementation status

The legacy path is complete end to end: `annotation` (GTF/GFF → pre-RNA spans) → `loci` → `design` (bins, masks, model reduction) → `counts` (BigWig aggregation) → `solvers/legacy` → `pipeline.fit` → `results` (versioned Parquet + manifest), with `cli` exposing `build-design` / `count` / `fit` / `validate`. 155 tests pass, 88 skip.

Three likelihoods: `solvers/legacy` (default, the only one with R parity), `solvers/poisson`, `solvers/negative_binomial`, plus `solvers/penalties` (ridge/lasso/elastic net) and the shared `solvers/_glm` exposure machinery. Selected via `fit(likelihood=...)` or `dienr fit --likelihood`.

**Not yet written:** BAM counting, TSS-evidence mapping, normalization factor computation, the multi-sample group penalty, and ML adapters beyond `FitResult.to_numpy`.

**R parity is established: 10/10 comparable scenarios verified** against DENR 1.0.0 at commit `09d6cde`. Locus membership, bin boundaries, mask indices, the reduced design matrix, equivalence classes, binned signal, predicted signal, objective values and full-rank abundances all agree. Coverage is printed on every run, so a regression to zero cannot pass unnoticed.

## Performance status

Measured against R DENR on identical data (`scripts/benchmark_*`, plan §14.2), model counts agreeing exactly:

| regime | total | fitting | aggregation | design | peak RSS |
|---|---|---|---|---|---|
| dense (124 loci, ~18 tx/locus) | **14.9×** | 9.3× | 24× | 27× | 301 MB vs 2417 MB (8.0×) |
| sparse (955 loci, ~3 models/locus) | **14.5×** | 3.8× | 38× | 14× | 249 MB vs 2265 MB (9.1×) |

Both the 3× wall-time and 2× memory targets are met. RSS is measured externally (`/usr/bin/time -l`) for both sides; R has no usable in-process figure since `gc()` reports heap cells, not RSS.

**Two performance traps this codebase has already fallen into**, both recorded in §14.2:

1. `Design.mask()` once filtered the `bins` DataFrame instead of indexing a cached list. It runs once per locus per sample from both `count_sample` and `collect_results`, and cost 5.57 s of a 5.92 s fitting stage — it alone was responsible for sparse fitting being *slower than R*. `Design` now caches `masks` and `bin_lengths` as lists; **any new per-locus accessor must index, never scan.**
2. Profiling a component instead of a stage sent the first fix at the wrong target. The `scipy.optimize.minimize` overhead was real (the lean `solvers/_lbfgsb` driver is 2.8× faster per call and bit-identical) but moved end-to-end sparse time by only 12%.

**JIT: evaluated twice, declined twice, for different reasons** (plan §14.1 and §14.3). Now that the scaffolding is gone the fused closures are 61–68% of each fitting stage, and numba would give **2.1× Poisson / 2.2× NB / 1.9× legacy** at the realistic 329×3 locus — but **0.3× (three times slower) at 4000×30**, where a hand loop can't beat BLAS. Declined because it needs size dispatch plus a second bit-identical implementation of every likelihood, for ~1.2× end-to-end on a pipeline already 14.5× faster than R. The free part was taken: fused closures fold the usable mask into precomputed vectors and preallocate all temporaries (~1.18× on Poisson/NB stages, no regression at any size). Don't add a JIT kernel without re-reading §14.3.

## How the suite is structured

395 tests, 99% line coverage, 10 skips (all `absent_contig`, which R cannot produce a golden for). `uv run pytest -q --cov=dienr --cov-report=term-missing` for the breakdown.

The layers, roughly in order of how much they buy:

1. **Cross-implementation parity (106 tests).** Eight comparison dimensions × 13 scenarios against frozen R DENR goldens. An independent implementation as oracle is the strongest evidence available here.
2. **Meta-tests that the parity checks can fail (19).** `test_parity_harness.py` corrupts goldens deliberately — shifts bins by one, transposes the matrix, blanks the masks, splits equivalence classes — and asserts each comparison catches it. Without these a green parity suite could be vacuous.
3. **Fast-path ≡ reference-path equivalence.** The lean L-BFGS-B driver is bit-identical to `scipy.optimize.minimize`; each fused closure equals its readable `objective`/`gradient` pair; the single-model fast path equals the general optimizer. This is what makes aggressive optimization safe without risking parity.
4. **Analytic gradients vs finite differences**, for all three likelihoods × penalty combinations.
5. **Known-truth recovery**, plus "error must fall as depth rises".
6. **Every `FitStatus` reached deliberately**, asserting NaN-not-zero — §10 encoded as tests.
7. **Findings pinned as regressions.** Every bug and surprise in the log has a test: the `ALL_ZERO` short-circuit, mask anchoring order, the uncached `mask()`, NB's naive formula, the pinning collapse, the ridge scaling law, `legacy_log`'s low-count misspecification.
8. **Derived rather than magic constants** — e.g. shrinkage ≈ `2λa/n_bins` is asserted against its prediction, so the thresholds in neighbouring tests are interpretable.

**Known gaps, deliberately.** No property-based testing (all example-based); no mutation testing, so coverage ≠ assertion quality; no real biological data — everything is synthetic or the R package's synthetic fixtures; no concurrency tests since there is no parallelism yet. The parity goldens also come from newer Bioconductor than the container, so parity is against a near-neighbour of the frozen reference, not the frozen reference itself.

## Likelihood guidance (Gate B evidence)

`scripts/validate_likelihoods.py` compares the objectives on prespecified regimes with contiguous held-out blocks and paired per-fold scoring (plan §17, Gate B). Stable across seeds:

- **`transform="log"` — the reference's default — is badly misspecified once zeros are common.** At mean 0.5 counts/bin its held-out deviance is ~3.1 vs ~1.2 for everything else. A `y = 0` bin contributes `[log(1e-3) − log(μ+1e-3)]²` ≈ 38 at μ=0.5, so empty bins dominate and drag the fit toward zero. Prefer `transform="identity"` for sparse data.
- **Poisson's edge over `legacy_identity` is small, and there is an exact reason.** The two estimating equations differ only by a weight `1/μ_b`, so when `μ` is constant on a model's support they are *the same equation*. For a single 0/1 column they are literally the same estimator (measured difference `0.000e+00`). **`bin_operation` rounding to 0/1 is what destroys the advantage** — with overlapping indicator columns `μ` takes just two values across a locus. The ceiling on a graded design is `Var(MSE)/Var(Poisson) = ΣM³ΣM/(ΣM²)² ≤ ~1.12` even for a 50× occupancy range (~6% in SE). So: **implementing shape profiles is a prerequisite for Poisson to pay off on accuracy**; until then its justification is the exposure term (depth-independent abundance) and inference, not point estimates.
- **Never ship a fixed penalty weight.** Relative shrinkage follows **`2λa/n_bins`** (verified within 25%), so `ridge(0.01)` is negligible at abundance ~1 and catastrophic at ~200 — held-out deviance 68.7 vs 1.06, winning 1% of folds. `λ ∝ 1/a` cancels the dependence. `--lambda2` has no nonzero default on purpose.

## R parity workflow

Golden values are frozen R output committed under `tests/fixtures/parity/`. Inputs are **generated by us** from `tests/fixtures/scenarios.py` and never copied from the R package, because its `LICENSE.md` (MIT) and `DESCRIPTION` (GPL ≥2) disagree about redistribution. R program output is not a derivative work, so committing it is fine; committing their `inst/extdata` fixtures would not be.

```bash
uv run python -m tests.fixtures.generate --output build/parity-inputs
# then, inside the DENR container (the frozen reference environment):
singularity exec denr_latest.sif Rscript scripts/export_r_golden.R     --inputs build/parity-inputs --output tests/fixtures/parity     --commit 09d6cde5ef7ad1a6eb140c6e25dbae91e2a9a819
```

A local R needs GenomicRanges, GenomicFeatures, rtracklayer and AnnotationDbi, none of which are installed on this machine — hence the container. See `tests/fixtures/parity/README.md`.

`tests/test_parity_harness.py` exercises the comparison machinery against a golden generated from *our own* output. That validates the plumbing (column-major reshape, locus matching by geometry, 0-based mask indices) and proves the comparisons can actually fail; it says nothing about R agreement, and `test_synthetic_goldens_are_never_committed` stops such a fixture from ever masquerading as parity.

## Design decisions already made (from the plan)

These are deliberate departures from the R package; do not "restore" the R behavior without discussion.

- **Reimplementation, not a translation.** Compatibility with the *published method* is required; legacy output conventions and silent-failure behavior are not.
- **Identifiability is a first-class output.** Unique pre-RNA models, not annotation transcripts, are the primary estimands. Exactly duplicate design columns are collapsed before fitting, with a many-to-one transcript→model map preserved and `equivalence_class_size` reported. Never split an equivalence class's abundance across its transcripts in primary output.
- **Failures never become zeros.** Every sample×locus fit returns an explicit status (`ok`, `all_zero`, `insufficient_bins`, `rank_deficient`, `nonconverged`, `numerical_error`, …). Invalid loci get `NaN` plus a reason. Solver convergence info, iteration count, gradient norm, and objective are retained — the R version discards them.
- **Coordinates:** 0-based half-open internally and in BED-like output; GTF/GFF converted exactly once at the parser boundary; strand must be `+`/`-`; contig renaming explicit and recorded, never guessed.
- **Separation of stages:** `build-design` / `count` / `fit` are distinct CLI commands so annotation geometry is computed once and reused across samples and cohorts. Any change to annotation, bin size, masking, TSS evidence, or shape profile must yield a new design digest.
- **Multi-sample by construction:** one reusable `Design` against many `SampleCounts`; the R object holds one sample at a time.
- **Fitting and normalization are separate operations.** `abundance_raw` always survives. Do not call a trimmed mean of fitted abundances "TMM" (the R package does; the plan explicitly forbids it).
- **Likelihoods, staged:** legacy identity/log least-squares parity first, then nonnegative Poisson, then ridge/lasso/elastic-net/group penalties, then NB2 only if overdispersion validation justifies it. Poisson requires count-equivalent input — reject normalized/smoothed BigWigs for count likelihoods by default.
- **Tabular layer is Polars, not pandas.** DataFrames live only at the boundaries (annotation parsing, result writing); data stays in NumPy/SciPy through fitting. Polars writes the Parquet outputs directly with `write_parquet(metadata=...)` for `schema_version`/digests. Pandas enters only via the optional AnnData adapter.
- **All BigWig access goes through `pybigtools`** — it is faster than `polars-bio`, O(bins) in memory regardless of input density, and needed anyway for header seqinfo. `values(chrom, start, end, bins=n, summary=..., exact=True, fillna=0)` is the aggregation call, and **both non-default flags are mandatory**: without `exact=True` it reads lossy zoom levels (250bp bins on the double-strand fixture give `[62.53, 69.53, ...]` vs. a true `[63, 68, ...]`, no error raised, small windows can coincidentally agree); without `fillna=0` uncovered positions come back `NaN` and a leaked `NaN` becomes a spurious numerical failure rather than a legitimate zero. Set both in one wrapper and test at several bin widths.
- **No genomics interval library in core.** Locus grouping is a sort + cumulative-max sweep in plain Polars (5 lines), verified bit-identical to `polars-bio.merge` at 200k transcripts / 52k loci for `min_dist` 0/250/5000. GTF parsing is `scan_csv(separator="\t", comment_prefix="#")` + two `str.extract` calls — faster than `scan_gtf` (32ms vs 53ms/50k rows) and coordinate-identical. Masking is bin-index arithmetic, not interval joins. Reconsider `polars-bio` only for Phase 4 TSS-evidence overlap joins if `join_where` is too slow.
- **A second, independent binning implementation exists as the test oracle**: `pybigtools.records()` streamed into a Polars fan-out, exact by construction. It agreed with the primary path to `max|diff| = 0.0`. If you ever collect that fanned frame eagerly, use `collect(engine="streaming")` — dense per-base input amplified 8k bins into 2M rows and 384MB RSS eagerly, ~0MB streamed. Long RLE runs are *not* the hazard (~1 row per bin).
- **Single-model loci take a fast path**, because profiling showed ~91% of a median realistic locus (64 bins, 1 model) was L-BFGS-B scaffolding, not arithmetic. Identity has a closed form; log uses a bounded scalar search. Any change there must keep `test_single_model_fast_path_matches_general_optimizer` passing.
- **Numba was evaluated and rejected** — see plan §14.1. It cannot beat BLAS on large loci (1.09×) and the small-loci win it does offer (2.5–3.3×) addresses the 9% of runtime that is arithmetic, not the 91% that was optimizer overhead.
- **Count likelihoods need `summary="sum"`, but the parity default is `"mean"`.** `fit` raises rather than fitting a per-base density with a count likelihood — this is the most likely way to misuse the Poisson solver, since counts built with defaults look fine.
- **The μ floor (`_glm.MU_FLOOR`) cannot move an interior optimum but does bound a boundary gradient that is truly infinite.** Consequence: **an L1 penalty cannot drive a model to exactly zero without a background**, because no finite penalty outweighs a divergent likelihood. Set `Exposure(background=...)` when sparsity is the goal.
- **`inactive` models must be excluded from the support test, not just bounded.** Pinning a model leaves bins only it reached with no free model; treating them as usable drove μ→0 against nonzero counts and failed the whole locus. `usable_bins` takes `pinned` for this.
- **NB2 cannot change the point estimate on a binary design, at any dispersion.** The identity-link score weights residuals by `1/V(μ)`; constant `μ` on the support makes that constant, so MSE, Poisson and NB2 agree to `0.000e+00` even at φ=3. Its value is *uncertainty*: Poisson SEs understate the true spread by 23–64% for φ = 0.1–1.0. But **we emit no standard errors**, so NB is currently unrealized — and **quasi-Poisson (one scalar, `sqrt(Pearson χ²/df)`) matches NB2 to within 1–4%** without a second likelihood. See plan Gate D before investing further in NB.
- **NB2 is not an outlier-robust estimator.** Its score is the GLM form `Σ M(μ−y)/V(μ)` — the residual enters linearly and drives the fit, but the weight `1/V(μ)` depends only on the fitted mean, so influence is unbounded and one spike pulls NB about as hard as Poisson. Weighting *by* the residual is not available in ML: the weight is fixed by the variance function, and any `y`-dependent weight breaks `E[w(y)(y−μ)] = 0`. Bounded influence means M-estimation, a different estimator. Adopt NB for calibrated uncertainty, not point estimates.
- **NB2 with the identity link is not globally convex** — zero-count bins are always concave (`f''= −r/(μ+r)²`), so it does *not* inherit Poisson's guarantee. Both count solvers carry a mean-collapse guard (`MU_FLOOR_CHECK`); NB needs it more, and lacked it until a pathological start returned zeros as a converged fit.
- **NB dispersion is never fitted per locus** (plan §4.3). `estimate_dispersion` pools across loci from a completed Poisson fit; `fit(likelihood="negative_binomial")` demands an explicit value.
- **Masking has two forms.** `mask_tss`/`mask_pas` take **bases** along the direction of transcription and are preferred; `mask_start_bins`/`mask_end_bins` take bin counts and exist for R parity. Mutually exclusive. Bin counts make the masked span depend on `bin_size` — 4 bins is 400bp at 100 and 2000bp at 500.
- **The mask is a union over every transcript in a locus, and has to be.** Masking says "signal in this bin is untrustworthy", a property of the bin not of a model; an initiation spike contaminates the bin for every transcript through it. Cost is real and now reported: the reference's ±4 bins discards **22.7% of all bins** on the benchmark. See `Design.masked_fraction`, `heavily_masked_loci`, and `locus_qc.masked_bin_fraction`.
- **`locus_qc.total_observed` is fit-bin signal, not coverage.** Masked bins are zeroed in place, so `total_observed_all_bins` (recorded pre-mask in `SampleCounts.totals_all_bins`) is the one to use for coverage.
- **The reference's "counts" are mean per-base coverage, not counts.** R's `add_data` hardcodes `summary_operation <- "mean"`, so DENR abundances are polymerase *density*. `count_sample` therefore defaults to `summary="mean"`; a `sum` summary gave exactly `bin_size` times R's values. Poisson/NB will need `"sum"`, and `require_count_equivalent` rejects the `mean` + count-likelihood combination.
- **R aborts on an unknown contig; we do not.** `assert_chromosome_exists` kills an entire DENR run over one bad contig name. We record `missing_contig` for that locus and continue. The `absent_contig` scenario documents this and is excluded from parity coverage since R cannot produce a golden for it.
- **Two verified `pybigtools` *writer* quirks**, worked around in `tests/fixtures/scenarios.py`: a BigWig written with zero intervals is unreadable (`BBIReadError`), and contigs passed in the size map but carrying no interval are **dropped from the header**. Since the header is what separates `missing_contig` from `all_zero`, a fixture needs an explicit zero-valued placeholder base on every declared contig.
- **Non-goals for v1:** Gviz plotting, TxDb download/caching, the legacy TSS CNN, de novo transcript discovery.

Target module layout, phase breakdown, output schemas (`model_abundance.parquet`, `transcript_model_map.parquet`, `locus_qc.parquet`, `normalization_factors.parquet`), and decision gates are all specified in the plan — follow it rather than re-deriving.

## Licensing caveat

`dienr` is MIT (© 2026 Adam Youlin He). The R reference has **conflicting** metadata: `_reference/DENR/LICENSE.md` says MIT, `_reference/DENR/DESCRIPTION` says GPL (≥2). Until that is resolved with the upstream maintainers, implement from the paper, the README methodology section, and observed fixture behavior — do not copy or line-by-line adapt R source into this repository.
