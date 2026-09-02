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

Phase 1 and the start of Phase 2 exist: `annotation` (GTF/GFF → pre-RNA spans), `loci` (grouping), `design` (bins, masks, model reduction), `counts` (BigWig aggregation), `solvers/legacy` (identity/log least squares), `diagnostics`, `status`. 105 tests pass.

**Not yet written:** `results.py` and the Parquet writers, `cli.py`, Poisson/NB/penalized solvers, BAM counting, evidence mapping, and — most importantly — the Phase 0 frozen R fixtures. Nothing here has been checked for numerical parity against R yet; that is the Phase 1 exit gate and the tests currently assert the *specified* behavior, not agreement with the reference.

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
- **Non-goals for v1:** Gviz plotting, TxDb download/caching, the legacy TSS CNN, de novo transcript discovery.

Target module layout, phase breakdown, output schemas (`model_abundance.parquet`, `transcript_model_map.parquet`, `locus_qc.parquet`, `normalization_factors.parquet`), and decision gates are all specified in the plan — follow it rather than re-deriving.

## Licensing caveat

`dienr` is MIT (© 2026 Adam Youlin He). The R reference has **conflicting** metadata: `_reference/DENR/LICENSE.md` says MIT, `_reference/DENR/DESCRIPTION` says GPL (≥2). Until that is resolved with the upstream maintainers, implement from the paper, the README methodology section, and observed fixture behavior — do not copy or line-by-line adapt R source into this repository.
