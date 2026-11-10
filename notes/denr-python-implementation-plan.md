# Python Reimplementation of DENR: Implementation Plan

Status: legacy pipeline implemented with R parity; count likelihoods and run-on
tails implemented. This file is the binding design spec **and** the validation
record (sections 17-18); user documentation lives in `docs/`. See
`notes/README.md` for how the two relate.  
Package name: `dienr`  
Target: a scientifically faithful, multi-sample, failure-transparent implementation of DENR for Python and machine-learning workflows

## 1. Executive decision

Implement a Python successor to DENR, beginning with an exact compatibility mode and then adding count-likelihood and regularized solvers.

The project should preserve DENR's central scientific idea—deconvolving nascent-transcription signal into identifiable pre-RNA transcript models—while replacing the current single-sample R/S4 workflow with reusable design matrices, explicit schemas, multi-sample fitting, numerical diagnostics, and array-native APIs.

This should be a reimplementation of the method, not a line-by-line translation of the R package. Compatibility with the published method is required, but legacy output conventions and silent failure behavior are not.

## 2. Motivation

The current DENR implementation has several limitations that matter for repeated analyses and downstream ML:

- It fits loci sequentially with `stats::optim` and retains only the fitted parameters, discarding optimizer convergence information and messages.
- Its primary object holds one sample's counts and abundances at a time.
- It relies heavily on S4, `GRanges`, `rtracklayer`, and an R-to-Python TensorFlow/Keras bridge.
- It creates dense per-locus transcript-model matrices and performs several R-level loops.
- Indistinguishable transcripts appear as repeated transcript-level abundance rows, while identifiability is represented indirectly through a model key.
- Its `tmm` normalization option is a trimmed mean of positive fitted abundances, not conventional between-sample TMM normalization.
- The optional TSS classifier uses legacy Keras interfaces and should not be a required component of quantification.

Relevant implementation sources:

- [DENR fitting loop](https://github.com/CshlSiepelLab/DENR/blob/master/R/model_fitting.R#L214-L244)
- [BigWig handling](https://github.com/CshlSiepelLab/DENR/blob/master/R/data_handling.R#L100-L238)
- [Abundance-table construction](https://github.com/CshlSiepelLab/DENR/blob/master/R/abundance_tables.R#L35-L110)
- [Legacy TSS predictor](https://github.com/CshlSiepelLab/DENR/blob/master/R/tss_predictor.R#L440-L467)
- [Published DENR method](https://academic.oup.com/bioinformatics/article/37/24/4727/6348164)

## 3. Goals and non-goals

### Goals

1. Reproduce DENR's annotation reduction, masking, binning, and legacy fitting behavior.
2. Make identifiability explicit by treating unique pre-RNA models—not annotation transcripts—as the primary estimands.
3. Quantify many samples against one reusable annotation/design object.
4. Support raw stranded counts from BAM and count-equivalent stranded BigWigs.
5. Add Poisson and, if justified by validation, negative-binomial likelihoods.
6. Add regularization for near-collinear transcript models and joint multi-sample fitting.
7. Expose NumPy/SciPy arrays and optional PyTorch, JAX, and AnnData adapters.
8. Report every numerical failure and important locus-level limitation explicitly.
9. Provide stable, versioned Parquet/TSV outputs suitable for workflow systems and ML pipelines.
10. Improve runtime and memory use through reusable matrices, vectorized I/O, sparse representations, and parallel locus fitting.

### Non-goals for the initial release

- Reproducing DENR's plotting and Gviz functionality.
- Reimplementing TxDb download or annotation caching utilities.
- Porting the legacy bundled TSS CNN.
- Calling regulatory elements or discovering unannotated transcripts.
- Claiming transcript-level resolution when models are exactly indistinguishable.
- Replacing differential-expression packages. The tool will produce well-defined abundance and count-like matrices for downstream inference.

## 4. Scientific model

Let:

- \(y_{bs}\) be the observed signal in genomic bin \(b\) and sample \(s\);
- \(M_{bk}\) describe the contribution of identifiable pre-RNA model \(k\) to bin \(b\);
- \(a_{ks}\ge 0\) be the abundance of model \(k\) in sample \(s\);
- \(c_s\) be a known sample scaling factor, such as a spike-in or library-size exposure;
- \(e_{bs}\) be the usable exposure of the bin, including effective length or mappability;
- \(\beta_{bs}\) be an optional fixed or separately estimated background component.

The expected count is

\[
\mu_{bs}=c_s e_{bs}(Ma_s)_b+\beta_{bs}.
\]

Transcript models represent continuous pre-RNA transcription units from annotated TSS to PAS, not mature exonic structures.

### 4.1 Legacy compatibility objectives

Implement both published DENR loss functions:

\[
L_{\mathrm{identity}}(a)=\sum_b[y_b-(Ma)_b]^2
\]

and

\[
L_{\mathrm{log}}(a)=
\sum_b\left[\log(y_b+10^{-3})-\log((Ma)_b+10^{-3})\right]^2,
\qquad a\ge0.
\]

Compatibility mode must reproduce the R design matrix, masking, initialization, and objective closely enough for numerical parity. It must additionally retain convergence status, iterations, final gradient norm, objective value, and optimizer message.

### 4.2 Poisson objective

For raw bin counts, implement

\[
y_{bs}\sim\operatorname{Poisson}(\mu_{bs})
\]

with negative log-likelihood

\[
L_{\mathrm{Pois}}(a_s)=
\sum_b\left[\mu_{bs}-y_{bs}\log\mu_{bs}\right].
\]

Requirements:

- Use exact zero-count handling; do not introduce a statistically meaningful pseudocount.
- Use positive initialization or numerically stable `xlogy`-style evaluation.
- Exclude bins with no model exposure unless an explicit background model covers them.
- Permit fixed library-size, spike-in, effective-length, and mappability exposures.
- Treat normalized or smoothed BigWig values as incompatible with Poisson fitting unless the user explicitly selects a non-count likelihood.
- Preserve raw fitted polymerase density separately from any reported normalized value.

With fixed nonnegative design, exposure, and background, the Poisson objective with nonnegativity and L1/L2 penalties remains convex. Rank deficiency can still make individual parameters non-identifiable, so model collapsing and diagnostics remain necessary.

Three implementation findings from building this, all with tests:

- **An L1 penalty cannot drive a model to exactly zero without a background.**
  A bin with nonzero counts reachable only by model ``k`` makes the negative log
  likelihood diverge as ``a_k -> 0``, so no finite penalty outweighs it. This is
  correct Poisson behavior, not a solver limitation; sparsity requires a
  background term that keeps ``mu > 0`` without the model. The numerical mean
  floor bounds that otherwise-infinite boundary gradient, so the penalty weight
  at which L1 *appears* to win without a background is an artifact of the
  constant and must not be read as a modeling statement.
- **Pinning a model inactive requires excluding the bins only it reached.**
  Otherwise those bins carry ``mu = 0`` against nonzero counts and abort the
  locus, which made inactive-transcript filtering unusable with a count
  likelihood until support was judged on the free columns alone.
- **Nonnegativity makes lasso smooth.** Since ``a >= 0``, the L1 term is
  ``lambda_1 sum a_k``, which is linear and differentiable in the feasible
  region. Ridge, lasso and elastic net therefore all run through the same
  L-BFGS-B driver with no proximal operator.

### 4.3 Negative-binomial objective

Add an NB2 model only after the Poisson implementation is validated:

\[
\operatorname{Var}(y_{bs})=\mu_{bs}+\phi_s\mu_{bs}^2.
\]

Initial versions should estimate dispersion across many loci or borrow it across replicates. Do not estimate an unconstrained dispersion separately for every small locus. Compare Poisson, quasi-Poisson, and NB predictive performance before making NB the default.

Implemented as the two-stage procedure this describes: `estimate_dispersion`
pools the moment relation ``E[(y - mu)^2 - mu] = phi mu^2`` across every locus
of a completed Poisson fit and returns one shared value, which `fit_locus` then
holds fixed. It refuses to run without an explicit dispersion, so per-locus
estimation cannot happen by accident. `compare_dispersion_models` reports the
pooled NB2 dispersion beside the quasi-Poisson Pearson statistic, which is the
comparison Gate D needs.

Two findings that bear on Gate D, and both argue for keeping NB low priority:

- **NB2 is not an outlier-robust estimator, and should not be presented as one.**
  Its score is the standard GLM form ``sum_b M_bk (mu_b - y_b) / V(mu_b)`` with
  ``V(mu) = mu + phi mu^2``: the residual enters *linearly* and drives the fit,
  while the weight ``1 / V(mu_b)`` depends only on the fitted mean. So the
  residual is fully used, but an observation's influence is unbounded — triple
  the excursion, triple its contribution — and bins sharing a fitted mean share
  a weight regardless of how far each observation sits from it. A single large
  excursion therefore pulls an NB2 fit essentially as hard as it pulls a
  Poisson fit.

  Weighting *by* the residual is not available within maximum likelihood. The
  weight is fixed by the assumed variance function, so changing it stops being
  NB2; and any weight depending on ``y`` breaks the unbiasedness of the
  estimating equation, since ``E[w(y)(y - mu)] != 0`` in general. Bounding
  influence means moving to M-estimation with a consistency correction — a
  different estimator with different guarantees, not a variant of this one. If
  robustness to excursions is the actual requirement, that is the honest route,
  and it should be scoped separately rather than expected from NB.

  What NB2 does change is the variance function and therefore inference. Where
  fitted means differ across a locus it reallocates abundance materially; where
  they are flat, Poisson and NB2 agree to within a few percent.
- **NB2 with an identity link is not globally convex, so it does not inherit
  the guarantee section 4.2 gives Poisson.** The per-bin second derivative is
  ``f''(mu) = r(-mu^2 + 2 mu y + r y) / (mu^2 + r mu)^2``, positive only while
  ``mu < y + sqrt(y^2 + r y)``, and a zero-count bin is always concave at
  ``f''(mu) = -r/(mu + r)^2``. Poisson escapes this because ``r -> inf`` makes
  the condition vacuous. No local optima have appeared in testing from the
  moment initialization, but the guarantee is genuinely absent, and a
  pathological start did terminate with a collapsed mean reported as a
  converged fit of exactly zero until `fit_locus` gained the same
  mean-collapse guard the Poisson solver already had.
- **The textbook NB2 likelihood is numerically unusable at small dispersion.**
  ``(y + r) log(r + mu) - r log r`` cancels two large numbers once ``r`` is
  large, and the naive form fails to reproduce the Poisson limit at all. The
  implementation uses the algebraically identical
  ``r log1p(mu/r) + y log1p(r/mu)``, which is stable and does reproduce it.

### 4.4 Regularization

Support penalties as explicit, named modeling choices:

- Ridge: \(\lambda_2\lVert a\rVert_2^2\), for stable allocation among near-collinear models.
- Lasso: \(\lambda_1\lVert a\rVert_1\), for sparse selection, with a warning that correlated models may be selected arbitrarily.
- Elastic net: \(\lambda_1\lVert a\rVert_1+\lambda_2\lVert a\rVert_2^2\).
- Multi-sample group penalty: \(\lambda_g\sum_k\lVert A_{k,\cdot}\rVert_2\), to select models consistently across samples while permitting sample-specific abundance.
- Optional fused penalties for ordered time courses or related conditions, deferred until after group-penalty validation.

The recommended first regularized model is nonnegative Poisson plus a small ridge penalty. Elastic net and multi-sample group penalties should follow.

Penalties must never be used to manufacture resolution between exactly identical models. Exact duplicate columns must be collapsed before fitting. Regularized allocations among merely near-identical columns must be labeled as model-dependent estimates.

## 5. Identifiability policy

Identifiability is a first-class output, not a hidden implementation detail.

1. Apply bin masks and configured model transformations.
2. Collapse exactly identical design columns into one identifiable model.
3. Preserve a many-to-one transcript-to-model mapping.
4. Compute matrix rank, smallest relevant singular value, and condition number per locus where feasible.
5. Flag near-collinear models using a configurable numerical threshold.
6. Estimate abundance primarily at the identifiable-model level.
7. Never split an equivalence-class abundance across its transcripts in the primary output.

An optional compatibility export may repeat the model abundance for every transcript, matching DENR's current convention, but the repeated values must carry `equivalence_class_size` and `is_identifiable = false` fields.

## 6. Coordinate and annotation conventions

Use one explicit internal coordinate convention:

- 0-based, half-open intervals internally and in BED-like outputs;
- GTF/GFF input converted exactly once at the parser boundary;
- original source coordinates retained in provenance metadata;
- strand required to be `+` or `-` for quantification;
- partial terminal bins represented with their true effective lengths;
- contig-name transformations explicit and recorded, never guessed silently.

Required boundary tests include one-base transcripts, terminal partial bins, transcripts touching but not overlapping, minus-strand TSS/PAS orientation, chromosome ends, missing contigs, and mixed `chr1`/`1` naming.

### 6.1 Masking is specified in bases, not bin counts

The reference specifies masking as a number of *bins* at each end of a
transcript, which makes the masked genomic span depend on `bin_size` and jitter
by up to a bin depending on where the TSS falls inside one. `mask_start_bins`
of 4 means 400 bp at `bin_size` 100 and 2000 bp at 500 — the same configuration
describing different biology.

`mask_tss` and `mask_pas` take `(upstream, downstream)` **bases** along the
direction of transcription, and mask any bin overlapping that genomic window.
The span is then reproducible across bin sizes, which is what section 6 asks
for everywhere else. The bin-count form is retained for R parity and the two
are mutually exclusive; mixing them raises.

Two properties to expect rather than treat as defects:

- **Windows are clipped to the locus,** because bins exist only inside it. An
  isolated transcript's locus begins at its own TSS, so an upstream window has
  nothing to act on; it bites only where another transcript extends the locus
  further back. No data is lost, since there are no bins outside the locus, but
  masked spans are not comparable between isolated and crowded loci.
- **The mask is a union across every transcript in the locus, and must be.**
  Masking encodes "the observed signal in this bin is untrustworthy", which is a
  property of the bin and not of a model: an initiation spike from one TSS
  contaminates that bin for every transcript running through it. Masking per
  model while leaving the shared count vector intact would ask the surviving
  models to explain signal they did not generate.

The union is correct but not free, and its cost was previously invisible. On the
benchmark annotation:

| masking | bins masked | loci losing >50% |
|---|---|---|
| legacy, ±4 bins | 22.7% | 0 |
| `mask_tss/pas = (1000, 1000)` | 24.9% | 1 |
| `mask_tss/pas = (500, 500)` | 15.8% | 0 |
| `mask_tss = (0, 1000)` only | 8.4% | 0 |

So the reference's recommended masking discards roughly a fifth of all bins.
That is a defensible trade against noisy initiation and termination, but it
should be a visible choice. `Design.masked_fraction` and
`Design.heavily_masked_loci` report it, `dienr build-design` warns when any
locus loses more than half its bins, and `locus_qc` now carries
`masked_bin_fraction` and `total_observed_all_bins` alongside `total_observed`
— which covers only the bins the fit used, and would otherwise make a heavily
masked locus look quiet rather than mostly discarded.

The way to reduce the cost is a narrower window, or excluding transcripts known
to be inactive so their TSS regions are not masked at all. The latter needs
activity estimates that do not exist at design time, so it implies a two-pass
fit and is deferred past Phase 4.

## 7. Proposed software architecture

```text
dienr/
  annotation.py       GTF/GFF parsing and pre-RNA span construction
  loci.py             strand-specific locus grouping
  design.py           bins, masks, model reduction, stable IDs
  counts.py           BAM/BigWig aggregation and exposure tracking
  evidence.py         external TSS/TRE evidence mapping
  solvers/
    legacy.py         DENR identity/log least-squares parity
    poisson.py        nonnegative Poisson fitting
    negative_binomial.py
    penalties.py      ridge, lasso, elastic net, group penalties
  diagnostics.py      rank, conditioning, residual and convergence QC
  results.py          result objects and schema-validated writers
  adapters.py         NumPy, SciPy, PyTorch, JAX and AnnData adapters
  cli.py              workflow-oriented command-line interface
```

### Core objects

- `Annotation`: validated transcript and gene records plus annotation digest.
- `Design`: immutable bins, block-diagonal sparse design matrix, locus offsets, masks, model map, and configuration digest.
- `SampleCounts`: sample-by-bin count matrix, exposures, input provenance, and missing-data mask.
- `FitConfig`: likelihood, penalty, normalization, tolerances, initialization, and parallelism.
- `FitResult`: model abundances, fitted signals, solver diagnostics, locus QC, and provenance.

The design must be serializable and reusable across samples. A change to annotation, bin size, masking, external TSS evidence, or shape profile must produce a new design digest.

### Recommended dependencies

- NumPy and SciPy for arrays, sparse matrices, numerical optimization, and special functions.
- **Polars** as the tabular layer and as the annotation parser, interval-grouping engine, and Parquet writer. See "Tabular layer" and "Why not a genomics interval library" below.
- **`figwig`** for all BigWig access: header seqinfo, exact per-base reads, and writing. (Replaced `pybigtools`; see "BigWig aggregation".)
- `pysam` for BAM/CRAM input.
- Pydantic or equivalent schema validation for configuration and result metadata.

PyTorch, JAX, AnnData, pandas, `polars-bio`, plotting, and negative-binomial support should be optional extras rather than core dependencies.

That is the whole core list. No genomics-specific interval or annotation library is required; the reasoning is below, because the obvious candidates were evaluated and rejected on evidence rather than overlooked.

#### Tabular layer

Choose **Polars** over pandas, and treat PyArrow as an implementation detail rather than the primary interface.

Rationale specific to this project:

- Section 14 already requires that data stay in NumPy/SciPy form through fitting. DataFrames therefore live only at the boundaries — annotation parsing and result writing — so pandas' ecosystem familiarity buys little, and there is no long user-facing analytical pipeline to hook into.
- Polars is Arrow-native, so the Parquet outputs in section 9 are written with no pandas-to-Arrow conversion layer. `DataFrame.write_parquet(..., metadata=...)` carries the required `schema_version`, software version, and digest fields as Parquet key-value metadata, so a separate PyArrow writer is not needed.
- Polars has no index concept and requires explicit dtypes, which matches the strict, versioned schema discipline this plan demands. Pandas' index semantics are a recurring source of silent row-alignment bugs in exactly the join-heavy transcript-to-model mapping code this project needs.
- Lazy scanning with predicate and projection pushdown matters at real annotation scale: a full GENCODE GTF need never be materialized to extract six required fields.

Pandas remains an optional extra, entering only through the AnnData adapter, whose `.obs`/`.var` are pandas DataFrames. Do not let it become a core dependency; note that PyRanges would have pulled it in, which is a further reason to prefer `polars-bio`.

#### Why not a genomics interval library

The earlier draft named PyRanges/NCLS as a placeholder, and `polars-bio` looked like a strong replacement: it offers `scan_gtf`/`scan_gff`, `merge`/`cluster`/`overlap`/`nearest`, and object-storage reads. On measurement, every capability this project actually needs from it is a short expression in plain Polars, matching bit-for-bit and usually faster. It is therefore demoted to an optional extra.

| Need | plain Polars | `polars-bio` | Result |
|---|---|---|---|
| GTF parse, 50k transcript rows | `scan_csv(separator="\t", comment_prefix="#")` + two `str.extract` calls, 32 ms | `scan_gtf(attr_fields=[...])`, 53 ms | Identical coordinates; plain is faster |
| Locus grouping, 200k transcripts to 52k loci | cumulative-max sweep, 5 lines, 13–18 ms | `merge(min_dist=...)`, 8–10 ms | **Bit-identical at `min_dist` 0, 250, 5000**; both negligible |
| BigWig interval streaming | not supported | `scan_bigwig` | Not needed: `figwig` reads exact per-base values, and binning is NumPy |

The locus-grouping idiom is worth stating, since it is the one genuinely graph-flavored operation in the pipeline and it needs no interval tree — DENR's `group_transcripts(distance = d)` is a sort plus a cumulative maximum:

```python
(
    tx.sort("chrom", "start")
    .with_columns(pm=pl.col("end").cum_max().shift(1).over("chrom"))
    .with_columns(new=pl.col("pm").is_null() | (pl.col("start") > pl.col("pm") + d))
    .with_columns(locus=pl.col("new").cum_sum().over("chrom"))
)
```

Run per strand, this is the strand-specific connected-component grouping of section 7. Masking is bin-index arithmetic, not interval joins, and the BigWig binning in "BigWig aggregation" below uses no interval-library operation at all.

One claim made for `polars-bio` in an earlier draft does not survive checking, and is worth recording so it is not repeated: its coordinate handling is **not** strict by default. `scan_gtf` with no `use_zero_based` argument silently returns 1-based GTF coordinates unchanged; only `use_zero_based=True` converts to the 0-based half-open convention this plan mandates. It does warn when a bare DataFrame with no coordinate metadata is passed to an interval operation, but the reader default is passthrough, not an error. Owning the single `start - 1` conversion in our own parser is both more explicit and easier to test than delegating it — and section 6 already requires that the conversion happen exactly once at the parser boundary.

Reconsider `polars-bio` at one specific point: Phase 4 TSS-evidence mapping needs a strand-aware windowed overlap join, and if Polars `join_where` proves too slow on genome-scale evidence sets, `overlap`/`nearest` are the natural remedy. Treat that as a localized, benchmarked decision. Against adoption, note that it is pre-1.0, is a compiled Rust plugin requiring `polars>=1.37.1` — so a Polars upgrade forces revalidation — pulls in a DataFusion runtime, has a `register_bigwig` whose argument order is `(path, name)` rather than the `(name, path)` its siblings suggest, and aborted with a raw Rust `PanicException` instead of a Python exception on one path form during evaluation.

#### BigWig aggregation

Use `figwig` for all BigWig access, through the single wrapper `dienr.counts.read_binned`. figwig reads exact per-base float32 values into NumPy and has **no zoom-level path at all**, so the lossy-summary hazard below cannot arise; binning is a reshape (or `reduceat` for a window that is not a whole number of bins) in our own code, where every semantic choice is visible and tested. It depends only on NumPy and numba.

```python
reader.read([chrom], [start], end - start, n_jobs=1, missing=np.nan, oob=np.nan)
```

`missing=nan` keeps an uncovered base distinguishable from an interval whose value is `0`, which matters because `mean`/`min`/`max`/`std` are taken over **covered bases only**, a zero-valued interval counting as covered; an empty bin summarizes to `0` (count semantics — an absent interval is zero polymerase, not a missing observation). `oob=nan` makes bases past a contig end uncovered. `std` is the sample standard deviation, `0` below two bases. Bin `i` of `[start, end)` is `[start + ⌊i·w/n⌋, start + ⌊(i+1)·w/n⌋)`. figwig only *warns* on a contig absent from the header and returns zeros, so the wrapper checks `chrom_sizes` and raises `KeyError` itself: that conflation is exactly `MISSING_CONTIG` vs `ALL_ZERO`.

These are the semantics the package had under `pybigtools` (`values(..., exact=True, fillna=0, oob=0)`), measured base by base before the switch. On 5,000 random windows (sparse, signed, zero-valued intervals; divisible, non-divisible, clipped and past-end) the figwig path agrees exactly with an independent per-base oracle built from `records()`, and with pybigtools on every whole-bin window. R parity was unchanged and abundances on the dense benchmark were identical.

**figwig is only fast when batched.** One call per locus left its thread pool idle and was *slower* than pybigtools (1.13 s vs 1.01 s, three dense samples): its speed comes from many equal-width windows in one multithreaded call. `count_sample` therefore goes through `read_loci_binned`, which cuts every in-bounds locus window into tiles of `TILE_BINS = 16` bins (a whole number of bins, so no bin straddles two tiles), reads them in calls capped at `MAX_CHUNK_BYTES = 64 MiB` of per-base output, and summarizes every bin of a chunk in one vectorized pass, split across `n_jobs` threads (NumPy releases the GIL there; single-threaded, binning cost as much as the read). A window overrunning the contig end keeps the one-window `read_binned` path, clipped as before. Measured on the dense benchmark:

| | per sample, steady state | three samples, incl. first-call warm-up | peak RSS |
|---|---|---|---|
| pybigtools, one binned call per locus | ~330 ms | 1.01 s | 430 MB |
| figwig, one call per locus | ~390 ms | 1.13 s | 454 MB |
| figwig, batched tiles, 8 threads | **~145 ms** | **0.68 s** | 566 MB |

The first call costs ~240 ms extra (numba kernels loading from cache), so the gap widens with more samples. Smaller caps were slower (16 MiB: 0.98 s, 32 MiB: 0.90 s) for ~80 MB less RSS; 16 threads did not beat 8. Equality with one-window reads is asserted for every summary, with the cap forced down to one window per call. The `isal` extra cuts figwig's inflate time a further ~40%.

##### History: why not pybigtools

`pybigtools` carried this path until it was replaced. It needed **two non-default arguments, each silently wrong if omitted**: without `exact=True`, `values(..., bins=N)` read precomputed zoom levels — on `test_double_strand_plus.bw`, 250 bp bins over `1:0-2000` have a true sum of `[63, 68, 72, 77]` and the default returned `[62.53, 69.53, 71.47, 76.76]`, no error, and small windows could coincidentally agree; without `fillna=0`, uncovered bins leaked `NaN`. The switch also uncovered three behaviors of its binned `values()` that the new path deliberately does **not** reproduce:

- **A bin straddling the contig end is zeroed**, discarding its in-bounds signal: an all-ones contig of length 100 read as `(0, 107, bins=4)` gives `[26, 27, 27, 0]`, truth `[26, 27, 27, 20]`. Reachable only from `shape.py`, for a gene whose rounded-up last bin overruns the contig.
- **Covered-base counts use floating-point bin edges** in non-divisible windows, so an interval clipped at a fractional edge loses a base from the `mean` denominator (26 counted, 27 true). Reachable from `count_sample`, which clips a locus window at the contig end.
- **`std` uses a one-pass formula**: `9e-8` for a constant bin whose true value is `0`.

Two *writer* quirks went with it: a file with zero intervals was unreadable, and contigs without an interval were dropped from the header, which forced zero-valued placeholder bases into every fixture. figwig keeps every declared contig and writes empty files; fixtures write with `missing=nan` so explicit zeros remain stored as data.

##### Parity oracle: independent binning in Polars

This oracle was built when `pybigtools` held the primary path, from its `records()` stream; the figwig switch was validated against a per-base oracle from the same stream. Stream intervals with `records(chrom, start, end)`, clip them to the window, fan each across the bins it touches, and sum `value * overlap` per bin. This is exact by construction with no opt-in flag, and it makes the section 6 requirement that partial terminal bins carry true effective lengths an explicit column. Both implementations agreed to `max|diff| = 0.0` on the RLE and dense fixtures, which makes this a genuinely informative cross-check rather than a restatement of the same code path.

Two facts about that route, recorded because they bear on any future decision to promote it:

| Input | Intervals in | Bins out | Rows after fan-out | Peak RSS, eager | Peak RSS, streaming |
|---|---|---|---|---|---|
| RLE, 5 kb runs over 10 Mb | 2,000 | 40,000 | 40,000 (1.0×) | 44 MB | — |
| Dense per-base over 2 Mb | 2,000,000 | 8,000 | 2,000,000 (250×) | 384 MB | ~0 MB |

First, the fan-out hazard is the opposite of what run-length encoding suggests. Long constant-value runs are harmless — BigWig intervals are non-overlapping, so interval-by-bin pairs are bounded by roughly bins plus intervals, and the RLE case produced exactly one row per bin. The amplification comes from **dense per-base signal, where intervals greatly outnumber bins**, which is exactly what a PRO-seq 3′-end BigWig is. Second, that amplification is fully mitigated by never materializing the fanned frame: aggregating lazily with `.collect(engine="streaming")` gave bit-identical results with no measurable RSS growth, and faster than eager collection (70 ms versus 200 ms). Even so, the library reader did the same 2 Mb aggregation in 51 ms with bounded memory and no discipline required from the caller, which is why it holds the primary path.

Still to settle: throughput on a real whole-genome sample, where the dense case extrapolates to order 10⁹ intervals per sample. (Batching loci into memory-capped multithreaded figwig reads is settled — see "BigWig aggregation".)

## 8. Input contracts

### Annotation

Required fields:

- `transcript_id`
- `gene_id` when available
- `contig`
- `start`
- `end`
- `strand`

For GTF/GFF input, derive each pre-RNA model from the minimum and maximum coordinates of all records belonging to a transcript. Record whether coordinates came from explicit transcript features or were reconstructed from exons.

### Signal

Preferred Poisson input:

- stranded BAM/CRAM containing one countable polymerase 3′ end per aligned read; or
- stranded BigWigs whose values are unnormalized count-equivalent signal.

Legacy/non-count input:

- stranded BigWigs containing mean coverage, scaled signal, or other continuous values.

Every input must record whether values are raw, spike-in scaled, library normalized, or otherwise transformed. The CLI must reject incompatible likelihood/input combinations by default.

**The reference's "counts" are not counts.** `add_data` hardcodes
`summary_operation <- "mean"`, so each bin holds *mean per-base coverage*, and
DENR abundances are in units of mean polymerase density. Confirmed against the R
export: a `sum` summary gave exactly `bin_size` times R's values. Two
consequences. Legacy-compatible extraction must use `mean`, which is why
`count_sample` defaults to it. And a Poisson or negative-binomial fit must use
`sum`: feeding it a per-base mean would hand a count likelihood a density
divided by bin width. `SignalFile.require_count_equivalent` now rejects that
combination alongside a transformed `value_kind`.

BigWig uncovered-region semantics must be pinned explicitly rather than inherited from library defaults. In nascent-RNA BigWigs an absent interval means zero observed polymerase, not a missing observation, and whether uncovered bases enter the denominator of a `mean` summary silently changes the fitted value. `read_binned` reads with `missing=nan, oob=nan` so uncovered bases are identifiable, excludes them from every summary, and returns `0` for an empty bin. Pin these at the single I/O boundary, record them in provenance, and cover them with fixtures — a `NaN` leaking into a bin vector would otherwise propagate into the objective as a numerical failure rather than a legitimate zero.

### TSS activity evidence

Accept optional BED-like evidence from dREG, PINTS, PRO-cap, GRO-cap, CAGE, or user-curated annotations. Map evidence to annotated TSSs using an explicit strand-aware window. Initial behavior should support:

- no filtering;
- hard active/inactive transcript masks;
- soft prior weights, deferred until regularized solvers are available.

Do not embed a peak caller in the initial package.

## 9. Output schemas

All outputs must carry `schema_version`, software version, annotation digest, design digest, and configuration digest.

### `model_abundance.parquet`

One row per sample and identifiable model:

- `sample_id`
- `locus_id`
- `model_id`
- `abundance_raw`
- `abundance_normalized`, nullable
- `normalization_method`, nullable
- `equivalence_class_size`
- `solver_status`
- `objective`
- `penalty_value`
- `iterations`
- `converged`

### `transcript_model_map.parquet`

- `transcript_id`
- `gene_id`
- `locus_id`
- `model_id`
- `contig`
- `start0`
- `end0`
- `strand`
- `equivalence_class_size`
- `exactly_identifiable`

### `locus_qc.parquet`

- `sample_id`
- `locus_id`
- `n_bins`
- `n_models_before_collapse`
- `n_models_after_collapse`
- `matrix_rank`
- `condition_number`
- `total_observed`
- `total_fitted`
- `deviance` or residual sum of squares
- `gradient_norm`
- `solver_status`
- `failure_reason`, nullable

### `normalization_factors.parquet`

- `sample_id`
- `method`
- `factor`
- `source`
- `is_external`

Raw abundance must always remain available. Conventional between-sample normalization should be performed separately from locus fitting, and terms such as TMM should only be used for their standard definitions.

## 10. Failure-handling policy

Never convert a failed fit to an apparent biological zero.

Each sample/locus fit must return one of a documented set of statuses, including:

- `ok`
- `all_zero`
- `insufficient_bins`
- `missing_contig`
- `invalid_exposure`
- `rank_deficient`
- `ill_conditioned`
- `nonconverged`
- `invalid_mean`
- `numerical_error`

Policy:

- A valid all-zero locus may return zero abundance with `all_zero` status.
- A failed or invalid locus returns `NaN` abundance and a reason, never zero.
- Rank deficiency after exact-column collapsing is reported even if a regularized estimate is produced.
- No automatic solver fallback may occur silently. If enabled, both attempts and the selected result must be recorded.
- A run summary must report counts of every status and exit nonzero when a configurable failure threshold is exceeded.

## 11. Public interfaces

### Python API

```python
import dienr

design = dienr.build_design(
    annotation="gencode.gtf",
    bin_size=250,
    mask_tss=(-1000, 1000),
    mask_pas=(-1000, 1000),
)

counts = dienr.count_samples(
    samples="samples.csv",
    design=design,
    signal="three_prime_ends",
)

result = dienr.fit(
    design,
    counts,
    likelihood="poisson",
    penalty={"type": "ridge", "lambda": 0.01},
)

result.write("results/")
X, feature_ids = result.to_numpy(value="abundance_raw")
```

### CLI

```bash
dienr build-design --gtf gencode.gtf --output design/
dienr count --design design/ --samples samples.csv --output counts/
dienr fit --design design/ --counts counts/ --model poisson-ridge --output results/
dienr validate --results results/
```

Build, count, and fit must be separate commands so static annotation work is reused across cohorts and workflow engines can cache each stage.

## 12. Implementation phases

### Phase 0: Specification and frozen R reference

Estimated effort: 3–5 working days.

Deliverables:

- Freeze the exact DENR commit and container used as the behavioral reference.
  **Decided:** commit `09d6cde` on `master`, and the container the DENR README
  publishes (`library://rhassett-cshl/stadyum/denr:latest`). A local R needs the
  Bioconductor stack, which is why the container is the reference environment.
- **Decided: how parity fixtures live.** Inputs are generated by us from
  `tests/fixtures/scenarios.py`, not copied from DENR's `inst/extdata`, which
  keeps the unresolved license question out of this repository entirely. R
  *program output* is committed as JSON under `tests/fixtures/parity/`. We do
  **not** submodule or clone the R repo at test time: that would make the suite
  network-dependent and still would not yield R-computed values without running
  R. `scripts/export_r_golden.R` performs the export; `tests/test_parity.py`
  consumes it and skips with an actionable message until it exists.
- Run the included synthetic datasets and export intermediate R objects: loci, bins, masks, full and reduced model matrices, count vectors, model map, fitted abundance, objectives, and predictions.
- Write the coordinate, masking, normalization, and model-ID specification.
- Establish benchmark datasets and capture R wall time and peak memory.
- Resolve the repository's conflicting license metadata before deriving or redistributing source code: `LICENSE.md` says MIT, while `DESCRIPTION` says GPL ≥2.

Exit gate:

- Every legacy behavior needed for compatibility is represented by a fixture or written specification.

### Phase 1: Annotation geometry and design construction

Estimated effort: 1–2 weeks.

Deliverables:

- GTF/GFF parser and continuous pre-RNA span construction.
- Strand-specific connected-component locus grouping.
- Exact bin generation, partial-bin exposure, masks, and transcript-model matrices.
- Exact-equivalence collapse and stable content-derived model identifiers.
- Serialized reusable `Design` object.
- Exhaustive coordinate and strand tests.

Exit gate:

- Python loci, masks, model matrices, and equivalence maps match the frozen R fixtures after coordinate normalization.

### Phase 2: Signal extraction and legacy solver

Estimated effort: 1–2 weeks.

Deliverables:

- Exact targeted BigWig aggregation.
- BAM/CRAM 3′-end counting with strand validation.
- Legacy identity and log least-squares solvers with R-compatible initialization.
- Per-locus convergence and conditioning diagnostics.
- Versioned primary outputs and optional legacy-format export.

Exit gate:

- On identifiable fixtures, fitted abundance and predicted signal match R within predefined numerical tolerance.
- On rank-deficient fixtures, predicted signal and objective match even when parameter allocations are non-unique.
- No test fixture can fail silently.

### Phase 3: Poisson likelihood

Estimated effort: 1–2 weeks.

Deliverables:

- Nonnegative Poisson solver with analytic gradients and stable zero handling.
- Exposure, spike-in factor, and optional background support.
- Poisson deviance and likelihood-based QC.
- Simulation comparison against legacy objectives.

Exit gate:

- Solver recovers known abundances in Poisson simulations.
- It improves held-out deviance or abundance error over legacy fitting in prespecified low-count and heterogeneous-depth regimes.
- It does not regress materially on high-count simulations.

### Phase 4: Regularized and multi-sample fitting

Estimated effort: 2–3 weeks.

Deliverables:

- Ridge, lasso, and elastic-net penalties.
- Joint multi-sample group penalty.
- Hyperparameter-selection workflow based on simulations, replicate prediction, or contiguous held-out regions—not random individual bins.
- Regularization path and stability diagnostics.
- Optional soft TSS evidence as prior or penalty weights.

Exit gate:

- Regularized estimates reduce replicate-to-replicate instability for near-collinear models without worsening locus-level prediction beyond a prespecified tolerance.
- Exactly indistinguishable transcripts remain collapsed.
- All regularized outputs are labeled with the penalty type and value.

### Phase 5: Overdispersion, performance, and release hardening

Estimated effort: 2–3 weeks.

Deliverables:

- Poisson goodness-of-fit assessment across representative datasets.
- NB2 or quasi-Poisson support only if justified.
- Parallel fitting, chunked I/O, sparse/block-matrix optimizations, and design caching.
- Python and CLI documentation, container, PyPI package, and Bioconda recipe.
- ML adapters and end-to-end workflow examples.
- Reproducible performance and scientific validation report.

Exit gate:

- Release candidate meets the definition of done below.

Expected total effort is approximately 8–12 engineer-weeks for one experienced scientific Python developer, excluding extended biological benchmarking. A legacy-parity MVP should be possible in roughly 3–5 weeks.

## 13. Testing and validation strategy

### Unit and property tests

- Coordinate conversions and terminal bins.
- Plus/minus-strand symmetry.
- Mask orientation around TSS and PAS.
- Exact and near-duplicate transcript models.
- All-zero and single-bin loci.
- Missing chromosomes and out-of-bound annotation.
- Poisson gradients checked against finite differences.
- Penalty gradients/proximal operators checked independently.
- Serialization round trips and schema compatibility.

### R parity tests

Use the DENR package's included synthetic BigWigs and TxDb fixtures. Compare:

- locus membership;
- bin boundaries;
- mask indices;
- full and reduced design matrices;
- transcript-to-model mapping;
- binned signal;
- fitted predictions and objective values;
- abundance when the solution is unique.

### Simulation benchmarks

Use `nascentRNASim` or equivalent controlled simulations spanning:

- sequencing depth;
- number of isoforms per locus;
- TSS/PAS separation;
- abundance imbalance;
- pausing and shape-profile misspecification;
- overdispersion;
- annotation errors;
- inactive TSS contamination;
- exact and near-collinearity.

Primary metrics:

- model-level RMSE and rank correlation;
- dominant-model accuracy;
- locus-signal prediction error;
- held-out Poisson/NB deviance;
- replicate stability;
- calibration or coverage of uncertainty intervals where available;
- failure-detection sensitivity.

### Real-data validation

- Reproduce key DENR analyses on K562 and immune-cell data from the publication.
- Compare dominant pre-RNA models and cell-type changes.
- Assess agreement with PRO-cap/GRO-cap/CAGE-supported TSS usage.
- Separate differences caused by implementation from those caused by a new likelihood or penalty.

## 14. Performance strategy

Do not assume Python alone is faster. Profile each stage and optimize the measured bottlenecks.

Planned improvements:

- Build locus geometry and design matrices once per annotation/configuration.
- Store matrices in sparse or compact block form where this reduces memory.
- Aggregate only requested BigWig intervals; never materialize whole-genome RLE coverage.
- Count BAM 3′ ends in genomic chunks and reuse sorted locus/bin indexes.
- Parallelize independent loci with controlled worker and BLAS thread counts.
- Batch loci of similar dimensions when a solver benefits from vectorization.
- Keep data in NumPy/SciPy form through fitting; avoid repeated DataFrame conversion.
- Profile before considering Numba, Cython, Rust, or GPU kernels.

### 14.1 Profiling results and the Numba question

Numba was evaluated against the initial implementation. The finding is that it
would optimize the wrong 9%.

Profiling the full pipeline put 88% of runtime in the solver and 12% in design
construction, with `objective` and `gradient` together accounting for about 70%
of total time. Two algorithmic fixes came first and cost nothing:

1. **Fuse the objective and gradient.** The optimizer evaluated them
   separately, computing `M @ a` twice per iteration.
2. **Hoist `log(observed + eps)` out of the loop.** It does not depend on the
   abundance, yet was recomputed on every call, at roughly a quarter of the
   cost of each one.

Together these cut the fit stage by 25%. Only then does the Numba comparison
become meaningful, and locus size decides it:

| Locus shape | Fused NumPy | Numba | Speedup |
|---|---|---|---|
| 10141 x 128 | 92.6 us | 84.7 us | 1.09x |
| 2000 x 20 | 18.5 us | 15.2 us | 1.22x |
| 200 x 4 | 3.1 us | 1.3 us | 2.48x |
| 40 x 3 | 2.4 us | 0.7 us | 3.28x |

Numba loses its advantage exactly where the matrices are large, because that
regime is BLAS-bound and Numba cannot beat OpenBLAS on a matrix-vector product.
It wins on small loci, where NumPy's per-call overhead and temporaries dominate
— and small loci are the realistic case. On a GENCODE-like transcript density
the median locus has **64 bins and 1 model**, with 100% of matrix-vector work in
loci under 500 bins.

That size distribution, though, also revealed the real bottleneck, which Numba
does not address at all. On a median 64-bin single-model locus, `fit_locus`
took 338 us of which only 30 us was objective evaluation: **91% was L-BFGS-B
scaffolding.** The remedy is to stop calling a general optimizer for problems
that do not need one. Single-model loci have a closed-form nonnegative
least-squares solution under the identity transform, and a cheap bounded scalar
search under the log transform; that path measured 1.07 us against 80.8 us and
agreed exactly. Adding it halved fit time on realistic density (6.03 s to
3.00 s over 18,756 loci).

**Decision: no Numba for now.** It targets the arithmetic, which is 9% of the
realistic hot locus, and costs a 38 MB `llvmlite` dependency. Revisit only after
the remaining scaffolding work is done, and only with a fresh profile:

- extend the batching idea in section 14 to small multi-model loci, whose
  per-call overhead is the same problem the single-model path just solved;
- consider calling L-BFGS-B below `scipy.optimize.minimize`, bypassing the
  `_differentiable_functions` wrappers that dominate small fits;
- drop rows where no model has support, which shrinks every matrix-vector
  product. Note these rows contribute a constant to the objective and zero to
  the gradient, so dropping them changes the reported objective value but not
  the argmin — the constant must be added back for R parity.

### 14.3 The JIT question, revisited after the scaffolding was removed

The section 14.1 verdict was correct for the code it measured and is now
obsolete, because the thing it deferred to has happened: with the lean L-BFGS-B
driver in place and the `Design.mask()` scan gone, the fused
objective-and-gradient closure is the dominant cost in every solver — 63% of the
legacy fitting stage, 61% Poisson, 68% NB. Arithmetic is no longer 9% of the
work; it is most of it.

The free option was taken first. Folding the usable mask into precomputed
vectors and preallocating every temporary removed per-call allocation and
boolean indexing, worth 1.3-1.4x on a Poisson or NB evaluation at realistic
size and, unlike a JIT, never a regression at any size. End to end it took the
Poisson fitting stage from 0.594 s to 0.504 s and NB from 0.891 s to 0.749 s.
Legacy barely moved, at 1.03x, because its closure was already lean.

What Numba would add on top, measured against the optimized NumPy and verified
bit-identical at every shape:

| locus shape | legacy | Poisson | NB2 |
|---|---|---|---|
| 120 x 2 | 3.4x | 3.6x | 3.8x |
| **329 x 3** (realistic median) | 1.9x | **2.1x** | **2.2x** |
| 1000 x 8 | 1.4x | 1.6x | 1.6x |
| 4000 x 30 | **0.29x** | **0.32x** | **0.41x** |

Two things stand out. The count likelihoods gain more than legacy, consistently,
because they carry more elementwise work per matrix-vector product — `xlogy`,
`log1p`, divisions — and that is exactly what a JIT fuses and NumPy cannot.
And the last row is a trap: a naive JIT kernel is **three times slower** than
NumPy on large loci, because a hand-written loop cannot compete with BLAS on a
4000 x 30 product. Any adoption therefore needs a size-dependent dispatch, not
a wholesale replacement.

Applying Amdahl to the realistic regime, a 2.1x Poisson evaluation inside a
stage that is 61% evaluation yields about 1.5x on fitting and roughly 1.2x end
to end.

**Decision: still no, and the reasoning has changed.** It is no longer that the
gain is negligible — it is real and now well quantified. The costs are:

- a 38 MB `llvmlite` dependency in the core, for a package whose current
  dependency list is deliberately five entries long;
- a size-dispatch threshold that has to be tuned per solver and revisited on
  new hardware;
- and the decisive one, a **second implementation of every likelihood** that
  must stay bit-identical to the first. That is precisely the duplication that
  produced the naive NB2 formula bug, and there are now three likelihoods plus
  four penalty combinations to keep in step.

Against that, wall time is already 14.5x better than R DENR with both section
14 targets met, so the marginal 1.2x buys nothing anyone is waiting for.

Revisit if any of these change: profiling shows the fused closure back above
roughly 70% of a stage that someone is actually blocked on; a whole-genome
multi-sample run moves the realistic locus distribution toward the small shapes
where the gain is 3x rather than 2x; or a JAX or PyTorch backend arrives for the
ML adapters anyway, at which point the JIT dependency is already paid for and
the batched-loci idea below becomes the natural way to use it.

### 14.2 Measured comparison against R DENR

Both implementations run on identical generated datasets
(`scripts/benchmark_*`): 3000 transcripts, 3 samples, bin size 250, masks ±4
bins, log transform. Model counts agree exactly, so the ratios compare like
with like.

Peak RSS is measured externally for both, with `/usr/bin/time -l`, because R
offers no reliable in-process figure: `gc()` reports heap cells rather than
resident set size, and deriving MB from it gave a nonsensical result.

Dense regime — 20 Mb contig, 124 loci, 2838 models, median 18 transcripts per
locus:

| stage | DENR | dienr | ratio |
|---|---|---|---|
| annotation parse | 0.54 s | 0.10 s | 5.7x |
| design construction | 3.65 s | 0.14 s | 27x |
| signal aggregation | 8.28 s | 0.34 s | 24x |
| fitting | 6.46 s | 0.70 s | 9.3x |
| **total** | **18.9 s** | **1.27 s** | **14.9x** |
| peak RSS | 2417 MB | 301 MB | 8.0x |

Sparse regime — 200 Mb contig, 955 loci, 2865 models, about 3 models per locus,
much closer to a real annotation:

| stage | DENR | dienr | ratio |
|---|---|---|---|
| annotation parse | 0.54 s | 0.01 s | 60x |
| design construction | 1.58 s | 0.11 s | 14x |
| signal aggregation | 12.28 s | 0.33 s | 38x |
| fitting | 2.82 s | 0.74 s | 3.8x |
| **total** | **17.2 s** | **1.19 s** | **14.5x** |
| peak RSS | 2265 MB | 249 MB | 9.1x |

The section 14 target of 3x end-to-end is met in both regimes, with the
narrowest single-stage margin being 3.8x on sparse fitting.

#### What the first round of these measurements got wrong

An earlier revision of this section reported 1.3x on sparse geometry, with
fitting at 0.4x — slower than R — and drew two conclusions from it. Both were
wrong, and the way they were wrong is worth keeping:

1. It concluded that `scipy.optimize.minimize` scaffolding made us lose to R on
   many small loci. Replacing it with a lean `setulb` driver (`solvers/_lbfgsb`)
   is genuinely 2.8x faster per call on small problems and bit-identical, but it
   moved the end-to-end sparse figure by only 12%. The scaffolding was real and
   worth removing; it was not the bottleneck.
2. It concluded that per-locus BigWig reads stop scaling and needed batching per
   contig. That was simply false.

The actual cause of both was one line: `Design.mask()` filtered the `bins`
DataFrame instead of indexing a cached list. It is called once per locus per
sample, from both `count_sample` and `collect_results`, so it cost **5.57 s of
a 5.92 s fitting stage** and dominated signal aggregation too. Fixing it took
sparse fitting from 6.19 s to 0.74 s and sparse aggregation from 5.68 s to
0.33 s.

Two lessons for the remaining performance work. First, profile before
attributing: the per-call microbenchmark of the optimizer was accurate and still
pointed at the wrong target, because it measured a component rather than the
stage. Second, an accessor that looks like a field lookup but scans a DataFrame
is the characteristic failure mode of this design; `Design` now caches `masks`
and `bin_lengths` as lists precisely so per-locus access cannot regress to a
scan. Any new per-locus accessor must do the same.

Provisional performance targets, to be finalized after Phase 0:

- At least 3× lower end-to-end wall time than R DENR on the agreed multi-sample
  benchmark. **Met: 14.9× dense and 14.5× sparse (section 14.2).** The
  narrowest stage margin is 3.8× on sparse fitting.
- At least 2× lower peak memory or a documented explanation of the limiting
  representation. **Met in both regimes: 301 MB against 2417 MB dense (8.0×),
  249 MB against 2265 MB sparse (9.1×). R's footprint is roughly flat at about
  2.3 GB across both, while ours tracks the data.**
- Near-linear useful scaling through four CPU workers for the fitting stage.
- Reusing a design for a new sample must avoid all annotation and equivalence-class reconstruction.

### 14.4 Genome scale: where the time goes (November 2026)

`scripts/profile_k562.py`, one process per run, each stage timed and run
under cProfile. K562 PRO-seq rep1, Ensembl GRCh37.87, contigs renamed; Python
3.13.13, polars 2.1.0, numpy 2.5.2, the Poisson single-model fast path. The
14.2 benchmarks were 124 and 955 loci; this is 18,967 loci / 49,511 models
(protein-coding, 81,579 transcripts, 11,633 multi-model loci, up to 32 models
and 9,219 bins) and 46,994 loci / 122,969 models (every transcript).

Same node (cbsubscb20, EPYC 9554), protein-coding, 8 data threads:

| stage | BLAS 8 threads | BLAS 1 thread |
|---|---|---|
| build_design | 9.60 s | 9.39 s |
| fit_legacy | 11.43 s | 7.85 s |
| fit_poisson | 8.84 s | 7.03 s |
| count (sum) | 2.50 s | 1.89 s |
| parse | 1.79 s | 1.51 s |
| runon_tail | 0.64 s | 1.16 s |
| strand check, apply tail, write | < 1 s each | < 1 s each |
| **total** | **35.8 s** | **29.7 s** |

- **A whole-genome run takes about half a minute.** Fitting is about half of
  it; design building about a third.
- **Multithreaded BLAS costs 1.26-1.46x on fitting** -- the matrices are tiny
  and the threads only add synchronization. Capping BLAS at one thread inside
  `fit` (threadpoolctl) is the cheapest speedup available.
- **An earlier "4x BLAS slowdown" was the node, not BLAS.** The first profile
  ran BLAS-8 on cbsubscb16 (EPYC 7763) and BLAS-1 on cbsubscb20, and every
  stage was ~4x slower on 16, including pure-Python ones (build_design 53 s vs
  14 s). 18.18's rep1 trial, also on cbsubscb16, took 2,407 s against rep2's
  323 s on cbsubscb17. Compare configurations on one node.
- **Cold start is a separate, larger cost on Lustre.** The first figwig read
  in a fresh process spent 24 s in `runon_tail` on a cold cache: 11.4 s in 372
  `stat` calls and 2.8 s in `open_code`, all from importing numba/llvmlite
  modules out of a `.venv` on /home2, plus ~1.6 s of numba compilation. Warm,
  the same stage takes 0.6-1.2 s. Each CLI command is a new process, so a
  `.venv` on local disk (/workdir) matters more than any kernel.
- **build_design's cost is Polars construction per locus**, not geometry:
  18,967 small DataFrames (4.2 s cumulative in Polars constructors) and
  326k calls to `design._mark` (2.3 s). Building the bins table once, in
  NumPy, is a plain vectorization job.
- **Numba: still no.** The arithmetic in the fitting closures is ~5 s of a
  30 s run; 14.3's ~2x on those would save ~2.5 s, against a second
  bit-identical implementation of every likelihood. The two items above
  (BLAS cap, vectorized design build) are worth more and cost less.

**Both done.** `fit(blas_threads=1)` (CLI `--blas-threads`, `auto` to leave
the library alone) wraps fitting in `threadpoolctl.threadpool_limits`; results
are unchanged, which a test pins. `build_design` now slices transcripts per
locus from one stable sort instead of `partition_by` + per-locus DataFrames,
vectorizes both mask forms over transcripts, and builds the bin table once.
Timed on one node (cbsubscb20, job 4108219, best of 3), every output
bit-identical to the previous code on the K562 annotations -- digest, bins,
models, masks, spans, maps, for paper, base-window and no masking:

| annotation, masks | before | after |
|---|---|---|
| protein-coding, paper | 5.49 s | 2.35 s |
| protein-coding, no PAS mask | 4.85 s | 2.10 s |
| every transcript, paper | 12.80 s | 5.27 s |
| every transcript, no PAS mask | 11.43 s | 4.89 s |

What remains is ~0.1 ms of NumPy call overhead per locus, spread evenly;
going further means batching loci together, which isn't worth it at ~2 s.

### 14.5 Lazy figwig import and parallel fitting (November 2026)

**Lazy import.** `import dienr` used to load figwig and with it numba and
llvmlite, so `build-design` and `fit` paid for a JIT stack they never use.
Every BigWig is now opened through `counts.open_bigwig`, which imports figwig
on first use; a test asserts `import dienr, dienr.cli` loads none of the three.
Fresh-process `import dienr.cli` on cbsubscb20 with the `.venv` on /home2
(job 4108222, alternating old/new): **15.9 s -> 1.8 s** on the first, cold
run; ~2.0 s -> 0.77 s warm.

**Parallel fitting.** `fit(n_jobs=N)` (CLI `--jobs`) fits loci in worker
processes. Every locus is fitted by the same function, so outputs are
identical for any N (tested for legacy, Poisson and quasi-Poisson). Each
locus's data goes to one worker once, in contiguous chunks balanced by
bins x models^2, eight chunks per worker. K562 protein-coding, one sample,
cbsubscb20 (job 4108225), every output identical to N = 1:

| | N = 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| legacy (defaults) | 7.73 s | 4.91 s | 3.17 s | 2.94 s |
| quasi-Poisson + tail | 7.17 s | 5.11 s | 3.82 s | 3.67 s |

About 2.5 s stays serial -- building per-locus tasks, assembling the result
tables, and quasi-Poisson's pooled standard-error pass -- so 4 workers get
nearly all of it. The CLI defaults to 4, capped by `os.sched_getaffinity`
(which honours a Slurm allocation); the library defaults to 1.

**Start method -- a trap found while timing.** `forkserver` and `spawn`
re-import the caller's `__main__`, so a script calling `fit(n_jobs=4)`
without an `if __name__ == "__main__":` guard crashed (the timing script
itself did). `fork` doesn't, but is unsafe while another thread is mid-work.
`_start_context` forks when every other Python thread is a
`concurrent.futures` pool worker -- in practice figwig's shared reader pool,
which outlives each read, is idle once `count` returns, and is keyed by PID --
and falls back to `forkserver` otherwise. Python's "multi-threaded fork"
DeprecationWarning is silenced around the pool only, because NumPy's OpenBLAS
threads trigger it in every process and OpenBLAS reinitializes itself after
fork. A test runs an unguarded script that counts, holds an idle executor,
and fits with N = 2; it fails under the stricter "no other thread" rule.

**Not done: warm starts across samples.** Starting each Poisson locus at the
previous sample's estimate would save iterations only at multi-model loci in
multi-sample runs, and would move results by solver tolerance (single-model
loci already take the closed form). Next to the above it isn't worth the
change in outputs.

## 15. Normalization and downstream inference

Fitting and between-sample normalization must be separate operations.

- `abundance_raw` represents the fitted polymerase density or count-rate parameter.
- Spike-in or known exposure factors may enter the likelihood directly and must also be reported.
- Library-size, median-ratio, or conventional TMM factors should be stored in a separate table.
- Composition-normalized values may be exported for visualization and ML features.
- Differential testing should use an explicitly defined count-like quantity and an appropriate downstream method; it should not be performed on ambiguously normalized values.

The initial release should not call any custom trimmed-mean calculation “TMM.”

## 16. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Coordinate off-by-one errors | Single internal convention, R fixtures, boundary property tests |
| Reimplementation changes the scientific method | Frozen legacy mode and staged validation gates |
| Poisson underestimates PRO-seq variability | Goodness-of-fit tests; add shared-dispersion NB/quasi-Poisson only when justified |
| Regularization creates false isoform certainty | Preserve equivalence classes; report conditioning; label model-dependent allocations |
| Normalized BigWigs are treated as counts | Strict input metadata and likelihood compatibility checks |
| Uncovered BigWig bins read as `NaN` instead of zero | Set `fillna`/`oob`/`uncovered` explicitly at the I/O boundary; fixtures for empty and terminal bins |
| Zoom-level approximation silently biases binned signal | Aggregate intervals in Polars, or pass `exact=True`; parity-test binned sums against per-base ground truth at multiple bin widths |
| `polars-bio` is pre-1.0 and coupled to Polars versions | Pin both narrowly; gate adoption on Phase 1 R parity; keep a small internal interval layer as fallback |
| Parallel execution changes results | Deterministic ordering, fixed seeds where relevant, tolerance-based reproducibility tests |
| Sparse representation is slower for small dense loci | Select dense versus sparse representation by profiled locus characteristics |
| External TSS evidence is assay-dependent | Record evidence source, mapping window, strand rules, and filtering mode |
| License ambiguity complicates derived distribution | Clarify with maintainers or implement from the paper/specification without copying source |

## 17. Decision gates

### Gate A: Is the Python geometry trustworthy?

Proceed beyond legacy MVP only if design matrices, masks, and equivalence classes match R fixtures and coordinate edge cases pass.

### Gate B: Does Poisson improve inference?

Adopt Poisson as the recommended count-data solver only if it improves prespecified simulation or held-out predictive metrics and behaves robustly at zero/low counts.

**Verdict: partially met, and the headline finding is about the legacy default
rather than about Poisson.** Evidence from `scripts/validate_likelihoods.py`,
four prespecified regimes, contiguous held-out blocks, paired per-fold
comparison, stable across four seeds. Numbers below are the fraction of folds
Poisson wins on held-out deviance.

| regime | vs `legacy_log` (DENR's default) | vs `legacy_identity` |
|---|---|---|
| low count (mean 0.5) | **78-88%** | 40-50% |
| moderate (mean 8) | 51-64% | 50-64% |
| high count (mean 200) | 45-57% | 42-62% |
| heterogeneous depth (10x) | 61-67% | 53-56% |

Three conclusions.

1. **DENR's default `transform = "log"` is badly misspecified once zeros are
   common.** At mean 0.5 counts per bin its held-out deviance is about 3.1
   against 1.2 for every alternative, and it loses 78-88% of folds. The cause
   is structural: a bin with ``y = 0`` contributes
   ``[log(1e-3) - log(mu + 1e-3)]^2``, roughly 38 at ``mu = 0.5``, so empty
   bins dominate the objective and drag the fit toward zero. This is a
   criticism of the reference method, not of our implementation of it, and it
   is the single most actionable result here.
2. **Switching the legacy transform to `identity` recovers nearly all of the
   achievable gain.** Poisson's *marginal* edge over `legacy_identity` is nil
   in the low-count regime the gate names explicitly (40-50%), modest at
   moderate counts and under heterogeneous depth (53-64%), and
   indistinguishable from noise at high counts. So the gate's own criterion —
   improvement over legacy fitting in the low-count regime — is **not met**
   against the strongest legacy objective, though it is met decisively against
   the default one.
3. **No regression at high counts**, as required.

#### Why Poisson barely beats least squares, and what would change that

The tie against `legacy_identity` is not a numerical accident, and it has an
exact explanation. The two estimating equations differ only in a weight:

    Poisson:  sum_b M_bk (y_b - mu_b) / mu_b = 0
    MSE:      sum_b M_bk (y_b - mu_b)        = 0

So when ``mu`` is constant across a model's support, ``1 / mu`` factors out and
the equations are **the same equation**. For a single 0/1 design column that is
exactly the case — both reduce to ``a = sum(y) / n_support`` — and measurement
confirms it: over 200 draws the two estimates differ by ``0.000e+00``. They are
not similar estimators; they are one estimator.

**`bin_operation` is what causes this.** Rounding fractional occupancy to 0/1
makes every design column an indicator, which removes precisely the
heterogeneity the Poisson weighting exists to exploit. With overlapping
indicator columns ``mu`` takes only two distinct values across the benchmark
locus, so the weight is nearly constant and the two estimators stay within a
few percent — a consistent 59-60% win on abundance error at every sequencing
depth, from 0.5 to 200 counts per bin. Depth-independent, because the
limitation is a property of the design, not of the counts.

Even a graded design has a low ceiling. For one model the variance ratio is

    Var(MSE) / Var(Poisson) = sum(M^3) sum(M) / sum(M^2)^2

which Cauchy-Schwarz puts at or above 1, with equality iff ``M`` is constant.
Measured against theory:

| design | predicted ratio | measured | Poisson wins |
|---|---|---|---|
| constant M (what rounding produces) | 1.0000 | 1.0000 | identical |
| M in [0.5, 1] | 1.034 | 1.059 | 58% |
| M in [0.05, 1] | 1.123 | 1.127 | 56% |
| shape-profile-like, M in [0.02, 1] | 1.125 | 1.143 | 56% |

A fiftyfold range in occupancy buys 12% in variance, about 6% in standard
error. That is the entire prize available from switching likelihood on this
class of design.

**That last sentence was wrong.** Every row
above draws ``M`` roughly *uniformly* over its range, and for a uniform ``M``
the ratio is insensitive to how wide the range is -- 38.8x gives 1.122 and
50x gives 1.125. The ratio `sum(M^3) sum(M) / sum(M^2)^2` is a kurtosis-like
quantity: it is driven by **outliers, not by range**. A real pausing profile is
exactly the outlier case, one bin far above a nearly flat remainder:

| M distribution | range | Var ratio | SE ratio |
|---|---|---|---|
| uniform on [0.02, 1] | 38.8x | 1.122 | 1.059 |
| uniform on [0.02, 1], evenly spaced | 50.0x | 1.125 | 1.061 |
| **actual fitted shape profile** | 14.3x | **2.329** | **1.526** |
| one bin at 14x, rest at 1 | 14.3x | 4.044 | 2.011 |

Measured on a reshaped design from `dienr.shape` over 200 Poisson resamplings,
against `legacy` identity least squares on the same data:

| design | occupancy range | Poisson RMSE | MSE RMSE | SE ratio |
|---|---|---|---|---|
| binary | 1.00x | 0.2367 | 0.2367 | **1.000** |
| shape-reshaped | 14.30x | 0.1882 | 0.2854 | **1.516** |

Predicted 1.526 against measured 1.516, so the theory is right and only the
assumed ``M`` distribution was wrong. The prize is about **50% in standard
error**, not 6%, and on a binary design it remains exactly zero.

Three actionable consequences.

- **The shape profile is a prerequisite for Poisson to be worth anything at
  all**, and it is now implemented in `dienr/shape.py`. On a binary design the
  two estimators are the same to `0.000e+00`; on a reshaped one Poisson is
  1.5x better in standard error. This reverses the earlier reading that the
  gain would be marginal even with a graded design.
- **Per-bin exposure is the other source of heterogeneity**, and it is already
  implemented. Mappability and partial terminal bins vary ``e_b`` and therefore
  ``mu``, and least squares cannot use that information at all.
- **Rounding the design matrix is a lever, not a fixed cost.** An unrounded
  fractional design would recover some of the gain; against that, exact
  equivalence classes and the whole identifiability apparatus depend on
  integer columns, so this trades one thing the plan cares about for another
  and should not be changed casually.

Poisson's justification therefore is not point accuracy. It is the exposure
term, which no legacy objective has: across a tenfold depth spread the
abundance estimate's relative spread is 0.16 for Poisson against 0.17-0.21 for
`legacy_identity` and 0.29 for `legacy_log`, because only Poisson can return a
depth-independent abundance. Add the principled variance model needed for
later inference and the case is real, but it is a narrower case than "Poisson
is more accurate."

**Actions taken.** Legacy stays the default, since it is the only likelihood
with R parity. Poisson is positioned for multi-sample and heterogeneous-depth
work rather than as a general accuracy improvement. The low-count behavior of
`transform = "log"` is documented where users will meet it.

**A fourth finding, from the same run: a fixed ridge weight does not transfer
across count scales and must never be a default.** The ridge term grows as
``a^2`` while the Poisson log-likelihood grows roughly as ``a``, so
``lambda = 0.01`` is negligible at ``a ~ 1`` and catastrophic at ``a ~ 200``:
held-out deviance 68.7 against 1.06 unpenalized, winning 1-3% of folds.
Scaling ``lambda`` by a moment estimate of the abundance removes the failure
entirely (57% of folds, best abundance error in the regime).

The mechanism is exact enough to give a rule. The ridge gradient is
``2 lambda a`` while the Poisson score accumulates over bins, so the relative
shrinkage a ridge weight imposes is

    shrinkage ~ 2 lambda a / n_bins

verified within 25% at three combinations of ``n_bins`` and abundance. Two
consequences follow. A weight chosen at one count scale is wrong at another, in
proportion to the abundance ratio. And ``lambda`` proportional to ``1 / a``
cancels the dependence, leaving shrinkage at ``2c / n_bins`` — still falling
with the amount of data, which is the desired behavior for a penalty.

This is concrete motivation for the Phase 4 hyperparameter-selection
deliverable, and until that exists any penalty weight must be chosen per
dataset rather than inherited. `dienr fit --lambda2` is deliberately not
defaulted to anything nonzero.

### Gate C: Does regularization improve stability rather than merely appearance?

Recommend a penalty only if it improves replicate or bootstrap stability and simulated abundance recovery without hiding non-identifiability.

### Gate D: Is NB complexity warranted?

Ship NB as a supported model only if overdispersion is reproducible, dispersion can be estimated stably, and held-out prediction improves materially over Poisson.

**Verdict on the criterion as written: no, and it cannot be met on this design.
But the criterion is the wrong one.**

NB2 cannot improve the point estimate here, and not marginally — at all. The
identity-link score weights each residual by ``1 / V(mu)``, so when ``mu`` is
constant across a model's support that weight factors out and the estimating
equation reduces to the sample mean *whatever the variance function is*.
Measured on a single 0/1 column, least squares, Poisson and NB2 agree to
``0.000e+00`` at dispersion 0.3 and again at 3.0. Severe overdispersion changes
nothing, because the variance function is algebraically irrelevant to the
estimate. "Held-out prediction improves materially over Poisson" is therefore
unachievable by construction, and holding NB to it would reject it for the
wrong reason.

**What NB2 is actually for is the uncertainty, and there the gap is large.**
With constant ``mu`` the model-based standard error is ``sqrt(V(mu) / n)``, so
Poisson reports ``sqrt(a/n)`` while the truth is ``sqrt(a(1 + phi a)/n)``.
Against the observed spread of 4000 replicate estimates:

| dispersion | true SD | Poisson SE | NB2 SE | Poisson error |
|---|---|---|---|---|
| 0.0 | 0.416 | 0.418 | 0.418 | +1% |
| 0.1 | 0.542 | 0.419 | 0.546 | **-23%** |
| 0.3 | 0.737 | 0.418 | 0.735 | **-43%** |
| 1.0 | 1.175 | 0.417 | 1.178 | **-64%** |

A Poisson-only pipeline is therefore not wrong about abundance but badly
overconfident about it, by up to a factor of three in variance. Anything
downstream that ranks or tests — differential expression, intervals, thresholds
on effect size — inherits that overconfidence.

**Two consequences, and the second undercuts NB2 specifically.**

1. **NB2's value is unrealized in the current output.** We emit no standard
   error or variance anywhere, so `fit(likelihood="negative_binomial")` today
   returns the same abundance as Poisson with a different objective value —
   close to useless. Adding a variance column to `model_abundance` is the
   prerequisite for NB to be worth selecting at all, and is a smaller change
   than NB2 itself was.
2. **Quasi-Poisson matches NB2 here, far more cheaply.** Inflating the Poisson
   standard error by ``sqrt(Pearson chi2 / df)`` — one scalar from a Poisson
   fit — tracked the true spread to within 1-4% across dispersions 0.1 to 1.0,
   indistinguishable from NB2's 1-2%. That is expected rather than lucky: for
   constant ``mu`` the Pearson statistic estimates ``1 + phi mu``, which is
   exactly the NB2 inflation factor.

**Implemented.** `dienr.solvers.inference` supplies Wald standard errors and a
pooled quasi-Poisson dispersion, and `fit(likelihood="quasi_poisson")` returns
the Poisson abundances with errors scaled by ``sqrt(phi)``. Three details worth
recording:

- The dispersion is pooled over every locus of **every sample** in the run,
  which needs a second pass after fitting. Per-locus estimation is forbidden
  here for the same reason section 4.3 forbids it for NB2.
- Standard errors are **NaN where a Wald error is not defined**: a model at the
  nonnegativity boundary has a one-sided sampling distribution, and a singular
  information matrix leaves the error undefined rather than large. Emitting a
  big finite number in either case would invite a symmetric interval that does
  not exist.
- `is_overdispersed` requires the estimate to clear 1 by ``z`` standard errors,
  where the Pearson statistic's standard error is ``sqrt(2 / df)`` under the
  null. A bare ``> 1`` test fires on about half of perfectly equidispersed
  datasets; it did so during development, at 1.0215 on Poisson data.

So the honest recommendation, now the shipped behavior, is to **report
quasi-Poisson standard errors from the Poisson fit, and treat NB2 as
optional**. QP needs no second likelihood, no
dispersion held fixed per fit, no non-convex objective, and no mean-collapse
guard — all of which NB2 required. NB2 earns its place only where ``mu`` varies
enough across a locus that the full variance function matters for the estimate
too, which on a rounded binary design it does not, and which brings this back
to the shape-profile prerequisite recorded under Gate B.

**Decision: NB2 removed.** Once 18.8 found that a shape profile is mostly not
worth applying, the one setting where NB2 could matter -- a graded design --
stopped being a recommended configuration, and nothing else justified a
non-convex likelihood that needs an externally pooled dispersion and a
mean-collapse guard. `solvers/negative_binomial.py` and its tests were deleted;
`fit` and `dienr fit` accept `legacy`, `poisson` and `quasi_poisson`, and the
`dispersion` argument went with it. The evidence above stays as the record of
why; re-adding NB needs new evidence against this gate, most plausibly on real
data with a design graded for reasons other than a shared shape profile.

## 18. Shape profiles: what the reference gets wrong

The shape profile was a v1 non-goal until Gate B and Gate D independently
converged on it: with a rounded binary `M`, `mu` is constant across a model's
support, so least squares, Poisson and NB2 are the *same estimating equation*
and no choice of likelihood can move a point estimate. A graded design is the
only thing that breaks that degeneracy. It is now implemented in
`dienr/shape.py`.

It was also reported to crash frequently in the reference. It does, and it does
something worse. Everything below was reproduced against DENR 1.0.0 at commit
`09d6cde` in the `dienr-rref` conda environment. `scripts/probe_r_shape_profile.R`
regenerates every number in this section:

```bash
/Users/adamhe/miniforge3/envs/dienr-rref/bin/Rscript scripts/probe_r_shape_profile.R
SHAPE_PROBE_TRIALS=700 ... # raise the 18.4 fuzz, which dominates runtime
```

It lives outside pytest because it needs the reference environment. The Python
side of the same findings is pinned as regressions in `tests/test_shape.py`,
which needs no R.

### 18.1 It crashes

**`rescale_fixed_width_position` assigns a zero-length sequence.** When the
gene is shorter than head plus tail it splits the available bins in the
head:tail ratio, and `round(prop_head * length_out)` can be 0. It then does
`rescaled_pos[1:0] <- seq(0, head_percent, length.out = 0)`, and `1:0` in R is
the two-element vector `c(1, 0)`:

```
len=2 head=1000 tail=5000 bs=250  -> ERROR: replacement has length zero
len=3 head=1000 tail=5000 bs=250  -> ERROR: replacement has length zero
len=5 head=1000 tail=9000 bs=250  -> ERROR: replacement has length zero
```

Reachable straight from the documented API whenever the tail region is longer
than the head and `min_transcript_length` is small:

```
min_tx_len=1e3 head=2e3 tail=6e4  -> ERROR: replacement has length zero
```

`length_out = 0` gives a different opaque failure,
`'length.out' must be a non-negative number`.

**`model_agreement` indexes a group table by transcript names.** It computes
`percent_err <- total_err / width(group_ranges[names(tx)])`, where `tx <-
unlist(tx_grps)`. `unlist` on a `GRangesList` concatenates the group name with
the inner name, so `names(tx)` is `"100_+.g227"` where `group_ranges` is keyed
`"100_+"`:

```
names(gr) <- gr$tx_name  -> ERROR: subscript contains invalid names
names(gr) NULL           -> ok
```

The function works *only* when the input `GRanges` has no names, which is an
accident of `GenomicFeatures::transcripts()` returning unnamed ranges.
`names(gr) <- gr$tx_name` is an ordinary idiom and kills the run.

**The documented defaults `bigwig_plus = NULL, bigwig_minus = NULL` crash** with
an rtracklayer error about `'filename' must be a single string`.

### 18.2 Worse, it is silently inert

Two independent defects in the signal-extraction step.

`plus_grps <- grep("+", names(final_bins))` uses `+` as a **regex**, where it is
a quantifier that matches every string:

```
names:  1_+  2_+  3_-  4_-
grep("+", names)             -> 1,2,3,4      # all of them
grep("+", names, fixed=TRUE) -> 1,2          # what was meant
grep("-", names)             -> 3,4
```

And the minus branch reads the **plus** file regardless:

```r
minus_counts <- lapply(summarize_bigwig(bigwig_file = bigwig_plus,
                                        final_bins[minus_grps]), rev)
```

So `bigwig_minus` never contributes signal to the profile at all; it is used
only in the filtering step. Every minus-strand gene is fed to the smoother
*twice* -- once unreversed via the regex bug, once reversed -- and both copies
carry antisense signal. Confirmed by the gene count on the two-strand fixture:
2 genes selected, "Fitting profile model using **3** genes".

On 255 simulated genes with a 14x pausing peak and correct stranded BigWigs:

| | curves fed to loess | all-zero curves | fitted dynamic range |
|---|---|---|---|
| DENR as written | 174 (from 116 genes) | 116 (67%) | **1.25x** |
| strand handling fixed | 116 | 0 | **14.63x** |

Two thirds of the training data is identically zero, and the profile that comes
out multiplies the design by about 1.0 everywhere. **This does not fail; it
produces a shape profile that does nothing.**

It cannot be caught by the reference's tests, because every shape-profile test
in `test-shape_profile.R` passes the same BigWig for both strands
(`bigwig_plus = bw_1000, bigwig_minus = bw_1000`), which makes a plus/minus
swap undetectable by construction. `test-shape_profile.R` also computes a
reference vector `ref_1` for the rescaler and never asserts it -- and it is
wrong, ending `0.1` where the function returns `1`.

### 18.3 Three further defects, not crashes

**Per-gene min-max rescaling manufactures data.** `(x - (min(x) - 1e-6)) /
(max(x) - (min(x) - 1e-6))` maps an all-zero gene to a constant 1.0 instead of
dividing by zero, so absent signal becomes evidence for a flat profile -- which
is precisely the state the strand bug puts most genes in. It also makes the
single largest bin, normally the pausing spike, set the scale for the whole
gene. `dienr` divides by the gene's own mean and *drops* silent genes, counting
them in `ShapeProfile.rejected`.

**`predict.loess` returns NA outside its training hull**, and
`apply_shape_profile` evaluates it on `seq(0, 1, by = 1e-5)` and writes the
result straight into the model matrix. With the reference's own defaults the
training positions happen to span exactly 0 and 1 so no NA appears, but nothing
enforces that. Confined to `[0.3, 0.7]`, **60,005 of the 100,001 grid points
come back NA**, and a NaN weight makes the objective non-finite so L-BFGS-B
rejects the locus. `dienr` fits at fixed grid nodes spanning `[0, 1]`, so the
case is unreachable rather than guarded.

**The defaults are internally inconsistent.** `min_transcript_length = 8e3` sits
below `linear_head_length + linear_tail_length = 15e3`, so with defaults every
gene between 8 and 15 kb takes the squeeze branch and contributes *no body
positions at all* -- only `[0, 0.2]` and `[0.8, 1]`. `ShapeConfig.min_length`
defaults to head plus tail and validates against it.

The R fit also emits `Warning: stack imbalance in '<-', 18 then 20` at scale, a
C-level symptom with no clear cause. `loess` itself scales fine (2M points in
0.82 s with `trace.hat = "approximate"`), so the smoother was never the
bottleneck the stripping and caching in the reference suggests.

### 18.4 What `apply_shape_profile` gets right

Worth recording, since it was the prime suspect. The reference's application
step averages member transcripts' coordinates into an "expected" bin range and
then deletes the entries that rounding or masking removed, keeping two index
spaces aligned by hand:

```r
omit <- -which(models[expected_range[1]:expected_range[2], m] == 0)
models[non_z, m] <- pre_profile[lookup_ind][omit]
```

That looks fragile and is not. 700 fuzzed annotations -- dense isoform
clusters, both strands, all three `bin_operation` values, start and end masking
-- produced no crash and no misalignment, checked against a monotone profile.
`ceiling((d+1)/b) == floor(d/b)+1` makes the expected start exactly the bin
containing the transcript start, and columns only collapse together when their
rounded patterns already agree, which bounds the averaging error to the bins
that `omit` then removes.

`dienr` still does not copy the approach, because it does not need it:
collapsing to equivalence classes first means a model column *has* one geometry,
so `Design.model_spans` records the pre-mask span and the mapping is direct.

### 18.5 Two findings of our own

**The reference's `span = 0.4` over-smooths the feature the profile exists to
measure.** A pausing peak is a few hundred bases wide, which the head rescaling
maps to well under a hundredth of the axis; a span of 0.4 gives a bandwidth
around 0.2. Against the unsmoothed training cloud on the simulated data
(empirical range 9.29x):

| span | fitted range | value at position 0 |
|---|---|---|
| 0.4 (reference default) | 2.38x | 1.95 |
| 0.2 | 3.00x | 2.49 |
| 0.1 | 4.00x | 3.33 |
| 0.05 (`dienr` default) | 5.59x | 4.66 |
| 0.02 | 8.47x | 7.08 |
| empirical, unsmoothed | 9.29x | 7.74 |

The default therefore discards about three quarters of the signal even once
every bug above is fixed. `ShapeConfig.span` defaults to 0.05, with
`min_bandwidth` flooring the neighbourhood at two head bins so a sparse
training set cannot be fitted more finely than the bins the profile multiplies.
Local linear regression can also undershoot below zero at the peak;
`PROFILE_FLOOR` clamps it and `ShapeProfile.n_floored` reports it.

**The prize from a count likelihood is larger than Gate B estimated.** The
correction is recorded under Gate B above: the earlier ceiling assumed a
uniform ``M``, and the relevant statistic is driven by outliers rather than by
range. A fitted pausing profile gives Poisson a ~1.5x standard-error advantage
over least squares, where a binary design gives it exactly none.

**A shape profile and TSS masking compete for the same bins**, and applying
both largely cancels the profile. Masking calls the pausing peak untrustworthy;
the profile calls it predictable and supplies its magnitude. On 40 kb simulated
genes with a profile fitted at 14.3x:

| masking | applied range |
|---|---|
| none | 14.30x |
| `mask_tss=(0, 10000)` | 1.60x |
| `mask_tss=(0, 20000)` | 1.40x |
| `mask_tss=(0, 20000)` + `mask_pas=(10000, 0)` | 1.16x |

Masking the first 10 kb of a 40 kb gene removes 89% of the grading. That is not
an argument against either feature, but a run that enables both and expects the
profile to do work is mistaken, and it looks identical from the outside to one
that works. `shape.applied_dynamic_range` reports the range surviving in the
reshaped design, `build-design` prints it, and it warns below 1.2x. **The
number to look at is the applied range, not `ShapeProfile.dynamic_range`.**


### 18.6 Validated on the authors' own simulator

The paper has **no real-data gold standard**: section 3.2 says so directly --
"Lacking an appropriate 'gold-standard' in the form of real biological data, we
chose to benchmark the software using simulated data" -- and built
`nascentRNASim` (MIT, `CshlSiepelLab/nascentRNASim`) for the purpose. It
resamples read-counts from 62 curated PRO-seq archetypes onto simulated loci
drawn from a real annotation, with abundances from a GTEx skeletal-muscle
kernel, so per-isoform truth is known while the shape stays real. Both are
bundled with the package, so the paper's benchmark is reproducible:
`scripts/simulate_nascentrnasim.R` then `scripts/benchmark_shape_accuracy.py`.

Two 1500-locus simulations (seeds 5, 17), the paper's masking (1 bin TSS, 4
bins PAS), profile fitted on half the contigs and scored on the other half, one
global calibration constant removed per arm. Results, seed 5 / seed 17:

| arm | isoform abs log2 err | isoform rho | zero estimates |
|---|---|---|---|
| binary / legacy_log (DENR default) | 0.344 / 0.271 | 0.669 / 0.555 | 0.186 / 0.166 |
| binary / legacy_identity | 0.338 / 0.332 | 0.611 / 0.514 | 0.221 / 0.190 |
| binary / poisson | 0.320 / 0.328 | 0.613 / 0.492 | 0.205 / 0.202 |
| shape / legacy_log | 0.399 / 0.287 | 0.700 / 0.634 | 0.156 / 0.135 |
| shape / legacy_identity | 0.562 / 0.384 | 0.650 / 0.566 | 0.199 / 0.178 |
| shape / poisson | 0.328 / 0.283 | 0.656 / 0.573 | 0.186 / 0.160 |

**Corrected by 18.8 below.** The first reading of these two seeds was that
shape modeling "improves isoform ranking and not magnitude". A five-seed paired
follow-up does not support the ranking half: about half of that gain was the
zero-reduction effect appearing in rank correlation, and the remainder is not
sign-stable. Magnitude error does rise, consistently, and sharply under least
squares. Dominant-isoform accuracy swings between +9.8 and -2.4 points across
these two seeds, which was the first sign that two seeds were not enough.

**Why magnitudes do not improve, and it is not a defect.** The magnitude
estimand is a mean, and a mean is insensitive to shape. For a single column,
least squares gives `a_hat = sum(M y) / sum(M^2)`, which for `M = 1` is
`mean(y)` -- unbiased for a mean-normalized truth *whatever* the gene's shape.
Measured over 377 training genes, the uniform design's multiplicative bias has
median 1.006. Weighting by a shared profile can therefore only inject
misspecification: `a_hat/a = <g,p>/<p,p>`, which is 1 only if the gene's own
profile `g` is proportional to `p`, and it is not. Across-gene SD of `g(0)` is
**6.32** against a shared `p(0)` of 4.06 -- the peak height varies more than its
own mean -- while `p^2` places 26.7% of its weight in the first 5% of the axis
against 5.5% for a uniform design. So the profile weights the estimate most
heavily exactly where genes disagree most, and bias dispersion rises from
SD(log2) 0.228 to 0.264. An average-gene profile is a good estimate of a mean
and a bad model of any individual gene.

What it buys instead is **conditioning**, a different estimand. Overlapping
isoforms are near-collinear under a uniform design, which asserts they are
identical wherever they overlap; a graded design breaks the tie because each
isoform's peak sits at its own TSS, inside the other's body. Median condition
number at multi-model loci falls 7.44 -> 4.98, 90th percentile 39.0 -> 17.6.
Stratified, with calibration taken within each stratum:

| arm | 1 model: err / rho | 2+ models: err / rho |
|---|---|---|
| binary / legacy_identity | **0.076** / 0.986 | 0.452 / 0.738 |
| binary / poisson | **0.076** / 0.986 | 0.438 / 0.747 |
| shape / legacy_identity | 0.342 / 0.969 | 0.553 / **0.766** |
| shape / poisson | 0.182 / 0.985 | 0.456 / **0.773** |

**At single-model loci a shape profile is pure loss** -- 4.5x worse magnitude
under least squares, 2.4x under Poisson, no ranking gain, because there is
nothing to apportion. The whole rank-correlation improvement lives at
multi-model loci. The two objectives agree to four decimals on the binary
single-model stratum, which is Gate B's collapse appearing in the data.
Applying the profile only at multi-model loci is the obvious consequence and is
**not yet validated**: such a design has two natural abundance scales and the
single global calibration constant here cannot score it fairly.

**And a graded design is what makes the likelihood matter**, which settles the
Gate B question on the authors' own benchmark rather than by simulation of our
own design:

| | Poisson vs identity least squares |
|---|---|
| binary design | -1% to -5% isoform error |
| shape-reshaped design | **-26% to -42%** isoform error |

The mechanism is the one Gate B derived. A fitted profile puts a 4-5x weight on
the first bin; unweighted least squares is dominated by the largest design
entries, so a gene whose own peak departs from the shared profile is fitted
mostly where that departure is worst. Poisson's `1/mu` weighting is exactly the
correction. So the recommendation is **shape profile with `poisson`, not with
least squares** -- `shape/poisson` is the only arm that buys ranking without
paying in magnitude. The paper could not reach this, having only least squares.

**The reference's `span = 0.4` is harmful, not merely wasteful.** At 0.4 the
profile spans 2.0x and isoform rho falls to 0.625, *below* the 0.673 of no
profile at all; at the `dienr` default of 0.05 it rises to 0.700.
Over-smoothing inverts the benefit rather than reducing it.

Caveats. Two seeds, one simulator, one archetype set. `nascentRNASim` gives
every simulated isoform an archetype's profile, which is the assumption a shape
correction encodes, so this benchmark is friendlier to shape modeling than
biology will be. No real data: the paper's orthogonal comparison (matched
PRO-seq/RNA-seq, entropy decomposition) rests on dbGaP `phs002146.v1.p1`
(controlled access); public K562 `GSE96869` and CD4+ T `GSE85337` PRO-seq would
support a real-data check and have not been run.

### 18.7 Two further reference defects, and one of ours

**`min_percent_transcribed = 0.8` is unsatisfiable on count data.**
`percent_transcribed` is the fraction of a transcript's *bases* with at least
one read, threshold hardcoded at 1. PRO-seq 3'-end counts are sparse per base:
on nascentRNASim output, 1.07% of bases are non-zero (213,377 of 20 Mb) and
`percent_transcribed` peaks at 0.487, median 0.069. So
`transcript_shape_profile` **fails outright on the DENR authors' own simulator
at their own defaults**: `Error: No transcripts passed filters to be used for
model fitting`. This is the default path on real data, not an edge case, and is
the most likely crash a user hits. Lowering the threshold is not an escape --
at 0.4 exactly one gene survives and a loess profile is fitted from it without
complaint.

**The published "U shape" is largely an artifact.** A third strand defect:
minus-strand BigWigs are conventionally negative (their own simulator writes
min -1122, max 0); `add_data` applies `abs()` and
`transcript_shape_profile` does not, so min-max rescaling of an all-negative
vector maps the 5' pausing peak to ~0 and the background to ~1, inverting every
minus-strand gene. Two bugs hide each other: the minus file is never opened for
signal, so the missing `abs()` only fires once the strand dispatch is fixed.
Decomposed on the same 57 genes:

| variant | range | 5'/3' | shape |
|---|---|---|---|
| DENR as written | 4.3x | 1.33 | symmetric U |
| strand fixed, still min-max | 2.6x | 1.66 | weakly 5' |
| strand fixed, no `abs()` | 1.2x | 1.01 | flat |
| bug kept, mean-normalized | 10.2x | 1.24 | U, exaggerated |
| all three fixed | 2.4x | **2.13** | 5'-peaked, mild 3' plateau |

A real PAS-proximal rise exists but is far smaller than the published profile.

**And the same class of bug was in our code.** `count_sample` applied `abs()`;
`dienr.shape` read through `read_binned` and did not, so on a conventionally
signed minus-strand file every minus-strand gene had a negative mean, failed
the expression filter, and the profile was fitted from plus-strand genes alone
(487 of 487 minus candidates dropped). No test caught it because **every
fixture wrote positive values on both strands** -- precisely the blind spot
that hides the reference's strand bugs from its own tests, reproduced in our
suite. Found only by running real-convention data through it. The fix is
structural rather than a second `abs()` call: both the sign and the
short-window padding now live in `read_binned`, at the I/O boundary, so there
is one reader and the corrections are not optional. Two readers, one of which
forgets a correction, is the shape of the defect.


### 18.8 Is a shape profile worth applying? Mostly not

`scripts/validate_shape_usefulness.py`, five 1500-locus simulations, scored on
multi-model loci only (the profile is pure loss where there is nothing to
apportion), paired at the same locus and likelihood, calibrated within the
stratum. Built to be capable of returning "no", and it does.

| what | result | replicates? |
|---|---|---|
| dominant isoform: share of *discordant* loci shape gets right | 0.53 (Poisson), 0.50 (legacy_log) | **wash** |
| magnitude: fraction of loci where shape has lower error | 0.50, 0.50 | **wash** |
| within-locus ranking: mean per-locus rho | +0.022 | 3 of 5 seeds |
| pooled rho, excluding zero estimates | +0.013 | 3 of 5 seeds |
| pooled rho, including zero estimates | +0.026 | 5 of 5 seeds |
| expressed models estimated as exactly zero | 18.9% -> 17.1% | **5 of 5 seeds** |

The last two rows are one finding. Zero estimates are a tie block at the bottom
of a ranking, so removing some raises rank correlation by itself -- which is why
the pooled gain halves when zeros are excluded and loses sign stability. **The
"improves ranking" claim in 18.6 was substantially the zero-reduction effect.**

**The proposed mechanism does not survive either.** Conditioning genuinely
improves -- median condition number at multi-model loci 7.44 -> 4.98, 90th
percentile 39.0 -> 17.6 -- and the natural story was that this breaks ties
between near-collinear overlapping isoforms. That predicts a *within-locus*
gain, and the within-locus measurements are exactly the ones that flip sign
between seeds. Better conditioning is real and does not cash out as accuracy.

**It does not improve dominant-isoform identification, the paper's headline use
case.** Of four arms, the best dominant accuracy is `binary/legacy_log` --
DENR's default with no profile at all (0.761 vs 0.750 shaped, 0.733
binary/poisson, 0.737 shape/poisson).
**Does not reproduce on the recorded seeds -- see 18.11:** there
`shape/legacy_log` edges it, 0.761 vs 0.757, a gap well inside seed-to-seed
noise; the wash verdict stands.

So the defensible uses are narrow:

1. **Fewer spurious zeros** (5 of 5 seeds, ~9% relative). A zero is a
   qualitative claim that an isoform is off, so this is worth something.
2. **The profile as a measurement.** A pausing index, a 5'/3' ratio, or a
   comparison of profiles across samples is a readout in its own right and
   needs no design reweighting -- fit it, inspect `profile.table()`, stop.
   This is what the paper's Supplementary Figure S5 is, and it is probably the
   most defensible use of the feature.
3. **As a prerequisite, not a benefit.** A graded design is what makes the
   likelihood choice matter at all (18.6), but `shape/poisson` does not beat
   `binary/poisson`, so the enabling does not cash out.

**Guidance: fit a profile and inspect it; be skeptical about applying it.** If
applied, apply with `poisson` and only at multi-model loci. The feature exists
because the published method specifies it and because the reference's
implementation is broken in ways worth documenting -- not because it has been
shown to improve estimates.

A caveat that cuts the other way: `nascentRNASim` gives every simulated isoform
one of 62 archetype profiles, so simulated shape heterogeneity is bounded by
the archetype set. Real genes may vary more (weakening the case further) or the
archetypes may understate how *predictable* shape is within a cell type
(strengthening it). Real data would settle it; see 18.6 for the accessions.


### 18.9 Could *any* density model help? The oracle ceiling

18.8 left one door open, and the DENR paper's Discussion points at it: a
per-gene density model fitted inside the likelihood instead of one shared
profile. Before building one, `scripts/oracle_shape_ceiling.py` asks what
bounds it -- how much better deconvolution gets if the design knows every
isoform's **true** expected density. No model estimated from data can beat that.

**The oracle is exact, not approximate.** nascentRNASim gives each simulated
transcript exactly the width of one archetype (the 62 archetype widths are
unique) and generates its sense signal by resampling that archetype's counts,
so `E[count at p] = score * L * sense_arch[p] / abundance_estimate_arch` for one
library constant `L` (`simulate_data`). `scripts/export_archetypes.R` exports
the vectors. Checked before use: on 60 isolated single-isoform loci the
observed/expected ratio was within a few percent of one `L` (0.0343) for 90% of
loci, per-bin correlation 0.975 against -0.06 with the column reversed; the
outliers each had an expressed transcript excluded from that check within the
6 kb flank.

**Corrected by 18.10.** The first version of this section scored every arm on
~200 multi-model loci per seed and concluded that the ceiling was "absent" by
the letter of the rule, that a general density model was not worth building, and
that most of the gain was run-on. All three were artifacts of a Poisson solver
bug (18.10): the exact-density oracle arms failed `invalid_mean` at ~55-65% of
loci, pairing kept only the loci where they happened to fit, and those were the
easy ones (binary accuracy 0.84 there vs 0.73 overall). The numbers below are
from the fixed solver; the old ones are kept in the last column.

Five 1500-locus seeds (5, 17, 29, 41, 53 -- the three 18.8 seeds besides 5 and 17
were never recorded, so these are new), the paper's masking, Poisson unless
stated, every arm scored on the same **~790 multi-model loci per seed** (>= 2
models, one expressed, fitted by every arm -- now all scorable loci). Means over
seeds:

| arm (what the design knows) | dominant accuracy | median abs log2 err | expressed at zero | *old dominant (~200 loci)* |
|---|---|---|---|---|
| binary/legacy_log (DENR default) | 0.731 | 0.381 | 16.5% | *0.840* |
| binary/poisson | 0.698 | 0.415 | 18.7% | *0.828* |
| shape/poisson (shared profile) | 0.715 | 0.433 | 17.1% | *0.821* |
| oracle_flat -- true per-isoform scale only | 0.697 | 0.389 | 18.6% | *0.831* |
| oracle_2level -- true head (2 kb) / body step | 0.745 | 0.336 | 14.2% | *0.855* |
| oracle_smooth -- true shape, 5-bin running mean | 0.822 | 0.261 | 6.7% | *0.886* |
| oracle -- exact expected density, annotated span | 0.877 | 0.177 | 3.7% | *0.912* |
| oracle_flank -- plus run-on / upstream signal in the locus | **0.976** | **0.042** | 2.0% | *0.978* |
| oracle/legacy_log | 0.909 | 0.170 | 2.7% | *0.945* |

At single-model loci every oracle arm reaches 0.027 median abs log2 error
(binary/poisson 0.078), confirming the construction end to end.

**The pre-registered rule says the ceiling EXISTS, as written.** Fixed before
running: `oracle/poisson` vs `binary/poisson`, paired, must win >= 60% of
discordant-dominant loci pooled and > 50% in every seed, AND show a within-locus
Spearman gain > 0 in every seed. **Pooled 0.949 (747 gained / 40 lost)**, > 0.5
every seed, rho gain +0.23 to +0.38 in every seed; lower error at 79% of loci.
(The first run "failed" the rho criterion on 9-16 loci per seed and was
reinterpreted post hoc; that reinterpretation is now moot, and the small count
was itself a symptom of the bug.)

What the ladder says (the gap from binary/poisson to oracle_flank is 0.28):

- **Knowing units alone does nothing for the dominant call** (flat 0.697 vs
  binary 0.698), and only modestly helps magnitude (error wins 58%). Truth is
  defined against a masked-mean abundance estimate, so a flat design is off by
  a per-archetype factor (CV 0.175) -- a magnitude effect, not a ranking one.
- **Body shape is about 0.18 of the gap, run-on about 0.10.** Not "most of it
  is read-through", which is what the easy subset showed.
- **A one-parameter-per-isoform density model has a small ceiling**: 2-level
  0.745, +1.5 points over DENR's default, winning 56% of discordant loci
  against it.
- **A smooth per-gene shape has a large ceiling**: 0.822 (+9 over DENR's
  default, winning 77% of discordant loci against it), and exact shape 0.877.
  The exact oracle overstates what is reachable: the simulator resamples each
  archetype's own counts, so every copy has its spikes in the same places and
  the exact column "knows the noise". Treat `oracle_smooth` as the realistic
  ceiling for body shape.
- **A shared profile captures almost none of it** (shape/poisson 0.715, 18.8).
  The body-shape ceiling is gene-specific shape, and nothing here shows that is
  estimable from the same sparse counts it would be used to deconvolve.

**Decision (revised): run-on is the cheap, estimable part -- trialed in 18.10
and it passes. Per-gene body shape has the larger ceiling but no estimator; it
is worth a design study, not code.** Caveats: the oracle shows the information
exists, not that it is estimable at the very loci where overlapping isoforms
confound it; the simulator flatters shape models (every isoform is an
archetype); about 30% of models are equivalence classes whose members have
different archetypes and got an equal-weight average column, which weakens the
oracle, so the ceiling is if anything understated. Real data before shipping
anything.


### 18.10 A fitted run-on tail -- and the solver bug that hid half the loci

`scripts/runon_trial.py`, same five seeds and masking as 18.9.

**The term.** Each binary model column gets a shared tail `r(d)` in the `d`-th
bin past its PAS (24 bins = the simulator's 6 kb flank), inside the locus
window, masked bins zero. `r` is a ratio of sums over isolated, expressed
single-isoform loci -- `sum tail_signal(d) / sum body_density`, 60-91 loci per
seed, no other transcript (either strand, +/- flank) reaching the tail window.
Training loci are single-isoform, scoring loci multi-isoform; no truth is used.

**The fitted tail is termination biology, not decay.** Mean over seeds, in body
units: 1.62 at 0.25 kb past the PAS, **rising to ~2.4 at 1.5-2 kb**, then 1.1
at 3 kb and 0.34 at 5.75 kb; total mass ~27 bins of body signal. Polymerase
slows and accumulates in the termination zone. The peak lies beyond the 4-bin
(1 kb) PAS mask, which is why masking does not remove the confusion.

**Pre-registered rule.** `runon/poisson` vs `binary/poisson`, paired: win >=
60% of discordant-dominant loci pooled and > 50% every seed, AND lower
per-locus error at > 50% of loci every seed. Recommendable only if it also
beats DENR's default on pooled win share.

| arm | dominant accuracy | median abs log2 err | expressed at zero |
|---|---|---|---|
| binary/legacy_log (DENR default) | 0.731 | 0.381 | 16.5% |
| binary/poisson | 0.697 | 0.415 | 18.7% |
| **runon/poisson (fitted tail)** | **0.833** | **0.282** | 13.6% |
| runon/legacy_log | 0.815 | 0.353 | 12.2% |
| runon_oracle/poisson (each isoform's exact tail) | 0.886 | 0.222 | 10.2% |

**Verdict: package candidate, and recommendable.** Pooled win share **0.810
(697 / 163)**, > 0.5 every seed; error wins **0.583-0.616**, > 0.5 every seed;
against DENR's default 0.761 (587 / 184), > 0.5 every seed. The fitted tail
realizes 72% of the exact tail's accuracy gain. Single-model loci are
unchanged (their tails fall outside the window).

**How the first run got it wrong.** The first run scored "fail": error wins
0.469 and 0.487 in two seeds. It was scored, like 18.9, on loci fitted by every
arm *including the exact-body oracles*, ~200 per seed. Diagnosing the
magnitude result found that the oracle arms returned `invalid_mean` at 493-575
of ~910 multi-model loci, and that on all ~790 loci the run-on arm won error
in every seed. The decision set was first narrowed to the non-oracle arms (a
post-hoc scoring change, recorded as such), then the cause was found and
fixed, after which the two sets coincide and the rule passes as originally
specified.

**The bug: Poisson's starting point was not scale-invariant.**
`_glm.moment_initialization` started each model at `M'y / M'scale`, the
M-weighted mean of `y`. That is the single-model MLE only for a 0/1 column;
for entries of size `s` it starts `~s^2` too high. Exact-density columns hold
expected reads per bin (median 226, max 1.7e4) against abundances ~0.04, so
`mu` started ~500x above the data and L-BFGS-B ran every abundance onto the
zero bound without converging -- all 493 failing loci on seed 5 had `a = 0`
for every model. scipy's L-BFGS-B fails identically from that start; rescaling
the columns to mean 1 rescues 492/493. Foreign reads (antisense signal,
neighbouring flanks) in bins where an oracle column is exactly zero explain
only 9. **Fix:** start at `sum(y over the support) / sum(scale * M)`, the
single-model Poisson MLE the docstring already described; identical for 0/1
columns. `test_fit_is_invariant_to_column_scale` fails on the old start.
**Impact:** binary designs were unaffected; 18.8's shape designs (mean ~1) move
only in the third decimal when rerun with each start; 18.9's oracle arms were
the casualty. "Failures never become zeros" held: every bad fit was
`invalid_mean` with NaN, which is how it was found.

**Where the residual error loss sits** (`scripts/runon_diagnose.py`, error-win
fraction vs binary/poisson, pooled):

- **Not a metric artifact.** Restricting the per-locus error to models positive
  in both arms (the tail rescues zeros) gives 0.606 vs 0.603.
- **Where the tail lands.** If some model's tail falls on another model's body:
  0.625 (exact tail 0.689). If it lands on nothing (14% of loci): 0.465
  (exact 0.446). **Corrected by 18.12: that is not a loss.** At those loci the
  tail is entirely clipped or masked, the two fits are bit-identical, and the
  error difference is only the per-arm calibration constant (0.5-1.2% apart).
- **Gene-specific tail size, at the extremes.** Where the shared tail is > 2x an
  isoform's true tail mass (6% of loci) error wins drop to 0.486; within 2x,
  0.678.
- **No length trend** across shortest expressed isoform.

The obvious refinement -- add the tail only where it overlaps another model's
body -- was trialed in 18.12 and is identical to this design.

**Caveats.** The simulator truncates run-on at 6 kb (archetype vectors end
there); real termination zones are often longer and gene-specific, so a shared
tail has an easier job here. No real data. Not yet a package feature: shipping
needs a tail estimator on real data (isolated-gene selection, strand handling
through `read_binned`) and a design-digest change.

### 18.11 Reproducibility: tracing the 18.8 drift

Rerunning 18.8 in November did not reproduce its table -- dominant accuracy
`binary/legacy_log` 0.757 (table 0.761), `shape/legacy_log` 0.761 (0.750),
`binary/poisson` 0.744 (0.733), `shape/poisson` 0.754 (0.737) -- even for the
arm no recent code change touches. Three causes, separated:

1. **Different simulations, mostly.** 18.8 used seeds 5, 17 and three seeds
   that were never recorded; the reruns use 5, 17, 29, 41, 53. Seeds 5 and 17
   do reproduce: regenerated with `scripts/simulate_nascentrnasim.R` on
   BioHPC, they match 18.6's recorded per-arm isoform errors and zero fractions
   to three decimals in every binary arm (the data would not). The September
   session's transcripts are on another machine, so the other three seeds are
   unrecoverable.
2. **Not code.** 18.8's script at its own commit (`4f6fcc0`, pybigtools era)
   and at HEAD give identical binary-arm results on the same data: the figwig
   switch moved nothing. The Poisson start fix (18.10) moves `shape/poisson` in
   the third decimal. Fits are bitwise deterministic run to run.
3. **The scoring was nondeterministic.** Two identical runs disagreed (seed 53
   `binary/legacy_log` 0.7852 vs 0.7877). Two tie-breaks depended on polars row
   order, which a join or `group_by` does not fix:
   - *Dominant call with tied truth.* nascentRNASim draws abundances from a
     discrete kernel grid (`rgtex` samples `dens$x`; 818 distinct values among
     2,564 expressed transcripts on seed 53), so two models can have equal
     truth; picking one by row order flipped 1 of 391 loci, exactly the 0.0025.
   - *Spearman with ties.* `argsort(argsort())` ranks tied values by row order,
     and zero estimates are a ~17-19% tie block.

   **Fixed in every harness** (`validate_shape_usefulness`,
   `benchmark_shape_accuracy`, `oracle_shape_ceiling` and so `runon_trial`,
   `validate_likelihoods`): Spearman uses average ranks
   (`scipy.stats.rankdata`), and a dominant call is correct when the top
   estimate is *a* truly dominant model (estimate ties broken by model index).
   Repeat runs are now bit-identical. 18.9 and 18.10 move only in the third
   decimal (numbers above are rescored); their verdicts are unchanged.

18.8 on the recorded seeds, rescored (multi-model loci, means of 5):

| arm | dominant | isoform rho | median abs log2 | zeros |
|---|---|---|---|---|
| binary/legacy_log | 0.757 | 0.636 | 0.381 | 16.5% |
| binary/poisson | 0.744 | 0.577 | 0.415 | 18.7% |
| shape/legacy_log | 0.761 | 0.657 | 0.412 | 15.0% |
| shape/poisson | 0.754 | 0.601 | 0.433 | 17.1% |

Paired, shape vs binary: share of discordant loci shape wins 0.54 (Poisson;
0.39-0.65 by seed) and 0.56 (legacy_log); fraction of loci with lower error
0.48 and 0.48. **The verdict stands** -- a wash on the dominant call and on
magnitude, a reproducible cut in zero estimates (Poisson 18.7% -> 17.1%,
legacy_log 16.5% -> 15.0%) -- but "the best of four arms is DENR's default" was
a seed draw: on these seeds `shape/legacy_log` is 0.004 ahead, well inside the
0.03-0.06 seed-to-seed spread. The within-locus and zero-excluded rho rows of
the 18.8 table came from analysis outside the committed script and were not
rerun. 18.6's rho column is tie-order dependent in the same way; its error and
zero columns reproduce.


### 18.12 An overlap-only tail is the same design

`scripts/runon_overlap_trial.py` was written to test the refinement 18.10
proposed: keep each model's tail only in bins where another model has unmasked
support (bin-level), plus a model-level gate as a secondary arm. The rule was
fixed before running (replace the full tail only if it still passes 18.10's
rule, wins error against the full tail in every seed with ties excluded, and
is no worse on the dominant call).

**All three designs are bit-identical at every locus**, so the rule returns
"keep the full tail" trivially (no discordant loci; 3,894 of 3,894 error
comparisons tied). The reason is structural: a locus window is the union of
its transcripts' spans, a connected component, so any bin inside it is covered
by some model. A tail bin inside the window therefore always overlaps another
model's support, unless it is masked (zero either way). The full tail already
*is* an overlap-only tail; it never adds a bin to the fit.

That also overturns the 18.10 diagnosis it came from. At the 14% of loci
labelled "tail lands on nothing", the tail is wholly outside the window or
masked: on 1,474 models there, fitted tail mass in the window is 0.0 and the
run-on and binary abundances are identical. Their 0.465 error-win rate is the
global calibration constant differing between arms (ratio 0.995-1.012 by seed)
shifting every calibrated error slightly, then compared with a strict `<`.
Counting identical fits as ties, the fitted tail's error wins are 0.62 at
overlap loci and undefined elsewhere. The one residual-loss finding that
survives is gene-specific tail size: where the shared tail is > 2x an
isoform's true tail mass, error wins are 0.486.

The lesson for paired error scoring: **a per-arm calibration constant makes
identical fits look different**. Where two arms share a fit, compare the
uncalibrated estimates, or count the locus as a tie.

### 18.13 Shape profile on top of the run-on tail: no

`scripts/shape_runon_trial.py`: `apply_shape_profile` (dienr defaults), then
18.10's fitted tail past each PAS; profile and tail both from the data alone.
Same five seeds and scoring (tie-aware; error wins exclude exact ties). Rule,
fixed before running: replace `runon/poisson` only if, paired against it, the
combination wins >= 60% of discordant-dominant loci pooled and > 50% every
seed, and wins per-locus error at > 50% of loci every seed.

| arm | dominant | median abs log2 (multi) | zeros (multi) | single-model err |
|---|---|---|---|---|
| binary/legacy_log (DENR default) | 0.730 | 0.381 | 16.5% | 0.106 |
| binary/poisson | 0.697 | 0.415 | 18.7% | 0.078 |
| shape/poisson | 0.714 | 0.433 | 17.1% | 0.185 |
| **runon/poisson** | 0.833 | **0.283** | 13.5% | **0.078** |
| shape_runon/poisson | **0.840** | 0.319 | 11.3% | 0.185 |
| shape_runon/legacy_log | 0.825 | 0.372 | 10.1% | 0.148 |

**Fails both criteria:** pooled win share 0.539 (214 / 183; 0.51-0.59 by seed),
and error wins 0.43-0.46 in every seed. Adding the profile reproduces its 18.8
signature on top of the tail: +0.7 points on the dominant call, ~16% fewer
zeros, and worse magnitudes at multi-model loci (0.283 -> 0.319) and at single-
model loci (0.078 -> 0.185). The gains do add (the profile's +1.7 alone is
roughly +0.7 on top of the tail), but the profile's share stays small and its
cost stays, because its information is the average gene's and the 18.9 ceiling
is in each gene's own shape.

**Where this leaves the density-model line.** Run-on is the one validated,
estimable component; a shared profile is not worth applying, alone or with it;
per-gene body shape has the larger ceiling and no estimator. Next steps, in
order of value:

1. **Real data before more simulation.** Every density result here comes from
   one simulator whose run-on is truncated at 6 kb and whose isoforms are
   archetypes. On public PRO-seq (K562 `GSE96869`): (a) the fitted tail's real
   extent and shape past 6 kb; (b) a truth-free test -- held-out-bin predictive
   deviance, binary vs run-on design, with contiguous blocks held out as in
   Gate B -- which asks whether the tail models real signal better; (c) as an
   orthogonal proxy, dominant-isoform agreement with matched RNA-seq or
   long-read data.
2. **Run-on as an opt-in package feature**, only if (1) holds up: a tail
   estimator (isolated-gene selection, strand handling through `read_binned`),
   a design operation that changes the digest, CLI exposure, and tests.
3. **Per-gene tail scale** for the residual loss where the shared tail is > 2x
   an isoform's own (6% of loci); only after (1), since real tails vary more.

### 18.14 Real data: the run-on tail predicts held-out K562 signal better

`scripts/k562_runon_check.py`. GEO `GSE96869` (Dukler et al. 2017 -- the data
nascentRNASim's archetypes come from), K562 0 min, two replicates, `primary`
3'-end bigWigs, hg19 (checked: `plus` holds + strand transcription on GAPDH and
MYC, `minus` holds - strand on ACTB and EEF1A1, as negative values). Ensembl
GRCh37.87 protein-coding transcripts on chr1-22 and X, contigs renamed
explicitly `N -> chrN`: 81,579 transcripts, 18,967 loci. Paper masking,
250 bp bins, `summary="sum"`, Poisson. No ground truth is needed: at
multi-model loci (>= 10 usable bins, >= 20 reads) five contiguous blocks are
held out in turn and each design is scored by Poisson deviance on the held-out
block. The run-on and binary designs fit the same bins (18.12) and the tail is
fixed before fitting, so they have the same per-locus parameters. **Tail and
profile are fitted on one replicate and scored on the other.**

**The real tail** (ratio of sums over ~845 isolated, expressed single-isoform
genes per replicate, same-strand isolation, 5 kb buffer), body-density units:

| kb past PAS | 0.25 | 1.25 | 2.25 | 3.25 | 4.25 | 5.25 | 7.25 | 10.25 | 14.25 |
|---|---|---|---|---|---|---|---|---|---|
| rep1 | 1.18 | 1.51 | 1.13 | 0.76 | 0.56 | 0.41 | 0.28 | 0.24 | 0.15 |
| rep2 | 1.04 | 1.37 | 1.02 | 0.70 | 0.52 | 0.38 | 0.26 | 0.22 | 0.15 |

Same form as the simulator's (a rise past the PAS, then decay) but flatter
(peak ~1.4x vs 2.4x; the archetypes are 62 curated, highly expressed genes)
and much longer: total mass ~28 bins of body signal, **27% of it beyond 6 kb**,
where nascentRNASim truncates, and still ~0.15 at 15 kb.

**Pre-registered rule** (fixed before running): run-on is supported only if, in
both replicates, `runon` vs `binary` has lower total held-out deviance AND
lower per-locus held-out deviance at > 50% of loci (exact ties excluded).

| scored replicate (tail from) | loci | runon vs binary: total deviance | locus wins | runon6 vs binary | shape vs binary |
|---|---|---|---|---|---|
| rep1 (rep2) | 8,358 | **-5.5%** | **0.572** | -3.2%, 0.565 | +15.9%, 0.628 |
| rep2 (rep1) | 8,264 | **-5.4%** | **0.574** | -3.2%, 0.565 | +16.0%, 0.562 |

**Passes in both replicates.** ~1,490 loci tie exactly: every tail there is
clipped or masked. Cutting the tail at the simulator's 6 kb (`runon6`) keeps
only ~58% of the gain, and the full tail beats it directly (-2.4%, 0.57 of
loci), so the length past 6 kb is real signal, not overfitting.

**The shape profile is worse in total and better at most loci** -- +16% total
held-out deviance, yet lower deviance at 56-63% of loci. The total is
dominated by high-count loci, where a shared profile's misfit of each gene's
own pause peak costs most (18.6's mechanism under a count likelihood). Its
real-data dynamic range is only ~1.5x with defaults against ~4.9x simulated,
which also limits what it could do.

**What this does and does not show.** Held-out deviance tests whether the
design models *real signal* better, not whether abundances are more accurate;
the simulator remains the only accuracy evidence (18.10). The tail may also be
absorbing unannotated downstream transcription, which is still signal a
binary design misattributes. One cell line, one lab, untreated cells.

**Next:** run-on as an opt-in package feature (tail estimator with isolated-gene
selection and strand handling through `read_binned`, a design operation that
changes the digest, CLI flags, tests), with the tail fitted to >= 15 kb rather
than the simulator's 6 kb. An orthogonal accuracy check (dominant isoform vs
matched K562 RNA-seq or long-read data) would be the next evidence after that.

### 18.15 Run-on in the package

`dienr.runon` (`RunonConfig`, `fit_runon_tail`, `apply_runon_tail`), the CLI
command `dienr runon-tail`, and `dienr build-design --runon-tail`. Design
choices, and where they differ from the analysis scripts:

- **Estimator**: the same ratio of sums, but self-contained -- it reads gene
  bodies directly instead of borrowing a design's counts. Body density excludes
  the first 250 bp (pause) and last 1 kb (PAS); the tail starts exactly at the
  PAS. Training genes need no other same-strand transcript within
  `buffer_bases` (5 kb) of their **whole span** or tail window -- stricter than
  the scripts, which checked the tail window only -- so multi-isoform genes are
  excluded outright. Identical spans are collapsed first. Opposite-strand
  genes are ignored (stranded data). Default tail 15 kb (18.14).
- **Application**: added past each model's pre-mask span, clipped to the
  window, masked bins zeroed; records `Design.runon_digest` and changes the
  design digest; the results manifest now records `shape_digest` and
  `runon_digest`. Refuses a second tail or a bin-size mismatch. Commutes with
  `apply_shape_profile` (a test asserts it).
- **Tests** (`tests/test_runon.py`, 26): recovery of a known tail; strand file
  and orientation (minus-only and plus-only training agree; swapped files
  fail); scale-freeness; isolation rules; placement on both strands;
  single-model loci untouched; **usable bins unchanged** (18.12 as a test);
  masked bins stay zero; digests and guards; a known-truth case where binary
  Poisson credits a short isoform's read-through to the long one and the
  run-on design does not; CLI fit/save/apply and errors.

**Re-validated on K562** with the packaged code (defaults), scored as in 18.14
(tail from the other replicate): held-out deviance **-6.7% / -6.8%** vs binary,
lower at **56.3% / 57.0%** of loci -- slightly better in total than the
script's tail. The stricter isolation leaves ~560 training genes (vs ~845) and
a larger tail (peak ~2.0x, mass ~43 body-bins vs ~28): the genes left are those
with no neighbour at all, whose read-through is not truncated by one.

**A bug this turned up, unrelated to run-on**: `read_annotation` let Polars
infer the GTF contig column's type, so an Ensembl GTF (contigs "1".."22", "X")
was parsed as integers and failed at "X" -- `dienr build-design` could not read
a stock Ensembl annotation. Contig is now always a string;
`test_ensembl_numeric_contig_names_stay_strings` fails on the old code. (The
analysis scripts never hit it: they parse the GTF themselves.)

### 18.16 A strand-orientation check

`dienr.strand.check_strand_orientation`, run by `dienr count` on every
sample. `dienr` takes strand from the declared file, never the sign, so a
swapped plus/minus pair was undetectable -- and a swap is the most likely
conversion error (single-end PRO-seq reads are the reverse complement of the
RNA). The check reads each locus window from both files, over loci that no
opposite-strand locus overlaps (there the opposite file legitimately holds
another gene's sense signal), capped at 2,000 loci taken evenly across the
design, and reports the fraction of gene-body signal in the declared-sense
file plus the fraction of loci where sense exceeds antisense.

Calibration (correct / deliberately swapped files):

| data | sense fraction | locus agreement |
|---|---|---|
| K562 PRO-seq rep1 (1,807 loci) | 0.876 / 0.124 | 0.748 / 0.226 |
| K562 PRO-seq rep2 (1,789 loci) | 0.879 / 0.121 | 0.750 / 0.224 |
| nascentRNASim, 5 seeds (~1,180 loci) | 0.958-0.963 / 0.037-0.042 | 0.953-0.961 / 0.039-0.046 |

Thresholds: `swapped` below 0.5 (count refuses; `--allow-strand-mismatch`
overrides), `low` below 0.7 (warning), `insufficient` under 20 informative
loci (warning). Both real-data margins are wide. The verdict and statistics
are stored in each sample's counts provenance. ~1-2 s per sample on K562.
11 tests (`tests/test_strand.py`), including that negative minus-strand files
pass and that `count` refuses a swapped sheet.

### 18.17 Per-locus termination scale and a depth gate: neither helps

`scripts/k562_pergene_trial.py`, rules committed before results were read
(`6083b53`). 18.15's packaged tail on K562, tail from the other replicate,
5-fold held-out Poisson deviance at ~8,300 multi-model loci.

**Per-locus termination scale (`runon_scaled`): fails.** Each fold fits the
run-on design on training bins, predicts the signal its tails put in an
isolated 15 kb window *downstream of the locus* (never scored), and rescales
that locus's tail by `clip((O + 20) / (E + 20), 0.25, 4)`. About 5,850 of the
loci have a usable window. The scales are broad (5th-95th percentile 0.35-3.6,
median 1.13-1.16), and using them makes the fit *worse*:

| scored (tail from) | vs `runon`: total deviance | locus wins | vs `binary` |
|---|---|---|---|
| rep1 (rep2) | +0.68% | 0.463 | -6.0% |
| rep2 (rep1) | +0.63% | 0.474 | -6.2% |

The downstream window does not predict the in-locus tail well enough to beat a
shared one: it also collects unannotated downstream transcription, and its
Poisson noise is large at most loci even with shrinkage. A per-gene tail needs
a better estimator than this, or none.

**Depth gate (`runon_gated`): no gate is chosen.** Picking the threshold on
the training replicate selects decile 0 -- apply the tail everywhere -- in both
directions, so the gated arm *is* `runon` and the rule (lower total deviance)
cannot pass. Training-replicate totals rise monotonically with the threshold:
the tail lowers held-out deviance in **every** depth decile, including the
shallowest (binary minus run-on, deciles 0-10 through 80-90: +558 ... +146k on
rep2-trained, +539 ... +109k on rep1-trained; all positive).

**Correction.** A stratification I reported in conversation after 18.15 --
"the tail slightly hurts the low-signal half (+1-2% deviance, 52% of loci)" --
was an artifact. It binned loci by the *binary* arm's own held-out deviance on
the scored replicate, i.e. it selected on the comparator's outcome: loci where
binary happens to fit unusually well are over-represented in the bottom bins,
so the challenger looks worse there by regression to the mean. Binned by an
independent depth measure (the other replicate), the effect is positive
everywhere. Same lesson as 18.10's calibration artifact: **stratify by
something measured independently of the outcome being compared.**

**Where that leaves run-on:** the shared tail, applied everywhere, stays the
recommendation. The residual per-gene variation is real (the downstream scales
are broad) but not recoverable from this signal.

### 18.18 With a run-on tail, drop the PAS mask

`scripts/k562_pasmask_trial.py`, rules written before running (job 4108105,
code at `8d0a720` plus the uncommitted script). The paper's PAS mask,
`mask_end_bins=(4, 4)`, hides 1 kb either side of every PAS because a binary
column can't explain signal there. The run-on tail now explains the part past
the PAS (its peak, ~1.4x body density at ~1.25 kb, lies inside the mask), so
the question was whether the mask still earns the bins it discards.

**Comparable scoring.** Smaller masks fit more bins, so every arm is scored on
the *same* test bins -- those usable under `(4, 4)` -- in 5 contiguous blocks,
and each arm trains on its own usable bins outside the block's span. A
smaller mask only adds training bins; the question is whether they improve
the prediction of bins every arm sees. Counts read once without a PAS mask;
TSS mask `(1, 1)` throughout; tail from the other replicate. 18,967 loci,
masked fraction 6.6% at `(4, 4)` vs 1.4% with no PAS mask.

| vs `runon_4_4` (rep1 / rep2) | total held-out deviance | locus wins |
|---|---|---|
| `runon_4_0` (unmask past the PAS only) | -0.80% / -0.64% | 0.581 / 0.576 |
| `runon_1_1` | -3.79% / -3.69% | 0.640 / 0.636 |
| **`runon_0_0`** (no PAS mask) | **-4.09% / -3.88%** | **0.641 / 0.628** |
| `binary_0_0` (no PAS mask, no tail) | -1.98% / -1.58% | 0.596 / 0.586 |

Also: `runon_4_4` vs `binary_4_4` -6.7% / -6.8% (18.15 reproduced on this
scoring), and **`binary_0_0` vs `binary_4_4` -8.5% / -8.3%, wins 0.68**.

**Decision: all three smaller masks pass in both replicates; the new
recommended PAS mask is none, `(0, 0)`.** Two readings:

- **Most of the gain is the *inside* half of the mask.** Unmasking past the
  PAS alone (`4_0`) buys under 1%; also unmasking the last 1 kb of gene body
  buys ~4%. Those bins carry abundance information about the gene that ends
  there, and masking them was costing more than the PAS pile-up it guarded
  against -- even without a tail (`binary_0_0` beats `binary_4_4` by 8%).
- **The tail still pays with no mask** (`runon_0_0` vs `binary_0_0`, -2.2% /
  -2.3%, derived from the totals above), less than its -6.8% under the paper's mask,
  because part of what it did was compensate for the bins that mask removed.

Held-out deviance scores signal prediction, not isoform calls; whether the
unmasked fit calls dominant isoforms better needs a truth set (section 20,
item 1). The R-parity default stays `(4, 4)` -- it's what R does -- but the
recommended configuration changes. The TSS mask is 18.19.

### 18.19 The TSS mask stays at one bin

`scripts/k562_tssmask_trial.py`, rules committed before running (`88657c3`;
job 4108181). 18.18's scoring, with the PAS mask at 18.18's choice, `(0, 0)`,
and every arm carrying the tail; test bins are those usable under the largest
mask, `(4, 4)`. Run on the rebuilt environment (CPython 3.13.13, polars 2.1.0,
the Poisson fast path), which all arms share.

| vs `tss_1_1` (rep1 / rep2) | total held-out deviance | locus wins |
|---|---|---|
| `tss_0_0` (no TSS mask) | +10.4% / +10.3% | 0.315 / 0.337 |
| `tss_1_0` (unmask upstream) | +0.83% / +0.75% | 0.523 / 0.535 |
| `tss_2_2` | -2.86% / -1.83% | 0.521 / **0.450** |
| `tss_4_4` | +2.22% / +3.63% | 0.436 / 0.394 |

**Decision: no arm passes; `(1, 1)` stays.** Unmasking the TSS is clearly
harmful, as expected: a 0/1 column can't explain the pause peak, and fitting
it inflates the models that start there. `tss_2_2` lowers the total on both
replicates but wins at fewer than half the loci on rep2: a gain concentrated
in a minority of (large) loci, not a better default. Unlike the PAS mask, the
TSS mask is protecting the fit from signal the model genuinely can't
represent.

**Recommended configuration from here:** run-on tail, `mask_start_bins=(1, 1)`,
no PAS mask, Poisson / quasi-Poisson, `summary="sum"`.

### 18.20 Dominant-isoform accuracy against K562 long reads: rule fails, one finding replicates

`scripts/k562_longread_accuracy.py`, rules written before running (job
4108192; the switch from ID.version to ID matching was made from match
coverage, before any fit). Truth: ENCODE K562 PacBio Sequel II long reads
(ENCSR983KDL, ENCFF302ZXZ + ENCFF448BDX pooled, 2.75M reads on *Known*
GENCODE v29 transcripts), matched to GRCh37.87 protein-coding transcripts by
Ensembl ID. Scored on transcript groups that every arm resolves. Funnel:
11,633 multi-group loci -> 7,683 with every transcript in v29 -> 3,144 with
>= 20 long reads -> 3,122 fitted by every arm -> **2,974 with a decisive
(>= 2x) top group**. Chance (1/n_groups) is 0.364.

| dominant-group accuracy | rep1 | rep2 |
|---|---|---|
| `denr_default` (PAS 4, no tail, legacy log) | 0.754 | 0.747 |
| `binary_poisson` (PAS 4, no tail) | 0.700 | 0.701 |
| `runon_4_4` (PAS 4, tail) | 0.733 | 0.704 |
| `binary_0_0` (no PAS mask, no tail) | 0.747 | 0.747 |
| `recommended` (no PAS mask, tail) | **0.778** | 0.745 |

Paired, `recommended` vs `denr_default`: rep1 wins 0.555 of discordant loci
(p = 0.006); **rep2 0.497 (p = 0.91)**. **The pre-registered rule fails**: the
recommended configuration does not beat DENR's default on both replicates.
The same pattern holds at 1.5x and 3x decisiveness. The exact-version subset
(103 loci) is too small to say anything.

What replicates, and what doesn't:

- **Dropping the PAS mask improves accuracy -- replicated.** `binary_0_0` vs
  `binary_poisson`: +4.7 points on both replicates, winning 0.745 / 0.732 of
  discordant loci (p < 1e-15). 18.18's held-out-deviance result holds up on
  accuracy, the measure that matters.
- **The run-on tail's accuracy gain does not replicate.** `runon_4_4` vs
  `binary_poisson`: +3.3 points on rep1 (p = 0.0002), +0.3 on rep2 (p = 0.73);
  `recommended` vs `binary_0_0`: +3.1 vs -0.2. It is never worse, and its
  held-out-deviance gain was consistent (18.14, 18.18), but the simulator's
  large accuracy gain (18.10, 0.697 -> 0.833) is **not confirmed on real
  data**. Why it depends on the replicate is open; the scored replicate's tail
  comes from the other one, so the rep2 arm uses the rep1-fitted tail.
- **DENR's legacy log least squares beats binary Poisson at dominant calls by
  ~5 points on both replicates**, even though it is far worse in held-out
  deviance (Gate B). The simulator showed the same ordering (18.9: DENR
  default 0.731 vs binary/poisson 0.698). Predicting signal and ranking
  isoforms are different jobs: `log(y + 1e-3)` compresses large bins, which
  seems to help rank models even as it misfits counts.
- Within-locus Spearman follows accuracy (rep1: recommended 0.628, DENR
  default 0.583; rep2: 0.589 vs 0.577).

**Consequences.** The "no PAS mask" recommendation now rests on accuracy as
well as fit. The run-on tail stays recommended for signal prediction, with
its accuracy benefit stated as unconfirmed. Poisson remains the recommendation
for fitting counts and for standard errors, but **not because it calls
dominant isoforms better** -- on that measure the reference's log objective
is at least as good. Untested and worth a pre-registered follow-up: legacy log
*with* no PAS mask (and with the tail), since the mask effect and the
objective effect may stack. Absolute accuracies are agreement with
steady-state mature RNA (long reads), shared by every arm.

### 18.21 Log objective with no PAS mask: neither candidate passes; the mask effect replicates a third time

`scripts/k562_longread_logmask.py`, rules committed before running
(`9aab6ab`; job 4108194). 18.20's truth, matching, groups and scoring; the
same 2,974 decisive loci.

| dominant-group accuracy | rep1 | rep2 |
|---|---|---|
| `denr_default` (PAS 4, no tail, log) | 0.754 | 0.747 |
| `log_0_0` (no PAS mask, no tail, log) | 0.776 | **0.769** |
| `log_runon_0_0` (no PAS mask, tail, log) | 0.744 | 0.734 |
| `recommended` (no PAS mask, tail, Poisson) | **0.778** | 0.746 |
| `binary_0_0` (no PAS mask, no tail, Poisson) | 0.747 | 0.748 |

**Decision: neither candidate passes; 18.20's conclusions stand.**
`log_0_0` beats `denr_default` on both replicates (wins 0.653 / 0.636,
p < 1e-5) but only ties `recommended` on rep1 (0.497, p = 0.91; rep2 0.548,
p = 0.011). `log_runon_0_0` is worse than `log_0_0` and loses or ties every
reference.

Readings, all replicated on both PRO-seq replicates:

- **Dropping the PAS mask helps under DENR's own objective too** (`log_0_0`
  vs `denr_default`, +2.3 / +2.2 points). With 18.18 (deviance) and 18.20
  (Poisson accuracy), the PAS mask result has now held three ways. The
  smallest change that improves on DENR is to keep everything else and drop
  `mask_end_bins`.
- **The log objective beats Poisson at dominant calls with no PAS mask as
  well** (`log_0_0` vs `binary_0_0`, +3.0 / +2.2, p <= 0.002).
- **The run-on tail hurts under the log objective** (`log_runon_0_0` vs
  `log_0_0`, -3.2 / -3.5 points, p < 1e-4), where with Poisson it was +3.1 /
  -0.2 (18.20). The tail was only ever validated with Poisson
  (`docs/runon-tails.md` already said so); now there is evidence against the
  combination, not just an absence of evidence for it.

What this leaves: for dominant-isoform calls, two configurations are
statistically indistinguishable overall and both beat DENR's default on at
least one replicate -- Poisson with tail and no PAS mask (wins on rep1), and
log least squares with no tail and no PAS mask (wins on rep2, and against the
default on both). The pre-registered rule couldn't separate them, and picking
one after seeing these numbers would be the forking path the rules exist to
prevent. The recommendation stays as 18.19 states it, with the PAS-mask change
also recommended to anyone who keeps the R-parity objective.

### 18.22 The built-in defaults are the dominant-isoform recommendation

`scripts/k562_longread_defaults.py`, rules committed before running
(`be1ad63`; job 4108220). 18.20's truth and scoring, same 2,974
decisive loci.

**Correction to 18.20-18.21.** Their `denr_default` arm was not DENR's
defaults. It was DENR's objective with the *paper's* masking (TSS 1 bin, PAS
4 bins). R DENR's function defaults mask nothing
(`transcript_quantifier(mask_start_bins = NULL, mask_end_bins = NULL)`), and
dienr's defaults reproduce those exactly: 250 bp bins, no masks, `round`, log
least squares on a `mean` summary. The comparisons in 18.20-18.21 stand; only
the label was wrong. Read `denr_default` there as "DENR, paper masking".

| dominant-group accuracy | rep1 | rep2 |
|---|---|---|
| `defaults` (no masks, log, no tail) | 0.771 | 0.769 |
| `log_tss_1_1` (TSS 1 bin, no PAS mask, log) | 0.776 | 0.769 |
| `denr_paper` (TSS 1, PAS 4, log) | 0.754 | 0.747 |
| `poisson_runon` (TSS 1, no PAS mask, tail, Poisson) | 0.778 | 0.746 |

**Decision: the TSS mask does not pass** (`log_tss_1_1` vs `defaults`: wins
0.571, p = 0.17; 0.500, p = 1.0). **The dominant-isoform recommendation is the
built-in defaults, with no flags.** The defaults beat DENR with the paper's
masking on both replicates (wins 0.604 / 0.625, p <= 0.001) and match the
Poisson configuration (0.485, p = 0.47; 0.549, p = 0.010 in the defaults'
favour).

So the defaults are a sound compromise as well as DENR-faithful: nothing in
the code changes. The recommendation splits by goal -- defaults for dominant
isoforms and R parity; Poisson / quasi-Poisson with run-on tail, TSS mask
`(1, 1)` and no PAS mask for magnitudes and standard errors -- and the TSS
mask's value depends on the objective (it matters to Poisson's fit, 18.19, not
to the log objective's calls).

### 18.23 The log offset matters; the bin size barely does

`scripts/k562_longread_sweep.py`, rules committed before running (`1a4ca1c`;
job 4108226). The defaults' configuration (legacy log, `mean`, no masks, no
tail) with bin size {100, 250, 500} x log offset `c` {1e-4, 1e-3, 1e-2,
1e-1, 1}; locus grouping fixed at `distance=250` so every arm sees the same
loci. 18.20's truth and scoring. Groups are rebuilt from these 15 arms, so
the decisive set (~3,010 loci, chance 0.33) differs from 18.20-18.22's.

Dominant-group accuracy, every decisive locus (rep1 / rep2; not used for the
decision):

| bin \ c | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1 |
|---|---|---|---|---|---|
| 100 | 0.626 / 0.627 | 0.663 / 0.667 | 0.705 / 0.713 | 0.755 / 0.757 | 0.774 / 0.774 |
| 250 | 0.721 / 0.713 | **0.741 / 0.737** (default) | 0.759 / 0.752 | 0.774 / 0.771 | 0.775 / 0.776 |
| 500 | 0.762 / 0.759 | 0.767 / 0.761 | 0.767 / 0.764 | 0.766 / 0.769 | 0.763 / 0.760 |

**Decision, by the pre-registered rule: `b100_c1` passes.** Selected on PRO-seq
rep1, odd chromosomes (1677 loci; accuracy 0.773, next
`b250_c1` 0.769); confirmed on rep2, even chromosomes and X
(1332 loci): 0.779 vs the default's 0.740, winning
0.691 of discordant loci (94 vs 42), p = 9.7e-06.

Readings:

- **The offset is the active ingredient.** At 250 bp, `c` from 1e-3 to 1 is
  +3.4 / +3.9 points; at `c = 1` the three bin sizes are within ~1 point,
  and `b250_c1` vs `b100_c1` splits 74 / 71 (rep1) and 76 / 71 (rep2)
  discordant loci. The rule picked 100 bp by a selection-set margin of 0.5
  points; recommending 250 bp instead would be a post-hoc choice, so that
  needs its own confirmation if wanted.
- **Why it helps.** On the `mean` (per-base) scale, PRO-seq bins hold ~0.01-0.1;
  with `c` = 1 the transform `log(y + c)` is nearly linear, so the objective
  approaches identity least squares and empty bins stop dominating. Small `c`
  with small bins is worst (more empty bins, each weighted heavily); larger
  bins with small `c` recover much of the loss by having fewer empty bins.
  This refines 18.20's reading: what made DENR's objective rank well was the
  absence of masks, not the published offset, which is close to the worst
  setting at 250 bp.
- **Not tested here:** `--transform identity` itself, which large `c`
  approximates, and anything with the Poisson configuration.

**Consequence.** Code defaults stay DENR's (R parity). The rule's outcome
was `b100_c1`. **Adopted recommendation (decided after seeing the grid):**
the defaults plus `fit --log-offset 1`, i.e. `b250_c1` -- keeping 250 bp
bins, because at `c = 1` bin size made no material difference (74 / 71 and
76 / 71 discordant loci against `b100_c1`) and 250 bp is the default,
faster, and needs no `--distance`. That choice is post hoc: `b250_c1`'s gain
over the default (0.775 / 0.776 vs 0.741 / 0.737) is measured on the same
data that suggested it and was not separately confirmed. 18.24 compares it
with the Poisson configuration.

### 18.24 Tuned log vs the Poisson configuration: two configurations stay -- and an unexplained interaction

`scripts/k562_longread_headtohead.py`, rules committed before running
(`0e76636`; job 4108232). 18.20's truth and scoring; groups rebuilt from these
arms, 3,460 decisive loci (chance 0.317).

| dominant-group accuracy | rep1 | rep2 |
|---|---|---|
| `tuned_log` (defaults + `c = 1`) | **0.771** | **0.773** |
| `identity` (defaults, `--transform identity`) | 0.770 | 0.772 |
| `defaults` (DENR, `c = 0.001`) | 0.728 | 0.730 |
| `poisson_runon` (TSS 1, no PAS mask, tail, Poisson) | 0.631 | 0.585 |

**Decision: outcome 1, two recommended configurations stay.** `tuned_log`
beats `poisson_runon` on both replicates (wins 0.725 / 0.757, p < 1e-15).
Also: `tuned_log` vs `defaults` +4.3 / +4.3 points (wins 0.706 / 0.705) on
this larger locus set, supporting 18.23's adopted `b250_c1`; and `identity`
is indistinguishable from `tuned_log` (23 / 20 and 29 / 32 discordant), as
the near-linear transform predicts.

**The Poisson configuration's collapse needs explaining.** It scored 0.778 /
0.746 in 18.20-18.22. Those trials included a paper-masked arm, whose 4-bin
PAS mask merges isoforms with PAS sites within ~1 kb into one group, so loci
told apart mainly by nearby alternative polyadenylation were never scored.
Here they are.

**Exploratory decomposition** (`scripts/k562_longread_explore.py`, job
4108234; *not* pre-registered, changes nothing by itself), same 3,460 loci:

| Poisson variant | rep1 | rep2 |
|---|---|---|
| plain (no masks, no tail) | 0.668 | 0.673 |
| + tail | 0.706 | 0.694 |
| + TSS mask (1, 1) | 0.680 | 0.687 |
| + tail + TSS mask (the recommended abundances configuration) | **0.631** | **0.585** |
| tuned log + tail | 0.762 | 0.754 |

The tail helps Poisson alone (+3.8 / +2.1, p <= 0.02); the TSS mask helps
alone (+1.2 / +1.4); **together they hurt** (vs TSS mask alone: -4.9 / -10.2,
wins 0.43 / 0.36, p < 1e-6). An interaction this strong may be a defect in how
the tail meets a masked design rather than biology; `apply_runon_tail` adds
the tail from the pre-mask span and zeroes masked bins afterwards, which looks
right, so the next step is to inspect loci where `poisson_tss` is right and
`poisson_runon` wrong. Until that is understood, the abundances
configuration's dominant calls should not be trusted, and the 18.19 TSS-mask
choice (made on held-out deviance with the tail) is in question. The tail
also costs the tuned log objective (-0.9 / -1.9).

### 18.25 Why tail + TSS mask fails: the window clips the longest isoform's tail

`scripts/k562_tail_mask_diagnose.py` (exploratory; jobs 4108236, 4108238),
18.24's 3,460 loci. Of the loci where Poisson + TSS mask calls the dominant
group right and Poisson + TSS mask + tail calls it wrong (675 / 815,
rep1 / rep2):

- the tail fit picks a group whose PAS is **upstream** of the true group's in
  97% / 97% (median -1.43 / -1.42 kb -- where the fitted tail peaks, ~1.4x
  body density at ~1.25 kb), and with a different TSS in 97% / 96%;
- **the true group's PAS is the locus window's end in 92% / 93%** (all other
  loci: 73% / 72%), so its tail contributes **0** bins (median), while the
  picked group's tail contributes 6;
- plain Poisson got 89% / 88% of these loci right, the tuned log objective
  93% / 94%, Poisson + tail without the mask 44% / 46%.

**Mechanism.** A locus window is the union of its transcripts' spans
(18.12), so it ends at the most downstream PAS and that isoform's tail falls
entirely outside it. Every shorter isoform's tail lands inside the window,
peaking ~1.4 kb past its PAS, on the longer isoform's 3' body; the longest
isoform has no tail to answer with. The shorter isoform plus its tail then
explains the longer one's 3' signal better than the longer one can. The TSS
mask makes this decisive by removing the pause peak, the remaining evidence
for which TSS is used. 18.12 recorded "a tail never adds usable bins" as
harmless; it is an asymmetry in favour of shorter isoforms. It went unseen in
18.20-18.22 because each included a paper-masked arm whose 4-bin PAS mask
merged isoforms with nearby PAS sites, removing these loci from scoring.

**Consequence.** The run-on tail as implemented is biased toward shorter
isoforms at loci with nearby alternative PAS sites, with or without the TSS
mask (Poisson + tail alone gets 44-46% of these loci right). The fix is to
make every model's tail observable: extend each locus window downstream by
the tail length (clipped at the next same-strand locus and the contig end)
whenever a tail is applied, so the longest isoform's tail -- and the
termination pile-up past its PAS -- is fitted like the others. Until then the
abundances configuration's isoform-level results are biased, and its held-out
deviance gains (18.14, 18.18, 18.19) may partly reflect the same asymmetry.

### 18.26 Extending the windows doesn't rescue the tail: not recommended for isoform calls

`scripts/k562_longread_tailfix.py`, rules committed with the fix before
running (`cd9b0eb`; job 4108240). `apply_runon_tail` now extends every window
downstream by the tail length (stopping at the next same-strand locus), so
the longest isoform's tail is observed like the others (18.25). 18.20's truth
and scoring, the same 3,461 loci as 18.24.

| dominant-group accuracy | rep1 | rep2 |
|---|---|---|
| `tuned_log` (defaults + `c = 1`) | 0.771 | 0.773 |
| `poisson_tss` (TSS 1, no tail) | 0.680 | 0.687 |
| `runon_fixed` (TSS 1, tail, extended) | 0.629 | 0.611 |
| `runon_clipped` (TSS 1, tail, `extend=False`) | 0.631 | 0.585 |
| `tail_fixed` (no masks, tail, extended) | 0.652 | 0.647 |
| `log_tail_fixed` (tuned log, tail, extended) | 0.694 | 0.690 |

**Decisions, by the pre-registered rules:**

1. **The fix works -- fails.** `runon_fixed` vs `runon_clipped`: rep1 0.496
   of discordant loci (p = 0.83), rep2 0.555 (p = 0.002). It removes the
   18.25 asymmetry where it bites (rep2) and costs nothing elsewhere, so
   `extend=True` stays the default as the correct construction, but it does
   not make the tail help.
2. **The tail stays in the abundances configuration -- fails.**
   `poisson_tss` beats `runon_fixed` on both replicates (wins 0.582 / 0.623,
   p < 1e-15). The docs drop the tail from that configuration, which becomes
   quasi-Poisson, TSS mask `(1, 1)`, no PAS mask.
3. **Head-to-head -- outcome 1 again:** `tuned_log` beats `runon_fixed` on
   both (0.750 / 0.762). Two recommended configurations stay.

Reported: with every tail visible, **the tail lowers accuracy in every
configuration**: `tail_fixed` vs `poisson_tss` -2.8 / -4.0 (p <= 0.004);
`log_tail_fixed` vs `tuned_log` -7.7 / -8.3 (p < 1e-15). The clipped tail's
apparent gains without the TSS mask (18.24 exploratory: +3.8 / +2.1) did not
survive making it symmetric.

**Where run-on stands.** It improves the fit to real signal (held-out
deviance, 18.14, 18.18) and helped isoform calls in the DENR authors'
simulator (18.10), whose isoforms all share archetype tails; on real data a
tail shared by every gene does not help -- and with every tail visible,
hurts -- telling isoforms apart, consistent with per-gene termination varying
widely (18.17: downstream scales 0.35-3.6x). Demoted like shape profiles
(18.8): kept and documented, `build-design --runon-tail` warns, not
recommended for isoform calls. The held-out-deviance recommendations that
included it (18.18's no-PAS-mask result was measured with the tail; 18.20-
18.22 replicated it without) stand on their own evidence.

### 18.27 Scope decision: replicate DENR; everything else is experimental

November 2026, the maintainer's decision after 18.20-18.26. Measured against
an independent truth (K562 long reads), no extension improved isoform calls
enough to justify departing from DENR: shape profiles were a wash in
simulation and worse on real data (18.8, 18.14); the run-on tail fits signal
better but makes calls worse once applied symmetrically (18.26); Poisson and
quasi-Poisson fit counts better but call dominant isoforms worse (18.24,
18.26); penalties were never validated (Gate C). The one consistent gain,
`--log-offset 1` (+3.4 to +4.3 points, 18.23-18.24), is modest and rests on
one cell line.

So: **the defaults -- DENR's, with its bugs fixed -- are the only
recommendation.** Every extension stays in the code, tested and documented,
and is labelled experimental in the docs and CLI help; nothing is removed.
Improvement work stops here. What remains in scope is replicating DENR well:
correctness, bug fixes, input checks, speed, packaging. Open items that would
reopen method work -- TSS evidence from PINTS/dREG calls, a second cell line
for the long-read comparison -- are deferred, not planned.

## 19. Definition of done for version 1.0

Version 1.0 is complete when the following hold. Progress as of November 2026
is in brackets.

1. Annotation geometry and legacy fits pass frozen R parity tests. [done: 12/12 scenarios]
2. Raw BAM and count-equivalent BigWig inputs are supported and validated. [BigWig done; BAM not written]
3. Multi-sample Poisson fitting and at least ridge regularization are production-ready. [Poisson/quasi-Poisson done; ridge implemented but unvalidated -- Gate C never run, no lambda selection]
4. Exact equivalence classes and locus conditioning are primary outputs. [done]
5. Every locus has an explicit status; numerical failures never appear as zeros. [done]
6. Output schemas are versioned, documented, and round-trip tested. [done; `docs/outputs.md`]
7. Design matrices can be reused without recomputing annotation geometry. [done]
8. Performance targets are met or revised with documented benchmark evidence. [done: 14.5x wall time, 8-9x memory, section 14]
9. A complete simulation and real-data validation report is published. [simulation and one real dataset (18.14); no orthogonal real-data accuracy check yet]
10. Python API, CLI, container, and one workflow example are available. [API, CLI and workflow in `docs/getting-started.md`; no container]

## 20. Next actions

The original Phase 0 list (reference commit, golden exports, schema tests,
benchmarks, `M`/`y` prototype) is complete. Licensing metadata is still
unresolved upstream, so implementation continues from the paper and observed
behaviour. Current priorities:

1. **Orthogonal real-data accuracy for run-on**: dominant-isoform calls against
   matched K562 RNA-seq or long-read isoform data (18.14 tested signal, not
   accuracy). [done, 18.20: the rule fails; dropping the PAS mask is
   confirmed on accuracy, the tail's accuracy gain is not, and DENR's log
   objective ranks isoforms as well as Poisson. Legacy log with no PAS mask
   (18.21): neither candidate passes; the PAS-mask gain replicates under the
   log objective too, and the tail hurts with log.]
2. **Contig renaming** at the CLI (an explicit, recorded map), since Ensembl
   GTFs and UCSC BigWigs do not match out of the box. [done, November 2026:
   `dienr.contigs`, `--contig-map FILE|PRESET`. Applied to the annotation, so
   the design digest changes; recorded as `Design.contig_map` and in the
   manifest. Presets cover primary human chromosomes only -- scaffold names
   differ in ways a prefix rule gets wrong (Ensembl `GL000192.1` is UCSC
   `chr1_gl000192_random`) -- and files from
   github.com/dpryan79/ChromosomeMappings (MIT) load unmodified. `count`
   refuses when no locus contig is in the BigWigs and names a preset that
   would fix it.]
3. **BAM input: not supported, by decision (November 2026).** Converting
   reads to polymerase positions is chemistry-specific -- single-end PRO-seq
   needs the read's 5' end with strands swapped; paired-end depends on which
   mate the kit puts on the RNA 3' end; GRO-seq is fragment-based; PRO-cap
   and similar measure TSSs, not polymerase -- and a wrong choice yields files
   that still look like nascent RNA. That decision belongs to the upstream
   pipeline that knows the library, and a generic converter in dienr would
   own defaults whose failure is silent. figwig's `bam2bw` would also still
   need pysam (its `bam2bw` extra), and in figwig 0.3.0 its
   `--opposite_strand` is silently ignored without `--mate_pairs`
   (`bam2bw.py:390`; worth reporting upstream). Instead: BigWig input only,
   chemistry guidance in `docs/input-data.md`, and a **strand-orientation
   check** on every sample in `dienr count` (18.16), which catches the
   commonest conversion error whatever produced the files. Normalization
   factors (section 15) remain to be written.
4. **Gate C** before any penalty is recommended: stability under resampling,
   with a lambda-selection rule.
5. **Per-gene run-on scale**: a downstream-window estimator was tried and
   made held-out fit worse (18.17); needs a better estimator, if any. A depth
   gate is unnecessary -- the tail helps at every depth (18.17).
6. **Dependencies (November 2026).** Python floor raised to 3.12: numpy 2.5
   and scipy 1.18 require it, so a 3.11 install resolved a separate, untested
   stack (and BioHPC ships no 3.11). The `bam`, `intervals` (polars-bio) and
   `ml` (torch, anndata) extras were removed: nothing imported them, `bam`
   contradicted item 3, and polars-bio pins `polars<2.2`. If Phase 4
   TSS-evidence joins ever need polars-bio (section 7), re-add it then.
   New `compat` extra (`polars[rtcompat]`): Polars' default runtime needs
   AVX2 and dies with `Illegal instruction` on Sandy/Ivy Bridge nodes
   (cbsubscb02-12); with the extra, the full suite passes there. Polars
   moved to 2.1.0 (suite, parity 12/12 and mypy unchanged); upper bounds
   `polars<3` and `figwig<0.4` (pre-1.0, all BigWig I/O).
   **Revised the same month: floor back to 3.11.** Raising it to 3.12 was
   about an *untested* 3.11 stack, not an incompatible one. With the suite run
   on 3.11-3.14, each in an environment resolved fresh for that version
   (3.11: numpy 2.4.6, scipy 1.17.1, polars 2.1.0; job 4108228), all pass,
   and `requires-python` is `>=3.11`. 3.10 stays out: end of life October
   2026, Polars 2 requires 3.11, and `FitStatus` is an `enum.StrEnum`.
