# Python Reimplementation of DENR: Implementation Plan

Status: proposed  
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


## 19. Definition of done for version 1.0

Version 1.0 is complete when:

1. Annotation geometry and legacy fits pass frozen R parity tests.
2. Raw BAM and count-equivalent BigWig inputs are supported and validated.
3. Multi-sample Poisson fitting and at least ridge regularization are production-ready.
4. Exact equivalence classes and locus conditioning are primary outputs.
5. Every locus has an explicit status; numerical failures never appear as zeros.
6. Output schemas are versioned, documented, and round-trip tested.
7. Design matrices can be reused without recomputing annotation geometry.
8. Performance targets are met or revised with documented benchmark evidence.
9. A complete simulation and real-data validation report is published.
10. Python API, CLI, container, and one workflow example are available.

## 20. Immediate next actions

1. Select and record the DENR reference commit and execution environment.
2. Export golden intermediate objects from the bundled synthetic tests.
3. Write the coordinate and output-schema specifications as executable tests.
4. Benchmark current DENR on one small fixture and one representative whole-genome sample set.
5. Implement a prototype that accepts an already constructed \(M\) and \(y\), then compares legacy, Poisson, and Poisson-ridge fits before building genomic I/O.
6. Resolve licensing metadata before copying or adapting implementation code.

The first go/no-go review should occur after Phase 2. At that point, the team will know whether parity is achievable, where the actual performance costs lie, and whether the new interface alone justifies continuation independent of the new statistical models.
