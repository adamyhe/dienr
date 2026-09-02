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
- **`pybigtools`** for all BigWig access: header seqinfo, interval streaming via `records()`, and writing.
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
| BigWig interval streaming | not supported | `scan_bigwig` | Use `pybigtools.records()`, which returns byte-identical tuples and is already a required dependency |

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

Use `pybigtools` for all BigWig access. It is faster than the alternatives, bounded in memory by construction, and already required for header seqinfo, so it carries the signal path with no additional dependency.

The primary aggregation call is:

```python
bw.values(chrom, start, end, bins=n, summary="sum", exact=True, fillna=0)
```

**Both non-default arguments are mandatory, and each is silently wrong if omitted.**

`exact=True` is the more dangerous of the two. Without it, `values(..., bins=N)` reads precomputed zoom levels, which are lossy. On `test_double_strand_plus.bw`, 250 bp bins over `1:0-2000` have a true per-base sum of `[63, 68, 72, 77]`; the default binned call returns `[62.53, 69.53, 71.47, 76.76]`. No error is raised, and every fitted abundance is biased. Small windows can coincidentally agree — 50 bp bins over `1:0-200` return exact integers — so a small unit test will not necessarily catch it. Fix the flag in a single wrapper and test it at several bin widths against per-base ground truth.

`fillna=0` supplies count semantics. Recent `pybigtools` returns uncovered positions and empty bins as `NaN`, but in a nascent-RNA BigWig an absent interval means zero observed polymerase, not a missing observation; a leaked `NaN` propagates into the objective as a numerical failure rather than a legitimate zero. Set `oob` for reads past a contig end and `uncovered` for whether uncovered bases enter a `mean` denominator at the same boundary, and record all four in provenance.

Because `bins=` accumulates into a fixed-size array in Rust, peak memory is O(bins) regardless of input density, and no per-locus loop overhead was measurable: 2,000 separate 5 kb locus calls over a 10 Mb file completed in under 5 ms.

##### Parity oracle: independent binning in Polars

Keep a second, independently derived implementation for tests. Stream intervals with `pybigtools.records(chrom, start, end)`, clip them to the window, fan each across the bins it touches, and sum `value * overlap` per bin. This is exact by construction with no opt-in flag, and it makes the section 6 requirement that partial terminal bins carry true effective lengths an explicit column. Both implementations agreed to `max|diff| = 0.0` on the RLE and dense fixtures, which makes this a genuinely informative cross-check rather than a restatement of the same code path.

Two facts about that route, recorded because they bear on any future decision to promote it:

| Input | Intervals in | Bins out | Rows after fan-out | Peak RSS, eager | Peak RSS, streaming |
|---|---|---|---|---|---|
| RLE, 5 kb runs over 10 Mb | 2,000 | 40,000 | 40,000 (1.0×) | 44 MB | — |
| Dense per-base over 2 Mb | 2,000,000 | 8,000 | 2,000,000 (250×) | 384 MB | ~0 MB |

First, the fan-out hazard is the opposite of what run-length encoding suggests. Long constant-value runs are harmless — BigWig intervals are non-overlapping, so interval-by-bin pairs are bounded by roughly bins plus intervals, and the RLE case produced exactly one row per bin. The amplification comes from **dense per-base signal, where intervals greatly outnumber bins**, which is exactly what a PRO-seq 3′-end BigWig is. Second, that amplification is fully mitigated by never materializing the fanned frame: aggregating lazily with `.collect(engine="streaming")` gave bit-identical results with no measurable RSS growth, and faster than eager collection (70 ms versus 200 ms). Even so, `pybigtools` did the same 2 Mb aggregation in 51 ms with bounded memory and no discipline required from the caller, which is why it holds the primary path.

Still to settle in Phase 2: throughput on a real whole-genome sample, where the dense case extrapolates to order 10⁹ intervals per sample, and whether per-contig or per-chunk batching of `values()` calls beats one call per locus.

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

BigWig uncovered-region semantics must be pinned explicitly rather than inherited from library defaults. In nascent-RNA BigWigs an absent interval means zero observed polymerase, not a missing observation, but recent `pybigtools` returns uncovered positions and empty bins as `NaN` by default; `fillna=0` is required to obtain count semantics, and `oob` governs reads past a contig end. The `uncovered` argument additionally controls whether uncovered bases enter the denominator of a `mean` summary, which silently changes the fitted value. Set all three at the single I/O boundary, record them in provenance, and cover them with fixtures — a `NaN` leaking into a bin vector would otherwise propagate into the objective as a numerical failure rather than a legitimate zero.

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

## 18. Definition of done for version 1.0

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

## 19. Immediate next actions

1. Select and record the DENR reference commit and execution environment.
2. Export golden intermediate objects from the bundled synthetic tests.
3. Write the coordinate and output-schema specifications as executable tests.
4. Benchmark current DENR on one small fixture and one representative whole-genome sample set.
5. Implement a prototype that accepts an already constructed \(M\) and \(y\), then compares legacy, Poisson, and Poisson-ridge fits before building genomic I/O.
6. Resolve licensing metadata before copying or adapting implementation code.

The first go/no-go review should occur after Phase 2. At that point, the team will know whether parity is achievable, where the actual performance costs lie, and whether the new interface alone justifies continuation independent of the new statistical models.
