# Shape profiles

A shape profile measures how polymerase density varies *along* a gene and
reweights the design matrix by it, replacing the uniform-density assumption
behind a 0/1 model column.

- [Why](#why)
- [Does it work?](#does-it-work)
- [Using it](#using-it)
- [What was wrong in DENR](#what-was-wrong-in-denr)
- [Improvements beyond the fixes](#improvements-beyond-the-fixes)
- [Caveats](#caveats)
- [Reproducing the findings](#reproducing-the-findings)

Reference: `dienr.shape`. Design notes and full evidence: plan §18.

## Why

Nascent RNA sequencing shows a sharp promoter-proximal pausing peak within a
few hundred bases of the TSS, a slow decline through the gene body, and often a
3′ rise near the poly(A) site. A binary design matrix asserts none of that: it
says a transcript covers a bin or it does not, so the model can only explain a
pausing peak as extra abundance spread over the whole transcript.

There is a second reason, specific to this package. With a binary `M`, the
fitted mean `μ` is constant across a model's support, and every generalized
linear model whose variance function is `V(μ)` then reduces to the *same*
estimating equation. Least squares, Poisson and NB2 agree to `0.000e+00` — this
was measured, twice, as decision Gates B and D. A shape profile is the only
thing in the pipeline that makes `M` non-binary, so it is a prerequisite for
any count likelihood to affect a point estimate at all.

How much it is worth, measured over 200 Poisson resamplings against
`legacy` identity least squares on the same data:

| design | occupancy range | Poisson RMSE | least-squares RMSE | SE ratio |
|---|---|---|---|---|
| binary | 1.00× | 0.2367 | 0.2367 | **1.000** |
| shape-reshaped | 14.30× | 0.1882 | 0.2854 | **1.516** |

## Does it work?

Measured against known truth on **`nascentRNASim`, the DENR authors' own
simulator** — the one the paper used for every accuracy claim it makes, since
section 3.2 states plainly that they lacked a real-data gold standard. It
resamples read-counts from 62 manually curated PRO-seq archetypes (real signal,
Dukler et al. 2017) onto simulated loci drawn from Ensembl GRCh37, with
abundances from a GTEx skeletal-muscle kernel, so per-isoform truth is known
while the shape stays real.

Two independent 1500-locus simulations (seeds 5 and 17; ~6600 transcripts,
~2600 expressed), the paper's masking (1 bin at the TSS, 4 at the PAS), profile
fitted on half the contigs and scored on the other half. Abundance units are
arbitrary and differ between objectives, so one global calibration constant is
removed before any error is computed — applied identically to every arm.
Reproduce with `scripts/simulate_nascentrnasim.R` then
`scripts/benchmark_shape_accuracy.py`.

| arm | isoform \|log2\| err | isoform rho | dominant isoform | zero estimates |
|---|---|---|---|---|
| binary / legacy_log (DENR default) | 0.344 / 0.271 | 0.669 / 0.555 | 0.708 / 0.585 | 0.186 / 0.166 |
| binary / legacy_identity | 0.338 / 0.332 | 0.611 / 0.514 | 0.653 / 0.512 | 0.221 / 0.190 |
| binary / poisson | 0.320 / 0.328 | 0.613 / 0.492 | 0.653 / 0.512 | 0.205 / 0.202 |
| shape / legacy_log | 0.399 / 0.287 | **0.700 / 0.634** | 0.750 / 0.561 | 0.156 / 0.135 |
| shape / legacy_identity | 0.562 / 0.384 | 0.650 / 0.566 | 0.694 / 0.610 | 0.199 / 0.178 |
| **shape / poisson** | **0.328 / 0.283** | 0.656 / 0.573 | 0.708 / 0.537 | 0.186 / 0.160 |

(seed 5 / seed 17.)

**Yes for ranking isoforms, no for calibrating their magnitudes.** Isoform-level
rank correlation improves with a shape profile in every objective and both
seeds, by 0.03–0.08 — and ordering isoforms within a locus is the deconvolution
problem. The fraction of models estimated as exactly zero drops as well
(0.186 → 0.156, 0.166 → 0.135). But the magnitude error gets *worse*,
consistently, and badly so under least squares.

**The magnitude regression has a mechanism, and it is the answer to the second
question.** A fitted profile puts a 4–5× weight on the first bin, and
unweighted least squares is dominated by the largest entries of the design, so
a gene whose own pausing peak departs from the shared average profile is fitted
mostly on the bins where that departure is largest. Poisson weights each bin by
`1/μ`, which is exactly the right correction, and the difference is stark:

| | Poisson vs identity least squares |
|---|---|
| binary design | **−1% to −5%** isoform error |
| shape-reshaped design | **−26% to −42%** isoform error |

So **a graded design is what makes the choice of likelihood matter**, by roughly
an order of magnitude — measured on the authors' own benchmark, and predicted in
advance by Gate B from the estimating equations. On a binary design the two are
nearly the same estimator; on a shaped one they are not close.

### Why magnitudes don't improve

Not a defect in the profile — the magnitude estimand is one a uniform design
already gets right.

**A magnitude is a mean, and a mean does not care about shape.** For a single
model column the least-squares estimate is `â = Σ M_b y_b / Σ M_b²`. With
`M ≡ 1` that is just `mean(y)`, so if the truth is defined as the gene's mean
density, the uniform design is unbiased *whatever the shape is*. Measured over
377 training genes, its multiplicative bias has median **1.006**. There is
nothing here for a shape profile to fix.

**Weighting by a shared profile can therefore only inject misspecification.**
With `M = p`, `â/a = ⟨g,p⟩/⟨p,p⟩` where `g` is the gene's own profile, and that
equals 1 only when `g ∝ p`. It doesn't:

| | value |
|---|---|
| shared profile at the peak, `p(0)` | 4.06 |
| **across-gene SD of `g(0)`** | **6.32** |
| median across-gene CV of `g(x)` | 0.83 |
| `p²` weight in the first 5% of the axis | **26.7%** (uniform: 5.5%) |
| bias dispersion, uniform design | SD(log2) 0.228 |
| bias dispersion, shaped design | SD(log2) **0.264** |

The peak height varies more between genes than its own average value, and `p²`
concentrates five times more weight there than a uniform design does. So the
profile weights the estimate most heavily at exactly the position where genes
disagree most. **An average-gene profile is a good estimate of a mean and a bad
model of any individual gene**, and magnitude estimation is a per-gene problem.

**What the profile does buy is conditioning, which is a different estimand.**
Overlapping isoforms are nearly collinear under a uniform design, which says
they are identical wherever they overlap, so their split is decided only by the
non-overlapping bins. A graded design breaks the tie, because each isoform's
peak sits at its own TSS — inside the other's body. At multi-model loci:

| design | median condition number | 90th percentile |
|---|---|---|
| binary | 7.44 | 39.00 |
| shape-reshaped | **4.98** | **17.61** |

And the gain is confined to exactly where that matters. Stratifying by locus
complexity, with the calibration constant taken within each stratum:

| arm | single-model loci (n=320) | | multi-model loci (n≈1300) | |
|---|---|---|---|---|
| | \|log2\| err | rho | \|log2\| err | rho |
| binary / legacy_identity | **0.076** | 0.986 | 0.452 | 0.738 |
| binary / poisson | **0.076** | 0.986 | 0.438 | 0.747 |
| shape / legacy_identity | 0.342 | 0.969 | 0.553 | **0.766** |
| shape / poisson | 0.182 | 0.985 | 0.456 | **0.773** |

**At single-model loci a shape profile is pure loss** — 4.5× worse magnitude
error under least squares, 2.4× under Poisson, and no gain in ranking, because
there is nothing to apportion. The rank-correlation improvement lives entirely
at multi-model loci. (Note the two objectives are *identical* to four decimal
places on the binary single-model stratum, which is Gate B's collapse showing
up in the data.)

This also explains the Poisson interaction exactly: the damage is `p²` leverage
on the peak, and Poisson's `1/μ` weighting downweights precisely those
high-mean bins, recovering most of it — 0.342 → 0.182 at single-model loci,
0.553 → 0.456 at multi-model ones.

Applying the profile only where it can help is the obvious consequence and is
**not yet validated**: a design that mixes binary and shaped loci has two
natural abundance scales, and the single global calibration constant this
benchmark removes cannot fairly score it. Rank correlation, which needs no
calibration, is best for that variant (0.816), but the error comparison would
need per-stratum calibration to mean anything.

**The practical recommendation follows: use a shape profile with `poisson`, not
with least squares.** `shape/poisson` is the only arm that improves ranking
without paying for it in magnitude — its error matches the best binary arm to
within 2.4% while gaining 0.04 in rank correlation. The paper could not reach
this conclusion because least squares was its only option.

**And the reference's `span = 0.4` is actively harmful here**, not merely
wasteful. At 0.4 the fitted profile spans 2.0× instead of 4.5×, and isoform rank
correlation falls to 0.625 — *below* the 0.673 of no shape profile at all. The
`dienr` default of 0.05 raises it to 0.700. Over-smoothing does not merely
discard the benefit; it inverts it.

**Limits of this evidence.** Two seeds, one simulator, one archetype set, one
masking scheme. Dominant-isoform accuracy moves too erratically between seeds
(+9.8 points on one, −2.4 on the other) to support any claim; the rank
correlation and the Poisson-versus-least-squares gap are the findings that
replicate. Nothing here is real data: `nascentRNASim` resamples real archetype
signal, which makes the shape realistic but still assumes each simulated
isoform inherits an archetype's profile — the very assumption a shape
correction encodes, so this benchmark is friendlier to shape modeling than
biology will be. The paper's own orthogonal comparison (matched PRO-seq and
RNA-seq, entropy decomposition) rests on dbGaP `phs002146.v1.p1`, which is
controlled access; the public K562 (`GSE96869`) and CD4+ T-cell (`GSE85337`)
PRO-seq libraries would support a real-data check but have not been run.

## Using it

Fitting a profile and applying it are separate stages, because a profile is
measured once per cell type or protocol and reused. Reshaping is part of design
geometry, so it happens in `build-design` and produces a new design digest.

```bash
# 1. Measure the curve from isolated, expressed genes.
dienr shape-profile \
    --gtf gencode.gtf \
    --bigwig-plus sample_plus.bw \
    --bigwig-minus sample_minus.bw \
    --output profile/

# 2. Build a design that carries it.
dienr build-design --gtf gencode.gtf --output design/ --shape-profile profile/

# 3. Count and fit as usual.
dienr count --design design/ --samples samples.csv --output counts/
dienr fit --design design/ --counts counts/ --output results/
```

`shape-profile` prints the curve so a flat or inverted profile is visible
without plotting:

```
shape profile bd263cf7f9b46eed: 186 training genes, 33965 bins, dynamic range 5.59x
  genes dropped: too_short=69
  position  0.00  0.02  0.05  0.10  0.25  0.50  0.75  1.00
  density   4.66  1.68  1.41  1.30  1.27  1.00  0.88  0.83
```

and `build-design` reports how much of that grading survived masking:

```
design 8b368f2b938592ba: 255 transcripts, 255 loci, 255 models
  reshaped by profile bd263cf7f9b46eed; applied range 5.59x
```

The Python API is equivalent:

```python
import dienr

annotation = dienr.read_annotation("gencode.gtf")
signal = (
    dienr.SignalFile(path="sample_plus.bw", strand="+"),
    dienr.SignalFile(path="sample_minus.bw", strand="-"),
)

profile = dienr.fit_shape_profile(annotation, signal)
design = dienr.apply_shape_profile(dienr.build_design(annotation), profile)

profile.table()  # position / relative_density, for plotting
profile.describe()  # provenance for a manifest
profile.rejected  # why each candidate gene was excluded
dienr.applied_dynamic_range(design)
```

`ShapeConfig` holds the parameters; every CLI flag maps to one of its fields.
The ones worth knowing are `span`, `head_bases`/`tail_bases`, and the three
gene-selection filters (`min_length`, `min_locus_fraction`,
`min_covered_fraction`).

## What was wrong in DENR

All of the following was reproduced against DENR 1.0.0 at commit `09d6cde`.
The reference's shape profile fails in two independent ways: it crashes on
ordinary inputs, and when it does not crash it silently does nothing.

### It crashes

| defect | symptom | trigger |
|---|---|---|
| `rescale_fixed_width_position` assigns a zero-length sequence to `rescaled_pos[1:0]`, and R's `1:0` is the two-element vector `c(1, 0)` | `Error: replacement has length zero` | any gene short enough that `round(prop_head * length_out) == 0`, i.e. whenever the tail region is longer than the head |
| the same function with no bins | `Error: 'length.out' must be a non-negative number` | `length_out == 0` |
| `model_agreement` indexes a group table by `names(unlist(tx_grps))`, which is `"1_+.t1.1"` where the table is keyed `"1_+"` | `Error: subscript contains invalid names` | **any input `GRanges` carrying names**, e.g. after `names(gr) <- gr$tx_name` |
| documented defaults `bigwig_plus = NULL, bigwig_minus = NULL` | opaque rtracklayer error about `'filename'` | omitting either BigWig |
| **`min_percent_transcribed = 0.8` is unsatisfiable on count data** | `Error: No transcripts passed filters to be used for model fitting` | **the documented defaults, on any PRO-seq count BigWig** |

The last one is the one to know about, because it is not an edge case — it is
the default path on real data. `percent_transcribed` is the fraction of a
transcript's **bases** carrying at least one read, with the threshold
hardcoded at 1 (`transcribed_thresh <- 1`, whose own comment concedes "control
of this is not accessible to user for now"). PRO-seq 3′-end counts are sparse
per base: on the authors' own simulated dataset, **1.07% of bases have a
non-zero value** (213,377 of 20 Mb), and `percent_transcribed` peaks at 0.487
with a median of 0.069. Nothing can reach 0.8.

So `transcript_shape_profile` **fails outright on the output of the DENR
authors' own simulator, at its own default settings.** Their archetype
curation criterion — "the transcript averages one read per 20 bp" — is itself
about 5% base coverage, an order of magnitude below the filter. Lowering the
threshold walks into the next problem rather than out of it: at 0.4 exactly
**one** gene survives and a loess profile is fitted, without complaint, from a
single gene.

The names bug is the other likely one: `model_agreement` works *only* when the
annotation has no names, which is an accident of
`GenomicFeatures::transcripts()` returning unnamed ranges.

### It is silently inert

Two defects compound in the signal-extraction step:

```r
plus_grps  <- grep("+", names(final_bins))   # "+" is a REGEX QUANTIFIER
minus_grps <- grep("-", names(final_bins))
plus_counts  <- summarize_bigwig(bigwig_file = bigwig_plus, final_bins[plus_grps])
minus_counts <- lapply(summarize_bigwig(bigwig_file = bigwig_plus,   # <- plus, again
                                        final_bins[minus_grps]), rev)
```

`grep("+", ...)` matches *every* group, because `+` on its own is a quantifier
rather than a literal. And the minus branch reads `bigwig_plus` regardless, so
**`bigwig_minus` never contributes signal to the profile at all** — it is used
only in the upstream filtering step. Every minus-strand gene therefore enters
the smoother twice, both times carrying antisense signal, once in reversed
orientation.

Measured on 255 simulated genes with a 14× pausing peak and correct stranded
BigWigs:

| | curves fed to the smoother | identically zero | fitted dynamic range |
|---|---|---|---|
| DENR as written | 174 (from 116 genes) | 116 (**67%**) | **1.25×** |
| strand handling corrected | 116 | 0 | **14.63×** |

Two thirds of the training data is a constant, and the resulting profile
multiplies the design by about 1.0 everywhere. **Nothing errors. The profile is
inert, not broken** — which is the harder failure to notice, because a run that
applies it looks identical to one that works.

This could not be caught by the reference's own tests: every shape-profile test
in `test-shape_profile.R` passes the same BigWig for both strands
(`bigwig_plus = bw_1000, bigwig_minus = bw_1000`), which makes a plus/minus
swap undetectable by construction. That file also computes a reference vector
for the rescaler and never asserts it — and it is wrong, ending `0.1` where the
function returns `1`.

### The "U shape" is largely an artifact

There is a third strand defect, and together the three manufacture the shape
the paper reports. **Minus-strand BigWigs are conventionally negative** so a
browser draws them below the axis — the authors' own simulator writes them that
way (min −1122, max 0). `add_data`, the counting path, applies `abs()`.
`transcript_shape_profile` does not. Min–max rescaling of an all-negative
vector maps the *largest-magnitude* value — the 5′ pausing peak — to ≈0 and the
background to ≈1, inverting every minus-strand gene.

Two of the bugs hide each other: because the minus groups are read from
`bigwig_plus`, the minus file is never opened for signal, so the missing
`abs()` never fires — until someone fixes the strand dispatch, at which point
it does.

Decomposed on the authors' simulated data, same 57 genes throughout, `5'/3'` =
the ratio of fitted density at position 0 to position 1:

| variant | range | 5′/3′ | shape |
|---|---|---|---|
| DENR as written | 4.3× | 1.33 | **symmetric U** |
| strand fixed, still min–max | 2.6× | 1.66 | weakly 5′ |
| strand fixed, no `abs()` | 1.2× | 1.01 | flat |
| bug kept, mean-normalized | 10.2× | 1.24 | **U, exaggerated** |
| **all three fixed** | 2.4× | **2.13** | **5′-peaked, mild 3′ plateau** |

The corrected profile is 5′-dominant by a factor of two with a small 3′
plateau, which is what PRO-seq looks like. DENR's is near-symmetric, with a 3′
arm about as tall as its 5′ peak. The paper is careful to say that "while we
sometimes refer to a 'U-shape correction', the 'U' shape is not assumed but is
derived from the data" — but on stranded input the 3′ arm of that U is
substantially the 5′ peak of every minus-strand gene, mirrored by the
unreversed duplicate and inverted by min–max on negative values. A genuine
PAS-proximal rise exists; it is much smaller than the published profile shows.

### Three further defects, not crashes

**Min–max rescaling manufactures data**, beyond the inversion above. Per gene, the reference computes
`(x - (min(x) - 1e-6)) / (max(x) - (min(x) - 1e-6))`. The `- 1e-6` guard maps a
gene with *no signal* to a constant 1.0 instead of dividing by zero, so absent
data becomes evidence for a flat profile — precisely the state the strand bug
puts most genes in. Min–max also makes the single largest bin, normally the
pausing spike, set the scale for the whole gene, so the body of a sharply
paused gene is compressed toward zero.

**`predict.loess` has no value outside its training hull** and returns `NA`
there, and `apply_shape_profile` evaluates it on `seq(0, 1, by = 1e-5)` and
writes the result straight into the model matrix. With the reference's own
defaults the training positions happen to span exactly 0 and 1, so no `NA`
appears — but nothing enforces that. Confined to `[0.3, 0.7]`, 60,005 of the
100,001 grid points come back `NA`, and a `NaN` weight makes the objective
non-finite so L-BFGS-B rejects the locus. The symptom is a failed *fit*, three
stages downstream of its cause.

**The defaults are internally inconsistent.** `min_transcript_length = 8e3`
sits below `linear_head_length + linear_tail_length = 15e3`, so with defaults
every gene between 8 and 15 kb takes a fallback branch that gives it **no body
positions at all** — it contributes only to `[0, 0.2]` and `[0.8, 1]`.

### What the reference gets right

`apply_shape_profile` was the prime suspect and is sound. It averages member
transcripts' coordinates into an "expected" bin range and then deletes the
entries that rounding or masking removed, keeping two index spaces aligned by
hand:

```r
omit <- -which(models[expected_range[1]:expected_range[2], m] == 0)
models[non_z, m] <- pre_profile[lookup_ind][omit]
```

That looks fragile and is not. 700 fuzzed annotations — dense isoform clusters,
both strands, all three `bin_operation` values, start and end masking —
produced no crash and no misalignment when checked against a monotone profile.
`ceiling((d+1)/b) == floor(d/b)+1` makes the expected start exactly the bin
containing the transcript start, and columns collapse together only when their
rounded patterns already agree, which bounds the averaging error to the bins
that `omit` then removes.

`dienr` does not copy the approach, because it does not need it: collapsing to
equivalence classes first means a model column *has* one geometry.

## Improvements beyond the fixes

### Strand is data

Each gene is read from the file matching its own strand, and minus-strand bins
are reversed so that position 0 is the TSS. There is no name matching anywhere.
`fit_shape_profile` requires a file for every strand its training genes lie on
and rejects duplicates.

### Degenerate genes are dropped and counted

Relative density is the gene's signal divided by its own **mean**, which has
expectation 1 and puts genes of any expression level on the same footing. A
gene with no signal fails the coverage filter and is dropped, never rescued
into a constant. `ShapeProfile.rejected` tallies why each candidate was
excluded, so a heavily filtered run is visible rather than merely small.

### Position rescaling is total on `[0, 1]`

`rescale_position` is guaranteed, for any positive bin count and any head/tail
ratio, to return an ascending vector of exactly that length, starting at 0.0
and — for two or more bins — ending at 1.0. It never raises. Genes shorter than
head plus tail are rejected by default rather than silently squeezed;
`allow_short_genes` opts in.

`min_length` defaults to `head_bases + tail_bases` and is validated against it,
so the reference's inconsistent-defaults trap cannot be reproduced by accident.

### NaN is unreachable, not guarded

The smoother is a tricube-weighted local linear regression evaluated *at* fixed
grid nodes spanning `[0, 1]`, so there is no region where it has no value.
`ShapeProfile.evaluate` interpolates between nodes and clamps outside, and
`PROFILE_FLOOR` keeps the curve strictly positive — a profile value of exactly
zero would silently unassign a bin from a transcript that covers it.

Observations are first aggregated onto the grid. This is exactly equivalent to
weighting each distinct position by its multiplicity (the least-squares design
depends on `x` alone) and it bounds the cost: training positions are heavily
tied, since every gene with the same bin count contributes an identical
position vector.

### `span = 0.4` is wrong even after the bugs are fixed

A pausing peak is a few hundred bases wide, which the head rescaling maps to
well under a hundredth of the axis, while a span of 0.4 gives a bandwidth
around 0.2 — some thirty times wider than the feature it exists to measure.
Against the unsmoothed training cloud on simulated data:

| span | fitted range | value at position 0 |
|---|---|---|
| 0.4 (DENR default) | 2.38× | 1.95 |
| 0.2 | 3.00× | 2.49 |
| 0.1 | 4.00× | 3.33 |
| **0.05 (`dienr` default)** | **5.59×** | **4.66** |
| 0.02 | 8.47× | 7.08 |
| empirical, unsmoothed | 9.29× | 7.74 |

The reference default discards about three quarters of the signal even once
every bug above is fixed. `min_bandwidth` floors the neighbourhood at two head
bins, so a sparse training set cannot be fitted more finely than the bins the
profile multiplies, and `ShapeProfile.n_floored` reports where the smoother
undershot below zero and was clamped.

### Positions come from the pre-mask span

`Design.model_spans` records each model's occupied bin range *before* masking,
because masking destroys it: a TSS mask zeroes the leading bins of a column, so
the surviving nonzero span begins a mask width into the transcript. Deriving
positions from it would assign the pausing-peak weight to a bin that is really
a kilobase in — and the profile is at its steepest exactly there, so the error
would be largest where it matters most.

### Reshaping is visible

`apply_shape_profile` returns a new `Design` with `shape_digest` set and a new
`digest`, because abundances fitted against a reshaped design mean something
different. Applying a second profile raises rather than silently composing.
The source design is never mutated. Weights are *multiplied* into the existing
occupancies, which is identical to overwriting while `bin_operation` rounds to
0/1 and stays correct if a fractional design is ever used.

## Caveats

**A shape profile and TSS masking compete for the same bins.** Masking asserts
that signal near the TSS is untrustworthy; a shape profile asserts that it is
predictable and supplies its magnitude. Applying both largely cancels the
profile, because the pausing peak is both the largest feature the profile
measures and the first thing a mask removes. On 40 kb simulated genes with a
profile fitted at 14.3×:

| masking | applied range |
|---|---|
| none | 14.30× |
| `--mask-tss 0 10000` | 1.60× |
| `--mask-tss 0 20000` | 1.40× |
| `--mask-tss 0 20000` + `--mask-pas 10000 0` | 1.16× |

Masking the first 10 kb of a 40 kb gene removes 89% of the grading. This is not
an argument against either feature, but a run that enables both and expects the
profile to do work is mistaken, and from the outside it looks identical to one
that works. **The number to look at is `applied_dynamic_range(design)`, not
`ShapeProfile.dynamic_range`.** `build-design` prints it and warns below 1.2×.

**A flat profile is a warning, not an error.** Uniform density is a legitimate
finding, so `fit_shape_profile` returns the profile and warns. If the dynamic
range is near 1, check that `--bigwig-plus` and `--bigwig-minus` are the right
way round and that the training genes are expressed.

**Gene selection is geometric, not model-based.** The reference scored per-base
agreement between the annotation and observed transcription
(`percent_match`, `percent_transcribed`). `dienr` requires a training gene to
be the longest in its locus and to fill at least `min_locus_fraction` of it,
which excludes genes whose signal a neighbour would contaminate without needing
a threshold on the signal itself. `min_covered_fraction` and `min_mean_signal`
then require the gene to be expressed. Same intent, fewer knobs — but not the
same selection, so a profile is not expected to match DENR's numerically even
with every bug above fixed.

**Not validated on real data.** Everything here is measured on synthetic
signal with a known shape. `test_reshaped_design_reduces_residual_error`
asserts that a reshaped design leaves a smaller residual sum of squares than a
flat one at equal parameter count, which is a fair comparison but an
**in-sample** one — the profile was fitted on the same signal. There is no
held-out or cross-validated evidence yet, and none on real data.

**Poisson has not been re-validated against a reshaped design.** The 1.516×
figure above is an analytic prediction confirmed on one locus geometry, not a
Gate B re-run over the prespecified regimes. `scripts/validate_likelihoods.py`
still operates on binary designs.

## Reproducing the findings

Against the R reference (needs DENR 1.0.0 in the `dienr-rref` environment):

```bash
/Users/adamhe/miniforge3/envs/dienr-rref/bin/Rscript scripts/probe_r_shape_profile.R
SHAPE_PROBE_TRIALS=700 /Users/adamhe/.../Rscript scripts/probe_r_shape_profile.R
```

Every number in this document and in plan §18 comes from that script. It lives
outside the test suite because it needs the reference environment; the Python
side of the same findings is pinned as regressions in `tests/test_shape.py`,
which needs no R:

```bash
uv run pytest tests/test_shape.py -q
```

Those tests are written as the properties the reference violates — position
rescaling is total and ascending, a profile built from the correct files
differs from one built with the files swapped, support is preserved exactly,
the peak lands at the transcription start on both strands — rather than as
examples, because the reference's defects are all cases its own example-based
tests happened not to cover.
