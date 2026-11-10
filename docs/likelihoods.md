# Choosing a likelihood

`dienr fit --likelihood` picks the objective each locus is fitted under. Every
option estimates the same nonnegative abundances `a` in `y ≈ M a`, where `y` is
the binned signal and `M` the design matrix. They differ in how they weigh the
mismatch.

| likelihood | objective | status |
|---|---|---|
| `legacy` + `--transform log` (default) | least squares on `log(y + c)`, `c = 0.001` | **DENR's; recommended**. `--log-offset` changes `c` (experimental) |
| `legacy` + `--transform identity` | least squares on `y` | DENR's identity option |
| `poisson` | Poisson likelihood, identity link | experimental |
| `quasi_poisson` | the same fit as `poisson`, with standard errors scaled by a pooled dispersion | experimental |

The default is the recommendation; everything else is an
[experimental option](getting-started.md#experimental-options). This page
explains how they differ, for anyone who uses them.

## What the default is good and bad at

**Bad at magnitudes.** The R package's default, least squares on
`log(y + 0.001)`, is badly misspecified once zero bins are common, which is
normal for nascent RNA at 250 bp. A bin with `y = 0` contributes
`[log(0.001) − log(μ + 0.001)]²`, about 38 at `μ = 0.5`. Empty bins then
dominate and drag estimates toward zero. In held-out tests at a mean of 0.5
counts per bin, its deviance was about 3.1 against about 1.2 for every other
objective (plan §17, Gate B).

**Good at ranking.** Against K562 long-read isoform data, it called the
dominant isoform group about 5 points more often than Poisson on the same
binary design, on both PRO-seq replicates, and as often as the best Poisson
configuration (plan §18.20–18.22). The DENR authors' simulator shows the
same ordering. Fitting signal well and ranking isoforms well turn out to be
different jobs. So the log objective is a sound choice when the question is
*which* isoform dominates, and a poor one when it's *how much*.

**Its offset was never tuned, and it matters.** `log(y + c)` with the
published `c = 0.001` weighs empty bins heavily, because on the per-base
`mean` scale nascent RNA bins hold only ~0.01–0.1. Raising it with
`fit --log-offset 1` makes the transform nearly linear and improved
dominant-isoform calls by 3–4 points against K562 long reads, confirmed on
held-out data (plan §18.23). Keep `0.001` for R parity.

## Poisson and quasi-Poisson

- **They need counts.** Use `dienr count --summary sum`, on raw 3′-end
  BigWigs. `fit` refuses a count likelihood on `--summary mean` input or on
  samples declared `library_normalized`/`other_transformed`. That's the
  easiest way to misuse them, and the error says so.
- **Same abundances, different errors.** `quasi_poisson` scales Poisson
  standard errors by `sqrt(φ)`, with the dispersion `φ` pooled over every
  locus of every sample. `abundance_standard_error` is `NaN` where a Wald
  error doesn't exist: a model at the zero boundary, or a singular
  information matrix. `legacy` reports none, because a sum of squares implies
  no variance model.
- **A binary design limits what Poisson can gain.** With 0/1 columns, the
  Poisson and least-squares estimating equations differ only by a weight that
  is constant on each column's support, so for a single column they're the
  same estimator. Poisson's advantage shows up once the design is graded, and
  the [run-on tail](runon-tails.md) is a graded design that pays off in fit
  to real signal. Don't combine the tail with `legacy`: there it made
  dominant-isoform calls worse (plan §18.21).

## Penalties (unvalidated)

`--penalty ridge|lasso|elastic_net` with `--lambda1` / `--lambda2` applies to
count likelihoods. **They have not been validated:** the planned stability test
(Gate C) was never run, and there's no λ selection. Never use a fixed weight
across datasets. Relative shrinkage scales as `2λa/n_bins`, so a weight that's
negligible at abundance 1 is catastrophic at 200. An L1 penalty can't drive a
model to exactly zero without a background term, because the Poisson
likelihood diverges at the boundary.

## Negative binomial

Removed. On a binary design it can't change a point estimate at any
dispersion, and quasi-Poisson matched its standard errors to within 1–4% with
one scalar (plan §17, Gate D).
