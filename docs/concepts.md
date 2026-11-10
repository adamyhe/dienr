# Concepts

## Pre-RNA models, not transcripts

Nascent RNA covers introns, so `dienr` represents each annotated transcript as
a **pre-RNA model**: the continuous span from its TSS to its PAS, binned at
`--bin-size` (250 bp by default). Each model is a column of a per-locus design
matrix `M`, holding its fractional occupancy of each bin, rounded by
`--bin-operation`.

## Loci

Transcripts are grouped into strand-specific **loci**: connected components of
transcripts within `--distance` of one another (the bin size, by default).
Each locus is fitted independently, as `y ≈ M a` with `a ≥ 0`.

## Equivalence classes: what can actually be estimated

Two transcripts whose binned columns are identical (for example, isoforms that
differ only inside one bin) can't be told apart by any amount of data. `dienr`
collapses them into one **model** before fitting and reports:

- `model_abundance.parquet`, with one estimate per model;
- `transcript_model_map.parquet`, mapping each transcript to its model, with
  `equivalence_class_size` and `exactly_identifiable`.

**An equivalence class's abundance is never split across its transcripts.**
The R package repeated it for every member, which reads as several
independent estimates. If you need per-transcript numbers, decide how to
apportion them yourself and say so.

## Masking

Signal right at a TSS (promoter-proximal pausing) and right at a PAS
(termination) doesn't follow a model's body density. Masked bins are left out
of the fit. There are two ways to specify masking:

- `--mask-start-bins INSIDE OUTSIDE` / `--mask-end-bins INSIDE OUTSIDE`: bin
  counts, matching the R package. The paper uses `1 1` and `4 4`.
- `--mask-tss UP DOWN` / `--mask-pas UP DOWN`: bases along the direction of
  transcription, independent of bin size.

These are mutually exclusive. **A mask is a union over every transcript in a
locus**, because pausing in one isoform contaminates that bin for every
isoform through it. The cost is real: the paper's masking discards about 23%
of bins on a typical annotation. `locus_qc.masked_bin_fraction` reports it,
and `build-design` warns about loci that lose more than half their bins.

## Fit status: failures are never zeros

Every sample × locus fit gets an explicit `solver_status`:

| status | meaning |
|---|---|
| `ok` | converged normally |
| `all_zero` | no signal at the locus; the abundance is a true zero |
| `insufficient_bins` | too few unmasked bins to constrain the models |
| `missing_contig` | the locus's contig isn't in the BigWig |
| `invalid_exposure` | a non-finite or non-positive exposure |
| `rank_deficient` | the design is rank deficient after collapsing duplicates |
| `ill_conditioned` | the condition number exceeds the threshold |
| `nonconverged` | the optimizer stopped without converging |
| `invalid_mean` | a fitted mean was invalid for the likelihood |
| `numerical_error` | an unexpected numerical failure, with a reason |

Failed fits are `NaN` with a `failure_reason`, never `0`. `dienr fit` and
`dienr validate` exit nonzero when the failure fraction exceeds
`--max-failure-fraction`, so a run that quietly failed isn't mistaken for a
successful one.

## What an abundance means

It depends on the counting summary:

- `--summary mean` (the default and R parity): the mean per-base signal in
  each bin, so an abundance is a **polymerase density**. That's what DENR
  reports.
- `--summary sum`: reads per bin, so an abundance is in **reads per bin of
  model body**. Count likelihoods need this.

The two differ by exactly the bin size. Either way, `abundance_raw` is never
rescaled. Normalization is a separate step with its own table
(`normalization_factors.parquet`).

## Designs are reusable and versioned

`build-design` output is sample-independent and identified by a **design
digest**. Anything that changes what a model column predicts gives a new
digest: the annotation, bin size, masking, a [run-on tail](runon-tails.md) or a
[shape profile](shape-profiles.md). Counts record the digest they were made
against, and `fit` refuses a mismatch. Every results directory records the
annotation, design, shape and run-on digests in `manifest.json`.
