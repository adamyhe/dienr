# Differences from R DENR

`dienr` reimplements the *published method* of DENR (Zhao & Dukler,
*Bioinformatics* 2021). By default it reproduces the R package's numbers:
locus membership, bins, masks, the reduced design matrix, binned signal,
objective values and abundances all agree across every comparable test
scenario against DENR at commit `09d6cde`. Where it departs, it's on purpose.

## Behaviour you'll notice

| | R DENR | `dienr` |
|---|---|---|
| Indistinguishable transcripts | one abundance repeated per transcript | one row per **model**, with a transcript→model map and `equivalence_class_size` |
| A failed locus fit | parameters kept, convergence info discarded | explicit `solver_status`, `NaN` abundance, iterations, gradient norm and objective retained |
| An unknown contig | aborts the whole run | that locus gets `missing_contig`; the run continues; `--contig-map` renames contigs explicitly |
| Swapped plus/minus BigWigs | fitted silently | `count` checks strand orientation and refuses a swapped pair |
| Samples | one per object | one reusable design against many samples |
| Stages | one object holds everything | `build-design` / `count` / `fit` are separate commands, so geometry is computed once |
| "TMM" | a trimmed mean of fitted abundances | not called TMM; normalization is separate and `abundance_raw` always survives |
| Likelihoods | least squares (`log` or `identity`) | the same (default), plus experimental `poisson`, `quasi_poisson` and `--log-offset` |
| Read-through past the PAS | not modelled | the same by default; an experimental [run-on tail](runon-tails.md) |
| Speed and memory | baseline | about 14.5× faster end to end and 8–9× less peak memory on the benchmarks |

## "Counts" are densities in both

R's `add_data` hardcodes a `mean` summary, so DENR abundances are mean
per-base polymerase density, not counts. `dienr count` defaults to the same
(`--summary mean`) for parity. Count likelihoods need `--summary sum`, which
gives exactly `bin_size` times R's values.

## Shape profiles

The reference's `transcript_shape_profile` crashes at its own defaults on
count data, and when it does run it's close to inert: it reads minus-strand
genes from the plus-strand file. `dienr shape-profile` fixes those defects,
but a shared profile turned out not to help much in validation. See
[shape profiles](shape-profiles.md) for both stories.

## Not implemented

- **The TSS-activity classifier.** The R package bundles a Keras CNN to mark
  inactive TSSs. `dienr` doesn't. Pin known-inactive transcripts with
  `fit --inactive-transcripts FILE` instead.
- **Gviz plotting, TxDb download and caching.** Pass a GTF instead.
- **BAM input, by design.** Converting reads to polymerase positions depends on
  the chemistry; do it upstream ([input data](input-data.md)).
- **Not yet written:** normalization factors and a multi-sample group penalty.

## Licensing note

The R reference's metadata conflicts: `LICENSE.md` says MIT and `DESCRIPTION`
says GPL (≥ 2). `dienr` (MIT) is implemented from the paper and from observed
behaviour, and doesn't copy R source.
