# DIENR

Deconvolution of Isoform Expression from Nascent RNAs.

A Python reimplementation of [DENR](https://github.com/CshlSiepelLab/DENR)
(Zhao & Dukler, *Bioinformatics* 2021). Nascent RNA sequencing (PRO-seq,
GRO-seq) measures where RNA polymerase is, not which isoform it's making.
`dienr` explains the signal at each locus as a nonnegative mixture of pre-RNA
isoform models and estimates each model's abundance.

By default it reproduces R DENR exactly, with its bugs fixed. What it adds
around the method:

- **Bug fixes.** DENR's shape profile reads minus-strand genes from the
  plus-strand file, inverts negative minus-strand values and crashes at its
  own defaults on count data; an unknown contig aborts a whole run. All are
  fixed ([differences from DENR](docs/differences-from-denr.md)).
- **Explicit failures.** Every fit has a status, and failed fits are `NaN`,
  never zero.
- **Identifiability as an output.** Indistinguishable transcripts are
  collapsed into one model and mapped, never given duplicated estimates.
- **Reusable, versioned designs.** Annotation geometry is built once and
  fitted against many samples.
- **Input checks.** Swapped plus/minus BigWigs and mismatched contig names
  are caught before counting, and contigs can be renamed explicitly.
- **Speed.** About 14.5× faster than R DENR on the benchmarks, with 8–9× less
  memory, and parallel fitting.

## Quick start

```bash
uv tool install dienr
dienr build-design --gtf genes.gtf --output design/
dienr count        --design design/ --samples samples.csv --output counts/
dienr fit          --design design/ --counts counts/ --output results/
dienr validate     --results results/
```

The defaults are DENR's, and they are the recommended configuration.

## Experimental options

`dienr` also implements extensions to the method. They're available,
tested and documented, but **none is recommended**: tested against K562
long-read isoform data, none improved isoform calls enough to justify
departing from DENR (plan §18.20–18.26).

| option | what it does | evidence |
|---|---|---|
| `fit --log-offset C` | changes the 0.001 in DENR's `log(y + 0.001)` | `C = 1` called dominant isoforms 3–4 points more often (one cell line) |
| `fit --likelihood poisson` / `quasi_poisson` | count likelihoods; quasi-Poisson adds standard errors | fits held-out signal better; calls dominant isoforms worse |
| `build-design --runon-tail` | models read-through past each PAS | fits signal better; makes isoform calls worse |
| `build-design --shape-profile` | reweights models by polymerase density | no gain in simulation; worse on real data |
| `fit --penalty` | ridge / lasso / elastic net | never validated |

[Getting started](docs/getting-started.md#experimental-options) has the
details.

## Documentation

User guide (`docs/`):

- [Getting started](docs/getting-started.md): install, inputs, the four-step
  workflow, recommended settings.
- [Preparing input BigWigs](docs/input-data.md): what the files must contain,
  chemistries, converting single-end PRO-seq, the strand check.
- [Concepts](docs/concepts.md): pre-RNA models, loci, equivalence classes,
  masking, fit statuses, what an abundance means.
- [Choosing a likelihood](docs/likelihoods.md): legacy vs Poisson vs
  quasi-Poisson, and penalties.
- [Run-on tails](docs/runon-tails.md): modelling read-through past the PAS.
- [Shape profiles](docs/shape-profiles.md): measuring polymerase density along
  genes, the defects in the R implementation, and why reweighting with a
  profile isn't recommended.
- [Outputs](docs/outputs.md): every output table and file.
- [Differences from R DENR](docs/differences-from-denr.md).

To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md). For developers (`notes/`): the [implementation plan](notes/denr-python-implementation-plan.md)
is the binding design spec and the record of every validation experiment. See
[notes/README.md](notes/README.md).

## Status

The DENR pipeline is complete, with R parity in its default mode; the
experimental options above are implemented and tested. Input is BigWig only, by
design ([input data](docs/input-data.md)). Not yet written: normalization
factors, multi-sample group penalties.
Validation so far is the DENR authors' simulator plus one real dataset (K562
PRO-seq).

## Citing

If you use `dienr`, cite it ([CITATION.cff](CITATION.cff); GitHub's "Cite this
repository" reads it) and the DENR paper, whose method it reimplements:

> Zhao Y, Dukler N, Barshad G, Toneyan S, Danko CG, Siepel A. Deconvolution of
> expression for nascent RNA-sequencing data (DENR) highlights pre-RNA isoform
> diversity in human cells. *Bioinformatics* 37(24):4727–4736 (2021).
> [doi:10.1093/bioinformatics/btab582](https://doi.org/10.1093/bioinformatics/btab582)

## License

MIT © 2026 Adam Youlin He. Implemented from the DENR paper and the R package's
observed behaviour, not its source (its license metadata conflicts; see
[differences](docs/differences-from-denr.md#licensing-note)).
