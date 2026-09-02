# DIENR

Deconvolution of Isoform Expression from Nascent RNAs

A Python reimplementation of the [DENR](https://github.com/CshlSiepelLab/DENR)
method (Zhao & Dukler, *Bioinformatics* 2021), which explains observed nascent
RNA sequencing signal at a locus as a nonnegative mixture of pre-RNA isoform
models.

## Documentation

- [Shape profiles](docs/shape-profiles.md) — measuring relative polymerase
  density along a gene and reweighting the design by it, plus the defects in
  the R reference's implementation and what changed here.
