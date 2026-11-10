# Getting started

## Install

`dienr` is on PyPI and needs Python 3.11 or newer. With
[uv](https://docs.astral.sh/uv/), which also fetches a suitable Python if you
don't have one:

```bash
uv tool install dienr        # the dienr command, in its own environment
dienr --help
```

To use it from Python in a project, add it as a dependency instead:

```bash
uv add dienr                 # or, in an existing virtualenv: uv pip install dienr
```

The core dependencies are NumPy, SciPy, Polars, figwig (BigWig I/O),
pydantic and threadpoolctl. No R, Bioconductor or genomics interval library is
needed.

**Older CPUs.** Polars' default build needs AVX2 (Intel Haswell, 2013, or
newer) and crashes with `Illegal instruction` on older x86 machines. Install
the `compat` extra there:

```bash
uv tool install 'dienr[compat]'      # or: uv add 'dienr[compat]'
```

Polars then uses the compatibility build on every machine, which is somewhat
slower; set `POLARS_FORCE_PKG=32` to use the default build on machines that
have AVX2. `grep -c avx2 /proc/cpuinfo` prints 0 on a machine that needs it.

**On shared clusters**, the system Python is often too old; uv installs its
own, so nothing else is needed. Keep the environment on local disk if you
can: importing NumPy and SciPy from a network filesystem can take seconds per
command.

To work on `dienr` itself, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Inputs

- **An annotation** (GTF or GFF). Each transcript becomes a *pre-RNA model*,
  the continuous span from its TSS to its PAS. Exons are ignored, because
  nascent transcription covers introns.
- **Stranded signal, as two BigWigs per sample** (plus and minus strand),
  holding one count per read at the RNA polymerase position: the usual PRO-seq,
  GRO-seq or PRO-cap 3′-end tracks. Minus-strand files may store negative
  values; `dienr` takes strand from what you declare, never from the sign.
- **They must be made correctly for your chemistry**, which is where most mistakes
  happen; see [preparing input BigWigs](input-data.md). `dienr count` checks
  each sample's strand orientation and refuses one whose files look swapped.
- **Contig names must match.** An Ensembl GTF (`1`, `X`) won't match UCSC
  BigWigs (`chr1`, `chrX`). `dienr` doesn't rename contigs for you. A locus on
  a contig the BigWig lacks gets status `missing_contig` rather than a silent
  zero. Rename in the GTF beforehand (e.g. `sed 's/^/chr/'` on non-comment
  lines).

## The workflow

There are four steps, each its own command, so the annotation geometry is
computed once and reused across samples and cohorts:

```bash
# 1. Geometry: loci, bins, masks and identifiable models. Sample-independent.
dienr build-design --gtf genes.gtf --output design/

# 2. Signal: binned values per sample.
dienr count --design design/ --samples samples.csv --output counts/

# 3. Abundances.
dienr fit --design design/ --counts counts/ --output results/

# 4. Check a results directory (exits nonzero above a failure threshold).
dienr validate --results results/ --max-failure-fraction 0.05
```

With no options this is DENR's own configuration, and the recommended one;
see [recommended settings](#recommended-settings).

`samples.csv` has one row per sample:

```csv
sample_id,bigwig_plus,bigwig_minus,value_kind
ctrl_1,ctrl_1_plus.bw,ctrl_1_minus.bw,raw
ctrl_2,ctrl_2_plus.bw,ctrl_2_minus.bw,raw
```

`value_kind` is optional and defaults to `raw`. It also accepts
`spike_in_scaled`, `library_normalized` and `other_transformed`. Count
likelihoods refuse anything that isn't count-equivalent, so declare it
honestly. An optional `exposure` column supplies a per-sample scale factor.

## Recommended settings

**Use the defaults.** They are DENR's, exactly: 250 bp bins, no masking,
least squares on `log(y + 0.001)` (`fit --likelihood legacy`), and a
per-base `mean` summary. They reproduce R DENR run with its defaults, with
its bugs fixed, and nothing tried improved on them enough to replace them.

> **Note: the log offset.** The one extension with a consistent gain is
> `fit --log-offset 1`, which replaces the 0.001 in DENR's `log(y + 0.001)`.
> Against K562 long-read isoform data it called the dominant isoform 3.4–4.3
> points more often than the default (for example 0.728 → 0.771 and
> 0.730 → 0.773 on 3,460 loci, two PRO-seq replicates), about one wrong call
> in seven fixed, and it was confirmed on held-out data (0.740 → 0.779,
> p = 1e-5; plan §18.23–18.24). Why: on DENR's per-base scale bins hold
> roughly 0.01–0.1, so with 0.001 the many empty bins each pull hard toward
> zero; with 1 the transform is nearly linear, and the result is the same as
> `--transform identity`. It stays experimental because the evidence is one
> cell line and one long-read dataset, and the gain is modest. Leave the
> default for R parity.

## Experimental options

`dienr` implements several extensions to the method. They are tested and
documented, and all were evaluated against an independent truth -- K562
PRO-seq scored against K562 long-read isoform data, on about 3,000 loci where
the long reads clearly favour one isoform group (plan §18.20–18.26). None is
recommended:

| option | effect on dominant-isoform calls (two PRO-seq replicates) | other evidence |
|---|---|---|
| `fit --log-offset 1` | **+3.4 to +4.3 points** over the default 0.001 (e.g. 0.728 → 0.771); the clearest gain, but one cell line | equivalent to `--transform identity` |
| `fit --likelihood poisson` / `quasi_poisson` (with `count --summary sum`) | worse: 0.680 / 0.687 with a 1-bin TSS mask, against 0.771 / 0.773 for `--log-offset 1` | fits held-out signal better; quasi-Poisson gives standard errors ([likelihoods](likelihoods.md)) |
| `build-design --runon-tail` | worse in every configuration tried | fits read-through signal better ([run-on tails](runon-tails.md)) |
| `build-design --shape-profile` | not tested on long reads | no gain in simulation, worse held-out fit on real data ([shape profiles](shape-profiles.md)) |
| `fit --penalty` | not tested | never validated ([likelihoods](likelihoods.md#penalties-unvalidated)) |
| `build-design --mask-end-bins 4 4` (the paper's PAS mask) | **worse**: dropping it gained 2–5 points | DENR's defaults already leave it off |

Masks are a DENR option rather than an extension: the defaults mask nothing,
the paper used one TSS bin and four PAS bins, and the evidence says not to
mask the PAS.

These results come from one cell line and one lab's PRO-seq, scored against
steady-state long-read RNA, which differs from nascent transcription by
isoform stability.

## Reading the results

`results/` holds four Parquet tables and a `manifest.json`; see
[outputs](outputs.md) for the full schemas. The table you'll use most is
`model_abundance.parquet`, with one row per sample × identifiable model:

```python
import polars as pl

abundance = pl.read_parquet("results/model_abundance.parquet")
ok = abundance.filter(pl.col("solver_status") == "ok")
```

Two habits matter:

- **Estimates are per model, not per transcript.** Transcripts that the
  annotation can't distinguish at the chosen bin size share one model; see
  [concepts](concepts.md). `transcript_model_map.parquet` maps between them.
- **Filter on `solver_status`.** A failed fit is `NaN` with a reason, never a
  zero. Only `ok` (and `all_zero`, a genuine zero) are estimates.

## From Python

Every command has a Python equivalent:

```python
import dienr

annotation = dienr.read_annotation("genes.gtf")
design = dienr.build_design(annotation, bin_size=250, mask_start_bins=(1, 1), mask_end_bins=(4, 4))
signal = (
    dienr.SignalFile(path="s_plus.bw", strand="+"),
    dienr.SignalFile(path="s_minus.bw", strand="-"),
)
counts = dienr.count_sample(design, "s", signal, summary="sum")
result = dienr.fit(design, counts, likelihood="poisson")
result.model_abundance  # a Polars DataFrame
result.write("results/")
```
