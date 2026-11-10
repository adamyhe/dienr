# Run-on tails

RNA polymerase doesn't stop at the annotated polyadenylation site (PAS). It
slows down, piles up in a termination zone and is released over several
kilobases, so nascent RNA signal carries on well past every transcript's
3′ end. A plain (binary) design matrix doesn't account for that signal. Inside
a locus with nested isoforms, a short isoform's read-through lands in a longer
isoform's body and looks like evidence for the longer one.

`dienr` can estimate one shared **run-on tail** from your data and add it
past every model's PAS. The tail is `r(d)`: the expected signal `d` bins past
the PAS, as a multiple of the gene's own body density.

> **Status: experimental, not recommended.** The tail improves the fit to real signal
> (held-out deviance, plan §18.14), and in the DENR authors' simulator it
> improved isoform calls a lot (§18.10). But against K562 long-read isoform
> data it made dominant-isoform calls **worse** in every configuration tried
> once every isoform's tail was visible: Poisson with a TSS mask, 0.680 /
> 0.687 without it vs 0.629 / 0.611 with it; the log objective, 0.771 / 0.773
> vs 0.694 / 0.690 (two replicates, plan §18.26). One tail shared by every
> gene doesn't help tell isoforms apart on real data. `build-design` warns
> when it's used. It's kept for research and for modelling signal.

## When to use it

- **Not to call dominant isoforms**, with any objective (see the status
  above).
- **For modelling the signal itself**, if you need to: it explains the
  read-through past each PAS. Fit with `--likelihood poisson` (or
  `quasi_poisson`), on counts made with `--summary sum`.
- **Don't combine it with a shape profile.** The pair didn't beat the tail
  alone (plan §18.13), and on real data a shared profile made held-out fit
  worse ([shape profiles](shape-profiles.md)).
- **Don't use it with the legacy least-squares objectives.** Every
  validation used Poisson, and with DENR's log least squares the tail made
  dominant-isoform calls about 3 points *worse* against K562 long reads, on
  both replicates (plan §18.21). `dienr fit` warns about the combination.

## Usage

```bash
# 1. Fit the tail from the sample(s) you'll quantify (or a matched sample).
dienr runon-tail --gtf genes.gtf \
    --bigwig-plus sample_plus.bw --bigwig-minus sample_minus.bw \
    --output runon/

# 2. Build the design with it. The design digest changes.
#    Keep the 1-bin TSS mask but drop the PAS mask: the tail models the
#    signal around the PAS that it hid (plan 18.18).
dienr build-design --gtf genes.gtf --output design/ --runon-tail runon/ \
    --mask-start-bins 1 1

# 3. Count and fit as usual, with a count likelihood.
dienr count --design design/ --samples samples.csv --output counts/ --summary sum
dienr fit --design design/ --counts counts/ --output results/ --likelihood poisson
```

`dienr runon-tail` prints the number of training genes, the tail's total mass
and peak, and a coarse version of the curve, so you can spot an empty or
implausible tail without plotting it. It also writes `runon_tail.parquet`,
with one row per bin (`bin`, `distance_start`, `distance_end`,
`relative_density`), which you can plot.

From Python:

```python
import dienr

annotation = dienr.read_annotation("genes.gtf")
signal = (
    dienr.SignalFile(path="plus.bw", strand="+"),
    dienr.SignalFile(path="minus.bw", strand="-"),
)
tail = dienr.fit_runon_tail(annotation, signal)  # RunonConfig() defaults
design = dienr.apply_runon_tail(dienr.build_design(annotation), tail)
```

### Options

| option | default | meaning |
|---|---|---|
| `--tail-bases` | 15000 | tail length past the PAS; a multiple of `--bin-size` |
| `--bin-size` | 250 | must match the design's bin size |
| `--buffer-bases` | 5000 | a training gene must have no other same-strand transcript this close to its span or tail |
| `--min-length` | 5000 | shortest training gene |
| `--min-body-signal` | 2.0 | minimum mean summed signal per body bin (reads per bin, for count data) |
| `--threads` | 8 | reader threads |

## How it works

**Estimation.** Training genes are transcripts with no other same-strand
transcript within `--buffer-bases` of their span or tail window. That
excludes multi-isoform genes and genes with close neighbours.
Opposite-strand genes are ignored, because stranded data can't put their
reads on this strand. Each gene is read from the BigWig for its own strand,
oriented so `d = 1` is the bin next to the PAS, and its body density is
measured with the first 250 bp and the last 1 kb excluded. The tail is a
ratio of sums:

    r(d) = Σ_genes signal(d) / Σ_genes body_density

This weights genes by expression, so it stays stable under Poisson noise. It
doesn't depend on the signal's scale.

**Application.** First, each locus window is extended downstream by the
tail's length (15 kb at the defaults), in whole bins, stopping short of the
next same-strand locus. Then, for each model, `r` is added starting at the
first bin past the model's pre-mask span (`Design.model_spans`), in the
direction of transcription. Masked bins stay zero.

The extension matters. A locus window ends at its most downstream PAS, so
without it the longest isoform's tail would fall outside the window while
every shorter isoform's tail landed on the longer isoform's 3′ end. That gave
shorter isoforms an advantage they hadn't earned, and decided the wrong
dominant isoform at loci with nearby alternative PAS sites (plan §18.25).
With the extension every model's tail sits on signal that's counted.
`apply_runon_tail(..., extend=False)` reproduces the old behaviour, for
comparison only.

Two more properties:

1. **No new parameters.** The tail is fixed before fitting, so each model
   still has exactly one abundance.
2. **Provenance.** `Design.runon_digest` and the design digest record the
   tail, and the results manifest carries `runon_digest`, so a run-on fit is
   never mistaken for a plain one.

## Evidence

**Simulation** (the DENR authors' `nascentRNASim`, 5 seeds, about 790
multi-model loci each; plan §18.10):

| design | dominant-isoform accuracy | median \|log₂ error\| |
|---|---|---|
| DENR's objective (binary, `legacy_log`) | 0.730 | 0.381 |
| binary, Poisson | 0.697 | 0.415 |
| **run-on, Poisson** | **0.833** | **0.282** |
| exact per-isoform tail (oracle) | 0.886 | 0.222 |

Paired against the binary Poisson design, the run-on design called the
dominant isoform correctly at 81% of the loci where the two disagreed. It had
lower error at 58–62% of loci in every seed. Both pass the rule fixed before
the run.

**Real data** (K562 PRO-seq, GEO GSE96869, two replicates; plan §18.14).
There's no ground truth here, so the test asks which design better predicts
signal it hasn't seen. Contiguous blocks of bins at about 8,300 multi-model
loci were held out, and the tail was fitted on the *other* replicate:

| scored replicate | held-out Poisson deviance, run-on vs binary | loci where run-on is better |
|---|---|---|
| rep1 | −5.5% | 57% |
| rep2 | −5.4% | 57% |

The packaged estimator with its defaults (stricter gene isolation) does
slightly better on the same test: −6.7% and −6.8%, lower at 56% and 57% of
loci (plan §18.15).

The real tail peaks at about 1.4× body density around 1.25 kb past the PAS,
and is still about 0.15× at 15 kb. 27% of its mass lies beyond 6 kb (where
the simulator stops), and cutting the tail at 6 kb keeps only about 58% of
the improvement. That's why the default is 15 kb.

## Limits

- **One shared tail.** Real termination varies by gene. In simulation, the
  remaining loss is concentrated where the shared tail is more than twice an
  isoform's own.
- **Accuracy on real data is unconfirmed.** Held-out deviance shows the tail
  models real signal better, on both K562 replicates. Against K562 long-read
  isoform data (plan §18.20), it improved dominant-isoform calls by about 3
  points on one PRO-seq replicate and not at all on the other; it was never
  worse. The tail may also absorb unannotated downstream transcription.
- **Narrow real-data evidence:** one cell line, one lab, untreated cells.
