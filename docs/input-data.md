# Preparing input BigWigs

`dienr` reads **stranded BigWigs of RNA polymerase positions**: one count per
read, at the 3′ end of the nascent RNA, with plus- and minus-strand
transcription in separate files. It doesn't read BAM files. Turning reads into
polymerase positions depends on the library chemistry, and the conversion is
best done by whoever knows the library. A wrong choice doesn't fail: it
produces files that still look like nascent RNA.

## What the files must contain

- **Strand is the RNA's strand, not the read's.** Transcription on the + strand
  goes in `bigwig_plus`. Minus-strand values may be stored as negative numbers;
  `dienr` uses their magnitude and takes strand only from the column you put
  the file in.
- **One position per read, at the RNA 3′ end** (the active site), not read
  coverage. Count likelihoods (`poisson`, `quasi_poisson`) assume each count
  is one molecule.
- **Raw counts** for count likelihoods. If you scaled the files (spike-in,
  CPM), declare it in the sample sheet's `value_kind` column. Count
  likelihoods then refuse them, rather than fitting the wrong noise model.

## Chemistries

| assay | polymerase position | strand |
|---|---|---|
| PRO-seq, single-end (standard) | the read's 5′ end | **opposite** the read's alignment |
| PRO-seq, paired-end | the 5′ end of the mate that reads the RNA 3′ end (kit-dependent) | depends on the mate |
| ChRO-seq, leChRO-seq | as PRO-seq: the same biotin-NTP run-on and library design, on isolated chromatin | as PRO-seq |
| NET-seq, mNET-seq | the RNA 3′ end protected inside immunoprecipitated Pol II, read through the 3′ adapter: single-nucleotide (which mate and strand depend on the kit) | kit-dependent |
| GRO-seq | not at base resolution: the BrUTP run-on extends tens to hundreds of nucleotides and any labelled fragment is recovered, so a read locates the polymerase only to within the run-on length; fine at 250 bp bins | usually the read's (protocols vary) |
| PRO-cap, CoPRO, other TSS assays | **not polymerase position**: these measure TSSs; don't use them | |

Two cautions for NET-seq and mNET-seq. First, they measure only the Pol II
that the antibody pulls down; mNET-seq with a CTD-phospho antibody (Ser5P,
Ser2P) sees a subpopulation whose density along the gene differs from total
polymerase, so the [run-on tail](runon-tails.md) and other density
assumptions validated on PRO-seq may not carry over. Fit the tail from the same
assay, and treat it as unvalidated there. Second, they also capture the 3′
ends of splicing intermediates, which pile up at exon ends and aren't
polymerase positions.

Every one of these assays fragments the RNA. What differs is whether the read
is anchored to the active site. In PRO-seq the biotin-NTP run-on stops after
about one nucleotide, biotin enrichment keeps the fragment that carries the
nascent 3′ end, and the 3′ adapter ligates to that end, so one read end marks
the polymerase. Pipelines also differ by one base on whether that position is
the last incorporated nucleotide; at 250 bp bins it doesn't matter.

### Single-end PRO-seq with figwig

`figwig bam2bw` writes `NAME.+.bw` and `NAME.-.bw` by the read's *alignment*
strand. For single-end PRO-seq the RNA is on the other strand, so swap them
when you fill in the sample sheet:

```bash
figwig bam2bw sample.bam -s chrom.sizes -n sample      # records each read's 5' end
```

```csv
sample_id,bigwig_plus,bigwig_minus
sample,sample.-.bw,sample.+.bw
```

Note that figwig's `--opposite_strand` flag does **nothing** without
`--mate_pairs` (as of figwig 0.3.0), so it can't do this swap for you.

For paired-end data, look up which mate your kit puts on the RNA 3′ end and
check figwig's `--mate_pairs`, `--rna5` and `--opposite_strand` options. Then
confirm the result with the strand check below.

## The strand check

`dienr count` checks every sample before counting it. It reads each locus's
window from both files and measures how much of the gene-body signal is in the
file declared as that locus's strand (loci overlapping an opposite-strand gene
are left out):

```
s1: strand check: 0.876 of gene-body signal is in the declared-sense file, sense > antisense at 0.75 of 1807 loci -> ok
```

| sense fraction | verdict | what `count` does |
|---|---|---|
| ≥ 0.7 | `ok` | counts |
| 0.5 – 0.7 | `low` | counts, with a warning |
| < 0.5 | `swapped` | **refuses** (override with `--allow-strand-mismatch`) |
| fewer than 20 informative loci | `insufficient` | counts, with a warning |

Measured values: K562 PRO-seq gives 0.88 with the files the right way round
and 0.12 swapped; the DENR authors' simulator gives 0.96 and 0.04 (plan
§18.16). The result is stored in each sample's counts provenance. From Python,
it's `dienr.check_strand_orientation(design, signal)`.

## Contig names

Contig names in the BigWigs must match the annotation's. Ensembl GTFs use
`1`, `X`, `MT`; UCSC-style BigWigs use `chr1`, `chrX`, `chrM`. `dienr` never
guesses a renaming, but you can give it one with `--contig-map` on
`build-design`, `runon-tail` and `shape-profile`:

```bash
dienr build-design --gtf Homo_sapiens.GRCh38.gtf.gz --output design/ \
    --contig-map ensembl-to-ucsc
```

`--contig-map` takes a preset or a file:

- **Presets** `ensembl-to-ucsc` and `ucsc-to-ensembl` cover the primary human
  chromosomes (1–22, X, Y, MT ↔ chrM) only. Scaffold and patch names differ
  in ways no prefix rule gets right, so the presets leave them alone.
- **A file** with one `old<TAB>new` pair per line (commas also work; `#`
  starts a comment). An empty `new` means "no equivalent". The
  [ChromosomeMappings](https://github.com/dpryan79/ChromosomeMappings)
  repository has ready-made files for many assemblies and naming schemes
  (Ensembl, UCSC, GENCODE, NCBI/RefSeq), and they load as they are,
  scaffolds included:

```bash
curl -O https://raw.githubusercontent.com/dpryan79/ChromosomeMappings/master/GRCh38_ensembl2UCSC.txt
dienr build-design --gtf genes.gtf --output design/ --contig-map GRCh38_ensembl2UCSC.txt
```

`build-design` prints how many contigs and transcripts were renamed and which
contigs had no entry. The map is stored in the design and in every results
`manifest.json` (`contig_map`), and changes the design digest.

`dienr count` checks contig names before counting. If no locus's contig is in
the BigWigs it stops (override with `--allow-contig-mismatch`), and suggests a
preset when one would fix it. If only
some are missing (unplaced scaffolds, say) it warns, and those loci get
status `missing_contig`.
