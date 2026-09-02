#!/usr/bin/env Rscript
# Generate a benchmark dataset with the DENR authors' own simulator.
#
#   Rscript scripts/simulate_nascentrnasim.R --n 1500 --out build/sim/sim1500
#
# This is the simulator the DENR paper used for every accuracy claim it makes:
# "Lacking an appropriate 'gold-standard' in the form of real biological data,
# we chose to benchmark the software using simulated data" (Zhao et al. 2021,
# section 3.2). It resamples read-counts from 62 manually curated PRO-seq
# archetypes -- real signal from Dukler et al. 2017, bundled with the package --
# onto simulated loci whose structure is sampled from a real annotation, with
# abundances drawn from a GTEx skeletal-muscle kernel. So the output has a known
# per-isoform truth while keeping realistic shape, overlap and depth.
#
# nascentRNASim is MIT-licensed and cloned into _reference/ (gitignored). It is
# not vendored, and nothing here is adapted from it; this script only calls its
# public API in the order its vignette prescribes.
#
# Writes <out>_plus.bw, <out>_minus.bw, <out>.bed.gz (truth in the score
# column) and <out>_archetypes.RData.

suppressMessages({
  library(nascentRNASim)
  library(GenomicRanges)
  library(rtracklayer)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[i + 1]
}
N_LOCI <- as.integer(get_arg("--n", "1500"))
OUT <- get_arg("--out", "build/sim/sim1500")
SEED <- as.integer(get_arg("--seed", "5"))
GTF <- get_arg("--gtf", "build/sim/Homo_sapiens.GRCh37.87.gtf.gz")
ARCHETYPES <- get_arg("--archetypes", "celastrol_dukler2017")

dir.create(dirname(OUT), showWarnings = FALSE, recursive = TRUE)
cat(sprintf("simulating %d loci, seed %d, archetypes '%s'\n", N_LOCI, SEED, ARCHETYPES))

## 1. Archetypes: real PRO-seq signal for 62 curated single-isoform loci.
ta <- load_archetype_set(id = ARCHETYPES)
cat(sprintf("archetypes: %d, widths %d-%d bp\n", length(ta@transcripts),
            min(width(ta@transcripts)), max(width(ta@transcripts))))

## 2. Template annotation. Only used to sample realistic locus structure --
## isoforms per locus, TSS offsets within a locus, inter-locus distances.
## Ensembl GRCh37 (release-99 serves the frozen 87 annotation), which is the
## assembly the archetype BigWigs are on.
cat("reading template annotation ...\n")
# awk rather than grep -P: BSD grep on macOS has no -P.
gtf <- fread(cmd = paste("gzip -dc", shQuote(GTF), "| awk -F'\t' '$3==\"transcript\"'"),
             sep = "\t", header = FALSE, showProgress = FALSE,
             col.names = c("seqnames", "src", "feature", "start", "end",
                           "score", "strand", "frame", "attr"))
gtf <- gtf[strand %in% c("+", "-")]
# Protein-coding autosomes plus X, as the paper's real-data analysis used.
gtf <- gtf[seqnames %in% c(as.character(1:22), "X")]
gtf <- gtf[grepl('transcript_biotype "protein_coding"', attr)]
gtf[, transcript_id := sub('.*transcript_id "([^"]+)".*', "\\1", attr)]

template <- GRanges(paste0("chr", gtf$seqnames),
                    IRanges(gtf$start, gtf$end),
                    strand = gtf$strand)
template$transcript_id <- gtf$transcript_id
# Seqlengths taken from the archetype BigWig rather than looked up over the
# network, so the run is reproducible offline and cannot silently pick up a
# different hg19 patch.
arch_dir <- system.file("extdata", "archetype_sets", ARCHETYPES,
                        package = "nascentRNASim")
bw_si <- seqinfo(BigWigFile(file.path(arch_dir, paste0(ARCHETYPES, "_plus.bw"))))
common <- intersect(seqlevels(template), seqlevels(bw_si))
template <- GenomeInfoDb::keepSeqlevels(template, common, pruning.mode = "coarse")
GenomeInfoDb::seqlevels(template) <- GenomeInfoDb::seqlevels(bw_si)
seqinfo(template) <- bw_si
cat(sprintf("template: %d protein-coding transcripts on %d contigs\n",
            length(template), length(common)))

## 3. Simulate loci, then abundances, then read-counts.
cat("simulating annotations ...\n")
sim_anno <- simulate_annotations(ta = ta, n = N_LOCI, template = template,
                                 seed = SEED)
cat(sprintf("simulated %d transcripts across %d loci\n",
            length(sim_anno), length(unique(sim_anno$gene_id))))

cat("simulating abundances from the GTEx skeletal-muscle kernel ...\n")
annotations <- simulate_abundances(annotations = sim_anno, sampling_dist = rgtex)

cat("resampling read-counts from archetypes ...\n")
# Named sim_dat deliberately: export_simulation() below reads a global of that
# name rather than its own `data` argument.
sim_dat <- simulate_data(ta, annotations = annotations, show_progress = FALSE)

## 4. Export.
cat("exporting ...\n")
export_simulation(annotations = annotations, data = sim_dat, ta = ta,
                  directory = dirname(OUT), simulation_id = basename(OUT))

truth <- as.data.table(annotations)
cat(sprintf("\nwrote %s_{plus,minus}.bw and %s.bed.gz\n", OUT, OUT))
cat(sprintf("transcripts: %d  nonzero-abundance: %d  median abundance %.3f\n",
            nrow(truth), sum(truth$score > 0), median(truth$score[truth$score > 0])))
cat(sprintf("strands: + %d, - %d\n", sum(truth$strand == "+"), sum(truth$strand == "-")))
print(summary(truth$score))
