#!/usr/bin/env Rscript
# Export nascentRNASim archetypes in a form Python can read without R.
#
#   Rscript scripts/export_archetypes.R build/sim/sim1500_s5_archetypes.RData build/oracle
#
# `export_simulation` saves the archetype set with saveRDS (despite the .RData
# extension). For each archetype this writes its width, the abundance estimate
# that defines the simulator's truth units, and the nonzero entries of its
# sense-strand count vector. That is everything needed to reconstruct the
# expected signal of a simulated transcript exactly:
#
#   E[count at vector index i] = score * L * sense[i] / abundance_estimate
#
# for one library-wide constant L (`simulate_data`). The vector spans
# width + 2 * flank bases, from `flank` upstream of the TSS to `flank`
# downstream of the PAS, in the direction of transcription.
#
# Writes <out>/archetypes_meta.tsv and <out>/archetypes_sense.tsv.gz.

suppressMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: export_archetypes.R <archetypes.RData> <out_dir>")
ta <- readRDS(args[1])
out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

flank <- ta@configs$flank
widths <- GenomicRanges::width(ta@transcripts)
if (anyDuplicated(widths)) stop("archetype widths are not unique; the width key is ambiguous")

meta <- data.table(
  archetype = seq_along(widths),
  width = widths,
  flank = flank,
  abundance_estimate = ta@abundance_estimate,
  vector_length = vapply(ta@data$sense, length, integer(1))
)
stopifnot(all(meta$vector_length == meta$width + 2 * flank))
fwrite(meta, file.path(out, "archetypes_meta.tsv"), sep = "\t")

entries <- rbindlist(lapply(seq_along(widths), function(i) {
  v <- as.numeric(ta@data$sense[[i]])
  nz <- which(v != 0)
  data.table(archetype = i, index0 = nz - 1L, value = v[nz])
}))
fwrite(entries, file.path(out, "archetypes_sense.tsv.gz"), sep = "\t", compress = "gzip")
cat(sprintf("wrote %d archetypes (flank %d), %d nonzero sense entries to %s\n",
            nrow(meta), flank, nrow(entries), out))
