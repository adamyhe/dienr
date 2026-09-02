#!/usr/bin/env Rscript
#
# Time the R DENR reference on the shared benchmark dataset.
#
# Stage boundaries are chosen to match scripts/benchmark_dienr.py:
#   makeTxDbFromGFF          ~ read_annotation
#   transcript_quantifier    ~ build_design
#   add_data (per sample)    ~ count_sample
#   fit (per sample)         ~ fit
#
# One caveat is recorded in the output rather than hidden: DENR's add_data also
# computes upstream polymerase ratios and per-transcript goodness-of-fit
# metrics, which dienr does not implement. Its counting stage therefore does
# strictly more work, and the count-stage ratio overstates the difference.

suppressPackageStartupMessages({
  library(DENR)
  library(GenomicFeatures)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) {
    if (is.null(default)) stop(paste("missing required argument", flag))
    return(default)
  }
  args[[hit + 1]]
}

inputs_dir <- get_arg("--inputs")
out_path <- get_arg("--output", "")
spec <- jsonlite::fromJSON(file.path(inputs_dir, "spec.json"), simplifyDataFrame = FALSE)

txdb_from_gtf <- function(path) {
  if (requireNamespace("txdbmaker", quietly = TRUE)) {
    return(suppressWarnings(txdbmaker::makeTxDbFromGFF(path, format = "gtf")))
  }
  suppressWarnings(GenomicFeatures::makeTxDbFromGFF(path, format = "gtf"))
}

elapsed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, seconds = proc.time()[["elapsed"]] - t0)
}

timings <- list()

step <- elapsed({
  txdb <- txdb_from_gtf(spec$gtf)
  tx <- GenomicFeatures::transcripts(txdb, columns = c("tx_name", "gene_id"))
  tx$gene_id <- as.character(unlist(tx$gene_id))
  tx
})
timings$read_annotation <- step$seconds
transcripts <- step$value

step <- elapsed(suppressMessages(DENR::transcript_quantifier(
  transcripts,
  transcript_name_column = "tx_name",
  gene_name_column = "gene_id",
  bin_size = as.integer(spec$bin_size),
  mask_start_bins = as.integer(unlist(spec$mask_start_bins)),
  mask_end_bins = as.integer(unlist(spec$mask_end_bins)),
  bin_operation = spec$bin_operation
)))
timings$build_design <- step$seconds
tq_base <- step$value

count_seconds <- 0
fit_seconds <- 0
total_abundance <- 0
for (sample in spec$samples) {
  step <- elapsed(suppressMessages(DENR::add_data(
    tq = tq_base, bigwig_plus = sample$plus, bigwig_minus = sample$minus
  )))
  count_seconds <- count_seconds + step$seconds
  tq <- step$value

  step <- elapsed(suppressMessages(DENR::fit(
    tq, transform = spec$transform, heuristic_inactive_override = FALSE
  )))
  fit_seconds <- fit_seconds + step$seconds
  total_abundance <- total_abundance + sum(unlist(step$value@model_abundance))
}
timings$count_all_samples <- count_seconds
timings$fit_all_samples <- fit_seconds
timings$total <- Reduce(`+`, timings)

report <- list(
  implementation = "DENR",
  denr_version = as.character(utils::packageVersion("DENR")),
  r_version = as.character(getRversion()),
  n_transcripts = length(transcripts),
  n_loci = length(tq_base@models),
  n_models = sum(vapply(tq_base@models, ncol, integer(1))),
  n_samples = length(spec$samples),
  timings_seconds = lapply(timings, function(x) round(x, 3)),
  # No in-process RSS figure: R's gc() reports heap cells and Vcells, not
  # resident set size, and deriving MB from them gave a nonsensical result.
  # Measure externally instead:  /usr/bin/time -l Rscript ...
  total_abundance = round(total_abundance, 6),
  caveat = paste(
    "add_data also computes upstream polymerase ratios and per-transcript",
    "goodness-of-fit metrics, which dienr does not implement."
  )
)

cat(jsonlite::toJSON(report, auto_unbox = TRUE, pretty = TRUE), "\n")
if (nzchar(out_path)) write(jsonlite::toJSON(report, auto_unbox = TRUE, pretty = TRUE), out_path)
