#!/usr/bin/env Rscript
# Reproduces the shape-profile defects recorded in plan section 18 against the
# frozen R reference. Every number in that section comes from here.
#
#   /Users/adamhe/miniforge3/envs/dienr-rref/bin/Rscript scripts/probe_r_shape_profile.R
#
# Needs DENR 1.0.0 (commit 09d6cde) installed, which is why it lives outside
# the pytest suite: the reference environment is a separate conda env and is
# not present on every machine. The Python side of the same findings is pinned
# as regressions in tests/test_shape.py, which needs no R at all.
suppressMessages({library(DENR); library(GenomicRanges); library(rtracklayer); library(data.table)})
set.seed(42)
WORK <- file.path(tempdir(), "shape_probe")
dir.create(WORK, showWarnings = FALSE, recursive = TRUE)

banner <- function(text) { cat("\n===", text, "===\n"); flush.console() }

# The 18.4 fuzz dominates runtime (each trial builds a full transcript
#_quantifier over a megabase at 250 bp bins). Section 18 quotes a 700-trial
# run; the default here is small enough to be routinely runnable.
TRIALS <- as.integer(Sys.getenv("SHAPE_PROBE_TRIALS", "40"))

# --------------------------------------------------------------------------
# 18.1 Crashes
# --------------------------------------------------------------------------
banner("18.1a rescale_fixed_width_position: 'replacement has length zero'")
resc <- DENR:::rescale_fixed_width_position
probe <- function(label, ...) {
  r <- tryCatch(resc(...), error = function(e) paste("ERROR:", conditionMessage(e)))
  cat(sprintf("  %-38s -> %s\n", label,
      if (is.character(r)) r else paste(sprintf("%.4g", r), collapse = " ")))
}
probe("len=2 head=1000 tail=5000 bs=250", bin_size = 250, length_out = 2,
      linear_head_length = 1000, linear_tail_length = 5000)
probe("len=3 head=1000 tail=5000 bs=250", bin_size = 250, length_out = 3,
      linear_head_length = 1000, linear_tail_length = 5000)
probe("len=5 head=1000 tail=9000 bs=250", bin_size = 250, length_out = 5,
      linear_head_length = 1000, linear_tail_length = 9000)
probe("len=0 (defaults)", bin_size = 250, length_out = 0,
      linear_head_length = 10000, linear_tail_length = 5000)
cat("  and the one-bin tail places the gene's last bin at 1 - tail_percent:\n")
probe("len=4 head=10000 tail=5000 bs=250", bin_size = 250, length_out = 4,
      linear_head_length = 10000, linear_tail_length = 5000)

banner("18.1b the reference test's own unasserted expectation is wrong")
ref_1 <- c(0, 0.2, seq(0.2, 0.8, length.out = 7)[c(-1, -7)], 0.8, 0.9, 0.1)
got <- resc(bin_size = 1, length_out = 10, linear_head_length = 2,
            linear_tail_length = 3, head_percent = 0.2, tail_percent = 0.2)
cat("  ref_1 (in test-shape_profile.R, never asserted):",
    paste(sprintf("%.4g", ref_1), collapse = " "), "\n")
cat("  actual:                                        ",
    paste(sprintf("%.4g", got), collapse = " "), "\n")
cat("  equal:", isTRUE(all.equal(ref_1, got)), "\n")

# --------------------------------------------------------------------------
# 18.2 Silently inert
# --------------------------------------------------------------------------
banner("18.2a grep('+', names) is a regex and matches every group")
nms <- c("1_+", "2_+", "3_-", "4_-")
cat("  names:                      ", paste(nms, collapse = "  "), "\n")
cat("  grep('+', names):           ", paste(grep("+", nms), collapse = ","), "\n")
cat("  grep('+', names, fixed=TRUE):", paste(grep("+", nms, fixed = TRUE), collapse = ","), "\n")
cat("  grep('-', names):           ", paste(grep("-", nms), collapse = ","), "\n")

banner("18.2b every minus gene is fed to the smoother twice")
ext <- system.file("extdata", package = "DENR")
txdb <- AnnotationDbi::loadDb(file.path(ext, "test_double_strand.txdb"))
gr2 <- GenomicFeatures::transcripts(txdb, c("tx_name", "gene_id"))
msgs <- character()
withCallingHandlers(
  tsp2 <- transcript_shape_profile(
    transcripts = gr2,
    bigwig_plus = file.path(ext, "test_double_strand_plus.bw"),
    bigwig_minus = file.path(ext, "test_double_strand_minus.bw"),
    bin_size = 10, min_transcript_length = 100,
    linear_head_length = 50, linear_tail_length = 50, span = 0.4),
  message = function(m) { msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage") })
cat("  selected transcripts:", length(tsp2@transcripts),
    "(+:", sum(strand(tsp2@transcripts) == "+"),
    " -:", sum(strand(tsp2@transcripts) == "-"), ")\n")
cat("  ", grep("Fitting profile model", msgs, value = TRUE), sep = "")
cat("  -> P + 2M curves from P + M genes\n")

banner("18.1c named input GRanges kills model_agreement")
named <- gr2
names(named) <- named$tx_name
for (label in c("names set", "names NULL")) {
  g <- if (label == "names set") named else gr2
  r <- tryCatch({
    suppressMessages(DENR:::model_agreement(
      DENR:::group_transcripts(g, distance = 0),
      file.path(ext, "test_double_strand_plus.bw"),
      file.path(ext, "test_double_strand_minus.bw")))
    "ok"
  }, error = function(e) paste("ERROR:", conditionMessage(e)))
  cat(sprintf("  %-12s -> %s\n", label, r))
}
u <- unlist(DENR:::group_transcripts(named, distance = 0))
cat("  names(unlist(groups)) are", paste(head(names(u), 2), collapse = ", "),
    "-- transcript names appended to group names\n")

# --------------------------------------------------------------------------
# 18.2c the headline: profile dynamic range on realistic stranded data
# --------------------------------------------------------------------------
banner("18.2c 255 simulated genes: DENR as written vs strand-fixed")
CHRLEN <- 12000000L
ngene <- 300
glen <- pmax(pmin(round(exp(rnorm(ngene, log(25000), 0.9))), 400000), 500)
starts <- cumsum(round(runif(ngene, 3000, 20000)) + c(0, head(glen, -1)))
keep <- (starts + glen) < CHRLEN - 1000
starts <- starts[keep]; glen <- glen[keep]; ngene <- length(starts)
strand_v <- sample(c("+", "-"), ngene, replace = TRUE)
genes <- GRanges("chr1", IRanges(starts, starts + glen), strand = strand_v)
genes$tx_name <- paste0("g", seq_len(ngene)); genes$gene_id <- genes$tx_name

mk_signal <- function(sel) {
  parts <- lapply(sel, function(i) {
    g <- genes[i]; w <- width(g); expr <- exp(rnorm(1, 1.2, 1.1))
    prof <- 0.6 * exp(-seq(0, 1, length.out = w) * 2.2) + 0.25
    pk <- min(w, 300); prof[1:pk] <- prof[1:pk] + 14 * exp(-seq(0, 1, length.out = pk) * 6)
    vals <- rpois(w, prof * expr)
    if (as.vector(strand(g)) == "-") vals <- rev(vals)
    r <- Rle(vals); idx <- cumsum(runLength(r))
    st <- start(g) + c(0, head(idx, -1)); en <- start(g) + idx - 1
    ok <- runValue(r) > 0
    GRanges("chr1", IRanges(st[ok], en[ok]), score = as.numeric(runValue(r)[ok]))
  })
  out <- sort(do.call(c, parts)); out <- out[!duplicated(start(out))]
  seqlengths(out) <- c(chr1 = CHRLEN); out
}
bwp <- file.path(WORK, "plus.bw"); bwm <- file.path(WORK, "minus.bw")
export.bw(mk_signal(which(strand_v == "+")), bwp)
export.bw(mk_signal(which(strand_v == "-")), bwm)
cat("  simulated", ngene, "genes with a 14x pausing peak\n")

BIN <- 250
tx <- genes[width(genes) >= 8000]
ma <- suppressMessages(DENR:::model_agreement(
  DENR:::group_transcripts(tx, distance = 0), bwp, bwm))
dt <- as.data.table(ma); dt[, group := names(ma)]
dt <- dt[percent_transcribed >= 0.8 & percent_match >= 0.8]
final <- dt[, .SD[which.max(percent_match)], by = "group"]
fgr <- with(final, GRanges(seqnames, IRanges(start, end), strand = strand))
fbins <- DENR:::create_bins(DENR:::group_transcripts(fgr), bin_size = BIN)
plus_regex <- grep("+", names(fbins))            # as DENR does it
plus_fixed <- grep("+", names(fbins), fixed = TRUE)
minus_idx  <- grep("-", names(fbins))

as_denr <- c(DENR:::summarize_bigwig(bigwig_file = bwp, fbins[plus_regex]),
             lapply(DENR:::summarize_bigwig(bigwig_file = bwp, fbins[minus_idx]), rev))
as_fixed <- c(DENR:::summarize_bigwig(bigwig_file = bwp, fbins[plus_fixed]),
              lapply(DENR:::summarize_bigwig(bigwig_file = bwm, fbins[minus_idx]), rev))

fit_profile <- function(raw) {
  d <- rbindlist(lapply(raw, function(x) {
    pos <- resc(bin_size = BIN, length_out = length(x), linear_head_length = 10000,
                linear_tail_length = 5000, head_percent = 0.2, tail_percent = 0.2)
    data.table(pos, scaled_count = (x - (min(x) - 1e-6)) / (max(x) - (min(x) - 1e-6)))
  }))
  ctrl <- stats::loess.control(); ctrl[["trace.hat"]] <- "approximate"
  lo <- stats::loess(scaled_count ~ pos, data = d, span = 0.4, control = ctrl)
  grid <- seq(0, 1, by = 1e-5); pred <- stats::predict(lo, newdata = grid)
  pred / stats::median(pred[grid >= 0.2 & grid <= 0.8])
}
report <- function(label, raw) {
  const <- sum(vapply(raw, function(x) max(x) == min(x), logical(1)))
  zero  <- sum(vapply(raw, function(x) all(x == 0), logical(1)))
  p <- fit_profile(raw)
  cat(sprintf("  %-16s curves=%3d  all-zero=%3d (%2.0f%%)  range=%6.2fx\n",
              label, length(raw), zero, 100 * zero / length(raw), max(p) / min(p)))
  invisible(p)
}
cat("  selected genes:", length(fgr), "\n")
pd <- report("DENR as-is", as_denr)
pf <- report("strand-fixed", as_fixed)
at <- c(1, 5001, 10001, 25001, 50001, 75001, 100001)
cat("\n                 pos=0.00   0.05   0.10   0.25   0.50   0.75   1.00\n")
cat("  DENR as-is: ", paste(sprintf("%6.3f", pd[at]), collapse = " "), "\n")
cat("  strand-fixed:", paste(sprintf("%6.3f", pf[at]), collapse = " "), "\n")

# --------------------------------------------------------------------------
# 18.3 predict.loess has no value outside its training hull
# --------------------------------------------------------------------------
banner("18.3 predict.loess returns NA outside the training hull")
x <- runif(20000, 0.3, 0.7)
lo <- stats::loess(y ~ x, data = data.frame(x = x, y = 1 + rnorm(length(x), 0, 0.05)),
                   span = 0.4, control = ctrl <- stats::loess.control(trace.hat = "approximate"))
pred <- stats::predict(lo, newdata = seq(0, 1, by = 1e-5))
cat("  training positions confined to [0.3, 0.7];",
    sum(is.na(pred)), "of", length(pred), "grid points are NA\n")
cat("  apply_shape_profile writes these straight into the model matrix\n")

# --------------------------------------------------------------------------
# 18.4 apply_shape_profile is actually sound
# --------------------------------------------------------------------------
banner("18.4 apply_shape_profile fuzz: no crash, no misalignment")
p <- seq(0, 1, length.out = 4000)
mono <- stats::loess(v ~ p, data = data.frame(p = p, v = exp(-3 * p)), span = 0.15,
                     control = stats::loess.control(surface = "direct"))
tsp <- methods::new("transcript_shape_profile", transcripts = GRanges(),
                    bin_size = 250L, linear_head_length = 10000L,
                    linear_tail_length = 5000L, head_percent = 0.2,
                    tail_percent = 0.2, shape_profile = mono, shape_scale_factor = 1)
set.seed(99); fails <- 0; misaligned <- 0; ok <- 0
for (trial in seq_len(TRIALS)) {
  base <- 0; chunks <- list()
  for (i in 1:4) {
    niso <- sample(2:8, 1); gl <- sample(c(12000, 40000, 90000), 1)
    s <- base + sample(0:round(gl * 0.4), niso, replace = TRUE)
    chunks[[i]] <- data.frame(s = s, e = s + sample(round(gl * 0.3):gl, niso, replace = TRUE),
                              st = sample(c("+", "-"), 1))
    base <- base + 400000
  }
  dd <- do.call(rbind, chunks)
  g <- GRanges("chr1", IRanges(dd$s + 1, dd$e), strand = dd$st)
  g$tx_name <- paste0("t", seq_len(nrow(dd))); g$gene_id <- g$tx_name
  res <- tryCatch({
    tq <- suppressMessages(transcript_quantifier(
      g, "tx_name", bin_size = 250,
      mask_start_bins = if (runif(1) < 0.6) sample(1:8, 2, replace = TRUE) else NULL,
      bin_operation = sample(c("round", "floor", "ceiling"), 1)))
    tqs <- suppressMessages(apply_shape_profile(tq, tsp))
    bad <- 0
    for (i in seq_along(tqs@models)) {
      M <- tqs@models[[i]]; st <- as.vector(strand(tq@bins[[i]]))[1]
      for (m in seq_len(ncol(M))) {
        v <- M[M[, m] != 0, m]
        if (length(v) < 3) next
        vv <- if (st == "+") v else rev(v)
        if (!all(diff(vv) <= 1e-9)) bad <- bad + 1
      }
    }
    bad
  }, error = function(e) NA_integer_)
  if (is.na(res)) fails <- fails + 1 else { ok <- ok + 1; misaligned <- misaligned + res }
}
cat(sprintf("  %d fuzzed annotations: ok=%d crashed=%d misaligned columns=%d\n",
            TRIALS, ok, fails, misaligned))
cat("  (section 18 quotes 700 trials; raise with SHAPE_PROBE_TRIALS)\n")
cat("\nDone. Numbers above populate plan section 18.\n")
