# Roadmap

Where `dienr` is going, and where it deliberately isn't. Section numbers (§)
refer to the [implementation plan](notes/denr-python-implementation-plan.md),
which is the binding spec; this page summarizes it and doesn't override it.
Released changes are in the [changelog](CHANGELOG.md).

## Scope

**dienr replicates DENR, with its bugs fixed, and its defaults are the only
recommendation** (§18.27). Measured against K562 long-read isoforms, none of
the extensions (count likelihoods, run-on tails, shape profiles, penalties)
improved isoform calls enough to depart from DENR. They stay in the package,
tested and labelled experimental, but method-improvement work is paused.

In scope: correctness, bug fixes, input checks, speed and packaging.

## Planned

- **Normalization factors** (§15; schema in §9,
  `normalization_factors.parquet`). Library-size and median-ratio factors in
  their own table, separate from `abundance_raw`. Not "TMM" unless it is TMM.
- **ML adapters** beyond `FitResult.to_numpy` (§12, Phase 5).
- **Container image and a Bioconda recipe** (§12 Phase 5, §19 item 10).
  `dienr` is on PyPI; there is no container yet.
- **Harness hygiene.** Five validation scripts still filter
  `solver_status == "ok"` and must keep every converged status before any of
  §18.6 or §18.9–18.13 is re-run (list in §18.29).
- **Correctness, input checks, speed** as issues arise. The one-job path is
  still ~6% (legacy) / ~11% (quasi-Poisson) slower than 0.1.0 (§14.7).

## Deferred: would reopen method work

These need new evidence or a decision from the maintainer before they start.

- **Gate C for penalties** (§17): stability under resampling and a λ
  selection rule. Until then no penalty is recommended and `--lambda2` has no
  nonzero default.
- **Joint multi-sample group penalty** (§12, Phase 4). Depends on Gate C.
- **TSS-activity evidence** from PINTS/dREG-style calls, as hard pins or soft
  penalty weights (§8, §12 Phase 4).
- **A second cell line** for the long-read accuracy comparison (§18.20–18.27).
  Every real-data accuracy result so far is K562.
- **A per-gene run-on scale.** The one estimator tried made held-out fit worse
  (§18.17).

## Not planned

Decided against, with the reasons recorded in the plan:

- **BAM input** (§20): converting reads to polymerase positions depends on the
  chemistry and belongs upstream. Input is BigWig only; see
  [preparing input BigWigs](docs/input-data.md).
- **Negative-binomial likelihood** (§17, Gate D): it cannot change an estimate
  on a binary design, and quasi-Poisson matches its standard errors to 1–4%.
- **A JIT kernel** (§14.1, §14.3): it is slower than BLAS on large loci, and on
  a pipeline already ~14.5× faster than R it would buy about 1.2× end to end.
- **v1 non-goals** (§3): Gviz-style plotting, TxDb download and caching, the
  legacy TSS CNN, de novo transcript discovery.

## Toward 1.0

The definition of done is §19. Outstanding as of November 2026:

- Ridge regularization is implemented but not production-ready (Gate C).
- The validation record has simulation and one real dataset, with no
  orthogonal real-data accuracy check beyond K562 long reads.
- No container.
- §19 item 2 asks for BAM input, which §20 has since ruled out; the criterion
  needs revising rather than meeting.
- The R package's license metadata conflicts (MIT vs GPL ≥ 2) and is
  unresolved upstream, so implementation continues from the paper and the
  package's observed behaviour.
