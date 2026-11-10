# Outputs

`dienr fit` writes a directory holding four Parquet tables and
`manifest.json`. Each Parquet file also carries the schema version and digests
as key-value metadata, so a table copied out on its own can still be traced
back to its inputs. Read it back with `dienr.FitResult.read(directory)`.

## `model_abundance.parquet`: one row per sample × identifiable model

| column | type | meaning |
|---|---|---|
| `sample_id` | str | |
| `locus_id`, `model_index` | u32 | the model's position in the design |
| `model_id` | str | stable model identifier |
| `abundance_raw` | f64 | the estimate; `NaN` for any failed fit |
| `abundance_standard_error` | f64 | `quasi_poisson`/`poisson` only; `NaN` where a Wald error is undefined; null for `legacy` |
| `abundance_normalized`, `normalization_method` | f64, str | filled only by an explicit normalization step |
| `equivalence_class_size` | u32 | transcripts sharing this model |
| `solver_status` | str | see [concepts](concepts.md#fit-status-failures-are-never-zeros) |
| `objective`, `penalty_value` | f64 | final objective and penalty term |
| `iterations`, `converged` | u32, bool | optimizer diagnostics, which the R package discards |

## `transcript_model_map.parquet`: one row per transcript

`transcript_id`, `gene_id`, `locus_id`, `model_id`, `contig`, `start`, `end`
(0-based, half-open), `strand`, `equivalence_class_size`,
`exactly_identifiable`, `has_support` (whether the model occupies any bin at
this bin size).

## `locus_qc.parquet`: one row per sample × locus

`n_bins`, `n_unmasked_bins`, models before and after collapsing
(`n_models_before_collapse`, `n_models_after_collapse`), `matrix_rank`,
`condition_number`, `total_observed` (signal in *fitted* bins),
`total_observed_all_bins` (including masked bins; use this for coverage),
`masked_bin_fraction`, `total_fitted`, `residual_sum_of_squares`,
`gradient_norm`, `solver_status`, `failure_reason`.

## `normalization_factors.parquet`

`sample_id`, `method`, `factor`, `source`, `is_external`. Fitting never
normalizes; this table holds factors from an explicit normalization step.
(Normalization isn't implemented yet, so it's empty.)

## `manifest.json`

`schema_version`, `dienr_version`, `annotation_digest`, `design_digest`,
`shape_digest`, `runon_digest`, bin size, distance, masking, `bin_operation`,
`n_loci`, `samples`, plus the fit's likelihood and penalty settings.

## Design and counts directories

- `build-design` writes `design.pkl` (the reusable `Design`) plus
  `bins.parquet`, `loci.parquet` and `models.parquet`, for inspection.
- `count` writes `counts.pkl`.
- `shape-profile` writes `shape_profile.pkl` and `shape_profile.parquet`.
- `runon-tail` writes `runon_tail.pkl` and `runon_tail.parquet`.

The `.pkl` files are Python pickles for passing objects between commands.
Treat them as an internal format tied to a `dienr` version, not as an archive.
