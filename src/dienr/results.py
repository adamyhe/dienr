"""Result objects and schema-validated writers.

Every output carries `SCHEMA_VERSION`, the software version, and the annotation,
design and configuration digests, so a table on disk can always be traced back
to the inputs that produced it (plan section 9).

Raw abundance is always present. Normalization is a separate operation and its
factors live in their own table; nothing here silently rescales an estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from .status import FitStatus, is_failure

if TYPE_CHECKING:
    from .counts import SampleCounts
    from .design import Design
    from .solvers.legacy import LocusFit

#: Bumped whenever an output schema changes incompatibly.
SCHEMA_VERSION = "1"

#: Polars accepts either dtype classes or instances in a schema, and its own
#: alias for that union is private, so the output schemas are typed loosely
#: here rather than scattering per-call ignores.
Schema = dict[str, Any]

MODEL_ABUNDANCE_SCHEMA: Schema = {
    "sample_id": pl.String,
    "locus_id": pl.UInt32,
    "model_id": pl.String,
    "model_index": pl.UInt32,
    "abundance_raw": pl.Float64,
    "abundance_normalized": pl.Float64,
    "normalization_method": pl.String,
    "equivalence_class_size": pl.UInt32,
    "solver_status": pl.String,
    "objective": pl.Float64,
    "penalty_value": pl.Float64,
    "iterations": pl.UInt32,
    "converged": pl.Boolean,
}

TRANSCRIPT_MODEL_MAP_SCHEMA: Schema = {
    "transcript_id": pl.String,
    "gene_id": pl.String,
    "locus_id": pl.UInt32,
    "model_id": pl.String,
    "contig": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "strand": pl.String,
    "equivalence_class_size": pl.UInt32,
    "exactly_identifiable": pl.Boolean,
    "has_support": pl.Boolean,
}

LOCUS_QC_SCHEMA: Schema = {
    "sample_id": pl.String,
    "locus_id": pl.UInt32,
    "locus_name": pl.String,
    "n_bins": pl.UInt32,
    "n_unmasked_bins": pl.UInt32,
    "n_models_before_collapse": pl.UInt32,
    "n_models_after_collapse": pl.UInt32,
    "matrix_rank": pl.UInt32,
    "condition_number": pl.Float64,
    "total_observed": pl.Float64,
    "total_observed_all_bins": pl.Float64,
    "masked_bin_fraction": pl.Float64,
    "total_fitted": pl.Float64,
    "residual_sum_of_squares": pl.Float64,
    "gradient_norm": pl.Float64,
    "solver_status": pl.String,
    "failure_reason": pl.String,
}

NORMALIZATION_FACTORS_SCHEMA: Schema = {
    "sample_id": pl.String,
    "method": pl.String,
    "factor": pl.Float64,
    "source": pl.String,
    "is_external": pl.Boolean,
}

#: Written alongside the tables so a directory is self-describing.
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class RunSummary:
    """Counts of each status across a run.

    Attributes:
        by_status: Number of sample x locus fits per status value.
        n_fits: Total fits attempted.
    """

    by_status: dict[str, int]
    n_fits: int

    @property
    def n_failures(self) -> int:
        """Fits whose status means no estimate exists."""
        return sum(count for name, count in self.by_status.items() if is_failure(FitStatus(name)))

    @property
    def failure_fraction(self) -> float:
        """Failures as a fraction of all fits, 0.0 for an empty run."""
        return self.n_failures / self.n_fits if self.n_fits else 0.0

    def exceeds(self, threshold: float) -> bool:
        """True when `failure_fraction` is above `threshold`.

        The CLI exits nonzero on this, so a run that quietly failed most of its
        loci cannot be mistaken for a successful one.
        """
        return self.failure_fraction > threshold


@dataclass(frozen=True)
class FitResult:
    """Fitted abundances plus diagnostics and provenance for one or more samples.

    Attributes:
        model_abundance: One row per sample and identifiable model.
        transcript_model_map: Many-to-one transcript-to-model mapping.
        locus_qc: One row per sample and locus. ``total_observed`` covers only
            the bins the fit used; ``total_observed_all_bins`` is coverage
            before masking, and ``masked_bin_fraction`` the share removed.
        normalization_factors: Between-sample factors, empty until computed.
        provenance: Digests, versions and configuration.
    """

    model_abundance: pl.DataFrame
    transcript_model_map: pl.DataFrame
    locus_qc: pl.DataFrame
    normalization_factors: pl.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> RunSummary:
        """Status tally over the locus QC table."""
        if self.locus_qc.height == 0:
            return RunSummary(by_status={}, n_fits=0)
        counts = self.locus_qc.group_by("solver_status").len()
        return RunSummary(
            by_status=dict(zip(counts["solver_status"], counts["len"], strict=True)),
            n_fits=self.locus_qc.height,
        )

    def to_numpy(self, *, value: str = "abundance_raw") -> tuple[np.ndarray, list[str], list[str]]:
        """Pivot to a dense sample-by-model matrix for downstream ML.

        Args:
            value: Column to fill the matrix with.

        Returns:
            ``(matrix, sample_ids, model_ids)`` where ``matrix`` is
            ``(n_samples, n_models)``. Absent combinations are NaN, never zero.
        """
        wide = self.model_abundance.pivot(
            on="model_id", index="sample_id", values=value, aggregate_function="first"
        ).sort("sample_id")
        sample_ids = wide["sample_id"].to_list()
        model_ids = [c for c in wide.columns if c != "sample_id"]
        matrix = wide.select(model_ids).to_numpy().astype(np.float64)
        return matrix, sample_ids, model_ids

    def write(
        self,
        directory: str | Path,
        *,
        compression: Literal["zstd", "snappy", "gzip", "lz4", "brotli", "uncompressed"] = "zstd",
    ) -> Path:
        """Write all tables plus a manifest to `directory`.

        Parquet carries the schema version and digests as key-value metadata, so
        a single file remains traceable even when separated from the manifest.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        metadata = {k: str(v) for k, v in self.provenance.items() if not isinstance(v, dict)}

        for name, frame in (
            ("model_abundance", self.model_abundance),
            ("transcript_model_map", self.transcript_model_map),
            ("locus_qc", self.locus_qc),
            ("normalization_factors", self.normalization_factors),
        ):
            frame.write_parquet(out / f"{name}.parquet", compression=compression, metadata=metadata)

        manifest = dict(self.provenance)
        manifest["summary"] = {
            "by_status": self.summary.by_status,
            "n_fits": self.summary.n_fits,
            "n_failures": self.summary.n_failures,
        }
        (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return out

    @classmethod
    def read(cls, directory: str | Path) -> FitResult:
        """Read back a directory written by `write`."""
        source = Path(directory)
        manifest_path = source / MANIFEST_FILENAME
        provenance = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        found = provenance.get("schema_version")
        if found is not None and str(found) != SCHEMA_VERSION:
            raise ValueError(
                f"{source}: schema_version {found!r} was written by a different "
                f"version of dienr; this build reads {SCHEMA_VERSION!r}"
            )
        return cls(
            model_abundance=pl.read_parquet(source / "model_abundance.parquet"),
            transcript_model_map=pl.read_parquet(source / "transcript_model_map.parquet"),
            locus_qc=pl.read_parquet(source / "locus_qc.parquet"),
            normalization_factors=pl.read_parquet(source / "normalization_factors.parquet"),
            provenance=provenance,
        )

    def legacy_transcript_abundance(self) -> pl.DataFrame:
        """Repeat each model's abundance across every transcript mapped to it.

        This matches the reference implementation's convention and exists only
        for comparison against it. The repeated rows carry
        ``equivalence_class_size`` and ``exactly_identifiable`` so a reader
        cannot mistake a shared value for transcript-level resolution.
        """
        return (
            self.transcript_model_map.join(
                self.model_abundance, on=["locus_id", "model_id"], how="inner"
            )
            .select(
                "sample_id",
                "transcript_id",
                "gene_id",
                "locus_id",
                "model_id",
                "abundance_raw",
                "equivalence_class_size",
                "exactly_identifiable",
                "solver_status",
            )
            .sort("sample_id", "transcript_id")
        )


def _empty(schema: Schema) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def build_transcript_model_map(design: Design) -> pl.DataFrame:
    """Join design coordinates onto the transcript-to-model mapping."""
    coordinates = design.loci.select("locus_id", "contig", "strand", pl.col("start"), pl.col("end"))
    return (
        design.transcript_map.join(coordinates, on="locus_id", how="left")
        .select(
            "transcript_id",
            "gene_id",
            "locus_id",
            "model_id",
            "contig",
            "start",
            "end",
            "strand",
            "equivalence_class_size",
            "exactly_identifiable",
            "has_support",
        )
        .cast(pl.Schema(TRANSCRIPT_MODEL_MAP_SCHEMA))
        .sort("locus_id", "transcript_id")
    )


def collect_results(
    design: Design,
    samples: list[tuple[SampleCounts, list[LocusFit]]],
    *,
    provenance: dict[str, Any] | None = None,
) -> FitResult:
    """Assemble per-locus fits into the versioned output tables.

    Args:
        design: The design the fits were made against.
        samples: One ``(counts, fits)`` pair per sample, where ``fits`` is
            indexed by ``locus_id``.
        provenance: Extra fields to merge into the manifest.

    Returns:
        A `FitResult`.

    Raises:
        ValueError: If a sample's counts were built against a different design,
            or if its fit list does not cover every locus.
    """
    from .diagnostics import assess, residual_sum_of_squares

    abundance_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    model_ids = {
        (row["locus_id"], row["model_index"]): row["model_id"]
        for row in design.model_table.iter_rows(named=True)
    }
    class_sizes = {
        (row["locus_id"], row["model_index"]): row["equivalence_class_size"]
        for row in design.model_table.iter_rows(named=True)
    }
    locus_names = dict(zip(design.loci["locus_id"], design.loci["locus_name"], strict=True))
    # Pre-collapse model counts come from the annotation, not the reduced matrix.
    before_collapse = dict(zip(design.loci["locus_id"], design.loci["n_transcripts"], strict=True))

    for counts, fits in samples:
        if counts.design_digest != design.digest:
            raise ValueError(
                f"sample {counts.sample_id!r} was counted against design "
                f"{counts.design_digest!r}, not {design.digest!r}"
            )
        if len(fits) != design.n_loci:
            raise ValueError(
                f"sample {counts.sample_id!r} has {len(fits)} fits for {design.n_loci} loci"
            )

        for locus_id, fit in enumerate(fits):
            models = design.models[locus_id]
            observed = counts.values[locus_id]
            conditioning = assess(models, design.mask(locus_id))

            if is_failure(fit.status):
                total_fitted = float("nan")
                rss = float("nan")
                total_observed = float("nan")
            else:
                total_fitted = float((models @ fit.abundance).sum())
                rss = residual_sum_of_squares(models, fit.abundance, observed)
                # Signal in the bins the fit actually used. Masked bins are
                # already zeroed in `observed`, so this is not locus coverage;
                # `total_observed_all_bins` alongside it is.
                total_observed = float(observed.sum())

            qc_rows.append(
                {
                    "sample_id": counts.sample_id,
                    "locus_id": locus_id,
                    "locus_name": locus_names[locus_id],
                    "n_bins": conditioning.n_bins,
                    "n_unmasked_bins": conditioning.n_unmasked_bins,
                    "n_models_before_collapse": before_collapse[locus_id],
                    "n_models_after_collapse": conditioning.n_models,
                    "matrix_rank": conditioning.matrix_rank,
                    "condition_number": conditioning.condition_number,
                    "total_observed": total_observed,
                    "total_observed_all_bins": (
                        counts.totals_all_bins[locus_id]
                        if locus_id < len(counts.totals_all_bins)
                        else float("nan")
                    ),
                    "masked_bin_fraction": design.masked_fraction(locus_id),
                    "total_fitted": total_fitted,
                    "residual_sum_of_squares": rss,
                    "gradient_norm": fit.gradient_norm,
                    "solver_status": _qualified_status(fit.status, conditioning),
                    "failure_reason": fit.failure_reason,
                }
            )

            for model_index, value in enumerate(fit.abundance):
                key = (locus_id, model_index)
                abundance_rows.append(
                    {
                        "sample_id": counts.sample_id,
                        "locus_id": locus_id,
                        "model_id": model_ids[key],
                        "model_index": model_index,
                        "abundance_raw": float(value),
                        "abundance_normalized": None,
                        "normalization_method": None,
                        "equivalence_class_size": class_sizes[key],
                        "solver_status": str(fit.status),
                        "objective": fit.objective,
                        "penalty_value": 0.0,
                        "iterations": fit.iterations,
                        "converged": fit.converged,
                    }
                )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "annotation_digest": design.annotation_digest,
        "design_digest": design.digest,
        "bin_size": design.config.bin_size,
        "distance": design.config.effective_distance,
        "mask_start_bins": list(design.config.mask_start_bins),
        "mask_end_bins": list(design.config.mask_end_bins),
        "bin_operation": design.config.bin_operation,
        "n_loci": design.n_loci,
        "samples": [counts.sample_id for counts, _ in samples],
    }
    from . import __version__

    manifest["dienr_version"] = __version__
    if provenance:
        manifest.update(provenance)

    return FitResult(
        model_abundance=(
            pl.DataFrame(abundance_rows, schema=MODEL_ABUNDANCE_SCHEMA)
            if abundance_rows
            else _empty(MODEL_ABUNDANCE_SCHEMA)
        ),
        transcript_model_map=build_transcript_model_map(design),
        locus_qc=(
            pl.DataFrame(qc_rows, schema=LOCUS_QC_SCHEMA) if qc_rows else _empty(LOCUS_QC_SCHEMA)
        ),
        normalization_factors=_empty(NORMALIZATION_FACTORS_SCHEMA),
        provenance=manifest,
    )


def _qualified_status(status: FitStatus, conditioning: Any) -> str:
    """Escalate an OK fit to a conditioning warning where warranted.

    A converged fit on a rank-deficient design is still numerically fine but its
    parameter allocation is not unique, so reporting plain ``ok`` would overstate
    what was learned.
    """
    if status is not FitStatus.OK:
        return str(status)
    if conditioning.rank_deficient:
        return str(FitStatus.RANK_DEFICIENT)
    if conditioning.ill_conditioned:
        return str(FitStatus.ILL_CONDITIONED)
    return str(status)
