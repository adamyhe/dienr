"""BigWig aggregation into per-bin signal, with exposure and provenance.

All BigWig access in this package goes through `read_binned`, which exists
mainly to pin two `pybigtools` arguments that are wrong by default and silently
so:

``exact=True``
    Without it, ``values(..., bins=N)`` reads precomputed zoom levels, which
    are lossy. On the reference package's ``test_double_strand_plus.bw``, 250 bp
    bins over ``1:0-2000`` have a true per-base sum of ``[63, 68, 72, 77]``
    while the zoom-level path returns ``[62.53, 69.53, 71.47, 76.76]``. No
    error is raised and every abundance downstream is biased. Small windows can
    agree by coincidence, so a small unit test may not catch it.

``fillna=0``
    Uncovered positions and empty bins come back as ``NaN``. In a nascent-RNA
    BigWig an absent interval means zero observed polymerase, not a missing
    observation, and a leaked ``NaN`` would surface as a spurious numerical
    failure instead of a legitimate zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pybigtools

from .design import Design
from .status import FitStatus

Summary = Literal["sum", "mean", "min", "max", "std"]

#: How values in a signal file should be interpreted, per plan section 8.
ValueKind = Literal["raw", "spike_in_scaled", "library_normalized", "other_transformed"]

#: Every permitted `ValueKind`, for validating untyped external input.
VALUE_KINDS: tuple[str, ...] = (
    "raw",
    "spike_in_scaled",
    "library_normalized",
    "other_transformed",
)

#: Likelihoods that require count-equivalent input.
_COUNT_LIKELIHOODS = frozenset({"poisson", "negative_binomial"})

#: Per-bin summaries that preserve counts. ``mean`` divides by bin width, so it
#: is a density and cannot be fed to a count likelihood.
_COUNT_SUMMARIES = frozenset({"sum"})


@dataclass(frozen=True)
class SignalFile:
    """One stranded BigWig plus the metadata needed to validate its use.

    Attributes:
        path: Path to the BigWig.
        strand: Which strand's signal this file holds.
        value_kind: Whether values are raw counts or have been transformed.
            Recorded rather than guessed; `require_count_equivalent` enforces it.
    """

    path: str | Path
    strand: Literal["+", "-"]
    value_kind: ValueKind = "raw"

    def __post_init__(self) -> None:
        # External input (a samples CSV) is untyped, and a typo here would
        # silently defeat the count-equivalence check below rather than
        # failing, so validate rather than trusting the annotation.
        if self.value_kind not in VALUE_KINDS:
            raise ValueError(
                f"{self.path}: value_kind {self.value_kind!r} is not one of {list(VALUE_KINDS)}"
            )
        if self.strand not in ("+", "-"):
            raise ValueError(f"{self.path}: strand must be '+' or '-', got {self.strand!r}")

    def require_count_equivalent(self, likelihood: str, summary: str = "sum") -> None:
        """Raise if `likelihood` needs counts but the input is not count-like.

        Two distinct ways the input can fail to be a count: the file itself was
        transformed, or the per-bin summary divided by bin width.
        """
        if likelihood not in _COUNT_LIKELIHOODS:
            return
        if self.value_kind != "raw":
            raise ValueError(
                f"{self.path}: likelihood {likelihood!r} requires count-equivalent "
                f"signal, but value_kind is {self.value_kind!r}. Use an untransformed "
                "BigWig or select a non-count likelihood explicitly."
            )
        if summary not in _COUNT_SUMMARIES:
            raise ValueError(
                f"{self.path}: likelihood {likelihood!r} requires a count-preserving "
                f"summary, but summary is {summary!r}, which is a per-base density. "
                f"Use one of {sorted(_COUNT_SUMMARIES)}."
            )


@dataclass
class SampleCounts:
    """Per-bin signal for one sample, aligned to a `Design`.

    Attributes:
        sample_id: Identifier for the sample.
        values: Per-locus bin vectors, indexed by ``locus_id``. Masked bins are
            already zeroed.
        locus_status: Per-locus status assigned during extraction. Only
            `FitStatus.MISSING_CONTIG` and `FitStatus.ALL_ZERO` originate here;
            everything else is decided by the solver.
        totals_all_bins: Per-locus signal summed over *all* bins, before
            masking zeroed any of them. Kept because `values` is masked in
            place, so the unmasked total is otherwise unrecoverable, and
            reporting only the masked total would understate observed coverage
            at loci where masking removed a lot.
        design_digest: Digest of the `Design` these counts were built against.
        provenance: Free-form record of inputs and reader settings.
        exposure: Optional known scaling factor for the sample, such as a
            spike-in or library size. Never applied silently; solvers take it
            explicitly.
    """

    sample_id: str
    values: list[np.ndarray]
    locus_status: list[FitStatus]
    design_digest: str
    totals_all_bins: list[float] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    exposure: float = 1.0

    def total_signal(self) -> float:
        """Sum of all unmasked signal across loci."""
        return float(sum(v.sum() for v in self.values))


def read_binned(
    path: str | Path,
    contig: str,
    start: int,
    end: int,
    n_bins: int,
    *,
    summary: Summary = "sum",
    reader: pybigtools.BBIReader | None = None,
) -> np.ndarray:
    """Read `n_bins` exact binned summaries from a BigWig.

    Args:
        path: BigWig path. Ignored when `reader` is supplied.
        contig: Contig name, which must appear in the file header.
        start: 0-based inclusive window start.
        end: Half-open window end.
        n_bins: Number of equal-width bins spanning the window.
        summary: Per-bin statistic. ``"sum"`` is the count-preserving choice.
        reader: An open reader to reuse across calls.

    Returns:
        A float64 array of length `n_bins`, with uncovered positions counted as
        zero.

    Raises:
        KeyError: If `contig` is absent from the file header.
    """
    handle = reader if reader is not None else pybigtools.open(str(path))
    values = handle.values(
        contig,
        start,
        end,
        bins=n_bins,
        summary=summary,
        exact=True,  # never read lossy zoom levels; see module docstring
        fillna=0.0,  # uncovered means zero polymerase, not missing
        oob=0.0,  # reads past a contig end are zero, not an error
    )
    return np.asarray(values, dtype=np.float64)


def declared_contigs(path: str | Path) -> dict[str, int]:
    """Contig names and lengths declared in a BigWig header.

    The header is the only way to tell a contig that is absent from the file
    (`FitStatus.MISSING_CONTIG`) from one that is present but carries no signal
    in the window of interest (`FitStatus.ALL_ZERO`). Inferring the contig list
    from observed data conflates the two, which section 10 of the plan forbids.
    """
    return dict(pybigtools.open(str(path)).chroms())


def count_sample(
    design: Design,
    sample_id: str,
    signal: SignalFile | tuple[SignalFile, ...],
    *,
    summary: Summary = "mean",
    exposure: float = 1.0,
) -> SampleCounts:
    """Aggregate stranded BigWig signal into `design`'s bins.

    Each locus is read from the file matching its strand. Masked bins are zeroed
    so that model and data agree, as the objective requires.

    Args:
        design: Geometry to align to.
        sample_id: Identifier for this sample.
        signal: One `SignalFile` per strand. A single file is accepted only when
            every locus lies on its strand.
        summary: Per-bin statistic. Defaults to ``"mean"`` because that is what
            the reference implementation uses -- ``add_data`` hardcodes
            ``summary_operation <- "mean"`` -- so its abundances are in units of
            mean per-base polymerase density, not counts. Verified against R:
            our ``"sum"`` was exactly ``bin_size`` times its values. A count
            likelihood needs ``"sum"`` instead, and must not be combined with
            ``"mean"``.
        exposure: Known sample scaling factor, recorded but not applied here.

    Returns:
        A `SampleCounts` aligned to `design`.

    Raises:
        ValueError: If a required strand has no signal file, or if two files
            claim the same strand.
    """
    files = (signal,) if isinstance(signal, SignalFile) else tuple(signal)
    by_strand: dict[str, SignalFile] = {}
    for item in files:
        if item.strand in by_strand:
            raise ValueError(f"two signal files given for strand {item.strand!r}")
        by_strand[item.strand] = item

    needed = set(design.loci["strand"].unique().to_list())
    missing = needed - set(by_strand)
    if missing:
        raise ValueError(f"design contains loci on strand(s) {sorted(missing)} with no signal file")

    readers = {s: pybigtools.open(str(f.path)) for s, f in by_strand.items()}
    headers = {s: dict(r.chroms()) for s, r in readers.items()}

    values: list[np.ndarray] = [np.empty(0) for _ in range(design.n_loci)]
    statuses: list[FitStatus] = [FitStatus.OK] * design.n_loci
    totals: list[float] = [float("nan")] * design.n_loci

    for locus in design.loci.iter_rows(named=True):
        locus_id = locus["locus_id"]
        strand = locus["strand"]
        n_bins = design.bin_count(locus_id)
        contig = locus["contig"]

        if contig not in headers[strand]:
            # A real failure, distinct from a locus with no signal.
            values[locus_id] = np.full(n_bins, np.nan)
            statuses[locus_id] = FitStatus.MISSING_CONTIG
            continue

        # Bins are anchored at the locus start, so no lookup is needed. Taking
        # this from the bins frame would also mean an unguarded `.min()` that
        # returns None for an empty selection.
        window_start = int(locus["start"])
        window_end = window_start + n_bins * design.config.bin_size

        raw = read_binned(
            by_strand[strand].path,
            contig,
            window_start,
            min(window_end, headers[strand][contig]),
            n_bins,
            summary=summary,
            reader=readers[strand],
        )
        # A window clipped at the contig end returns fewer bins than requested;
        # pad with zeros so the vector still aligns to the design.
        if raw.size < n_bins:
            raw = np.concatenate([raw, np.zeros(n_bins - raw.size)])

        # Minus-strand BigWigs are conventionally negative; magnitude is the count.
        np.abs(raw, out=raw)
        # Record the total before masking; after this the masked bins are gone.
        totals[locus_id] = float(raw.sum())
        raw[design.mask(locus_id)] = 0.0
        values[locus_id] = raw
        if not raw.any():
            statuses[locus_id] = FitStatus.ALL_ZERO

    return SampleCounts(
        sample_id=sample_id,
        values=values,
        locus_status=statuses,
        design_digest=design.digest,
        totals_all_bins=totals,
        provenance={
            "signal_files": [
                {"path": str(f.path), "strand": f.strand, "value_kind": f.value_kind} for f in files
            ],
            "summary": summary,
            "reader": "pybigtools",
            "reader_settings": {"exact": True, "fillna": 0.0, "oob": 0.0},
        },
        exposure=exposure,
    )
