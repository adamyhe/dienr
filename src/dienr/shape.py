"""Empirical shape profiles: relative polymerase density along a gene.

The uniform-density assumption behind a 0/1 design matrix is wrong in a known,
reproducible way. Nascent RNA sequencing shows a sharp promoter-proximal
pausing peak within a few hundred bases of the TSS, a slow decline through the
gene body, and often a 3' rise near the poly(A) site. A shape profile measures
that curve once, from genes clean enough to measure it on, and then reweights
every model column by it, so ``M[b, t]`` becomes expected *relative density*
rather than mere occupancy.

Two reasons this matters here beyond fitting accuracy. It is the only thing
that makes ``M`` non-binary by construction, and a binary ``M`` is exactly why
Gate B found Poisson indistinguishable from least squares and Gate D found NB2
unable to move a point estimate at all: with ``mu`` constant across a model's
support, every variance function gives the same estimating equation. A graded
design breaks that degeneracy.

Method, following Zhao & Dukler (2021):

1. Select genes whose signal can be attributed to them alone -- long enough to
   have a body, isolated enough not to overlap a neighbour, expressed enough to
   have a measurable shape.
2. Bin each one, orient it 5' to 3', and divide by its own mean, giving a
   scale-free relative density.
3. Map bin positions onto a common ``[0, 1]`` axis under a tripartite rescaling
   -- a fixed-length head, a fixed-length tail, and a proportionally stretched
   middle -- so that the pausing peak, which sits a fixed *distance* from the
   TSS, lands at the same place for a 10 kb and a 200 kb gene.
4. Smooth the pooled cloud into a curve, and normalize it to 1 in the body.

Departures from the R reference, each one fixing a defect reproduced against
DENR 1.0.0 at commit ``09d6cde``. Section 18 of the plan records the evidence.

- **Strand is carried as data, never matched by name.** The reference selects
  plus-strand groups with ``grep("+", names)``, and ``+`` is a regex
  quantifier, so it matches *every* group. It then reads the minus groups from
  the plus BigWig as well, so ``bigwig_minus`` never contributes signal to the
  profile at all. Measured effect on 255 simulated genes: 116 of 174 training
  curves were identically zero and the fitted profile spanned 1.25x instead of
  the true 14.6x. The profile was not crashing there -- it was silently inert.
- **A gene is dropped, never rescued into a constant.** The reference min-max
  rescales each gene with a ``- 1e-6`` guard so that a gene with no signal
  becomes a constant 1.0 instead of dividing by zero. That converts missing
  data into evidence for a flat profile. `fit_shape_profile` drops such genes
  and reports the count.
- **Relative density is the gene's mean, not its min-max range.** Min-max
  scaling makes the single largest bin -- normally the pausing spike -- define
  the scale for the whole gene, so the body of a sharply paused gene is
  compressed toward zero and the profile reflects pausing intensity rather than
  shape.
- **Position rescaling is total on ``[0, 1]`` and cannot raise.** The
  reference's rescaler indexes ``rescaled_pos[1:linear_head_pos]`` with
  ``linear_head_pos`` possibly zero, which is R for "elements 1 and 0", and
  assigns a zero-length sequence to it: ``Error: replacement has length zero``.
  It is reachable from the documented API whenever the tail region is longer
  than the head and the gene is short. Its degenerate one-bin tail also places
  the final bin of the gene at 0.8 rather than 1.0.
- **The smoother is evaluated where it is needed, so it cannot return NaN.**
  ``predict.loess`` has no values outside the convex hull of its training
  positions and returns NA there; those NAs would flow straight into the design
  matrix and then into an objective, where L-BFGS-B rejects a non-finite value
  and the whole locus fails. `_local_linear` fits at fixed grid nodes spanning
  ``[0, 1]``, and `ShapeProfile.evaluate` interpolates between them, so the
  profile is finite and positive on the whole interval by construction.
- **Defaults are self-consistent.** The reference defaults
  ``min_transcript_length = 8e3`` below ``linear_head_length +
  linear_tail_length = 15e3``, so with defaults every gene between 8 and 15 kb
  silently takes a fallback branch that gives it no body positions at all --
  it contributes only to ``[0, 0.2]`` and ``[0.8, 1]``. Here `min_length`
  defaults to head plus tail and is validated against it.

One interaction is worth knowing before turning this on. **A shape profile and
TSS masking compete for the same bins.** Masking asserts that signal near the
TSS is untrustworthy; a shape profile asserts that it is predictable and
supplies its magnitude. Applying both largely cancels the profile, because the
pausing peak is both the largest feature it measures and the first thing a mask
removes. Measured on the same simulated data, profile fitted at 5.59x:

===========================  =============
masking                      applied range
===========================  =============
none                                 5.59x
``mask_tss=(0, 250)``                4.59x
``mask_tss=(1000, 1000)``            1.94x
``mask_start_bins=(4, 4)``           2.76x
===========================  =============

So the reference's own +/-1 kb default would discard most of what the profile
has to say. `applied_dynamic_range` reports the range actually surviving in a
reshaped design, which is the number to look at -- not
`ShapeProfile.dynamic_range`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import polars as pl
from figwig import BigWigReader

from .annotation import Annotation
from .counts import SignalFile, read_binned
from .design import Design
from .loci import assign_loci

#: Floor applied to the fitted profile, in units of the normalized body
#: density. A model column entry of exactly zero means "this transcript does
#: not reach this bin", so a profile reaching zero would silently unassign a
#: bin from a transcript that does cover it. The floor is needed rather than
#: merely defensive: local linear regression can undershoot below zero at a
#: sharp edge, and the pausing peak is the sharpest edge there is. Measured on
#: simulated data, the raw smoother went negative just past the peak at
#: ``span=0.01``. `ShapeProfile.n_floored` reports how often it engaged.
PROFILE_FLOOR = 1e-3

#: Grid nodes at which the smoother is evaluated. The profile is a piecewise
#: linear interpolant of these, so this sets its resolution: 2001 nodes put
#: about 400 of them inside a 0.2-wide head region, which at the default 10 kb
#: head is 25 bases per node -- an order of magnitude finer than a bin.
GRID_NODES = 2001

Normalization = Literal["body_median", "whole_mean"]


@dataclass(frozen=True)
class ShapeConfig:
    """Shape-profile parameters.

    Attributes:
        bin_size: Bin width used to sample the training genes. Need not match
            the design's, since positions are rescaled to ``[0, 1]``, but
            matching it means the profile is measured at the resolution it will
            be applied at.
        head_bases: Length of the fixed-scale 5' head region, in bases. Chosen
            to contain the pausing peak.
        tail_bases: Length of the fixed-scale 3' tail region, in bases.
        head_fraction: Share of the ``[0, 1]`` axis given to the head.
        tail_fraction: Share of the ``[0, 1]`` axis given to the tail.
        span: Fraction of the training observations in each local
            neighbourhood of the smoother. **Not the reference's 0.4.** A
            pausing peak is a few hundred bases wide, which the head rescaling
            maps to well under a hundredth of the axis, while a span of 0.4
            gives a bandwidth of roughly 0.2 -- some thirty times wider than
            the feature it is meant to measure. Measured against the
            unsmoothed training cloud on simulated data with a 14x peak: the
            empirical profile spans 9.3x, ``span=0.4`` recovers 2.4x, and
            0.05 recovers 5.6x. Over-smoothing here is not a cosmetic issue,
            since flattening the profile is exactly equivalent to not having
            one.
        min_bandwidth: Smallest neighbourhood half-width, in axis units, used
            as a floor on the span-derived bandwidth. Defaults to two head
            bins' worth of axis, which is the resolution the profile is
            applied at, so a small training set cannot produce a profile
            spikier than the bins it multiplies.
        min_length: Shortest training gene accepted. Defaults to
            ``head_bases + tail_bases``, the shortest gene that can supply a
            head, a body and a tail; below that a gene has no body to
            contribute and the head and tail would have to be squeezed.
        min_locus_fraction: A training gene must span at least this share of
            its locus, which is how a gene overlapping a neighbour is excluded.
            The reference instead scored per-base agreement between annotation
            and observed transcription; this is the same intent, expressed as
            geometry, and needs no threshold on the signal.
        min_covered_fraction: A training gene must have signal in at least this
            share of its bins. The analogue of the reference's
            ``min_percent_transcribed``.
        min_mean_signal: A training gene must average at least this much signal
            per bin, so that shape is measured on genes rather than on noise.
        normalization: Where the profile is pinned to 1. ``"body_median"``
            matches the reference and makes an abundance a body density.
            ``"whole_mean"`` pins the mean over the whole axis, which keeps
            total predicted signal comparable to the uniform-density model.
            Note this is not a global rescaling: because each gene's positions
            are stretched to its own length, the choice shifts abundances
            between genes of different lengths, not just overall.
        allow_short_genes: Permit genes shorter than ``head_bases +
            tail_bases`` by shrinking the head and tail proportionally. Off by
            default; such a gene contributes no body positions, and in the
            reference this branch was silently the common case.
    """

    bin_size: int = 250
    head_bases: int = 10_000
    tail_bases: int = 5_000
    head_fraction: float = 0.2
    tail_fraction: float = 0.2
    span: float = 0.05
    min_bandwidth: float | None = None
    min_length: int | None = None
    min_locus_fraction: float = 0.9
    min_covered_fraction: float = 0.5
    min_mean_signal: float = 0.1
    normalization: Normalization = "body_median"
    allow_short_genes: bool = False

    def __post_init__(self) -> None:
        if self.bin_size <= 0:
            raise ValueError(f"bin_size must be positive, got {self.bin_size}")
        for name in ("head_bases", "tail_bases"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.bin_size >= self.head_bases or self.bin_size >= self.tail_bases:
            raise ValueError(
                f"bin_size ({self.bin_size}) must be smaller than head_bases "
                f"({self.head_bases}) and tail_bases ({self.tail_bases}); "
                "otherwise the fixed-scale regions are under one bin wide"
            )
        for name in ("head_fraction", "tail_fraction"):
            value = getattr(self, name)
            if not 0 < value < 0.5:
                raise ValueError(f"{name} must be in (0, 0.5), got {value}")
        if not 0 < self.span <= 1:
            raise ValueError(f"span must be in (0, 1], got {self.span}")
        if self.min_bandwidth is not None and not 0 < self.min_bandwidth <= 1:
            raise ValueError(f"min_bandwidth must be in (0, 1], got {self.min_bandwidth}")
        for name in ("min_locus_fraction", "min_covered_fraction"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.min_mean_signal < 0:
            raise ValueError(f"min_mean_signal must be non-negative, got {self.min_mean_signal}")
        if self.normalization not in ("body_median", "whole_mean"):
            raise ValueError(
                f"normalization must be 'body_median' or 'whole_mean', got {self.normalization!r}"
            )
        if self.min_length is not None:
            if self.min_length <= 0:
                raise ValueError(f"min_length must be positive, got {self.min_length}")
            if self.min_length < self.head_bases + self.tail_bases and not self.allow_short_genes:
                raise ValueError(
                    f"min_length ({self.min_length}) is below head_bases + tail_bases "
                    f"({self.head_bases + self.tail_bases}), so training genes would have "
                    "no body region. Raise min_length or set allow_short_genes=True to "
                    "accept a proportionally squeezed head and tail."
                )

    @property
    def effective_min_length(self) -> int:
        """Shortest accepted gene, defaulting to head plus tail."""
        if self.min_length is not None:
            return self.min_length
        return self.head_bases + self.tail_bases

    @property
    def effective_min_bandwidth(self) -> float:
        """Bandwidth floor, defaulting to two head bins' worth of axis."""
        if self.min_bandwidth is not None:
            return self.min_bandwidth
        head_bins = -(-self.head_bases // self.bin_size)
        return 2.0 * self.head_fraction / head_bins


@dataclass(frozen=True)
class ShapeProfile:
    """A fitted relative-density curve on ``[0, 1]``.

    Attributes:
        grid: Ascending positions spanning exactly ``[0, 1]``.
        values: Profile value at each grid node, strictly positive and finite.
        config: The `ShapeConfig` used.
        n_genes: Training genes that contributed.
        n_observations: Training bins that contributed.
        n_floored: Grid nodes clamped up to `PROFILE_FLOOR`. Nonzero means the
            smoother undershot below zero somewhere -- normally just past the
            pausing peak -- so the profile there is a floor, not a fit.
        rejected: Gene counts per rejection reason, for auditing what the
            filters removed rather than reporting only what survived.
        digest: Content digest, folded into a reshaped `Design`'s digest.
    """

    grid: np.ndarray
    values: np.ndarray
    config: ShapeConfig
    n_genes: int
    n_observations: int
    n_floored: int
    rejected: dict[str, int]
    digest: str

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Profile value at `positions`, by linear interpolation on the grid.

        Total on ``[0, 1]`` and clamped outside it, so unlike the reference's
        ``predict.loess`` it has no region where it returns a missing value.
        """
        return np.interp(np.asarray(positions, dtype=np.float64), self.grid, self.values)

    @property
    def dynamic_range(self) -> float:
        """Ratio of the largest profile value to the smallest.

        Worth checking: a value near 1 means the profile is flat and reshaping
        the design will do nothing. That is the state the reference silently
        ended up in, so `fit_shape_profile` warns when this is near 1.
        """
        return float(self.values.max() / self.values.min())

    def describe(self) -> dict[str, float | str | int]:
        """Provenance record for the manifest."""
        return {
            "shape_digest": self.digest,
            "shape_n_genes": self.n_genes,
            "shape_n_observations": self.n_observations,
            "shape_dynamic_range": self.dynamic_range,
            "shape_n_floored": self.n_floored,
            "shape_normalization": self.config.normalization,
            "shape_span": self.config.span,
            "shape_head_bases": self.config.head_bases,
            "shape_tail_bases": self.config.tail_bases,
        }

    def table(self) -> pl.DataFrame:
        """The curve as a two-column frame, for writing or plotting."""
        return pl.DataFrame({"position": self.grid, "relative_density": self.values})


def rescale_position(
    n_bins: int,
    config: ShapeConfig,
    *,
    allow_short: bool | None = None,
) -> np.ndarray:
    """Map `n_bins` bins onto ``[0, 1]`` under the tripartite rescaling.

    A fixed head of ``head_bases`` occupies ``[0, head_fraction]``, a fixed
    tail of ``tail_bases`` occupies ``[1 - tail_fraction, 1]``, and whatever
    lies between is stretched over the rest. That is what lets a pausing peak
    at a fixed distance from the TSS appear at a fixed position for genes of
    very different lengths.

    Guarantees, none of which the reference's version provides: the result is
    ascending, has exactly `n_bins` entries, starts at 0.0 and -- for
    ``n_bins >= 2`` -- ends at 1.0, and never raises for any positive
    `n_bins`.

    Args:
        n_bins: Number of bins spanning the gene, in transcription order.
        config: Supplies the head and tail lengths and their axis shares.
        allow_short: Override `ShapeConfig.allow_short_genes`. When the head
            and tail together need more bins than the gene has, they are
            shrunk proportionally; with this false, that raises instead.

    Returns:
        A float64 array of length `n_bins`.

    Raises:
        ValueError: If `n_bins` is not positive, or if the gene is too short
            and short genes are not allowed.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if n_bins == 1:
        # One bin cannot express a shape; put it at the gene start.
        return np.zeros(1)

    permit_short = config.allow_short_genes if allow_short is None else allow_short
    head_bins = -(-config.head_bases // config.bin_size)  # ceil
    tail_bins = -(-config.tail_bases // config.bin_size)

    if head_bins + tail_bins > n_bins:
        if not permit_short:
            raise ValueError(
                f"a gene of {n_bins} bins cannot hold a {head_bins}-bin head and a "
                f"{tail_bins}-bin tail; raise min_length or set allow_short_genes=True"
            )
        # Split the available bins in the head:tail ratio, leaving at least one
        # bin on each side so that position 0.0 and position 1.0 both exist.
        share = config.head_bases / (config.head_bases + config.tail_bases)
        head_bins = int(np.clip(round(share * n_bins), 1, n_bins - 1))
        tail_bins = n_bins - head_bins

    body_bins = n_bins - head_bins - tail_bins
    head_stop = config.head_fraction
    tail_start = 1.0 - config.tail_fraction

    positions = np.empty(n_bins, dtype=np.float64)
    # linspace(0, head_stop, 1) is [0.0], which is what a one-bin head means.
    positions[:head_bins] = np.linspace(0.0, head_stop, head_bins)
    if tail_bins == 1:
        # A one-bin tail is the gene's final bin, so it belongs at 1.0. The
        # reference's seq(1 - tail_percent, 1, length.out = 1) puts it at
        # 1 - tail_percent instead, reporting the end of the gene as 80% of it.
        positions[-1] = 1.0
    else:
        positions[n_bins - tail_bins :] = np.linspace(tail_start, 1.0, tail_bins)
    if body_bins > 0:
        # Interior points only: the endpoints already belong to head and tail.
        positions[head_bins : head_bins + body_bins] = np.linspace(
            head_stop, tail_start, body_bins + 2
        )[1:-1]
    return positions


def fit_shape_profile(
    annotation: Annotation,
    signal: SignalFile | tuple[SignalFile, ...],
    *,
    config: ShapeConfig | None = None,
    warn_flat_below: float = 1.2,
) -> ShapeProfile:
    """Fit a shape profile from isolated, expressed genes.

    Args:
        annotation: Candidate training transcripts.
        signal: One `SignalFile` per strand. Each gene is read from the file
            matching *its own* strand -- the defect that made the reference's
            profile inert was reading minus-strand genes from the plus file.
        config: Parameters; defaults to `ShapeConfig()`.
        warn_flat_below: Emit a warning when the fitted dynamic range falls
            below this. A flat profile is not an error, but it means reshaping
            will not change the design, and the reference reached that state
            silently.

    Returns:
        A `ShapeProfile`.

    Raises:
        ValueError: If a needed strand has no signal file, two files claim one
            strand, or no gene survives the filters. The last case is an error
            rather than a flat profile because a profile fitted from nothing
            would multiply every design column by a constant and look like a
            successful run.
    """
    config = config or ShapeConfig()
    by_strand = _index_by_strand(signal)

    candidates, rejected = _select_training_genes(annotation, config)
    needed = set(candidates["strand"].unique().to_list()) if len(candidates) else set()
    missing = needed - set(by_strand)
    if missing:
        raise ValueError(f"training genes lie on strand(s) {sorted(missing)} with no signal file")

    # One reader per file, reused for every gene: figwig parses the data index
    # once per reader, so reopening per gene would re-read it each time.
    readers = {strand: BigWigReader(str(item.path)) for strand, item in by_strand.items()}
    positions: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    n_genes = 0

    for gene in candidates.iter_rows(named=True):
        strand = gene["strand"]
        if gene["contig"] not in readers[strand].chrom_sizes:
            rejected["missing_contig"] = rejected.get("missing_contig", 0) + 1
            continue
        width = gene["end"] - gene["start"]
        n_bins = -(-width // config.bin_size)
        values = read_binned(
            by_strand[strand].path,
            gene["contig"],
            gene["start"],
            gene["start"] + n_bins * config.bin_size,
            n_bins,
            summary="mean",
            reader=readers[strand],
        )
        # Orient 5' to 3'. A minus-strand gene's TSS is at its highest
        # coordinate, so bin order has to be reversed before it can share an
        # axis with a plus-strand gene.
        if strand == "-":
            values = values[::-1]

        covered = float(np.count_nonzero(values)) / n_bins
        if covered < config.min_covered_fraction:
            rejected["low_coverage"] = rejected.get("low_coverage", 0) + 1
            continue
        gene_mean = float(values.mean())
        if not np.isfinite(gene_mean) or gene_mean < config.min_mean_signal:
            rejected["low_signal"] = rejected.get("low_signal", 0) + 1
            continue

        # Relative density: expectation 1 by construction, so genes of any
        # expression level contribute on the same footing. A gene with no
        # signal never reaches here, so nothing is rescued into a constant.
        densities.append(values / gene_mean)
        positions.append(rescale_position(n_bins, config))
        n_genes += 1

    if n_genes == 0:
        raise ValueError(
            "no gene passed the shape-profile filters "
            f"({_format_rejections(rejected)}). Relax min_covered_fraction, "
            "min_mean_signal or min_locus_fraction, or check that the signal "
            "files match the annotation's contigs and strands."
        )

    x = np.concatenate(positions)
    y = np.concatenate(densities)
    grid = np.linspace(0.0, 1.0, GRID_NODES)
    fitted = _local_linear(x, y, grid, config.span, config.effective_min_bandwidth)
    # Normalize first, then floor, so that PROFILE_FLOOR means a fixed fraction
    # of body density rather than a fixed number on whatever scale the
    # smoother happened to land on.
    values = fitted / _normalizer(grid, fitted, config)
    n_floored = int(np.count_nonzero(values < PROFILE_FLOOR))
    values = np.maximum(values, PROFILE_FLOOR)

    profile = ShapeProfile(
        grid=grid,
        values=values,
        config=config,
        n_genes=n_genes,
        n_observations=int(x.size),
        n_floored=n_floored,
        rejected=dict(sorted(rejected.items())),
        digest=_shape_digest(config, values),
    )
    if profile.dynamic_range < warn_flat_below:
        import warnings

        warnings.warn(
            f"shape profile spans only {profile.dynamic_range:.3f}x from {n_genes} genes, "
            "so reshaping the design will barely change it. Check that each strand's "
            "signal file is the right one and that training genes are expressed.",
            stacklevel=2,
        )
    return profile


def apply_shape_profile(design: Design, profile: ShapeProfile) -> Design:
    """Reweight `design`'s model columns by `profile`, returning a new `Design`.

    Each model column's occupied bins are mapped onto ``[0, 1]`` along the
    direction of transcription and multiplied by the profile there. Multiplying
    rather than overwriting -- the reference overwrites -- is identical while
    `dienr.design.DesignConfig.bin_operation` rounds occupancies to 0 and 1,
    and stays correct if a fractional design is ever used.

    Positions come from `Design.model_spans`, the span *before* masking. Using
    the surviving nonzero bins instead would place the first unmasked bin at
    position 0, giving it the pausing-peak weight when it is really a mask
    width into the gene -- and the profile is at its steepest exactly there.
    This is also what lets the mapping be a direct one: the reference had to
    average member transcripts' coordinates into an "expected" bin range and
    then delete the entries that rounding or masking had removed, keeping two
    parallel index spaces aligned by construction. Here an equivalence class
    has one span because its members are geometrically indistinguishable, which
    is the point of collapsing them.

    Args:
        design: Geometry to reshape. Unchanged; a new `Design` is returned.
        profile: The fitted curve.

    Returns:
        A `Design` with reweighted `models` and a digest that reflects the
        profile, so a reshaped design can never be mistaken for its unshaped
        source.

    Raises:
        ValueError: If `design` has already been reshaped.
    """
    if design.shape_digest is not None:
        raise ValueError(
            f"design has already been reshaped by profile {design.shape_digest}; "
            "rebuild it from the annotation to apply a different profile"
        )

    strands = design.loci["strand"].to_list()
    models: list[np.ndarray] = []
    for locus_id, matrix in enumerate(design.models):
        reshaped = matrix.copy()
        spans = design.model_spans[locus_id]
        minus = strands[locus_id] == "-"
        for model_index in range(reshaped.shape[1]):
            first, last = int(spans[model_index, 0]), int(spans[model_index, 1])
            if first < 0:
                continue  # no occupied bin even before masking
            n_span = last - first + 1
            positions = rescale_position(n_span, profile.config, allow_short=True)
            weights = profile.evaluate(positions)
            if minus:
                # Positions run 5' to 3'; bin index runs along the genome.
                weights = weights[::-1]
            reshaped[first : last + 1, model_index] *= weights
        models.append(np.ascontiguousarray(reshaped))

    return replace(
        design,
        models=models,
        shape_digest=profile.digest,
        digest=_reshaped_digest(design.digest, profile.digest),
    )


def applied_dynamic_range(design: Design) -> float:
    """Ratio of the largest to smallest nonzero model weight in `design`.

    For a reshaped design this is how much grading actually survived masking,
    which is the figure that matters: `ShapeProfile.dynamic_range` describes
    the curve, not the design it was applied to, and masking the TSS removes
    the peak the curve is mostly about. Returns 1.0 for an unshaped design,
    whose weights are all 1.
    """
    values = [matrix[matrix != 0] for matrix in design.models if matrix.size]
    nonzero = np.concatenate(values) if values else np.array([])
    if nonzero.size == 0:
        return 1.0
    return float(nonzero.max() / nonzero.min())


def _index_by_strand(
    signal: SignalFile | tuple[SignalFile, ...],
) -> dict[str, SignalFile]:
    """Map strand to signal file, rejecting duplicates."""
    files = (signal,) if isinstance(signal, SignalFile) else tuple(signal)
    by_strand: dict[str, SignalFile] = {}
    for item in files:
        if item.strand in by_strand:
            raise ValueError(f"two signal files given for strand {item.strand!r}")
        by_strand[item.strand] = item
    return by_strand


def _select_training_genes(
    annotation: Annotation,
    config: ShapeConfig,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Pick one isolated, long transcript per locus.

    Returns the candidates and a per-reason tally of what was dropped.
    """
    transcripts = annotation.transcripts
    rejected: dict[str, int] = {}

    long_enough = transcripts.filter(
        (pl.col("end") - pl.col("start")) >= config.effective_min_length
    )
    rejected["too_short"] = len(transcripts) - len(long_enough)
    if not len(long_enough):
        return long_enough, rejected

    # Group at zero distance: any two transcripts that touch share a locus, so
    # a gene whose locus it does not nearly fill has a neighbour whose signal
    # would contaminate it.
    grouped = assign_loci(long_enough, distance=0)
    with_width = grouped.with_columns(_width=pl.col("end") - pl.col("start"))
    # The longest transcript of each locus, ties broken by id for determinism.
    longest = (
        with_width.sort(["locus_id", "_width", "transcript_id"], descending=[False, True, False])
        .group_by("locus_id", maintain_order=True)
        .first()
    )
    rejected["not_longest_in_locus"] = len(with_width) - len(longest)

    locus_extent = grouped.group_by("locus_id").agg(
        _locus_start=pl.col("start").min(), _locus_end=pl.col("end").max()
    )
    joined = longest.join(locus_extent, on="locus_id", how="left").with_columns(
        _fraction=pl.col("_width") / (pl.col("_locus_end") - pl.col("_locus_start"))
    )
    isolated = joined.filter(pl.col("_fraction") >= config.min_locus_fraction)
    rejected["overlapping_neighbour"] = len(joined) - len(isolated)

    return isolated.select("transcript_id", "contig", "start", "end", "strand"), rejected


def _local_linear(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    span: float,
    min_bandwidth: float,
) -> np.ndarray:
    """Tricube-weighted local linear regression evaluated at `grid`.

    A LOESS fit, with two deliberate differences from ``stats::loess``.

    It is evaluated *at* the grid nodes rather than fitted once and
    interpolated through a kd-tree, so there is no region outside a convex hull
    where it has no value: every node in ``[0, 1]`` gets a number, and a node
    whose neighbourhood is empty falls back to the nearest node that is not.
    That is what makes NaN unreachable in the design matrix.

    And the observations are first aggregated onto the grid. Training positions
    are not scattered -- every gene of the same bin count contributes the
    identical position vector -- so grouping is nearly lossless, and for the
    fitted mean it is exactly equivalent to weighting each distinct position by
    its multiplicity, since the least-squares design depends only on ``x``.
    It also bounds the cost: the reference pools millions of rows and hands
    them to a Fortran smoother that emits R stack-imbalance warnings at that
    size.

    Args:
        x: Positions in ``[0, 1]``.
        y: Relative densities.
        grid: Ascending evaluation nodes.
        span: Fraction of observations in each neighbourhood.
        min_bandwidth: Floor on the neighbourhood half-width, in axis units.

    Returns:
        Fitted values at `grid`, finite everywhere. Not guaranteed
        non-negative: local linear regression undershoots at a sharp edge, so
        the caller floors the result. See `PROFILE_FLOOR`.
    """
    n_nodes = grid.size
    # Aggregate onto the nearest grid node.
    index = np.clip(np.rint(x * (n_nodes - 1)).astype(np.int64), 0, n_nodes - 1)
    counts = np.bincount(index, minlength=n_nodes).astype(np.float64)
    sums = np.bincount(index, weights=y, minlength=n_nodes)
    occupied = counts > 0
    nodes = grid[occupied]
    weight = counts[occupied]
    response = sums[occupied] / weight
    total = float(weight.sum())

    if nodes.size == 1:
        return np.full(n_nodes, float(response[0]))

    # Neighbourhood radius per grid node: the smallest radius holding at least
    # span * total observations. Bisection, vectorized over the grid.
    cumulative = np.concatenate([[0.0], np.cumsum(weight)])
    target = max(span * total, np.finfo(float).tiny)

    def enclosed(radius: np.ndarray) -> np.ndarray:
        hi = np.searchsorted(nodes, grid + radius, side="right")
        lo = np.searchsorted(nodes, grid - radius, side="left")
        return cumulative[hi] - cumulative[lo]

    low = np.zeros(n_nodes)
    high = np.full(n_nodes, 1.0)
    for _ in range(60):
        mid = 0.5 * (low + high)
        enough = enclosed(mid) >= target
        high = np.where(enough, mid, high)
        low = np.where(enough, low, mid)
    # The floor does two jobs: it keeps the kernel from collapsing onto a
    # single node, and it stops a sparse training set from being fitted more
    # finely than the bins the profile will multiply.
    radius = np.maximum(high, max(min_bandwidth, 1.0 / (n_nodes - 1)))

    fitted = np.full(n_nodes, np.nan)
    # Chunk the (grid x nodes) kernel so memory stays bounded for a fine grid.
    chunk = max(1, int(4_000_000 // max(nodes.size, 1)))
    for begin in range(0, n_nodes, chunk):
        stop = min(begin + chunk, n_nodes)
        offset = nodes[None, :] - grid[begin:stop, None]
        scaled = np.abs(offset) / radius[begin:stop, None]
        kernel = np.where(scaled < 1.0, (1.0 - scaled**3) ** 3, 0.0) * weight[None, :]
        s0 = kernel.sum(axis=1)
        s1 = (kernel * offset).sum(axis=1)
        s2 = (kernel * offset**2).sum(axis=1)
        t0 = (kernel * response[None, :]).sum(axis=1)
        t1 = (kernel * offset * response[None, :]).sum(axis=1)
        # Solve the 2x2 weighted normal equations; the intercept is the fit at
        # the node, since offsets are centred there.
        determinant = s0 * s2 - s1**2
        with np.errstate(divide="ignore", invalid="ignore"):
            local = np.where(
                np.abs(determinant) > 1e-12 * np.maximum(s0 * s2, 1e-300),
                (t0 * s2 - t1 * s1) / determinant,
                np.where(s0 > 0, t0 / np.maximum(s0, 1e-300), np.nan),
            )
        fitted[begin:stop] = local

    return _fill_gaps(fitted)


def _fill_gaps(values: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the nearest finite one.

    The last line of defence for the no-NaN guarantee: a degenerate
    neighbourhood must still yield a number, because a NaN here becomes a NaN
    in a design matrix and then a failed locus.
    """
    good = np.isfinite(values)
    if good.all():
        return values
    if not good.any():
        raise ValueError("local regression produced no finite value anywhere")
    index = np.arange(values.size)
    return np.interp(index, index[good], values[good])


def _normalizer(grid: np.ndarray, values: np.ndarray, config: ShapeConfig) -> float:
    """Divisor that pins the profile to 1 where `config` says.

    Raises:
        ValueError: If the reference level is not positive. Dividing by it
            would flip the profile's sign and turn every design entry
            negative, which no downstream check would recognize as a
            measurement failure rather than a model.
    """
    if config.normalization == "whole_mean":
        level = float(values.mean())
    else:
        body = (grid >= config.head_fraction) & (grid <= 1.0 - config.tail_fraction)
        level = float(np.median(values[body])) if body.any() else float(np.median(values))
    if not np.isfinite(level) or level <= 0:
        raise ValueError(
            f"shape profile reference level is {level}, not positive, so the fitted "
            "curve carries no usable gene-body density. Check that training genes are "
            "expressed and that span is not so small that the fit is noise."
        )
    return level


def _format_rejections(rejected: dict[str, int]) -> str:
    parts = [f"{reason}: {count}" for reason, count in sorted(rejected.items()) if count]
    return ", ".join(parts) if parts else "no candidates at all"


def _shape_digest(config: ShapeConfig, values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for field in sorted(config.__dataclass_fields__):
        hasher.update(f"{field}={getattr(config, field)}".encode())
    hasher.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return hasher.hexdigest()[:16]


def _reshaped_digest(design_digest: str, shape_digest: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(design_digest.encode())
    hasher.update(b"|shape|")
    hasher.update(shape_digest.encode())
    return hasher.hexdigest()[:16]
