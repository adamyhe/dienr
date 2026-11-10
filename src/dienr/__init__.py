"""dienr: Deconvolution of Isoform Expression from Nascent RNAs.

A Python reimplementation of the DENR method, which explains observed nascent
RNA signal at a locus as a nonnegative mixture of pre-RNA isoform models.

Typical use::

    import dienr

    annotation = dienr.read_annotation("gencode.gtf")
    design = dienr.build_design(annotation, bin_size=250, mask_start_bins=(4, 4))
    counts = dienr.count_sample(design, "sample1", (plus, minus))
    result = dienr.fit(design, counts)
    result.write("results/")

Three likelihoods are available through `fit`. ``"legacy"`` is the default and
the only one checked against R DENR. The count likelihoods --- ``"poisson"``
and ``"quasi_poisson"`` --- require count-equivalent
input, so counts must be extracted with ``summary="sum"`` rather than the
parity default of ``"mean"``; `fit` rejects the mismatch rather than quietly
fitting a density.

``"quasi_poisson"`` returns the same abundances as ``"poisson"`` and differs
only in the reported standard errors. It is the recommended choice for
overdispersed counts: a Poisson standard error understates the true spread by
23% at dispersion 0.1 and 64% at 1.0, and quasi-Poisson matched a negative
binomial's errors to within a few percent, which is why there is no NB solver
(plan Gate D).

Coordinates are 0-based half-open throughout; see `dienr.annotation` for the
single conversion point.
"""

from importlib import metadata as _metadata

from .annotation import Annotation, read_annotation
from .contigs import ContigMap, check_contigs, rename_contigs
from .counts import (
    SampleCounts,
    SignalFile,
    count_sample,
    declared_contigs,
    read_binned,
    read_loci_binned,
)
from .design import Design, DesignConfig, build_design
from .diagnostics import LocusConditioning, assess
from .loci import assign_loci, locus_table
from .pipeline import fit, status_report
from .results import SCHEMA_VERSION, FitResult, RunSummary, collect_results
from .runon import RunonConfig, RunonTail, apply_runon_tail, fit_runon_tail
from .shape import (
    ShapeConfig,
    ShapeProfile,
    applied_dynamic_range,
    apply_shape_profile,
    fit_shape_profile,
    rescale_position,
)
from .solvers import inference, penalties, poisson
from .solvers._glm import Exposure
from .solvers.legacy import LocusFit, fit_locus
from .solvers.penalties import NO_PENALTY, Penalty
from .status import FitStatus, is_failure
from .strand import StrandCheck, check_strand_orientation

try:
    # pyproject.toml is the only place the version is written.
    __version__ = _metadata.version("dienr")
except _metadata.PackageNotFoundError:  # imported from a source tree, not installed
    __version__ = "0+unknown"

__all__ = [
    "SCHEMA_VERSION",
    "Annotation",
    "Design",
    "DesignConfig",
    "NO_PENALTY",
    "Exposure",
    "FitResult",
    "FitStatus",
    "LocusConditioning",
    "LocusFit",
    "Penalty",
    "RunSummary",
    "RunonConfig",
    "RunonTail",
    "SampleCounts",
    "SignalFile",
    "StrandCheck",
    "ContigMap",
    "check_contigs",
    "rename_contigs",
    "ShapeConfig",
    "ShapeProfile",
    "__version__",
    "applied_dynamic_range",
    "apply_runon_tail",
    "apply_shape_profile",
    "assess",
    "assign_loci",
    "build_design",
    "check_strand_orientation",
    "collect_results",
    "count_sample",
    "declared_contigs",
    "fit",
    "fit_locus",
    "fit_runon_tail",
    "fit_shape_profile",
    "is_failure",
    "inference",
    "locus_table",
    "penalties",
    "poisson",
    "read_annotation",
    "read_binned",
    "read_loci_binned",
    "rescale_position",
    "status_report",
]
