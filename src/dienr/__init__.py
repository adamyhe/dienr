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
the only one checked against R DENR. ``"poisson"`` and ``"negative_binomial"``
require count-equivalent input, so counts must be extracted with
``summary="sum"`` rather than the parity default of ``"mean"``; `fit` rejects
the mismatch rather than quietly fitting a density.

Coordinates are 0-based half-open throughout; see `dienr.annotation` for the
single conversion point.
"""

from .annotation import Annotation, read_annotation
from .counts import SampleCounts, SignalFile, count_sample, declared_contigs, read_binned
from .design import Design, DesignConfig, build_design
from .diagnostics import LocusConditioning, assess
from .loci import assign_loci, locus_table
from .pipeline import fit, status_report
from .results import SCHEMA_VERSION, FitResult, RunSummary, collect_results
from .solvers import negative_binomial, penalties, poisson
from .solvers._glm import Exposure
from .solvers.legacy import LocusFit, fit_locus
from .solvers.penalties import NO_PENALTY, Penalty
from .status import FitStatus, is_failure

__version__ = "0.1.0.dev0"

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
    "SampleCounts",
    "SignalFile",
    "__version__",
    "assess",
    "assign_loci",
    "build_design",
    "collect_results",
    "count_sample",
    "declared_contigs",
    "fit",
    "fit_locus",
    "is_failure",
    "locus_table",
    "negative_binomial",
    "penalties",
    "poisson",
    "read_annotation",
    "read_binned",
    "status_report",
]
