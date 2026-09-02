"""dienr: Deconvolution of Isoform Expression from Nascent RNAs.

A Python reimplementation of the DENR method, which explains observed nascent
RNA signal at a locus as a nonnegative mixture of pre-RNA isoform models.

Typical use::

    import dienr

    annotation = dienr.read_annotation("gencode.gtf")
    design = dienr.build_design(annotation, bin_size=250, mask_start_bins=(4, 4))
    counts = dienr.count_sample(design, "sample1", (plus, minus))
    fits = [
        dienr.fit_locus(design.models[i], counts.values[i],
                        prior_status=counts.locus_status[i])
        for i in range(design.n_loci)
    ]

Coordinates are 0-based half-open throughout; see `dienr.annotation` for the
single conversion point.
"""

from .annotation import Annotation, read_annotation
from .counts import SampleCounts, SignalFile, count_sample, declared_contigs, read_binned
from .design import Design, DesignConfig, build_design
from .diagnostics import LocusConditioning, assess
from .loci import assign_loci, locus_table
from .solvers.legacy import LocusFit, fit_locus
from .status import FitStatus, is_failure

__version__ = "0.1.0.dev0"

__all__ = [
    "Annotation",
    "Design",
    "DesignConfig",
    "FitStatus",
    "LocusConditioning",
    "LocusFit",
    "SampleCounts",
    "SignalFile",
    "__version__",
    "assess",
    "assign_loci",
    "build_design",
    "count_sample",
    "declared_contigs",
    "fit_locus",
    "is_failure",
    "locus_table",
    "read_annotation",
    "read_binned",
]
