"""Tuned log objective vs the Poisson configuration, head to head (plan 18.24).

The docs recommend two configurations: for dominant isoforms, DENR's defaults
plus ``fit --log-offset 1`` (18.23); for magnitudes and standard errors,
quasi-Poisson with a run-on tail, a 1-bin TSS mask and no PAS mask (18.19).
They were never compared directly at the tuned offset. If the Poisson
configuration calls dominant isoforms as well, one configuration serves both
goals and the docs get simpler; if the tuned log objective is clearly better,
two configurations are justified. 18.20's truth and scoring
(`k562_longread_accuracy.main`).

**Arms** (250 bp bins, default locus grouping):

- ``tuned_log`` -- no masks, no tail, legacy log, ``c = 1``, ``mean``.
- ``poisson_runon`` -- TSS mask ``(1, 1)``, no PAS mask, tail, Poisson,
  ``sum`` (the abundances recommendation).
- ``defaults`` -- as ``tuned_log`` with ``c = 0.001`` (DENR's defaults).
- ``identity`` -- as ``tuned_log`` with ``--transform identity``, which a large
  offset approximates.

**Decision rule, fixed before running.** ``tuned_log`` vs ``poisson_runon``,
on each PRO-seq replicate: "better" means higher accuracy AND > 50% of the
loci where exactly one is correct, with a two-sided sign test p < 0.05.

1. ``tuned_log`` better on **both** replicates -> keep two recommended
   configurations (tuned log for dominant isoforms).
2. ``poisson_runon`` better on **both** -> one recommended configuration,
   Poisson / quasi-Poisson with the tail, for both goals; tuned log listed as
   the DENR-compatible option.
3. Anything else (equivalent or mixed) -> also one recommended configuration,
   Poisson with the tail, because it calls isoforms as well and adds
   count-scale abundances and standard errors; tuned log listed as the
   DENR-compatible option.

Reported, not part of the rule: ``tuned_log`` vs ``defaults`` (18.23's gain
on this locus set), ``identity`` vs ``tuned_log``.

    uv run python scripts/k562_longread_headtohead.py \\
        --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --longread build/k562/longread --score rep1 --tail-from rep2 \\
        --output build/k562/headtohead_rep1.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_longread_accuracy import main as run  # noqa: E402

_LOG = {"tss": (0, 0), "pas": (0, 0), "tail": False, "likelihood": "legacy"}
ARMS: dict[str, dict] = {
    "tuned_log": {**_LOG, "log_offset": 1.0},
    "poisson_runon": {"tss": (1, 1), "pas": (0, 0), "tail": True, "likelihood": "poisson"},
    "defaults": {**_LOG, "log_offset": 1e-3},
    "identity": {**_LOG, "transform": "identity"},
}
CANDIDATES = {"tuned_log": ("poisson_runon",), "poisson_runon": ("tuned_log",)}
EXTRA_PAIRS = (("tuned_log", "defaults"), ("identity", "tuned_log"))


if __name__ == "__main__":
    raise SystemExit(
        run(arms=ARMS, candidates=CANDIDATES, extra_pairs=EXTRA_PAIRS, description=__doc__)
    )
