"""Does DENR's log objective plus no PAS mask call isoforms best? (plan 18.21)

18.20 found two separate effects on dominant-isoform accuracy against K562
long reads: dropping the PAS mask helps (+4.7 points with Poisson, both
replicates), and the reference's log least squares beats Poisson at dominant
calls (~5 points with the paper's mask, both replicates). They may stack.
This tests the combination, with everything else exactly as in 18.20 (same
truth, matching, groups, evaluable-locus filter, scoring;
`k562_longread_accuracy.main`). Groups are rebuilt from these arms' designs,
so the evaluable set can differ slightly from 18.20's.

**Arms** (TSS mask ``(1, 1)``; tail from the other PRO-seq replicate):

- ``denr_default`` -- PAS mask ``(4, 4)``, no tail, legacy log, ``mean``.
- ``log_0_0`` -- **no PAS mask**, no tail, legacy log, ``mean``.
- ``log_runon_0_0`` -- no PAS mask, **tail**, legacy log, ``mean``.
- ``recommended`` -- no PAS mask, tail, Poisson, ``sum`` (18.19).
- ``binary_0_0`` -- no PAS mask, no tail, Poisson, ``sum``.

**Decision rule, fixed before running.** Two candidates, ``log_0_0`` and
``log_runon_0_0``, each tested against **both** ``denr_default`` and
``recommended``. A candidate passes if, on **both** PRO-seq replicates and
against each reference:

1. higher dominant-group accuracy, AND
2. it wins > 50% of the loci where exactly one of the two is correct, with a
   two-sided sign test **p < 0.025** (0.05 split over the two candidates).

If exactly one passes, it becomes the recommended configuration *for calling
dominant isoforms* (Poisson stays the recommendation for count fits and
standard errors); if both pass, the one with the higher mean accuracy over the
two replicates. If neither passes, 18.20's conclusions stand.

    uv run python scripts/k562_longread_logmask.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --longread build/k562/longread --score rep1 --tail-from rep2 \\
        --output build/k562/logmask_rep1.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_longread_accuracy import main as run  # noqa: E402

ARMS: dict[str, dict] = {
    "denr_default": {"pas": (4, 4), "tail": False, "likelihood": "legacy"},
    "log_0_0": {"pas": (0, 0), "tail": False, "likelihood": "legacy"},
    "log_runon_0_0": {"pas": (0, 0), "tail": True, "likelihood": "legacy"},
    "recommended": {"pas": (0, 0), "tail": True, "likelihood": "poisson"},
    "binary_0_0": {"pas": (0, 0), "tail": False, "likelihood": "poisson"},
}
CANDIDATES = {
    "log_0_0": ("denr_default", "recommended"),
    "log_runon_0_0": ("denr_default", "recommended"),
}
EXTRA_PAIRS = (
    ("log_0_0", "binary_0_0"),  # objective effect without the tail
    ("log_runon_0_0", "log_0_0"),  # tail effect under the log objective
    ("recommended", "denr_default"),  # 18.20's primary comparison, re-run
)


if __name__ == "__main__":
    raise SystemExit(
        run(
            arms=ARMS,
            candidates=CANDIDATES,
            extra_pairs=EXTRA_PAIRS,
            alpha=0.025,
            description=__doc__,
        )
    )
