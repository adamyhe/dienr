"""Does extending the windows fix the run-on tail's bias? (plan 18.26)

18.25 found the tail biased toward shorter isoforms: a locus window ended at
its most downstream PAS, so that isoform's tail fell outside it while shorter
isoforms' tails landed on its 3' body. `apply_runon_tail` now extends each
window downstream by the tail's length (stopping at the next same-strand
locus). This re-tests the tail with 18.20's truth and scoring
(`k562_longread_accuracy.main`).

**Arms** (250 bp bins):

- ``runon_fixed`` -- Poisson, TSS mask ``(1, 1)``, no PAS mask, tail, windows
  extended (the abundances configuration, as now built).
- ``runon_clipped`` -- the same with ``extend=False`` (18.24's ``poisson_runon``).
- ``poisson_tss`` -- Poisson, TSS mask ``(1, 1)``, no tail.
- ``tail_fixed`` -- Poisson, no masks, tail, extended.
- ``tuned_log`` -- defaults + ``c = 1`` (the dominant-isoforms configuration).
- ``log_tail_fixed`` -- tuned log with the extended tail.

"Better" below means, on a PRO-seq replicate: higher dominant-group accuracy
AND > 50% of the loci where exactly one is correct, with a two-sided sign test
p < 0.05.

**Decision rules, fixed before running**, each on **both** replicates:

1. **The fix works** if ``runon_fixed`` is better than ``runon_clipped``.
2. **The tail stays in the abundances configuration** only if ``poisson_tss``
   is not better than ``runon_fixed`` on either replicate; otherwise the docs
   drop the tail from it.
3. **18.24's head-to-head, re-applied:** ``tuned_log`` vs ``runon_fixed`` with
   18.24's three outcomes (tuned log better on both -> two configurations;
   ``runon_fixed`` better on both -> one Poisson configuration; anything else
   -> one Poisson configuration, tuned log as the DENR-compatible option).

Reported, not decided on: ``tail_fixed`` vs ``poisson_tss``; ``log_tail_fixed``
vs ``tuned_log`` (18.21 found the clipped tail hurt the log objective).

    uv run python scripts/k562_longread_tailfix.py \\
        --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --longread build/k562/longread --score rep1 --tail-from rep2 \\
        --output build/k562/tailfix_rep1.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_longread_accuracy import main as run  # noqa: E402

_TSS = {"tss": (1, 1), "pas": (0, 0)}
_NOMASK = {"tss": (0, 0), "pas": (0, 0)}
ARMS: dict[str, dict] = {
    "runon_fixed": {**_TSS, "tail": True, "likelihood": "poisson"},
    "runon_clipped": {**_TSS, "tail": True, "extend": False, "likelihood": "poisson"},
    "poisson_tss": {**_TSS, "tail": False, "likelihood": "poisson"},
    "tail_fixed": {**_NOMASK, "tail": True, "likelihood": "poisson"},
    "tuned_log": {**_NOMASK, "tail": False, "likelihood": "legacy", "log_offset": 1.0},
    "log_tail_fixed": {**_NOMASK, "tail": True, "likelihood": "legacy", "log_offset": 1.0},
}
# `rule_passes_by_candidate` in the output then reads: runon_fixed -> rule 1
# passes; poisson_tss -> rule 2 fails (if on both replicates); tuned_log -> rule
# 3's first outcome. Every pairwise sign test is reported either way.
CANDIDATES = {
    "runon_fixed": ("runon_clipped",),
    "poisson_tss": ("runon_fixed",),
    "tuned_log": ("runon_fixed",),
}
EXTRA_PAIRS = (("tail_fixed", "poisson_tss"), ("log_tail_fixed", "tuned_log"))


if __name__ == "__main__":
    raise SystemExit(
        run(arms=ARMS, candidates=CANDIDATES, extra_pairs=EXTRA_PAIRS, description=__doc__)
    )
