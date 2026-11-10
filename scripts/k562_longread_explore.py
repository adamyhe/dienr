"""EXPLORATORY: why the Poisson configuration lost 18.24 so badly (plan 18.24).

Not a pre-registered trial; nothing here changes a recommendation. 18.24 found
the abundances configuration (Poisson, run-on tail, TSS mask (1, 1), no PAS
mask) calling dominant isoforms at 0.631 / 0.585 against 0.771 / 0.773 for the
tuned log objective -- far below its 0.778 / 0.746 in 18.20-18.22. The
evaluable loci differ: earlier trials included a paper-masked arm, whose 4-bin
PAS mask merges isoforms with PAS sites within ~1 kb, so loci distinguished
mainly by alternative polyadenylation dropped out; 18.24 has no such arm.

This separates the configuration's ingredients on 18.24's locus set (same
arms' groups, plus the extras below, which can only merge groups further if
they collapse models; reported funnel shows the set):

- ``tuned_log`` and ``poisson_runon`` -- 18.24's arms, for reference.
- ``poisson_plain`` -- Poisson, no masks, no tail: the objective alone.
- ``poisson_tss`` -- Poisson, TSS mask (1, 1), no tail: adds the mask.
- ``poisson_tail`` -- Poisson, no masks, tail: adds the tail.
- ``log_tail`` -- tuned log with the tail.

    uv run python scripts/k562_longread_explore.py --gtf ... --bigwigs build/k562 \\
        --longread build/k562/longread --score rep1 --tail-from rep2 \\
        --output build/k562/explore_rep1.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_longread_accuracy import main as run  # noqa: E402

_NOMASK = {"tss": (0, 0), "pas": (0, 0)}
ARMS: dict[str, dict] = {
    "tuned_log": {**_NOMASK, "tail": False, "likelihood": "legacy", "log_offset": 1.0},
    "poisson_runon": {"tss": (1, 1), "pas": (0, 0), "tail": True, "likelihood": "poisson"},
    "poisson_plain": {**_NOMASK, "tail": False, "likelihood": "poisson"},
    "poisson_tss": {"tss": (1, 1), "pas": (0, 0), "tail": False, "likelihood": "poisson"},
    "poisson_tail": {**_NOMASK, "tail": True, "likelihood": "poisson"},
    "log_tail": {**_NOMASK, "tail": True, "likelihood": "legacy", "log_offset": 1.0},
}
EXTRA_PAIRS = (
    ("poisson_plain", "tuned_log"),
    ("poisson_tail", "poisson_plain"),
    ("poisson_tss", "poisson_plain"),
    ("poisson_runon", "poisson_tss"),
    ("log_tail", "tuned_log"),
)


if __name__ == "__main__":
    raise SystemExit(run(arms=ARMS, candidates={}, extra_pairs=EXTRA_PAIRS, description=__doc__))
