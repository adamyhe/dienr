"""Are dienr's built-in defaults -- DENR's own -- good for dominant calls? (plan 18.22)

dienr's defaults reproduce R DENR's function defaults exactly: 250 bp bins,
**no masks at all** (`transcript_quantifier(mask_start_bins = NULL,
mask_end_bins = NULL)`), `round`, log least squares on a `mean` summary.
18.20-18.21's ``denr_default`` arm was *not* that: it was DENR with the
paper's masking (TSS 1 bin, PAS 4 bins). 18.21's best dominant-call
configuration, log least squares with TSS mask ``(1, 1)`` and no PAS mask,
differs from the true defaults only by the 1-bin TSS mask, which was tested
only with Poisson and only on held-out deviance (18.19).

If the TSS mask doesn't help the log objective's dominant calls, the
"dominant isoforms" recommendation *is* the built-in defaults, and the
defaults are both DENR-faithful and a sound choice. This measures it, with
18.20's truth, matching, groups and scoring (`k562_longread_accuracy.main`).

**Arms** (no run-on tail, legacy log, ``mean``, unless stated):

- ``defaults`` -- no masks: dienr's and DENR's built-in defaults.
- ``log_tss_1_1`` -- TSS mask ``(1, 1)``, no PAS mask (18.21's ``log_0_0``).
- ``denr_paper`` -- TSS ``(1, 1)``, PAS ``(4, 4)``: 18.20's ``denr_default``.
- ``poisson_runon`` -- TSS ``(1, 1)``, no PAS mask, tail, Poisson, ``sum``
  (the abundances recommendation).

**Decision rule, fixed before running.** ``log_tss_1_1`` vs ``defaults``.
The dominant-isoforms recommendation keeps ``--mask-start-bins 1 1`` only if
``log_tss_1_1`` is better on **both** PRO-seq replicates: higher accuracy AND
> 50% of discordant loci with a two-sided sign test p < 0.05. Otherwise the
recommendation becomes the built-in defaults with no flags. Reported, not part
of the rule: ``defaults`` against ``denr_paper`` and ``poisson_runon``.

    uv run python scripts/k562_longread_defaults.py --gtf build/sim/Homo_sapiens.GRCh37.87.gtf.gz \\
        --bigwigs build/k562 --longread build/k562/longread --score rep1 --tail-from rep2 \\
        --output build/k562/defaults_rep1.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k562_longread_accuracy import main as run  # noqa: E402

ARMS: dict[str, dict] = {
    "defaults": {"tss": (0, 0), "pas": (0, 0), "tail": False, "likelihood": "legacy"},
    "log_tss_1_1": {"tss": (1, 1), "pas": (0, 0), "tail": False, "likelihood": "legacy"},
    "denr_paper": {"tss": (1, 1), "pas": (4, 4), "tail": False, "likelihood": "legacy"},
    "poisson_runon": {"tss": (1, 1), "pas": (0, 0), "tail": True, "likelihood": "poisson"},
}
CANDIDATES = {"log_tss_1_1": ("defaults",)}
EXTRA_PAIRS = (("defaults", "denr_paper"), ("defaults", "poisson_runon"))


if __name__ == "__main__":
    raise SystemExit(
        run(arms=ARMS, candidates=CANDIDATES, extra_pairs=EXTRA_PAIRS, description=__doc__)
    )
