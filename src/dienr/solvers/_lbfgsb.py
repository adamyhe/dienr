"""A lean L-BFGS-B driver, for many small independent problems.

`scipy.optimize.minimize` is the wrong shape for this workload. Section 14.2 of
the plan measured us *slower than R* at fitting a realistic annotation — 955
loci at about 3 models each — and section 14.1 found that on a median locus
roughly 91% of the runtime was optimizer scaffolding rather than arithmetic.
R wins there because its `optim` is a thin shim over C while
`scipy.optimize.minimize` interposes `ScalarFunction`, `_minimize_lbfgsb` and an
`OptimizeResult` per iteration, all in Python, all paid per locus.

This module calls the same Fortran routine, `scipy.optimize._lbfgsb.setulb`,
through the same driver loop, and drops everything around it:

- no `ScalarFunction` wrapper; the fused objective/gradient is called directly;
- no per-iteration `OptimizeResult` or callback dispatch;
- no `LbfgsInvHessProduct` built on exit;
- workspace arrays are cached and reused across problems of equal size, rather
  than reallocated per call. At ``m = 10`` the ``wa`` array alone is
  ``2mn + 5n + 11m² + 8m`` doubles.

It is the same algorithm with the same defaults, so results match
`minimize(method="L-BFGS-B")` exactly. `test_lean_driver_matches_scipy_minimize`
asserts that on the parity fixtures and on random problems, which is what makes
this safe to use where R parity has already been established.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg.lapack import HAS_ILP64
from scipy.optimize import _lbfgsb

#: Integer width the bundled LAPACK was built with. scipy derives the same
#: value inside `_minimize_lbfgsb`; passing the wrong width silently corrupts
#: setulb's integer workspace.
int_dtype = np.int64 if HAS_ILP64 else np.int32

#: scipy's L-BFGS-B defaults, mirrored so behavior is identical.
MAXCOR = 10
FTOL = 2.2204460492503131e-09
GTOL = 1e-5
MAXLS = 20

#: `factr` as scipy derives it from `ftol`.
FACTR = FTOL / np.finfo(float).eps

#: setulb task codes.
_TASK_NEW_ITERATION = 1
_TASK_NEED_FG = 3
_TASK_CONVERGED = 4
_TASK_STOP = 5

#: Reusable workspaces, keyed by problem size. Each entry is only ever used by
#: one call at a time, and every array is fully reinitialized on entry.
_WORKSPACES: dict[int, dict[str, np.ndarray]] = {}

#: Cap on cached sizes, so a run over wildly varying locus shapes cannot grow
#: this without bound.
_MAX_CACHED_SIZES = 256


def _workspace(n: int, m: int) -> dict[str, np.ndarray]:
    """Return zeroed workspace arrays for an `n`-parameter problem."""
    cached = _WORKSPACES.get(n)
    if cached is None:
        cached = {
            "x": np.zeros(n, dtype=np.float64),
            "low": np.zeros(n, dtype=np.float64),
            "upp": np.zeros(n, dtype=np.float64),
            "nbd": np.zeros(n, dtype=int_dtype),
            "g": np.zeros(n, dtype=np.float64),
            "wa": np.zeros(2 * m * n + 5 * n + 11 * m * m + 8 * m, dtype=np.float64),
            "iwa": np.zeros(3 * n, dtype=int_dtype),
            "task": np.zeros(2, dtype=int_dtype),
            "ln_task": np.zeros(2, dtype=int_dtype),
            "lsave": np.zeros(4, dtype=int_dtype),
            "isave": np.zeros(44, dtype=int_dtype),
            "dsave": np.zeros(29, dtype=np.float64),
        }
        if len(_WORKSPACES) < _MAX_CACHED_SIZES:
            _WORKSPACES[n] = cached
    else:
        # Reused arrays must be cleared: setulb reads its own state back out of
        # wa/iwa/isave/dsave, so leftovers from a previous problem would seed
        # the Hessian approximation with someone else's curvature.
        for key in ("wa", "iwa", "task", "ln_task", "lsave", "isave", "dsave", "g"):
            cached[key][...] = 0
    return cached


def minimize_bounded(
    value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    maxiter: int = 15_000,
    maxfun: int = 15_000,
    maxcor: int = MAXCOR,
) -> tuple[np.ndarray, float, np.ndarray, int, bool, str]:
    """Minimize a bounded problem with L-BFGS-B.

    Args:
        value_and_grad: Returns ``(objective, gradient)`` at a point.
        x0: Starting point.
        lower: Per-parameter lower bounds, finite.
        upper: Per-parameter upper bounds, finite.
        maxiter: Iteration cap.
        maxfun: Function-evaluation cap.
        maxcor: Number of stored corrections.

    Returns:
        ``(x, objective, gradient, iterations, converged, message)``.
    """
    n = x0.size
    space = _workspace(n, maxcor)

    x = space["x"]
    x[:] = x0
    space["low"][:] = lower
    space["upp"][:] = upper
    # 2 means "both bounds present", which is always the case here: the lower
    # bound is nonnegativity and the upper bound is finite by construction.
    space["nbd"][:] = 2

    g = space["g"]
    f = np.array(0.0, dtype=np.float64)
    task = space["task"]

    iterations = 0
    evaluations = 0
    while True:
        _lbfgsb.setulb(
            maxcor,
            x,
            space["low"],
            space["upp"],
            space["nbd"],
            f,
            g,
            FACTR,
            GTOL,
            space["wa"],
            space["iwa"],
            task,
            space["lsave"],
            space["isave"],
            space["dsave"],
            MAXLS,
            space["ln_task"],
        )
        code = task[0]
        if code == _TASK_NEED_FG:
            value, gradient = value_and_grad(x)
            f = np.array(value, dtype=np.float64)
            g[:] = gradient
            evaluations += 1
        elif code == _TASK_NEW_ITERATION:
            iterations += 1
            if iterations >= maxiter:
                task[0] = _TASK_STOP
                task[1] = 504
            elif evaluations > maxfun:
                task[0] = _TASK_STOP
                task[1] = 502
        else:
            break

    converged = task[0] == _TASK_CONVERGED
    if converged:
        message = "CONVERGENCE"
    elif iterations >= maxiter or evaluations > maxfun:
        message = "STOP: TOTAL NO. of ITERATIONS or FUNCTION EVALUATIONS EXCEEDS LIMIT"
    else:
        message = f"ABNORMAL: task {int(task[0])}/{int(task[1])}"

    # x lives in the shared workspace, so hand back a copy.
    return x.copy(), float(f), g.copy(), iterations, converged, message
