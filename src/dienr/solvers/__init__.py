"""Solvers for locus-level abundance estimation.

Two likelihoods, in the order the plan stages them:

- `legacy`: the published identity and log least-squares objectives. Verified
  against R DENR and the default, because it is the only one with parity.
- `poisson`: nonnegative Poisson for raw counts. Requires count-equivalent
  input; a per-base mean summary is rejected.

Quasi-Poisson is not a third solver: it is the Poisson fit with standard errors
from `inference`. A negative-binomial solver existed and was removed (plan
Gate D): it cannot move a point estimate on a binary design, and quasi-Poisson
matched its standard errors to within 1-4%.

`penalties` supplies ridge, lasso and elastic net for the count likelihoods.
Both solvers share the lean L-BFGS-B driver in `_lbfgsb` and the exposure
and mean machinery in `_glm`.
"""

from . import penalties, poisson
from ._glm import Exposure
from .legacy import LocusFit, fit_locus, gradient, objective, predict
from .penalties import NO_PENALTY, Penalty, elastic_net, lasso, ridge

__all__ = [
    "NO_PENALTY",
    "Exposure",
    "LocusFit",
    "Penalty",
    "elastic_net",
    "fit_locus",
    "gradient",
    "lasso",
    "objective",
    "penalties",
    "poisson",
    "predict",
    "ridge",
]
