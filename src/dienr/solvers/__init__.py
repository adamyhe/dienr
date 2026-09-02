"""Solvers for locus-level abundance estimation.

Three likelihoods, in the order the plan stages them:

- `legacy`: the published identity and log least-squares objectives. Verified
  against R DENR and the default, because it is the only one with parity.
- `poisson`: nonnegative Poisson for raw counts. Requires count-equivalent
  input; a per-base mean summary is rejected.
- `negative_binomial`: NB2 at a shared, externally estimated dispersion. Not a
  default, and not to become one without the evidence Gate D asks for.

`penalties` supplies ridge, lasso and elastic net for the count likelihoods.
All three solvers share the lean L-BFGS-B driver in `_lbfgsb` and the exposure
and mean machinery in `_glm`.
"""

from . import negative_binomial, penalties, poisson
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
    "negative_binomial",
    "objective",
    "penalties",
    "poisson",
    "predict",
    "ridge",
]
