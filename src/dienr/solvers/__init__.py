"""Solvers for locus-level abundance estimation.

`legacy` reproduces the published sum-of-squares objectives. Poisson,
negative-binomial and penalized solvers arrive in later phases; see the plan's
phase breakdown before adding one.
"""

from .legacy import LocusFit, fit_locus, gradient, objective, predict

__all__ = ["LocusFit", "fit_locus", "gradient", "objective", "predict"]
