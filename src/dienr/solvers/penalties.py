"""Ridge, lasso and elastic-net penalties on nonnegative abundance.

A convenient consequence of the nonnegativity constraint: because ``a >= 0``,
the L1 penalty ``lambda_1 * sum |a_k|`` is just ``lambda_1 * sum a_k``, which is
linear and therefore differentiable everywhere in the feasible region. Lasso
needs no proximal operator or subgradient here, and the same smooth L-BFGS-B
driver handles ridge, lasso and elastic net alike.

Section 4.4 of the plan sets the policy these obey:

- a penalty is a named modeling choice, never a silent default;
- penalties must never manufacture resolution between exactly identical models,
  which is why exact duplicate columns are collapsed in `dienr.design` before
  any of this runs;
- an allocation among merely *near*-identical columns is model-dependent, and
  `dienr.diagnostics` reports the conditioning that says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

PenaltyType = Literal["none", "ridge", "lasso", "elastic_net"]


@dataclass(frozen=True)
class Penalty:
    """A penalty on the abundance vector.

    Attributes:
        kind: Which penalty to apply.
        lambda1: L1 weight, used by ``lasso`` and ``elastic_net``.
        lambda2: L2 weight, used by ``ridge`` and ``elastic_net``.
    """

    kind: PenaltyType = "none"
    lambda1: float = 0.0
    lambda2: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda1 < 0 or self.lambda2 < 0:
            raise ValueError(
                f"penalty weights must be non-negative, got "
                f"lambda1={self.lambda1}, lambda2={self.lambda2}"
            )
        if self.kind == "ridge" and self.lambda2 == 0:
            raise ValueError("ridge penalty requires lambda2 > 0")
        if self.kind == "lasso" and self.lambda1 == 0:
            raise ValueError("lasso penalty requires lambda1 > 0")
        if self.kind == "elastic_net" and (self.lambda1 == 0 or self.lambda2 == 0):
            raise ValueError("elastic_net penalty requires lambda1 > 0 and lambda2 > 0")
        if self.kind == "none" and (self.lambda1 or self.lambda2):
            raise ValueError(
                "penalty kind is 'none' but a weight was given; name the penalty "
                "explicitly so it is recorded in the output"
            )

    @property
    def is_active(self) -> bool:
        """True when this penalty contributes anything."""
        return self.kind != "none"

    @property
    def effective_l1(self) -> float:
        """L1 weight actually applied."""
        return self.lambda1 if self.kind in ("lasso", "elastic_net") else 0.0

    @property
    def effective_l2(self) -> float:
        """L2 weight actually applied."""
        return self.lambda2 if self.kind in ("ridge", "elastic_net") else 0.0

    def value(self, abundance: np.ndarray) -> float:
        """Penalty contribution to the objective at `abundance`."""
        if not self.is_active:
            return 0.0
        total = 0.0
        if self.effective_l1:
            # abundance >= 0, so |a| == a.
            total += self.effective_l1 * float(abundance.sum())
        if self.effective_l2:
            total += self.effective_l2 * float(abundance @ abundance)
        return total

    def gradient(self, abundance: np.ndarray) -> np.ndarray:
        """Penalty contribution to the gradient at `abundance`."""
        if not self.is_active:
            return np.zeros_like(abundance)
        grad = np.zeros_like(abundance)
        if self.effective_l1:
            grad += self.effective_l1
        if self.effective_l2:
            grad += 2.0 * self.effective_l2 * abundance
        return grad

    def describe(self) -> dict[str, Any]:
        """Provenance record, so a penalized estimate is never unlabelled."""
        return {
            "penalty_type": self.kind,
            "penalty_lambda1": self.effective_l1,
            "penalty_lambda2": self.effective_l2,
        }


#: The unpenalized default.
NO_PENALTY = Penalty()


def ridge(lambda2: float) -> Penalty:
    """Ridge penalty. The plan's recommended first regularized model, with
    a nonnegative Poisson likelihood and a small `lambda2`."""
    return Penalty(kind="ridge", lambda2=lambda2)


def lasso(lambda1: float) -> Penalty:
    """Lasso penalty.

    Correlated models may be selected arbitrarily among themselves; check
    `dienr.diagnostics` conditioning before reading a zero as absence.
    """
    return Penalty(kind="lasso", lambda1=lambda1)


def elastic_net(lambda1: float, lambda2: float) -> Penalty:
    """Elastic net, combining both weights."""
    return Penalty(kind="elastic_net", lambda1=lambda1, lambda2=lambda2)
