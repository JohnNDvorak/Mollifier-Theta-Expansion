"""Theta feasibility checker: symbolic exponent evaluation and binary search.

Reconciles three representations of theta_max:
  1. Symbolic: solve E(theta) = 1 via SymPy  ->  sp.Rational(4, 7) exactly
  2. Known constant: KNOWN_THETA_MAX = Fraction(4, 7) (regression guard)
  3. Numerical: binary search on the ledger's BoundOnly terms

The admissibility check uses strict inequality E(theta) < 1, so theta_max = 4/7
is the *supremum* of admissible values (4/7 itself is NOT admissible).  The
ThetaMaxResult dataclass makes this explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.core.scale_model import ScaleModel, theta
from mollifier_theta.lemmas.di_kloosterman import (
    DIExponentModel,
    KNOWN_THETA_MAX,
    ThetaBarrierMismatch,
)


@dataclass(frozen=True)
class ThetaMaxResult:
    """Reconciled theta_max from symbolic, known-constant, and numerical paths.

    Attributes:
        symbolic: Exact rational value from solving E(theta) = 1.
        numerical: Midpoint of final binary-search interval.
        numerical_lo: Last admissible theta found by binary search.
        numerical_hi: First inadmissible theta found by binary search.
        tol: Binary search tolerance used.
        is_supremum: Always True — theta_max itself is inadmissible because
                     E(theta_max) = 1 (not < 1).  Every theta < theta_max
                     is admissible.
    """

    symbolic: sp.Rational
    numerical: float
    numerical_lo: float
    numerical_hi: float
    tol: float
    is_supremum: bool = True

    @property
    def symbolic_float(self) -> float:
        return float(self.symbolic)

    @property
    def gap(self) -> float:
        """Absolute difference between numerical midpoint and symbolic value."""
        return abs(self.numerical - self.symbolic_float)


def theta_admissible(terms: list[Term], theta_val: float) -> bool:
    """Check all BoundOnly terms satisfy E(theta) < 1  (strict inequality).

    Returns True iff every BoundOnly error term has T-exponent strictly
    less than 1 at the given theta, meaning the error is o(main term).

    Note: at theta = 4/7, E(4/7) = 1.0 exactly, so theta_admissible
    returns False.  4/7 is the supremum, not the maximum.
    """
    for term in terms:
        if term.status != TermStatus.BOUND_ONLY:
            continue

        scale_dict = term.metadata.get("scale_model_dict")
        if scale_dict:
            sm = ScaleModel.from_dict(scale_dict)
            if not sm.is_dominated_at(theta_val):
                return False
        elif term.metadata.get("error_exponent"):
            expr = sp.sympify(
                term.metadata["error_exponent"],
                locals={"theta": theta},
            )
            val = float(expr.subs(theta, theta_val))
            if val >= 1:
                return False

    return True


def find_theta_max(
    terms: list[Term],
    lo: float = 0.01,
    hi: float = 0.99,
    tol: float = 1e-6,
) -> ThetaMaxResult:
    """Locate the supremum of admissible theta by three convergent methods.

    1. Binary search on theta_admissible(terms, ·) to find the numerical
       boundary within ±tol.
    2. Symbolic derivation: solve E(theta) = 1 via DIExponentModel.
    3. Cross-check symbolic result against KNOWN_THETA_MAX = 4/7.

    Raises ThetaBarrierMismatch if any pair of the three disagrees beyond
    the binary-search tolerance.
    """
    # --- Numerical: binary search ---
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if theta_admissible(terms, mid):
            lo = mid
        else:
            hi = mid

    numerical_theta_max = (lo + hi) / 2

    # --- Symbolic: derived from exponent algebra (Layer 1) ---
    model = DIExponentModel()
    symbolic_theta_max = model.theta_max()

    # --- Regression constant (Layer 2) ---
    known = sp.Rational(KNOWN_THETA_MAX.numerator, KNOWN_THETA_MAX.denominator)
    if symbolic_theta_max != known:
        raise ThetaBarrierMismatch(
            f"Symbolic theta_max = {symbolic_theta_max}, "
            f"known constant = {known}"
        )

    # --- Reconcile numerical with symbolic ---
    symbolic_float = float(symbolic_theta_max)
    delta = abs(numerical_theta_max - symbolic_float)
    # The binary search can only resolve to ±tol, so allow 2*tol as the
    # maximum acceptable gap (midpoint can be up to tol/2 off, plus float
    # rounding).
    max_acceptable_gap = 2 * tol
    if delta > max_acceptable_gap:
        raise ThetaBarrierMismatch(
            f"Numerical theta_max = {numerical_theta_max:.10f}, "
            f"symbolic = {symbolic_float:.10f}, "
            f"gap = {delta:.2e} exceeds 2*tol = {max_acceptable_gap:.2e}"
        )

    return ThetaMaxResult(
        symbolic=symbolic_theta_max,
        numerical=numerical_theta_max,
        numerical_lo=lo,
        numerical_hi=hi,
        tol=tol,
    )
