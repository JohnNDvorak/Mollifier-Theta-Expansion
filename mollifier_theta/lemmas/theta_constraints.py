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

from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.core.scale_model import ScaleModel
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

    symbolic: Fraction
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
            val = ScaleModel.evaluate_expr(
                term.metadata["error_exponent"], theta_val,
            )
            if val >= 1:
                return False

    return True


def find_theta_max(
    terms: list[Term],
    lo: float = 0.01,
    hi: float = 0.99,
    tol: float = 1e-6,
    known_theta_max: Fraction | None = None,
) -> ThetaMaxResult:
    """Locate the supremum of admissible theta by convergent methods.

    When known_theta_max is None (default):
        Three-method reconciliation (baseline pipeline):
        1. Binary search on theta_admissible(terms, ·)
        2. Symbolic derivation from DIExponentModel
        3. Cross-check against KNOWN_THETA_MAX = 4/7

    When known_theta_max is provided:
        Two-method reconciliation (variant pipelines):
        1. Binary search on theta_admissible(terms, ·)
        2. Cross-check numerical result against provided constant
        (DIExponentModel symbolic derivation is skipped since the
        binding constraint may come from a different bound family.)

    Raises ThetaBarrierMismatch if the numerical and symbolic/known
    values disagree beyond the binary-search tolerance.
    """
    # --- Numerical: binary search ---
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if theta_admissible(terms, mid):
            lo = mid
        else:
            hi = mid

    numerical_theta_max = (lo + hi) / 2

    if known_theta_max is None:
        # Default path: derive symbolically from DI and cross-check
        model = DIExponentModel()
        symbolic_theta_max_sp = model.theta_max()
        symbolic_theta_max = Fraction(int(symbolic_theta_max_sp.p), int(symbolic_theta_max_sp.q))

        # Layer 2 cross-check
        if symbolic_theta_max != KNOWN_THETA_MAX:
            raise ThetaBarrierMismatch(
                f"Symbolic theta_max = {symbolic_theta_max}, "
                f"known constant = {KNOWN_THETA_MAX}"
            )
    else:
        # Variant pipeline path: use provided known constant
        symbolic_theta_max = known_theta_max

    # --- Reconcile numerical with symbolic/known ---
    symbolic_float = float(symbolic_theta_max)
    delta = abs(numerical_theta_max - symbolic_float)
    max_acceptable_gap = 2 * tol
    if delta > max_acceptable_gap:
        raise ThetaBarrierMismatch(
            f"Numerical theta_max = {numerical_theta_max:.10f}, "
            f"symbolic/known = {symbolic_float:.10f}, "
            f"gap = {delta:.2e} exceeds 2*tol = {max_acceptable_gap:.2e}"
        )

    return ThetaMaxResult(
        symbolic=symbolic_theta_max,
        numerical=numerical_theta_max,
        numerical_lo=lo,
        numerical_hi=hi,
        tol=tol,
    )
