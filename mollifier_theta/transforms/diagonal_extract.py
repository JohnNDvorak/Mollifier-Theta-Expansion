"""Diagonal main term extraction.

Diagonal terms -> MainTerm (scale T^1, polynomial in theta) + Error (scale T^{1-delta}).
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger


class MainTermPoly:
    """Symbolic polynomial representing the diagonal main term as a function of theta.

    The main term is T * P(theta) * (log T)^k where P(theta) is a polynomial
    built from mollifier coefficients.
    """

    def __init__(
        self,
        coefficients: list[tuple[str, str]],
        description: str = "",
    ) -> None:
        # Each entry is (label, symbolic_expression_in_theta)
        self.coefficients = coefficients
        self.description = description

    def evaluate(self, theta_val: float) -> float:
        """Evaluate the polynomial at a given theta (uses eval for symbolic)."""
        import sympy as sp
        from mollifier_theta.core.scale_model import theta

        total = sp.Integer(0)
        for _label, expr_str in self.coefficients:
            total += sp.sympify(expr_str, locals={"theta": theta})
        return float(total.subs(theta, theta_val))

    def to_sympy(self):
        """Return SymPy expression for the polynomial."""
        import sympy as sp
        from mollifier_theta.core.scale_model import theta

        total = sp.Integer(0)
        for _label, expr_str in self.coefficients:
            total += sp.sympify(expr_str, locals={"theta": theta})
        return total

    def to_dict(self) -> dict:
        return {
            "coefficients": self.coefficients,
            "description": self.description,
        }


class DiagonalExtract:
    """Extract main term and error from diagonal terms."""

    def __init__(self, K: int = 3) -> None:
        self.K = K

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if term.kind == TermKind.DIAGONAL:
                extracted = self._apply_one(term)
                results.extend(extracted)
                new_terms.extend(extracted)
            else:
                results.append(term)
        ledger.add_many(new_terms)
        return results

    def _apply_one(self, term: Term) -> list[Term]:
        history_main = HistoryEntry(
            transform="DiagonalExtract",
            parent_ids=[term.id],
            description="Extracted diagonal main term: T * P(theta) * (log T)^k",
        )
        history_error = HistoryEntry(
            transform="DiagonalExtract",
            parent_ids=[term.id],
            description="Diagonal error: T^{1-delta} from truncation/Perron",
        )

        # Build the main term polynomial from mollifier structure
        # For K mollifier terms, the diagonal produces a polynomial in theta
        # with coefficients from the Ramanujan function / Mertens estimates
        poly = MainTermPoly(
            coefficients=[
                ("leading", "1"),
                ("mollifier_correction", f"-1/{self.K}"),
                ("cross_contribution", f"(({self.K}-1)/(2*{self.K}))"),
            ],
            description=f"Diagonal main term polynomial, K={self.K}",
        )

        main_term = Term(
            kind=TermKind.DIAGONAL,
            expression=f"T * P(theta) * (log T)^k [K={self.K}]",
            variables=[],
            ranges=[],
            kernels=list(term.kernels),
            phases=list(term.phases),
            scale_model="T^1",
            status=TermStatus.MAIN_TERM,
            history=list(term.history) + [history_main],
            parents=[term.id],
            multiplicity=term.multiplicity,
            metadata={
                **term.metadata,
                "diagonal_role": "main_term",
                "main_term_poly": poly.to_dict(),
                "T_exponent": "1",
                "log_power": self.K - 1,
            },
        )

        error_term = Term(
            kind=TermKind.ERROR,
            expression=f"O(T^{{1-delta}}) diagonal error [K={self.K}]",
            variables=[],
            ranges=[],
            kernels=list(term.kernels),
            phases=[],
            scale_model="T^(1-delta)",
            status=TermStatus.ERROR,
            history=list(term.history) + [history_error],
            parents=[term.id],
            metadata={
                **term.metadata,
                "diagonal_role": "error",
                "T_exponent": "1 - delta",
            },
        )

        return [main_term, error_term]

    def describe(self) -> str:
        return (
            f"Diagonal extraction (K={self.K}): Diagonal -> MainTerm "
            f"(T * polynomial in theta * (log T)^k) + Error (T^{{1-delta}})."
        )
