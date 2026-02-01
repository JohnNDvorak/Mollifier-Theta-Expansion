"""Approximate functional equation: zeta integral -> Dirichlet sums + error.

Input: integral term int_0^T |M*zeta|^2 dt
Output: 2 main Dirichlet-sum terms (short + long) + 1 error term

The AFE kernel W is attached as a first-class Kernel to the main terms.
The error term has status=Error with scale T^{-A}.
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger


class ApproxFunctionalEq:
    """Approximate functional equation transform."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        for term in terms:
            results.extend(self._apply_one(term))
        ledger.add_many(results)
        return results

    def _apply_one(self, term: Term) -> list[Term]:
        history = HistoryEntry(
            transform="ApproxFunctionalEq",
            parent_ids=[term.id],
            description="Applied approximate functional equation",
        )

        # Short sum: sum_{n <= sqrt(t/2pi)} a_n n^{-s} W(n/sqrt(t/2pi))
        short_sum = Term(
            kind=TermKind.DIRICHLET_SUM,
            expression="sum_{n<=x} a_n n^{-1/2-it} W(n/x)",
            variables=["n", "t"],
            ranges=[
                Range(variable="n", lower="1", upper="sqrt(t/2pi)"),
                Range(variable="t", lower="0", upper="T"),
            ],
            kernels=[
                Kernel(
                    name="W_AFE",
                    support="(0, inf)",
                    argument="n/sqrt(t/2pi)",
                    description="Approximate functional equation kernel (short sum)",
                    properties={
                        "mellin_transform": "Gamma(s/2) pi^{-s/2} / Gamma(1/4)",
                        "residue_structure": "Pole at s=1 with residue 1",
                        "rapid_decay": True,
                    },
                )
            ],
            phases=list(term.phases),
            history=list(term.history) + [history],
            parents=[term.id],
            metadata={"afe_role": "short_sum"},
        )

        # Long sum: chi(s) * sum_{n <= sqrt(t/2pi)} a_n n^{s-1} W_tilde(n/sqrt(t/2pi))
        long_sum = Term(
            kind=TermKind.DIRICHLET_SUM,
            expression="chi(s) sum_{n<=x} a_n n^{-1/2+it} W_tilde(n/x)",
            variables=["n", "t"],
            ranges=[
                Range(variable="n", lower="1", upper="sqrt(t/2pi)"),
                Range(variable="t", lower="0", upper="T"),
            ],
            kernels=[
                Kernel(
                    name="W_AFE_tilde",
                    support="(0, inf)",
                    argument="n/sqrt(t/2pi)",
                    description="Approximate functional equation kernel (long sum, functional eq side)",
                    properties={
                        "mellin_transform": "Gamma((1-s)/2) pi^{-(1-s)/2} / Gamma(1/4)",
                        "residue_structure": "Pole at s=0 with residue 1",
                        "rapid_decay": True,
                    },
                )
            ],
            phases=list(term.phases) + [
                Phase(
                    expression="chi(1/2+it)",
                    depends_on=["t"],
                    description="Functional equation chi factor",
                )
            ],
            history=list(term.history) + [history],
            parents=[term.id],
            metadata={"afe_role": "long_sum"},
        )

        # Error term: O(T^{-A}) for any A > 0
        error = Term(
            kind=TermKind.ERROR,
            expression="O(T^{-A})",
            variables=[],
            ranges=[],
            scale_model="T^(-A)",
            status=TermStatus.ERROR,
            history=list(term.history) + [history],
            parents=[term.id],
            metadata={"afe_role": "error"},
        )

        return [short_sum, long_sum, error]

    def describe(self) -> str:
        return (
            "Approximate functional equation: replaces zeta integral with "
            "two Dirichlet sums (short + long via functional equation) "
            "plus O(T^{-A}) error. Attaches AFE kernel W to main terms."
        )
