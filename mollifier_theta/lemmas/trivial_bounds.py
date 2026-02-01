"""Trivial and Weil individual sum bounds."""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.scale_model import ScaleModel, theta


class TrivialBound:
    """Trivial bound for error terms: just use absolute values."""

    CITATION = "Trivial bound (absolute convergence)"

    def applies(self, term: Term) -> bool:
        return term.status == TermStatus.ERROR

    def bound(self, term: Term) -> Term:
        history = HistoryEntry(
            transform="TrivialBound",
            parent_ids=[term.id],
            description="Applied trivial bound (absolute values).",
        )
        return Term(
            kind=term.kind,
            expression=f"Trivially bounded: {term.expression}",
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=[],
            scale_model=term.scale_model,
            status=TermStatus.BOUND_ONLY,
            history=list(term.history) + [history],
            parents=[term.id],
            lemma_citation=self.CITATION,
            multiplicity=term.multiplicity,
            kernel_state=term.kernel_state,
            metadata={**term.metadata, "trivial_bound": True},
        )

    def explain(self) -> str:
        return "Trivial bound: use absolute values. No cancellation exploited."


class WeilBound:
    """Weil bound for individual Kloosterman sums: |S(m,n;c)| <= tau(c) * sqrt(gcd(m,n,c)) * c^{1/2}."""

    CITATION = "Weil 1948, Kloosterman sum bound"

    def applies(self, term: Term) -> bool:
        return (
            term.kind == TermKind.KLOOSTERMAN
            and term.status == TermStatus.ACTIVE
        )

    def bound(self, term: Term) -> Term:
        # Weil gives |S(m,n;c)| << c^{1/2+eps}
        # For the sum over m,n ~ T^theta, c ~ T^{1-theta}:
        # Total ~ T^{2*theta} * T^{(1-theta)/2} = T^{2*theta + (1-theta)/2}
        # = T^{(3*theta + 1)/2}
        scale = ScaleModel(
            T_exponent=(3 * theta + 1) / 2,
            description="Weil bound: individual Kloosterman sum bound",
        )

        history = HistoryEntry(
            transform="WeilBound",
            parent_ids=[term.id],
            description="Applied Weil bound to individual Kloosterman sums.",
        )

        return Term(
            kind=term.kind,
            expression=f"Weil bounded: T^((3*theta+1)/2) [from {term.expression}]",
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=list(term.phases),
            scale_model=scale.to_str(),
            status=TermStatus.BOUND_ONLY,
            history=list(term.history) + [history],
            parents=[term.id],
            lemma_citation=self.CITATION,
            multiplicity=term.multiplicity,
            kernel_state=term.kernel_state,
            metadata={
                **term.metadata,
                "weil_bound": True,
                "error_exponent": str(scale.T_exponent),
            },
        )

    def explain(self) -> str:
        return (
            "Weil bound (1948): |S(m,n;c)| <= tau(c) * sqrt(gcd(m,n,c)) * c^{1/2}.\n"
            "Applied individually to each Kloosterman sum in the off-diagonal.\n"
            "This gives weaker results than the DI bilinear bound — theta < 1/3\n"
            "instead of theta < 4/7."
        )
