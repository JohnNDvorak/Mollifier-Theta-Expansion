"""Reorganize off-diagonal into canonical Kloosterman bilinear form S(m,n;c).

After the delta method, the additive characters e(am/c) and e(-bn/c) combine
to form Kloosterman sums S(m,n;c) = sum_{d (mod c), gcd(d,c)=1} e((md+n*d_bar)/c).
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    Phase,
    Range,
    Term,
    TermKind,
)
from mollifier_theta.core.ledger import TermLedger


class KloostermanForm:
    """Reorganize into canonical S(m,n;c)/c * (smooth integral) form."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if term.kind == TermKind.OFF_DIAGONAL and term.metadata.get(
                "delta_method_applied"
            ):
                transformed = self._apply_one(term)
                results.append(transformed)
                new_terms.append(transformed)
            else:
                results.append(term)
        ledger.add_many(new_terms)
        return results

    def _apply_one(self, term: Term) -> Term:
        history = HistoryEntry(
            transform="KloostermanForm",
            parent_ids=[term.id],
            description=(
                "Reorganized additive characters into Kloosterman sum "
                "S(m,n;c)/c with smooth integral factor."
            ),
        )

        # The Kloosterman structure replaces the separated additive character phases.
        # The e(am/c) and e(-bn/c) phases are consumed into S(m,n;c).
        new_phases: list[Phase] = []
        for p in term.phases:
            if p.expression in ("e(am/c)", "e(-bn/c)"):
                # Consumed into Kloosterman sum
                continue
            new_phases.append(p)

        # Add Kloosterman phase as a combined non-separable phase
        new_phases.append(
            Phase(
                expression="S(m,n;c)/c",
                depends_on=["m", "n", "c"],
                is_separable=False,
            )
        )

        return Term(
            kind=TermKind.KLOOSTERMAN,
            expression=(
                f"sum_c (1/c) sum_{{m,n}} a_m b_n S(m,n;c) V(m,n,c) "
                f"[from {term.expression}]"
            ),
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=new_phases,
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            metadata={
                **term.metadata,
                "kloosterman_form": True,
                "kloosterman_variables": ["m", "n", "c"],
            },
        )

    def describe(self) -> str:
        return (
            "Kloosterman form: reorganize additive characters e(am/c), e(-bn/c) "
            "into canonical S(m,n;c)/c bilinear form with smooth integral factor."
        )
