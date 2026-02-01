"""Reorganize off-diagonal into canonical Kloosterman bilinear form S(m,n;c).

After the delta method, the additive characters e(am/c) and e(-bn/c) combine
to form Kloosterman sums S(m,n;c) = sum_{d (mod c), gcd(d,c)=1} e((md+n*d_bar)/c).
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import KloostermanMeta
from mollifier_theta.core.sum_structures import SumStructure


class KloostermanForm:
    """Reorganize into canonical S(m,n;c)/c * (smooth integral) form."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if (
                term.kind == TermKind.OFF_DIAGONAL
                and term.kernel_state == KernelState.COLLAPSED
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

        # Build set of phase expressions to consume.
        # Data-driven: read SumStructure twists if available.
        consumed_expressions: set[str] = set()
        ss_data = term.metadata.get("sum_structure")
        used_fallback = False
        if ss_data:
            ss = SumStructure.model_validate(ss_data)
            for twist in ss.additive_twists:
                consumed_expressions.add(twist.format_phase_expression())
        else:
            # Fallback: hard-coded legacy expressions (logged as fallback)
            consumed_expressions.add("e(am/c)")
            consumed_expressions.add("e(-bn/c)")
            used_fallback = True

        # Determine the sum variables involved for the Kloosterman phase.
        # Use the actual term variables (which may be renamed after Voronoi).
        kloosterman_vars = list(term.variables)

        new_phases: list[Phase] = []
        actually_consumed: list[str] = []
        for p in term.phases:
            if p.expression in consumed_expressions:
                # Consumed into Kloosterman sum
                actually_consumed.append(p.expression)
                continue
            new_phases.append(p)

        # Add Kloosterman phase as a combined non-separable phase
        # Use actual term variables (may include n* after Voronoi)
        new_phases.append(
            Phase(
                expression="S(m,n;c)/c",
                depends_on=kloosterman_vars,
                is_separable=False,
                unit_modulus=False,
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
            kernel_state=KernelState.KLOOSTERMANIZED,
            metadata={
                **term.metadata,
                "kloosterman_form": True,
                "kloosterman_variables": kloosterman_vars,
                "kloosterman_used_fallback": used_fallback,
                "_kloosterman": KloostermanMeta(
                    formed=True,
                    variables=kloosterman_vars,
                    consumed_phases=actually_consumed,
                ).model_dump(),
            },
        )

    def describe(self) -> str:
        return (
            "Kloosterman form: reorganize additive characters e(am/c), e(-bn/c) "
            "into canonical S(m,n;c)/c bilinear form with smooth integral factor."
        )
