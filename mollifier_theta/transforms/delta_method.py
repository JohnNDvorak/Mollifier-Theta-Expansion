"""Delta method insertion for off-diagonal terms.

Off-diagonal -> sum over moduli c with additive character / Ramanujan sum structure.
The smooth kernel from the delta method is tracked as a first-class Kernel.
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


class DeltaMethodInsert:
    """Apply delta method to off-diagonal terms, introducing modulus sum."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if term.kind == TermKind.OFF_DIAGONAL:
                transformed = self._apply_one(term)
                results.append(transformed)
                new_terms.append(transformed)
            else:
                results.append(term)
        ledger.add_many(new_terms)
        return results

    def _apply_one(self, term: Term) -> Term:
        history = HistoryEntry(
            transform="DeltaMethodInsert",
            parent_ids=[term.id],
            description=(
                "Delta method: introduced sum over moduli c with "
                "additive characters e(am/c) and smooth delta-method kernel."
            ),
        )

        delta_kernel = Kernel(
            name="DeltaMethodKernel",
            support="(0, inf)",
            argument="(am-bn)/c",
            description=(
                "Smooth kernel from delta method approximation to "
                "delta(am-bn). NOT a literal delta function."
            ),
            properties={
                "is_delta_method": True,
                "smooth": True,
                "compact_support_in_c": False,
            },
        )

        # Add modulus variable c with range
        new_variables = list(term.variables) + ["c"]
        new_ranges = list(term.ranges) + [
            Range(
                variable="c",
                lower="1",
                upper="C(T,theta)",
                description="Modulus range from delta method, C ~ T^{1+epsilon}/y where y = T^theta",
            )
        ]

        # Add additive character phases
        new_phases = list(term.phases) + [
            Phase(
                expression="e(am/c)",
                depends_on=["m", "c"],
                is_separable=True,
            ),
            Phase(
                expression="e(-bn/c)",
                depends_on=["n", "c"],
                is_separable=True,
            ),
        ]

        # Keep all existing kernels plus the delta method kernel
        new_kernels = list(term.kernels) + [delta_kernel]

        return Term(
            kind=TermKind.OFF_DIAGONAL,
            expression=f"sum_c sum_{{m,n}} a_m b_n e((am-bn)/c) V(...) [from {term.expression}]",
            variables=new_variables,
            ranges=new_ranges,
            kernels=new_kernels,
            phases=new_phases,
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            metadata={
                **term.metadata,
                "delta_method_applied": True,
                "modulus_variable": "c",
            },
        )

    def describe(self) -> str:
        return (
            "Delta method: off-diagonal -> sum over moduli c with additive "
            "characters e(am/c), e(-bn/c). Smooth delta-method kernel "
            "tracked as first-class Kernel object."
        )
