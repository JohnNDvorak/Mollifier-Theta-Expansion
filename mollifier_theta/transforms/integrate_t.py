"""Integrate over t: replace t-integral with Fourier kernel.

This transform replaces the t-integration on each cross-term with a
Fourier kernel K(log(am/bn)). It does NOT evaluate to a delta function —
that would violate the DO NOT SIMPLIFY invariant.
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


class IntegrateOverT:
    """Replace t-integral with Fourier kernel on each term."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        for term in terms:
            results.append(self._apply_one(term))
        ledger.add_many(results)
        return results

    def _apply_one(self, term: Term) -> Term:
        history = HistoryEntry(
            transform="IntegrateOverT",
            parent_ids=[term.id],
            description="Replaced t-integral with Fourier kernel K(log(am/bn)). NOT delta-approximated.",
        )

        fourier_kernel = Kernel(
            name="FourierKernel",
            support="R",
            argument="log(am/bn)",
            description=(
                "Fourier kernel from integrating (am/bn)^{it} over [0,T]. "
                "Concentrates near am=bn but is NOT approximated as delta."
            ),
            properties={
                "is_fourier": True,
                "not_delta_approximated": True,
                "concentration_scale": "1/T",
            },
        )

        # Remove t from variables, remove t-range, keep all other ranges
        new_variables = [v for v in term.variables if v != "t"]
        new_ranges = [r for r in term.ranges if r.variable != "t"]

        # Retain all kernels, add the Fourier kernel
        new_kernels = list(term.kernels) + [fourier_kernel]

        # Phases that depend only on t are consumed by the integration;
        # phases depending on other variables are retained
        new_phases: list[Phase] = []
        for phase in term.phases:
            if phase.depends_on == ["t"]:
                # Pure t-phase consumed by integration
                continue
            elif "t" in phase.depends_on:
                # Mixed phase: the t-dependent part becomes the Fourier kernel argument,
                # but the phase record is retained (marking it as partially consumed)
                new_phases.append(
                    Phase(
                        expression=phase.expression,
                        depends_on=[v for v in phase.depends_on if v != "t"],
                        is_separable=phase.is_separable,
                        absorbed=False,
                    )
                )
            else:
                new_phases.append(phase)

        return Term(
            kind=term.kind,
            expression=f"sum_{{m,n}} ... * K(log(am/bn)) [from {term.expression}]",
            variables=new_variables,
            ranges=new_ranges,
            kernels=new_kernels,
            phases=new_phases,
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            metadata={**term.metadata, "t_integrated": True},
        )

    def describe(self) -> str:
        return (
            "Integrate over t: replaces t-integration with Fourier kernel "
            "K(log(am/bn)). Does NOT approximate as delta(am-bn). "
            "Pure t-phases consumed; mixed phases have t-dependence removed."
        )
