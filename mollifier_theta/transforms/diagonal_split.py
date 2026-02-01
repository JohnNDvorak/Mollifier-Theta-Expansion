"""Diagonal / off-diagonal split.

Each input term -> 2 outputs:
- Diagonal (am=bn): oscillatory phase removed, kind=Diagonal
- OffDiagonal (am!=bn): all phases retained, kind=OffDiagonal
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Phase,
    Term,
    TermKind,
)
from mollifier_theta.core.ledger import TermLedger


class DiagonalSplit:
    """Split each term into diagonal (am=bn) and off-diagonal (am!=bn) parts."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        for term in terms:
            results.extend(self._apply_one(term))
        ledger.add_many(results)
        return results

    def _apply_one(self, term: Term) -> list[Term]:
        history_diag = HistoryEntry(
            transform="DiagonalSplit",
            parent_ids=[term.id],
            description="Diagonal part (am=bn): oscillatory phase removed",
        )
        history_offdiag = HistoryEntry(
            transform="DiagonalSplit",
            parent_ids=[term.id],
            description="Off-diagonal part (am!=bn): all phases retained",
        )

        # Diagonal: remove oscillatory phases (those depending on both m and n)
        # since am=bn kills the oscillation
        diag_phases: list[Phase] = []
        for p in term.phases:
            deps = set(p.depends_on)
            if "m" in deps and "n" in deps:
                # This phase vanishes on the diagonal am=bn
                continue
            diag_phases.append(p)

        diagonal = Term(
            kind=TermKind.DIAGONAL,
            expression=f"DIAG[{term.expression}] (am=bn)",
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=diag_phases,
            history=list(term.history) + [history_diag],
            parents=[term.id],
            multiplicity=term.multiplicity,
            metadata={**term.metadata, "split_role": "diagonal"},
        )

        # Off-diagonal: all phases retained
        off_diagonal = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression=f"OFFDIAG[{term.expression}] (am!=bn)",
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=list(term.phases),
            history=list(term.history) + [history_offdiag],
            parents=[term.id],
            multiplicity=term.multiplicity,
            metadata={**term.metadata, "split_role": "off_diagonal"},
        )

        return [diagonal, off_diagonal]

    def describe(self) -> str:
        return (
            "Diagonal/off-diagonal split: each term -> Diagonal (am=bn, "
            "oscillatory phases removed) + OffDiagonal (am!=bn, all phases retained)."
        )
