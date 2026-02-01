"""Transform protocol: every transform is list[Term] -> list[Term], never mutates input."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mollifier_theta.core.ir import Term
from mollifier_theta.core.ledger import TermLedger


@runtime_checkable
class Transform(Protocol):
    """Protocol for all transforms in the pipeline."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        """Apply the transform to a list of terms, returning new terms.

        Must not mutate input terms. All new terms should be added to ledger.
        """
        ...

    def describe(self) -> str:
        """Human-readable description of this transform."""
        ...
