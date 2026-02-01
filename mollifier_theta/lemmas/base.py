"""Lemma protocol: applies(), bound(), explain()."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mollifier_theta.core.ir import Term


@runtime_checkable
class Lemma(Protocol):
    """Protocol for bound-application lemmas."""

    def applies(self, term: Term) -> bool:
        """Whether this lemma can be applied to the given term."""
        ...

    def bound(self, term: Term) -> Term:
        """Apply the bound, returning a new term with status=BoundOnly,
        scale_model set, and lemma_citation filled."""
        ...

    def explain(self) -> str:
        """Human-readable explanation of the lemma and its proof context."""
        ...
