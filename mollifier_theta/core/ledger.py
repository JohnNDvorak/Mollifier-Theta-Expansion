"""TermLedger: central collection of terms with query/filter/serialize."""

from __future__ import annotations

import json
from typing import Callable

from mollifier_theta.core.ir import Term, TermKind, TermStatus
from mollifier_theta.core.invariants import validate_all


class TermLedger:
    """Dict-backed collection of Term objects with query and serialization."""

    def __init__(self) -> None:
        self._terms: dict[str, Term] = {}
        self._pruned_ids: set[str] = set()

    def add(self, term: Term) -> Term:
        """Add a term to the ledger. Returns the term."""
        if term.id in self._terms:
            raise ValueError(f"Duplicate term id: {term.id}")
        self._terms[term.id] = term
        return term

    def add_many(self, terms: list[Term]) -> list[Term]:
        """Add multiple terms. Returns the list."""
        for t in terms:
            self.add(t)
        return terms

    def get(self, term_id: str) -> Term:
        """Retrieve a term by id. Works for all terms including pruned ones."""
        return self._terms[term_id]

    def clone(self) -> "TermLedger":
        """Shallow-copy the ledger. Terms are immutable so sharing is safe."""
        new = TermLedger()
        new._terms = dict(self._terms)
        new._pruned_ids = set(self._pruned_ids)
        return new

    def all_terms(self) -> list[Term]:
        """All non-pruned terms in insertion order."""
        return [t for t in self._terms.values() if t.id not in self._pruned_ids]

    def all_terms_including_pruned(self) -> list[Term]:
        """All terms including pruned ones, in insertion order."""
        return list(self._terms.values())

    def filter(
        self,
        kind: TermKind | None = None,
        status: TermStatus | None = None,
        predicate: Callable[[Term], bool] | None = None,
    ) -> list[Term]:
        """Filter non-pruned terms by kind, status, and/or arbitrary predicate."""
        result = self.all_terms()
        if kind is not None:
            result = [t for t in result if t.kind == kind]
        if status is not None:
            result = [t for t in result if t.status == status]
        if predicate is not None:
            result = [t for t in result if predicate(t)]
        return result

    def active_terms(self) -> list[Term]:
        """Terms with status Active."""
        return self.filter(status=TermStatus.ACTIVE)

    def count(self) -> int:
        return len(self._terms) - len(self._pruned_ids)

    def count_total(self) -> int:
        """Total terms including pruned."""
        return len(self._terms)

    def validate_all(self) -> list[str]:
        """Run invariant checks on all non-pruned terms."""
        return validate_all(self.all_terms())

    def to_json(self) -> str:
        """Serialize ledger to JSON string."""
        terms_data = [t.model_dump(mode="json") for t in self._terms.values()]
        return json.dumps({"terms": terms_data}, indent=2, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "TermLedger":
        """Deserialize ledger from JSON string."""
        data = json.loads(json_str)
        ledger = cls()
        for td in data["terms"]:
            term = Term(**td)
            ledger.add(term)  # Route through add() for duplicate detection
        return ledger

    def prune(self, keep_statuses: set[TermStatus] | None = None) -> int:
        """Mark intermediate terms as pruned (non-destructive).

        Pruned terms are hidden from all_terms(), filter(), count() but
        still accessible via get() for parent chain traversal.
        Returns the number of terms pruned.
        """
        if keep_statuses is None:
            keep_statuses = {TermStatus.MAIN_TERM, TermStatus.BOUND_ONLY, TermStatus.ERROR}

        newly_pruned = {
            tid for tid, t in self._terms.items()
            if t.status not in keep_statuses and tid not in self._pruned_ids
        }
        self._pruned_ids |= newly_pruned
        return len(newly_pruned)

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, term_id: str) -> bool:
        return term_id in self._terms
