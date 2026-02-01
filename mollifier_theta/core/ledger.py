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
        """Retrieve a term by id."""
        return self._terms[term_id]

    def all_terms(self) -> list[Term]:
        """All terms in insertion order."""
        return list(self._terms.values())

    def filter(
        self,
        kind: TermKind | None = None,
        status: TermStatus | None = None,
        predicate: Callable[[Term], bool] | None = None,
    ) -> list[Term]:
        """Filter terms by kind, status, and/or arbitrary predicate."""
        result = list(self._terms.values())
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
        return len(self._terms)

    def validate_all(self) -> list[str]:
        """Run invariant checks on all terms."""
        return validate_all(list(self._terms.values()))

    def to_json(self) -> str:
        """Serialize ledger to JSON string."""
        terms_data = [t.model_dump() for t in self._terms.values()]
        # Convert enums to their values for clean JSON
        for td in terms_data:
            td["kind"] = td["kind"].value if hasattr(td["kind"], "value") else td["kind"]
            td["status"] = td["status"].value if hasattr(td["status"], "value") else td["status"]
        return json.dumps({"terms": terms_data}, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TermLedger":
        """Deserialize ledger from JSON string."""
        data = json.loads(json_str)
        ledger = cls()
        for td in data["terms"]:
            term = Term(**td)
            ledger._terms[term.id] = term
        return ledger

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, term_id: str) -> bool:
        return term_id in self._terms
