"""Tests for TermLedger."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Term, TermKind, TermStatus
from mollifier_theta.core.ledger import TermLedger


class TestLedgerBasics:
    def test_empty_ledger(self, empty_ledger: TermLedger) -> None:
        assert len(empty_ledger) == 0
        assert empty_ledger.all_terms() == []

    def test_add_and_get(self, empty_ledger: TermLedger) -> None:
        t = Term(kind=TermKind.INTEGRAL)
        empty_ledger.add(t)
        assert empty_ledger.get(t.id) == t
        assert len(empty_ledger) == 1

    def test_add_many(self, empty_ledger: TermLedger) -> None:
        terms = [Term(kind=TermKind.INTEGRAL) for _ in range(5)]
        empty_ledger.add_many(terms)
        assert len(empty_ledger) == 5

    def test_duplicate_id_rejected(self, empty_ledger: TermLedger) -> None:
        t = Term(id="fixed_id", kind=TermKind.INTEGRAL)
        empty_ledger.add(t)
        t2 = Term(id="fixed_id", kind=TermKind.DIAGONAL)
        with pytest.raises(ValueError, match="Duplicate"):
            empty_ledger.add(t2)

    def test_contains(self, empty_ledger: TermLedger) -> None:
        t = Term(kind=TermKind.INTEGRAL)
        empty_ledger.add(t)
        assert t.id in empty_ledger
        assert "nonexistent" not in empty_ledger


class TestLedgerFilter:
    def test_filter_by_kind(self, populated_ledger: TermLedger) -> None:
        integrals = populated_ledger.filter(kind=TermKind.INTEGRAL)
        assert len(integrals) == 1

    def test_filter_by_status(self, populated_ledger: TermLedger) -> None:
        bound = populated_ledger.filter(status=TermStatus.BOUND_ONLY)
        assert len(bound) == 1

    def test_filter_by_predicate(self, populated_ledger: TermLedger) -> None:
        with_kernels = populated_ledger.filter(
            predicate=lambda t: len(t.kernels) > 0
        )
        assert len(with_kernels) == 1

    def test_active_terms(self, populated_ledger: TermLedger) -> None:
        active = populated_ledger.active_terms()
        assert all(t.status == TermStatus.ACTIVE for t in active)

    def test_combined_filters(self) -> None:
        ledger = TermLedger()
        ledger.add(Term(kind=TermKind.DIAGONAL, status=TermStatus.ACTIVE))
        ledger.add(Term(kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM))
        ledger.add(Term(kind=TermKind.OFF_DIAGONAL, status=TermStatus.ACTIVE))
        result = ledger.filter(kind=TermKind.DIAGONAL, status=TermStatus.ACTIVE)
        assert len(result) == 1


class TestLedgerSerialization:
    def test_json_roundtrip(self, populated_ledger: TermLedger) -> None:
        json_str = populated_ledger.to_json()
        restored = TermLedger.from_json(json_str)
        assert len(restored) == len(populated_ledger)

    def test_json_roundtrip_preserves_kinds(
        self, populated_ledger: TermLedger
    ) -> None:
        json_str = populated_ledger.to_json()
        restored = TermLedger.from_json(json_str)
        original_kinds = {t.kind for t in populated_ledger.all_terms()}
        restored_kinds = {t.kind for t in restored.all_terms()}
        assert original_kinds == restored_kinds

    def test_json_roundtrip_preserves_status(
        self, populated_ledger: TermLedger
    ) -> None:
        json_str = populated_ledger.to_json()
        restored = TermLedger.from_json(json_str)
        original_statuses = {t.status for t in populated_ledger.all_terms()}
        restored_statuses = {t.status for t in restored.all_terms()}
        assert original_statuses == restored_statuses

    def test_empty_ledger_roundtrip(self) -> None:
        ledger = TermLedger()
        json_str = ledger.to_json()
        restored = TermLedger.from_json(json_str)
        assert len(restored) == 0


class TestLedgerValidation:
    def test_validate_all_clean(self, populated_ledger: TermLedger) -> None:
        assert populated_ledger.validate_all() == []
