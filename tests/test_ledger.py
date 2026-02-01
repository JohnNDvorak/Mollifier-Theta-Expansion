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


class TestLedgerClone:
    def test_clone_produces_independent_copy(self) -> None:
        ledger = TermLedger()
        t1 = Term(kind=TermKind.INTEGRAL)
        ledger.add(t1)

        clone = ledger.clone()
        t2 = Term(kind=TermKind.DIAGONAL)
        clone.add(t2)

        # Clone has both, original has only t1
        assert len(clone) == 2
        assert len(ledger) == 1
        assert t2.id not in ledger

    def test_clone_shares_immutable_terms(self) -> None:
        ledger = TermLedger()
        t = Term(kind=TermKind.INTEGRAL)
        ledger.add(t)

        clone = ledger.clone()
        assert clone.get(t.id) is ledger.get(t.id)  # Same object (shared)

    def test_clone_copies_pruned_ids(self) -> None:
        ledger = TermLedger()
        t = Term(kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE)
        ledger.add(t)
        ledger.prune(keep_statuses={TermStatus.MAIN_TERM})

        clone = ledger.clone()
        assert len(clone) == 0  # Pruned in clone too
        assert clone.get(t.id) == t  # But still accessible via get


class TestLedgerPrune:
    def test_prune_non_destructive(self) -> None:
        """Pruned terms are hidden from all_terms but accessible via get."""
        ledger = TermLedger()
        active = Term(kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE)
        main = Term(kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM)
        ledger.add(active)
        ledger.add(main)

        pruned_count = ledger.prune(keep_statuses={TermStatus.MAIN_TERM})
        assert pruned_count == 1

        # Active term hidden from all_terms
        visible = ledger.all_terms()
        assert len(visible) == 1
        assert visible[0].id == main.id

        # But still accessible via get (parent chain preservation)
        assert ledger.get(active.id) == active

    def test_prune_preserves_parent_chain(self) -> None:
        """After pruning, parent terms are still accessible for lineage traversal."""
        ledger = TermLedger()
        parent = Term(id="parent_01", kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE)
        child = Term(
            id="child_01",
            kind=TermKind.DIAGONAL,
            status=TermStatus.MAIN_TERM,
            parents=["parent_01"],
        )
        ledger.add(parent)
        ledger.add(child)

        ledger.prune(keep_statuses={TermStatus.MAIN_TERM})

        # Child is visible
        assert child.id in [t.id for t in ledger.all_terms()]
        # Parent is pruned but accessible
        retrieved = ledger.get("parent_01")
        assert retrieved == parent

    def test_all_terms_including_pruned(self) -> None:
        ledger = TermLedger()
        t1 = Term(kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE)
        t2 = Term(kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM)
        ledger.add(t1)
        ledger.add(t2)

        ledger.prune(keep_statuses={TermStatus.MAIN_TERM})

        assert len(ledger.all_terms()) == 1
        assert len(ledger.all_terms_including_pruned()) == 2

    def test_count_vs_count_total(self) -> None:
        ledger = TermLedger()
        for _ in range(3):
            ledger.add(Term(kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE))
        ledger.add(Term(kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM))

        ledger.prune(keep_statuses={TermStatus.MAIN_TERM})
        assert ledger.count() == 1
        assert ledger.count_total() == 4

    def test_prune_idempotent(self) -> None:
        ledger = TermLedger()
        ledger.add(Term(kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE))
        ledger.add(Term(kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM))

        first = ledger.prune(keep_statuses={TermStatus.MAIN_TERM})
        second = ledger.prune(keep_statuses={TermStatus.MAIN_TERM})
        assert first == 1
        assert second == 0  # Already pruned

    def test_filter_respects_pruned(self) -> None:
        ledger = TermLedger()
        ledger.add(Term(kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE))
        ledger.add(Term(kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM))

        ledger.prune(keep_statuses={TermStatus.MAIN_TERM})
        # filter() should only see non-pruned terms
        result = ledger.filter(kind=TermKind.INTEGRAL)
        assert len(result) == 0


class TestSerializationHardening:
    def test_to_json_sort_keys(self) -> None:
        """to_json output uses sort_keys=True for determinism."""
        import json

        ledger = TermLedger()
        ledger.add(Term(id="t1", kind=TermKind.INTEGRAL, metadata={"z": 1, "a": 2}))
        json_str = ledger.to_json()
        data = json.loads(json_str)
        # Re-serialize with sort_keys and compare
        reserialized = json.dumps(data, indent=2, sort_keys=True, default=str)
        assert json_str.strip() == reserialized.strip()

    def test_from_json_duplicate_detection(self) -> None:
        """from_json rejects JSON with duplicate term IDs."""
        import json

        dup_data = {
            "terms": [
                {"id": "dup", "kind": "Integral"},
                {"id": "dup", "kind": "Diagonal"},
            ]
        }
        with pytest.raises(ValueError, match="Duplicate"):
            TermLedger.from_json(json.dumps(dup_data))

    def test_roundtrip_with_pruned_terms(self) -> None:
        """JSON roundtrip includes pruned terms (they're preserved in _terms)."""
        ledger = TermLedger()
        t1 = Term(kind=TermKind.INTEGRAL, status=TermStatus.ACTIVE)
        t2 = Term(kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM)
        ledger.add(t1)
        ledger.add(t2)
        ledger.prune(keep_statuses={TermStatus.MAIN_TERM})

        json_str = ledger.to_json()
        restored = TermLedger.from_json(json_str)
        # Pruned terms are included in serialization (they're in _terms)
        assert restored.count_total() == 2


class TestLedgerValidation:
    def test_validate_all_clean(self, populated_ledger: TermLedger) -> None:
        assert populated_ledger.validate_all() == []
