"""Tests for OpenSquare transform."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Kernel, Phase, Range, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.open_square import OpenSquare


@pytest.fixture
def dirichlet_term() -> Term:
    return Term(
        kind=TermKind.DIRICHLET_SUM,
        expression="sum a_n n^{-s} W(n/x)",
        variables=["n", "t"],
        ranges=[
            Range(variable="n", lower="1", upper="T^theta"),
            Range(variable="t", lower="0", upper="T"),
        ],
        kernels=[Kernel(name="W_AFE", argument="n/x")],
    )


class TestOpenSquareTermCount:
    def test_k3_produces_6_terms(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        assert len(results) == 6  # 3*(3+1)/2 = 6

    def test_k2_produces_3_terms(self, dirichlet_term) -> None:
        os = OpenSquare(K=2)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        assert len(results) == 3  # 2*(2+1)/2 = 3

    def test_k1_produces_1_term(self, dirichlet_term) -> None:
        os = OpenSquare(K=1)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        assert len(results) == 1

    def test_k4_produces_10_terms(self, dirichlet_term) -> None:
        os = OpenSquare(K=4)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        assert len(results) == 10  # 4*(4+1)/2 = 10


class TestOpenSquareMultiplicity:
    def test_diagonal_pairs_multiplicity_1(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        diag = [t for t in results if t.metadata.get("is_diagonal_pair")]
        assert len(diag) == 3  # (1,1), (2,2), (3,3)
        assert all(t.multiplicity == 1 for t in diag)

    def test_off_diagonal_pairs_multiplicity_2(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        off_diag = [t for t in results if not t.metadata.get("is_diagonal_pair")]
        assert len(off_diag) == 3  # (1,2), (1,3), (2,3)
        assert all(t.multiplicity == 2 for t in off_diag)

    def test_total_multiplicity(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        total = sum(t.multiplicity for t in results)
        assert total == 9  # K^2 = 9


class TestOpenSquarePhases:
    def test_off_diagonal_has_phase(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        off_diag = [t for t in results if not t.metadata.get("is_diagonal_pair")]
        for t in off_diag:
            phase_exprs = [p.expression for p in t.phases]
            assert any("^{it}" in expr for expr in phase_exprs)

    def test_diagonal_pair_no_extra_phase(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        diag = [t for t in results if t.metadata.get("is_diagonal_pair")]
        for t in diag:
            # Should only have phases from parent, not conjugation phases
            assert len(t.phases) == len(dirichlet_term.phases)

    def test_phases_from_parent_preserved(self) -> None:
        parent_phase = Phase(expression="chi(1/2+it)", depends_on=["t"])
        term = Term(
            kind=TermKind.DIRICHLET_SUM,
            variables=["n", "t"],
            phases=[parent_phase],
        )
        os = OpenSquare(K=2)
        ledger = TermLedger()
        ledger.add(term)
        results = os.apply([term], ledger)
        for t in results:
            assert any(p.expression == "chi(1/2+it)" for p in t.phases)


class TestOpenSquareKernels:
    def test_kernels_preserved(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        for t in results:
            kernel_names = [k.name for k in t.kernels]
            assert "W_AFE" in kernel_names


class TestOpenSquareKind:
    def test_all_cross_kind(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        results = os.apply([dirichlet_term], ledger)
        assert all(t.kind == TermKind.CROSS for t in results)


class TestOpenSquarePurity:
    def test_does_not_mutate_input(self, dirichlet_term) -> None:
        os = OpenSquare(K=3)
        ledger = TermLedger()
        ledger.add(dirichlet_term)
        orig_phases = len(dirichlet_term.phases)
        os.apply([dirichlet_term], ledger)
        assert len(dirichlet_term.phases) == orig_phases
