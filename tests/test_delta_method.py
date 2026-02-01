"""Tests for DeltaMethodInsert transform."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Kernel, Phase, Range, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.delta_method import DeltaMethodInsert


@pytest.fixture
def delta() -> DeltaMethodInsert:
    return DeltaMethodInsert()


@pytest.fixture
def off_diagonal_term() -> Term:
    return Term(
        kind=TermKind.OFF_DIAGONAL,
        expression="OFFDIAG[sum ...] (am!=bn)",
        variables=["m", "n"],
        ranges=[
            Range(variable="m", lower="1", upper="T^theta"),
            Range(variable="n", lower="1", upper="T^theta"),
        ],
        kernels=[
            Kernel(name="W_AFE"),
            Kernel(name="FourierKernel", argument="log(am/bn)"),
        ],
        phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
    )


class TestDeltaMethodStructure:
    def test_modulus_variable_added(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        assert "c" in results[0].variables

    def test_modulus_range_added(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        range_vars = {r.variable for r in results[0].ranges}
        assert "c" in range_vars

    def test_additive_character_phases(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        phase_exprs = {p.expression for p in results[0].phases}
        assert "e(am/c)" in phase_exprs
        assert "e(-bn/c)" in phase_exprs

    def test_additive_phases_are_separable(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        additive = [p for p in results[0].phases if p.expression.startswith("e(")]
        assert all(p.is_separable for p in additive)


class TestDeltaMethodKernels:
    def test_delta_method_kernel_added(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        kernel_names = {k.name for k in results[0].kernels}
        assert "DeltaMethodKernel" in kernel_names

    def test_original_kernels_preserved(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        kernel_names = {k.name for k in results[0].kernels}
        assert "W_AFE" in kernel_names
        assert "FourierKernel" in kernel_names

    def test_delta_kernel_properties(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert dk.properties.get("smooth") is True


class TestDeltaMethodPassthrough:
    def test_non_off_diagonal_passes_through(self, delta) -> None:
        diag = Term(kind=TermKind.DIAGONAL)
        ledger = TermLedger()
        ledger.add(diag)
        results = delta.apply([diag], ledger)
        assert len(results) == 1
        assert results[0].kind == TermKind.DIAGONAL


class TestDeltaMethodHistory:
    def test_history_appended(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        assert results[0].history[-1].transform == "DeltaMethodInsert"

    def test_metadata_flag(self, delta, off_diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = delta.apply([off_diagonal_term], ledger)
        assert results[0].metadata.get("delta_method_applied") is True
