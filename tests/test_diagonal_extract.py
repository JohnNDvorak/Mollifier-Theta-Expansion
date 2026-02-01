"""Tests for DiagonalExtract transform."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Kernel, Range, Term, TermKind, TermStatus
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.diagonal_extract import DiagonalExtract, MainTermPoly


@pytest.fixture
def extract() -> DiagonalExtract:
    return DiagonalExtract(K=3)


@pytest.fixture
def diagonal_term() -> Term:
    return Term(
        kind=TermKind.DIAGONAL,
        expression="DIAG[sum ...] (am=bn)",
        variables=["m"],
        ranges=[Range(variable="m", lower="1", upper="T^theta")],
        kernels=[Kernel(name="W_AFE"), Kernel(name="FourierKernel")],
        multiplicity=1,
    )


class TestDiagonalExtractOutputs:
    def test_two_outputs_from_diagonal(self, extract, diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(diagonal_term)
        results = extract.apply([diagonal_term], ledger)
        assert len(results) == 2

    def test_main_term_and_error(self, extract, diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(diagonal_term)
        results = extract.apply([diagonal_term], ledger)
        statuses = {t.status for t in results}
        assert TermStatus.MAIN_TERM in statuses
        assert TermStatus.ERROR in statuses

    def test_non_diagonal_passes_through(self, extract) -> None:
        offdiag = Term(kind=TermKind.OFF_DIAGONAL)
        ledger = TermLedger()
        ledger.add(offdiag)
        results = extract.apply([offdiag], ledger)
        assert len(results) == 1
        assert results[0].kind == TermKind.OFF_DIAGONAL


class TestMainTermScale:
    def test_main_term_T_exponent(self, extract, diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(diagonal_term)
        results = extract.apply([diagonal_term], ledger)
        main = [t for t in results if t.status == TermStatus.MAIN_TERM][0]
        assert main.metadata.get("T_exponent") == "1"

    def test_error_T_exponent_less_than_1(self, extract, diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(diagonal_term)
        results = extract.apply([diagonal_term], ledger)
        error = [t for t in results if t.status == TermStatus.ERROR][0]
        t_exp = error.metadata.get("T_exponent", "")
        assert "delta" in t_exp and "1" in t_exp


class TestMainTermPoly:
    def test_poly_evaluate(self) -> None:
        poly = MainTermPoly(
            coefficients=[("a", "1"), ("b", "theta")],
        )
        val = poly.evaluate(0.5)
        assert abs(val - 1.5) < 1e-10

    def test_poly_to_sympy(self) -> None:
        poly = MainTermPoly(
            coefficients=[("a", "1"), ("b", "-1/3")],
        )
        expr = poly.to_sympy()
        assert expr is not None

    def test_poly_to_dict(self) -> None:
        poly = MainTermPoly(
            coefficients=[("a", "1"), ("b", "theta")],
            description="test",
        )
        d = poly.to_dict()
        assert "coefficients" in d
        assert len(d["coefficients"]) == 2

    def test_main_term_has_poly_metadata(self, extract, diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(diagonal_term)
        results = extract.apply([diagonal_term], ledger)
        main = [t for t in results if t.status == TermStatus.MAIN_TERM][0]
        assert "main_term_poly" in main.metadata
        assert "coefficients" in main.metadata["main_term_poly"]


class TestKernelPreservation:
    def test_kernels_on_main_term(self, extract, diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(diagonal_term)
        results = extract.apply([diagonal_term], ledger)
        main = [t for t in results if t.status == TermStatus.MAIN_TERM][0]
        kernel_names = {k.name for k in main.kernels}
        assert "W_AFE" in kernel_names
        assert "FourierKernel" in kernel_names


class TestHistory:
    def test_history_chain(self, extract, diagonal_term) -> None:
        ledger = TermLedger()
        ledger.add(diagonal_term)
        results = extract.apply([diagonal_term], ledger)
        for t in results:
            assert t.history[-1].transform == "DiagonalExtract"
            assert diagonal_term.id in t.parents
