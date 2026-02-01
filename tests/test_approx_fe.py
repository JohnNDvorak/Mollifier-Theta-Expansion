"""Tests for ApproxFunctionalEq transform."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Term, TermKind, TermStatus, Range
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.approx_fe import ApproxFunctionalEq


@pytest.fixture
def afe() -> ApproxFunctionalEq:
    return ApproxFunctionalEq()


@pytest.fixture
def input_integral() -> Term:
    return Term(
        kind=TermKind.INTEGRAL,
        expression="int_0^T |M*zeta|^2 dt",
        variables=["t"],
        ranges=[Range(variable="t", lower="0", upper="T")],
    )


class TestApproxFEOutputCount:
    def test_three_outputs(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        assert len(results) == 3

    def test_two_dirichlet_sums_one_error(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        dirichlet = [t for t in results if t.kind == TermKind.DIRICHLET_SUM]
        errors = [t for t in results if t.kind == TermKind.ERROR]
        assert len(dirichlet) == 2
        assert len(errors) == 1


class TestApproxFEKernels:
    def test_kernel_attached_to_short_sum(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        short = [t for t in results if t.metadata.get("afe_role") == "short_sum"][0]
        assert len(short.kernels) == 1
        assert short.kernels[0].name == "W_AFE"

    def test_kernel_attached_to_long_sum(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        long = [t for t in results if t.metadata.get("afe_role") == "long_sum"][0]
        assert len(long.kernels) == 1
        assert long.kernels[0].name == "W_AFE_tilde"

    def test_kernel_has_properties(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        short = [t for t in results if t.metadata.get("afe_role") == "short_sum"][0]
        assert short.kernels[0].properties.get("rapid_decay") is True


class TestApproxFEError:
    def test_error_status(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        error = [t for t in results if t.kind == TermKind.ERROR][0]
        assert error.status == TermStatus.ERROR

    def test_error_scale(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        error = [t for t in results if t.kind == TermKind.ERROR][0]
        assert "T^(-A)" in error.scale_model


class TestApproxFEHistory:
    def test_history_chain(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        for t in results:
            assert len(t.history) >= 1
            assert t.history[-1].transform == "ApproxFunctionalEq"
            assert input_integral.id in t.history[-1].parent_ids

    def test_parents_set(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        results = afe.apply([input_integral], ledger)
        for t in results:
            assert input_integral.id in t.parents


class TestApproxFEPurity:
    def test_does_not_mutate_input(self, afe, input_integral) -> None:
        ledger = TermLedger()
        ledger.add(input_integral)
        original_id = input_integral.id
        original_kind = input_integral.kind
        afe.apply([input_integral], ledger)
        assert input_integral.id == original_id
        assert input_integral.kind == original_kind

    def test_describe(self, afe) -> None:
        desc = afe.describe()
        assert "approximate functional equation" in desc.lower()
