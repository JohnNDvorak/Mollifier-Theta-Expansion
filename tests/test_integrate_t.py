"""Tests for IntegrateOverT transform."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Kernel, Phase, Range, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.integrate_t import IntegrateOverT


@pytest.fixture
def cross_term() -> Term:
    return Term(
        kind=TermKind.CROSS,
        expression="sum a_m conj(b_n) (m/n)^{it} W(m) W(n)",
        variables=["m", "n", "t"],
        ranges=[
            Range(variable="m", lower="1", upper="T^theta"),
            Range(variable="n", lower="1", upper="T^theta"),
            Range(variable="t", lower="0", upper="T"),
        ],
        kernels=[Kernel(name="W_AFE", argument="n/x")],
        phases=[
            Phase(
                expression="(m/n)^{it}",
                depends_on=["m", "n", "t"],
            )
        ],
    )


@pytest.fixture
def integrate() -> IntegrateOverT:
    return IntegrateOverT()


class TestFourierKernelAttachment:
    def test_fourier_kernel_added(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        assert len(results) == 1
        kernel_names = [k.name for k in results[0].kernels]
        assert "FourierKernel" in kernel_names

    def test_fourier_kernel_argument(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        fk = [k for k in results[0].kernels if k.name == "FourierKernel"][0]
        assert fk.argument == "log(am/bn)"

    def test_not_delta_approximated(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        fk = [k for k in results[0].kernels if k.name == "FourierKernel"][0]
        assert fk.properties.get("not_delta_approximated") is True


class TestFourierKernelRetention:
    def test_original_kernels_retained(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        kernel_names = [k.name for k in results[0].kernels]
        assert "W_AFE" in kernel_names
        assert "FourierKernel" in kernel_names

    def test_kernel_count_increases(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        assert len(results[0].kernels) == len(cross_term.kernels) + 1


class TestTVariableRemoval:
    def test_t_removed_from_variables(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        assert "t" not in results[0].variables

    def test_other_variables_retained(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        assert "m" in results[0].variables
        assert "n" in results[0].variables

    def test_t_range_removed(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        range_vars = [r.variable for r in results[0].ranges]
        assert "t" not in range_vars


class TestPhaseHandling:
    def test_mixed_phase_t_removed(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        for p in results[0].phases:
            assert "t" not in p.depends_on

    def test_pure_t_phase_consumed(self, integrate) -> None:
        term = Term(
            kind=TermKind.CROSS,
            variables=["m", "n", "t"],
            ranges=[Range(variable="t", lower="0", upper="T")],
            phases=[Phase(expression="e^{2pi i t}", depends_on=["t"])],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = integrate.apply([term], ledger)
        # Pure t-phase consumed
        assert len(results[0].phases) == 0

    def test_non_t_phase_retained(self, integrate) -> None:
        term = Term(
            kind=TermKind.CROSS,
            variables=["m", "n", "t"],
            ranges=[Range(variable="t", lower="0", upper="T")],
            phases=[Phase(expression="e(m/c)", depends_on=["m", "c"])],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = integrate.apply([term], ledger)
        assert len(results[0].phases) == 1
        assert results[0].phases[0].expression == "e(m/c)"


class TestHistoryChain:
    def test_history_appended(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        assert results[0].history[-1].transform == "IntegrateOverT"

    def test_parent_id_recorded(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        results = integrate.apply([cross_term], ledger)
        assert cross_term.id in results[0].parents


class TestPurity:
    def test_does_not_mutate_input(self, integrate, cross_term) -> None:
        ledger = TermLedger()
        ledger.add(cross_term)
        orig_vars = list(cross_term.variables)
        integrate.apply([cross_term], ledger)
        assert cross_term.variables == orig_vars

    def test_multiple_terms(self, integrate) -> None:
        terms = [
            Term(
                kind=TermKind.CROSS,
                variables=["m", "n", "t"],
                ranges=[Range(variable="t", lower="0", upper="T")],
            )
            for _ in range(3)
        ]
        ledger = TermLedger()
        ledger.add_many(terms)
        results = integrate.apply(terms, ledger)
        assert len(results) == 3
