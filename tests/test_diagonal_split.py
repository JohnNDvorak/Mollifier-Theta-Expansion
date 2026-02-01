"""Tests for DiagonalSplit transform."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Kernel, Phase, Range, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.diagonal_split import DiagonalSplit


@pytest.fixture
def split() -> DiagonalSplit:
    return DiagonalSplit()


@pytest.fixture
def cross_term_with_phases() -> Term:
    return Term(
        kind=TermKind.CROSS,
        expression="sum a_m conj(b_n) K(log(m/n))",
        variables=["m", "n"],
        ranges=[
            Range(variable="m", lower="1", upper="T^theta"),
            Range(variable="n", lower="1", upper="T^theta"),
        ],
        kernels=[
            Kernel(name="W_AFE", argument="n/x"),
            Kernel(name="FourierKernel", argument="log(am/bn)"),
        ],
        phases=[
            Phase(expression="(m/n)^{it}", depends_on=["m", "n"]),
        ],
        multiplicity=2,
    )


class TestSplitOutputCount:
    def test_two_outputs_per_input(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        assert len(results) == 2

    def test_multiple_inputs(self, split) -> None:
        terms = [
            Term(kind=TermKind.CROSS, variables=["m", "n"])
            for _ in range(4)
        ]
        ledger = TermLedger()
        ledger.add_many(terms)
        results = split.apply(terms, ledger)
        assert len(results) == 8


class TestSplitKindAssignment:
    def test_diagonal_kind(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        diag = [t for t in results if t.kind == TermKind.DIAGONAL]
        assert len(diag) == 1

    def test_off_diagonal_kind(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        offdiag = [t for t in results if t.kind == TermKind.OFF_DIAGONAL]
        assert len(offdiag) == 1


class TestSplitPhaseRetention:
    def test_off_diagonal_retains_all_phases(
        self, split, cross_term_with_phases
    ) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        offdiag = [t for t in results if t.kind == TermKind.OFF_DIAGONAL][0]
        assert len(offdiag.phases) == len(cross_term_with_phases.phases)

    def test_diagonal_removes_mn_oscillatory_phase(
        self, split, cross_term_with_phases
    ) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        diag = [t for t in results if t.kind == TermKind.DIAGONAL][0]
        # The (m/n)^{it} phase should be removed on diagonal
        mn_phases = [
            p for p in diag.phases
            if "m" in p.depends_on and "n" in p.depends_on
        ]
        assert len(mn_phases) == 0

    def test_non_mn_phase_retained_on_diagonal(self, split) -> None:
        term = Term(
            kind=TermKind.CROSS,
            variables=["m", "n"],
            phases=[
                Phase(expression="(m/n)^{it}", depends_on=["m", "n"]),
                Phase(expression="e(m/c)", depends_on=["m", "c"]),
            ],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = split.apply([term], ledger)
        diag = [t for t in results if t.kind == TermKind.DIAGONAL][0]
        assert len(diag.phases) == 1
        assert diag.phases[0].expression == "e(m/c)"


class TestSplitKernelPreservation:
    def test_both_outputs_keep_kernels(
        self, split, cross_term_with_phases
    ) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        for t in results:
            kernel_names = {k.name for k in t.kernels}
            assert "W_AFE" in kernel_names
            assert "FourierKernel" in kernel_names


class TestSplitMultiplicity:
    def test_multiplicity_preserved(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        for t in results:
            assert t.multiplicity == 2


class TestSplitHistory:
    def test_history_appended(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        for t in results:
            assert t.history[-1].transform == "DiagonalSplit"

    def test_parent_recorded(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        for t in results:
            assert cross_term_with_phases.id in t.parents


class TestSplitMetadata:
    def test_diagonal_metadata(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        diag = [t for t in results if t.kind == TermKind.DIAGONAL][0]
        assert diag.metadata.get("split_role") == "diagonal"

    def test_off_diagonal_metadata(self, split, cross_term_with_phases) -> None:
        ledger = TermLedger()
        ledger.add(cross_term_with_phases)
        results = split.apply([cross_term_with_phases], ledger)
        offdiag = [t for t in results if t.kind == TermKind.OFF_DIAGONAL][0]
        assert offdiag.metadata.get("split_role") == "off_diagonal"
