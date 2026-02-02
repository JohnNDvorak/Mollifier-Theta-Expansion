"""Tests for KernelState state machine and invariants."""

from __future__ import annotations

import pytest

from mollifier_theta.core.invariants import (
    check_kernel_state_consistency,
    check_kernel_state_transition,
)
from mollifier_theta.core.ir import (
    KERNEL_STATE_TRANSITIONS,
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.delta_method import (
    DeltaMethodCollapse,
    DeltaMethodSetup,
)
from mollifier_theta.transforms.kloosterman_form import KloostermanForm


@pytest.fixture
def off_diagonal_term() -> Term:
    return Term(
        kind=TermKind.OFF_DIAGONAL,
        expression="OFFDIAG test",
        variables=["m", "n"],
        ranges=[
            Range(variable="m", lower="1", upper="T^theta"),
            Range(variable="n", lower="1", upper="T^theta"),
        ],
        kernels=[Kernel(name="W_AFE"), Kernel(name="FourierKernel")],
        phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
    )


class TestKernelStateEnum:
    def test_all_states_exist(self) -> None:
        assert KernelState.NONE is not None
        assert KernelState.UNCOLLAPSED_DELTA is not None
        assert KernelState.VORONOI_APPLIED is not None
        assert KernelState.COLLAPSED is not None
        assert KernelState.KLOOSTERMANIZED is not None

    def test_default_is_none(self) -> None:
        t = Term(kind=TermKind.INTEGRAL)
        assert t.kernel_state == KernelState.NONE


class TestKernelStateTransitions:
    def test_none_to_uncollapsed_legal(self) -> None:
        assert check_kernel_state_transition(
            KernelState.NONE, KernelState.UNCOLLAPSED_DELTA
        ) == []

    def test_uncollapsed_to_collapsed_legal(self) -> None:
        assert check_kernel_state_transition(
            KernelState.UNCOLLAPSED_DELTA, KernelState.COLLAPSED
        ) == []

    def test_uncollapsed_to_voronoi_legal(self) -> None:
        assert check_kernel_state_transition(
            KernelState.UNCOLLAPSED_DELTA, KernelState.VORONOI_APPLIED
        ) == []

    def test_voronoi_to_collapsed_legal(self) -> None:
        assert check_kernel_state_transition(
            KernelState.VORONOI_APPLIED, KernelState.COLLAPSED
        ) == []

    def test_collapsed_to_kloostermanized_legal(self) -> None:
        assert check_kernel_state_transition(
            KernelState.COLLAPSED, KernelState.KLOOSTERMANIZED
        ) == []

    def test_none_to_collapsed_illegal(self) -> None:
        violations = check_kernel_state_transition(
            KernelState.NONE, KernelState.COLLAPSED
        )
        assert len(violations) == 1
        assert "Illegal" in violations[0]

    def test_collapsed_to_uncollapsed_illegal(self) -> None:
        violations = check_kernel_state_transition(
            KernelState.COLLAPSED, KernelState.UNCOLLAPSED_DELTA
        )
        assert len(violations) == 1

    def test_kloostermanized_to_spectralized_legal(self) -> None:
        assert check_kernel_state_transition(
            KernelState.KLOOSTERMANIZED, KernelState.SPECTRALIZED
        ) == []

    def test_kloostermanized_other_transitions_illegal(self) -> None:
        for state in KernelState:
            if state in (KernelState.KLOOSTERMANIZED, KernelState.SPECTRALIZED):
                continue
            violations = check_kernel_state_transition(
                KernelState.KLOOSTERMANIZED, state
            )
            assert len(violations) == 1

    def test_spectralized_terminal(self) -> None:
        for state in KernelState:
            if state == KernelState.SPECTRALIZED:
                continue
            violations = check_kernel_state_transition(
                KernelState.SPECTRALIZED, state
            )
            assert len(violations) == 1

    def test_collapsed_to_spectralized_illegal(self) -> None:
        violations = check_kernel_state_transition(
            KernelState.COLLAPSED, KernelState.SPECTRALIZED
        )
        assert len(violations) == 1


class TestKernelStateInPipeline:
    def test_setup_produces_uncollapsed(self, off_diagonal_term: Term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        results = DeltaMethodSetup().apply([off_diagonal_term], ledger)
        assert results[0].kernel_state == KernelState.UNCOLLAPSED_DELTA

    def test_collapse_produces_collapsed(self, off_diagonal_term: Term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        setup = DeltaMethodSetup().apply([off_diagonal_term], ledger)
        results = DeltaMethodCollapse().apply(setup, ledger)
        assert results[0].kernel_state == KernelState.COLLAPSED

    def test_kloosterman_produces_kloostermanized(self, off_diagonal_term: Term) -> None:
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        setup = DeltaMethodSetup().apply([off_diagonal_term], ledger)
        collapsed = DeltaMethodCollapse().apply(setup, ledger)
        results = KloostermanForm().apply(collapsed, ledger)
        assert results[0].kernel_state == KernelState.KLOOSTERMANIZED


class TestKernelStateConsistency:
    def test_uncollapsed_consistent(self) -> None:
        t = Term(
            kind=TermKind.OFF_DIAGONAL,
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            metadata={"delta_method_applied": True, "delta_method_collapsed": False},
        )
        assert check_kernel_state_consistency(t) == []

    def test_uncollapsed_missing_applied_flag(self) -> None:
        t = Term(
            kind=TermKind.OFF_DIAGONAL,
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            metadata={},
        )
        violations = check_kernel_state_consistency(t)
        assert len(violations) >= 1

    def test_uncollapsed_but_collapsed_true(self) -> None:
        t = Term(
            kind=TermKind.OFF_DIAGONAL,
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            metadata={"delta_method_applied": True, "delta_method_collapsed": True},
        )
        violations = check_kernel_state_consistency(t)
        assert len(violations) >= 1
