"""Tests for PhaseAbsorb transform."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Kernel, Phase, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.phase_absorb import PhaseAbsorb, spot_check_norm_preservation


@pytest.fixture
def absorb() -> PhaseAbsorb:
    return PhaseAbsorb()


@pytest.fixture
def term_with_separable_phases() -> Term:
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="sum S(m,n;c)/c ...",
        variables=["m", "n", "c"],
        kernels=[Kernel(name="W_AFE"), Kernel(name="DeltaMethodKernel")],
        phases=[
            Phase(expression="e(am/c)", depends_on=["m", "c"], is_separable=True),
            Phase(expression="e(-bn/c)", depends_on=["n", "c"], is_separable=True),
            Phase(expression="S(m,n;c)/c", depends_on=["m", "n", "c"], is_separable=False),
        ],
    )


class TestAbsorption:
    def test_separable_phases_absorbed(self, absorb, term_with_separable_phases) -> None:
        ledger = TermLedger()
        ledger.add(term_with_separable_phases)
        results = absorb.apply([term_with_separable_phases], ledger)
        absorbed = [p for p in results[0].phases if p.absorbed]
        assert len(absorbed) == 2

    def test_non_separable_phase_not_absorbed(self, absorb, term_with_separable_phases) -> None:
        ledger = TermLedger()
        ledger.add(term_with_separable_phases)
        results = absorb.apply([term_with_separable_phases], ledger)
        kloosterman = [
            p for p in results[0].phases if p.expression == "S(m,n;c)/c"
        ]
        assert len(kloosterman) == 1
        assert not kloosterman[0].absorbed

    def test_all_phases_still_present(self, absorb, term_with_separable_phases) -> None:
        ledger = TermLedger()
        ledger.add(term_with_separable_phases)
        results = absorb.apply([term_with_separable_phases], ledger)
        # Phases are not removed, just marked absorbed
        assert len(results[0].phases) == len(term_with_separable_phases.phases)

    def test_already_absorbed_not_reabsorbed(self, absorb) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="e(x)", depends_on=["x"], is_separable=True, absorbed=True),
            ],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = absorb.apply([term], ledger)
        # Should remain absorbed, not double-absorbed
        assert results[0].phases[0].absorbed

    def test_no_separable_phases_noop(self, absorb) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="S(m,n;c)/c", depends_on=["m", "n", "c"], is_separable=False),
            ],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = absorb.apply([term], ledger)
        assert not results[0].phases[0].absorbed


class TestNormPreservation:
    def test_spot_check_passes(self) -> None:
        passed, norm_before, norm_after = spot_check_norm_preservation()
        assert passed
        assert abs(norm_before - norm_after) < 1e-10

    def test_spot_check_different_seeds(self) -> None:
        for seed in [1, 2, 3, 42, 100]:
            passed, _, _ = spot_check_norm_preservation(seed=seed)
            assert passed

    def test_spot_check_various_lengths(self) -> None:
        for length in [1, 10, 100, 500]:
            passed, _, _ = spot_check_norm_preservation(length=length)
            assert passed


class TestPhaseAbsorbHistory:
    def test_history_appended(self, absorb, term_with_separable_phases) -> None:
        ledger = TermLedger()
        ledger.add(term_with_separable_phases)
        results = absorb.apply([term_with_separable_phases], ledger)
        assert results[0].history[-1].transform == "PhaseAbsorb"

    def test_parent_recorded(self, absorb, term_with_separable_phases) -> None:
        ledger = TermLedger()
        ledger.add(term_with_separable_phases)
        results = absorb.apply([term_with_separable_phases], ledger)
        assert term_with_separable_phases.id in results[0].parents


class TestPhaseAbsorbMetadata:
    def test_absorption_flag_set(self, absorb, term_with_separable_phases) -> None:
        ledger = TermLedger()
        ledger.add(term_with_separable_phases)
        results = absorb.apply([term_with_separable_phases], ledger)
        assert results[0].metadata.get("phases_absorbed") is True
