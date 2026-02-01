"""Tests for deterministic PhaseAbsorb with structural proof."""

from __future__ import annotations

from mollifier_theta.core.ir import Kernel, Phase, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.transforms.phase_absorb import PhaseAbsorb, verify_absorption_invariant


class TestStructuralProof:
    def test_unit_modulus_phase_absorbed(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="e(am/c)", is_separable=True, unit_modulus=True),
            ],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = PhaseAbsorb().apply([term], ledger)
        assert results[0].phases[0].absorbed

    def test_non_unit_modulus_phase_not_absorbed(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="c^{1/2}", is_separable=True, unit_modulus=False),
            ],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = PhaseAbsorb().apply([term], ledger)
        assert not results[0].phases[0].absorbed

    def test_absorption_proof_metadata(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="e(am/c)", is_separable=True, unit_modulus=True),
            ],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = PhaseAbsorb().apply([term], ledger)
        assert results[0].metadata.get("absorption_proof") == "unit_modulus_isometry"


class TestVerifyAbsorptionInvariant:
    def test_valid_absorption(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="e(am/c)", is_separable=True, absorbed=True, unit_modulus=True),
            ],
        )
        assert verify_absorption_invariant(term) == []

    def test_invalid_non_unit_modulus_absorbed(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="bad", is_separable=True, absorbed=True, unit_modulus=False),
            ],
        )
        violations = verify_absorption_invariant(term)
        assert len(violations) == 1
        assert "unit_modulus=False" in violations[0]

    def test_invalid_non_separable_absorbed(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            phases=[
                Phase(expression="S(m,n;c)", is_separable=False, absorbed=True, unit_modulus=True),
            ],
        )
        violations = verify_absorption_invariant(term)
        assert len(violations) == 1
        assert "is_separable=False" in violations[0]
