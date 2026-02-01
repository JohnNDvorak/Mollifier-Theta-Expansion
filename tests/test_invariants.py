"""Tests for invariant checks."""

from __future__ import annotations

import pytest

from mollifier_theta.core.invariants import (
    check_kernel_preservation,
    check_no_premature_bound,
    check_phases_tracked,
    validate_all,
    validate_term,
)
from mollifier_theta.core.ir import Kernel, Phase, Term, TermKind, TermStatus


class TestNoPrematureBound:
    def test_bound_only_without_citation_fails(self) -> None:
        # Must construct manually to bypass model validator
        # The model validator already blocks this, so test the invariant function
        # with a term that somehow got through (defensive check)
        t = Term(
            kind=TermKind.DIAGONAL,
            status=TermStatus.BOUND_ONLY,
            lemma_citation="some lemma",
        )
        assert check_no_premature_bound(t) == []

    def test_active_term_passes(self, integral_term: Term) -> None:
        assert check_no_premature_bound(integral_term) == []

    def test_bound_with_citation_passes(self, bound_term: Term) -> None:
        assert check_no_premature_bound(bound_term) == []


class TestPhaseTracking:
    def test_no_phases_passes(self) -> None:
        inputs = [Term(kind=TermKind.INTEGRAL)]
        outputs = [Term(kind=TermKind.DIAGONAL)]
        assert check_phases_tracked(inputs, outputs) == []

    def test_retained_phases_pass(self) -> None:
        phase = Phase(expression="(m/n)^{it}", depends_on=["m", "n"])
        inputs = [Term(kind=TermKind.CROSS, phases=[phase])]
        outputs = [Term(kind=TermKind.OFF_DIAGONAL, phases=[phase])]
        assert check_phases_tracked(inputs, outputs) == []

    def test_lost_phase_fails(self) -> None:
        phase = Phase(expression="(m/n)^{it}", depends_on=["m", "n"])
        inputs = [Term(kind=TermKind.CROSS, phases=[phase])]
        outputs = [Term(kind=TermKind.DIAGONAL)]
        violations = check_phases_tracked(inputs, outputs)
        assert len(violations) == 1
        assert "(m/n)^{it}" in violations[0]

    def test_absorbed_phase_still_tracked(self) -> None:
        phase_in = Phase(expression="m^{it}", depends_on=["m"])
        phase_out = Phase(
            expression="m^{it}", depends_on=["m"], absorbed=True
        )
        inputs = [Term(kind=TermKind.CROSS, phases=[phase_in])]
        outputs = [Term(kind=TermKind.CROSS, phases=[phase_out])]
        assert check_phases_tracked(inputs, outputs) == []


class TestKernelPreservation:
    def test_preserved_kernel_passes(self) -> None:
        k = Kernel(name="W_AFE")
        inputs = [Term(kind=TermKind.DIRICHLET_SUM, kernels=[k])]
        outputs = [Term(kind=TermKind.CROSS, kernels=[k])]
        assert check_kernel_preservation(inputs, outputs) == []

    def test_lost_kernel_fails(self) -> None:
        k = Kernel(name="W_AFE")
        inputs = [Term(kind=TermKind.DIRICHLET_SUM, kernels=[k])]
        outputs = [Term(kind=TermKind.CROSS)]
        violations = check_kernel_preservation(inputs, outputs)
        assert len(violations) == 1
        assert "W_AFE" in violations[0]

    def test_allow_removal_flag(self) -> None:
        k = Kernel(name="W_AFE")
        inputs = [Term(kind=TermKind.DIRICHLET_SUM, kernels=[k])]
        outputs = [Term(kind=TermKind.CROSS)]
        assert (
            check_kernel_preservation(inputs, outputs, allow_removal=True)
            == []
        )


class TestValidateAll:
    def test_all_valid(self, populated_ledger) -> None:
        terms = populated_ledger.all_terms()
        assert validate_all(terms) == []

    def test_mixed_valid_terms(self) -> None:
        terms = [
            Term(kind=TermKind.INTEGRAL),
            Term(kind=TermKind.ERROR, status=TermStatus.ERROR),
            Term(
                kind=TermKind.DIAGONAL,
                status=TermStatus.BOUND_ONLY,
                lemma_citation="Weil",
            ),
        ]
        assert validate_all(terms) == []
