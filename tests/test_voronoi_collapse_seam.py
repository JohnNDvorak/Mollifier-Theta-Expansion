"""Tests for the Voronoi→Collapse seam (WI-4).

Validates that:
  - After Voronoi renames n→n*, DeltaMethodCollapse produces phases referencing n*
  - Without Voronoi, phases reference n (unchanged)
  - Phase depends_on ⊆ term.variables at every stage
  - Kernel state progression is legal
  - Full conrey89_voronoi_pipeline still produces correct theta_max
"""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import (
    Kernel,
    KernelState,
    KERNEL_STATE_TRANSITIONS,
    Phase,
    Range,
    Term,
    TermKind,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
from mollifier_theta.transforms.delta_method import (
    DeltaMethodCollapse,
    DeltaMethodSetup,
)
from mollifier_theta.transforms.kloosterman_form import KloostermanForm
from mollifier_theta.transforms.voronoi import VoronoiTransform


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


@pytest.fixture
def setup_term(off_diagonal_term: Term) -> tuple[Term, TermLedger]:
    ledger = TermLedger()
    ledger.add(off_diagonal_term)
    setup = DeltaMethodSetup()
    results = setup.apply([off_diagonal_term], ledger)
    return results[0], ledger


@pytest.fixture
def voronoi_term(setup_term: tuple[Term, TermLedger]) -> tuple[Term, TermLedger]:
    term, ledger = setup_term
    voronoi = VoronoiTransform(target_variable="n")
    results = voronoi.apply([term], ledger)
    return results[0], ledger


@pytest.fixture
def collapsed_after_voronoi(voronoi_term: tuple[Term, TermLedger]) -> tuple[Term, TermLedger]:
    term, ledger = voronoi_term
    collapse = DeltaMethodCollapse()
    results = collapse.apply([term], ledger)
    return results[0], ledger


@pytest.fixture
def collapsed_no_voronoi(setup_term: tuple[Term, TermLedger]) -> tuple[Term, TermLedger]:
    term, ledger = setup_term
    collapse = DeltaMethodCollapse()
    results = collapse.apply([term], ledger)
    return results[0], ledger


# ============================================================
# Phase variable tracking after Voronoi
# ============================================================
class TestPhaseVariablesAfterVoronoi:
    def test_voronoi_phases_reference_n_star(self, collapsed_after_voronoi: tuple) -> None:
        """After Setup→Voronoi→Collapse, the n-twist phase references n*."""
        term, _ = collapsed_after_voronoi
        additive = [p for p in term.phases if p.expression.startswith("e(")]
        n_phase = [p for p in additive if "n" in p.expression]
        assert len(n_phase) >= 1
        # The phase should reference n*, not plain n
        for p in n_phase:
            assert "n*" in p.expression, f"Expected n* in phase '{p.expression}'"

    def test_voronoi_phase_depends_on_n_star(self, collapsed_after_voronoi: tuple) -> None:
        """Phase depends_on should list n*, not n."""
        term, _ = collapsed_after_voronoi
        additive = [p for p in term.phases if p.expression.startswith("e(")]
        n_phase = [p for p in additive if "n*" in p.expression]
        assert len(n_phase) >= 1
        for p in n_phase:
            assert "n*" in p.depends_on, f"Expected n* in depends_on: {p.depends_on}"

    def test_no_voronoi_phases_reference_n(self, collapsed_no_voronoi: tuple) -> None:
        """Without Voronoi, the n-twist phase references plain n."""
        term, _ = collapsed_no_voronoi
        additive = [p for p in term.phases if p.expression.startswith("e(")]
        n_phase = [p for p in additive if "n" in p.expression and "n*" not in p.expression]
        assert len(n_phase) >= 1

    def test_m_phase_unchanged_after_voronoi(self, collapsed_after_voronoi: tuple) -> None:
        """The m-twist phase is unaffected by Voronoi on n."""
        term, _ = collapsed_after_voronoi
        additive = [p for p in term.phases if p.expression.startswith("e(")]
        m_phase = [p for p in additive if "/c)" in p.expression and "m" in p.expression and "n" not in p.expression]
        assert len(m_phase) >= 1
        for p in m_phase:
            assert "m" in p.depends_on


# ============================================================
# Phase depends_on ⊆ term.variables at every stage
# ============================================================
class TestPhaseDependsOnSubset:
    def test_setup_phase_deps_subset(self, setup_term: tuple) -> None:
        term, _ = setup_term
        var_set = set(term.variables)
        for p in term.phases:
            assert set(p.depends_on) <= var_set, (
                f"Phase '{p.expression}' depends_on {p.depends_on} "
                f"not subset of variables {term.variables}"
            )

    def test_voronoi_phase_deps_subset(self, voronoi_term: tuple) -> None:
        term, _ = voronoi_term
        var_set = set(term.variables)
        for p in term.phases:
            assert set(p.depends_on) <= var_set, (
                f"Phase '{p.expression}' depends_on {p.depends_on} "
                f"not subset of variables {term.variables}"
            )

    def test_collapse_after_voronoi_phase_deps_subset(self, collapsed_after_voronoi: tuple) -> None:
        term, _ = collapsed_after_voronoi
        var_set = set(term.variables)
        for p in term.phases:
            assert set(p.depends_on) <= var_set, (
                f"Phase '{p.expression}' depends_on {p.depends_on} "
                f"not subset of variables {term.variables}"
            )

    def test_collapse_no_voronoi_phase_deps_subset(self, collapsed_no_voronoi: tuple) -> None:
        term, _ = collapsed_no_voronoi
        var_set = set(term.variables)
        for p in term.phases:
            assert set(p.depends_on) <= var_set, (
                f"Phase '{p.expression}' depends_on {p.depends_on} "
                f"not subset of variables {term.variables}"
            )


# ============================================================
# Kernel state progression
# ============================================================
class TestKernelStateProgression:
    def test_setup_state(self, setup_term: tuple) -> None:
        term, _ = setup_term
        assert term.kernel_state == KernelState.UNCOLLAPSED_DELTA

    def test_voronoi_state(self, voronoi_term: tuple) -> None:
        term, _ = voronoi_term
        assert term.kernel_state == KernelState.VORONOI_APPLIED

    def test_collapse_after_voronoi_state(self, collapsed_after_voronoi: tuple) -> None:
        term, _ = collapsed_after_voronoi
        assert term.kernel_state == KernelState.COLLAPSED

    def test_collapse_no_voronoi_state(self, collapsed_no_voronoi: tuple) -> None:
        term, _ = collapsed_no_voronoi
        assert term.kernel_state == KernelState.COLLAPSED

    def test_kloosterman_state(self, collapsed_after_voronoi: tuple) -> None:
        term, ledger = collapsed_after_voronoi
        kloos = KloostermanForm()
        results = kloos.apply([term], ledger)
        assert results[0].kernel_state == KernelState.KLOOSTERMANIZED

    def test_full_progression_legal(self, off_diagonal_term: Term) -> None:
        """Every transition in Setup→Voronoi→Collapse→Kloosterman is legal."""
        ledger = TermLedger()
        ledger.add(off_diagonal_term)

        states = [off_diagonal_term.kernel_state]  # NONE

        setup = DeltaMethodSetup()
        terms = setup.apply([off_diagonal_term], ledger)
        states.append(terms[0].kernel_state)  # UNCOLLAPSED_DELTA

        voronoi = VoronoiTransform(target_variable="n")
        terms = voronoi.apply(terms, ledger)
        states.append(terms[0].kernel_state)  # VORONOI_APPLIED

        collapse = DeltaMethodCollapse()
        terms = collapse.apply(terms, ledger)
        states.append(terms[0].kernel_state)  # COLLAPSED

        kloos = KloostermanForm()
        terms = kloos.apply(terms, ledger)
        states.append(terms[0].kernel_state)  # KLOOSTERMANIZED

        expected = [
            KernelState.NONE,
            KernelState.UNCOLLAPSED_DELTA,
            KernelState.VORONOI_APPLIED,
            KernelState.COLLAPSED,
            KernelState.KLOOSTERMANIZED,
        ]
        assert states == expected

        # Verify each transition is legal per KERNEL_STATE_TRANSITIONS
        for prev, curr in zip(states[:-1], states[1:]):
            assert curr in KERNEL_STATE_TRANSITIONS[prev], (
                f"Illegal transition: {prev} -> {curr}"
            )


# ============================================================
# SumStructure propagation
# ============================================================
class TestSumStructurePropagation:
    def test_voronoi_updates_sum_structure(self, voronoi_term: tuple) -> None:
        """After Voronoi, the SumStructure's twist references n*."""
        term, _ = voronoi_term
        ss = term.metadata["sum_structure"]
        twists = ss["additive_twists"]
        n_twists = [t for t in twists if "n*" in t["sum_variable"]]
        assert len(n_twists) >= 1

    def test_collapse_reads_updated_structure(self, collapsed_after_voronoi: tuple) -> None:
        """DeltaMethodCollapse builds phases from the Voronoi-updated SumStructure."""
        term, _ = collapsed_after_voronoi
        # The SumStructure should still be present
        assert "sum_structure" in term.metadata
        # And the phases should reflect the updated structure
        additive = [p for p in term.phases if p.expression.startswith("e(")]
        exprs = {p.expression for p in additive}
        # Should have an n*-referencing phase
        assert any("n*" in e for e in exprs), f"No n* phase found in {exprs}"


# ============================================================
# Full pipeline integration
# ============================================================
class TestVoronoiPipelineIntegration:
    def test_voronoi_pipeline_theta_max(self) -> None:
        """Voronoi pipeline gives theta_max = 5/8 (PostVoronoi bound is binding)."""
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        assert result.theta_max is not None
        assert abs(result.theta_max - 5 / 8) < 1e-10

    def test_voronoi_pipeline_admissible(self) -> None:
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        assert result.theta_admissible is True

    def test_voronoi_pipeline_inadmissible_above(self) -> None:
        # PostVoronoi theta_max = 5/8 = 0.625, so 0.63 is inadmissible
        result = conrey89_voronoi_pipeline(theta_val=0.63)
        assert result.theta_admissible is False

    def test_voronoi_pipeline_transform_chain(self) -> None:
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        chain = result.report_data["transform_chain"]
        assert "DeltaMethodSetup" in chain
        assert "VoronoiTransform(n)" in chain
        assert "DeltaMethodCollapse" in chain
