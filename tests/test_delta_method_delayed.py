"""Tests for the two-stage delta method architecture (DeltaMethodSetup + DeltaMethodCollapse)."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import Kernel, KernelState, Phase, Range, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.transforms.delta_method import (
    DeltaMethodCollapse,
    DeltaMethodInsert,
    DeltaMethodSetup,
)
from mollifier_theta.transforms.kloosterman_form import KloostermanForm


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
def setup_result(off_diagonal_term: Term) -> tuple[list[Term], TermLedger]:
    ledger = TermLedger()
    ledger.add(off_diagonal_term)
    setup = DeltaMethodSetup()
    results = setup.apply([off_diagonal_term], ledger)
    return results, ledger


@pytest.fixture
def collapse_result(setup_result: tuple[list[Term], TermLedger]) -> tuple[list[Term], TermLedger]:
    intermediate, ledger = setup_result
    collapse = DeltaMethodCollapse()
    results = collapse.apply(intermediate, ledger)
    return results, ledger


# ============================================================
# TestDeltaMethodSetup
# ============================================================
class TestDeltaMethodSetup:
    def test_modulus_variable_added(self, setup_result: tuple) -> None:
        results, _ = setup_result
        assert "c" in results[0].variables

    def test_modulus_range_added(self, setup_result: tuple) -> None:
        results, _ = setup_result
        range_vars = {r.variable for r in results[0].ranges}
        assert "c" in range_vars

    def test_no_additive_phases(self, setup_result: tuple) -> None:
        """Stage 1 should NOT add additive character phases."""
        results, _ = setup_result
        phase_exprs = {p.expression for p in results[0].phases}
        assert "e(a*m/c)" not in phase_exprs
        assert "e(-b*n/c)" not in phase_exprs
        assert "e(am/c)" not in phase_exprs
        assert "e(-bn/c)" not in phase_exprs

    def test_kernel_uncollapsed(self, setup_result: tuple) -> None:
        results, _ = setup_result
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert dk.properties["collapsed"] is False

    def test_kernel_has_test_function(self, setup_result: tuple) -> None:
        results, _ = setup_result
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert dk.properties["test_function"] == "h"

    def test_kernel_has_oscillatory_argument(self, setup_result: tuple) -> None:
        results, _ = setup_result
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert dk.properties["oscillatory_argument"] == "x(am-bn)/cq"

    def test_kernel_has_collapse_conditions(self, setup_result: tuple) -> None:
        results, _ = setup_result
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert "stationary_phase_valid" in dk.properties["collapse_conditions"]
        assert "test_function_smooth" in dk.properties["collapse_conditions"]

    def test_integral_form_argument(self, setup_result: tuple) -> None:
        results, _ = setup_result
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert "integral" in dk.argument

    def test_passthrough_non_offdiag(self) -> None:
        diag = Term(kind=TermKind.DIAGONAL)
        ledger = TermLedger()
        ledger.add(diag)
        setup = DeltaMethodSetup()
        results = setup.apply([diag], ledger)
        assert len(results) == 1
        assert results[0].kind == TermKind.DIAGONAL

    def test_original_kernels_preserved(self, setup_result: tuple) -> None:
        results, _ = setup_result
        kernel_names = {k.name for k in results[0].kernels}
        assert "W_AFE" in kernel_names
        assert "FourierKernel" in kernel_names

    def test_metadata_flags(self, setup_result: tuple) -> None:
        results, _ = setup_result
        m = results[0].metadata
        assert m["delta_method_applied"] is True
        assert m["delta_method_collapsed"] is False
        assert m["delta_method_stage"] == "setup"

    def test_history_entry(self, setup_result: tuple) -> None:
        results, _ = setup_result
        assert results[0].history[-1].transform == "DeltaMethodSetup"


# ============================================================
# TestDeltaMethodCollapse
# ============================================================
class TestDeltaMethodCollapse:
    def test_additive_phases_added(self, collapse_result: tuple) -> None:
        results, _ = collapse_result
        phase_exprs = {p.expression for p in results[0].phases}
        assert "e(a*m/c)" in phase_exprs
        assert "e(-b*n/c)" in phase_exprs

    def test_additive_phases_separable(self, collapse_result: tuple) -> None:
        results, _ = collapse_result
        additive = [p for p in results[0].phases if p.expression.startswith("e(")]
        assert all(p.is_separable for p in additive)

    def test_kernel_collapsed(self, collapse_result: tuple) -> None:
        results, _ = collapse_result
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert dk.properties["collapsed"] is True

    def test_kernel_argument_collapsed(self, collapse_result: tuple) -> None:
        results, _ = collapse_result
        dk = [k for k in results[0].kernels if k.name == "DeltaMethodKernel"][0]
        assert dk.argument == "(a*m-b*n)/c"

    def test_metadata_flags(self, collapse_result: tuple) -> None:
        results, _ = collapse_result
        m = results[0].metadata
        assert m["delta_method_applied"] is True
        assert m["delta_method_collapsed"] is True
        assert m["delta_method_stage"] == "collapsed"

    def test_passthrough_already_collapsed(self) -> None:
        """A term that's already collapsed should pass through."""
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            metadata={
                "delta_method_applied": True,
                "delta_method_collapsed": True,
            },
        )
        ledger = TermLedger()
        ledger.add(term)
        collapse = DeltaMethodCollapse()
        results = collapse.apply([term], ledger)
        assert len(results) == 1
        assert results[0].id == term.id  # unchanged

    def test_passthrough_no_delta_metadata(self) -> None:
        """A term without delta_method metadata should pass through."""
        term = Term(kind=TermKind.OFF_DIAGONAL)
        ledger = TermLedger()
        ledger.add(term)
        collapse = DeltaMethodCollapse()
        results = collapse.apply([term], ledger)
        assert results[0].id == term.id

    def test_history_entry(self, collapse_result: tuple) -> None:
        results, _ = collapse_result
        assert results[0].history[-1].transform == "DeltaMethodCollapse"


# ============================================================
# TestDeltaMethodComposition
# ============================================================
class TestDeltaMethodComposition:
    """setup + collapse should produce equivalent output to DeltaMethodInsert."""

    @pytest.fixture(autouse=True)
    def _run_both(self, off_diagonal_term: Term) -> None:
        # Two-stage
        ledger_2 = TermLedger()
        ledger_2.add(off_diagonal_term)
        setup = DeltaMethodSetup()
        collapse = DeltaMethodCollapse()
        intermediate = setup.apply([off_diagonal_term], ledger_2)
        self.two_stage = collapse.apply(intermediate, ledger_2)
        self.ledger_2 = ledger_2

        # Single-stage (backward-compatible wrapper)
        ledger_1 = TermLedger()
        ledger_1.add(off_diagonal_term)
        insert = DeltaMethodInsert()
        self.single_stage = insert.apply([off_diagonal_term], ledger_1)
        self.ledger_1 = ledger_1

    def test_same_phases(self) -> None:
        two_phases = {p.expression for p in self.two_stage[0].phases}
        one_phases = {p.expression for p in self.single_stage[0].phases}
        assert two_phases == one_phases

    def test_same_kernel_names(self) -> None:
        two_kernels = {k.name for k in self.two_stage[0].kernels}
        one_kernels = {k.name for k in self.single_stage[0].kernels}
        assert two_kernels == one_kernels

    def test_same_variables(self) -> None:
        assert set(self.two_stage[0].variables) == set(self.single_stage[0].variables)

    def test_same_range_variables(self) -> None:
        two_rv = {r.variable for r in self.two_stage[0].ranges}
        one_rv = {r.variable for r in self.single_stage[0].ranges}
        assert two_rv == one_rv

    def test_kernel_collapsed_in_both(self) -> None:
        for result in (self.two_stage, self.single_stage):
            dk = [k for k in result[0].kernels if k.name == "DeltaMethodKernel"][0]
            assert dk.properties["collapsed"] is True

    def test_two_stage_has_two_history_entries(self) -> None:
        """Two-stage should have one more history entry than the original term."""
        # The two-stage result has: original history + DeltaMethodSetup + DeltaMethodCollapse
        transforms = [h.transform for h in self.two_stage[0].history]
        assert "DeltaMethodSetup" in transforms
        assert "DeltaMethodCollapse" in transforms

    def test_intermediate_in_ledger(self) -> None:
        """The intermediate (setup) term should be in the two-stage ledger."""
        all_terms = self.ledger_2.all_terms()
        setup_terms = [
            t for t in all_terms
            if t.metadata.get("delta_method_stage") == "setup"
        ]
        assert len(setup_terms) > 0


# ============================================================
# TestKloostermanFormWithUncollapsed
# ============================================================
class TestKloostermanFormWithUncollapsed:
    def test_uncollapsed_passes_through(self, off_diagonal_term: Term) -> None:
        """KloostermanForm should not consume uncollapsed delta-method terms."""
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        setup = DeltaMethodSetup()
        intermediate = setup.apply([off_diagonal_term], ledger)

        kloos = KloostermanForm()
        results = kloos.apply(intermediate, ledger)
        # Should pass through since delta_method_collapsed is False
        assert results[0].kind == TermKind.OFF_DIAGONAL
        assert results[0].metadata.get("delta_method_collapsed") is False

    def test_collapsed_consumed(self, off_diagonal_term: Term) -> None:
        """KloostermanForm should consume collapsed delta-method terms."""
        ledger = TermLedger()
        ledger.add(off_diagonal_term)
        setup = DeltaMethodSetup()
        collapse = DeltaMethodCollapse()
        intermediate = setup.apply([off_diagonal_term], ledger)
        collapsed = collapse.apply(intermediate, ledger)

        kloos = KloostermanForm()
        results = kloos.apply(collapsed, ledger)
        assert results[0].kind == TermKind.KLOOSTERMAN

    def test_legacy_term_without_kernel_state_passes_through(self) -> None:
        """A legacy term with metadata flags but kernel_state=NONE is NOT consumed.

        kernel_state is the sole gating authority (WI-3).
        """
        legacy_term = Term(
            kind=TermKind.OFF_DIAGONAL,
            metadata={
                "delta_method_applied": True,
            },
            phases=[
                Phase(expression="e(am/c)", is_separable=True),
                Phase(expression="e(-bn/c)", is_separable=True),
            ],
        )
        ledger = TermLedger()
        ledger.add(legacy_term)
        kloos = KloostermanForm()
        results = kloos.apply([legacy_term], ledger)
        # Passes through because kernel_state is NONE, not COLLAPSED
        assert results[0].kind == TermKind.OFF_DIAGONAL

    def test_kernel_state_collapsed_consumed(self) -> None:
        """A term with kernel_state=COLLAPSED is consumed by KloostermanForm."""
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            kernel_state=KernelState.COLLAPSED,
            phases=[
                Phase(expression="e(am/c)", is_separable=True),
                Phase(expression="e(-bn/c)", is_separable=True),
            ],
        )
        ledger = TermLedger()
        ledger.add(term)
        kloos = KloostermanForm()
        results = kloos.apply([term], ledger)
        assert results[0].kind == TermKind.KLOOSTERMAN


# ============================================================
# TestDelayedCollapseIntegration
# ============================================================
class TestDelayedCollapseIntegration:
    def test_full_pipeline_theta_max(self) -> None:
        """Full pipeline with two-stage delta method still gives theta_max = 4/7."""
        result = conrey89_pipeline(theta_val=0.56)
        assert result.theta_max is not None
        assert abs(result.theta_max - 4 / 7) < 1e-10

    def test_full_pipeline_admissible(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert result.theta_admissible is True

    def test_full_pipeline_inadmissible_above(self) -> None:
        result = conrey89_pipeline(theta_val=0.58)
        assert result.theta_admissible is False

    def test_transform_chain_updated(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        chain = result.report_data["transform_chain"]
        assert "DeltaMethodSetup" in chain
        assert "DeltaMethodCollapse" in chain
