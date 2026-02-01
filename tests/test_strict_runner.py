"""Tests for StrictPipelineRunner (WI-9).

Validates:
  - Strict mode passes on both pipelines end-to-end
  - Injecting a phase-dropping mock transform raises PipelineInvariantViolation
  - Non-strict mode (default) does not run per-stage checks
  - Phase dependency subset check catches violations
"""

from __future__ import annotations

import pytest

from mollifier_theta.core.invariants import PipelineInvariantViolation
from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
from mollifier_theta.pipelines.strict_runner import StrictPipelineRunner


class _PhaseDropper:
    """Mock transform that silently drops all phases."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results = []
        for term in terms:
            new_term = Term(
                kind=term.kind,
                expression=term.expression,
                variables=list(term.variables),
                ranges=list(term.ranges),
                kernels=list(term.kernels),
                phases=[],  # Drop all phases!
                history=list(term.history) + [
                    HistoryEntry(transform="PhaseDropper", parent_ids=[term.id])
                ],
                parents=[term.id],
                kernel_state=term.kernel_state,
                metadata=dict(term.metadata),
            )
            results.append(new_term)
            ledger.add(new_term)
        return results


class _BadDepTransform:
    """Mock transform that adds a phase with invalid depends_on."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results = []
        for term in terms:
            bad_phase = Phase(
                expression="e(ghost_var/c)",
                depends_on=["ghost_var", "c"],  # ghost_var not in variables
            )
            new_term = Term(
                kind=term.kind,
                expression=term.expression,
                variables=list(term.variables),
                ranges=list(term.ranges),
                kernels=list(term.kernels),
                phases=list(term.phases) + [bad_phase],
                history=list(term.history) + [
                    HistoryEntry(transform="BadDepTransform", parent_ids=[term.id])
                ],
                parents=[term.id],
                kernel_state=term.kernel_state,
                metadata=dict(term.metadata),
            )
            results.append(new_term)
            ledger.add(new_term)
        return results


class TestStrictBaselinePipeline:
    def test_strict_baseline_passes(self) -> None:
        """Baseline pipeline passes strict mode end-to-end."""
        result = conrey89_pipeline(theta_val=0.56, strict=True)
        assert result.theta_admissible is True
        assert result.theta_max is not None

    def test_non_strict_by_default(self) -> None:
        """Default (non-strict) does not raise on any pipeline."""
        result = conrey89_pipeline(theta_val=0.56)
        assert result is not None


class TestStrictVoronoiPipeline:
    def test_strict_voronoi_passes(self) -> None:
        """Voronoi pipeline passes strict mode end-to-end."""
        result = conrey89_voronoi_pipeline(theta_val=0.56, strict=True)
        assert result.theta_admissible is True


class TestPhaseDropDetection:
    def test_phase_dropper_raises(self) -> None:
        """A transform that drops phases should raise PipelineInvariantViolation."""
        ledger = TermLedger()
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression="test",
            variables=["m", "n"],
            phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
        )
        ledger.add(term)

        runner = StrictPipelineRunner(ledger)
        dropper = _PhaseDropper()

        with pytest.raises(PipelineInvariantViolation, match="Phases lost"):
            runner.run_stage(dropper, [term], "PhaseDropper")


class TestPhaseDepsSubsetDetection:
    def test_bad_deps_raises(self) -> None:
        """A transform that adds a phase with invalid depends_on should raise."""
        ledger = TermLedger()
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression="test",
            variables=["m", "n"],
        )
        ledger.add(term)

        runner = StrictPipelineRunner(ledger)
        bad = _BadDepTransform()

        with pytest.raises(PipelineInvariantViolation, match="depends_on"):
            runner.run_stage(bad, [term], "BadDepTransform")


class _FanOutTransform:
    """Splits each input term into two output terms."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        out = []
        for t in terms:
            for suffix in ("_a", "_b"):
                child = t.with_updates(
                    kind=t.kind,
                    expression=t.expression + suffix,
                    parents=[t.id],
                    history=list(t.history) + [
                        HistoryEntry(
                            transform="FanOut", parent_ids=[t.id],
                            description=f"split{suffix}",
                        )
                    ],
                )
                out.append(child)
                ledger.add(child)
        return out


class _OrphanTransform:
    """Creates output referencing a non-existent parent."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        return [
            t.with_updates(parents=["nonexistent_id_999"])
            for t in terms
        ]


class _IllegalStateJump:
    """Jumps kernel state NONE -> KLOOSTERMANIZED (skips intermediates)."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        return [
            t.with_updates(
                kernel_state=KernelState.KLOOSTERMANIZED,
                parents=[t.id],
            )
            for t in terms
        ]


class _LedgerMutatingDropper:
    """Adds a junk term to the ledger then drops phases (triggers violation)."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        junk = Term(id="junk_rollback", kind=TermKind.ERROR, status=TermStatus.ERROR)
        ledger.add(junk)
        return [
            t.with_updates(phases=[], parents=[t.id])
            for t in terms
        ]


class TestStageLog:
    def test_stage_log_populated(self) -> None:
        """Stage log should have entries after running stages."""
        ledger = TermLedger()
        term = Term(
            kind=TermKind.DIAGONAL,
            expression="test",
        )
        ledger.add(term)

        class _Identity:
            def apply(self, terms, ledger):
                return terms

        runner = StrictPipelineRunner(ledger)
        runner.run_stage(_Identity(), [term], "Identity")

        assert len(runner.stage_log) == 1
        assert runner.stage_log[0]["stage"] == "Identity"
        assert runner.stage_log[0]["violations"] == []


class TestFanOutLineage:
    def test_fan_out_passes_strict(self) -> None:
        """Fan-out (1 -> 2 terms) passes when lineage is correct."""
        ledger = TermLedger()
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression="sum_od",
            variables=["m", "n"],
            kernels=[Kernel(name="W")],
            phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
        )
        ledger.add(term)

        runner = StrictPipelineRunner(ledger)
        result = runner.run_stage(_FanOutTransform(), [term], "FanOut")

        assert len(result) == 2
        for child in result:
            assert term.id in child.parents

        log = runner.stage_log[0]
        assert log["input_count"] == 1
        assert log["output_count"] == 2
        assert log["violations"] == []

    def test_fan_out_multi_input(self) -> None:
        """Fan-out from 2 input terms produces 4 output terms."""
        ledger = TermLedger()
        t1 = Term(kind=TermKind.DIAGONAL, expression="diag", variables=["m"])
        t2 = Term(kind=TermKind.OFF_DIAGONAL, expression="od", variables=["m"])
        ledger.add(t1)
        ledger.add(t2)

        runner = StrictPipelineRunner(ledger)
        result = runner.run_stage(_FanOutTransform(), [t1, t2], "FanOut")
        assert len(result) == 4


class TestOrphanDetection:
    def test_orphan_parent_raises(self) -> None:
        """Output referencing a non-existent parent is caught."""
        ledger = TermLedger()
        term = Term(kind=TermKind.INTEGRAL, expression="test")
        ledger.add(term)

        runner = StrictPipelineRunner(ledger)
        with pytest.raises(PipelineInvariantViolation, match="not found"):
            runner.run_stage(_OrphanTransform(), [term], "orphan")


class TestIllegalStateTransition:
    def test_state_jump_raises(self) -> None:
        """NONE -> KLOOSTERMANIZED skips intermediates and is illegal."""
        ledger = TermLedger()
        term = Term(
            kind=TermKind.INTEGRAL,
            expression="test",
            kernel_state=KernelState.NONE,
        )
        ledger.add(term)

        runner = StrictPipelineRunner(ledger)
        with pytest.raises(PipelineInvariantViolation, match="Illegal kernel state"):
            runner.run_stage(_IllegalStateJump(), [term], "jump")


class TestRollbackTransaction:
    def test_rollback_on_violation(self) -> None:
        """When a violation occurs, the original ledger is untouched."""
        ledger = TermLedger()
        original = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression="test",
            variables=["m", "n"],
            phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
        )
        ledger.add(original)

        runner = StrictPipelineRunner(ledger)
        with pytest.raises(PipelineInvariantViolation):
            runner.run_stage(_LedgerMutatingDropper(), [original], "mutating")

        # Junk term should NOT be in the original ledger
        assert "junk_rollback" not in ledger
        # Original term still accessible
        assert original.id in ledger

    def test_commit_on_success(self) -> None:
        """On success, the trial ledger changes are committed."""
        ledger = TermLedger()
        term = Term(
            kind=TermKind.DIAGONAL,
            expression="test",
            variables=["m"],
            kernels=[Kernel(name="W")],
            phases=[Phase(expression="e(x)", depends_on=["m"], unit_modulus=True)],
        )
        ledger.add(term)

        runner = StrictPipelineRunner(ledger)
        result = runner.run_stage(_FanOutTransform(), [term], "fanout")

        # Fan-out children should now be in the committed ledger
        for child in result:
            assert child.id in ledger

    def test_violation_still_logs_stage(self) -> None:
        """Even on failure, the stage log records the violations."""
        runner = StrictPipelineRunner()
        term = Term(
            kind=TermKind.INTEGRAL,
            expression="test",
            variables=["m"],
            phases=[Phase(expression="e(x)", depends_on=["m"], unit_modulus=True)],
        )

        with pytest.raises(PipelineInvariantViolation):
            runner.run_stage(_PhaseDropper(), [term], "drop")

        assert len(runner.stage_log) == 1
        assert runner.stage_log[0]["violations"]  # non-empty


class TestLineageLedgerFallback:
    def test_parent_found_in_ledger(self) -> None:
        """Parent lookup falls back to ledger when not in stage inputs."""
        ledger = TermLedger()
        grandparent = Term(
            kind=TermKind.INTEGRAL,
            expression="gp",
            kernel_state=KernelState.NONE,
        )
        ledger.add(grandparent)

        parent = grandparent.with_updates(
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            parents=[grandparent.id],
            metadata={"delta_method_applied": True},
        )
        ledger.add(parent)

        runner = StrictPipelineRunner(ledger)

        # Child references grandparent (in ledger, not in stage inputs)
        # NONE -> UNCOLLAPSED_DELTA is legal
        child = parent.with_updates(
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            parents=[grandparent.id],
            metadata={"delta_method_applied": True},
        )

        class _LedgerParentTransform:
            def apply(self, terms, ledger):
                ledger.add(child)
                return [child]

        result = runner.run_stage(_LedgerParentTransform(), [parent], "ledger_lookup")
        assert len(result) == 1


class TestMultiStagePipeline:
    def test_sequential_stages_both_validated(self) -> None:
        """Two stages run in sequence, each independently validated."""
        runner = StrictPipelineRunner()
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression="test",
            variables=["m", "n"],
            kernels=[Kernel(name="W")],
            phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
        )

        class _Identity:
            def apply(self, terms, ledger):
                return list(terms)

        result1 = runner.run_stage(_Identity(), [term], "stage1")
        result2 = runner.run_stage(_FanOutTransform(), result1, "stage2")

        assert len(result2) == 2
        assert len(runner.stage_log) == 2
        assert all(log["violations"] == [] for log in runner.stage_log)

    def test_failure_mid_pipeline_preserves_earlier(self) -> None:
        """Violation in stage 2 doesn't affect stage 1's state."""
        ledger = TermLedger()
        runner = StrictPipelineRunner(ledger)
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression="test",
            variables=["m", "n"],
            kernels=[Kernel(name="W")],
            phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
        )

        class _Identity:
            def apply(self, terms, ledger):
                return list(terms)

        result1 = runner.run_stage(_Identity(), [term], "stage1")
        with pytest.raises(PipelineInvariantViolation):
            runner.run_stage(_PhaseDropper(), result1, "stage2")

        assert len(runner.stage_log) == 2
        assert runner.stage_log[0]["violations"] == []
        assert runner.stage_log[1]["violations"] != []
