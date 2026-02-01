"""StrictPipelineRunner: validates invariants after every transform stage.

Uses transactional semantics (clone-on-write) so invariant failures
never leave invalid terms in the ledger.

Kernel state transition checks are lineage-based (via term.parents),
not positional, so fan-out transforms are fully covered.

Opt-in via strict=True parameter. CI should run strict.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mollifier_theta.core.invariants import (
    PipelineInvariantViolation,
    check_kernel_preservation,
    check_kernel_state_transition,
    check_phase_deps_subset,
    check_phases_tracked_with_context,
    validate_all,
)
from mollifier_theta.core.ir import KernelState, Term
from mollifier_theta.core.ledger import TermLedger


@runtime_checkable
class Transform(Protocol):
    """Protocol for transform objects."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]: ...


class StrictPipelineRunner:
    """Runs a sequence of transforms with per-stage invariant validation.

    After each transform stage, checks:
    - validate_all() on all output terms
    - check_phases_tracked_with_context() between input/output
    - check_kernel_preservation() between input/output
    - lineage-based kernel state transitions (via term.parents)
    - check_phase_deps_subset() on output terms

    Uses copy-on-write: the transform runs against a cloned ledger.
    If invariants fail, the original ledger is untouched (rollback).
    """

    def __init__(self, ledger: TermLedger | None = None) -> None:
        self.ledger = ledger or TermLedger()
        self._stage_log: list[dict[str, Any]] = []

    def run_stage(
        self,
        transform: Transform,
        terms: list[Term],
        stage_name: str = "",
        allow_kernel_removal: bool = False,
        _allow_phase_drop: bool = False,
    ) -> list[Term]:
        """Run a single transform stage with full invariant checking.

        Uses copy-on-write: if invariants fail, the ledger is unchanged.
        Raises PipelineInvariantViolation if any check fails.

        When _allow_phase_drop=True (used by bounding stages), phase tracking
        is relaxed since BoundOnly terms legitimately simplify structure.
        """
        name = stage_name or type(transform).__name__
        input_terms = list(terms)

        # Build input lookup for lineage-based state checks
        input_by_id: dict[str, Term] = {t.id: t for t in input_terms}

        # Copy-on-write: run against a clone
        trial_ledger = self.ledger.clone()
        output_terms = transform.apply(terms, trial_ledger)

        # Collect violations
        violations: list[str] = []

        # 1. Per-term invariants
        violations.extend(validate_all(output_terms))

        # 2. Phase tracking (with Kloosterman/absorption awareness)
        #    Skipped for bounding stages where phase simplification is expected
        if not _allow_phase_drop:
            violations.extend(
                check_phases_tracked_with_context(input_terms, output_terms, name)
            )

        # 3. Kernel preservation
        violations.extend(
            check_kernel_preservation(
                input_terms, output_terms, allow_removal=allow_kernel_removal,
            )
        )

        # 4. Lineage-based kernel state transitions
        for out in output_terms:
            for parent_id in out.parents:
                # Look up parent: try stage inputs first, then ledger
                parent = input_by_id.get(parent_id)
                if parent is None:
                    try:
                        parent = trial_ledger.get(parent_id)
                    except KeyError:
                        violations.append(
                            f"Term {out.id}: parent '{parent_id}' not found "
                            f"in stage inputs or ledger"
                        )
                        continue

                if parent.kernel_state != out.kernel_state:
                    violations.extend(
                        check_kernel_state_transition(
                            parent.kernel_state, out.kernel_state,
                        )
                    )

        # 5. Phase dependency subset
        violations.extend(check_phase_deps_subset(output_terms))

        # Log the stage
        self._stage_log.append({
            "stage": name,
            "input_count": len(input_terms),
            "output_count": len(output_terms),
            "violations": violations,
        })

        if violations:
            # Rollback: trial_ledger is discarded, self.ledger is untouched
            raise PipelineInvariantViolation(name, violations)

        # Commit: adopt the trial ledger
        self.ledger._terms = trial_ledger._terms
        self.ledger._pruned_ids = trial_ledger._pruned_ids
        return output_terms

    def run_bounding_stage(
        self,
        bound_fn: Any,
        terms: list[Term],
        stage_name: str = "",
    ) -> list[Term]:
        """Run a bounding step through the strict runner.

        Wraps a bound object (with .applies() and .bound() methods) into
        the transform protocol.  Bounding stages are validated with relaxed
        checks: phase dropping and kernel removal are allowed because
        BoundOnly terms legitimately simplify the symbolic structure.

        Per-term invariants (validate_all) and phase-dep checks still run.
        Kernel state transitions are still checked via lineage.
        """

        class _BoundAdapter:
            """Adapts .applies()/.bound() to the Transform protocol."""

            def __init__(self, bound_obj: Any) -> None:
                self._bound = bound_obj

            def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
                out: list[Term] = []
                for t in terms:
                    if self._bound.applies(t):
                        bounded = self._bound.bound(t)
                        ledger.add(bounded)
                        out.append(bounded)
                    else:
                        out.append(t)
                return out

        adapter = _BoundAdapter(bound_fn)
        name = stage_name or type(bound_fn).__name__

        # Use run_stage with relaxed checks for bounding
        return self.run_stage(
            adapter,
            terms,
            stage_name=name,
            allow_kernel_removal=True,
            _allow_phase_drop=True,
        )

    @property
    def stage_log(self) -> list[dict[str, Any]]:
        """Log of all stages run and their validation results."""
        return list(self._stage_log)
