"""Invariant checks enforced across the pipeline.

These are the "DO NOT SIMPLIFY" guards:
- No premature bounding without citation
- Phases never silently dropped
- Kernels survive transforms
"""

from __future__ import annotations

from mollifier_theta.core.ir import KERNEL_STATE_TRANSITIONS, KernelState, Term, TermStatus


def check_no_premature_bound(term: Term) -> list[str]:
    """BoundOnly terms must have a lemma_citation."""
    violations: list[str] = []
    if term.status == TermStatus.BOUND_ONLY and not term.lemma_citation:
        violations.append(
            f"Term {term.id}: BoundOnly without lemma_citation"
        )
    return violations


def check_phases_tracked(input_terms: list[Term], output_terms: list[Term]) -> list[str]:
    """All input phases must be accounted for in outputs (absorbed or retained)."""
    violations: list[str] = []
    input_phase_exprs = set()
    for t in input_terms:
        for p in t.phases:
            input_phase_exprs.add(p.expression)

    output_phase_exprs = set()
    for t in output_terms:
        for p in t.phases:
            output_phase_exprs.add(p.expression)

    missing = input_phase_exprs - output_phase_exprs
    if missing:
        violations.append(
            f"Phases lost in transform: {missing}"
        )
    return violations


def check_kernel_preservation(
    input_terms: list[Term],
    output_terms: list[Term],
    allow_removal: bool = False,
) -> list[str]:
    """Kernels from input terms must appear in output terms unless explicitly allowed."""
    violations: list[str] = []
    if allow_removal:
        return violations

    input_kernel_names = set()
    for t in input_terms:
        for k in t.kernels:
            input_kernel_names.add(k.name)

    output_kernel_names = set()
    for t in output_terms:
        for k in t.kernels:
            output_kernel_names.add(k.name)

    missing = input_kernel_names - output_kernel_names
    if missing:
        violations.append(
            f"Kernels lost in transform: {missing}"
        )
    return violations


def check_kernel_state_transition(
    from_state: KernelState,
    to_state: KernelState,
) -> list[str]:
    """Verify a kernel state transition is legal."""
    violations: list[str] = []
    allowed = KERNEL_STATE_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        violations.append(
            f"Illegal kernel state transition: {from_state.value} -> {to_state.value}. "
            f"Allowed from {from_state.value}: {[s.value for s in allowed]}"
        )
    return violations


def check_kernel_state_consistency(term: Term) -> list[str]:
    """Verify that a term's kernel_state is consistent with its metadata."""
    violations: list[str] = []
    ks = term.kernel_state

    if ks == KernelState.UNCOLLAPSED_DELTA:
        if not term.metadata.get("delta_method_applied"):
            violations.append(
                f"Term {term.id}: kernel_state=UNCOLLAPSED_DELTA but "
                f"delta_method_applied not set"
            )
        if term.metadata.get("delta_method_collapsed"):
            violations.append(
                f"Term {term.id}: kernel_state=UNCOLLAPSED_DELTA but "
                f"delta_method_collapsed=True"
            )

    if ks == KernelState.COLLAPSED:
        if term.metadata.get("delta_method_applied") and not term.metadata.get(
            "delta_method_collapsed", False
        ):
            violations.append(
                f"Term {term.id}: kernel_state=COLLAPSED but "
                f"delta_method_collapsed=False"
            )

    return violations


class PipelineInvariantViolation(Exception):
    """Raised by StrictPipelineRunner when a per-stage invariant fails."""

    def __init__(self, stage: str, violations: list[str]) -> None:
        self.stage = stage
        self.violations = violations
        msg = f"Invariant violation after {stage}:\n" + "\n".join(violations)
        super().__init__(msg)


def check_phase_deps_subset(terms: list[Term]) -> list[str]:
    """Check that all phase depends_on entries are subsets of term.variables."""
    violations: list[str] = []
    for term in terms:
        var_set = set(term.variables)
        for p in term.phases:
            if not set(p.depends_on) <= var_set:
                extra = set(p.depends_on) - var_set
                violations.append(
                    f"Term {term.id}: phase '{p.expression}' depends_on "
                    f"includes {extra} not in variables {list(term.variables)}"
                )
    return violations


def check_phases_tracked_with_context(
    input_terms: list[Term],
    output_terms: list[Term],
    stage: str = "",
) -> list[str]:
    """Enhanced phase tracking that understands absorption and Kloosterman consumption.

    Phases are accounted for if:
    - They appear in output (possibly with absorbed=True)
    - They were recorded as consumed in _kloosterman metadata (authoritative)
    - They were consumed by Fourier integration (t-dependent phases)
    - They appear absorbed in output
    """
    violations: list[str] = []
    input_phase_exprs = set()
    for t in input_terms:
        for p in t.phases:
            input_phase_exprs.add(p.expression)

    output_phase_exprs = set()
    for t in output_terms:
        for p in t.phases:
            output_phase_exprs.add(p.expression)

    missing = input_phase_exprs - output_phase_exprs

    if not missing:
        return violations

    # Check authoritative metadata for consumed phases (from KloostermanForm)
    kloosterman_consumed: set[str] = set()
    for t in output_terms:
        km = t.metadata.get("_kloosterman")
        if km and isinstance(km, dict):
            for expr in km.get("consumed_phases", []):
                kloosterman_consumed.add(expr)
    # Also check heuristic fallback for backward compatibility:
    # If S(m,n;c)/c is in output and no _kloosterman metadata exists,
    # treat e(...) phases as consumed.
    if not kloosterman_consumed and "S(m,n;c)/c" in output_phase_exprs:
        for expr in missing:
            if expr.startswith("e(") and "/" in expr and expr.endswith(")"):
                kloosterman_consumed.add(expr)

    # Check if Fourier integration consumed t-dependent phases
    output_kernel_names = set()
    for t in output_terms:
        for k in t.kernels:
            output_kernel_names.add(k.name)
    fourier_consumed: set[str] = set()
    if "FourierKernel" in output_kernel_names:
        for expr in missing:
            for t_in in input_terms:
                for p in t_in.phases:
                    if p.expression == expr and "t" in p.depends_on:
                        fourier_consumed.add(expr)

    # Check if absorption accounts for missing phases
    absorbed_exprs: set[str] = set()
    for t in output_terms:
        for p in t.phases:
            if p.absorbed:
                absorbed_exprs.add(p.expression)

    remaining = missing - kloosterman_consumed - fourier_consumed - absorbed_exprs
    if remaining:
        violations.append(
            f"Phases lost in {stage or 'transform'}: {remaining}"
        )
    return violations


def validate_term(term: Term) -> list[str]:
    """Run all single-term invariant checks."""
    violations: list[str] = []
    violations.extend(check_no_premature_bound(term))
    violations.extend(check_kernel_state_consistency(term))
    return violations


def validate_all(terms: list[Term]) -> list[str]:
    """Run all invariant checks on a list of terms."""
    violations: list[str] = []
    for term in terms:
        violations.extend(validate_term(term))
    return violations
