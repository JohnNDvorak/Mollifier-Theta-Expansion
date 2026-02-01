"""Invariant checks enforced across the pipeline.

These are the "DO NOT SIMPLIFY" guards:
- No premature bounding without citation
- Phases never silently dropped
- Kernels survive transforms
"""

from __future__ import annotations

from mollifier_theta.core.ir import Term, TermStatus


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


def validate_term(term: Term) -> list[str]:
    """Run all single-term invariant checks."""
    violations: list[str] = []
    violations.extend(check_no_premature_bound(term))
    return violations


def validate_all(terms: list[Term]) -> list[str]:
    """Run all invariant checks on a list of terms."""
    violations: list[str] = []
    for term in terms:
        violations.extend(validate_term(term))
    return violations
