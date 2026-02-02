"""Trace diff: compare two DerivationTrace objects and report differences.

Useful for comparing pipeline outputs across strategy branches or
between pipeline versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from mollifier_theta.pipelines.derivation_trace import DerivationTrace


@dataclass(frozen=True)
class TraceDiff:
    """Structured diff between two DerivationTrace objects."""

    theta_max_a: Fraction | None
    theta_max_b: Fraction | None
    binding_family_a: str
    binding_family_b: str
    added_case_ids: frozenset[str]
    removed_case_ids: frozenset[str]
    added_families: frozenset[str]
    removed_families: frozenset[str]
    case_count_changes: dict[str, tuple[int, int]]

    @property
    def is_empty(self) -> bool:
        """True if there are no differences between the two traces."""
        return (
            self.theta_max_a == self.theta_max_b
            and self.binding_family_a == self.binding_family_b
            and not self.added_case_ids
            and not self.removed_case_ids
            and not self.added_families
            and not self.removed_families
            and not self.case_count_changes
        )

    def format(self, label_a: str = "A", label_b: str = "B") -> str:
        """Human-readable diff summary."""
        lines: list[str] = []

        if self.theta_max_a != self.theta_max_b:
            lines.append(
                f"theta_max: {label_a}={self.theta_max_a} "
                f"vs {label_b}={self.theta_max_b}"
            )

        if self.binding_family_a != self.binding_family_b:
            lines.append(
                f"binding_family: {label_a}={self.binding_family_a!r} "
                f"vs {label_b}={self.binding_family_b!r}"
            )

        if self.added_families:
            lines.append(f"added families: {sorted(self.added_families)}")
        if self.removed_families:
            lines.append(f"removed families: {sorted(self.removed_families)}")

        if self.added_case_ids:
            lines.append(f"added cases: {sorted(self.added_case_ids)}")
        if self.removed_case_ids:
            lines.append(f"removed cases: {sorted(self.removed_case_ids)}")

        if self.case_count_changes:
            for case_id, (count_a, count_b) in sorted(
                self.case_count_changes.items()
            ):
                lines.append(
                    f"  {case_id}: {count_a} -> {count_b}"
                )

        if not lines:
            lines.append("(no differences)")

        return "\n".join(lines)


def diff_traces(
    trace_a: DerivationTrace,
    trace_b: DerivationTrace,
    *,
    theta_max_a: Fraction | None = None,
    theta_max_b: Fraction | None = None,
    binding_family_a: str = "",
    binding_family_b: str = "",
) -> TraceDiff:
    """Compare two DerivationTrace objects.

    Args:
        trace_a: First trace.
        trace_b: Second trace.
        theta_max_a: Optional theta_max for trace A (from find_theta_max).
        theta_max_b: Optional theta_max for trace B.
        binding_family_a: Binding family for trace A.
        binding_family_b: Binding family for trace B.

    Returns:
        TraceDiff with structural differences.
    """
    cases_a = trace_a.case_summary
    cases_b = trace_b.case_summary

    keys_a = set(cases_a.keys())
    keys_b = set(cases_b.keys())

    families_a = set(trace_a.families.keys())
    families_b = set(trace_b.families.keys())

    # Case count changes: cases present in both but with different counts
    case_count_changes: dict[str, tuple[int, int]] = {}
    for key in keys_a & keys_b:
        if cases_a[key] != cases_b[key]:
            case_count_changes[key] = (cases_a[key], cases_b[key])

    return TraceDiff(
        theta_max_a=theta_max_a,
        theta_max_b=theta_max_b,
        binding_family_a=binding_family_a,
        binding_family_b=binding_family_b,
        added_case_ids=frozenset(keys_b - keys_a),
        removed_case_ids=frozenset(keys_a - keys_b),
        added_families=frozenset(families_b - families_a),
        removed_families=frozenset(families_a - families_b),
        case_count_changes=case_count_changes,
    )
