"""DerivationTrace: human-parseable trace of the transform stack and bound strategies.

Reconstructs the derivation path for any term in the ledger, showing
which transforms were applied, what metadata was attached, and which
bound strategy (with case ID) produced BoundOnly terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.core.stage_meta import (
    get_bound_meta,
    get_kuznetsov_meta,
    get_voronoi_meta,
)


@dataclass(frozen=True)
class TraceStep:
    """One step in a term's derivation path."""

    stage_name: str
    parent_ids: list[str]
    description: str
    kernel_state: str
    metadata_keys: list[str]


@dataclass(frozen=True)
class TermTrace:
    """Full derivation trace for a single term."""

    term_id: str
    kind: str
    status: str
    kernel_state: str
    steps: list[TraceStep]
    bound_family: str = ""
    case_id: str = ""
    voronoi_kind: str = ""
    kuznetsov_applied: bool = False
    lemma_citation: str = ""

    def format(self, indent: int = 2) -> str:
        """Format as human-readable multi-line string."""
        pad = " " * indent
        lines = [
            f"Term {self.term_id}:",
            f"{pad}kind: {self.kind}",
            f"{pad}status: {self.status}",
            f"{pad}kernel_state: {self.kernel_state}",
        ]
        if self.voronoi_kind:
            lines.append(f"{pad}voronoi: {self.voronoi_kind}")
        if self.kuznetsov_applied:
            lines.append(f"{pad}kuznetsov: applied")
        if self.bound_family:
            lines.append(f"{pad}bound_family: {self.bound_family}")
        if self.case_id:
            lines.append(f"{pad}case_id: {self.case_id}")
        if self.lemma_citation:
            lines.append(f"{pad}citation: {self.lemma_citation}")
        lines.append(f"{pad}derivation ({len(self.steps)} steps):")
        for i, step in enumerate(self.steps):
            lines.append(f"{pad}  [{i}] {step.stage_name}")
            if step.description:
                lines.append(f"{pad}      {step.description}")
            if step.parent_ids:
                lines.append(f"{pad}      parents: {step.parent_ids}")
            lines.append(f"{pad}      state: {step.kernel_state}")
        return "\n".join(lines)


@dataclass
class DerivationTrace:
    """Collection of traces for pipeline terms, with summary statistics.

    Build via `DerivationTrace.from_terms()` or by appending to `traces`.
    """

    traces: list[TermTrace] = field(default_factory=list)
    stage_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_terms(
        cls,
        terms: list[Term],
        stage_log: list[dict[str, Any]] | None = None,
    ) -> DerivationTrace:
        """Build traces for all provided terms."""
        traces = [_trace_term(t) for t in terms]
        return cls(traces=traces, stage_log=stage_log or [])

    @property
    def bound_traces(self) -> list[TermTrace]:
        """Traces for BoundOnly terms only."""
        return [t for t in self.traces if t.status == TermStatus.BOUND_ONLY.value]

    @property
    def families(self) -> dict[str, list[TermTrace]]:
        """Group bound traces by bound_family."""
        result: dict[str, list[TermTrace]] = {}
        for t in self.bound_traces:
            family = t.bound_family or "unknown"
            result.setdefault(family, []).append(t)
        return result

    @property
    def case_summary(self) -> dict[str, int]:
        """Count of bound terms by family:case_id."""
        counts: dict[str, int] = {}
        for t in self.bound_traces:
            key = f"{t.bound_family}:{t.case_id}" if t.case_id else t.bound_family
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def format_summary(self) -> str:
        """Human-readable summary of the derivation trace."""
        lines = [
            f"DerivationTrace: {len(self.traces)} terms traced",
            f"  BoundOnly: {len(self.bound_traces)}",
        ]
        if self.stage_log:
            lines.append(f"  Stages: {len(self.stage_log)}")
            for entry in self.stage_log:
                v_str = "" if not entry.get("violations") else " [VIOLATIONS]"
                lines.append(
                    f"    {entry['stage']}: "
                    f"{entry['input_count']}→{entry['output_count']}{v_str}"
                )

        families = self.families
        if families:
            lines.append("  Bound families:")
            for family, traces in sorted(families.items()):
                case_ids = sorted(set(t.case_id for t in traces if t.case_id))
                if case_ids:
                    lines.append(
                        f"    {family}: {len(traces)} terms, "
                        f"cases={case_ids}"
                    )
                else:
                    lines.append(f"    {family}: {len(traces)} terms")

        return "\n".join(lines)

    def format_full(self) -> str:
        """Full trace of all bound terms."""
        parts = [self.format_summary(), ""]
        for trace in self.bound_traces:
            parts.append(trace.format())
            parts.append("")
        return "\n".join(parts)


def _trace_term(term: Term) -> TermTrace:
    """Build a TermTrace from a single term's history."""
    steps: list[TraceStep] = []
    for h in term.history:
        steps.append(
            TraceStep(
                stage_name=h.transform,
                parent_ids=list(h.parent_ids),
                description=h.description,
                kernel_state=term.kernel_state.value,
                metadata_keys=sorted(term.metadata.keys()) if term.metadata else [],
            )
        )

    bm = get_bound_meta(term)
    vm = get_voronoi_meta(term)
    km = get_kuznetsov_meta(term)

    return TermTrace(
        term_id=term.id,
        kind=term.kind.value,
        status=term.status.value,
        kernel_state=term.kernel_state.value,
        steps=steps,
        bound_family=bm.bound_family if bm else "",
        case_id=bm.case_id if bm else "",
        voronoi_kind=vm.kind.value if vm else "",
        kuznetsov_applied=km.applied if km else False,
        lemma_citation=term.lemma_citation or "",
    )
