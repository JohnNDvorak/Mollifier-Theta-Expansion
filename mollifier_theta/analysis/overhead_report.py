"""Theta-loss decomposition report.

For each binding/near-binding BoundOnly term, computes:
  - raw inequality exponent (from LengthAwareDI raw formula)
  - pipeline-implied exponent (current error_exponent used in proof)
  - overhead := pipeline - raw

If the overhead is systematic, it identifies the *real target* for
improving theta: remove or reduce that overhead by altering the
transform sequence (or delaying simplifications further).

Output: a table sorted by overhead magnitude, with derivation paths.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from mollifier_theta.analysis.length_aware_di import LengthAwareDIModel
from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.core.scale_model import ScaleModel
from mollifier_theta.core.stage_meta import get_bound_meta, get_voronoi_meta


@dataclass(frozen=True)
class OverheadRecord:
    """Overhead analysis for a single BoundOnly term."""

    term_id: str
    bound_family: str
    pipeline_exponent: str
    pipeline_E_val: float
    raw_di_exponent: str
    raw_di_E_val: float
    overhead: float  # pipeline_E_val - raw_di_E_val
    di_model_label: str
    derivation_path: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OverheadReport:
    """Theta-loss decomposition for a set of terms."""

    theta_val: float
    records: list[OverheadRecord]

    @property
    def bottleneck(self) -> OverheadRecord | None:
        """Record with the highest pipeline exponent (closest to binding)."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.pipeline_E_val)

    @property
    def max_overhead(self) -> OverheadRecord | None:
        """Record with the largest overhead (pipeline - raw)."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.overhead)

    def format_table(self) -> str:
        """Human-readable table sorted by overhead magnitude."""
        if not self.records:
            return "No overhead records."

        lines = [
            f"Theta-loss decomposition at theta={self.theta_val}",
            "",
            f"{'Family':<25} {'Pipeline E':<12} {'Raw DI E':<12} "
            f"{'Overhead':<12} {'Model':<15} {'Path'}",
            "-" * 100,
        ]

        for r in sorted(self.records, key=lambda x: x.overhead, reverse=True):
            path_str = " → ".join(r.derivation_path[-3:]) if r.derivation_path else ""
            lines.append(
                f"{r.bound_family:<25} "
                f"{r.pipeline_E_val:<12.6f} "
                f"{r.raw_di_E_val:<12.6f} "
                f"{r.overhead:<12.6f} "
                f"{r.di_model_label:<15} "
                f"{path_str}"
            )

        if self.max_overhead:
            lines.append("")
            lines.append(
                f"Largest overhead: {self.max_overhead.bound_family} "
                f"(+{self.max_overhead.overhead:.6f})"
            )

        return "\n".join(lines)

    def to_json(self) -> list[dict]:
        return [r.to_dict() for r in self.records]


def _get_derivation_path(term: Term) -> list[str]:
    """Extract the transform chain from term history."""
    return [h.transform for h in term.history]


def _select_raw_model(term: Term) -> LengthAwareDIModel:
    """Select the appropriate raw DI model for this term."""
    vm = get_voronoi_meta(term)
    if (
        vm is not None
        and vm.applied
        and vm.dual_length
    ):
        return LengthAwareDIModel.voronoi_dual()
    return LengthAwareDIModel.symmetric()


def compute_overhead(
    terms: list[Term],
    theta_val: float,
) -> OverheadReport:
    """Compute pipeline vs raw DI overhead for all BoundOnly terms.

    Args:
        terms: All terms (filters to BoundOnly internally).
        theta_val: Theta value for evaluation.

    Returns:
        OverheadReport sorted by overhead magnitude.
    """
    records: list[OverheadRecord] = []

    for term in terms:
        if term.status != TermStatus.BOUND_ONLY:
            continue

        bm = get_bound_meta(term)
        bound_family = bm.bound_family if bm else ""
        pipeline_exponent = bm.error_exponent if bm else term.metadata.get(
            "error_exponent", ""
        )

        if not pipeline_exponent:
            continue

        # Evaluate pipeline exponent
        try:
            pipeline_E_val = ScaleModel.evaluate_expr(pipeline_exponent, theta_val)
        except (ValueError, TypeError):
            continue

        # Select and evaluate raw DI model
        raw_model = _select_raw_model(term)
        raw_di_E_val = raw_model.evaluate_error(theta_val)
        raw_di_exponent = raw_model.error_exponent_str

        records.append(
            OverheadRecord(
                term_id=term.id,
                bound_family=bound_family,
                pipeline_exponent=pipeline_exponent,
                pipeline_E_val=pipeline_E_val,
                raw_di_exponent=raw_di_exponent,
                raw_di_E_val=raw_di_E_val,
                overhead=pipeline_E_val - raw_di_E_val,
                di_model_label=raw_model.label,
                derivation_path=_get_derivation_path(term),
            )
        )

    return OverheadReport(theta_val=theta_val, records=records)
