"""Per-term theta breakdown: binding constraints, slack, and overhead.

Given a validated MathParamsEnvelope and a theta value, produces a
deterministic per-term breakdown showing which constraint binds,
the error exponent at theta, slack, raw DI comparison, and the
overall theta_max with binding constraint identification.

Output is a versioned ThetaBreakdownEnvelope (sorted, deterministic).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from mollifier_theta.analysis.exponent_model import ExponentConstraint
from mollifier_theta.analysis.length_aware_di import LengthAwareDIModel
from mollifier_theta.core.scale_model import ScaleModel
from mollifier_theta.reports.envelope_loader import MathParamsEnvelope
from mollifier_theta.reports.math_parameter_export import MathParameterRecord

FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class TermBreakdown:
    """Per-term breakdown at a specific theta value."""

    term_id: str
    bound_family: str
    error_exponent: str
    E_val: float
    slack: float
    is_binding: bool
    raw_di_E_val: float
    overhead: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ThetaBreakdown:
    """Full theta breakdown result."""

    theta_val: float
    theta_max: float
    binding_term_id: str
    binding_family: str
    terms: tuple[TermBreakdown, ...]

    def to_envelope(self) -> dict:
        """Versioned JSON envelope with records sorted by slack ascending."""
        sorted_terms = sorted(self.terms, key=lambda t: (t.slack, t.term_id))
        return {
            "format_version": FORMAT_VERSION,
            "theta_val": self.theta_val,
            "theta_max": self.theta_max,
            "binding_term_id": self.binding_term_id,
            "binding_family": self.binding_family,
            "record_count": len(sorted_terms),
            "records": [t.to_dict() for t in sorted_terms],
        }

    def to_json(self) -> str:
        """Canonical JSON string (sorted keys, indent=2)."""
        return json.dumps(self.to_envelope(), sort_keys=True, indent=2)


def _select_raw_model(record: MathParameterRecord) -> LengthAwareDIModel:
    """Select the appropriate raw DI model for a math-param record."""
    if "voronoi" in record.bound_family.lower():
        return LengthAwareDIModel.voronoi_dual()
    return LengthAwareDIModel.symmetric()


def _solve_theta_max_for_record(record: MathParameterRecord) -> float:
    """Solve E(theta) = 1 for a single record's error exponent."""
    if not record.error_exponent:
        return 1.0
    try:
        return ScaleModel.solve_expr_equals_one(record.error_exponent)
    except ValueError:
        return 1.0


def compute_theta_breakdown(
    envelope: MathParamsEnvelope,
    theta_val: float,
) -> ThetaBreakdown:
    """Compute per-term theta breakdown from a validated MathParamsEnvelope.

    Args:
        envelope: Validated v1.0 math-parameters envelope.
        theta_val: Theta value for evaluation.

    Returns:
        ThetaBreakdown with per-term analysis and overall theta_max.
    """
    breakdowns: list[TermBreakdown] = []
    theta_maxes: list[tuple[float, MathParameterRecord]] = []

    for record in envelope.records:
        # Evaluate pipeline error exponent
        if not record.error_exponent:
            continue
        try:
            E_val = ScaleModel.evaluate_expr(record.error_exponent, theta_val)
        except (ValueError, TypeError):
            continue

        slack = 1.0 - E_val

        # Raw DI model comparison
        raw_model = _select_raw_model(record)
        raw_di_E_val = raw_model.evaluate_error(theta_val)
        overhead = E_val - raw_di_E_val

        # Solve theta_max for this record
        tm = _solve_theta_max_for_record(record)
        theta_maxes.append((tm, record))

        breakdowns.append(
            TermBreakdown(
                term_id=record.term_id,
                bound_family=record.bound_family,
                error_exponent=record.error_exponent,
                E_val=E_val,
                slack=slack,
                is_binding=False,  # set below
                raw_di_E_val=raw_di_E_val,
                overhead=overhead,
            )
        )

    # Identify binding term (smallest theta_max = tightest constraint)
    if theta_maxes:
        theta_maxes.sort(key=lambda x: (x[0], x[1].term_id))
        overall_theta_max = theta_maxes[0][0]
        binding_record = theta_maxes[0][1]
        binding_term_id = binding_record.term_id
        binding_family = binding_record.bound_family
    else:
        overall_theta_max = 1.0
        binding_term_id = ""
        binding_family = ""

    # Mark binding term(s)
    final_breakdowns = tuple(
        TermBreakdown(
            term_id=tb.term_id,
            bound_family=tb.bound_family,
            error_exponent=tb.error_exponent,
            E_val=tb.E_val,
            slack=tb.slack,
            is_binding=(tb.term_id == binding_term_id),
            raw_di_E_val=tb.raw_di_E_val,
            overhead=tb.overhead,
        )
        for tb in breakdowns
    )

    return ThetaBreakdown(
        theta_val=theta_val,
        theta_max=overall_theta_max,
        binding_term_id=binding_term_id,
        binding_family=binding_family,
        terms=final_breakdowns,
    )
