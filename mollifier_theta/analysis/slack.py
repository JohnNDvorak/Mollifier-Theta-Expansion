"""Slack computation for BoundOnly terms.

Identifies which term is the binding constraint on theta_max by measuring
how close each BoundOnly term's T-exponent is to 1.

Supports model-aware grouping by lemma/bound family and pipeline stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.core.scale_model import ScaleModel
from mollifier_theta.lemmas.di_kloosterman import DIExponentModel
from mollifier_theta.lemmas.theta_constraints import find_theta_max
from mollifier_theta.pipelines.conrey89 import PipelineResult, conrey89_pipeline


@dataclass(frozen=True)
class TermSlack:
    """Slack measurement for a single BoundOnly term."""

    term_id: str
    expression: str
    error_exponent: str
    E_val: float
    slack: float
    sub_exponents: dict[str, float] = field(default_factory=dict)
    lemma_citation: str = ""
    bound_family: str = ""
    pipeline_stage: str = ""


@dataclass(frozen=True)
class DiagnoseResult:
    """Result of the slack diagnosis pipeline."""

    theta_val: float
    theta_max: float
    headroom: float
    term_slacks: list[TermSlack]
    bottleneck: TermSlack | None
    pipeline_variant: str = "conrey89"

    def group_by_family(self) -> dict[str, list[TermSlack]]:
        """Group term slacks by bound family."""
        groups: dict[str, list[TermSlack]] = {}
        for ts in self.term_slacks:
            key = ts.bound_family or "unknown"
            groups.setdefault(key, []).append(ts)
        return groups

    def group_by_stage(self) -> dict[str, list[TermSlack]]:
        """Group term slacks by pipeline stage."""
        groups: dict[str, list[TermSlack]] = {}
        for ts in self.term_slacks:
            key = ts.pipeline_stage or "unknown"
            groups.setdefault(key, []).append(ts)
        return groups


def _classify_bound_family(term: Term) -> str:
    """Determine the bound family from term metadata."""
    if term.metadata.get("di_bound_applied"):
        return "DI_Kloosterman"
    if term.metadata.get("post_voronoi_bound"):
        return "PostVoronoi"
    if term.metadata.get("weil_bound"):
        return "Weil"
    if term.metadata.get("trivial_bound"):
        return "Trivial"
    return "unknown"


def _classify_pipeline_stage(term: Term) -> str:
    """Determine the pipeline stage from term history."""
    if not term.history:
        return "unknown"
    last_transform = term.history[-1].transform
    return last_transform


def _compute_slack_for_term(
    term: Term, theta_val: float,
) -> TermSlack:
    """Compute slack = 1.0 - E(theta_val) for a single BoundOnly term."""
    scale_dict = term.metadata.get("scale_model_dict")
    if scale_dict:
        sm = ScaleModel.from_dict(scale_dict)
        E_val = sm.evaluate(theta_val)
        sub_exp_vals = sm.evaluate_sub_exponents(theta_val)
        error_exponent = str(sm.T_exponent)
    elif term.metadata.get("error_exponent"):
        error_exponent = term.metadata["error_exponent"]
        E_val = ScaleModel.evaluate_expr(error_exponent, theta_val)
        sub_exp_vals = {}
    else:
        E_val = 0.0
        sub_exp_vals = {}
        error_exponent = "unknown"

    return TermSlack(
        term_id=term.id,
        expression=term.expression,
        error_exponent=error_exponent,
        E_val=E_val,
        slack=1.0 - E_val,
        sub_exponents=sub_exp_vals,
        lemma_citation=term.lemma_citation,
        bound_family=_classify_bound_family(term),
        pipeline_stage=_classify_pipeline_stage(term),
    )


def diagnose_pipeline(
    theta_val: float = 0.56,
    K: int = 3,
    result: PipelineResult | None = None,
    pipeline_variant: str = "conrey89",
) -> DiagnoseResult:
    """Run the pipeline (or accept a pre-computed result) and compute slack for all BoundOnly terms.

    Returns a DiagnoseResult with terms sorted ascending by slack
    (the bottleneck — closest to violating — comes first).
    """
    if result is None:
        result = conrey89_pipeline(theta_val=theta_val, K=K)

    theta_max = result.theta_max if result.theta_max is not None else 4 / 7

    all_terms = result.ledger.all_terms()
    bound_only = [t for t in all_terms if t.status == TermStatus.BOUND_ONLY]

    slacks = [_compute_slack_for_term(t, theta_val) for t in bound_only]
    # Deterministic tie-break: (slack, error_exponent, bound_family, pipeline_stage, term_id)
    # bound_family and pipeline_stage are content-derived (stable across runs),
    # term_id is the final fallback (random UUIDs, so run-specific)
    slacks.sort(key=lambda s: (s.slack, s.error_exponent, s.bound_family, s.pipeline_stage, s.term_id))

    bottleneck = slacks[0] if slacks else None
    headroom = theta_max - theta_val

    return DiagnoseResult(
        theta_val=theta_val,
        theta_max=theta_max,
        headroom=headroom,
        term_slacks=slacks,
        bottleneck=bottleneck,
        pipeline_variant=pipeline_variant,
    )


def compare_pipelines(
    theta_val: float = 0.56,
    K: int = 3,
) -> dict:
    """Compare slack diagnostics between conrey89 and conrey89_voronoi pipelines.

    Returns a summary showing which pipeline has what binding constraint.
    """
    from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline

    baseline = conrey89_pipeline(theta_val=theta_val, K=K)
    voronoi = conrey89_voronoi_pipeline(theta_val=theta_val, K=K)

    diag_baseline = diagnose_pipeline(
        theta_val=theta_val, result=baseline, pipeline_variant="conrey89",
    )
    diag_voronoi = diagnose_pipeline(
        theta_val=theta_val, result=voronoi, pipeline_variant="conrey89_voronoi",
    )

    return {
        "theta_val": theta_val,
        "baseline": {
            "pipeline": "conrey89",
            "theta_max": diag_baseline.theta_max,
            "headroom": diag_baseline.headroom,
            "bottleneck": diag_baseline.bottleneck.error_exponent if diag_baseline.bottleneck else None,
            "bottleneck_family": diag_baseline.bottleneck.bound_family if diag_baseline.bottleneck else None,
            "num_constraints": len(diag_baseline.term_slacks),
            "families": list(diag_baseline.group_by_family().keys()),
        },
        "voronoi": {
            "pipeline": "conrey89_voronoi",
            "theta_max": diag_voronoi.theta_max,
            "headroom": diag_voronoi.headroom,
            "bottleneck": diag_voronoi.bottleneck.error_exponent if diag_voronoi.bottleneck else None,
            "bottleneck_family": diag_voronoi.bottleneck.bound_family if diag_voronoi.bottleneck else None,
            "num_constraints": len(diag_voronoi.term_slacks),
            "families": list(diag_voronoi.group_by_family().keys()),
        },
    }
