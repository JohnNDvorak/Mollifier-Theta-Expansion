"""Proof certificate exporter.

Generates a human-readable audit trail showing:
- Transform chain
- Per-stage term counts
- Constraint list feeding theta_max
- Binding constraint and its history path back to the initial integral

Every theta change is explainable in one artifact.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from mollifier_theta.analysis.slack import DiagnoseResult, diagnose_pipeline
from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.pipelines.conrey89 import PipelineResult


def _environment_stamp() -> dict:
    """Capture reproducibility-relevant environment information."""
    stamp: dict = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Pydantic version
    try:
        import pydantic
        stamp["pydantic_version"] = pydantic.__version__
    except ImportError:
        stamp["pydantic_version"] = "unknown"

    # SymPy version
    try:
        import sympy
        stamp["sympy_version"] = sympy.__version__
    except ImportError:
        stamp["sympy_version"] = "unknown"

    # Git SHA (best-effort)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            stamp["git_sha"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return stamp


def _trace_history_path(term: Term) -> list[dict]:
    """Trace the derivation history from a term back to its roots."""
    return [
        {
            "transform": h.transform,
            "parent_ids": h.parent_ids,
            "description": h.description,
        }
        for h in term.history
    ]


def _count_by_stage(terms: list[Term]) -> dict[str, int]:
    """Count terms grouped by their last transform (pipeline stage)."""
    counts: dict[str, int] = {}
    for t in terms:
        if t.history:
            stage = t.history[-1].transform
        else:
            stage = "initial"
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def _count_by_status(terms: list[Term]) -> dict[str, int]:
    """Count terms by TermStatus."""
    counts: dict[str, int] = {}
    for t in terms:
        key = t.status.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def generate_proof_certificate(
    result: PipelineResult,
    diagnose: DiagnoseResult | None = None,
) -> dict:
    """Generate a proof certificate dict for a pipeline result.

    This is the single artifact that makes every theta change explainable.
    """
    all_terms = result.ledger.all_terms()

    if diagnose is None:
        diagnose = diagnose_pipeline(
            theta_val=result.theta_val, result=result,
        )

    # Build constraint table
    constraint_table = []
    for ts in diagnose.term_slacks:
        constraint_table.append({
            "term_id": ts.term_id,
            "error_exponent": ts.error_exponent,
            "E_val": ts.E_val,
            "slack": ts.slack,
            "bound_family": ts.bound_family,
            "pipeline_stage": ts.pipeline_stage,
            "lemma_citation": ts.lemma_citation,
        })

    # Trace the binding constraint history
    binding_history = None
    if diagnose.bottleneck:
        # Find the actual term in the ledger
        binding_term = None
        for t in all_terms:
            if t.id == diagnose.bottleneck.term_id:
                binding_term = t
                break
        if binding_term:
            binding_history = {
                "term_id": binding_term.id,
                "error_exponent": diagnose.bottleneck.error_exponent,
                "bound_family": diagnose.bottleneck.bound_family,
                "lemma_citation": diagnose.bottleneck.lemma_citation,
                "derivation_path": _trace_history_path(binding_term),
                "sub_exponents": diagnose.bottleneck.sub_exponents,
            }

    # Build a content fingerprint from run-stable fields (excludes term IDs)
    fingerprint_data = json.dumps({
        "theta_val": result.theta_val,
        "theta_admissible": result.theta_admissible,
        "theta_max": diagnose.theta_max,
        "transform_chain": result.report_data.get("transform_chain", []),
        "constraint_exponents": sorted(
            (c["error_exponent"], c["bound_family"]) for c in constraint_table
        ),
        "binding_exponent": binding_history["error_exponent"] if binding_history else None,
        "binding_family": binding_history["bound_family"] if binding_history else None,
    }, sort_keys=True, default=str)
    content_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    certificate = {
        "theta_val": result.theta_val,
        "theta_admissible": result.theta_admissible,
        "theta_max": diagnose.theta_max,
        "headroom": diagnose.headroom,
        "pipeline_variant": result.report_data.get("pipeline_variant", "conrey89"),
        "transform_chain": result.report_data.get("transform_chain", []),
        "term_counts": {
            "total": len(all_terms),
            "by_status": _count_by_status(all_terms),
            "by_stage": _count_by_stage(all_terms),
        },
        "constraints": constraint_table,
        "binding_constraint": binding_history,
        "verification": {
            "theta_max_symbolic": str(result.theta_max_result.symbolic) if result.theta_max_result else None,
            "theta_max_numerical": result.theta_max_result.numerical if result.theta_max_result else None,
            "gap": result.theta_max_result.gap if result.theta_max_result else None,
            "is_supremum": result.theta_max_result.is_supremum if result.theta_max_result else None,
        },
        "content_fingerprint": content_fingerprint,
        "environment": _environment_stamp(),
    }

    return certificate


def render_proof_certificate_md(certificate: dict) -> str:
    """Render a proof certificate as Markdown."""
    lines = []
    lines.append("# Proof Certificate")
    lines.append("")
    lines.append(f"**theta** = {certificate['theta_val']}")
    lines.append(f"**theta_max** = {certificate['theta_max']}")
    lines.append(f"**admissible** = {certificate['theta_admissible']}")
    lines.append(f"**headroom** = {certificate['headroom']:.6f}")
    lines.append(f"**pipeline** = {certificate['pipeline_variant']}")
    lines.append("")

    lines.append("## Transform Chain")
    lines.append("")
    for i, t in enumerate(certificate["transform_chain"], 1):
        lines.append(f"{i}. {t}")
    lines.append("")

    lines.append("## Term Counts")
    lines.append("")
    lines.append(f"Total: {certificate['term_counts']['total']}")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status, count in certificate["term_counts"]["by_status"].items():
        lines.append(f"| {status} | {count} |")
    lines.append("")

    lines.append("| Stage | Count |")
    lines.append("|-------|-------|")
    for stage, count in certificate["term_counts"]["by_stage"].items():
        lines.append(f"| {stage} | {count} |")
    lines.append("")

    lines.append("## Constraints Feeding theta_max")
    lines.append("")
    lines.append("| E(theta) | Slack | Family | Citation |")
    lines.append("|----------|-------|--------|----------|")
    for c in certificate["constraints"]:
        lines.append(
            f"| {c['E_val']:.6f} | {c['slack']:.6f} | "
            f"{c['bound_family']} | {c['lemma_citation'][:40]} |"
        )
    lines.append("")

    bc = certificate["binding_constraint"]
    if bc:
        lines.append("## Binding Constraint")
        lines.append("")
        lines.append(f"**Error exponent**: {bc['error_exponent']}")
        lines.append(f"**Bound family**: {bc['bound_family']}")
        lines.append(f"**Citation**: {bc['lemma_citation']}")
        lines.append("")
        lines.append("### Derivation Path (initial integral → bound)")
        lines.append("")
        for i, step in enumerate(bc["derivation_path"], 1):
            lines.append(f"{i}. **{step['transform']}**: {step['description']}")
        lines.append("")

        if bc.get("sub_exponents"):
            lines.append("### Sub-exponents at theta = " + str(certificate["theta_val"]))
            lines.append("")
            for k, v in bc["sub_exponents"].items():
                lines.append(f"- {k}: {v:.6f}")
            lines.append("")

    v = certificate["verification"]
    lines.append("## Verification")
    lines.append("")
    lines.append(f"- Symbolic theta_max: {v['theta_max_symbolic']}")
    lines.append(f"- Numerical theta_max: {v['theta_max_numerical']}")
    lines.append(f"- Gap: {v['gap']}")
    lines.append(f"- Is supremum: {v['is_supremum']}")
    lines.append("")

    return "\n".join(lines)


def export_proof_certificate(
    result: PipelineResult,
    output_dir: Path,
    diagnose: DiagnoseResult | None = None,
) -> None:
    """Export proof certificate as both JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cert = generate_proof_certificate(result, diagnose)

    # JSON
    json_path = output_dir / "proof_certificate.json"
    json_path.write_text(json.dumps(cert, indent=2, sort_keys=True, default=str))

    # Markdown
    md_path = output_dir / "proof_certificate.md"
    md_path.write_text(render_proof_certificate_md(cert))
