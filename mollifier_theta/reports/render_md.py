"""Markdown report generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mollifier_theta.pipelines.conrey89 import PipelineResult


def render_report(result: "PipelineResult") -> str:
    """Render a PipelineResult as a Markdown report."""
    rd = result.report_data
    lines: list[str] = []

    lines.append("# Conrey89 Reproduction Report")
    lines.append("")
    lines.append(f"**Theta value:** {rd['theta_val']}")
    status = "PASS" if rd["theta_admissible"] else "FAIL"
    lines.append(f"**Result:** {status}")
    lines.append(f"**Theta max (symbolic):** {rd['theta_max']}  (= 4/7 exactly)")
    lines.append(f"**Theta max (numerical):** {rd.get('theta_max_numerical', 'N/A')}")
    gap = rd.get("theta_max_gap", None)
    if gap is not None:
        lines.append(f"**Symbolic/numerical gap:** {gap:.2e}")
    lines.append(f"**Semantics:** supremum (strict inequality E(theta) < 1; theta = 4/7 itself is inadmissible)")
    lines.append(f"**Mollifier length K:** {rd['K']}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total terms in ledger: {rd['total_terms']}")
    lines.append(f"- Main terms: {rd['main_term_count']}")
    lines.append(f"- Bound-only terms: {rd['bound_only_count']}")
    lines.append(f"- Error terms: {rd['error_count']}")
    lines.append("")

    lines.append("## Transform Chain")
    lines.append("")
    for i, t in enumerate(rd["transform_chain"], 1):
        lines.append(f"{i}. {t}")
    lines.append("")

    lines.append("## Where 4/7 Comes From")
    lines.append("")
    lines.append(
        "The theta < 4/7 barrier arises from the Deshouillers-Iwaniec (DI) "
        "bilinear Kloosterman bound applied to the off-diagonal terms of the "
        "mollified second moment."
    )
    lines.append("")
    lines.append(f"**Error exponent:** E(theta) = {rd['di_error_exponent']}")
    lines.append("")
    lines.append(
        "The off-diagonal error is O(T^{E(theta)+epsilon}). For this to be "
        "negligible compared to the main term T * P(theta), we need E(theta) < 1."
    )
    lines.append("")
    lines.append("E(theta) < 1  iff  7*theta/4 < 1  iff  theta < 4/7.")
    lines.append("")

    lines.append("### Sub-exponent Breakdown")
    lines.append("")
    lines.append("| Component | Symbol | Exponent | Contribution |")
    lines.append("|-----------|--------|----------|-------------|")
    for row in rd["di_exponent_table"]:
        lines.append(
            f"| {row['component']} | {row['symbol']} | {row['exponent']} | {row['contribution']} |"
        )
    lines.append("")

    lines.append("## Analytic vs Numerical Reconciliation")
    lines.append("")
    lines.append("Three independent paths determine theta_max:")
    lines.append("")
    lines.append(f"1. **Symbolic (Layer 1):** Solve E(theta) = 7*theta/4 = 1 via SymPy -> theta = 4/7 exactly.")
    lines.append(f"2. **Known constant (Layer 2):** KNOWN_THETA_MAX = 4/7 (regression guard from Conrey 1989).")
    gap_str = f"{gap:.2e}" if gap is not None else "N/A"
    numerical_str = f"{rd.get('theta_max_numerical', 'N/A')}"
    lines.append(f"3. **Numerical (binary search):** theta_max ~ {numerical_str} (gap from symbolic: {gap_str}).")
    lines.append("")
    lines.append(
        "The admissibility check uses **strict inequality** E(theta) < 1. "
        "At theta = 4/7, E(4/7) = 1.0 exactly, so 4/7 itself is *not* admissible. "
        "Thus 4/7 is the **supremum** of admissible theta values, not the maximum. "
        "The binary search converges to a value slightly below 4/7 (within ±tol of "
        "the true boundary), and the gap between the numerical midpoint and the "
        "symbolic value is bounded by the search tolerance."
    )
    lines.append("")

    lines.append("## Citations")
    lines.append("")
    lines.append("- Conrey, J.B. (1989). \"More than two fifths of the zeros of the Riemann zeta function are on the critical line.\" *J. reine angew. Math.* **399**, 1-26.")
    lines.append("- Deshouillers, J.-M. and Iwaniec, H. (1982). \"Kloosterman sums and Fourier coefficients of cusp forms.\" *Invent. Math.* **70**, 219-288.")
    lines.append("- Deshouillers, J.-M. and Iwaniec, H. (1983). \"An additive divisor problem.\" *J. London Math. Soc.* **26**, 1-14.")
    lines.append("")

    return "\n".join(lines)
