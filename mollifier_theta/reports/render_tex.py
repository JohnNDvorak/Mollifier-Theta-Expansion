"""LaTeX report generator (optional)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mollifier_theta.pipelines.conrey89 import PipelineResult


def render_tex_report(result: "PipelineResult") -> str:
    """Render a PipelineResult as a LaTeX document."""
    rd = result.report_data
    lines: list[str] = []

    lines.append(r"\documentclass{article}")
    lines.append(r"\usepackage{amsmath,amssymb}")
    lines.append(r"\title{Conrey89 Reproduction Report}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append("")

    lines.append(r"\section{Parameters}")
    lines.append(f"Theta value: $\\theta = {rd['theta_val']}$")
    lines.append("")
    status = "PASS" if rd["theta_admissible"] else "FAIL"
    lines.append(f"Result: \\textbf{{{status}}}")
    lines.append("")
    lines.append(f"Derived $\\theta_{{\\max}} = {rd['theta_max']}$")
    lines.append("")

    lines.append(r"\section{The 4/7 Barrier}")
    lines.append(
        r"The off-diagonal error exponent is $E(\theta) = \frac{7\theta}{4}$."
    )
    lines.append(
        r"The condition $E(\theta) < 1$ yields $\theta < \frac{4}{7}$."
    )
    lines.append("")

    lines.append(r"\section{Sub-exponent Table}")
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\hline")
    lines.append(r"Component & Symbol & Exponent & Contribution \\")
    lines.append(r"\hline")
    for row in rd["di_exponent_table"]:
        c = row["component"].replace("_", r"\_")
        s = row["symbol"].replace("_", r"\_")
        e = row["exponent"]
        con = row["contribution"].replace("_", r"\_")
        lines.append(f"{c} & ${s}$ & ${e}$ & {con} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")

    lines.append(r"\end{document}")
    return "\n".join(lines)
