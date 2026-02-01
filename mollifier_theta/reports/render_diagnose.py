"""Rich tables and JSON export for diagnose output."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mollifier_theta.analysis.slack import DiagnoseResult
from mollifier_theta.analysis.what_if import WhatIfResult


def render_slack_table(result: DiagnoseResult, console: Console | None = None) -> None:
    """Print a Rich table summarizing slack for each BoundOnly term."""
    if console is None:
        console = Console()

    console.print(f"\n[bold]Diagnose: slack analysis at theta = {result.theta_val}[/bold]")
    console.print(f"theta_max = {result.theta_max:.10f}   headroom = {result.headroom:.6f}\n")

    table = Table(title="BoundOnly Term Slacks (ascending)")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("E(theta)", justify="right")
    table.add_column("Slack", justify="right")
    table.add_column("Error Exponent")
    table.add_column("Citation", style="dim")

    for i, ts in enumerate(result.term_slacks, 1):
        style = "bold red" if ts.slack <= 0 else ("yellow" if ts.slack < 0.05 else "green")
        table.add_row(
            str(i),
            f"{ts.E_val:.6f}",
            f"{ts.slack:.6f}",
            ts.error_exponent,
            ts.lemma_citation[:50],
            style=style,
        )

    console.print(table)

    if result.bottleneck:
        console.print(f"\n[bold]Bottleneck:[/bold] {result.bottleneck.error_exponent}")
        if result.bottleneck.sub_exponents:
            console.print("  Sub-exponents:")
            for k, v in result.bottleneck.sub_exponents.items():
                console.print(f"    {k}: {v:.6f}")


def render_what_if_table(result: WhatIfResult, console: Console | None = None) -> None:
    """Print a Rich summary of a what-if analysis."""
    if console is None:
        console = Console()

    console.print(f"\n[bold]What-If Analysis:[/bold] {result.scenario.name}")
    console.print(f"  Old expression: {result.scenario.old_expr}")
    console.print(f"  New expression: {result.scenario.new_expr}")
    console.print(f"  Old E(theta): {result.old_E_expr}")
    console.print(f"  New E(theta): {result.new_E_expr}")
    console.print(f"  Old theta_max: {result.old_theta_max:.10f}")
    console.print(f"  New theta_max: {result.new_theta_max:.10f}")

    diff = result.improvement
    if diff > 0:
        console.print(f"  [green]Improvement: +{diff:.10f}[/green]")
    elif diff < 0:
        console.print(f"  [red]Regression:  {diff:.10f}[/red]")
    else:
        console.print("  [dim]No change[/dim]")


def slack_to_json(result: DiagnoseResult) -> dict:
    """Convert DiagnoseResult to a JSON-serializable dict."""
    return {
        "theta_val": result.theta_val,
        "theta_max": result.theta_max,
        "headroom": result.headroom,
        "bottleneck": asdict(result.bottleneck) if result.bottleneck else None,
        "term_slacks": [asdict(ts) for ts in result.term_slacks],
    }


def what_if_to_json(result: WhatIfResult) -> dict:
    """Convert WhatIfResult to a JSON-serializable dict."""
    return {
        "scenario": asdict(result.scenario),
        "old_theta_max": result.old_theta_max,
        "new_theta_max": result.new_theta_max,
        "improvement": result.improvement,
        "old_E_expr": result.old_E_expr,
        "new_E_expr": result.new_E_expr,
    }


def export_diagnose_json(data: dict, path: Path) -> None:
    """Write a diagnose result dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
