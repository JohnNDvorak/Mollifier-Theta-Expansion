"""Theta sweep: run pipeline over a grid, report pass/fail boundary."""

from __future__ import annotations

import csv
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mollifier_theta.pipelines.conrey89 import conrey89_pipeline


def theta_sweep(
    theta_min: float = 0.45,
    theta_max: float = 0.65,
    step: float = 0.005,
    K: int = 3,
) -> list[dict]:
    """Sweep theta grid, run pipeline at each value, return results."""
    results: list[dict] = []
    theta_val = theta_min

    while theta_val <= theta_max + 1e-9:
        try:
            pr = conrey89_pipeline(theta_val=round(theta_val, 6), K=K)
            results.append({
                "theta": round(theta_val, 6),
                "admissible": pr.theta_admissible,
                "theta_max": pr.theta_max,
                "total_terms": pr.ledger.count(),
            })
        except Exception as e:
            results.append({
                "theta": round(theta_val, 6),
                "admissible": False,
                "theta_max": None,
                "total_terms": 0,
                "error": str(e),
            })
        theta_val += step

    return results


def run_theta_sweep(
    theta_min: float = 0.45,
    theta_max: float = 0.65,
    step: float = 0.005,
    K: int = 3,
) -> None:
    """CLI entry point: sweep theta grid and write artifacts."""
    console = Console()
    console.print(
        f"[bold]Theta sweep[/bold] [{theta_min}, {theta_max}] step={step} K={K}"
    )

    results = theta_sweep(theta_min, theta_max, step, K)

    # Write CSV
    artifact_dir = Path("artifacts/theta_sweep")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifact_dir / "theta_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["theta", "admissible", "theta_max", "total_terms"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "theta": r["theta"],
                "admissible": r["admissible"],
                "theta_max": r.get("theta_max", ""),
                "total_terms": r.get("total_terms", ""),
            })

    # Summary table
    table = Table(title="Theta Sweep Results")
    table.add_column("theta", style="cyan")
    table.add_column("admissible", style="green")

    for r in results:
        status = "PASS" if r["admissible"] else "FAIL"
        style = "green" if r["admissible"] else "red"
        table.add_row(f'{r["theta"]:.4f}', f"[{style}]{status}[/{style}]")

    console.print(table)
    console.print(f"CSV written to {csv_path}")

    # Find boundary
    pass_thetas = [r["theta"] for r in results if r["admissible"]]
    fail_thetas = [r["theta"] for r in results if not r["admissible"]]
    if pass_thetas and fail_thetas:
        boundary = (max(pass_thetas) + min(fail_thetas)) / 2
        console.print(f"Pass/fail boundary approximately at theta = {boundary:.4f}")
        console.print(f"Known theta_max = 4/7 = {4/7:.6f}")
