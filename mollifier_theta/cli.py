"""CLI entry point using Typer."""

from __future__ import annotations

import typer

app = typer.Typer(name="mollifier", help="Mollifier theta expansion framework")
repro_app = typer.Typer(help="Reproduction pipelines")
app.add_typer(repro_app, name="repro")


@repro_app.command("conrey89")
def repro_conrey89(
    theta: float = typer.Option(0.56, help="Theta value for the mollifier length"),
) -> None:
    """Run the full Conrey89 reproduction pipeline."""
    from mollifier_theta.pipelines.conrey89 import run_conrey89_pipeline

    run_conrey89_pipeline(theta=theta)


@app.command("theta-sweep")
def theta_sweep(
    pipeline: str = typer.Argument("conrey89", help="Pipeline name"),
    theta_min: float = typer.Option(0.45, help="Minimum theta"),
    theta_max: float = typer.Option(0.65, help="Maximum theta"),
    step: float = typer.Option(0.005, help="Theta step size"),
) -> None:
    """Sweep theta grid and report pass/fail boundary."""
    from mollifier_theta.pipelines.theta_sweep import run_theta_sweep

    run_theta_sweep(theta_min=theta_min, theta_max=theta_max, step=step)


export_app = typer.Typer(help="Export utilities")
app.add_typer(export_app, name="export")

mathematica_app = typer.Typer(help="Mathematica export")
export_app.add_typer(mathematica_app, name="mathematica")


@mathematica_app.command("diagonal-main-term")
def export_mathematica_diagonal() -> None:
    """Export diagonal main term to Mathematica .wl file."""
    from mollifier_theta.reports.mathematica_export import export_diagonal_main_term

    export_diagonal_main_term()


if __name__ == "__main__":
    app()
