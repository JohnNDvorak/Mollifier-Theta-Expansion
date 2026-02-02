"""CLI entry point using Typer."""

from __future__ import annotations

from pathlib import Path

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


# --- Diagnose sub-app ---
diagnose_app = typer.Typer(help="Diagnostic analysis commands")
app.add_typer(diagnose_app, name="diagnose")


@diagnose_app.command("slack")
def diagnose_slack(
    theta: float = typer.Option(0.56, help="Theta value to diagnose"),
    k: int = typer.Option(3, "--K", help="Mollifier length K"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Identify the binding constraint (smallest slack) on theta."""
    from mollifier_theta.analysis.slack import diagnose_pipeline
    from mollifier_theta.reports.render_diagnose import (
        export_diagnose_json,
        render_slack_table,
        slack_to_json,
    )

    result = diagnose_pipeline(theta_val=theta, K=k)

    if json_output:
        import json

        typer.echo(json.dumps(slack_to_json(result), indent=2, default=str))
    else:
        render_slack_table(result)

    artifact_path = Path("artifacts/diagnose/slack.json")
    export_diagnose_json(slack_to_json(result), artifact_path)


@diagnose_app.command("what-if")
def diagnose_what_if(
    name: str = typer.Argument(..., help="Sub-exponent name (e.g. di_saving)"),
    new_expr: str = typer.Argument(..., help='New expression (e.g. "-theta/3")'),
    theta: float = typer.Option(0.56, help="Theta value for context"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Explore hypothetical sub-exponent changes."""
    from mollifier_theta.analysis.what_if import what_if_analysis
    from mollifier_theta.reports.render_diagnose import (
        export_diagnose_json,
        render_what_if_table,
        what_if_to_json,
    )

    try:
        result = what_if_analysis(name, new_expr)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    if json_output:
        import json

        typer.echo(json.dumps(what_if_to_json(result), indent=2, default=str))
    else:
        render_what_if_table(result)

    artifact_path = Path("artifacts/diagnose/what_if.json")
    export_diagnose_json(what_if_to_json(result), artifact_path)


@repro_app.command("conrey89-voronoi")
def repro_conrey89_voronoi(
    theta: float = typer.Option(0.56, help="Theta value for the mollifier length"),
    k: int = typer.Option(3, "--K", help="Mollifier length K"),
) -> None:
    """Run the Conrey89 pipeline with Voronoi summation on the off-diagonal."""
    from mollifier_theta.pipelines.conrey89_voronoi import run_conrey89_voronoi_pipeline

    run_conrey89_voronoi_pipeline(theta=theta, K=k)


@export_app.command("proof-cert")
def export_proof_cert(
    theta: float = typer.Option(0.56, help="Theta value"),
    k: int = typer.Option(3, "--K", help="Mollifier length K"),
    output_dir: Path = typer.Option(
        Path("artifacts/proof_certificate"), help="Output directory",
    ),
    pipeline: str = typer.Option("conrey89", help="Pipeline variant (conrey89 or conrey89-voronoi)"),
) -> None:
    """Export a proof certificate (JSON + Markdown) for a pipeline run."""
    from mollifier_theta.reports.proof_certificate import export_proof_certificate

    if pipeline == "conrey89-voronoi":
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline

        result = conrey89_voronoi_pipeline(theta_val=theta, K=k)
    else:
        from mollifier_theta.pipelines.conrey89 import conrey89_pipeline

        result = conrey89_pipeline(theta_val=theta, K=k)

    export_proof_certificate(result, output_dir)
    typer.echo(f"Proof certificate written to {output_dir}")


@diagnose_app.command("compare")
def diagnose_compare(
    theta: float = typer.Option(0.56, help="Theta value to diagnose"),
    k: int = typer.Option(3, "--K", help="Mollifier length K"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Compare slack diagnostics between conrey89 and conrey89-voronoi pipelines."""
    import json as json_mod

    from mollifier_theta.analysis.slack import compare_pipelines

    comparison = compare_pipelines(theta_val=theta, K=k)

    if json_output:
        typer.echo(json_mod.dumps(comparison, indent=2, default=str))
    else:
        typer.echo(f"Pipeline comparison at theta={theta}")
        typer.echo("")
        for variant in ("baseline", "voronoi"):
            info = comparison[variant]
            typer.echo(f"  {info['pipeline']}:")
            typer.echo(f"    theta_max   = {info['theta_max']:.10f}")
            typer.echo(f"    headroom    = {info['headroom']:.6f}")
            typer.echo(f"    bottleneck  = {info['bottleneck']}")
            typer.echo(f"    family      = {info['bottleneck_family']}")
            typer.echo(f"    constraints = {info['num_constraints']}")
            typer.echo(f"    families    = {info['families']}")
            typer.echo("")

    artifact_path = Path("artifacts/diagnose/comparison.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json_mod.dumps(comparison, indent=2, default=str))


@export_app.command("math-params")
def export_math_params(
    theta: float = typer.Option(0.56, help="Theta value"),
    k: int = typer.Option(3, "--K", help="Mollifier length K"),
    pipeline: str = typer.Option(
        "conrey89",
        help="Pipeline variant (conrey89, conrey89-voronoi, conrey89-spectral)",
    ),
    output: Path = typer.Option(
        Path("artifacts/math_params.json"), "--out", help="Output JSON path",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print to stdout as JSON"),
) -> None:
    """Export math parameters for BoundOnly terms (bottleneck + top constraints)."""
    import json as json_mod

    from mollifier_theta.reports.math_parameter_export import export_math_parameters_json

    if pipeline == "conrey89-voronoi":
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline

        result = conrey89_voronoi_pipeline(theta_val=theta, K=k)
    elif pipeline == "conrey89-spectral":
        from mollifier_theta.pipelines.conrey89_spectral import conrey89_spectral_pipeline

        result = conrey89_spectral_pipeline(theta_val=theta, K=k)
    else:
        from mollifier_theta.pipelines.conrey89 import conrey89_pipeline

        result = conrey89_pipeline(theta_val=theta, K=k)

    all_terms = result.ledger.all_terms()
    records = export_math_parameters_json(all_terms)

    if json_output:
        typer.echo(json_mod.dumps(records, indent=2))
    else:
        typer.echo(f"Exported {len(records)} math parameter records")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_mod.dumps(records, indent=2))
    typer.echo(f"Written to {output}")


@diagnose_app.command("overhead")
def diagnose_overhead(
    theta: float = typer.Option(0.56, help="Theta value"),
    k: int = typer.Option(3, "--K", help="Mollifier length K"),
    pipeline: str = typer.Option(
        "conrey89",
        help="Pipeline variant (conrey89, conrey89-voronoi)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show theta-loss decomposition: pipeline vs raw DI overhead."""
    import json as json_mod

    from mollifier_theta.analysis.overhead_report import compute_overhead

    if pipeline == "conrey89-voronoi":
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline

        result = conrey89_voronoi_pipeline(theta_val=theta, K=k)
    else:
        from mollifier_theta.pipelines.conrey89 import conrey89_pipeline

        result = conrey89_pipeline(theta_val=theta, K=k)

    all_terms = result.ledger.all_terms()
    report = compute_overhead(all_terms, theta_val=theta)

    if json_output:
        typer.echo(json_mod.dumps(report.to_json(), indent=2))
    else:
        typer.echo(report.format_table())

    artifact_path = Path("artifacts/diagnose/overhead.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json_mod.dumps(report.to_json(), indent=2))


if __name__ == "__main__":
    app()
