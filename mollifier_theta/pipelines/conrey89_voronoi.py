"""Conrey89 pipeline variant with Voronoi summation on the off-diagonal.

Identical to conrey89 except:
  DeltaMethodSetup -> VoronoiTransform(n) -> DeltaMethodCollapse -> ...

This is the "shadow pipeline" for testing Voronoi infrastructure.
The bounding step uses a placeholder post-Voronoi bound until real
bounds are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from mollifier_theta.core.ir import (
    HistoryEntry,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.serialize import export_dict, export_ledger
from mollifier_theta.lemmas.bound_strategy import PostVoronoiBound
from mollifier_theta.lemmas.di_kloosterman import (
    DIExponentModel,
    DIKloostermanBound,
)
from mollifier_theta.lemmas.theta_constraints import (
    ThetaMaxResult,
    find_theta_max,
    theta_admissible,
)
from mollifier_theta.lemmas.trivial_bounds import TrivialBound
from mollifier_theta.pipelines.conrey89 import PipelineResult
from mollifier_theta.transforms.approx_fe import ApproxFunctionalEq
from mollifier_theta.transforms.delta_method import (
    DeltaMethodCollapse,
    DeltaMethodSetup,
)
from mollifier_theta.transforms.diagonal_extract import DiagonalExtract
from mollifier_theta.transforms.diagonal_split import DiagonalSplit
from mollifier_theta.transforms.integrate_t import IntegrateOverT
from mollifier_theta.transforms.kloosterman_form import KloostermanForm
from mollifier_theta.transforms.open_square import OpenSquare
from mollifier_theta.transforms.phase_absorb import PhaseAbsorb
from mollifier_theta.transforms.voronoi import VoronoiTransform


def conrey89_voronoi_pipeline(
    theta_val: float = 0.56,
    K: int = 3,
    strict: bool = False,
) -> PipelineResult:
    """Run the Conrey89 pipeline with Voronoi inserted between Setup and Collapse.

    When strict=True, validates invariants after every transform stage.
    """
    ledger = TermLedger()

    runner = None
    if strict:
        from mollifier_theta.pipelines.strict_runner import StrictPipelineRunner
        runner = StrictPipelineRunner(ledger)

    def _apply(transform, terms, name="", allow_kernel_removal=False):
        if runner:
            return runner.run_stage(
                transform, terms, stage_name=name,
                allow_kernel_removal=allow_kernel_removal,
            )
        return transform.apply(terms, ledger)

    # Step 0: Initial integral term
    initial = Term(
        kind=TermKind.INTEGRAL,
        expression="int_0^T |M(1/2+it) zeta(1/2+it)|^2 dt",
        variables=["t"],
        ranges=[Range(variable="t", lower="0", upper="T")],
        metadata={"mollifier_length": K, "theta": theta_val},
    )
    ledger.add(initial)

    # Step 1: Approximate functional equation
    afe = ApproxFunctionalEq()
    afe_terms = _apply(afe, [initial], "ApproxFunctionalEq")

    main_afe_terms = [t for t in afe_terms if t.status != TermStatus.ERROR]
    afe_errors = [t for t in afe_terms if t.status == TermStatus.ERROR]

    # Step 2: Open the square
    open_sq = OpenSquare(K=K)
    cross_terms = _apply(open_sq, main_afe_terms, f"OpenSquare(K={K})")

    # Step 3: Integrate over t
    integrate = IntegrateOverT()
    integrated = _apply(integrate, cross_terms, "IntegrateOverT")

    # Step 4: Diagonal / off-diagonal split
    split = DiagonalSplit()
    split_terms = _apply(split, integrated, "DiagonalSplit")

    diagonal_terms = [t for t in split_terms if t.kind == TermKind.DIAGONAL]
    off_diagonal_terms = [t for t in split_terms if t.kind == TermKind.OFF_DIAGONAL]

    # Step 5a: Diagonal extraction (same as original)
    diag_extract = DiagonalExtract(K=K)
    diag_results = _apply(diag_extract, diagonal_terms, "DiagonalExtract")

    # Step 5b: Off-diagonal with Voronoi
    delta_setup = DeltaMethodSetup()
    voronoi = VoronoiTransform(target_variable="n")
    delta_collapse = DeltaMethodCollapse()

    delta_intermediate = _apply(delta_setup, off_diagonal_terms, "DeltaMethodSetup")
    voronoi_terms = _apply(voronoi, delta_intermediate, "VoronoiTransform")
    delta_terms = _apply(delta_collapse, voronoi_terms, "DeltaMethodCollapse")

    kloosterman = KloostermanForm()
    kloos_terms = _apply(kloosterman, delta_terms, "KloostermanForm")

    phase_abs = PhaseAbsorb()
    absorbed_terms = _apply(phase_abs, kloos_terms, "PhaseAbsorb")

    # Step 6: Apply bounds — PostVoronoiBound for voronoi-path terms,
    # DIKloostermanBound for any remaining non-voronoi Kloosterman terms.
    pv_bound = PostVoronoiBound()
    di_bound = DIKloostermanBound()

    if runner:
        pv_results = runner.run_bounding_stage(
            pv_bound, absorbed_terms, "PostVoronoiBound",
        )
        # Terms not handled by PostVoronoi go to DI
        remaining = [t for t in pv_results if t.status != TermStatus.BOUND_ONLY]
        if remaining:
            runner.run_bounding_stage(di_bound, remaining, "DIKloostermanBound")
    else:
        for term in absorbed_terms:
            if pv_bound.applies(term):
                bounded = pv_bound.bound(term)
                ledger.add(bounded)
            elif di_bound.applies(term):
                bounded = di_bound.bound(term)
                ledger.add(bounded)

    # Step 7: Trivial bounds for AFE errors
    trivial = TrivialBound()
    if runner:
        runner.run_bounding_stage(trivial, afe_errors, "TrivialBound")
    else:
        for err in afe_errors:
            if trivial.applies(err):
                bounded_err = trivial.bound(err)
                ledger.add(bounded_err)

    # Step 8: Theta check
    # The voronoi pipeline's binding constraint comes from PostVoronoiBound
    # (E(theta) = 2*theta - 1/4, theta_max = 5/8), not DI. Pass the known
    # constant so find_theta_max skips the DI symbolic derivation.
    from fractions import Fraction

    VORONOI_KNOWN_THETA_MAX = Fraction(5, 8)

    all_terms = ledger.all_terms()
    is_admissible = theta_admissible(all_terms, theta_val)

    bound_only_terms = [t for t in all_terms if t.status == TermStatus.BOUND_ONLY]
    di_model = DIExponentModel()

    if bound_only_terms:
        theta_max_res = find_theta_max(
            all_terms, known_theta_max=VORONOI_KNOWN_THETA_MAX,
        )
    else:
        theta_max_res = ThetaMaxResult(
            symbolic=VORONOI_KNOWN_THETA_MAX,
            numerical=float(VORONOI_KNOWN_THETA_MAX),
            numerical_lo=float(VORONOI_KNOWN_THETA_MAX),
            numerical_hi=float(VORONOI_KNOWN_THETA_MAX),
            tol=0.0,
        )

    main_terms = [t for t in all_terms if t.status == TermStatus.MAIN_TERM]
    error_terms = [t for t in all_terms if t.status == TermStatus.ERROR]

    report_data = {
        "theta_val": theta_val,
        "theta_admissible": is_admissible,
        "theta_max": theta_max_res.symbolic_float,
        "theta_max_numerical": theta_max_res.numerical,
        "theta_max_gap": theta_max_res.gap,
        "K": K,
        "total_terms": ledger.count(),
        "main_term_count": len(main_terms),
        "bound_only_count": len(bound_only_terms),
        "error_count": len(error_terms),
        "di_error_exponent": str(di_model.error_exponent),
        "pipeline_variant": "conrey89_voronoi",
        "transform_chain": [
            "ApproxFunctionalEq",
            f"OpenSquare(K={K})",
            "IntegrateOverT",
            "DiagonalSplit",
            "DiagonalExtract",
            "DeltaMethodSetup",
            "VoronoiTransform(n)",
            "DeltaMethodCollapse",
            "KloostermanForm",
            "PhaseAbsorb",
            "PostVoronoiBound",
        ],
    }

    return PipelineResult(
        ledger=ledger,
        theta_val=theta_val,
        theta_admissible=is_admissible,
        theta_max_result=theta_max_res,
        main_terms=main_terms,
        bounded_terms=bound_only_terms,
        error_terms=error_terms,
        report_data=report_data,
    )


def run_conrey89_voronoi_pipeline(theta: float = 0.56, K: int = 3) -> None:
    """CLI entry point for the Voronoi variant pipeline."""
    console = Console()
    console.print(f"[bold]Running Conrey89+Voronoi pipeline[/bold] theta={theta}, K={K}")

    result = conrey89_voronoi_pipeline(theta_val=theta, K=K)

    artifact_dir = Path("artifacts/repro_conrey89_voronoi")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    export_ledger(result.ledger, artifact_dir / "ledger.json")

    tmr = result.theta_max_result
    theta_report = {
        "theta_val": result.theta_val,
        "theta_admissible": result.theta_admissible,
        "theta_max_symbolic": str(tmr.symbolic) if tmr else None,
        "theta_max": result.theta_max,
        "pipeline_variant": "conrey89_voronoi",
        "derivation": result.report_data,
    }
    export_dict(theta_report, artifact_dir / "theta_report.json")

    status = "[green]PASS[/green]" if result.theta_admissible else "[red]FAIL[/red]"
    console.print(f"theta={theta}: {status}")
    if tmr is not None:
        console.print(f"theta_max (symbolic) = {tmr.symbolic}  ({tmr.symbolic_float})")
    console.print(f"Total terms in ledger: {result.ledger.count()}")
    console.print(f"Artifacts written to {artifact_dir}")
