"""Full Conrey89 reproduction pipeline.

Chains: ApproxFE -> OpenSquare(K=3) -> IntegrateOverT -> DiagonalSplit ->
  Diagonal path: DiagonalExtract
  Off-diagonal path: DeltaMethod -> KloostermanForm -> PhaseAbsorb -> DI bound
-> theta check

Returns PipelineResult with ledger, theta_max, main_term, report_data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from mollifier_theta.core.ir import (
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.serialize import export_dict, export_ledger
from mollifier_theta.lemmas.di_kloosterman import (
    DIExponentModel,
    DIKloostermanBound,
    ThetaBarrierMismatch,
)
from mollifier_theta.lemmas.theta_constraints import (
    ThetaMaxResult,
    find_theta_max,
    theta_admissible,
)
from mollifier_theta.lemmas.trivial_bounds import TrivialBound
from mollifier_theta.reports.mathematica_export import export_diagonal_main_term
from mollifier_theta.reports.render_md import render_report
from mollifier_theta.transforms.approx_fe import ApproxFunctionalEq
from mollifier_theta.transforms.delta_method import DeltaMethodInsert
from mollifier_theta.transforms.diagonal_extract import DiagonalExtract
from mollifier_theta.transforms.diagonal_split import DiagonalSplit
from mollifier_theta.transforms.integrate_t import IntegrateOverT
from mollifier_theta.transforms.kloosterman_form import KloostermanForm
from mollifier_theta.transforms.open_square import OpenSquare
from mollifier_theta.transforms.phase_absorb import PhaseAbsorb


@dataclass
class PipelineResult:
    """Result of running the Conrey89 pipeline."""

    ledger: TermLedger
    theta_val: float
    theta_admissible: bool
    theta_max_result: ThetaMaxResult | None
    main_terms: list[Term]
    bounded_terms: list[Term]
    error_terms: list[Term]
    report_data: dict = field(default_factory=dict)

    @property
    def theta_max(self) -> float | None:
        """Symbolic theta_max as float (backward-compatible accessor)."""
        if self.theta_max_result is None:
            return None
        return self.theta_max_result.symbolic_float


def conrey89_pipeline(theta_val: float = 0.56, K: int = 3) -> PipelineResult:
    """Run the full Conrey89 reproduction pipeline at a given theta.

    Returns a PipelineResult with the full ledger, theta analysis, and report data.
    """
    ledger = TermLedger()

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
    afe_terms = afe.apply([initial], ledger)

    # Separate error terms early
    main_afe_terms = [t for t in afe_terms if t.status != TermStatus.ERROR]
    afe_errors = [t for t in afe_terms if t.status == TermStatus.ERROR]

    # Step 2: Open the square for each main AFE term
    open_sq = OpenSquare(K=K)
    cross_terms = open_sq.apply(main_afe_terms, ledger)

    # Step 3: Integrate over t
    integrate = IntegrateOverT()
    integrated = integrate.apply(cross_terms, ledger)

    # Step 4: Diagonal / off-diagonal split
    split = DiagonalSplit()
    split_terms = split.apply(integrated, ledger)

    diagonal_terms = [t for t in split_terms if t.kind == TermKind.DIAGONAL]
    off_diagonal_terms = [t for t in split_terms if t.kind == TermKind.OFF_DIAGONAL]

    # Step 5a: Diagonal extraction
    diag_extract = DiagonalExtract(K=K)
    diag_results = diag_extract.apply(diagonal_terms, ledger)

    # Step 5b: Off-diagonal reduction
    delta = DeltaMethodInsert()
    delta_terms = delta.apply(off_diagonal_terms, ledger)

    kloosterman = KloostermanForm()
    kloos_terms = kloosterman.apply(delta_terms, ledger)

    phase_abs = PhaseAbsorb()
    absorbed_terms = phase_abs.apply(kloos_terms, ledger)

    # Step 6: Apply DI Kloosterman bound
    di_bound = DIKloostermanBound()
    bounded_off_diag: list[Term] = []
    active_off_diag: list[Term] = []

    for term in absorbed_terms:
        if di_bound.applies(term):
            bounded = di_bound.bound(term)
            ledger.add(bounded)
            bounded_off_diag.append(bounded)

            # Keep an Active copy as the "promotion hook" for future work
            active_off_diag.append(term)
        else:
            active_off_diag.append(term)

    # Step 7: Apply trivial bounds to AFE error terms
    trivial = TrivialBound()
    for err in afe_errors:
        if trivial.applies(err):
            bounded_err = trivial.bound(err)
            ledger.add(bounded_err)

    # Step 8: Theta check
    all_terms = ledger.all_terms()
    is_admissible = theta_admissible(all_terms, theta_val)

    # Compute and reconcile theta_max via all three paths:
    #   symbolic (solve E=1), regression constant (4/7), numerical (binary search)
    bound_only_terms = [t for t in all_terms if t.status == TermStatus.BOUND_ONLY]
    theta_max_res: ThetaMaxResult | None = None
    di_model = DIExponentModel()

    if bound_only_terms:
        theta_max_res = find_theta_max(all_terms)
        # find_theta_max already does the Layer 1 + Layer 2 cross-check
        # and raises ThetaBarrierMismatch on disagreement.
    else:
        # No bounded terms to binary-search over; fall back to pure symbolic
        symbolic_max = di_model.theta_max_with_crosscheck()
        theta_max_res = ThetaMaxResult(
            symbolic=symbolic_max,
            numerical=float(symbolic_max),
            numerical_lo=float(symbolic_max),
            numerical_hi=float(symbolic_max),
            tol=0.0,
        )

    # Gather results
    main_terms = [t for t in all_terms if t.status == TermStatus.MAIN_TERM]
    error_terms = [t for t in all_terms if t.status == TermStatus.ERROR]

    report_data = {
        "theta_val": theta_val,
        "theta_admissible": is_admissible,
        "theta_max": theta_max_res.symbolic_float,
        "theta_max_numerical": theta_max_res.numerical,
        "theta_max_gap": theta_max_res.gap,
        "theta_max_is_supremum": theta_max_res.is_supremum,
        "K": K,
        "total_terms": ledger.count(),
        "main_term_count": len(main_terms),
        "bound_only_count": len(bound_only_terms),
        "error_count": len(error_terms),
        "di_exponent_table": di_model.sub_exponent_table(),
        "di_error_exponent": str(di_model.error_exponent),
        "transform_chain": [
            "ApproxFunctionalEq",
            f"OpenSquare(K={K})",
            "IntegrateOverT",
            "DiagonalSplit",
            "DiagonalExtract",
            "DeltaMethod",
            "KloostermanForm",
            "PhaseAbsorb",
            "DIKloostermanBound",
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


def run_conrey89_pipeline(theta: float = 0.56, K: int = 3) -> None:
    """CLI entry point: run pipeline and write artifacts."""
    console = Console()

    console.print(f"[bold]Running Conrey89 pipeline[/bold] theta={theta}, K={K}")

    result = conrey89_pipeline(theta_val=theta, K=K)

    # Write artifacts
    artifact_dir = Path("artifacts/repro_conrey89")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    export_ledger(result.ledger, artifact_dir / "ledger.json")

    tmr = result.theta_max_result
    theta_report = {
        "theta_val": result.theta_val,
        "theta_admissible": result.theta_admissible,
        "theta_max_symbolic": str(tmr.symbolic) if tmr else None,
        "theta_max_symbolic_float": tmr.symbolic_float if tmr else None,
        "theta_max_numerical": tmr.numerical if tmr else None,
        "theta_max_gap": tmr.gap if tmr else None,
        "theta_max_is_supremum": tmr.is_supremum if tmr else None,
        "theta_max": result.theta_max,  # backward-compatible key
        "derivation": result.report_data,
    }
    export_dict(theta_report, artifact_dir / "theta_report.json")

    report_md = render_report(result)
    (artifact_dir / "report.md").write_text(report_md)

    # Mathematica export
    export_diagonal_main_term(result.ledger, artifact_dir / "mathematica")

    # Summary
    status = "[green]PASS[/green]" if result.theta_admissible else "[red]FAIL[/red]"
    console.print(f"theta={theta}: {status}")
    tmr = result.theta_max_result
    if tmr is not None:
        console.print(f"theta_max (symbolic) = {tmr.symbolic}  ({tmr.symbolic_float})")
        console.print(f"theta_max (numerical) = {tmr.numerical:.10f}  (binary search, tol={tmr.tol})")
        console.print(f"  gap = {tmr.gap:.2e}  (supremum, not attained: E(4/7) = 1)")
    console.print(f"Total terms in ledger: {result.ledger.count()}")
    console.print(f"Artifacts written to {artifact_dir}")
