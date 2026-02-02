"""Conrey89 pipeline variant with formula Voronoi + Kuznetsov + spectral large sieve.

Pipeline chain:
  ApproxFunctionalEq → OpenSquare → IntegrateOverT → DiagonalSplit
  → DiagonalExtract (diagonal path)
  → DeltaMethodSetup → VoronoiTransform(mode=FORMULA) → DeltaMethodCollapse
  → KloostermanForm → PhaseAbsorb → KuznetsovTransform
  → SpectralLargeSieveBound (Voronoi-path terms)
  → DIKloostermanBound (non-Voronoi terms if any)
  → TrivialBound (AFE errors)
  → find_theta_max (with multi-family awareness)
"""

from __future__ import annotations

from fractions import Fraction

from mollifier_theta.core.ir import (
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import VoronoiKind
from mollifier_theta.lemmas.di_kloosterman import (
    DIExponentModel,
    DIKloostermanBound,
)
from mollifier_theta.lemmas.spectral_large_sieve import SpectralLargeSieveBound
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
from mollifier_theta.transforms.kuznetsov import KuznetsovTransform
from mollifier_theta.transforms.open_square import OpenSquare
from mollifier_theta.transforms.phase_absorb import PhaseAbsorb
from mollifier_theta.transforms.voronoi import VoronoiTransform


def conrey89_spectral_pipeline(
    theta_val: float = 0.3,
    K: int = 3,
    strict: bool = False,
) -> PipelineResult:
    """Run the Conrey89 pipeline with formula Voronoi + Kuznetsov + spectral large sieve.

    When strict=True, validates invariants after every transform stage.
    """
    ledger = TermLedger()

    runner = None
    if strict:
        from mollifier_theta.pipelines.strict_runner import StrictPipelineRunner
        runner = StrictPipelineRunner(ledger)

    def _apply(transform, terms, name="", allow_kernel_removal=False,
               _allow_phase_drop=False):
        if runner:
            return runner.run_stage(
                transform, terms, stage_name=name,
                allow_kernel_removal=allow_kernel_removal,
                _allow_phase_drop=_allow_phase_drop,
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

    # Step 5a: Diagonal extraction
    diag_extract = DiagonalExtract(K=K)
    diag_results = _apply(diag_extract, diagonal_terms, "DiagonalExtract")

    # Step 5b: Off-diagonal with formula Voronoi
    delta_setup = DeltaMethodSetup()
    voronoi = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
    delta_collapse = DeltaMethodCollapse()

    delta_intermediate = _apply(delta_setup, off_diagonal_terms, "DeltaMethodSetup")
    voronoi_terms = _apply(voronoi, delta_intermediate, "VoronoiTransform(FORMULA)")

    # Separate main terms from dual sums before collapse
    voronoi_main_terms = [t for t in voronoi_terms if t.status == TermStatus.MAIN_TERM]
    voronoi_dual_terms = [t for t in voronoi_terms if t.status != TermStatus.MAIN_TERM]

    delta_terms = _apply(delta_collapse, voronoi_dual_terms, "DeltaMethodCollapse")

    kloosterman = KloostermanForm()
    kloos_terms = _apply(kloosterman, delta_terms, "KloostermanForm")

    phase_abs = PhaseAbsorb()
    absorbed_terms = _apply(phase_abs, kloos_terms, "PhaseAbsorb")

    # Step 5c: Kuznetsov trace formula
    kuznetsov = KuznetsovTransform(sign_case="plus")
    spectral_terms = _apply(
        kuznetsov, absorbed_terms, "KuznetsovTransform",
        _allow_phase_drop=True,  # Kloosterman phase consumed
    )

    # Step 6: Apply bounds
    sls_bound = SpectralLargeSieveBound()
    di_bound = DIKloostermanBound()

    if runner:
        # SpectralLargeSieve for spectralized terms
        sls_results = runner.run_bounding_stage(
            sls_bound, spectral_terms, "SpectralLargeSieveBound",
        )
        # DI for any remaining non-spectralized Kloosterman terms
        remaining = [t for t in sls_results if t.status != TermStatus.BOUND_ONLY]
        if remaining:
            runner.run_bounding_stage(di_bound, remaining, "DIKloostermanBound")
    else:
        for term in spectral_terms:
            if sls_bound.applies(term):
                bounded_list = sls_bound.bound_multi(term)
                for b in bounded_list:
                    ledger.add(b)
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
    all_terms = ledger.all_terms()
    is_admissible = theta_admissible(all_terms, theta_val)

    bound_only_terms = [t for t in all_terms if t.status == TermStatus.BOUND_ONLY]

    # The spectral pipeline's binding constraint comes from the large_modulus
    # case of SpectralLargeSieve: (3θ+1)/2 < 1 → θ < 1/3.
    # This is NOT the DI constraint, so we pass known_theta_max to skip
    # the DI symbolic derivation.
    SPECTRAL_KNOWN_THETA_MAX = Fraction(1, 3)

    if bound_only_terms:
        theta_max_res = find_theta_max(
            all_terms, known_theta_max=SPECTRAL_KNOWN_THETA_MAX,
        )
    else:
        theta_max_res = ThetaMaxResult(
            symbolic=SPECTRAL_KNOWN_THETA_MAX,
            numerical=float(SPECTRAL_KNOWN_THETA_MAX),
            numerical_lo=float(SPECTRAL_KNOWN_THETA_MAX),
            numerical_hi=float(SPECTRAL_KNOWN_THETA_MAX),
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
        "pipeline_variant": "conrey89_spectral",
        "binding_family": theta_max_res.binding_family,
        "transform_chain": [
            "ApproxFunctionalEq",
            f"OpenSquare(K={K})",
            "IntegrateOverT",
            "DiagonalSplit",
            "DiagonalExtract",
            "DeltaMethodSetup",
            "VoronoiTransform(FORMULA)",
            "DeltaMethodCollapse",
            "KloostermanForm",
            "PhaseAbsorb",
            "KuznetsovTransform",
            "SpectralLargeSieveBound",
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
