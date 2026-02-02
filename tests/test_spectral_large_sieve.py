"""Tests for SpectralLargeSieveBound (Wave 1C)."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import (
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    KuznetsovMeta,
    VoronoiKind,
    VoronoiMeta,
    get_bound_meta,
)
from mollifier_theta.lemmas.spectral_large_sieve import SpectralLargeSieveBound
from mollifier_theta.lemmas.theta_constraints import theta_admissible


@pytest.fixture
def spectralized_term() -> Term:
    """A SPECTRALIZED term with formula Voronoi and Kuznetsov metadata."""
    return Term(
        kind=TermKind.SPECTRAL,
        expression="Spectral expansion from Kuznetsov",
        variables=["m", "n*"],
        ranges=[
            Range(variable="m", lower="1", upper="N"),
            Range(variable="n*", lower="1", upper="N_dual"),
        ],
        kernels=[
            Kernel(name="W_AFE"),
            Kernel(name="SpectralKernel"),
            Kernel(name="KuznetsovKernel"),
        ],
        phases=[
            Phase(
                expression="spectral_expansion(lambda_f(m)*lambda_f(n), h(t_f))",
                is_separable=True,
                depends_on=["m"],
                unit_modulus=False,
            ),
        ],
        kernel_state=KernelState.SPECTRALIZED,
        metadata={
            "voronoi_applied": True,
            "_voronoi": VoronoiMeta(
                applied=True,
                target_variable="n",
                dual_variable="n*",
                dual_length="c^2/T^theta",
                kind=VoronoiKind.FORMULA,
            ).model_dump(),
            "_kuznetsov": KuznetsovMeta(
                applied=True,
                sign_case="plus",
                bessel_transform="Phi_Kuznetsov",
                spectral_window_scale="K",
                spectral_components=["discrete_maass", "holomorphic", "eisenstein"],
                level="1",
            ).model_dump(),
        },
    )


@pytest.fixture
def structural_voronoi_term() -> Term:
    """A SPECTRALIZED term with structural (not formula) Voronoi — should be rejected."""
    return Term(
        kind=TermKind.SPECTRAL,
        expression="Spectral expansion",
        kernel_state=KernelState.SPECTRALIZED,
        metadata={
            "_voronoi": VoronoiMeta(
                applied=True,
                target_variable="n",
                kind=VoronoiKind.STRUCTURAL_ONLY,
            ).model_dump(),
            "_kuznetsov": KuznetsovMeta(applied=True).model_dump(),
        },
    )


class TestSpectralLargeSieveBasic:
    def test_produces_three_terms(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        results = sls.bound_multi(spectralized_term)
        assert len(results) == 3

    def test_all_bound_only(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        results = sls.bound_multi(spectralized_term)
        for t in results:
            assert t.status == TermStatus.BOUND_ONLY

    def test_distinct_case_ids(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        results = sls.bound_multi(spectralized_term)
        case_ids = set()
        for t in results:
            bm = get_bound_meta(t)
            assert bm is not None
            case_ids.add(bm.case_id)
        assert len(case_ids) == 3
        assert "small_modulus" in case_ids
        assert "large_modulus" in case_ids
        assert "bessel_transition" in case_ids


class TestSpectralLargeSieveMetadata:
    def test_each_has_scale_model(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        results = sls.bound_multi(spectralized_term)
        for t in results:
            assert "scale_model_dict" in t.metadata

    def test_each_has_lemma_citation(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        results = sls.bound_multi(spectralized_term)
        for t in results:
            assert t.lemma_citation != ""

    def test_bound_family_is_spectral(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        results = sls.bound_multi(spectralized_term)
        for t in results:
            bm = get_bound_meta(t)
            assert bm.bound_family == "SpectralLargeSieve"


class TestSpectralLargeSieveGating:
    def test_rejects_structural_voronoi(self, structural_voronoi_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        assert sls.applies(structural_voronoi_term) is False

    def test_rejects_non_spectralized(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            kernel_state=KernelState.KLOOSTERMANIZED,
        )
        sls = SpectralLargeSieveBound()
        assert sls.applies(term) is False

    def test_rejects_missing_kuznetsov(self) -> None:
        term = Term(
            kind=TermKind.SPECTRAL,
            kernel_state=KernelState.SPECTRALIZED,
            metadata={
                "_voronoi": VoronoiMeta(
                    applied=True, kind=VoronoiKind.FORMULA,
                ).model_dump(),
                "_kuznetsov": KuznetsovMeta(applied=True).model_dump(),
            },
        )
        # Passes because applied=True
        sls = SpectralLargeSieveBound()
        assert sls.applies(term) is True

    def test_rejects_kuznetsov_not_applied(self) -> None:
        term = Term(
            kind=TermKind.SPECTRAL,
            kernel_state=KernelState.SPECTRALIZED,
            metadata={
                "_voronoi": VoronoiMeta(
                    applied=True, kind=VoronoiKind.FORMULA,
                ).model_dump(),
                "_kuznetsov": KuznetsovMeta(applied=False).model_dump(),
            },
        )
        sls = SpectralLargeSieveBound()
        assert sls.applies(term) is False

    def test_accepts_valid_spectralized(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        assert sls.applies(spectralized_term) is True


class TestSpectralLargeSieveConstraints:
    def test_produces_constraints(self) -> None:
        sls = SpectralLargeSieveBound()
        constraints = sls.constraints()
        assert len(constraints) == 3

    def test_constraints_have_expressions(self) -> None:
        sls = SpectralLargeSieveBound()
        for c in sls.constraints():
            assert c.expression_str != ""
            assert c.bound_family == "SpectralLargeSieve"

    def test_constraints_solvable(self) -> None:
        sls = SpectralLargeSieveBound()
        for c in sls.constraints():
            tm = c.solve_theta_max()
            assert 0 < tm < 1


class TestThetaAdmissibleMultiCase:
    def test_admissible_at_low_theta(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        bound_terms = sls.bound_multi(spectralized_term)
        # At theta=0.3, all cases should be admissible
        assert theta_admissible(bound_terms, 0.3) is True

    def test_inadmissible_at_high_theta(self, spectralized_term: Term) -> None:
        sls = SpectralLargeSieveBound()
        bound_terms = sls.bound_multi(spectralized_term)
        # At theta=0.9, should be inadmissible
        assert theta_admissible(bound_terms, 0.9) is False


class TestStrictRunnerMultiBound:
    def test_strict_runner_handles_multi_bound(self, spectralized_term: Term) -> None:
        from mollifier_theta.pipelines.strict_runner import StrictPipelineRunner
        ledger = TermLedger()
        ledger.add(spectralized_term)
        runner = StrictPipelineRunner(ledger)
        sls = SpectralLargeSieveBound()
        results = runner.run_bounding_stage(
            sls, [spectralized_term], "SpectralLargeSieveBound",
        )
        # Should produce 3 BoundOnly terms
        bound_results = [t for t in results if t.status == TermStatus.BOUND_ONLY]
        assert len(bound_results) == 3


class TestRedFlagBInvariant:
    def test_structural_voronoi_rejected_by_invariant(self) -> None:
        """Red Flag B: SpectralLargeSieve bound with structural Voronoi must fail validation."""
        from mollifier_theta.core.invariants import validate_term
        term = Term(
            kind=TermKind.SPECTRAL,
            kernel_state=KernelState.SPECTRALIZED,
            status=TermStatus.BOUND_ONLY,
            lemma_citation="test",
            metadata={
                "_voronoi": VoronoiMeta(
                    applied=True,
                    kind=VoronoiKind.STRUCTURAL_ONLY,
                ).model_dump(),
                "_kuznetsov": KuznetsovMeta(applied=True).model_dump(),
                "_bound": BoundMeta(
                    strategy="SpectralLargeSieve",
                    bound_family="SpectralLargeSieve",
                    citation="test",
                ).model_dump(),
            },
        )
        violations = validate_term(term)
        assert any("SpectralLargeSieve" in v for v in violations)

    def test_formula_voronoi_passes_invariant(self) -> None:
        from mollifier_theta.core.invariants import validate_term
        term = Term(
            kind=TermKind.SPECTRAL,
            kernel_state=KernelState.SPECTRALIZED,
            status=TermStatus.BOUND_ONLY,
            lemma_citation="test",
            metadata={
                "_voronoi": VoronoiMeta(
                    applied=True,
                    kind=VoronoiKind.FORMULA,
                ).model_dump(),
                "_kuznetsov": KuznetsovMeta(applied=True).model_dump(),
                "_bound": BoundMeta(
                    strategy="SpectralLargeSieve",
                    bound_family="SpectralLargeSieve",
                    citation="test",
                ).model_dump(),
            },
        )
        violations = validate_term(term)
        assert not any("SpectralLargeSieve" in v for v in violations)
