"""Integration tests for the conrey89_spectral pipeline (Wave 1D)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.core.ir import KernelState, TermKind, TermStatus
from mollifier_theta.core.stage_meta import VoronoiKind, get_bound_meta, get_voronoi_meta
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.pipelines.conrey89_spectral import conrey89_spectral_pipeline
from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline


class TestSpectralPipelineBasic:
    def test_runs_without_errors(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert result is not None

    def test_admissible_at_low_theta(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert result.theta_admissible is True

    def test_inadmissible_above_theta_max(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.34, K=3)
        assert result.theta_admissible is False

    def test_theta_max_is_one_third(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert result.theta_max_result.symbolic == Fraction(1, 3)


class TestSpectralPipelineTermStructure:
    def test_has_bound_only_terms(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert len(result.bounded_terms) > 0

    def test_has_main_terms(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert len(result.main_terms) > 0

    def test_formula_voronoi_produces_main_and_dual(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        # Main terms should include Voronoi polar residuals
        voronoi_mains = [
            t for t in result.main_terms
            if t.metadata.get("voronoi_main_term")
        ]
        assert len(voronoi_mains) > 0

    def test_spectral_large_sieve_produces_multiple_bounds(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        sls_terms = [
            t for t in result.bounded_terms
            if get_bound_meta(t) and get_bound_meta(t).bound_family == "SpectralLargeSieve"
        ]
        # Should have multiple case-tree terms
        assert len(sls_terms) >= 3


class TestSpectralPipelineKernelState:
    def test_spectralized_terms_exist(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        spectralized = [
            t for t in all_terms
            if t.kernel_state == KernelState.SPECTRALIZED
        ]
        assert len(spectralized) > 0

    def test_spectral_kind_terms_exist(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        spectral = [t for t in all_terms if t.kind == TermKind.SPECTRAL]
        assert len(spectral) > 0


class TestSpectralPipelineBindingFamily:
    def test_binding_family_identified(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert result.theta_max_result.binding_family != ""

    def test_binding_family_is_spectral(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert result.theta_max_result.binding_family == "SpectralLargeSieve"


class TestBaselinePipelineUnchanged:
    def test_conrey89_still_four_sevenths(self) -> None:
        result = conrey89_pipeline(theta_val=0.56, K=3)
        assert result.theta_max_result.symbolic == Fraction(4, 7)

    def test_conrey89_voronoi_still_five_eighths(self) -> None:
        result = conrey89_voronoi_pipeline(theta_val=0.56, K=3)
        assert result.theta_max_result.symbolic == Fraction(5, 8)


class TestPipelineComparison:
    def test_three_distinct_families(self) -> None:
        """Compare pipelines show 3 distinct bound families."""
        r1 = conrey89_pipeline(theta_val=0.3, K=3)
        r2 = conrey89_voronoi_pipeline(theta_val=0.3, K=3)
        r3 = conrey89_spectral_pipeline(theta_val=0.3, K=3)

        families: set[str] = set()
        for result in [r1, r2, r3]:
            for t in result.bounded_terms:
                bm = get_bound_meta(t)
                if bm and bm.bound_family:
                    families.add(bm.bound_family)

        # Should have at least DI_Kloosterman, PostVoronoi, SpectralLargeSieve
        assert "DI_Kloosterman" in families
        assert "PostVoronoi" in families
        assert "SpectralLargeSieve" in families

    def test_theta_max_ordering(self) -> None:
        """Spectral < DI < PostVoronoi for theta_max."""
        r1 = conrey89_pipeline(theta_val=0.3, K=3)
        r2 = conrey89_voronoi_pipeline(theta_val=0.3, K=3)
        r3 = conrey89_spectral_pipeline(theta_val=0.3, K=3)

        # 1/3 < 4/7 < 5/8
        assert r3.theta_max < r1.theta_max < r2.theta_max


class TestStrictModeSpectral:
    def test_strict_mode_passes(self) -> None:
        """Strict mode should pass all invariant checks."""
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3, strict=True)
        assert result is not None
        assert result.theta_admissible is True
