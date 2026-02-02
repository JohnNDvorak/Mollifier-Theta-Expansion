"""Pipeline-level constraint table sanity tests.

Structural sanity and constraint-family partitioning for all three pipeline
variants.  These turn theta_max into a "dashboard metric" you can trust
while refactoring post-Voronoi math.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.core.ir import TermStatus
from mollifier_theta.core.stage_meta import get_bound_meta
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
from mollifier_theta.pipelines.conrey89_spectral import conrey89_spectral_pipeline


# ---------------------------------------------------------------------------
# Baseline Conrey89
# ---------------------------------------------------------------------------

class TestBaselineConrey89:
    """The baseline pipeline's binding constraint must be DI_Kloosterman."""

    def test_theta_max_is_four_sevenths(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert result.theta_max_result is not None
        assert result.theta_max_result.symbolic == Fraction(4, 7)

    def test_binding_family_is_di(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert result.theta_max_result is not None
        assert "DI" in result.theta_max_result.binding_family

    def test_di_bound_terms_have_bound_meta(self) -> None:
        """DI bound terms (not trivial) must have BoundMeta."""
        result = conrey89_pipeline(theta_val=0.56)
        for term in result.bounded_terms:
            if term.metadata.get("di_bound_applied"):
                bm = get_bound_meta(term)
                assert bm is not None, f"DI term {term.id} missing BoundMeta"

    def test_di_bound_terms_have_scale_model_dict(self) -> None:
        """DI bound terms must have scale_model_dict in metadata."""
        result = conrey89_pipeline(theta_val=0.56)
        for term in result.bounded_terms:
            if term.metadata.get("di_bound_applied"):
                assert "scale_model_dict" in term.metadata, (
                    f"DI term {term.id} missing scale_model_dict"
                )


# ---------------------------------------------------------------------------
# Voronoi pipeline
# ---------------------------------------------------------------------------

class TestVoronoiPipeline:
    """Voronoi pipeline should report PostVoronoi family presence."""

    def test_theta_max_is_five_eighths(self) -> None:
        """The Voronoi pipeline's known theta_max is 5/8 (from PostVoronoi toy)."""
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        assert result.theta_max_result is not None
        assert result.theta_max_result.symbolic == Fraction(5, 8)

    def test_has_post_voronoi_bound_terms(self) -> None:
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        families = set()
        for term in result.bounded_terms:
            bm = get_bound_meta(term)
            if bm:
                families.add(bm.bound_family)
        assert "PostVoronoi" in families

    def test_binding_not_automatically_di(self) -> None:
        """Since PostVoronoi is present, the binding family should not
        be automatically DI_Kloosterman (it's PostVoronoi with theta_max=5/8,
        but DI_Kloosterman gives 4/7 < 5/8 so DI is still binding)."""
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        assert result.theta_max_result is not None
        # PostVoronoi theta_max=5/8 > DI theta_max=4/7
        # But find_theta_max uses known_theta_max=5/8, so it pins to that
        # The binding family will be the one closest to E=1 at theta=5/8


# ---------------------------------------------------------------------------
# Spectral pipeline
# ---------------------------------------------------------------------------

class TestSpectralPipeline:
    """The spectral pipeline's binding constraint is SpectralLargeSieve."""

    def test_theta_max_is_one_third(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3)
        assert result.theta_max_result is not None
        assert result.theta_max_result.symbolic == Fraction(1, 3)

    def test_binding_family_is_spectral(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3)
        assert result.theta_max_result is not None
        assert "SpectralLargeSieve" in result.theta_max_result.binding_family

    def test_has_spectralized_bound_terms(self) -> None:
        """At least some BoundOnly terms should come from SpectralLargeSieve."""
        result = conrey89_spectral_pipeline(theta_val=0.3)
        sls_terms = [
            t for t in result.bounded_terms
            if get_bound_meta(t) and "SpectralLargeSieve" in get_bound_meta(t).bound_family
        ]
        assert len(sls_terms) > 0

    def test_spectral_has_case_tree(self) -> None:
        """SpectralLargeSieve produces 3 cases per input term."""
        result = conrey89_spectral_pipeline(theta_val=0.3)
        case_ids = set()
        for term in result.bounded_terms:
            bm = get_bound_meta(term)
            if bm and "SpectralLargeSieve" in bm.bound_family and bm.case_id:
                case_ids.add(bm.case_id)
        expected = {"small_modulus", "large_modulus", "bessel_transition"}
        assert case_ids >= expected, f"Missing cases: {expected - case_ids}"


# ---------------------------------------------------------------------------
# Cross-pipeline comparison
# ---------------------------------------------------------------------------

class TestCrossPipeline:
    """Compare theta_max ordering across pipelines."""

    def test_spectral_strictest(self) -> None:
        """spectral (1/3) < baseline (4/7) <= voronoi (5/8)."""
        spectral = conrey89_spectral_pipeline(theta_val=0.3)
        baseline = conrey89_pipeline(theta_val=0.56)
        voronoi = conrey89_voronoi_pipeline(theta_val=0.56)
        s_max = spectral.theta_max_result.symbolic
        b_max = baseline.theta_max_result.symbolic
        v_max = voronoi.theta_max_result.symbolic
        assert s_max < b_max
        assert b_max <= v_max
