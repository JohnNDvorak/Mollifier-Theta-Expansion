"""Tests for strategy-branch enumerator."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.analysis.strategy_enumerator import (
    EnumerationResult,
    StrategyMatch,
    enumerate_strategies,
)
from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    KuznetsovMeta,
    VoronoiKind,
    VoronoiMeta,
    _KUZNETSOV_KEY,
    _VORONOI_KEY,
)
from mollifier_theta.lemmas.bound_strategy import PostVoronoiBound
from mollifier_theta.lemmas.spectral_large_sieve import SpectralLargeSieveBound


def _make_spectralized_term() -> Term:
    """Active SPECTRALIZED term eligible for SpectralLargeSieve."""
    return Term(
        kind=TermKind.SPECTRAL,
        expression="spectral sum",
        variables=["m", "n", "c"],
        ranges=[
            Range(variable="m", lower="1", upper="N"),
            Range(variable="n", lower="1", upper="N"),
        ],
        kernels=[
            Kernel(name="SpectralKernel"),
            Kernel(name="KuznetsovKernel"),
        ],
        phases=[
            Phase(
                expression="spectral_expansion(lambda_f(m)*lambda_f(n), h(t_f))",
                depends_on=["m"],
                is_separable=True,
            ),
        ],
        kernel_state=KernelState.SPECTRALIZED,
        status=TermStatus.ACTIVE,
        metadata={
            "kloosterman_form": True,
            "voronoi_applied": True,
            _VORONOI_KEY: VoronoiMeta(
                applied=True, kind=VoronoiKind.FORMULA,
            ).model_dump(),
            _KUZNETSOV_KEY: KuznetsovMeta(applied=True).model_dump(),
        },
    )


def _make_kloosterman_structural_term() -> Term:
    """Active KLOOSTERMANIZED term eligible for PostVoronoiBound."""
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="kloosterman sum",
        variables=["m", "n", "c"],
        ranges=[
            Range(variable="m", lower="1", upper="N"),
            Range(variable="n", lower="1", upper="N"),
        ],
        kernels=[Kernel(name="KloostermanKernel")],
        phases=[Phase(expression="S(m,n;c)/c", depends_on=["m", "n", "c"])],
        kernel_state=KernelState.KLOOSTERMANIZED,
        status=TermStatus.ACTIVE,
        metadata={
            "kloosterman_form": True,
            "voronoi_applied": True,
            _VORONOI_KEY: VoronoiMeta(
                applied=True, kind=VoronoiKind.STRUCTURAL_ONLY,
            ).model_dump(),
        },
    )


class TestEnumerateStrategies:
    def test_sls_matches_spectralized(self) -> None:
        terms = [_make_spectralized_term()]
        sls = SpectralLargeSieveBound()
        result = enumerate_strategies(terms, [sls], known_theta_max=Fraction(1, 3))
        assert len(result.matches) == 1
        assert result.matches[0].strategy_name == "SpectralLargeSieve"
        assert result.matches[0].matched_term_ids == [terms[0].id]
        assert result.matches[0].bound_count == 3  # 3 cases

    def test_post_voronoi_matches_structural(self) -> None:
        terms = [_make_kloosterman_structural_term()]
        pv = PostVoronoiBound()
        result = enumerate_strategies(terms, [pv], known_theta_max=Fraction(5, 8))
        assert len(result.matches) == 1
        assert result.matches[0].strategy_name == "PostVoronoi"
        assert result.matches[0].bound_count == 1

    def test_sls_does_not_match_structural(self) -> None:
        terms = [_make_kloosterman_structural_term()]
        sls = SpectralLargeSieveBound()
        result = enumerate_strategies(terms, [sls])
        assert result.matches[0].matched_term_ids == []
        assert result.matches[0].bound_count == 0

    def test_multiple_strategies(self) -> None:
        terms = [_make_spectralized_term(), _make_kloosterman_structural_term()]
        sls = SpectralLargeSieveBound()
        pv = PostVoronoiBound()
        result = enumerate_strategies(
            terms, [sls, pv], known_theta_max=Fraction(1, 3),
        )
        assert len(result.matches) == 2
        sls_match = next(m for m in result.matches if m.strategy_name == "SpectralLargeSieve")
        pv_match = next(m for m in result.matches if m.strategy_name == "PostVoronoi")
        assert sls_match.bound_count == 3
        assert pv_match.bound_count == 1

    def test_best_strategy_identified(self) -> None:
        terms = [_make_spectralized_term()]
        sls = SpectralLargeSieveBound()
        result = enumerate_strategies(terms, [sls], known_theta_max=Fraction(1, 3))
        assert result.best_strategy == "SpectralLargeSieve"
        assert result.best_theta_max == Fraction(1, 3)

    def test_case_summary_in_branch(self) -> None:
        terms = [_make_spectralized_term()]
        sls = SpectralLargeSieveBound()
        result = enumerate_strategies(terms, [sls], known_theta_max=Fraction(1, 3))
        branch = result.branches[0]
        assert "SpectralLargeSieve:small_modulus" in branch.case_summary
        assert "SpectralLargeSieve:large_modulus" in branch.case_summary
        assert "SpectralLargeSieve:bessel_transition" in branch.case_summary

    def test_format_summary(self) -> None:
        terms = [_make_spectralized_term()]
        sls = SpectralLargeSieveBound()
        result = enumerate_strategies(terms, [sls], known_theta_max=Fraction(1, 3))
        summary = result.format_summary()
        assert "SpectralLargeSieve" in summary
        assert "1/3" in summary

    def test_empty_terms(self) -> None:
        result = enumerate_strategies([], [SpectralLargeSieveBound()])
        assert len(result.matches) == 1
        assert result.matches[0].bound_count == 0
        assert result.best_strategy == ""
