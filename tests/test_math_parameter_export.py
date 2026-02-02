"""Tests for the math-parameter ledger export."""

from __future__ import annotations

import json

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
from mollifier_theta.core.scale_model import ScaleModel, theta
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    VoronoiKind,
    VoronoiMeta,
    _BOUND_KEY,
    _VORONOI_KEY,
)
from mollifier_theta.reports.math_parameter_export import (
    MathParameterRecord,
    export_math_parameters,
    export_math_parameters_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_di_bound_term() -> Term:
    """A DI Kloosterman BoundOnly term."""
    scale = ScaleModel(T_exponent=7 * theta / 4, description="DI bound")
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="DI bound term",
        status=TermStatus.BOUND_ONLY,
        lemma_citation="DI 1982",
        kernel_state=KernelState.KLOOSTERMANIZED,
        scale_model=scale.to_str(),
        metadata={
            "error_exponent": "7*theta/4",
            "scale_model_dict": scale.to_dict(),
            _BOUND_KEY: BoundMeta(
                strategy="DI_Kloosterman",
                error_exponent="7*theta/4",
                citation="DI 1982",
                bound_family="DI_Kloosterman",
            ).model_dump(),
        },
    )


def _make_voronoi_bound_term() -> Term:
    """A Voronoi-dual BoundOnly term."""
    scale = ScaleModel(T_exponent=2 * theta - 1, description="dual bound")
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="Voronoi dual bound",
        status=TermStatus.BOUND_ONLY,
        lemma_citation="length-aware DI",
        kernel_state=KernelState.KLOOSTERMANIZED,
        scale_model=scale.to_str(),
        metadata={
            "error_exponent": "2*theta - 1",
            "di_model_label": "voronoi_dual",
            "scale_model_dict": scale.to_dict(),
            _BOUND_KEY: BoundMeta(
                strategy="LengthAwareDI",
                error_exponent="2*theta - 1",
                citation="length-aware DI",
                bound_family="LengthAwareDI_voronoi_dual",
            ).model_dump(),
            _VORONOI_KEY: VoronoiMeta(
                applied=True,
                target_variable="n",
                dual_variable="n_star",
                dual_length="T^(2-3*theta)",
                kind=VoronoiKind.STRUCTURAL_ONLY,
            ).model_dump(),
        },
    )


def _make_active_term() -> Term:
    """An active term (not BoundOnly) — should be skipped."""
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="active term",
        status=TermStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportMathParameters:

    def test_bound_only_terms_produce_records(self) -> None:
        terms = [_make_di_bound_term()]
        records = export_math_parameters(terms)
        assert len(records) == 1
        assert records[0].bound_family == "DI_Kloosterman"
        assert records[0].error_exponent == "7*theta/4"

    def test_active_terms_skipped(self) -> None:
        terms = [_make_active_term(), _make_di_bound_term()]
        records = export_math_parameters(terms)
        assert len(records) == 1

    def test_defaults_for_symmetric_term(self) -> None:
        terms = [_make_di_bound_term()]
        records = export_math_parameters(terms)
        r = records[0]
        assert r.m_length_exponent == "theta"
        assert r.n_length_exponent == "theta"
        assert r.modulus_exponent == "1-theta"

    def test_voronoi_term_has_dual_length(self) -> None:
        terms = [_make_voronoi_bound_term()]
        records = export_math_parameters(terms)
        r = records[0]
        assert "2-3*theta" in r.n_length_exponent or "voronoi" in r.n_length_exponent.lower()

    def test_citation_from_bound_meta(self) -> None:
        terms = [_make_di_bound_term()]
        records = export_math_parameters(terms)
        assert records[0].citation == "DI 1982"

    def test_json_serializable(self) -> None:
        terms = [_make_di_bound_term(), _make_voronoi_bound_term()]
        result = export_math_parameters_json(terms)
        # Must be JSON-serializable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert len(parsed) == 2

    def test_record_to_dict(self) -> None:
        terms = [_make_di_bound_term()]
        records = export_math_parameters(terms)
        d = records[0].to_dict()
        assert "term_id" in d
        assert "bound_family" in d
        assert "error_exponent" in d

    def test_empty_input(self) -> None:
        assert export_math_parameters([]) == []
        assert export_math_parameters_json([]) == []
