"""Golden envelope tests: byte-identical regression for v1.0 envelope contracts.

Compares live output from the envelope producers against committed golden
fixture files.  If the schema changes intentionally, regenerate the golden
files; any unintended drift is a test failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mollifier_theta.analysis.overhead_report import compute_overhead
from mollifier_theta.core.ir import (
    HistoryEntry,
    Term,
    TermKind,
    TermStatus,
    KernelState,
)
from mollifier_theta.core.scale_model import ScaleModel, theta
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    VoronoiKind,
    VoronoiMeta,
    _BOUND_KEY,
    _VORONOI_KEY,
)
from mollifier_theta.reports.math_parameter_export import export_math_parameters_envelope

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, indent=2)


def _load_golden(name: str) -> dict:
    path = FIXTURE_DIR / name
    return json.loads(path.read_text())


# ── Fixture terms (deterministic IDs) ─────────────────────────────


def _make_fixture_terms() -> list[Term]:
    """Build the canonical fixture terms matching the golden files."""
    di_scale = ScaleModel(T_exponent=7 * theta / 4, description="DI bound")
    di_term = Term(
        id="fixture_di_001",
        kind=TermKind.KLOOSTERMAN,
        expression="DI bound term",
        status=TermStatus.BOUND_ONLY,
        lemma_citation="Deshouillers-Iwaniec 1982/83, Theorem 12",
        kernel_state=KernelState.KLOOSTERMANIZED,
        scale_model=di_scale.to_str(),
        history=[
            HistoryEntry(
                transform="DeltaMethod",
                parent_ids=[],
                description="delta decomposition",
            ),
            HistoryEntry(
                transform="KloostermanForm",
                parent_ids=[],
                description="Kloosterman formation",
            ),
            HistoryEntry(
                transform="DIBound",
                parent_ids=[],
                description="DI bilinear bound",
            ),
        ],
        metadata={
            "error_exponent": "7*theta/4",
            "scale_model_dict": di_scale.to_dict(),
            _BOUND_KEY: BoundMeta(
                strategy="DI_Kloosterman",
                error_exponent="7*theta/4",
                citation="Deshouillers-Iwaniec 1982/83, Theorem 12",
                bound_family="DI_Kloosterman",
                case_id="symmetric",
            ).model_dump(),
        },
    )

    voronoi_scale = ScaleModel(
        T_exponent=2 * theta - 1, description="Voronoi dual bound"
    )
    voronoi_term = Term(
        id="fixture_voronoi_001",
        kind=TermKind.KLOOSTERMAN,
        expression="Voronoi dual bound term",
        status=TermStatus.BOUND_ONLY,
        lemma_citation="Deshouillers-Iwaniec 1982/83; length-aware parametric model",
        kernel_state=KernelState.KLOOSTERMANIZED,
        scale_model=voronoi_scale.to_str(),
        history=[
            HistoryEntry(
                transform="DeltaMethod",
                parent_ids=[],
                description="delta decomposition",
            ),
            HistoryEntry(
                transform="VoronoiSummation",
                parent_ids=[],
                description="Voronoi on n-variable",
            ),
            HistoryEntry(
                transform="KloostermanForm",
                parent_ids=[],
                description="Kloosterman formation",
            ),
            HistoryEntry(
                transform="LengthAwareDIBound",
                parent_ids=[],
                description="length-aware DI bound",
            ),
        ],
        metadata={
            "error_exponent": "2*theta - 1",
            "di_model_label": "voronoi_dual",
            "scale_model_dict": voronoi_scale.to_dict(),
            _BOUND_KEY: BoundMeta(
                strategy="LengthAwareDI",
                error_exponent="2*theta - 1",
                citation="Deshouillers-Iwaniec 1982/83; length-aware parametric model",
                bound_family="LengthAwareDI_voronoi_dual",
                case_id="dual",
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

    return [di_term, voronoi_term]


# ── Tests ──────────────────────────────────────────────────────────


class TestGoldenMathParamsEnvelope:
    def test_matches_golden_file(self) -> None:
        """Live math params envelope matches committed golden fixture."""
        terms = _make_fixture_terms()
        live = export_math_parameters_envelope(terms)
        golden = _load_golden("golden_math_params_v1_0.json")
        assert _canonical(live) == _canonical(golden)

    def test_golden_file_is_canonically_sorted(self) -> None:
        """Golden file uses sort_keys=True, indent=2."""
        golden = _load_golden("golden_math_params_v1_0.json")
        reserialized = _canonical(golden)
        original = (FIXTURE_DIR / "golden_math_params_v1_0.json").read_text().strip()
        assert reserialized == original

    def test_deterministic_repeated_calls(self) -> None:
        """Two calls with same input produce identical output."""
        terms = _make_fixture_terms()
        env1 = export_math_parameters_envelope(terms)
        env2 = export_math_parameters_envelope(terms)
        assert _canonical(env1) == _canonical(env2)


class TestGoldenOverheadEnvelope:
    def test_matches_golden_file(self) -> None:
        """Live overhead envelope matches committed golden fixture."""
        terms = _make_fixture_terms()
        theta_val = 4.0 / 7.0
        report = compute_overhead(terms, theta_val)
        live = report.to_envelope()
        golden = _load_golden("golden_overhead_v1_0.json")
        assert _canonical(live) == _canonical(golden)

    def test_golden_file_is_canonically_sorted(self) -> None:
        """Golden file uses sort_keys=True, indent=2."""
        golden = _load_golden("golden_overhead_v1_0.json")
        reserialized = _canonical(golden)
        original = (FIXTURE_DIR / "golden_overhead_v1_0.json").read_text().strip()
        assert reserialized == original

    def test_deterministic_repeated_calls(self) -> None:
        """Two calls with same input produce identical output."""
        terms = _make_fixture_terms()
        theta_val = 4.0 / 7.0
        r1 = compute_overhead(terms, theta_val)
        r2 = compute_overhead(terms, theta_val)
        assert _canonical(r1.to_envelope()) == _canonical(r2.to_envelope())
