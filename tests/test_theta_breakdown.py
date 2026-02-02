"""Tests for the theta breakdown compute module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mollifier_theta.analysis.theta_breakdown import (
    FORMAT_VERSION,
    ThetaBreakdown,
    TermBreakdown,
    compute_theta_breakdown,
)
from mollifier_theta.reports.envelope_loader import MathParamsEnvelope
from mollifier_theta.reports.math_parameter_export import MathParameterRecord

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ── Helpers ────────────────────────────────────────────────────────


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, indent=2)


def _make_di_record() -> MathParameterRecord:
    return MathParameterRecord(
        term_id="test_di_001",
        bound_family="DI_Kloosterman",
        case_id="symmetric",
        error_exponent="7*theta/4",
        m_length_exponent="theta",
        n_length_exponent="theta",
        modulus_exponent="1-theta",
        kernel_family_tags=[],
        citation="DI 1982",
    )


def _make_voronoi_record() -> MathParameterRecord:
    return MathParameterRecord(
        term_id="test_voronoi_001",
        bound_family="LengthAwareDI_voronoi_dual",
        case_id="dual",
        error_exponent="2*theta - 1",
        m_length_exponent="theta",
        n_length_exponent="T^(2-3*theta)",
        modulus_exponent="1-theta",
        kernel_family_tags=[],
        citation="length-aware DI",
    )


def _make_envelope(records: list[MathParameterRecord]) -> MathParamsEnvelope:
    """Build a MathParamsEnvelope from records, pre-sorted."""
    sorted_records = sorted(
        records, key=lambda r: (r.bound_family, r.case_id, r.term_id)
    )
    return MathParamsEnvelope(
        format_version="1.0",
        record_count=len(sorted_records),
        records=tuple(sorted_records),
    )


def _two_term_envelope() -> MathParamsEnvelope:
    return _make_envelope([_make_di_record(), _make_voronoi_record()])


# ── Basic functionality ───────────────────────────────────────────


class TestComputeThetaBreakdown:
    def test_two_term_breakdown(self) -> None:
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 4.0 / 7.0)
        assert len(result.terms) == 2
        assert result.theta_val == 4.0 / 7.0

    def test_binding_term_is_di(self) -> None:
        """DI_Kloosterman (7*theta/4) has theta_max=4/7, which is tighter
        than voronoi dual (2*theta-1) with theta_max=1."""
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 4.0 / 7.0)
        assert result.binding_term_id == "test_di_001"
        assert result.binding_family == "DI_Kloosterman"

    def test_theta_max_is_four_sevenths(self) -> None:
        """Overall theta_max should be 4/7 from the DI constraint."""
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 4.0 / 7.0)
        assert abs(result.theta_max - 4.0 / 7.0) < 1e-8

    def test_di_slack_near_zero_at_theta_max(self) -> None:
        """At theta=4/7, the DI constraint is exactly binding (E=1, slack=0)."""
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 4.0 / 7.0)
        di_terms = [t for t in result.terms if t.term_id == "test_di_001"]
        assert len(di_terms) == 1
        assert abs(di_terms[0].slack) < 1e-8
        assert abs(di_terms[0].E_val - 1.0) < 1e-8


class TestDeterminism:
    def test_repeated_calls_identical(self) -> None:
        env = _two_term_envelope()
        r1 = compute_theta_breakdown(env, 0.5)
        r2 = compute_theta_breakdown(env, 0.5)
        assert _canonical(r1.to_envelope()) == _canonical(r2.to_envelope())

    def test_json_round_trip_lossless(self) -> None:
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 0.5)
        envelope = result.to_envelope()
        json_str = json.dumps(envelope, sort_keys=True, indent=2)
        parsed = json.loads(json_str)
        assert parsed == envelope


class TestEnvelopeFormat:
    def test_format_version_present(self) -> None:
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 0.5)
        envelope = result.to_envelope()
        assert envelope["format_version"] == FORMAT_VERSION

    def test_record_count_matches(self) -> None:
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 0.5)
        envelope = result.to_envelope()
        assert envelope["record_count"] == len(envelope["records"])

    def test_records_sorted_by_slack_ascending(self) -> None:
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 4.0 / 7.0)
        envelope = result.to_envelope()
        slacks = [r["slack"] for r in envelope["records"]]
        assert slacks == sorted(slacks)

    def test_to_json_is_canonical(self) -> None:
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 0.5)
        json_str = result.to_json()
        envelope = result.to_envelope()
        expected = json.dumps(envelope, sort_keys=True, indent=2)
        assert json_str == expected


class TestEdgeCases:
    def test_single_term(self) -> None:
        env = _make_envelope([_make_di_record()])
        result = compute_theta_breakdown(env, 0.5)
        assert len(result.terms) == 1
        assert result.binding_term_id == "test_di_001"

    def test_empty_envelope(self) -> None:
        env = MathParamsEnvelope(
            format_version="1.0", record_count=0, records=()
        )
        result = compute_theta_breakdown(env, 0.5)
        assert len(result.terms) == 0
        assert result.theta_max == 1.0
        assert result.binding_term_id == ""

    def test_is_binding_flag_set_correctly(self) -> None:
        env = _two_term_envelope()
        result = compute_theta_breakdown(env, 4.0 / 7.0)
        binding = [t for t in result.terms if t.is_binding]
        non_binding = [t for t in result.terms if not t.is_binding]
        assert len(binding) == 1
        assert binding[0].term_id == "test_di_001"
        assert len(non_binding) == 1
