"""Tests for the envelope loader / validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mollifier_theta.reports.envelope_loader import (
    EnvelopeValidationError,
    MathParamsEnvelope,
    OverheadEnvelope,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ── Helpers ────────────────────────────────────────────────────────


def _load_golden(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def _golden_math_params() -> dict:
    return _load_golden("golden_math_params_v1_0.json")


def _golden_overhead() -> dict:
    return _load_golden("golden_overhead_v1_0.json")


# ── MathParamsEnvelope tests ──────────────────────────────────────


class TestMathParamsEnvelopeLoad:
    def test_from_dict_golden(self) -> None:
        env = MathParamsEnvelope.from_dict(_golden_math_params())
        assert env.format_version == "1.0"
        assert env.record_count == 2
        assert len(env.records) == 2

    def test_from_json_golden(self) -> None:
        text = json.dumps(_golden_math_params())
        env = MathParamsEnvelope.from_json(text)
        assert env.record_count == 2

    def test_from_file_golden(self) -> None:
        env = MathParamsEnvelope.from_file(
            FIXTURE_DIR / "golden_math_params_v1_0.json"
        )
        assert env.format_version == "1.0"

    def test_wrong_version_fails(self) -> None:
        data = _golden_math_params()
        data["format_version"] = "2.0"
        with pytest.raises(EnvelopeValidationError, match="format_version"):
            MathParamsEnvelope.from_dict(data)

    def test_record_count_mismatch_fails(self) -> None:
        data = _golden_math_params()
        data["record_count"] = 999
        with pytest.raises(EnvelopeValidationError, match="record_count"):
            MathParamsEnvelope.from_dict(data)

    def test_missing_record_field_fails(self) -> None:
        data = _golden_math_params()
        del data["records"][0]["citation"]
        with pytest.raises(EnvelopeValidationError, match="missing fields"):
            MathParamsEnvelope.from_dict(data)

    def test_unsorted_records_fail(self) -> None:
        data = _golden_math_params()
        data["records"] = list(reversed(data["records"]))
        with pytest.raises(EnvelopeValidationError, match="sort order"):
            MathParamsEnvelope.from_dict(data)

    def test_missing_top_level_field_fails(self) -> None:
        data = _golden_math_params()
        del data["records"]
        with pytest.raises(EnvelopeValidationError, match="Missing top-level"):
            MathParamsEnvelope.from_dict(data)

    def test_invalid_json_fails(self) -> None:
        with pytest.raises(EnvelopeValidationError, match="Invalid JSON"):
            MathParamsEnvelope.from_json("not json")

    def test_records_are_frozen(self) -> None:
        env = MathParamsEnvelope.from_dict(_golden_math_params())
        assert isinstance(env.records, tuple)


# ── OverheadEnvelope tests ────────────────────────────────────────


class TestOverheadEnvelopeLoad:
    def test_from_dict_golden(self) -> None:
        env = OverheadEnvelope.from_dict(_golden_overhead())
        assert env.format_version == "1.0"
        assert env.record_count == 2
        assert abs(env.theta_val - 4.0 / 7.0) < 1e-10

    def test_from_json_golden(self) -> None:
        text = json.dumps(_golden_overhead())
        env = OverheadEnvelope.from_json(text)
        assert env.record_count == 2

    def test_from_file_golden(self) -> None:
        env = OverheadEnvelope.from_file(
            FIXTURE_DIR / "golden_overhead_v1_0.json"
        )
        assert env.format_version == "1.0"

    def test_wrong_version_fails(self) -> None:
        data = _golden_overhead()
        data["format_version"] = "0.9"
        with pytest.raises(EnvelopeValidationError, match="format_version"):
            OverheadEnvelope.from_dict(data)

    def test_record_count_mismatch_fails(self) -> None:
        data = _golden_overhead()
        data["record_count"] = 0
        with pytest.raises(EnvelopeValidationError, match="record_count"):
            OverheadEnvelope.from_dict(data)

    def test_missing_theta_val_fails(self) -> None:
        data = _golden_overhead()
        del data["theta_val"]
        with pytest.raises(EnvelopeValidationError, match="Missing top-level"):
            OverheadEnvelope.from_dict(data)

    def test_missing_record_field_fails(self) -> None:
        data = _golden_overhead()
        del data["records"][0]["overhead"]
        with pytest.raises(EnvelopeValidationError, match="missing fields"):
            OverheadEnvelope.from_dict(data)

    def test_unsorted_records_fail(self) -> None:
        data = _golden_overhead()
        data["records"] = list(reversed(data["records"]))
        with pytest.raises(EnvelopeValidationError, match="sort order"):
            OverheadEnvelope.from_dict(data)

    def test_records_are_frozen(self) -> None:
        env = OverheadEnvelope.from_dict(_golden_overhead())
        assert isinstance(env.records, tuple)

    def test_invalid_json_fails(self) -> None:
        with pytest.raises(EnvelopeValidationError, match="Invalid JSON"):
            OverheadEnvelope.from_json("{bad")
