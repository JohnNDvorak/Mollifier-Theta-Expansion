"""Strict envelope loader for v1.0 export contracts.

Enforces the v1.0 envelope schema on load: version check, record count,
required fields, and canonical sort order.  These are the import-side
counterparts to the export functions in math_parameter_export.py and
overhead_report.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mollifier_theta.analysis.overhead_report import OverheadRecord
from mollifier_theta.reports.math_parameter_export import MathParameterRecord

SUPPORTED_VERSION = "1.0"


class EnvelopeValidationError(ValueError):
    """Raised when an envelope fails validation."""


# ── Math-params envelope ──────────────────────────────────────────


_MATH_PARAM_FIELDS = frozenset(MathParameterRecord.__dataclass_fields__)
_MATH_PARAM_SORT_KEY = ("bound_family", "case_id", "term_id")


def _validate_math_param_record(rec: dict, index: int) -> MathParameterRecord:
    """Validate and convert a single math-param record dict."""
    missing = _MATH_PARAM_FIELDS - rec.keys()
    if missing:
        raise EnvelopeValidationError(
            f"Record {index}: missing fields {sorted(missing)}"
        )
    return MathParameterRecord(**{k: rec[k] for k in MathParameterRecord.__dataclass_fields__})


@dataclass(frozen=True)
class MathParamsEnvelope:
    """Validated v1.0 math-parameters envelope."""

    format_version: str
    record_count: int
    records: tuple[MathParameterRecord, ...]

    @classmethod
    def from_dict(cls, data: dict) -> MathParamsEnvelope:
        """Parse and validate from a dict."""
        if not isinstance(data, dict):
            raise EnvelopeValidationError("Envelope must be a dict")

        # Version check
        version = data.get("format_version")
        if version != SUPPORTED_VERSION:
            raise EnvelopeValidationError(
                f"Unsupported format_version {version!r} (expected {SUPPORTED_VERSION!r})"
            )

        # Required top-level fields
        for key in ("format_version", "record_count", "records"):
            if key not in data:
                raise EnvelopeValidationError(f"Missing top-level field {key!r}")

        raw_records = data["records"]
        if not isinstance(raw_records, list):
            raise EnvelopeValidationError("'records' must be a list")

        record_count = data["record_count"]
        if not isinstance(record_count, int):
            raise EnvelopeValidationError("'record_count' must be an int")
        if record_count != len(raw_records):
            raise EnvelopeValidationError(
                f"record_count={record_count} but len(records)={len(raw_records)}"
            )

        # Validate each record
        records = tuple(
            _validate_math_param_record(r, i) for i, r in enumerate(raw_records)
        )

        # Canonical sort order check
        sort_keys = [
            tuple(getattr(r, k) for k in _MATH_PARAM_SORT_KEY) for r in records
        ]
        if sort_keys != sorted(sort_keys):
            raise EnvelopeValidationError(
                f"Records not in canonical sort order {_MATH_PARAM_SORT_KEY}"
            )

        return cls(
            format_version=version,
            record_count=record_count,
            records=records,
        )

    @classmethod
    def from_json(cls, text: str) -> MathParamsEnvelope:
        """Parse and validate from a JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EnvelopeValidationError(f"Invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: Path) -> MathParamsEnvelope:
        """Parse and validate from a JSON file."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EnvelopeValidationError(f"Cannot read file: {exc}") from exc
        return cls.from_json(text)


# ── Overhead-report envelope ──────────────────────────────────────


_OVERHEAD_FIELDS = frozenset(OverheadRecord.__dataclass_fields__)
_OVERHEAD_SORT_KEY = ("bound_family", "term_id")


def _validate_overhead_record(rec: dict, index: int) -> OverheadRecord:
    """Validate and convert a single overhead record dict."""
    missing = _OVERHEAD_FIELDS - rec.keys()
    if missing:
        raise EnvelopeValidationError(
            f"Record {index}: missing fields {sorted(missing)}"
        )
    return OverheadRecord(**{k: rec[k] for k in OverheadRecord.__dataclass_fields__})


@dataclass(frozen=True)
class OverheadEnvelope:
    """Validated v1.0 overhead-report envelope."""

    format_version: str
    theta_val: float
    record_count: int
    records: tuple[OverheadRecord, ...]

    @classmethod
    def from_dict(cls, data: dict) -> OverheadEnvelope:
        """Parse and validate from a dict."""
        if not isinstance(data, dict):
            raise EnvelopeValidationError("Envelope must be a dict")

        # Version check
        version = data.get("format_version")
        if version != SUPPORTED_VERSION:
            raise EnvelopeValidationError(
                f"Unsupported format_version {version!r} (expected {SUPPORTED_VERSION!r})"
            )

        # Required top-level fields
        for key in ("format_version", "theta_val", "record_count", "records"):
            if key not in data:
                raise EnvelopeValidationError(f"Missing top-level field {key!r}")

        theta_val = data["theta_val"]
        if not isinstance(theta_val, (int, float)):
            raise EnvelopeValidationError("'theta_val' must be a number")

        raw_records = data["records"]
        if not isinstance(raw_records, list):
            raise EnvelopeValidationError("'records' must be a list")

        record_count = data["record_count"]
        if not isinstance(record_count, int):
            raise EnvelopeValidationError("'record_count' must be an int")
        if record_count != len(raw_records):
            raise EnvelopeValidationError(
                f"record_count={record_count} but len(records)={len(raw_records)}"
            )

        # Validate each record
        records = tuple(
            _validate_overhead_record(r, i) for i, r in enumerate(raw_records)
        )

        # Canonical sort order check
        sort_keys = [
            tuple(getattr(r, k) for k in _OVERHEAD_SORT_KEY) for r in records
        ]
        if sort_keys != sorted(sort_keys):
            raise EnvelopeValidationError(
                f"Records not in canonical sort order {_OVERHEAD_SORT_KEY}"
            )

        return cls(
            format_version=version,
            theta_val=float(theta_val),
            record_count=record_count,
            records=records,
        )

    @classmethod
    def from_json(cls, text: str) -> OverheadEnvelope:
        """Parse and validate from a JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EnvelopeValidationError(f"Invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: Path) -> OverheadEnvelope:
        """Parse and validate from a JSON file."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EnvelopeValidationError(f"Cannot read file: {exc}") from exc
        return cls.from_json(text)
