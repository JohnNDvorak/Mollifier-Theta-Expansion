"""Tests for diagnose render / JSON serialization."""

from __future__ import annotations

import json

from mollifier_theta.analysis.slack import diagnose_pipeline
from mollifier_theta.analysis.what_if import what_if_analysis
from mollifier_theta.reports.render_diagnose import (
    slack_to_json,
    what_if_to_json,
)


class TestSlackJsonSerializable:
    def test_json_serializable(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        data = slack_to_json(result)
        # Must be JSON-serializable without error
        text = json.dumps(data, default=str)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_json_keys(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        data = slack_to_json(result)
        required_keys = {"theta_val", "theta_max", "headroom", "bottleneck", "term_slacks"}
        assert required_keys <= set(data.keys())

    def test_bottleneck_has_sub_exponents(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        data = slack_to_json(result)
        assert data["bottleneck"] is not None
        assert "sub_exponents" in data["bottleneck"]


class TestWhatIfJsonSerializable:
    def test_json_serializable(self) -> None:
        result = what_if_analysis("di_saving", "-theta/3")
        data = what_if_to_json(result)
        text = json.dumps(data, default=str)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_json_keys(self) -> None:
        result = what_if_analysis("di_saving", "-theta/3")
        data = what_if_to_json(result)
        required_keys = {
            "scenario", "old_theta_max", "new_theta_max",
            "improvement", "old_E_expr", "new_E_expr",
        }
        assert required_keys <= set(data.keys())
