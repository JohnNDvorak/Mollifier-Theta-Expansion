"""Tests for what-if analysis."""

from __future__ import annotations

import pytest

from mollifier_theta.analysis.what_if import WhatIfResult, what_if_analysis


class TestWhatIfIdentity:
    def test_what_if_identity(self) -> None:
        """Replacing di_saving with the same value should give ~0 improvement."""
        result = what_if_analysis("di_saving", "-theta/4")
        assert abs(result.improvement) < 1e-10
        assert abs(result.new_theta_max - result.old_theta_max) < 1e-10


class TestWhatIfImproved:
    def test_what_if_improved(self) -> None:
        """di_saving = -theta/3 is a bigger saving, so theta_max should increase."""
        result = what_if_analysis("di_saving", "-theta/3")
        # new E = 7*theta/4 - (-theta/4) + (-theta/3) = 7*theta/4 + theta/4 - theta/3
        # = 2*theta - theta/3 = 5*theta/3
        # Solve 5*theta/3 = 1 -> theta = 3/5 = 0.6
        assert abs(result.new_theta_max - 0.6) < 1e-10
        assert result.improvement > 0
        assert abs(result.old_theta_max - 4 / 7) < 1e-10


class TestWhatIfWorse:
    def test_what_if_worse(self) -> None:
        """di_saving = 0 means no saving; theta_max should decrease."""
        result = what_if_analysis("di_saving", "0")
        # new E = 7*theta/4 - (-theta/4) + 0 = 7*theta/4 + theta/4 = 2*theta
        # Solve 2*theta = 1 -> theta = 1/2 = 0.5
        assert abs(result.new_theta_max - 0.5) < 1e-10
        assert result.improvement < 0


class TestWhatIfInvalidName:
    def test_what_if_invalid_name_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown sub-exponent"):
            what_if_analysis("nonexistent_sub", "-theta/3")


class TestWhatIfOtherSubExponents:
    def test_modulus_range(self) -> None:
        """Changing modulus_range should also work."""
        result = what_if_analysis("modulus_range", "1 - theta/2")
        assert isinstance(result, WhatIfResult)
        assert result.scenario.name == "modulus_range"
