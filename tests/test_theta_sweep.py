"""Tests for theta sweep."""

from __future__ import annotations

import pytest

from mollifier_theta.pipelines.theta_sweep import theta_sweep


class TestThetaSweep:
    def test_sweep_produces_results(self) -> None:
        results = theta_sweep(theta_min=0.50, theta_max=0.60, step=0.05)
        assert len(results) >= 3

    def test_sweep_has_correct_fields(self) -> None:
        results = theta_sweep(theta_min=0.55, theta_max=0.60, step=0.05)
        for r in results:
            assert "theta" in r
            assert "admissible" in r

    def test_monotone_pass_fail_boundary(self) -> None:
        """Once admissible goes False, it should stay False (monotone boundary)."""
        results = theta_sweep(theta_min=0.50, theta_max=0.62, step=0.02)
        seen_fail = False
        for r in results:
            if not r["admissible"]:
                seen_fail = True
            if seen_fail:
                assert not r["admissible"], (
                    f"theta={r['theta']} is admissible after a failure — "
                    f"boundary is not monotone"
                )

    def test_boundary_near_four_sevenths(self) -> None:
        results = theta_sweep(theta_min=0.55, theta_max=0.60, step=0.01)
        pass_thetas = [r["theta"] for r in results if r["admissible"]]
        fail_thetas = [r["theta"] for r in results if not r["admissible"]]
        assert pass_thetas, "No passing thetas found"
        assert fail_thetas, "No failing thetas found"
        boundary = (max(pass_thetas) + min(fail_thetas)) / 2
        assert abs(boundary - 4 / 7) < 0.02

    def test_sweep_theta_max_consistent(self) -> None:
        results = theta_sweep(theta_min=0.55, theta_max=0.58, step=0.01)
        for r in results:
            if r.get("theta_max") is not None:
                assert abs(r["theta_max"] - 4 / 7) < 0.001
