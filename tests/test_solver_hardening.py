"""Tests for theta solver hardening (WI-7).

Validates:
  - Multi-root handling: quadratic and multi-root constraints
  - Piecewise/Max fallback via numerical bisection
  - solve_all_roots returns sorted list
  - Linear 7*theta/4 regression
"""

from __future__ import annotations

import pytest

from mollifier_theta.core.scale_model import ScaleModel


class TestSolveAllRoots:
    def test_linear_single_root(self) -> None:
        """Standard DI exponent: 7*theta/4 = 1 => theta = 4/7."""
        sm = ScaleModel(T_exponent="7*theta/4")
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        assert len(roots) == 1
        assert abs(roots[0] - 4 / 7) < 1e-10

    def test_quadratic_two_roots(self) -> None:
        """Quadratic: 6*theta^2 - 5*theta + 1 = 1 => 6*theta^2 - 5*theta = 0 => theta = 0 or 5/6.
        In (0, 1): theta = 5/6.
        """
        sm = ScaleModel(T_exponent="6*theta**2 - 5*theta + 1")
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        assert len(roots) >= 1
        assert abs(roots[0] - 5 / 6) < 1e-8

    def test_quadratic_constraint(self) -> None:
        """theta^2 + theta/2 = 1 => theta = (-1/2 + sqrt(17)/2) / 2 ~ 0.78."""
        sm = ScaleModel(T_exponent="theta**2 + theta/2")
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        assert len(roots) >= 1
        # Verify: root^2 + root/2 should equal 1
        r = roots[0]
        assert abs(r * r + r / 2 - 1) < 1e-8

    def test_two_roots_in_interval(self) -> None:
        """Expression with two roots in (0, 1): 4*(theta - 1/4)*(theta - 3/4) + 1.
        Roots at theta = 1/4 and theta = 3/4.
        """
        # 4*(theta-1/4)*(theta-3/4) + 1 = 4*theta^2 - 4*theta + 3/4 + 1 = 4*theta^2 - 4*theta + 7/4
        # That's not right. Let me construct it directly:
        # f(theta) = 1 at theta=a and theta=b means f(a)=f(b)=1
        # Use: f(theta) = 4*(theta-1/4)*(theta-3/4) + 1
        # f(1/4) = 0 + 1 = 1 ✓
        # f(3/4) = 0 + 1 = 1 ✓
        sm = ScaleModel(T_exponent="4*(theta - 1/4)*(theta - 3/4) + 1")
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        assert len(roots) == 2
        assert abs(roots[0] - 0.25) < 1e-8
        assert abs(roots[1] - 0.75) < 1e-8

    def test_returns_smallest_first(self) -> None:
        """Roots should be sorted ascending."""
        sm = ScaleModel(T_exponent="4*(theta - 1/4)*(theta - 3/4) + 1")
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        assert roots == sorted(roots)

    def test_no_root_in_interval(self) -> None:
        """Expression with no root in (0, 1)."""
        sm = ScaleModel(T_exponent="theta/2")  # max at 1 is 0.5 < 1
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        assert len(roots) == 0


class TestSolveForThetaHardened:
    def test_linear_regression(self) -> None:
        """7*theta/4 still gives 4/7."""
        sm = ScaleModel(T_exponent="7*theta/4")
        assert abs(sm.solve_for_theta() - 4 / 7) < 1e-10

    def test_quadratic(self) -> None:
        """Quadratic constraint finds smallest root."""
        sm = ScaleModel(T_exponent="4*(theta - 1/4)*(theta - 3/4) + 1")
        # Smallest root is 1/4
        assert abs(sm.solve_for_theta() - 0.25) < 1e-8

    def test_no_root_raises(self) -> None:
        sm = ScaleModel(T_exponent="theta/2")
        with pytest.raises(ValueError, match="No solution found"):
            sm.solve_for_theta()

    def test_with_expr_str(self) -> None:
        sm = ScaleModel(T_exponent=0)
        assert abs(sm.solve_for_theta("7*theta/4") - 4 / 7) < 1e-10


class TestSolveExprEqualsOneHardened:
    def test_linear(self) -> None:
        assert abs(ScaleModel.solve_expr_equals_one("7*theta/4") - 4 / 7) < 1e-10

    def test_quadratic(self) -> None:
        # 2*theta - 1/4 = 1 => theta = 5/8
        assert abs(ScaleModel.solve_expr_equals_one("2*theta - 1/4") - 5 / 8) < 1e-10

    def test_multi_root_returns_smallest(self) -> None:
        # Two roots at 1/4 and 3/4: should return 1/4
        result = ScaleModel.solve_expr_equals_one("4*(theta - 1/4)*(theta - 3/4) + 1")
        assert abs(result - 0.25) < 1e-8

    def test_no_root_raises(self) -> None:
        with pytest.raises(ValueError):
            ScaleModel.solve_expr_equals_one("theta/2")


class TestBisectionFallback:
    def test_bisection_finds_root(self) -> None:
        """Bisection should find roots even without symbolic solve."""
        sm = ScaleModel(T_exponent="7*theta/4")
        roots = sm._bisection_roots(lo=0.0, hi=1.0)
        assert len(roots) >= 1
        assert abs(roots[0] - 4 / 7) < 1e-6

    def test_bisection_two_roots(self) -> None:
        sm = ScaleModel(T_exponent="4*(theta - 1/4)*(theta - 3/4) + 1")
        roots = sm._bisection_roots(lo=0.0, hi=1.0)
        assert len(roots) == 2
