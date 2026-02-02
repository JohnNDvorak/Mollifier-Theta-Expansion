"""Math contract tests for LengthAwareDIModel.

These tests lock down *model semantics* and *solver correctness* without
freezing fragile SymPy string forms.  They prevent "silent optimism" where
a structurally incorrect DI formula appears to improve theta simply because
it omitted some pipeline loss.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.analysis.length_aware_di import LengthAwareDIModel
from mollifier_theta.core.scale_model import ScaleModel


FOUR_SEVENTHS = float(Fraction(4, 7))
TOL = 1e-6


class TestRawDIAtConreyBarrier:
    """The raw DI inequality must be strictly less than 1 at theta=4/7.

    This is the key finding: the 7*theta/4 barrier arises from full
    second-moment bookkeeping, not the raw DI inequality alone.
    """

    def test_symmetric_raw_di_strictly_less_than_one(self) -> None:
        model = LengthAwareDIModel.symmetric()
        E_raw = model.evaluate_error(FOUR_SEVENTHS)
        assert E_raw < 1.0 - TOL, (
            f"Raw DI at theta=4/7 should be strictly < 1, got {E_raw}"
        )

    def test_voronoi_dual_raw_di_at_four_sevenths(self) -> None:
        """Voronoi dual should also be evaluable at 4/7."""
        model = LengthAwareDIModel.voronoi_dual()
        E_raw = model.evaluate_error(FOUR_SEVENTHS)
        # Just verify it's finite and evaluable
        assert E_raw == E_raw  # not NaN
        assert E_raw >= 0


class TestSymmetricModelYieldsFourSevenths:
    """Symmetric model's overall theta_max must equal 4/7 because it
    includes the pipeline constraint (7*theta/4), not just the raw DI.
    """

    def test_symmetric_overall_theta_max_is_four_sevenths(self) -> None:
        model = LengthAwareDIModel.symmetric()
        constraints = model.constraints()
        # Some constraints (raw DI) may have no root in (0,1) — only
        # the solvable ones participate.  The binding is 7*theta/4.
        theta_maxes = []
        for c in constraints:
            try:
                theta_maxes.append(c.solve_theta_max())
            except ValueError:
                pass  # No root in (0,1) → not binding
        assert len(theta_maxes) >= 1
        overall = min(theta_maxes)
        assert abs(overall - FOUR_SEVENTHS) < TOL

    def test_binding_constraint_is_7theta4(self) -> None:
        """The 7*theta/4 constraint must be the binding one (smallest theta_max).

        The raw DI constraint has no root in (0,1) so the only solvable
        constraint is 7*theta/4.
        """
        model = LengthAwareDIModel.symmetric()
        constraints = model.constraints()

        solvable = []
        for c in constraints:
            try:
                solvable.append((c.solve_theta_max(), c))
            except ValueError:
                pass

        assert len(solvable) >= 1
        _, binding = min(solvable, key=lambda x: x[0])
        assert binding.name == "di_conrey_7theta4"
        assert binding.bound_family == "DI_Kloosterman"


class TestVoronoiDualExcludes7Theta4:
    """The voronoi_dual model must NOT include the known 7*theta/4 constraint.

    This prevents accidentally pinning future variants to the old barrier.
    """

    def test_no_7theta4_expression(self) -> None:
        model = LengthAwareDIModel.voronoi_dual()
        for c in model.constraints():
            assert "7*theta/4" not in c.expression_str, (
                f"Voronoi dual should not contain 7*theta/4, found in {c.name}"
            )

    def test_no_di_kloosterman_family(self) -> None:
        model = LengthAwareDIModel.voronoi_dual()
        for c in model.constraints():
            assert c.bound_family != "DI_Kloosterman", (
                f"Voronoi dual should not have DI_Kloosterman family in {c.name}"
            )


class TestMaxPiecewiseBranchCorrectness:
    """Verify that Max(...) expressions are handled correctly by the solver.

    Uses a synthetic model to test branch switching.
    """

    def test_synthetic_max_root(self) -> None:
        """Max(theta + 1/10, 2*theta - 1/5) / 2 = 1.

        Branch A: theta + 1/10  → dominates when theta < 0.3
        Branch B: 2*theta - 1/5  → dominates when theta >= 0.3
        At crossover theta=0.3: both = 0.4

        Solving Max/2 = 1:
        Branch B dominates at the root: (2*theta - 1/5)/2 = 1
          → 2*theta - 1/5 = 2 → theta = 1.1 (outside (0,1))
        Branch A: (theta + 1/10)/2 = 1 → theta = 1.9 (outside (0,1))

        So no root in (0,1) — theta_max should be 1.0.
        """
        sm = ScaleModel(T_exponent="Max(theta + 1/10, 2*theta - 1/5) / 2")
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        # No root in (0,1) means the expression is always < 1 in that range
        if roots:
            # If solver finds a root, verify it's actually correct
            for r in roots:
                val = ScaleModel.evaluate_expr(
                    "Max(theta + 1/10, 2*theta - 1/5) / 2", r,
                )
                assert abs(val - 1.0) < 1e-6
        else:
            # Correct: no root in (0,1)
            val_at_09 = ScaleModel.evaluate_expr(
                "Max(theta + 1/10, 2*theta - 1/5) / 2", 0.9,
            )
            assert val_at_09 < 1.0

    def test_synthetic_max_with_root_in_unit(self) -> None:
        """Max(3*theta, 2*theta + 1/4) / 2 = 1.

        Branch A: 3*theta  → dominates when theta > 0.25
        Branch B: 2*theta + 1/4 → dominates when theta <= 0.25

        Solving:
        Branch A: 3*theta/2 = 1 → theta = 2/3
        Branch B: (2*theta + 1/4)/2 = 1 → theta = 7/8

        At theta=2/3 > 0.25, Branch A dominates → root at 2/3.
        """
        sm = ScaleModel(T_exponent="Max(3*theta, 2*theta + Rational(1,4)) / 2")
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        assert len(roots) >= 1
        assert abs(roots[0] - 2.0 / 3.0) < 1e-6

    def test_branch_evaluation_correct_below_crossover(self) -> None:
        """Below the crossover, the smaller branch should be the max."""
        model = LengthAwareDIModel.symmetric()
        # At theta=0.2 (small): sub_A = 0.2 + 0.8 = 1.0
        #                        sub_B = 0.4 + 0.4 = 0.8
        # Max = 1.0
        assert abs(model.sub_A_at(0.2) - 1.0) < TOL
        assert abs(model.sub_B_at(0.2) - 0.8) < TOL
        # Error = max(1.0, 0.8)/2 = 0.5
        assert abs(model.evaluate_error(0.2) - 0.5) < TOL

    def test_branch_evaluation_correct_above_crossover(self) -> None:
        """Above the crossover, sub_B dominates."""
        model = LengthAwareDIModel.symmetric()
        # sub_A = 1.0 always for symmetric, sub_B = 3*theta/2 + 1/2
        # Crossover at 3*theta/2 + 1/2 = 1 → theta = 1/3
        # At theta=0.5: sub_A = 1.0, sub_B = 0.75 + 0.5 = 1.25
        assert abs(model.sub_A_at(0.5) - 1.0) < TOL
        assert abs(model.sub_B_at(0.5) - 1.25) < TOL
        # Error = max(1.0, 1.25)/2 = 0.625
        assert abs(model.evaluate_error(0.5) - 0.625) < TOL


class TestVoronoiDualModelDifference:
    """The voronoi_dual model must produce a *different* theta_max from symmetric."""

    def test_different_theta_max(self) -> None:
        sym = LengthAwareDIModel.symmetric()
        dual = LengthAwareDIModel.voronoi_dual()
        sym_tm = sym.theta_max()
        dual_tm = dual.theta_max()
        assert abs(sym_tm - dual_tm) > TOL, (
            f"Symmetric ({sym_tm}) and dual ({dual_tm}) should differ"
        )

    def test_dual_sub_exponents_at_multiple_theta(self) -> None:
        """Verify the dual model's sub-exponents change with theta."""
        dual = LengthAwareDIModel.voronoi_dual()
        # At theta=0.3: alpha=0.3, beta=2-0.9=1.1, gamma=0.7
        # sub_A = (0.3+1.1)/2 + 0.7 = 0.7 + 0.7 = 1.4
        # sub_B = 0.3 + 1.1 + 0.35 = 1.75
        assert abs(dual.sub_A_at(0.3) - 1.4) < TOL
        assert abs(dual.sub_B_at(0.3) - 1.75) < TOL

    def test_dual_evaluates_correctly_at_boundary(self) -> None:
        """At theta=2/3, the dual beta = 2-3*(2/3) = 0. Sum lengths degenerate."""
        dual = LengthAwareDIModel.voronoi_dual()
        theta_val = 2.0 / 3.0
        # alpha=2/3, beta=0, gamma=1/3
        # sub_A = (2/3)/2 + 1/3 = 1/3 + 1/3 = 2/3
        # sub_B = 2/3 + 0 + 1/6 = 5/6
        assert abs(dual.sub_A_at(theta_val) - 2.0 / 3.0) < TOL
        assert abs(dual.sub_B_at(theta_val) - 5.0 / 6.0) < TOL
