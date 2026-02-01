"""Tests for ExponentModel, ParametricSolver, BoundStrategy, and upgraded diagnostics."""

from __future__ import annotations

import pytest

from mollifier_theta.analysis.exponent_model import (
    DIExponentConstraintModel,
    ExponentConstraint,
    ParametricSolver,
    ParametricSolverResult,
    sweep_theta_for_constraints,
)
from mollifier_theta.analysis.slack import compare_pipelines, diagnose_pipeline
from mollifier_theta.lemmas.bound_strategy import (
    BoundStrategyRegistry,
    PostVoronoiBound,
    create_default_registry,
)


class TestExponentConstraint:
    def test_evaluate(self) -> None:
        c = ExponentConstraint(name="test", expression_str="7*theta/4")
        assert abs(c.evaluate(0.56) - 0.98) < 1e-10

    def test_is_satisfied_below(self) -> None:
        c = ExponentConstraint(name="test", expression_str="7*theta/4")
        assert c.is_satisfied(0.56)

    def test_is_not_satisfied_above(self) -> None:
        c = ExponentConstraint(name="test", expression_str="7*theta/4")
        assert not c.is_satisfied(0.58)

    def test_solve_theta_max(self) -> None:
        c = ExponentConstraint(name="test", expression_str="7*theta/4")
        assert abs(c.solve_theta_max() - 4 / 7) < 1e-10


class TestDIExponentConstraintModel:
    def test_has_constraints(self) -> None:
        model = DIExponentConstraintModel()
        assert len(model.constraints()) > 0

    def test_theta_max(self) -> None:
        model = DIExponentConstraintModel()
        assert abs(model.theta_max() - 4 / 7) < 1e-10

    def test_name(self) -> None:
        model = DIExponentConstraintModel()
        assert model.name == "DI_Kloosterman"


class TestParametricSolver:
    def test_solve_single(self) -> None:
        model = DIExponentConstraintModel()
        solver = ParametricSolver()
        result = solver.solve_single(model)
        assert abs(result.theta_max - 4 / 7) < 1e-10
        assert result.binding_constraint == "di_bilinear"

    def test_solve_all(self) -> None:
        solver = ParametricSolver(models=[DIExponentConstraintModel()])
        results = solver.solve_all()
        assert len(results) == 1

    def test_compare_single_model(self) -> None:
        solver = ParametricSolver(models=[DIExponentConstraintModel()])
        comparison = solver.compare()
        assert comparison["best_model"] == "DI_Kloosterman"
        assert abs(comparison["theta_max"] - 4 / 7) < 1e-10

    def test_compare_empty(self) -> None:
        solver = ParametricSolver()
        comparison = solver.compare()
        assert comparison["best_model"] is None


class TestSweep:
    def test_sweep_basic(self) -> None:
        c = ExponentConstraint(name="di", expression_str="7*theta/4")
        results = sweep_theta_for_constraints([c], theta_min=0.5, theta_max=0.6, step=0.01)
        assert len(results) > 0
        assert all("di" in r for r in results)

    def test_sweep_boundary(self) -> None:
        c = ExponentConstraint(name="di", expression_str="7*theta/4")
        results = sweep_theta_for_constraints([c], theta_min=0.5, theta_max=0.6, step=0.01)
        # Should be admissible below 4/7 and inadmissible above
        admissible = [r for r in results if r["admissible"]]
        inadmissible = [r for r in results if not r["admissible"]]
        assert len(admissible) > 0
        assert len(inadmissible) > 0


class TestBoundStrategyRegistry:
    def test_register_and_get(self) -> None:
        registry = BoundStrategyRegistry()
        strategy = PostVoronoiBound()
        registry.register(strategy)
        assert registry.get("PostVoronoi") is not None

    def test_list_strategies(self) -> None:
        registry = create_default_registry()
        names = registry.list_strategies()
        assert "PostVoronoi" in names

    def test_post_voronoi_constraints(self) -> None:
        bound = PostVoronoiBound()
        constraints = bound.constraints()
        assert len(constraints) > 0
        # E(theta) = 2*theta - 1/4 = 1 => theta = 5/8
        assert abs(constraints[0].solve_theta_max() - 5 / 8) < 1e-10

    def test_post_voronoi_constraint_family(self) -> None:
        bound = PostVoronoiBound()
        constraints = bound.constraints()
        assert constraints[0].bound_family == "PostVoronoi"


class TestModelAwareSlack:
    def test_bound_family_populated(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        for ts in result.term_slacks:
            assert ts.bound_family != ""

    def test_pipeline_stage_populated(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        for ts in result.term_slacks:
            assert ts.pipeline_stage != ""

    def test_group_by_family(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        groups = result.group_by_family()
        assert len(groups) > 0
        # Should have DI_Kloosterman family
        assert "DI_Kloosterman" in groups or "Trivial" in groups

    def test_group_by_stage(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        groups = result.group_by_stage()
        assert len(groups) > 0


class TestComparePipelines:
    def test_comparison_runs(self) -> None:
        comparison = compare_pipelines(theta_val=0.56)
        assert comparison["theta_val"] == 0.56
        assert "baseline" in comparison
        assert "voronoi" in comparison

    def test_baseline_theta_max(self) -> None:
        comparison = compare_pipelines(theta_val=0.56)
        assert abs(comparison["baseline"]["theta_max"] - 4 / 7) < 1e-10

    def test_voronoi_theta_max(self) -> None:
        comparison = compare_pipelines(theta_val=0.56)
        # PostVoronoi bound is binding: theta_max = 5/8
        assert abs(comparison["voronoi"]["theta_max"] - 5 / 8) < 1e-10

    def test_both_have_families(self) -> None:
        comparison = compare_pipelines(theta_val=0.56)
        assert len(comparison["baseline"]["families"]) > 0
        assert len(comparison["voronoi"]["families"]) > 0

    def test_voronoi_has_post_voronoi_family(self) -> None:
        comparison = compare_pipelines(theta_val=0.56)
        assert "PostVoronoi" in comparison["voronoi"]["families"]

    def test_different_constraint_families(self) -> None:
        comparison = compare_pipelines(theta_val=0.56)
        baseline_families = set(comparison["baseline"]["families"])
        voronoi_families = set(comparison["voronoi"]["families"])
        # Voronoi should have PostVoronoi family that baseline doesn't
        assert "PostVoronoi" in voronoi_families - baseline_families
