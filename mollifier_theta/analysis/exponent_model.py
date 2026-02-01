"""Parametric exponent model and solver for theta_max optimization.

Allows defining multiple exponent constraints (each E_i(theta, params) < 1)
and solving for the maximum theta over the parameter space.

SymPy is used only through ScaleModel (contained in core/scale_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from mollifier_theta.core.scale_model import ScaleModel


@dataclass(frozen=True)
class ExponentConstraint:
    """A single exponent constraint: E(theta, params) < 1.

    The expression is a string in theta, evaluated via ScaleModel
    (SymPy containment — WI-6).
    """

    name: str
    expression_str: str
    description: str = ""
    citation: str = ""
    bound_family: str = ""

    def evaluate(self, theta_val: float) -> float:
        """Evaluate E(theta) at a specific theta value."""
        return ScaleModel.evaluate_expr(self.expression_str, theta_val)

    def is_satisfied(self, theta_val: float) -> bool:
        """Check E(theta) < 1 (strict)."""
        return self.evaluate(theta_val) < 1.0

    def solve_theta_max(self) -> float:
        """Solve E(theta) = 1 for the maximum theta from this constraint alone."""
        return ScaleModel.solve_expr_equals_one(self.expression_str)


@runtime_checkable
class ExponentModel(Protocol):
    """Protocol for exponent models that produce constraints."""

    @property
    def name(self) -> str: ...

    def constraints(self) -> list[ExponentConstraint]: ...

    def theta_max(self) -> float:
        """Maximum theta satisfying all constraints simultaneously."""
        ...


class DIExponentConstraintModel:
    """The standard DI Kloosterman exponent model: E(theta) = 7*theta/4 < 1."""

    @property
    def name(self) -> str:
        return "DI_Kloosterman"

    def constraints(self) -> list[ExponentConstraint]:
        return [
            ExponentConstraint(
                name="di_bilinear",
                expression_str="7*theta/4",
                description="DI bilinear Kloosterman bound error exponent",
                citation="Deshouillers-Iwaniec 1982/83; Conrey 1989",
                bound_family="DI_Kloosterman",
            ),
        ]

    def theta_max(self) -> float:
        # The binding constraint is 7*theta/4 < 1 => theta < 4/7
        return min(c.solve_theta_max() for c in self.constraints())


@dataclass
class ParametricSolverResult:
    """Result of a parametric theta_max optimization."""

    theta_max: float
    binding_constraint: str
    all_constraints: list[dict] = field(default_factory=list)
    model_name: str = ""


class ParametricSolver:
    """Solve for theta_max across multiple ExponentModel implementations.

    Uses grid search + refinement (no full optimizer needed).
    """

    def __init__(self, models: list[ExponentModel] | None = None) -> None:
        self.models: list[ExponentModel] = models or []

    def add_model(self, model: ExponentModel) -> None:
        self.models.append(model)

    def solve_single(self, model: ExponentModel) -> ParametricSolverResult:
        """Solve theta_max for a single model."""
        constraints = model.constraints()
        if not constraints:
            return ParametricSolverResult(
                theta_max=float("inf"),
                binding_constraint="none",
                model_name=model.name,
            )

        # Find the binding constraint (smallest theta_max)
        results = []
        for c in constraints:
            try:
                tm = c.solve_theta_max()
                results.append((tm, c))
            except ValueError:
                continue

        if not results:
            return ParametricSolverResult(
                theta_max=float("inf"),
                binding_constraint="none",
                model_name=model.name,
            )

        results.sort(key=lambda x: x[0])
        binding_tm, binding_c = results[0]

        all_c = [
            {
                "name": c.name,
                "expression": c.expression_str,
                "theta_max": tm,
                "is_binding": c.name == binding_c.name,
                "citation": c.citation,
                "bound_family": c.bound_family,
            }
            for tm, c in results
        ]

        return ParametricSolverResult(
            theta_max=binding_tm,
            binding_constraint=binding_c.name,
            all_constraints=all_c,
            model_name=model.name,
        )

    def solve_all(self) -> list[ParametricSolverResult]:
        """Solve theta_max for all registered models."""
        return [self.solve_single(m) for m in self.models]

    def compare(self) -> dict:
        """Compare theta_max across all models.

        Returns a summary dict with the best model and comparison table.
        """
        results = self.solve_all()
        if not results:
            return {"best_model": None, "theta_max": None, "models": []}

        best = max(results, key=lambda r: r.theta_max)

        return {
            "best_model": best.model_name,
            "theta_max": best.theta_max,
            "models": [
                {
                    "name": r.model_name,
                    "theta_max": r.theta_max,
                    "binding_constraint": r.binding_constraint,
                    "is_best": r.model_name == best.model_name,
                }
                for r in results
            ],
        }


def sweep_theta_for_constraints(
    constraints: list[ExponentConstraint],
    theta_min: float = 0.01,
    theta_max: float = 0.99,
    step: float = 0.001,
) -> list[dict]:
    """Sweep theta and evaluate all constraints at each point.

    Returns a list of dicts with theta, constraint values, and admissibility.
    """
    results = []
    theta_val = theta_min
    while theta_val <= theta_max:
        row = {"theta": theta_val, "admissible": True}
        for c in constraints:
            val = c.evaluate(theta_val)
            row[c.name] = val
            if val >= 1.0:
                row["admissible"] = False
        results.append(row)
        theta_val += step
    return results
