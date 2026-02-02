"""Pluggable bound strategy interface and registry.

Allows comparing multiple bounding approaches (DI, spectral large sieve,
hybrid, post-Voronoi) on the same off-diagonal terms.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mollifier_theta.core.ir import (
    HistoryEntry,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.scale_model import ScaleModel, theta
from mollifier_theta.core.stage_meta import BoundMeta, VoronoiKind, get_voronoi_meta
from mollifier_theta.analysis.exponent_model import ExponentConstraint


@runtime_checkable
class BoundStrategy(Protocol):
    """Protocol for a bounding strategy that can be applied to off-diagonal terms."""

    @property
    def name(self) -> str: ...

    @property
    def citation(self) -> str: ...

    def applies(self, term: Term) -> bool: ...

    def bound(self, term: Term) -> Term: ...

    def constraints(self) -> list[ExponentConstraint]: ...


@runtime_checkable
class MultiBoundStrategy(Protocol):
    """Protocol for a bounding strategy that produces multiple BoundOnly terms.

    Used when a single input term gives rise to multiple bound terms
    (e.g. case-tree bounds with different regimes).
    """

    @property
    def name(self) -> str: ...

    @property
    def citation(self) -> str: ...

    def applies(self, term: Term) -> bool: ...

    def bound_multi(self, term: Term) -> list[Term]: ...

    def constraints(self) -> list[ExponentConstraint]: ...


class BoundStrategyRegistry:
    """Registry of available bound strategies for pipeline selection.

    Accepts both BoundStrategy (single output) and MultiBoundStrategy
    (multiple outputs per input term).
    """

    def __init__(self) -> None:
        self._strategies: dict[str, BoundStrategy | MultiBoundStrategy] = {}

    def register(self, strategy: BoundStrategy | MultiBoundStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> BoundStrategy | MultiBoundStrategy | None:
        return self._strategies.get(name)

    def list_strategies(self) -> list[str]:
        return list(self._strategies.keys())

    def all_strategies(self) -> list[BoundStrategy | MultiBoundStrategy]:
        return list(self._strategies.values())


class PostVoronoiBound:
    """Toy post-Voronoi bound with structurally different exponent.

    After Voronoi summation, the dual sum has length n* ~ c^2/N, which
    changes the exponent balance.  This toy bound uses:

        E(theta) = 2*theta - 1/4

    giving theta_max = 5/8 = 0.625 (from E(theta) = 1).

    This is NOT a rigorous derivation — it is a structurally different
    placeholder that exercises the infrastructure for having multiple
    bound families with different theta_max values in the constraint table.
    The overall pipeline theta_max = min(4/7, 5/8) = 4/7 (DI is still
    binding), but the diagnostic output now distinguishes the two families.
    """

    @property
    def name(self) -> str:
        return "PostVoronoi"

    @property
    def citation(self) -> str:
        return (
            "Toy post-Voronoi bound: E(theta) = 2*theta - 1/4. "
            "Not a rigorous derivation. See Bui-Conrey-Young 2011 for "
            "the real analysis."
        )

    def applies(self, term: Term) -> bool:
        if not (
            term.kind == TermKind.KLOOSTERMAN
            and term.status == TermStatus.ACTIVE
            and term.metadata.get("kloosterman_form", False)
            and term.metadata.get("voronoi_applied", False)
        ):
            return False
        # Red Flag B: PostVoronoiBound only for structural Voronoi or missing metadata
        vm = get_voronoi_meta(term)
        if vm is not None and vm.kind == VoronoiKind.FORMULA:
            return False
        return True

    def bound(self, term: Term) -> Term:
        scale = ScaleModel(
            T_exponent=(8 * theta - 1) / 4,
            description="Toy post-Voronoi bound: E(theta) = 2*theta - 1/4",
            sub_exponents={
                "dual_sum_contribution": 2 * theta,
            },
        )

        history = HistoryEntry(
            transform="PostVoronoiBound",
            parent_ids=[term.id],
            description=(
                "Applied toy post-Voronoi bound. "
                "Error exponent E(theta) = 2*theta - 1/4. "
                "theta_max for this constraint alone = 5/8."
            ),
        )

        return Term(
            kind=TermKind.KLOOSTERMAN,
            expression=f"Post-Voronoi bound: T^(2*theta-1/4) [from {term.expression}]",
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=list(term.phases),
            scale_model=scale.to_str(),
            status=TermStatus.BOUND_ONLY,
            history=list(term.history) + [history],
            parents=[term.id],
            lemma_citation=self.citation,
            multiplicity=term.multiplicity,
            kernel_state=term.kernel_state,
            metadata={
                **term.metadata,
                "post_voronoi_bound": True,
                "bound_strategy": self.name,
                "error_exponent": str(scale.T_exponent),
                "scale_model_dict": scale.to_dict(),
                "_bound": BoundMeta(
                    strategy=self.name,
                    error_exponent=str(scale.T_exponent),
                    citation=self.citation,
                    bound_family="PostVoronoi",
                ).model_dump(),
            },
        )

    def constraints(self) -> list[ExponentConstraint]:
        return [
            ExponentConstraint(
                name="post_voronoi_toy",
                expression_str="2*theta - 1/4",
                description=(
                    "Toy post-Voronoi bound: E(theta) = 2*theta - 1/4. "
                    "theta_max = 5/8 for this constraint alone."
                ),
                citation=self.citation,
                bound_family="PostVoronoi",
            ),
        ]


def create_default_registry() -> BoundStrategyRegistry:
    """Create a registry pre-populated with available strategies."""
    registry = BoundStrategyRegistry()
    registry.register(PostVoronoiBound())
    return registry
