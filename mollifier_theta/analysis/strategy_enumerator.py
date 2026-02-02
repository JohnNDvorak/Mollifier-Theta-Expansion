"""Strategy-branch enumerator: compare bound strategies on the same terms.

Given a set of terms and a list of BoundStrategy/MultiBoundStrategy objects,
enumerate which strategies apply to which terms, produce bounds, and compare
the resulting theta_max across strategy combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.core.stage_meta import get_bound_meta
from mollifier_theta.lemmas.bound_strategy import BoundStrategy, MultiBoundStrategy
from mollifier_theta.lemmas.theta_constraints import find_theta_max, theta_admissible


@dataclass(frozen=True)
class StrategyMatch:
    """Records which strategy applies to which terms."""

    strategy_name: str
    is_multi: bool
    matched_term_ids: list[str]
    bound_count: int


@dataclass(frozen=True)
class StrategyBranchResult:
    """Result of applying a single strategy to all its matched terms."""

    strategy_name: str
    bound_terms: list[Term]
    theta_max: Fraction | None
    binding_family: str
    case_summary: dict[str, int]


@dataclass
class EnumerationResult:
    """Full result of enumerating all strategies."""

    matches: list[StrategyMatch]
    branches: list[StrategyBranchResult]
    best_strategy: str = ""
    best_theta_max: Fraction | None = None

    def format_summary(self) -> str:
        """Human-readable comparison table."""
        lines = ["Strategy Enumeration Summary:", ""]
        lines.append(f"{'Strategy':<30} {'Matches':<10} {'Bounds':<10} {'θ_max':<12} {'Binding'}")
        lines.append("-" * 80)
        for branch in sorted(self.branches, key=lambda b: b.theta_max or Fraction(0), reverse=True):
            theta_str = str(branch.theta_max) if branch.theta_max else "N/A"
            lines.append(
                f"{branch.strategy_name:<30} "
                f"{sum(1 for m in self.matches if m.strategy_name == branch.strategy_name):<10} "
                f"{len(branch.bound_terms):<10} "
                f"{theta_str:<12} "
                f"{branch.binding_family}"
            )
        if self.best_strategy:
            lines.append("")
            lines.append(f"Best: {self.best_strategy} (θ_max = {self.best_theta_max})")
        return "\n".join(lines)


def enumerate_strategies(
    terms: list[Term],
    strategies: list[BoundStrategy | MultiBoundStrategy],
    baseline_bound_terms: list[Term] | None = None,
    known_theta_max: Fraction | None = None,
) -> EnumerationResult:
    """Enumerate and compare strategies on the given terms.

    Args:
        terms: Active terms to try binding.
        strategies: Available bound strategies.
        baseline_bound_terms: Pre-existing BoundOnly terms (e.g., from TrivialBound)
            that should be included in theta evaluation for all branches.
        known_theta_max: If provided, passed to find_theta_max for cross-checking.

    Returns:
        EnumerationResult with per-strategy branches and comparison.
    """
    baseline = list(baseline_bound_terms) if baseline_bound_terms else []
    matches: list[StrategyMatch] = []
    branches: list[StrategyBranchResult] = []

    for strategy in strategies:
        is_multi = hasattr(strategy, "bound_multi")
        matched_ids: list[str] = []
        all_bounds: list[Term] = []

        for t in terms:
            if t.status != TermStatus.ACTIVE:
                continue
            if strategy.applies(t):
                matched_ids.append(t.id)
                if is_multi:
                    bounds = strategy.bound_multi(t)  # type: ignore[union-attr]
                    all_bounds.extend(bounds)
                else:
                    bound = strategy.bound(t)  # type: ignore[union-attr]
                    all_bounds.append(bound)

        matches.append(
            StrategyMatch(
                strategy_name=strategy.name,
                is_multi=is_multi,
                matched_term_ids=matched_ids,
                bound_count=len(all_bounds),
            )
        )

        if not all_bounds:
            branches.append(
                StrategyBranchResult(
                    strategy_name=strategy.name,
                    bound_terms=[],
                    theta_max=None,
                    binding_family="",
                    case_summary={},
                )
            )
            continue

        # Combine with baseline bounds and unmatched active terms for theta eval
        eval_terms = baseline + all_bounds
        # Add active terms not matched by this strategy (they don't constrain theta)
        # but include any pre-existing bound terms
        for t in terms:
            if t.status == TermStatus.BOUND_ONLY and t.id not in matched_ids:
                eval_terms.append(t)

        # Case summary
        case_counts: dict[str, int] = {}
        for bt in all_bounds:
            bm = get_bound_meta(bt)
            if bm:
                key = f"{bm.bound_family}:{bm.case_id}" if bm.case_id else bm.bound_family
                case_counts[key] = case_counts.get(key, 0) + 1
        case_summary = dict(sorted(case_counts.items()))

        # Find theta_max for this branch
        try:
            result = find_theta_max(eval_terms, known_theta_max=known_theta_max)
            theta_max = result.symbolic
            binding = result.binding_family
        except Exception:
            theta_max = None
            binding = ""

        branches.append(
            StrategyBranchResult(
                strategy_name=strategy.name,
                bound_terms=all_bounds,
                theta_max=theta_max,
                binding_family=binding,
                case_summary=case_summary,
            )
        )

    # Find best strategy (highest theta_max)
    best_strategy = ""
    best_theta_max: Fraction | None = None
    for branch in branches:
        if branch.theta_max is not None:
            if best_theta_max is None or branch.theta_max > best_theta_max:
                best_theta_max = branch.theta_max
                best_strategy = branch.strategy_name

    return EnumerationResult(
        matches=matches,
        branches=branches,
        best_strategy=best_strategy,
        best_theta_max=best_theta_max,
    )
