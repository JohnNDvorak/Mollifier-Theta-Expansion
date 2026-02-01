"""What-if analysis: explore hypothetical sub-exponent changes.

Answers questions like: "What would theta_max be if di_saving were -theta/3
instead of -theta/4?"
"""

from __future__ import annotations

from dataclasses import dataclass

from mollifier_theta.core.scale_model import ScaleModel
from mollifier_theta.lemmas.di_kloosterman import DIExponentModel


@dataclass(frozen=True)
class WhatIfScenario:
    """Description of a hypothetical sub-exponent change."""

    name: str
    old_expr: str
    new_expr: str


@dataclass(frozen=True)
class WhatIfResult:
    """Result of a what-if analysis."""

    scenario: WhatIfScenario
    old_theta_max: float
    new_theta_max: float
    improvement: float  # new - old (positive means better)
    old_E_expr: str
    new_E_expr: str


def what_if_analysis(
    sub_exponent_name: str,
    new_expr_str: str,
) -> WhatIfResult:
    """Compute the effect of replacing a sub-exponent on theta_max.

    Parameters
    ----------
    sub_exponent_name:
        Key in DIExponentModel.sub_exponents (e.g. "di_saving").
    new_expr_str:
        New symbolic expression as a string (e.g. "-theta/3").

    Raises
    ------
    KeyError
        If *sub_exponent_name* is not a valid sub-exponent key.
    """
    model = DIExponentModel()

    if sub_exponent_name not in model.sub_exponents:
        valid = list(model.sub_exponents.keys())
        raise KeyError(
            f"Unknown sub-exponent '{sub_exponent_name}'. "
            f"Valid names: {valid}"
        )

    old_sub = model.sub_exponents[sub_exponent_name]
    old_E = model.error_exponent

    old_E_str = str(old_E)
    # Build new_E = old_E - old_sub + new_sub as a string expression
    new_E_str_raw = f"({old_E_str}) - ({old_sub}) + ({new_expr_str})"
    new_E_str = ScaleModel.simplify_expr(new_E_str_raw)

    # Solve new_E = 1 for theta via ScaleModel containment
    new_theta_max = ScaleModel.solve_expr_equals_one(new_E_str)

    old_theta_max = float(model.theta_max())

    scenario = WhatIfScenario(
        name=sub_exponent_name,
        old_expr=str(old_sub),
        new_expr=new_expr_str,
    )

    return WhatIfResult(
        scenario=scenario,
        old_theta_max=old_theta_max,
        new_theta_max=new_theta_max,
        improvement=new_theta_max - old_theta_max,
        old_E_expr=old_E_str,
        new_E_expr=new_E_str,
    )
