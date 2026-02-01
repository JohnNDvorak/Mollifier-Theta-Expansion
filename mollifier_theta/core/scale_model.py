"""ScaleModel: symbolic T/theta exponent tracking.

This is one of two places where SymPy is allowed (the other is reports/).
"""

from __future__ import annotations

from functools import lru_cache

import sympy as sp


# Canonical symbols
T = sp.Symbol("T", positive=True)
theta = sp.Symbol("theta", positive=True)


@lru_cache(maxsize=1)
def _symbols() -> tuple[sp.Symbol, sp.Symbol]:
    return T, theta


class ScaleModel:
    """Tracks the T-exponent of a term as a symbolic expression in theta.

    A term with scale T^{f(theta)} * (log T)^k is represented as
    ScaleModel(T_exponent=f(theta), log_power=k).
    """

    # Shared locals for sympify so "theta" always maps to the canonical symbol
    _PARSE_LOCALS: dict[str, sp.Basic] = {"theta": theta, "T": T}

    def __init__(
        self,
        T_exponent: sp.Expr | str | int | float,
        log_power: int = 0,
        description: str = "",
        sub_exponents: dict[str, sp.Expr] | None = None,
    ) -> None:
        if isinstance(T_exponent, (str,)):
            T_exponent = sp.sympify(T_exponent, locals=self._PARSE_LOCALS)
        elif isinstance(T_exponent, (int, float)):
            T_exponent = sp.Rational(T_exponent) if isinstance(T_exponent, int) else sp.nsimplify(T_exponent, rational=True)
        self.T_exponent: sp.Expr = T_exponent
        self.log_power: int = log_power
        self.description: str = description
        self.sub_exponents: dict[str, sp.Expr] = sub_exponents or {}

    def evaluate(self, theta_val: float | sp.Rational) -> float:
        """Evaluate T_exponent at a specific theta value."""
        _, th = _symbols()
        result = self.T_exponent.subs(th, theta_val)
        return float(result)

    def is_negligible_at(self, theta_val: float | sp.Rational) -> bool:
        """True if T_exponent < 0 at the given theta (term is o(1))."""
        return self.evaluate(theta_val) < 0

    def is_dominated_at(self, theta_val: float | sp.Rational) -> bool:
        """True if T_exponent < 1 at the given theta (term is o(T))."""
        return self.evaluate(theta_val) < 1

    def product(self, other: "ScaleModel") -> "ScaleModel":
        """Product of two terms: exponents add, log powers add."""
        merged_sub = {**self.sub_exponents, **other.sub_exponents}
        return ScaleModel(
            T_exponent=sp.simplify(self.T_exponent + other.T_exponent),
            log_power=self.log_power + other.log_power,
            description=f"({self.description}) * ({other.description})",
            sub_exponents=merged_sub,
        )

    def sum_with(self, other: "ScaleModel") -> "ScaleModel":
        """Sum of two terms: take max exponent (dominant term)."""
        # Symbolic max — we use Piecewise or just keep the larger
        diff = sp.simplify(self.T_exponent - other.T_exponent)
        # If we can determine sign, pick the larger; otherwise keep self
        if diff.is_nonnegative:
            return ScaleModel(
                T_exponent=self.T_exponent,
                log_power=max(self.log_power, other.log_power),
                description=f"max({self.description}, {other.description})",
            )
        elif diff.is_nonpositive:
            return ScaleModel(
                T_exponent=other.T_exponent,
                log_power=max(self.log_power, other.log_power),
                description=f"max({self.description}, {other.description})",
            )
        else:
            # Can't determine statically — return symbolic Max
            return ScaleModel(
                T_exponent=sp.Max(self.T_exponent, other.T_exponent),
                log_power=max(self.log_power, other.log_power),
                description=f"max({self.description}, {other.description})",
            )

    def to_str(self) -> str:
        """Human-readable representation."""
        _, th = _symbols()
        exp_str = str(self.T_exponent)
        if self.log_power == 0:
            return f"T^({exp_str})"
        return f"T^({exp_str}) * (log T)^{self.log_power}"

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "T_exponent": str(self.T_exponent),
            "log_power": self.log_power,
            "description": self.description,
            "sub_exponents": {k: str(v) for k, v in self.sub_exponents.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScaleModel":
        """Deserialize from dict."""
        sub_exp = {k: sp.sympify(v, locals={"theta": theta, "T": T}) for k, v in data.get("sub_exponents", {}).items()}
        return cls(
            T_exponent=data["T_exponent"],
            log_power=data.get("log_power", 0),
            description=data.get("description", ""),
            sub_exponents=sub_exp,
        )

    def __repr__(self) -> str:
        return f"ScaleModel({self.to_str()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScaleModel):
            return NotImplemented
        return (
            sp.simplify(self.T_exponent - other.T_exponent) == 0
            and self.log_power == other.log_power
        )
