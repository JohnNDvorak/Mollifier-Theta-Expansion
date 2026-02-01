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

    def solve_for_theta(self, expr_str: str | None = None) -> float:
        """Solve expr = 1 for theta and return the smallest root in (0, 1).

        If *expr_str* is provided, parse it; otherwise use self.T_exponent.
        Falls back to numerical bisection for expressions SymPy can't
        handle analytically (Piecewise, Max, etc.).
        """
        if expr_str is not None:
            model = ScaleModel(T_exponent=expr_str)
        else:
            model = self

        roots = model.solve_all_roots(lo=0.0, hi=1.0)
        if not roots:
            raise ValueError(
                f"No solution found for ({model.T_exponent}) = 1 in (0, 1)"
            )
        return roots[0]  # smallest root

    def evaluate_sub_exponents(self, theta_val: float) -> dict[str, float]:
        """Evaluate all sub_exponents at *theta_val*."""
        _, th = _symbols()
        return {
            k: float(v.subs(th, theta_val))
            for k, v in self.sub_exponents.items()
        }

    def __repr__(self) -> str:
        return f"ScaleModel({self.to_str()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScaleModel):
            return NotImplemented
        return (
            sp.simplify(self.T_exponent - other.T_exponent) == 0
            and self.log_power == other.log_power
        )

    # ---- Containment helpers (WI-6) ----
    # All SymPy usage outside reports/ should go through these methods.

    def solve_all_roots(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        tol: float = 1e-10,
    ) -> list[float]:
        """Find ALL roots of T_exponent - 1 = 0 in (lo, hi).

        Uses symbolic solve first; falls back to numerical bisection for
        expressions SymPy can't handle (Piecewise, Max, etc.).
        Returns sorted list of roots.
        """
        _, th = _symbols()
        eq = self.T_exponent - 1

        # Try symbolic solve first
        roots: list[float] = []
        try:
            solutions = sp.solve(eq, th)
            for s in solutions:
                try:
                    val = float(s)
                    if lo < val < hi:
                        roots.append(val)
                except (TypeError, ValueError, OverflowError):
                    continue
        except (NotImplementedError, ValueError):
            pass

        # If symbolic solve found nothing (e.g. Piecewise/Max), use bisection
        if not roots:
            roots = self._bisection_roots(lo, hi, tol)

        # Verify roots numerically
        verified = []
        for r in roots:
            val = float(eq.subs(th, r))
            if abs(val) < max(tol, 1e-8):
                verified.append(r)

        return sorted(verified)

    def _bisection_roots(
        self,
        lo: float,
        hi: float,
        tol: float = 1e-10,
        n_samples: int = 200,
    ) -> list[float]:
        """Numerical root-finding via bisection on a grid."""
        _, th = _symbols()
        eq = self.T_exponent - 1
        roots: list[float] = []

        # Sample the interval
        step = (hi - lo) / n_samples
        prev_val = float(eq.subs(th, lo))
        prev_x = lo

        for i in range(1, n_samples + 1):
            x = lo + i * step
            try:
                val = float(eq.subs(th, x))
            except (TypeError, ValueError):
                prev_val = float("nan")
                prev_x = x
                continue

            # Exact root on grid point
            if abs(val) < tol and lo < x < hi:
                roots.append(x)
                prev_val = val
                prev_x = x
                continue

            # Sign change detected — bisect to find root
            if prev_val * val < 0:
                a, b = prev_x, x
                for _ in range(100):  # enough iterations for tol ~ 1e-10
                    mid = (a + b) / 2
                    mid_val = float(eq.subs(th, mid))
                    if abs(mid_val) < tol:
                        break
                    if prev_val * mid_val < 0:
                        b = mid
                    else:
                        a = mid
                        prev_val = mid_val
                roots.append((a + b) / 2)

            prev_val = val
            prev_x = x

        return roots

    @classmethod
    def evaluate_expr(cls, expr_str: str, theta_val: float) -> float:
        """Parse and evaluate a symbolic expression at a given theta."""
        expr = sp.sympify(expr_str, locals=cls._PARSE_LOCALS)
        return float(expr.subs(theta, theta_val))

    @classmethod
    def solve_expr_equals_one(cls, expr_str: str) -> float:
        """Solve expr = 1 for theta and return the smallest root in (0, 1).

        Delegates to solve_all_roots for robust handling of multi-root
        and non-algebraic (Piecewise, Max) expressions.
        """
        model = cls(T_exponent=expr_str)
        roots = model.solve_all_roots(lo=0.0, hi=1.0)
        if not roots:
            raise ValueError(f"No solution found for ({expr_str}) = 1 in (0, 1)")
        return roots[0]

    @classmethod
    def simplify_expr(cls, expr_str: str) -> str:
        """Simplify a symbolic expression and return as string."""
        expr = sp.sympify(expr_str, locals=cls._PARSE_LOCALS)
        return str(sp.simplify(expr))

    @classmethod
    def expr_to_rational(cls, expr_str: str) -> "Fraction":
        """Convert a symbolic expression to a Fraction (must be rational)."""
        from fractions import Fraction
        expr = sp.sympify(expr_str, locals=cls._PARSE_LOCALS)
        r = sp.Rational(expr)
        return Fraction(int(r.p), int(r.q))
