"""SymPy-free AST for linear/rational phase expressions.

Used by transforms to pattern-match additive twist structure without
importing SymPy (respecting CLAUDE.md rule #7: SymPy containment).

Grammar:
  expr := Var(name) | Const(value) | Neg(expr) | Add(left, right)
        | Mul(Const, expr) | Div(expr, Var)

All nodes are frozen Pydantic models for consistency with the IR.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag


class Var(BaseModel):
    """A symbolic variable (m, n, a, c, etc.)."""

    model_config = {"frozen": True}
    tag: Literal["var"] = "var"
    name: str

    def evaluate(self, env: dict[str, Fraction]) -> Fraction:
        if self.name not in env:
            raise KeyError(f"Variable '{self.name}' not in environment")
        return env[self.name]

    def __str__(self) -> str:
        return self.name


class Const(BaseModel):
    """A rational constant."""

    model_config = {"frozen": True}
    tag: Literal["const"] = "const"
    value: Fraction

    def evaluate(self, env: dict[str, Fraction]) -> Fraction:
        return self.value

    def __str__(self) -> str:
        if self.value.denominator == 1:
            return str(self.value.numerator)
        return str(self.value)


class Neg(BaseModel):
    """Negation: -expr."""

    model_config = {"frozen": True}
    tag: Literal["neg"] = "neg"
    operand: PhaseExpr

    def evaluate(self, env: dict[str, Fraction]) -> Fraction:
        return -self.operand.evaluate(env)

    def __str__(self) -> str:
        return f"-({self.operand})"


class Add(BaseModel):
    """Sum of two expressions."""

    model_config = {"frozen": True}
    tag: Literal["add"] = "add"
    left: PhaseExpr
    right: PhaseExpr

    def evaluate(self, env: dict[str, Fraction]) -> Fraction:
        return self.left.evaluate(env) + self.right.evaluate(env)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


class Sub(BaseModel):
    """Difference of two expressions."""

    model_config = {"frozen": True}
    tag: Literal["sub"] = "sub"
    left: PhaseExpr
    right: PhaseExpr

    def evaluate(self, env: dict[str, Fraction]) -> Fraction:
        return self.left.evaluate(env) - self.right.evaluate(env)

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


class Mul(BaseModel):
    """Product: scalar * expr (or expr * expr for variable products)."""

    model_config = {"frozen": True}
    tag: Literal["mul"] = "mul"
    left: PhaseExpr
    right: PhaseExpr

    def evaluate(self, env: dict[str, Fraction]) -> Fraction:
        return self.left.evaluate(env) * self.right.evaluate(env)

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


class Div(BaseModel):
    """Division: numerator / denominator."""

    model_config = {"frozen": True}
    tag: Literal["div"] = "div"
    numerator: PhaseExpr
    denominator: PhaseExpr

    def evaluate(self, env: dict[str, Fraction]) -> Fraction:
        d = self.denominator.evaluate(env)
        if d == 0:
            raise ZeroDivisionError("Division by zero in phase expression")
        return self.numerator.evaluate(env) / d

    def __str__(self) -> str:
        return f"({self.numerator} / {self.denominator})"


# Discriminated union type for all phase expressions
PhaseExpr = Annotated[
    Union[
        Annotated[Var, Tag("var")],
        Annotated[Const, Tag("const")],
        Annotated[Neg, Tag("neg")],
        Annotated[Add, Tag("add")],
        Annotated[Sub, Tag("sub")],
        Annotated[Mul, Tag("mul")],
        Annotated[Div, Tag("div")],
    ],
    Discriminator("tag"),
]

# Rebuild models now that PhaseExpr is defined (forward references)
Neg.model_rebuild()
Add.model_rebuild()
Sub.model_rebuild()
Mul.model_rebuild()
Div.model_rebuild()


# ---- Helpers ----

def var(name: str) -> Var:
    """Shorthand constructor for Var."""
    return Var(name=name)


def const(value: int | Fraction) -> Const:
    """Shorthand constructor for Const."""
    if isinstance(value, int):
        value = Fraction(value)
    return Const(value=value)


def extract_linear_coefficient(
    expr: PhaseExpr,
    target_var: str,
) -> tuple[PhaseExpr | None, PhaseExpr | None]:
    """Extract the coefficient of *target_var* from a linear expression.

    If *expr* has the form `coeff * target_var + remainder` (where coeff
    and remainder are independent of target_var), return (coeff, remainder).

    Returns (None, None) if *target_var* does not appear linearly.

    This is a structural pattern match, not a symbolic solver. It recognizes:
      - Var(target_var) -> (Const(1), Const(0))
      - Mul(coeff, Var(target_var)) -> (coeff, Const(0))
      - Mul(Var(target_var), coeff) -> (coeff, Const(0))
      - Div(Var(target_var), denom) -> (Div(Const(1), denom), Const(0))
      - Add(left, right) where one side contains target_var
      - Sub(left, right) where one side contains target_var
      - Neg(inner) where inner contains target_var
    """
    zero = const(0)
    one = const(1)

    if isinstance(expr, Var):
        if expr.name == target_var:
            return one, zero
        return None, None

    if isinstance(expr, Const):
        return None, None

    if isinstance(expr, Neg):
        coeff, remainder = extract_linear_coefficient(expr.operand, target_var)
        if coeff is not None:
            return Neg(operand=coeff), Neg(operand=remainder) if remainder != zero else zero
        return None, None

    if isinstance(expr, Mul):
        # Check if one factor is the target var
        if isinstance(expr.left, Var) and expr.left.name == target_var:
            if not _contains_var(expr.right, target_var):
                return expr.right, zero
        if isinstance(expr.right, Var) and expr.right.name == target_var:
            if not _contains_var(expr.left, target_var):
                return expr.left, zero
        return None, None

    if isinstance(expr, Div):
        # e.g. (a * n) / c -> coefficient of n is a/c
        num_coeff, num_rem = extract_linear_coefficient(expr.numerator, target_var)
        if num_coeff is not None and not _contains_var(expr.denominator, target_var):
            result_coeff = Div(numerator=num_coeff, denominator=expr.denominator)
            result_rem = Div(numerator=num_rem, denominator=expr.denominator) if num_rem != zero else zero
            return result_coeff, result_rem
        return None, None

    if isinstance(expr, Add):
        # Try left contains target
        lc, lr = extract_linear_coefficient(expr.left, target_var)
        if lc is not None and not _contains_var(expr.right, target_var):
            remainder = Add(left=lr, right=expr.right) if lr != zero else expr.right
            return lc, remainder
        # Try right contains target
        rc, rr = extract_linear_coefficient(expr.right, target_var)
        if rc is not None and not _contains_var(expr.left, target_var):
            remainder = Add(left=expr.left, right=rr) if rr != zero else expr.left
            return rc, remainder
        return None, None

    if isinstance(expr, Sub):
        # left - right
        lc, lr = extract_linear_coefficient(expr.left, target_var)
        if lc is not None and not _contains_var(expr.right, target_var):
            remainder = Sub(left=lr, right=expr.right) if lr != zero else Neg(operand=expr.right)
            return lc, remainder
        rc, rr = extract_linear_coefficient(expr.right, target_var)
        if rc is not None and not _contains_var(expr.left, target_var):
            remainder = Sub(left=expr.left, right=rr) if rr != zero else expr.left
            return Neg(operand=rc), remainder
        return None, None

    return None, None


def _contains_var(expr: PhaseExpr, target_var: str) -> bool:
    """Check whether *target_var* appears anywhere in *expr*."""
    if isinstance(expr, Var):
        return expr.name == target_var
    if isinstance(expr, Const):
        return False
    if isinstance(expr, Neg):
        return _contains_var(expr.operand, target_var)
    if isinstance(expr, (Add, Sub)):
        return _contains_var(expr.left, target_var) or _contains_var(expr.right, target_var)
    if isinstance(expr, Mul):
        return _contains_var(expr.left, target_var) or _contains_var(expr.right, target_var)
    if isinstance(expr, Div):
        return _contains_var(expr.numerator, target_var) or _contains_var(expr.denominator, target_var)
    return False


def variables_in(expr: PhaseExpr) -> set[str]:
    """Return all variable names appearing in *expr*."""
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, Neg):
        return variables_in(expr.operand)
    if isinstance(expr, (Add, Sub)):
        return variables_in(expr.left) | variables_in(expr.right)
    if isinstance(expr, Mul):
        return variables_in(expr.left) | variables_in(expr.right)
    if isinstance(expr, Div):
        return variables_in(expr.numerator) | variables_in(expr.denominator)
    return set()


def build_additive_twist(
    numerator_var: str,
    sum_var: str,
    modulus_var: str,
    sign: int = 1,
) -> Div:
    """Build the standard additive twist expression ±(a*n)/c.

    Returns Div(Mul(Var(a), Var(n)), Var(c)) or its negation.
    """
    product = Mul(left=Var(name=numerator_var), right=Var(name=sum_var))
    twist = Div(numerator=product, denominator=Var(name=modulus_var))
    if sign < 0:
        return Div(numerator=Neg(operand=product), denominator=Var(name=modulus_var))
    return twist
