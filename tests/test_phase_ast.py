"""Tests for SymPy-free phase AST."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.core.phase_ast import (
    Add,
    Const,
    Div,
    Mul,
    Neg,
    Sub,
    Var,
    build_additive_twist,
    const,
    extract_linear_coefficient,
    var,
    variables_in,
)


class TestBasicNodes:
    def test_var_evaluate(self) -> None:
        v = var("n")
        assert v.evaluate({"n": Fraction(5)}) == Fraction(5)

    def test_var_missing_raises(self) -> None:
        v = var("n")
        with pytest.raises(KeyError):
            v.evaluate({"m": Fraction(1)})

    def test_const_evaluate(self) -> None:
        c = const(3)
        assert c.evaluate({}) == Fraction(3)

    def test_neg_evaluate(self) -> None:
        n = Neg(operand=const(5))
        assert n.evaluate({}) == Fraction(-5)

    def test_add_evaluate(self) -> None:
        e = Add(left=const(2), right=const(3))
        assert e.evaluate({}) == Fraction(5)

    def test_sub_evaluate(self) -> None:
        e = Sub(left=const(5), right=const(3))
        assert e.evaluate({}) == Fraction(2)

    def test_mul_evaluate(self) -> None:
        e = Mul(left=const(3), right=var("n"))
        assert e.evaluate({"n": Fraction(4)}) == Fraction(12)

    def test_div_evaluate(self) -> None:
        e = Div(numerator=var("a"), denominator=var("c"))
        assert e.evaluate({"a": Fraction(6), "c": Fraction(3)}) == Fraction(2)

    def test_div_by_zero_raises(self) -> None:
        e = Div(numerator=const(1), denominator=const(0))
        with pytest.raises(ZeroDivisionError):
            e.evaluate({})


class TestStringRepresentation:
    def test_var_str(self) -> None:
        assert str(var("n")) == "n"

    def test_const_str(self) -> None:
        assert str(const(3)) == "3"

    def test_complex_str(self) -> None:
        e = Div(
            numerator=Mul(left=var("a"), right=var("n")),
            denominator=var("c"),
        )
        s = str(e)
        assert "a" in s and "n" in s and "c" in s


class TestExtractLinearCoefficient:
    def test_simple_var(self) -> None:
        coeff, rem = extract_linear_coefficient(var("n"), "n")
        assert coeff is not None
        assert coeff.evaluate({}) == Fraction(1)
        assert rem.evaluate({}) == Fraction(0)

    def test_scaled_var(self) -> None:
        e = Mul(left=const(3), right=var("n"))
        coeff, rem = extract_linear_coefficient(e, "n")
        assert coeff is not None
        assert coeff.evaluate({}) == Fraction(3)

    def test_scaled_var_reversed(self) -> None:
        e = Mul(left=var("n"), right=const(5))
        coeff, rem = extract_linear_coefficient(e, "n")
        assert coeff is not None
        assert coeff.evaluate({}) == Fraction(5)

    def test_div_by_modulus(self) -> None:
        # a*n / c -> coefficient of n is a/c
        e = Div(
            numerator=Mul(left=var("a"), right=var("n")),
            denominator=var("c"),
        )
        coeff, rem = extract_linear_coefficient(e, "n")
        assert coeff is not None
        # coeff should be a/c
        assert coeff.evaluate({"a": Fraction(6), "c": Fraction(3)}) == Fraction(2)

    def test_not_present(self) -> None:
        coeff, rem = extract_linear_coefficient(var("m"), "n")
        assert coeff is None and rem is None

    def test_constant_only(self) -> None:
        coeff, rem = extract_linear_coefficient(const(5), "n")
        assert coeff is None and rem is None

    def test_add_with_remainder(self) -> None:
        # 3*n + 5
        e = Add(left=Mul(left=const(3), right=var("n")), right=const(5))
        coeff, rem = extract_linear_coefficient(e, "n")
        assert coeff is not None
        assert coeff.evaluate({}) == Fraction(3)
        assert rem.evaluate({}) == Fraction(5)

    def test_sub_with_target_on_left(self) -> None:
        # 2*n - m
        e = Sub(left=Mul(left=const(2), right=var("n")), right=var("m"))
        coeff, rem = extract_linear_coefficient(e, "n")
        assert coeff is not None
        assert coeff.evaluate({}) == Fraction(2)

    def test_sub_with_target_on_right(self) -> None:
        # m - 3*n  -> coefficient of n is -3
        e = Sub(left=var("m"), right=Mul(left=const(3), right=var("n")))
        coeff, rem = extract_linear_coefficient(e, "n")
        assert coeff is not None
        # coeff should be Neg(Const(3)) which evaluates to -3
        assert coeff.evaluate({}) == Fraction(-3)


class TestVariablesIn:
    def test_single_var(self) -> None:
        assert variables_in(var("n")) == {"n"}

    def test_const(self) -> None:
        assert variables_in(const(5)) == set()

    def test_complex(self) -> None:
        e = Div(
            numerator=Sub(
                left=Mul(left=var("a"), right=var("m")),
                right=Mul(left=var("b"), right=var("n")),
            ),
            denominator=var("c"),
        )
        assert variables_in(e) == {"a", "m", "b", "n", "c"}


class TestBuildAdditiveTwist:
    def test_positive_twist(self) -> None:
        twist = build_additive_twist("a", "n", "c", sign=1)
        result = twist.evaluate({"a": Fraction(3), "n": Fraction(4), "c": Fraction(6)})
        assert result == Fraction(2)  # 3*4/6 = 2

    def test_negative_twist(self) -> None:
        twist = build_additive_twist("b", "n", "c", sign=-1)
        result = twist.evaluate({"b": Fraction(3), "n": Fraction(4), "c": Fraction(6)})
        assert result == Fraction(-2)  # -(3*4)/6 = -2


class TestPydanticSerialization:
    def test_round_trip(self) -> None:
        e = Div(
            numerator=Mul(left=var("a"), right=var("n")),
            denominator=var("c"),
        )
        d = e.model_dump()
        rebuilt = Div.model_validate(d)
        result = rebuilt.evaluate({"a": Fraction(6), "n": Fraction(2), "c": Fraction(3)})
        assert result == Fraction(4)

    def test_golden_ast_serialization(self) -> None:
        """Golden test: (a*m - b*n) / c AST serializes deterministically."""
        import json

        e = Div(
            numerator=Sub(
                left=Mul(left=var("a"), right=var("m")),
                right=Mul(left=var("b"), right=var("n")),
            ),
            denominator=var("c"),
        )
        dumped = json.dumps(e.model_dump(), sort_keys=True)
        # Verify deterministic
        dumped2 = json.dumps(e.model_dump(), sort_keys=True)
        assert dumped == dumped2
        # Verify round-trip
        rebuilt = Div.model_validate(json.loads(dumped))
        env = {"a": Fraction(3), "m": Fraction(5), "b": Fraction(2), "n": Fraction(4), "c": Fraction(7)}
        assert rebuilt.evaluate(env) == e.evaluate(env)
        assert rebuilt.evaluate(env) == Fraction(7, 7)  # (15-8)/7 = 1

    def test_nested_expression_round_trip(self) -> None:
        """Complex nested expression survives JSON round-trip."""
        import json

        e = Add(
            left=Div(
                numerator=Mul(left=var("a"), right=var("n")),
                denominator=var("c"),
            ),
            right=Neg(operand=Div(
                numerator=Mul(left=var("b"), right=var("m")),
                denominator=var("c"),
            )),
        )
        d = json.loads(json.dumps(e.model_dump(), sort_keys=True))
        rebuilt = Add.model_validate(d)
        env = {"a": Fraction(2), "n": Fraction(3), "b": Fraction(1), "m": Fraction(4), "c": Fraction(6)}
        assert rebuilt.evaluate(env) == e.evaluate(env)
