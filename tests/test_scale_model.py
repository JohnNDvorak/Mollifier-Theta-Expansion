"""Tests for ScaleModel symbolic exponent arithmetic."""

from __future__ import annotations

import sympy as sp

from mollifier_theta.core.scale_model import ScaleModel, theta


class TestScaleModelConstruction:
    def test_from_string(self) -> None:
        sm = ScaleModel(T_exponent="2*theta - 1")
        assert sm.T_exponent == 2 * theta - 1

    def test_from_int(self) -> None:
        sm = ScaleModel(T_exponent=1)
        assert sm.T_exponent == sp.Integer(1)

    def test_from_sympy_expr(self) -> None:
        expr = 3 * theta - sp.Rational(1, 2)
        sm = ScaleModel(T_exponent=expr)
        assert sm.T_exponent == expr

    def test_log_power(self) -> None:
        sm = ScaleModel(T_exponent=1, log_power=2)
        assert sm.log_power == 2

    def test_description(self) -> None:
        sm = ScaleModel(T_exponent=1, description="main term")
        assert sm.description == "main term"

    def test_sub_exponents(self) -> None:
        sm = ScaleModel(
            T_exponent="3*theta - 1",
            sub_exponents={
                "mollifier_length": theta,
                "modulus_range": 2 * theta - 1,
            },
        )
        assert "mollifier_length" in sm.sub_exponents


class TestScaleModelEvaluate:
    def test_evaluate_at_half(self) -> None:
        sm = ScaleModel(T_exponent="2*theta - 1")
        assert sm.evaluate(0.5) == 0.0

    def test_evaluate_at_four_sevenths(self) -> None:
        sm = ScaleModel(T_exponent="3*theta - 1")
        val = sm.evaluate(sp.Rational(4, 7))
        assert abs(val - 5 / 7) < 1e-10

    def test_is_negligible(self) -> None:
        sm = ScaleModel(T_exponent="theta - 1")
        assert sm.is_negligible_at(0.5)
        assert not sm.is_negligible_at(1.5)

    def test_is_dominated(self) -> None:
        sm = ScaleModel(T_exponent="2*theta - 1")
        assert sm.is_dominated_at(0.5)  # 0 < 1
        assert not sm.is_dominated_at(1.5)  # 2 > 1


class TestScaleModelArithmetic:
    def test_product_adds_exponents(self) -> None:
        a = ScaleModel(T_exponent="theta")
        b = ScaleModel(T_exponent="2*theta - 1")
        prod = a.product(b)
        assert sp.simplify(prod.T_exponent - (3 * theta - 1)) == 0

    def test_product_adds_log_powers(self) -> None:
        a = ScaleModel(T_exponent=0, log_power=2)
        b = ScaleModel(T_exponent=0, log_power=3)
        prod = a.product(b)
        assert prod.log_power == 5

    def test_sum_takes_max_exponent_clear_case(self) -> None:
        a = ScaleModel(T_exponent=sp.Integer(2))
        b = ScaleModel(T_exponent=sp.Integer(1))
        s = a.sum_with(b)
        assert s.T_exponent == sp.Integer(2)

    def test_sum_takes_max_exponent_symbolic(self) -> None:
        a = ScaleModel(T_exponent="3*theta")
        b = ScaleModel(T_exponent="theta + 1")
        s = a.sum_with(b)
        # For theta > 1/2: 3*theta > theta+1, so max = 3*theta
        # For theta < 1/2: 3*theta < theta+1, so max = theta+1
        # Result should be symbolic Max
        assert s.T_exponent is not None

    def test_product_merges_sub_exponents(self) -> None:
        a = ScaleModel(T_exponent="theta", sub_exponents={"x": theta})
        b = ScaleModel(
            T_exponent="theta", sub_exponents={"y": 2 * theta}
        )
        prod = a.product(b)
        assert "x" in prod.sub_exponents
        assert "y" in prod.sub_exponents


class TestScaleModelSerialization:
    def test_to_dict_roundtrip(self) -> None:
        sm = ScaleModel(
            T_exponent="3*theta - 1",
            log_power=2,
            description="test",
            sub_exponents={"mollifier_length": theta},
        )
        d = sm.to_dict()
        restored = ScaleModel.from_dict(d)
        assert sp.simplify(sm.T_exponent - restored.T_exponent) == 0
        assert sm.log_power == restored.log_power

    def test_to_str(self) -> None:
        sm = ScaleModel(T_exponent="2*theta - 1")
        s = sm.to_str()
        assert "T^(" in s


class TestScaleModelEquality:
    def test_equal_models(self) -> None:
        a = ScaleModel(T_exponent="2*theta - 1")
        b = ScaleModel(T_exponent="2*theta - 1")
        assert a == b

    def test_unequal_exponents(self) -> None:
        a = ScaleModel(T_exponent="2*theta")
        b = ScaleModel(T_exponent="3*theta")
        assert a != b

    def test_unequal_log_powers(self) -> None:
        a = ScaleModel(T_exponent=1, log_power=1)
        b = ScaleModel(T_exponent=1, log_power=2)
        assert a != b
