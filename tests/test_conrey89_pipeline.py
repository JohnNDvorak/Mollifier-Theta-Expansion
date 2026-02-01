"""E2E tests for the Conrey89 pipeline.

Critical regression gates:
- theta=0.56 PASS
- theta=0.58 FAIL
- theta_max ~ 4/7
- Layer 1 + Layer 2 cross-check
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from mollifier_theta.core.ir import TermKind, TermStatus
from mollifier_theta.lemmas.di_kloosterman import (
    DIExponentModel,
    DIKloostermanBound,
    KNOWN_THETA_MAX,
    ThetaBarrierMismatch,
)
from mollifier_theta.lemmas.theta_constraints import (
    ThetaMaxResult,
    find_theta_max,
    theta_admissible,
)
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline


class TestCriticalRegressionGates:
    def test_theta_056_passes(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert result.theta_admissible is True

    def test_theta_058_fails(self) -> None:
        result = conrey89_pipeline(theta_val=0.58)
        assert result.theta_admissible is False

    def test_theta_max_is_four_sevenths(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert result.theta_max is not None
        assert abs(result.theta_max - 4 / 7) < 0.001


class TestDIExponentModel:
    def test_layer1_symbolic_theta_max(self) -> None:
        model = DIExponentModel()
        theta_max = model.theta_max()
        assert theta_max == sp.Rational(4, 7)

    def test_layer2_crosscheck_passes(self) -> None:
        model = DIExponentModel()
        theta_max = model.theta_max_with_crosscheck()
        assert theta_max == sp.Rational(4, 7)

    def test_error_exponent_at_four_sevenths(self) -> None:
        model = DIExponentModel()
        val = model.evaluate_error(4 / 7)
        assert abs(val - 1.0) < 1e-10

    def test_error_exponent_below_barrier(self) -> None:
        model = DIExponentModel()
        val = model.evaluate_error(0.5)
        assert val < 1.0

    def test_error_exponent_above_barrier(self) -> None:
        model = DIExponentModel()
        val = model.evaluate_error(0.6)
        assert val > 1.0

    def test_sub_exponent_table(self) -> None:
        model = DIExponentModel()
        table = model.sub_exponent_table()
        assert len(table) >= 4
        components = {row["component"] for row in table}
        assert "DI bilinear saving" in components
        assert "Mollifier summation length" in components


class TestFindThetaMax:
    def test_find_theta_max_returns_result(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        all_terms = result.ledger.all_terms()
        tmr = find_theta_max(all_terms)
        assert isinstance(tmr, ThetaMaxResult)

    def test_symbolic_is_exact(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        tmr = find_theta_max(result.ledger.all_terms())
        assert tmr.symbolic == sp.Rational(4, 7)

    def test_numerical_within_tolerance(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        tmr = find_theta_max(result.ledger.all_terms())
        assert tmr.gap < 2 * tmr.tol

    def test_numerical_below_symbolic(self) -> None:
        """Binary search last-admissible must be strictly below the supremum."""
        result = conrey89_pipeline(theta_val=0.56)
        tmr = find_theta_max(result.ledger.all_terms())
        assert tmr.numerical_lo < tmr.symbolic_float

    def test_is_supremum(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        tmr = find_theta_max(result.ledger.all_terms())
        assert tmr.is_supremum is True

    def test_theta_admissible_boundary(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        all_terms = result.ledger.all_terms()
        assert theta_admissible(all_terms, 0.56) is True
        assert theta_admissible(all_terms, 0.58) is False

    def test_four_sevenths_itself_not_admissible(self) -> None:
        """4/7 is the supremum: E(4/7) = 1.0 exactly, so it fails strict < 1."""
        result = conrey89_pipeline(theta_val=0.56)
        all_terms = result.ledger.all_terms()
        assert theta_admissible(all_terms, 4 / 7) is False


class TestPipelineStructure:
    def test_ledger_nonempty(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert result.ledger.count() > 0

    def test_main_terms_exist(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert len(result.main_terms) > 0

    def test_bounded_terms_exist(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert len(result.bounded_terms) > 0

    def test_all_bounded_terms_have_citations(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        for term in result.bounded_terms:
            assert term.lemma_citation, f"Term {term.id} missing citation"

    def test_kloosterman_form_terms_exist(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        kloos = result.ledger.filter(kind=TermKind.KLOOSTERMAN)
        assert len(kloos) > 0

    def test_kloosterman_has_bounded_and_active_copies(self) -> None:
        """Off-diagonal Kloosterman terms exist both as BoundOnly and Active."""
        result = conrey89_pipeline(theta_val=0.56)
        kloos_bound = result.ledger.filter(
            kind=TermKind.KLOOSTERMAN, status=TermStatus.BOUND_ONLY
        )
        kloos_active = result.ledger.filter(
            kind=TermKind.KLOOSTERMAN, status=TermStatus.ACTIVE
        )
        assert len(kloos_bound) > 0, "No BoundOnly Kloosterman terms"
        assert len(kloos_active) > 0, "No Active Kloosterman terms (promotion hook)"


class TestReportData:
    def test_report_data_complete(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        rd = result.report_data
        assert "theta_val" in rd
        assert "theta_max" in rd
        assert "theta_max_numerical" in rd
        assert "theta_max_gap" in rd
        assert "theta_max_is_supremum" in rd
        assert "di_exponent_table" in rd
        assert "transform_chain" in rd

    def test_report_data_reconciliation(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        rd = result.report_data
        assert rd["theta_max_is_supremum"] is True
        assert abs(rd["theta_max"] - rd["theta_max_numerical"]) == rd["theta_max_gap"]

    def test_transform_chain_complete(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        chain = result.report_data["transform_chain"]
        assert "ApproxFunctionalEq" in chain
        assert "DIKloostermanBound" in chain

    def test_di_exponent_in_report(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        assert "7*theta/4" in result.report_data["di_error_exponent"]


class TestLedgerSerialization:
    def test_ledger_json_valid(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        json_str = result.ledger.to_json()
        parsed = json.loads(json_str)
        assert "terms" in parsed
        assert len(parsed["terms"]) > 0

    def test_ledger_json_roundtrip(self) -> None:
        from mollifier_theta.core.ledger import TermLedger

        result = conrey89_pipeline(theta_val=0.56)
        json_str = result.ledger.to_json()
        restored = TermLedger.from_json(json_str)
        assert len(restored) == len(result.ledger)

    def test_ledger_validates_clean(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        violations = result.ledger.validate_all()
        assert violations == []
