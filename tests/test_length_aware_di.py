"""Tests for the length-aware DI model and bound strategy."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.analysis.exponent_model import ExponentConstraint
from mollifier_theta.analysis.length_aware_di import LengthAwareDIModel
from mollifier_theta.analysis.strategy_enumerator import enumerate_strategies
from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    VoronoiKind,
    VoronoiMeta,
    get_bound_meta,
    _VORONOI_KEY,
)
from mollifier_theta.lemmas.length_aware_di import LengthAwareDIBound


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FOUR_SEVENTHS = Fraction(4, 7)
TOL = 1e-6


def _make_symmetric_kloosterman_term() -> Term:
    """Active KLOOSTERMAN term without Voronoi (symmetric case)."""
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="sum_{m,n} a_m b_n S(m,n;c)/c",
        variables=["m", "n", "c"],
        ranges=[
            Range(variable="m", lower="1", upper="T^theta"),
            Range(variable="n", lower="1", upper="T^theta"),
        ],
        kernels=[Kernel(name="KloostermanKernel")],
        phases=[Phase(expression="S(m,n;c)/c", depends_on=["m", "n", "c"])],
        kernel_state=KernelState.KLOOSTERMANIZED,
        status=TermStatus.ACTIVE,
        metadata={"kloosterman_form": True},
    )


def _make_voronoi_dual_term() -> Term:
    """Active KLOOSTERMAN term with structural Voronoi applied."""
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="sum_{m,n*} a_m b_{n*} S(m,n*;c)/c",
        variables=["m", "n_star", "c"],
        ranges=[
            Range(variable="m", lower="1", upper="T^theta"),
            Range(variable="n_star", lower="1", upper="T^(2-3*theta)"),
        ],
        kernels=[Kernel(name="KloostermanKernel")],
        phases=[Phase(expression="S(m,n*;c)/c", depends_on=["m", "n_star", "c"])],
        kernel_state=KernelState.KLOOSTERMANIZED,
        status=TermStatus.ACTIVE,
        metadata={
            "kloosterman_form": True,
            "voronoi_applied": True,
            _VORONOI_KEY: VoronoiMeta(
                applied=True,
                target_variable="n",
                dual_variable="n_star",
                dual_length="T^(2-3*theta)",
                kind=VoronoiKind.STRUCTURAL_ONLY,
            ).model_dump(),
        },
    )


# ---------------------------------------------------------------------------
# TestLengthAwareDIModel
# ---------------------------------------------------------------------------

class TestLengthAwareDIModel:
    """Tests for the parametric DI exponent model."""

    def test_symmetric_sub_A_at_four_sevenths(self) -> None:
        model = LengthAwareDIModel.symmetric()
        # sub_A = (theta + theta)/2 + (1 - theta) = 1
        val = model.sub_A_at(float(FOUR_SEVENTHS))
        assert abs(val - 1.0) < TOL

    def test_symmetric_sub_B_at_four_sevenths(self) -> None:
        model = LengthAwareDIModel.symmetric()
        # sub_B = 2*theta + (1 - theta)/2 = 3*theta/2 + 1/2
        # At theta=4/7: 6/7 + 1/2 = 19/14
        expected = 19.0 / 14.0
        val = model.sub_B_at(float(FOUR_SEVENTHS))
        assert abs(val - expected) < TOL

    def test_symmetric_error_at_four_sevenths(self) -> None:
        model = LengthAwareDIModel.symmetric()
        # max(1, 19/14) = 19/14, so E = 19/28
        val = model.evaluate_error(float(FOUR_SEVENTHS))
        assert abs(val - 19.0 / 28.0) < TOL

    def test_symmetric_raw_formula_less_than_one_at_four_sevenths(self) -> None:
        """The raw DI formula gives E < 1 at theta=4/7.

        This confirms the 7*theta/4 constraint is MORE restrictive.
        """
        model = LengthAwareDIModel.symmetric()
        val = model.evaluate_error(float(FOUR_SEVENTHS))
        assert val < 1.0

    def test_symmetric_theta_max_exceeds_four_sevenths(self) -> None:
        """Raw DI formula alone allows theta > 4/7."""
        model = LengthAwareDIModel.symmetric()
        tm = model.theta_max()
        assert tm > float(FOUR_SEVENTHS) + TOL

    def test_symmetric_constraints_include_7theta4(self) -> None:
        """Symmetric model returns both raw formula and 7*theta/4."""
        model = LengthAwareDIModel.symmetric()
        constraints = model.constraints()
        names = [c.name for c in constraints]
        assert "di_conrey_7theta4" in names
        assert any("di_raw" in n for n in names)

    def test_symmetric_7theta4_constraint_gives_four_sevenths(self) -> None:
        model = LengthAwareDIModel.symmetric()
        constraints = model.constraints()
        conrey = next(c for c in constraints if c.name == "di_conrey_7theta4")
        assert abs(conrey.solve_theta_max() - float(FOUR_SEVENTHS)) < TOL

    def test_voronoi_dual_different_sub_exponents(self) -> None:
        sym = LengthAwareDIModel.symmetric()
        dual = LengthAwareDIModel.voronoi_dual()
        # At theta=4/7, the dual sub-exponents should differ from symmetric
        theta_val = float(FOUR_SEVENTHS)
        assert abs(sym.sub_B_at(theta_val) - dual.sub_B_at(theta_val)) > TOL

    def test_voronoi_dual_theta_max_differs_from_symmetric(self) -> None:
        sym = LengthAwareDIModel.symmetric()
        dual = LengthAwareDIModel.voronoi_dual()
        assert abs(sym.theta_max() - dual.theta_max()) > TOL

    def test_voronoi_dual_label(self) -> None:
        dual = LengthAwareDIModel.voronoi_dual()
        assert dual.label == "voronoi_dual"

    def test_symmetric_label(self) -> None:
        sym = LengthAwareDIModel.symmetric()
        assert sym.label == "symmetric"

    def test_voronoi_dual_no_7theta4_constraint(self) -> None:
        """The voronoi_dual model should NOT include 7*theta/4."""
        dual = LengthAwareDIModel.voronoi_dual()
        constraints = dual.constraints()
        names = [c.name for c in constraints]
        assert "di_conrey_7theta4" not in names

    def test_name_property(self) -> None:
        assert LengthAwareDIModel.symmetric().name == "LengthAwareDI_symmetric"
        assert LengthAwareDIModel.voronoi_dual().name == "LengthAwareDI_voronoi_dual"

    def test_constraints_are_exponent_constraints(self) -> None:
        for model in [LengthAwareDIModel.symmetric(), LengthAwareDIModel.voronoi_dual()]:
            for c in model.constraints():
                assert isinstance(c, ExponentConstraint)


# ---------------------------------------------------------------------------
# TestLengthAwareDIBound
# ---------------------------------------------------------------------------

class TestLengthAwareDIBound:
    """Tests for the LengthAwareDIBound strategy."""

    def test_applies_to_kloosterman_active(self) -> None:
        strategy = LengthAwareDIBound()
        term = _make_symmetric_kloosterman_term()
        assert strategy.applies(term) is True

    def test_does_not_apply_to_bound_only(self) -> None:
        strategy = LengthAwareDIBound()
        term = _make_symmetric_kloosterman_term().with_updates(
            status=TermStatus.BOUND_ONLY,
            lemma_citation="test",
        )
        assert strategy.applies(term) is False

    def test_does_not_apply_without_kloosterman_form(self) -> None:
        strategy = LengthAwareDIBound()
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            status=TermStatus.ACTIVE,
            metadata={},
        )
        assert strategy.applies(term) is False

    def test_bound_produces_bound_only(self) -> None:
        strategy = LengthAwareDIBound()
        term = _make_symmetric_kloosterman_term()
        result = strategy.bound(term)
        assert result.status == TermStatus.BOUND_ONLY

    def test_bound_has_citation(self) -> None:
        strategy = LengthAwareDIBound()
        term = _make_symmetric_kloosterman_term()
        result = strategy.bound(term)
        assert result.lemma_citation != ""

    def test_bound_has_bound_meta(self) -> None:
        strategy = LengthAwareDIBound()
        term = _make_symmetric_kloosterman_term()
        result = strategy.bound(term)
        bm = get_bound_meta(result)
        assert bm is not None
        assert "LengthAwareDI" in bm.bound_family

    def test_symmetric_term_gets_symmetric_model(self) -> None:
        strategy = LengthAwareDIBound()
        term = _make_symmetric_kloosterman_term()
        result = strategy.bound(term)
        assert result.metadata.get("di_model_label") == "symmetric"

    def test_voronoi_dual_term_gets_dual_model(self) -> None:
        strategy = LengthAwareDIBound()
        term = _make_voronoi_dual_term()
        result = strategy.bound(term)
        assert result.metadata.get("di_model_label") == "voronoi_dual"

    def test_constraints_cover_both_models(self) -> None:
        strategy = LengthAwareDIBound()
        constraints = strategy.constraints()
        families = {c.bound_family for c in constraints}
        assert any("symmetric" in f for f in families)
        assert any("voronoi_dual" in f for f in families)

    def test_name_property(self) -> None:
        assert LengthAwareDIBound().name == "LengthAwareDI"


# ---------------------------------------------------------------------------
# TestEnumeratorComparison
# ---------------------------------------------------------------------------

class TestEnumeratorComparison:
    """Tests for comparing LengthAwareDI with other strategies via enumerator."""

    def test_enumerate_length_aware_di(self) -> None:
        terms = [_make_symmetric_kloosterman_term()]
        strategy = LengthAwareDIBound()
        result = enumerate_strategies(terms, [strategy])
        assert len(result.matches) == 1
        assert result.matches[0].strategy_name == "LengthAwareDI"
        assert result.matches[0].bound_count == 1

    def test_enumerate_with_voronoi_term(self) -> None:
        terms = [_make_voronoi_dual_term()]
        strategy = LengthAwareDIBound()
        result = enumerate_strategies(terms, [strategy])
        assert result.matches[0].bound_count == 1
