"""Tests for the trace diff utility."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.analysis.trace_diff import TraceDiff, diff_traces
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
from mollifier_theta.core.stage_meta import BoundMeta
from mollifier_theta.pipelines.derivation_trace import DerivationTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bound_term(
    family: str = "DI_Kloosterman",
    case_id: str = "",
    expression: str = "bound term",
) -> Term:
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression=expression,
        status=TermStatus.BOUND_ONLY,
        lemma_citation="test citation",
        kernel_state=KernelState.KLOOSTERMANIZED,
        metadata={
            "_bound": BoundMeta(
                strategy=family,
                bound_family=family,
                case_id=case_id,
            ).model_dump(),
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTraceDiff:

    def test_identical_traces_empty_diff(self) -> None:
        terms = [_make_bound_term()]
        trace = DerivationTrace.from_terms(terms)
        result = diff_traces(trace, trace)
        assert result.is_empty

    def test_different_families_detected(self) -> None:
        terms_a = [_make_bound_term(family="DI_Kloosterman")]
        terms_b = [_make_bound_term(family="PostVoronoi")]
        trace_a = DerivationTrace.from_terms(terms_a)
        trace_b = DerivationTrace.from_terms(terms_b)
        result = diff_traces(trace_a, trace_b)
        assert "PostVoronoi" in result.added_families
        assert "DI_Kloosterman" in result.removed_families

    def test_case_id_changes_detected(self) -> None:
        terms_a = [_make_bound_term(family="SLS", case_id="small")]
        terms_b = [_make_bound_term(family="SLS", case_id="large")]
        trace_a = DerivationTrace.from_terms(terms_a)
        trace_b = DerivationTrace.from_terms(terms_b)
        result = diff_traces(trace_a, trace_b)
        assert "SLS:large" in result.added_case_ids
        assert "SLS:small" in result.removed_case_ids

    def test_case_count_changes(self) -> None:
        terms_a = [_make_bound_term(family="DI", case_id="x")]
        terms_b = [
            _make_bound_term(family="DI", case_id="x"),
            _make_bound_term(family="DI", case_id="x"),
        ]
        trace_a = DerivationTrace.from_terms(terms_a)
        trace_b = DerivationTrace.from_terms(terms_b)
        result = diff_traces(trace_a, trace_b)
        assert "DI:x" in result.case_count_changes
        assert result.case_count_changes["DI:x"] == (1, 2)

    def test_theta_max_diff(self) -> None:
        terms = [_make_bound_term()]
        trace = DerivationTrace.from_terms(terms)
        result = diff_traces(
            trace, trace,
            theta_max_a=Fraction(4, 7),
            theta_max_b=Fraction(5, 8),
        )
        assert result.theta_max_a == Fraction(4, 7)
        assert result.theta_max_b == Fraction(5, 8)
        assert not result.is_empty

    def test_format_no_differences(self) -> None:
        terms = [_make_bound_term()]
        trace = DerivationTrace.from_terms(terms)
        result = diff_traces(trace, trace)
        output = result.format()
        assert "no differences" in output

    def test_format_with_differences(self) -> None:
        terms_a = [_make_bound_term(family="DI")]
        terms_b = [_make_bound_term(family="PV")]
        trace_a = DerivationTrace.from_terms(terms_a)
        trace_b = DerivationTrace.from_terms(terms_b)
        result = diff_traces(
            trace_a, trace_b,
            theta_max_a=Fraction(4, 7),
            theta_max_b=Fraction(5, 8),
            binding_family_a="DI",
            binding_family_b="PV",
        )
        output = result.format(label_a="baseline", label_b="voronoi")
        assert "baseline" in output
        assert "voronoi" in output

    def test_binding_family_diff(self) -> None:
        terms = [_make_bound_term()]
        trace = DerivationTrace.from_terms(terms)
        result = diff_traces(
            trace, trace,
            binding_family_a="DI",
            binding_family_b="SLS",
        )
        assert not result.is_empty
        assert "DI" in result.format()
        assert "SLS" in result.format()
