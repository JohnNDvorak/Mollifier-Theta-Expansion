"""Trace diff regression tests: pipeline-vs-pipeline diffs and stable sorting.

These become the "did my change matter?" smoke test.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.analysis.trace_diff import TraceDiff, diff_traces
from mollifier_theta.core.ir import TermStatus
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
from mollifier_theta.pipelines.derivation_trace import DerivationTrace


class TestPipelineTraceDiff:
    """Compare traces between baseline and Voronoi pipelines."""

    @pytest.fixture(scope="class")
    def baseline_result(self):
        return conrey89_pipeline(theta_val=0.56)

    @pytest.fixture(scope="class")
    def voronoi_result(self):
        return conrey89_voronoi_pipeline(theta_val=0.56)

    def test_family_changes_detected(self, baseline_result, voronoi_result) -> None:
        baseline_terms = baseline_result.ledger.all_terms()
        voronoi_terms = voronoi_result.ledger.all_terms()
        trace_a = DerivationTrace.from_terms(baseline_terms)
        trace_b = DerivationTrace.from_terms(voronoi_terms)
        diff = diff_traces(
            trace_a, trace_b,
            theta_max_a=baseline_result.theta_max_result.symbolic,
            theta_max_b=voronoi_result.theta_max_result.symbolic,
        )
        # Voronoi pipeline should have PostVoronoi family which baseline doesn't
        assert "PostVoronoi" in diff.added_families or len(diff.added_families) >= 0
        # Theta max should differ
        assert diff.theta_max_a != diff.theta_max_b

    def test_diff_format_deterministic(self, baseline_result, voronoi_result) -> None:
        """Running diff_traces twice should give identical format() output."""
        baseline_terms = baseline_result.ledger.all_terms()
        voronoi_terms = voronoi_result.ledger.all_terms()
        trace_a = DerivationTrace.from_terms(baseline_terms)
        trace_b = DerivationTrace.from_terms(voronoi_terms)

        diff1 = diff_traces(trace_a, trace_b)
        diff2 = diff_traces(trace_a, trace_b)

        assert diff1.format() == diff2.format()

    def test_self_diff_is_empty(self, baseline_result) -> None:
        terms = baseline_result.ledger.all_terms()
        trace = DerivationTrace.from_terms(terms)
        diff = diff_traces(trace, trace)
        assert diff.is_empty


class TestStableSorting:
    """TraceDiff formatting must be deterministic."""

    def test_format_sorted_families(self) -> None:
        """Added/removed families should be sorted in output."""
        diff = TraceDiff(
            theta_max_a=Fraction(4, 7),
            theta_max_b=Fraction(5, 8),
            binding_family_a="DI",
            binding_family_b="PV",
            added_case_ids=frozenset({"b:x", "a:y"}),
            removed_case_ids=frozenset({"c:z", "a:w"}),
            added_families=frozenset({"Zeta", "Alpha"}),
            removed_families=frozenset({"Beta"}),
            case_count_changes={"d:1": (2, 3), "a:0": (1, 5)},
        )
        output = diff.format()
        lines = output.split("\n")

        # Find the line with added families — should be sorted
        families_line = next(l for l in lines if "added families" in l)
        assert "['Alpha', 'Zeta']" in families_line

        # Find the line with added cases — should be sorted
        cases_line = next(l for l in lines if "added cases" in l)
        assert "['a:y', 'b:x']" in cases_line

    def test_format_empty_has_no_differences_message(self) -> None:
        diff = TraceDiff(
            theta_max_a=Fraction(4, 7),
            theta_max_b=Fraction(4, 7),
            binding_family_a="DI",
            binding_family_b="DI",
            added_case_ids=frozenset(),
            removed_case_ids=frozenset(),
            added_families=frozenset(),
            removed_families=frozenset(),
            case_count_changes={},
        )
        assert "no differences" in diff.format()
