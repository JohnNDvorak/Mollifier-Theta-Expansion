"""Tests for slack diagnosis pipeline."""

from __future__ import annotations

import pytest

from mollifier_theta.analysis.slack import DiagnoseResult, TermSlack, diagnose_pipeline
from mollifier_theta.core.ir import TermStatus


class TestSlackPositiveBelowBarrier:
    """theta=0.56 is below 4/7 ~ 0.5714, so all slacks should be positive."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self) -> None:
        self.result = diagnose_pipeline(theta_val=0.56)

    def test_slack_positive_below_barrier(self) -> None:
        for ts in self.result.term_slacks:
            assert ts.slack > 0, f"Expected positive slack, got {ts.slack}"

    def test_headroom_positive(self) -> None:
        assert self.result.headroom > 0

    def test_bottleneck_exists(self) -> None:
        assert self.result.bottleneck is not None


class TestSlackZeroAtBarrier:
    """theta = 4/7: E(4/7) = 1 exactly, so slack = 0."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self) -> None:
        self.result = diagnose_pipeline(theta_val=4.0 / 7.0)

    def test_slack_zero_at_barrier(self) -> None:
        assert self.result.bottleneck is not None
        assert abs(self.result.bottleneck.slack) < 1e-10

    def test_headroom_zero(self) -> None:
        assert abs(self.result.headroom) < 1e-10


class TestSlackNegativeAboveBarrier:
    """theta=0.58 is above 4/7, so at least one slack should be negative."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self) -> None:
        self.result = diagnose_pipeline(theta_val=0.58)

    def test_slack_negative_above_barrier(self) -> None:
        assert self.result.bottleneck is not None
        assert self.result.bottleneck.slack < 0

    def test_headroom_negative(self) -> None:
        assert self.result.headroom < 0


class TestSlackOrdering:
    @pytest.fixture(autouse=True)
    def _run_pipeline(self) -> None:
        self.result = diagnose_pipeline(theta_val=0.56)

    def test_sorted_ascending(self) -> None:
        slacks = [ts.slack for ts in self.result.term_slacks]
        assert slacks == sorted(slacks)

    def test_bottleneck_is_first(self) -> None:
        if self.result.term_slacks:
            assert self.result.bottleneck == self.result.term_slacks[0]


class TestSubExponentsPopulated:
    def test_sub_exponents_populated(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        # DI-bounded terms should have sub_exponents from the ScaleModel
        di_slacks = [
            ts for ts in result.term_slacks
            if "DI" in ts.lemma_citation or "Deshouillers" in ts.lemma_citation
        ]
        assert len(di_slacks) > 0
        for ts in di_slacks:
            assert len(ts.sub_exponents) > 0


class TestHeadroomComputation:
    def test_headroom_computation(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        expected = result.theta_max - 0.56
        assert abs(result.headroom - expected) < 1e-12


class TestNonBoundOnlyExcluded:
    def test_non_bound_only_excluded(self) -> None:
        result = diagnose_pipeline(theta_val=0.56)
        # All term_slacks should come from BoundOnly terms
        # (verified by the fact that diagnose_pipeline only processes BoundOnly)
        assert len(result.term_slacks) > 0
        # term_slacks should only include BoundOnly terms
        # We check that the count matches the number of BoundOnly in the pipeline
        from mollifier_theta.pipelines.conrey89 import conrey89_pipeline

        pipeline_result = conrey89_pipeline(theta_val=0.56)
        bound_count = sum(
            1 for t in pipeline_result.ledger.all_terms()
            if t.status == TermStatus.BOUND_ONLY
        )
        assert len(result.term_slacks) == bound_count
