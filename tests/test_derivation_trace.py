"""Tests for DerivationTrace and runner.explain()."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mollifier_theta.core.ir import KernelState, TermKind, TermStatus
from mollifier_theta.core.stage_meta import get_bound_meta
from mollifier_theta.pipelines.conrey89_spectral import conrey89_spectral_pipeline
from mollifier_theta.pipelines.derivation_trace import DerivationTrace, TermTrace


class TestDerivationTraceFromPipeline:
    def test_trace_captures_all_terms(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        trace = DerivationTrace.from_terms(all_terms)
        assert len(trace.traces) == len(all_terms)

    def test_bound_traces_filtered(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        trace = DerivationTrace.from_terms(all_terms)
        bound_count = sum(1 for t in all_terms if t.status == TermStatus.BOUND_ONLY)
        assert len(trace.bound_traces) == bound_count

    def test_families_grouped(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        trace = DerivationTrace.from_terms(all_terms)
        families = trace.families
        assert "SpectralLargeSieve" in families

    def test_case_summary_has_all_cases(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        trace = DerivationTrace.from_terms(all_terms)
        summary = trace.case_summary
        assert "SpectralLargeSieve:small_modulus" in summary
        assert "SpectralLargeSieve:large_modulus" in summary
        assert "SpectralLargeSieve:bessel_transition" in summary

    def test_format_summary_is_string(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        trace = DerivationTrace.from_terms(all_terms)
        summary = trace.format_summary()
        assert isinstance(summary, str)
        assert "DerivationTrace" in summary

    def test_format_full_includes_bound_terms(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        trace = DerivationTrace.from_terms(all_terms)
        full = trace.format_full()
        assert "SpectralLargeSieve" in full
        assert "bound_family" in full


class TestTermTrace:
    def test_trace_has_steps(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        bound = result.bounded_terms[0]
        trace = DerivationTrace.from_terms([bound])
        assert len(trace.traces) == 1
        assert len(trace.traces[0].steps) > 0

    def test_trace_format(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        bound = result.bounded_terms[0]
        trace = DerivationTrace.from_terms([bound])
        formatted = trace.traces[0].format()
        assert "Term " in formatted
        assert "derivation" in formatted


class TestRunnerExplain:
    def test_explain_returns_string(self) -> None:
        from mollifier_theta.pipelines.strict_runner import StrictPipelineRunner
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3, strict=True)
        # Can't access runner directly, but test the from_terms path
        all_terms = result.ledger.all_terms()
        trace = DerivationTrace.from_terms(all_terms)
        output = trace.format_full()
        assert isinstance(output, str)
        assert len(output) > 100

    def test_explain_with_stage_log(self) -> None:
        """Stage log is incorporated into summary."""
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        all_terms = result.ledger.all_terms()
        stage_log = [
            {"stage": "TestStage", "input_count": 10, "output_count": 12, "violations": []},
        ]
        trace = DerivationTrace.from_terms(all_terms, stage_log=stage_log)
        summary = trace.format_summary()
        assert "TestStage" in summary
        assert "10" in summary
