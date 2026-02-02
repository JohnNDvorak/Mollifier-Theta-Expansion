"""Golden end-to-end test for conrey89_spectral pipeline.

Captures the full pipeline output structure (term counts by kind/status/
kernel_state, bound case distribution, theta_max, binding_family) and
asserts it reproduces exactly.  Any change to transform logic that alters
the pipeline output will break this test — update the golden file if the
change is intentional.
"""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from mollifier_theta.core.stage_meta import get_bound_meta
from mollifier_theta.pipelines.conrey89_spectral import conrey89_spectral_pipeline

GOLDEN_DIR = Path(__file__).parent
GOLDEN_FILE = GOLDEN_DIR / "golden_spectral_pipeline.json"


def _load_golden() -> dict:
    return json.loads(GOLDEN_FILE.read_text())


def _build_actual() -> dict:
    """Run the spectral pipeline and extract structural counts."""
    result = conrey89_spectral_pipeline(theta_val=0.3, K=3, strict=True)
    all_terms = result.ledger.all_terms()

    kind_counts = dict(sorted(Counter(t.kind.value for t in all_terms).items()))
    status_counts = dict(sorted(Counter(t.status.value for t in all_terms).items()))
    kernel_state_counts = dict(
        sorted(Counter(t.kernel_state.value for t in all_terms).items())
    )

    bound_cases: list[str] = []
    for t in all_terms:
        bm = get_bound_meta(t)
        if bm and bm.bound_family:
            label = (
                f"{bm.bound_family}:{bm.case_id}" if bm.case_id else bm.bound_family
            )
            bound_cases.append(label)
    bound_case_counts = dict(sorted(Counter(bound_cases).items()))

    return {
        "pipeline_variant": "conrey89_spectral",
        "theta_val": 0.3,
        "K": 3,
        "theta_admissible": result.theta_admissible,
        "theta_max_symbolic": str(result.theta_max_result.symbolic),
        "binding_family": result.theta_max_result.binding_family,
        "total_terms": len(all_terms),
        "main_term_count": len(result.main_terms),
        "bound_only_count": len(result.bounded_terms),
        "error_count": len(result.error_terms),
        "kind_counts": kind_counts,
        "status_counts": status_counts,
        "kernel_state_counts": kernel_state_counts,
        "bound_case_counts": bound_case_counts,
        "transform_chain": result.report_data["transform_chain"],
    }


class TestSpectralPipelineGolden:
    def test_structural_counts_match_golden(self) -> None:
        golden = _load_golden()
        actual = _build_actual()
        assert actual == golden

    def test_theta_max_pinned(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert result.theta_max_result.symbolic == Fraction(1, 3)

    def test_binding_family_pinned(self) -> None:
        result = conrey89_spectral_pipeline(theta_val=0.3, K=3)
        assert result.theta_max_result.binding_family == "SpectralLargeSieve"

    def test_strict_mode_consistent(self) -> None:
        """Strict and non-strict produce same structural counts."""
        strict = conrey89_spectral_pipeline(theta_val=0.3, K=3, strict=True)
        relaxed = conrey89_spectral_pipeline(theta_val=0.3, K=3, strict=False)
        assert strict.ledger.count() == relaxed.ledger.count()
        assert len(strict.bounded_terms) == len(relaxed.bounded_terms)
        assert len(strict.main_terms) == len(relaxed.main_terms)

    def test_golden_file_is_sorted(self) -> None:
        """Golden file uses sort_keys=True."""
        data = _load_golden()
        reserialized = json.dumps(data, sort_keys=True, indent=2)
        original = GOLDEN_FILE.read_text().strip()
        assert reserialized == original

    def test_bound_case_symmetry(self) -> None:
        """Each SLS case should have equal count (one per spectralized input)."""
        golden = _load_golden()
        cases = golden["bound_case_counts"]
        counts = list(cases.values())
        # All 3 cases should have same count
        assert len(set(counts)) == 1, f"Asymmetric case counts: {cases}"
