"""Math-parameter ledger export: extract quantitative parameters from bound terms.

Produces structured records suitable for external analysis tools
(JSON export, comparison tables, parameter sweeps).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from mollifier_theta.core.ir import Term, TermStatus
from mollifier_theta.core.stage_meta import get_bound_meta, get_voronoi_meta
from mollifier_theta.core.sum_structures import SumStructure


@dataclass(frozen=True)
class MathParameterRecord:
    """Quantitative parameters extracted from a single BoundOnly term."""

    term_id: str
    bound_family: str
    case_id: str
    error_exponent: str
    m_length_exponent: str
    n_length_exponent: str
    modulus_exponent: str
    kernel_family_tags: list[str]
    citation: str

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def _extract_sum_structure(term: Term) -> SumStructure | None:
    """Try to extract SumStructure from term metadata."""
    raw = term.metadata.get("sum_structure")
    if raw is None:
        return None
    if isinstance(raw, SumStructure):
        return raw
    if isinstance(raw, dict):
        return SumStructure.model_validate(raw)
    return None


def _extract_length_exponents(
    term: Term,
) -> tuple[str, str, str]:
    """Extract m-length, n-length, and modulus exponents from metadata.

    Falls back to defaults when structured metadata is not available.
    """
    m_length = "theta"
    n_length = "theta"
    modulus = "1-theta"

    # Check VoronoiMeta for dual length
    vm = get_voronoi_meta(term)
    if vm is not None and vm.applied and vm.dual_length:
        n_length = vm.dual_length

    # Check SumStructure for explicit ranges
    ss = _extract_sum_structure(term)
    if ss is not None and ss.sum_indices:
        for idx in ss.sum_indices:
            if idx.name in ("m", "m1"):
                m_length = idx.range_upper
            elif idx.name in ("n", "n1", "n*", "n_star"):
                n_length = idx.range_upper

    # Check di_model_label for model-specific info
    label = term.metadata.get("di_model_label", "")
    if label == "voronoi_dual":
        n_length = "T^(2-3*theta)"

    return m_length, n_length, modulus


def _extract_kernel_tags(term: Term) -> list[str]:
    """Extract kernel family tags from SumStructure weight_kernels."""
    ss = _extract_sum_structure(term)
    if ss is None:
        return []
    return [
        wk.bessel_family.value
        for wk in ss.weight_kernels
        if wk.bessel_family.value != "unspecified"
    ]


def export_math_parameters(terms: list[Term]) -> list[MathParameterRecord]:
    """Extract MathParameterRecords from BoundOnly terms.

    Only processes terms with status=BOUND_ONLY.
    """
    records: list[MathParameterRecord] = []

    for term in terms:
        if term.status != TermStatus.BOUND_ONLY:
            continue

        bm = get_bound_meta(term)
        bound_family = bm.bound_family if bm else ""
        case_id = bm.case_id if bm else ""
        error_exponent = bm.error_exponent if bm else term.metadata.get(
            "error_exponent", ""
        )
        citation_str = bm.citation if bm else term.lemma_citation

        m_length, n_length, modulus = _extract_length_exponents(term)
        kernel_tags = _extract_kernel_tags(term)

        records.append(
            MathParameterRecord(
                term_id=term.id,
                bound_family=bound_family,
                case_id=case_id,
                error_exponent=error_exponent,
                m_length_exponent=m_length,
                n_length_exponent=n_length,
                modulus_exponent=modulus,
                kernel_family_tags=kernel_tags,
                citation=citation_str,
            )
        )

    return records


def export_math_parameters_json(terms: list[Term]) -> list[dict]:
    """Export math parameters as JSON-serializable dicts."""
    return [r.to_dict() for r in export_math_parameters(terms)]
