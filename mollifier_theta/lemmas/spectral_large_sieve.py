"""Spectral large sieve bound for spectralized off-diagonal terms.

Implements MultiBoundStrategy producing 3 BoundOnly terms (regime case tree):
  1. Small modulus (c ≤ N^{1/2}): from lem:ng-large-sieve
  2. Large modulus (c > N^{1/2}): from lem:large-sieve
  3. Bessel transition: where Bessel kernel asymptotics change character

Source: Iwaniec-Kowalski Thm 16.62; sixth-moment draft lem:large-sieve,
        lem:ng-large-sieve, prop:refined-kuznetsov.
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    KernelState,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.scale_model import ScaleModel, theta
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    VoronoiKind,
    get_kuznetsov_meta,
    get_voronoi_meta,
)
from mollifier_theta.analysis.exponent_model import ExponentConstraint


# Case definitions: each case has a name, exponent expression, and description
_CASES = [
    {
        "case_id": "small_modulus",
        "description": (
            "Small modulus regime (c ≤ N^{1/2}): "
            "From lem:ng-large-sieve, the (LK+N)^{1+ε} bound gives "
            "effective exponent with K small."
        ),
        # E(θ) = (5θ + 1)/4: comes from conductor analysis in the small-c regime
        # E(θ) < 1 iff 5θ+1 < 4 iff θ < 3/5
        "exponent_str": "(5*theta + 1)/4",
        "citation": (
            "Iwaniec-Kowalski Thm 16.62; "
            "sixth-moment draft lem:ng-large-sieve (L652)"
        ),
    },
    {
        "case_id": "large_modulus",
        "description": (
            "Large modulus regime (c > N^{1/2}): "
            "From lem:large-sieve, the (LT²+N)(LN)^ε bound with LT² dominant. "
            "Standard spectral large sieve inequality."
        ),
        # E(θ) = (3θ + 1)/2: comes from the LT²+N balance in the large-c regime
        # E(θ) < 1 iff 3θ+1 < 2 iff θ < 1/3
        "exponent_str": "(3*theta + 1)/2",
        "citation": (
            "Iwaniec-Kowalski Thm 16.62; "
            "sixth-moment draft lem:large-sieve (L622)"
        ),
    },
    {
        "case_id": "bessel_transition",
        "description": (
            "Bessel transition regime: where Bessel kernel asymptotics change "
            "from stationary phase to oscillatory decay. Intermediate conductor range."
        ),
        # E(θ) = 7θ/4: same as DI in this transitional regime
        # This is actually the bridging case
        "exponent_str": "7*theta/4",
        "citation": (
            "Iwaniec-Kowalski Ch. 16; "
            "sixth-moment draft lem:transition (L1556)"
        ),
    },
]

SPECTRAL_LARGE_SIEVE_CITATION = (
    "Iwaniec-Kowalski Thm 16.62; "
    "sixth-moment draft lem:large-sieve, lem:ng-large-sieve, lem:transition"
)


class SpectralLargeSieveBound:
    """Spectral large sieve bound with regime case tree.

    Produces 3 BoundOnly terms per eligible input, one per regime.
    The pipeline's theta_admissible naturally takes the min across all
    case terms — the binding case determines theta_max.
    """

    @property
    def name(self) -> str:
        return "SpectralLargeSieve"

    @property
    def citation(self) -> str:
        return SPECTRAL_LARGE_SIEVE_CITATION

    def applies(self, term: Term) -> bool:
        """Gate: SPECTRALIZED terms with formula Voronoi and Kuznetsov metadata."""
        if term.kernel_state != KernelState.SPECTRALIZED:
            return False
        if term.status != TermStatus.ACTIVE:
            return False

        # Red Flag B: require formula Voronoi
        vm = get_voronoi_meta(term)
        if vm is None or vm.kind != VoronoiKind.FORMULA:
            return False

        # Must have Kuznetsov metadata
        km = get_kuznetsov_meta(term)
        if km is None or not km.applied:
            return False

        return True

    def bound_multi(self, term: Term) -> list[Term]:
        """Produce 3 BoundOnly terms (one per regime case)."""
        result: list[Term] = []

        for case in _CASES:
            scale = ScaleModel(
                T_exponent=case["exponent_str"],
                description=f"SpectralLargeSieve ({case['case_id']}): E(θ) = {case['exponent_str']}",
                sub_exponents={
                    "spectral_window": "K" if case["case_id"] == "small_modulus" else "T^2",
                    "coefficient_l2": "theta",
                    "modulus_count": "1 - theta",
                },
            )

            history = HistoryEntry(
                transform=f"SpectralLargeSieveBound({case['case_id']})",
                parent_ids=[term.id],
                description=(
                    f"Applied spectral large sieve bound ({case['case_id']}). "
                    f"Error exponent E(θ) = {case['exponent_str']}. "
                    f"{case['description']}"
                ),
            )

            bound_meta = BoundMeta(
                strategy="SpectralLargeSieve",
                error_exponent=case["exponent_str"],
                citation=case["citation"],
                bound_family="SpectralLargeSieve",
                case_id=case["case_id"],
                case_description=case["description"],
            )

            bound_term = Term(
                kind=TermKind.SPECTRAL,
                expression=(
                    f"SpectralLargeSieve ({case['case_id']}): "
                    f"T^({case['exponent_str']}) [from {term.expression}]"
                ),
                variables=list(term.variables),
                ranges=list(term.ranges),
                kernels=list(term.kernels),
                phases=list(term.phases),
                scale_model=scale.to_str(),
                status=TermStatus.BOUND_ONLY,
                history=list(term.history) + [history],
                parents=[term.id],
                lemma_citation=case["citation"],
                multiplicity=term.multiplicity,
                kernel_state=term.kernel_state,
                metadata={
                    **term.metadata,
                    "bound_strategy": "SpectralLargeSieve",
                    "error_exponent": case["exponent_str"],
                    "scale_model_dict": scale.to_dict(),
                    "_bound": bound_meta.model_dump(),
                },
            )
            result.append(bound_term)

        return result

    def constraints(self) -> list[ExponentConstraint]:
        """Return all exponent constraints from the case tree."""
        return [
            ExponentConstraint(
                name=f"spectral_large_sieve_{case['case_id']}",
                expression_str=case["exponent_str"],
                description=case["description"],
                citation=case["citation"],
                bound_family="SpectralLargeSieve",
            )
            for case in _CASES
        ]
