"""Length-aware DI bound strategy that reads sum lengths from metadata.

Implements BoundStrategy.  For terms with Voronoi-applied metadata and
dual-length information in VoronoiMeta, uses the asymmetric DI model.
For symmetric terms, uses the standard model.
"""

from __future__ import annotations

from mollifier_theta.analysis.exponent_model import ExponentConstraint
from mollifier_theta.analysis.length_aware_di import LengthAwareDIModel
from mollifier_theta.core.ir import (
    HistoryEntry,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.scale_model import ScaleModel
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    VoronoiKind,
    get_voronoi_meta,
)


CITATION = (
    "Deshouillers-Iwaniec 1982/83, Theorem 12; "
    "length-aware parametric model"
)


class LengthAwareDIBound:
    """DI bound that reads sum lengths from SumStructure metadata.

    For terms where Voronoi summation has been applied (structurally)
    and dual-length information is available, uses the asymmetric
    LengthAwareDIModel.  Otherwise falls back to the symmetric model.
    """

    @property
    def name(self) -> str:
        return "LengthAwareDI"

    @property
    def citation(self) -> str:
        return CITATION

    def applies(self, term: Term) -> bool:
        """Gate: KLOOSTERMAN + ACTIVE + kloosterman_form."""
        return (
            term.kind == TermKind.KLOOSTERMAN
            and term.status == TermStatus.ACTIVE
            and term.metadata.get("kloosterman_form", False)
        )

    def _extract_model(self, term: Term) -> LengthAwareDIModel:
        """Determine the appropriate DI model from term metadata."""
        vm = get_voronoi_meta(term)
        if (
            vm is not None
            and vm.applied
            and vm.dual_length
            and vm.kind == VoronoiKind.STRUCTURAL_ONLY
        ):
            return LengthAwareDIModel.voronoi_dual()
        return LengthAwareDIModel.symmetric()

    def bound(self, term: Term) -> Term:
        model = self._extract_model(term)
        error_expr = model.error_exponent_str

        scale = ScaleModel(
            T_exponent=error_expr,
            description=f"Length-aware DI bound ({model.label})",
        )

        history = HistoryEntry(
            transform="LengthAwareDIBound",
            parent_ids=[term.id],
            description=(
                f"Applied length-aware DI bound ({model.label}). "
                f"alpha={model.alpha_str}, beta={model.beta_str}, "
                f"gamma={model.gamma_str}. "
                f"Error exponent E(theta) = {error_expr}."
            ),
        )

        return Term(
            kind=TermKind.KLOOSTERMAN,
            expression=(
                f"LengthAwareDI bound ({model.label}): "
                f"T^({error_expr}) [from {term.expression}]"
            ),
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=list(term.phases),
            scale_model=scale.to_str(),
            status=TermStatus.BOUND_ONLY,
            history=list(term.history) + [history],
            parents=[term.id],
            lemma_citation=self.citation,
            multiplicity=term.multiplicity,
            kernel_state=term.kernel_state,
            metadata={
                **term.metadata,
                "length_aware_di_bound": True,
                "di_model_label": model.label,
                "error_exponent": error_expr,
                "scale_model_dict": scale.to_dict(),
                "_bound": BoundMeta(
                    strategy="LengthAwareDI",
                    error_exponent=error_expr,
                    citation=self.citation,
                    bound_family=f"LengthAwareDI_{model.label}",
                ).model_dump(),
            },
        )

    def constraints(self) -> list[ExponentConstraint]:
        """Constraints from both symmetric and voronoi-dual models."""
        sym = LengthAwareDIModel.symmetric()
        dual = LengthAwareDIModel.voronoi_dual()
        return sym.constraints() + dual.constraints()
