"""GL(2) Voronoi summation transform for off-diagonal terms.

Operates between DeltaMethodSetup and DeltaMethodCollapse on terms with
kernel_state=UNCOLLAPSED_DELTA. Applies Voronoi summation to a sum
variable that has an additive twist with modulus c.

Mathematical operation (schematic):
  Sum_{n ~ N} a_n e(±a n/c) W(n)
  →  Sum_{n* ~ N*} a*_n* e(∓ā n*/c) W*(n*/c^2)  + (Bessel branch terms)

where N* = c^2/N (dual length) and W* is the Bessel/Voronoi transform
of the original weight.

This transform does NOT simplify — it replaces sum structure with the
dual sum structure, tracking all Bessel-type kernels explicitly.

Two modes:
  STRUCTURAL_ONLY (default): existing behavior, renames variables/dualizes
  FORMULA: emits TWO terms per eligible input (main term + dual sum) with
           explicit Bessel kernel families and Voronoi main kernel metadata
"""

from __future__ import annotations

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
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import VoronoiKind, VoronoiMeta
from mollifier_theta.core.sum_structures import (
    AdditiveTwist,
    ArithmeticType,
    BesselKernelFamily,
    CoeffSeq,
    SumIndex,
    SumStructure,
    VoronoiEligibility,
    VoronoiMainKernel,
    WeightKernel,
)


# Mapping from ArithmeticType to BesselKernelFamily for formula mode
_BESSEL_FAMILY_MAP: dict[ArithmeticType, BesselKernelFamily] = {
    ArithmeticType.DIVISOR: BesselKernelFamily.J_PLUS_K,
    ArithmeticType.HECKE: BesselKernelFamily.J_BESSEL,
    ArithmeticType.MOLLIFIER: BesselKernelFamily.J_PLUS_K,
}

# Arithmetic types with known Voronoi formulae (formula mode gating)
_FORMULA_ELIGIBLE_TYPES: set[ArithmeticType] = {
    ArithmeticType.DIVISOR,
    ArithmeticType.HECKE,
    ArithmeticType.MOLLIFIER,
}


def _rename_variable_in_string(s: str, old: str, new: str) -> str:
    """Rename a variable in a symbolic expression string (token-aware).

    Replaces standalone occurrences of `old` with `new`, avoiding
    partial matches inside longer identifiers. Uses word-boundary-like
    logic for math expression tokens.

    Examples:
        _rename_variable_in_string("(m/n)^{it}", "n", "n*")
        → "(m/n*)^{it}"
        _rename_variable_in_string("e(-bn/c)", "n", "n*")
        → "e(-bn*/c)"  (the 'n' in 'bn' is still a separate variable)
    """
    import re
    pattern = re.compile(
        r'(?<![a-zA-Z_])' + re.escape(old) + r'(?![a-zA-Z0-9_])'
    )
    return pattern.sub(new, s)


class VoronoiTransform:
    """Apply GL(2) Voronoi summation to the n-variable of an uncollapsed delta-method term.

    Gating conditions (all must hold):
      1. term.kernel_state == UNCOLLAPSED_DELTA
      2. term has sum_structure metadata
      3. sum_structure has a Voronoi-eligible twist

    In FORMULA mode, additional gating:
      4. coefficient arithmetic type must be in {DIVISOR, HECKE, MOLLIFIER}
    """

    def __init__(
        self,
        target_variable: str = "n",
        mode: VoronoiKind = VoronoiKind.STRUCTURAL_ONLY,
    ) -> None:
        self.target_variable = target_variable
        self.mode = mode

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if self._should_apply(term):
                if self.mode == VoronoiKind.FORMULA:
                    transformed = self._apply_one_formula(term)
                    results.extend(transformed)
                    new_terms.extend(transformed)
                else:
                    transformed = self._apply_one_structural(term)
                    results.append(transformed)
                    new_terms.append(transformed)
            else:
                results.append(term)
        ledger.add_many(new_terms)
        return results

    def _should_apply(self, term: Term) -> bool:
        """Check all gating conditions for Voronoi applicability."""
        if term.kernel_state != KernelState.UNCOLLAPSED_DELTA:
            return False

        ss_data = term.metadata.get("sum_structure")
        if not ss_data:
            return False

        ss = SumStructure.model_validate(ss_data)

        # Need a twist on the target variable
        twist = ss.get_twist_for_variable(self.target_variable)
        if twist is None:
            return False

        # Need Voronoi-eligible coefficients on the target variable
        cs = ss.get_coeff_for_variable(self.target_variable)
        if cs is None:
            return False
        if cs.voronoi_eligible != VoronoiEligibility.ELIGIBLE:
            return False

        # Formula mode extra gating
        if self.mode == VoronoiKind.FORMULA:
            if cs.arithmetic_type not in _FORMULA_ELIGIBLE_TYPES:
                return False

        return True

    def _apply_one_structural(self, term: Term) -> Term:
        """Structural-only Voronoi: existing behavior unchanged."""
        ss = SumStructure.model_validate(term.metadata["sum_structure"])
        twist = ss.get_twist_for_variable(self.target_variable)
        cs = ss.get_coeff_for_variable(self.target_variable)

        history = HistoryEntry(
            transform="VoronoiTransform",
            parent_ids=[term.id],
            description=(
                f"GL(2) Voronoi summation applied to variable '{self.target_variable}'. "
                f"Twist e({'+' if twist.sign > 0 else '-'}{twist.numerator}*{self.target_variable}/{twist.modulus}) "
                f"dualized. Coefficient sequence '{cs.name}' replaced by Voronoi dual. "
                f"Weight kernel replaced by Bessel transform."
            ),
        )

        # Build the Voronoi dual kernel
        voronoi_kernel = Kernel(
            name="VoronoiDualKernel",
            support="(0, inf)",
            argument=f"Bessel_transform({self.target_variable}*/{twist.modulus}^2)",
            description=(
                f"Bessel/Voronoi transform of original weight kernel. "
                f"Dual length {self.target_variable}* ~ {twist.modulus}^2 / {self.target_variable}."
            ),
            properties={
                "is_voronoi_dual": True,
                "original_variable": self.target_variable,
                "modulus": twist.modulus,
                "smooth": True,
                "bessel_type": "J+K",  # Both J- and K-Bessel branches
                "dual_length_formula": f"{twist.modulus}^2/{self.target_variable}",
            },
        )

        # Keep original kernels + add the Voronoi dual
        new_kernels = list(term.kernels) + [voronoi_kernel]

        # Update ranges: the target variable's range becomes the dual range
        new_ranges: list[Range] = []
        for r in term.ranges:
            if r.variable == self.target_variable:
                new_ranges.append(Range(
                    variable=f"{self.target_variable}*",
                    lower="1",
                    upper=f"C(T,theta)^2/T^theta",
                    description=(
                        f"Voronoi dual range for {self.target_variable}: "
                        f"{self.target_variable}* ~ c^2/N"
                    ),
                ))
            else:
                new_ranges.append(r)

        # Update variables
        new_variables = [
            f"{v}*" if v == self.target_variable else v
            for v in term.variables
        ]

        # Build new SumStructure reflecting the Voronoi dual
        new_sum_indices = []
        for idx in ss.sum_indices:
            if idx.name == self.target_variable:
                new_sum_indices.append(SumIndex(
                    name=f"{self.target_variable}*",
                    range_lower="1",
                    range_upper=f"C(T,theta)^2/T^theta",
                    range_description=f"Voronoi dual: {self.target_variable}* ~ c^2/N",
                ))
            else:
                new_sum_indices.append(idx)

        new_coeff_seqs = []
        for c in ss.coeff_seqs:
            if c.variable == self.target_variable:
                new_coeff_seqs.append(CoeffSeq(
                    name=f"{cs.name}*",
                    variable=f"{self.target_variable}*",
                    arithmetic_type=cs.arithmetic_type,
                    voronoi_eligible=VoronoiEligibility.INELIGIBLE,
                    norm_bound=f"Voronoi dual of {cs.norm_bound}",
                    description=f"Voronoi dual of {cs.name}",
                    citations=list(cs.citations) + [
                        "Voronoi 1903; Miller-Schmid 2006, Theorem 1.1"
                    ],
                ))
            else:
                new_coeff_seqs.append(c)

        new_twists = []
        for tw in ss.additive_twists:
            if tw.sum_variable == self.target_variable:
                # Voronoi dualizes the twist: e(an/c) -> e(-ā n*/c)
                new_twists.append(AdditiveTwist(
                    modulus=tw.modulus,
                    numerator=tw.numerator,
                    sum_variable=f"{self.target_variable}*",
                    sign=-tw.sign,
                    invert_numerator=True,
                    description=(
                        f"Voronoi dual twist: e({'+' if -tw.sign > 0 else '-'}"
                        f"{tw.numerator}bar*{self.target_variable}*/{tw.modulus})"
                    ),
                ))
            else:
                new_twists.append(tw)

        new_weight_kernels = list(ss.weight_kernels) + [
            WeightKernel(
                kind="bessel_transform",
                original_name="DeltaMethodKernel",
                parameters={
                    "bessel_type": "J+K",
                    "modulus": twist.modulus,
                    "dual_variable": f"{self.target_variable}*",
                },
                description=f"Bessel transform from Voronoi on {self.target_variable}",
            ),
        ]

        new_sum_structure = SumStructure(
            sum_indices=new_sum_indices,
            coeff_seqs=new_coeff_seqs,
            additive_twists=new_twists,
            weight_kernels=new_weight_kernels,
        )

        return Term(
            kind=TermKind.OFF_DIAGONAL,
            expression=(
                f"sum_c sum_{{m,{self.target_variable}*}} "
                f"a_m {cs.name}*({self.target_variable}*) "
                f"e(∓ā{self.target_variable}*/{twist.modulus}) "
                f"W*(Bessel) [Voronoi from {term.expression}]"
            ),
            variables=new_variables,
            ranges=new_ranges,
            kernels=new_kernels,
            phases=self._rename_phase_deps(list(term.phases)),
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            kernel_state=KernelState.VORONOI_APPLIED,
            metadata={
                **term.metadata,
                "voronoi_applied": True,
                "voronoi_target_variable": self.target_variable,
                "voronoi_dual_variable": f"{self.target_variable}*",
                "voronoi_dual_length": f"c^2/T^theta",
                "sum_structure": new_sum_structure.model_dump(),
                "_voronoi": VoronoiMeta(
                    applied=True,
                    target_variable=self.target_variable,
                    dual_variable=f"{self.target_variable}*",
                    dual_length=f"c^2/T^theta",
                ).model_dump(),
            },
        )

    def _apply_one_formula(self, term: Term) -> list[Term]:
        """Formula-mode Voronoi: emit main term + dual sum.

        Returns [main_term, dual_sum] for each eligible input.
        """
        ss = SumStructure.model_validate(term.metadata["sum_structure"])
        twist = ss.get_twist_for_variable(self.target_variable)
        cs = ss.get_coeff_for_variable(self.target_variable)

        bessel_family = _BESSEL_FAMILY_MAP.get(
            cs.arithmetic_type, BesselKernelFamily.UNSPECIFIED,
        )
        argument_structure = f"4*pi*sqrt(m*{self.target_variable}_star)/{twist.modulus}"

        # --- Term 1: Voronoi main term (polar residual) ---
        main_kernel = VoronoiMainKernel(
            arithmetic_type=cs.arithmetic_type,
            modulus=twist.modulus,
            residue_structure="simple_pole",
            test_function="W(x)",
            polar_order=1,
            description=(
                f"Polar residual of Estermann function for {cs.arithmetic_type.value} coefficients"
            ),
        )

        main_history = HistoryEntry(
            transform="VoronoiTransform(FORMULA)",
            parent_ids=[term.id],
            description=(
                f"Voronoi main term (polar residual) from {cs.arithmetic_type.value} "
                f"coefficients on '{self.target_variable}'. "
                f"This is the residual contribution that must be bounded separately."
            ),
        )

        main_term = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression=(
                f"Voronoi main term (polar residual) from "
                f"{cs.arithmetic_type.value} on {self.target_variable} "
                f"[from {term.expression}]"
            ),
            variables=list(term.variables),
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=[],  # Main term has no oscillatory phases
            history=list(term.history) + [main_history],
            parents=[term.id],
            status=TermStatus.MAIN_TERM,
            multiplicity=term.multiplicity,
            kernel_state=KernelState.VORONOI_APPLIED,
            metadata={
                **term.metadata,
                "voronoi_applied": True,
                "voronoi_main_term": True,
                "voronoi_main_kernel": main_kernel.model_dump(),
                "_voronoi": VoronoiMeta(
                    applied=True,
                    target_variable=self.target_variable,
                    dual_variable="",
                    dual_length="",
                    kind=VoronoiKind.FORMULA,
                ).model_dump(),
            },
        )

        # --- Term 2: Voronoi dual sum ---
        dual_history = HistoryEntry(
            transform="VoronoiTransform(FORMULA)",
            parent_ids=[term.id],
            description=(
                f"GL(2) Voronoi dual sum for variable '{self.target_variable}'. "
                f"Bessel family: {bessel_family.value}. "
                f"Argument structure: {argument_structure}. "
                f"Coefficient sequence '{cs.name}' replaced by Voronoi dual."
            ),
        )

        # Build dual kernel with explicit Bessel family
        voronoi_kernel = Kernel(
            name="VoronoiDualKernel",
            support="(0, inf)",
            argument=f"Bessel_transform({self.target_variable}*/{twist.modulus}^2)",
            description=(
                f"Bessel/Voronoi transform of original weight kernel. "
                f"Family: {bessel_family.value}. "
                f"Dual length {self.target_variable}* ~ {twist.modulus}^2 / {self.target_variable}."
            ),
            properties={
                "is_voronoi_dual": True,
                "original_variable": self.target_variable,
                "modulus": twist.modulus,
                "smooth": True,
                "bessel_type": bessel_family.value,
                "bessel_family": bessel_family.value,
                "argument_structure": argument_structure,
                "dual_length_formula": f"{twist.modulus}^2/{self.target_variable}",
            },
        )

        new_kernels = list(term.kernels) + [voronoi_kernel]

        new_ranges: list[Range] = []
        for r in term.ranges:
            if r.variable == self.target_variable:
                new_ranges.append(Range(
                    variable=f"{self.target_variable}*",
                    lower="1",
                    upper=f"C(T,theta)^2/T^theta",
                    description=(
                        f"Voronoi dual range for {self.target_variable}: "
                        f"{self.target_variable}* ~ c^2/N"
                    ),
                ))
            else:
                new_ranges.append(r)

        new_variables = [
            f"{v}*" if v == self.target_variable else v
            for v in term.variables
        ]

        # Build dual SumStructure
        new_sum_indices = []
        for idx in ss.sum_indices:
            if idx.name == self.target_variable:
                new_sum_indices.append(SumIndex(
                    name=f"{self.target_variable}*",
                    range_lower="1",
                    range_upper=f"C(T,theta)^2/T^theta",
                    range_description=f"Voronoi dual: {self.target_variable}* ~ c^2/N",
                ))
            else:
                new_sum_indices.append(idx)

        new_coeff_seqs = []
        for c in ss.coeff_seqs:
            if c.variable == self.target_variable:
                new_coeff_seqs.append(CoeffSeq(
                    name=f"{cs.name}*",
                    variable=f"{self.target_variable}*",
                    arithmetic_type=cs.arithmetic_type,
                    voronoi_eligible=VoronoiEligibility.INELIGIBLE,
                    norm_bound=f"Voronoi dual of {cs.norm_bound}",
                    description=f"Voronoi dual of {cs.name}",
                    citations=list(cs.citations) + [
                        "Voronoi 1903; Miller-Schmid 2006, Theorem 1.1"
                    ],
                ))
            else:
                new_coeff_seqs.append(c)

        new_twists = []
        for tw in ss.additive_twists:
            if tw.sum_variable == self.target_variable:
                new_twists.append(AdditiveTwist(
                    modulus=tw.modulus,
                    numerator=tw.numerator,
                    sum_variable=f"{self.target_variable}*",
                    sign=-tw.sign,
                    invert_numerator=True,
                    description=(
                        f"Voronoi dual twist: e({'+' if -tw.sign > 0 else '-'}"
                        f"{tw.numerator}bar*{self.target_variable}*/{tw.modulus})"
                    ),
                ))
            else:
                new_twists.append(tw)

        new_weight_kernels = list(ss.weight_kernels) + [
            WeightKernel(
                kind="bessel_transform",
                original_name="DeltaMethodKernel",
                bessel_family=bessel_family,
                argument_structure=argument_structure,
                parameters={
                    "bessel_type": bessel_family.value,
                    "modulus": twist.modulus,
                    "dual_variable": f"{self.target_variable}*",
                },
                description=f"Bessel transform from Voronoi on {self.target_variable}",
            ),
        ]

        new_sum_structure = SumStructure(
            sum_indices=new_sum_indices,
            coeff_seqs=new_coeff_seqs,
            additive_twists=new_twists,
            weight_kernels=new_weight_kernels,
        )

        dual_term = Term(
            kind=TermKind.OFF_DIAGONAL,
            expression=(
                f"sum_c sum_{{m,{self.target_variable}*}} "
                f"a_m {cs.name}*({self.target_variable}*) "
                f"e(∓ā{self.target_variable}*/{twist.modulus}) "
                f"W*(Bessel:{bessel_family.value}) "
                f"[Formula Voronoi from {term.expression}]"
            ),
            variables=new_variables,
            ranges=new_ranges,
            kernels=new_kernels,
            phases=self._rename_phase_deps(list(term.phases)),
            history=list(term.history) + [dual_history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            kernel_state=KernelState.VORONOI_APPLIED,
            metadata={
                **term.metadata,
                "voronoi_applied": True,
                "voronoi_target_variable": self.target_variable,
                "voronoi_dual_variable": f"{self.target_variable}*",
                "voronoi_dual_length": f"c^2/T^theta",
                "sum_structure": new_sum_structure.model_dump(),
                "_voronoi": VoronoiMeta(
                    applied=True,
                    target_variable=self.target_variable,
                    dual_variable=f"{self.target_variable}*",
                    dual_length=f"c^2/T^theta",
                    kind=VoronoiKind.FORMULA,
                ).model_dump(),
            },
        )

        return [main_term, dual_term]

    def _rename_phase_deps(self, phases: list[Phase]) -> list[Phase]:
        """Rename target variable → dual variable in phase depends_on AND expression."""
        dual = f"{self.target_variable}*"
        result: list[Phase] = []
        for p in phases:
            if self.target_variable in p.depends_on:
                new_deps = [
                    dual if v == self.target_variable else v
                    for v in p.depends_on
                ]
                # Also rename in expression string for consistency
                new_expr = _rename_variable_in_string(
                    p.expression, self.target_variable, dual,
                )
                result.append(Phase(
                    expression=new_expr,
                    is_separable=p.is_separable,
                    absorbed=p.absorbed,
                    depends_on=new_deps,
                    unit_modulus=p.unit_modulus,
                ))
            else:
                result.append(p)
        return result

    def describe(self) -> str:
        mode_desc = f" (mode={self.mode.value})" if self.mode != VoronoiKind.STRUCTURAL_ONLY else ""
        return (
            f"GL(2) Voronoi summation on variable '{self.target_variable}'{mode_desc}: "
            f"replaces sum_{{n ~ N}} a_n e(an/c) W(n) with "
            f"sum_{{n* ~ c^2/N}} a*_n* e(-ā n*/c) W*(Bessel). "
            f"Dual length n* ~ c^2/N. Weight becomes Bessel transform."
        )
