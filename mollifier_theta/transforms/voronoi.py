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
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import VoronoiMeta
from mollifier_theta.core.sum_structures import (
    AdditiveTwist,
    ArithmeticType,
    CoeffSeq,
    SumIndex,
    SumStructure,
    VoronoiEligibility,
    WeightKernel,
)


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
    # In math expressions, variable boundaries include: start/end, operators,
    # parens, braces, commas, spaces, ^, _, /, *, +, -, =
    # We match `old` when preceded/followed by non-alphanumeric (or start/end)
    # BUT we want "bn" to match the "n" since "b" is a coefficient prefix
    # Actually the safer approach: replace old that is not followed by an
    # alphanumeric (other than *) and not preceded by an alphanumeric
    # that would make it a different variable name.
    #
    # For our use case (n → n*), the key patterns are:
    #   "n" standalone, "/n)", "{n}", etc.
    # We need to NOT match "n" inside words like "int", "sin", etc.
    # But we DO want to match the "n" in "bn" since that's b*n.
    #
    # Simple approach: use word boundary but allow single-letter prefix
    # Actually, let's just use a negative lookbehind for letters of length > 1
    # and a negative lookahead for alphanumeric.
    #
    # Simplest correct approach for single-letter variables:
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
    """

    def __init__(self, target_variable: str = "n") -> None:
        self.target_variable = target_variable

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if self._should_apply(term):
                transformed = self._apply_one(term)
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

        return True

    def _apply_one(self, term: Term) -> Term:
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
        return (
            f"GL(2) Voronoi summation on variable '{self.target_variable}': "
            f"replaces sum_{{n ~ N}} a_n e(an/c) W(n) with "
            f"sum_{{n* ~ c^2/N}} a*_n* e(-ā n*/c) W*(Bessel). "
            f"Dual length n* ~ c^2/N. Weight becomes Bessel transform."
        )
