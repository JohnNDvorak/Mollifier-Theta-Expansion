"""Delta method insertion for off-diagonal terms.

Off-diagonal -> sum over moduli c with additive character / Ramanujan sum structure.
The smooth kernel from the delta method is tracked as a first-class Kernel.

Two-stage architecture:
  - DeltaMethodSetup: introduce modulus c + integral-form kernel (uncollapsed)
  - DeltaMethodCollapse: stationary phase -> additive characters e(am/c), e(-bn/c)
  - DeltaMethodInsert: backward-compatible wrapper that chains both stages
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
from mollifier_theta.core.phase_ast import Div, Mul, Sub, Var, build_additive_twist
from mollifier_theta.core.stage_meta import DeltaMethodMeta
from mollifier_theta.core.sum_structures import (
    AdditiveTwist,
    ArithmeticType,
    CoeffSeq,
    SumIndex,
    SumStructure,
    VoronoiEligibility,
    WeightKernel,
)


class DeltaMethodSetup:
    """Stage 1: introduce modulus variable c and integral-form kernel.

    Does NOT add additive character phases. The kernel is in uncollapsed
    (integral) form, creating an extension point for future transforms.
    """

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if term.kind == TermKind.OFF_DIAGONAL:
                transformed = self._apply_one(term)
                results.append(transformed)
                new_terms.append(transformed)
            else:
                results.append(term)
        ledger.add_many(new_terms)
        return results

    def _apply_one(self, term: Term) -> Term:
        history = HistoryEntry(
            transform="DeltaMethodSetup",
            parent_ids=[term.id],
            description=(
                "Delta method setup: introduced sum over moduli c with "
                "integral-form delta-method kernel (uncollapsed)."
            ),
        )

        delta_kernel = Kernel(
            name="DeltaMethodKernel",
            support="(0, inf)",
            argument="integral h(x) e(x(am-bn)/cq) dx",
            description=(
                "Integral-form kernel from delta method. Not yet collapsed "
                "via stationary phase."
            ),
            properties={
                "is_delta_method": True,
                "smooth": True,
                "compact_support_in_c": False,
                "collapsed": False,
                "test_function": "h",
                "oscillatory_argument": "x(am-bn)/cq",
                "collapse_conditions": [
                    "stationary_phase_valid",
                    "test_function_smooth",
                ],
            },
        )

        new_variables = list(term.variables) + ["c"]
        new_ranges = list(term.ranges) + [
            Range(
                variable="c",
                lower="1",
                upper="C(T,theta)",
                description="Modulus range from delta method, C ~ T^{1+epsilon}/y where y = T^theta",
            )
        ]

        new_kernels = list(term.kernels) + [delta_kernel]

        # Build structured AST for the oscillatory argument: (a*m - b*n) / c
        osc_ast = Div(
            numerator=Sub(
                left=Mul(left=Var(name="a"), right=Var(name="m")),
                right=Mul(left=Var(name="b"), right=Var(name="n")),
            ),
            denominator=Var(name="c"),
        )

        # Build structured sum description for Voronoi pattern-matching
        sum_structure = SumStructure(
            sum_indices=[
                SumIndex(name="m", range_upper="T^theta",
                         range_description="Mollifier summation range"),
                SumIndex(name="n", range_upper="T^theta",
                         range_description="Mollifier summation range"),
                SumIndex(name="c", range_lower="1", range_upper="C(T,theta)",
                         range_description="Modulus range from delta method"),
            ],
            coeff_seqs=[
                CoeffSeq(
                    name="a_m", variable="m",
                    arithmetic_type=ArithmeticType.MOLLIFIER,
                    voronoi_eligible=VoronoiEligibility.ELIGIBLE,
                    norm_bound="||a||_2 << T^{theta/2+eps}",
                    description="Mollifier coefficients mu(m) * P(log m / log y)",
                ),
                CoeffSeq(
                    name="b_n", variable="n",
                    arithmetic_type=ArithmeticType.MOLLIFIER,
                    voronoi_eligible=VoronoiEligibility.ELIGIBLE,
                    norm_bound="||b||_2 << T^{theta/2+eps}",
                    description="Conjugate mollifier coefficients",
                ),
            ],
            additive_twists=[
                AdditiveTwist(
                    modulus="c", numerator="a", sum_variable="m",
                    sign=1, description="Twist from delta method: e(am/c)",
                ),
                AdditiveTwist(
                    modulus="c", numerator="b", sum_variable="n",
                    sign=-1, description="Twist from delta method: e(-bn/c)",
                ),
            ],
            weight_kernels=[
                WeightKernel(
                    kind="smooth", original_name="DeltaMethodKernel",
                    description="Integral-form kernel h(x)e(x(am-bn)/cq)",
                ),
            ],
        )

        return Term(
            kind=TermKind.OFF_DIAGONAL,
            expression=f"sum_c int h(x) e(x(am-bn)/cq) dx ... [from {term.expression}]",
            variables=new_variables,
            ranges=new_ranges,
            kernels=new_kernels,
            phases=list(term.phases),
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            metadata={
                **term.metadata,
                "delta_method_applied": True,
                "delta_method_collapsed": False,
                "delta_method_stage": "setup",
                "modulus_variable": "c",
                "oscillatory_ast": osc_ast.model_dump(),
                "sum_structure": sum_structure.model_dump(),
                "_delta": DeltaMethodMeta(
                    applied=True, collapsed=False,
                    stage="setup", modulus_variable="c",
                ).model_dump(),
            },
        )

    def describe(self) -> str:
        return (
            "Delta method setup: off-diagonal -> sum over moduli c with "
            "integral-form kernel h(x)e(x(am-bn)/cq). Uncollapsed — "
            "additive characters not yet separated."
        )


class DeltaMethodCollapse:
    """Stage 2: stationary phase collapse -> additive characters.

    Takes terms with delta_method_applied=True and delta_method_collapsed=False,
    adds phases e(am/c) and e(-bn/c), and collapses the kernel.
    """

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        new_terms: list[Term] = []
        for term in terms:
            if term.kernel_state in (
                KernelState.UNCOLLAPSED_DELTA,
                KernelState.VORONOI_APPLIED,
            ):
                transformed = self._apply_one(term)
                results.append(transformed)
                new_terms.append(transformed)
            else:
                results.append(term)
        ledger.add_many(new_terms)
        return results

    def _apply_one(self, term: Term) -> Term:
        # Data-driven phase construction from SumStructure (WI-4).
        # Reads additive_twists from the SumStructure metadata to build
        # correct phases that reference the current variable names
        # (e.g. n* after Voronoi, not n).
        ss_data = term.metadata.get("sum_structure")
        if ss_data:
            ss = SumStructure.model_validate(ss_data)
            new_phases_from_twists = self._phases_from_sum_structure(ss)
            description = (
                "Delta method collapse: stationary phase applied, "
                "additive character phases built from SumStructure."
            )
        else:
            # Fallback for terms without SumStructure: hard-coded behavior
            new_phases_from_twists = [
                Phase(
                    expression="e(am/c)",
                    depends_on=["m", "c"],
                    is_separable=True,
                    unit_modulus=True,
                ),
                Phase(
                    expression="e(-bn/c)",
                    depends_on=["n", "c"],
                    is_separable=True,
                    unit_modulus=True,
                ),
            ]
            description = (
                "Delta method collapse: stationary phase applied, kernel "
                "collapsed to additive characters e(am/c) and e(-bn/c). "
                "(Fallback: no SumStructure metadata found.)"
            )

        history = HistoryEntry(
            transform="DeltaMethodCollapse",
            parent_ids=[term.id],
            description=description,
        )

        # Build the kernel argument string from the actual variable names.
        # After Voronoi, "n" may have become "n*" in the SumStructure.
        if ss_data:
            # Use sum index names for the kernel argument
            idx_names = [idx.name for idx in ss.sum_indices if idx.name != "c"]
            if len(idx_names) >= 2:
                kernel_arg = f"(a*{idx_names[0]}-b*{idx_names[1]})/c"
            else:
                kernel_arg = "(am-bn)/c"
        else:
            kernel_arg = "(am-bn)/c"

        # Update the DeltaMethodKernel to collapsed form
        new_kernels: list[Kernel] = []
        for k in term.kernels:
            if k.name == "DeltaMethodKernel" and not k.properties.get("collapsed", True):
                collapsed_kernel = Kernel(
                    name="DeltaMethodKernel",
                    support=k.support,
                    argument=kernel_arg,
                    description=(
                        "Smooth kernel from delta method approximation to "
                        "delta(am-bn). NOT a literal delta function."
                    ),
                    properties={
                        **k.properties,
                        "collapsed": True,
                    },
                )
                new_kernels.append(collapsed_kernel)
            else:
                new_kernels.append(k)

        new_phases = list(term.phases) + new_phases_from_twists

        return Term(
            kind=TermKind.OFF_DIAGONAL,
            expression=f"sum_c sum_{{m,n}} a_m b_n e((am-bn)/c) V(...) [from {term.expression}]",
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=new_kernels,
            phases=new_phases,
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            kernel_state=KernelState.COLLAPSED,
            metadata={
                **term.metadata,
                "delta_method_applied": True,
                "delta_method_collapsed": True,
                "delta_method_stage": "collapsed",
                "modulus_variable": "c",
                "_delta": DeltaMethodMeta(
                    applied=True, collapsed=True,
                    stage="collapsed", modulus_variable="c",
                ).model_dump(),
            },
        )

    @staticmethod
    def _phases_from_sum_structure(ss: SumStructure) -> list[Phase]:
        """Build additive character phases from SumStructure twists."""
        phases: list[Phase] = []
        for twist in ss.additive_twists:
            phases.append(Phase(
                expression=twist.format_phase_expression(),
                depends_on=[twist.sum_variable, twist.modulus],
                is_separable=True,
                unit_modulus=True,
            ))
        return phases

    def describe(self) -> str:
        return (
            "Delta method collapse: stationary phase reduces integral-form "
            "kernel to additive characters e(am/c), e(-bn/c)."
        )


class DeltaMethodInsert:
    """Backward-compatible wrapper: chains DeltaMethodSetup + DeltaMethodCollapse.

    Produces identical output to the original single-stage implementation
    (same phases, kernels, metadata keys, variables, ranges) except:
    - Two history entries instead of one
    - Extra metadata keys (delta_method_collapsed, delta_method_stage)
    """

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        setup = DeltaMethodSetup()
        collapse = DeltaMethodCollapse()
        intermediate = setup.apply(terms, ledger)
        return collapse.apply(intermediate, ledger)

    def describe(self) -> str:
        return (
            "Delta method: off-diagonal -> sum over moduli c with additive "
            "characters e(am/c), e(-bn/c). Smooth delta-method kernel "
            "tracked as first-class Kernel object."
        )
