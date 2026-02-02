"""Tests for formula-mode Voronoi transform (Wave 1A)."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import (
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import VoronoiKind, VoronoiMeta, get_voronoi_meta
from mollifier_theta.core.sum_structures import (
    AdditiveTwist,
    ArithmeticType,
    BesselKernelFamily,
    CoeffSeq,
    SumIndex,
    SumStructure,
    VoronoiEligibility,
    WeightKernel,
)
from mollifier_theta.transforms.delta_method import DeltaMethodSetup
from mollifier_theta.transforms.voronoi import VoronoiTransform


@pytest.fixture
def formula_eligible_term() -> Term:
    """An off-diagonal term with divisor coefficients — eligible for formula Voronoi."""
    term = Term(
        kind=TermKind.OFF_DIAGONAL,
        expression="OFFDIAG test",
        variables=["m", "n"],
        ranges=[
            Range(variable="m", lower="1", upper="T^theta"),
            Range(variable="n", lower="1", upper="T^theta"),
        ],
        kernels=[Kernel(name="W_AFE"), Kernel(name="FourierKernel")],
        phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
    )
    # Run through DeltaMethodSetup to get sum_structure
    ledger = TermLedger()
    ledger.add(term)
    results = DeltaMethodSetup().apply([term], ledger)
    return results[0]


@pytest.fixture
def generic_eligible_term() -> Term:
    """An off-diagonal term with GENERIC arithmetic type — NOT formula-eligible."""
    ss = SumStructure(
        sum_indices=[
            SumIndex(name="m", range_upper="T^theta"),
            SumIndex(name="n", range_upper="T^theta"),
        ],
        coeff_seqs=[
            CoeffSeq(
                name="a_m", variable="m",
                arithmetic_type=ArithmeticType.GENERIC,
                voronoi_eligible=VoronoiEligibility.ELIGIBLE,
            ),
            CoeffSeq(
                name="b_n", variable="n",
                arithmetic_type=ArithmeticType.GENERIC,
                voronoi_eligible=VoronoiEligibility.ELIGIBLE,
            ),
        ],
        additive_twists=[
            AdditiveTwist(modulus="c", numerator="a", sum_variable="m"),
            AdditiveTwist(modulus="c", numerator="b", sum_variable="n", sign=-1),
        ],
    )
    return Term(
        kind=TermKind.OFF_DIAGONAL,
        variables=["m", "n"],
        kernel_state=KernelState.UNCOLLAPSED_DELTA,
        kernels=[Kernel(name="W_AFE")],
        phases=[Phase(expression="(m/n)^{it}", depends_on=["m", "n"])],
        metadata={
            "sum_structure": ss.model_dump(),
            "delta_method_applied": True,
        },
    )


class TestStructuralModeUnchanged:
    """Structural mode must remain backward-compatible."""

    def test_structural_mode_produces_one_term(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.STRUCTURAL_ONLY)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        # One term per input (structural mode)
        assert len(results) == 1

    def test_structural_default_mode(self) -> None:
        v = VoronoiTransform()
        assert v.mode == VoronoiKind.STRUCTURAL_ONLY

    def test_structural_voronoi_meta_kind(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.STRUCTURAL_ONLY)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        vm = get_voronoi_meta(results[0])
        assert vm is not None
        assert vm.kind == VoronoiKind.STRUCTURAL_ONLY


class TestFormulaModeBasic:
    """Formula mode must emit two terms per eligible input."""

    def test_formula_emits_two_terms(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        assert len(results) == 2

    def test_first_term_is_main(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        main = results[0]
        assert main.status == TermStatus.MAIN_TERM
        assert main.metadata.get("voronoi_main_term") is True

    def test_second_term_is_dual(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        dual = results[1]
        assert dual.status == TermStatus.ACTIVE
        assert dual.kind == TermKind.OFF_DIAGONAL

    def test_dual_has_formula_voronoi_kind(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        dual = results[1]
        vm = get_voronoi_meta(dual)
        assert vm is not None
        assert vm.kind == VoronoiKind.FORMULA


class TestFormulaModeBessel:
    """Dual term must carry explicit Bessel kernel family."""

    def test_dual_has_bessel_family_in_weight_kernel(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        dual = results[1]
        ss = SumStructure.model_validate(dual.metadata["sum_structure"])
        bessel_wks = [wk for wk in ss.weight_kernels if wk.kind == "bessel_transform"]
        assert len(bessel_wks) >= 1
        assert bessel_wks[0].bessel_family != BesselKernelFamily.UNSPECIFIED

    def test_dual_kernel_properties_have_bessel(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        dual = results[1]
        voronoi_kernels = [k for k in dual.kernels if k.name == "VoronoiDualKernel"]
        assert len(voronoi_kernels) == 1
        props = voronoi_kernels[0].properties
        assert "bessel_family" in props
        assert "argument_structure" in props

    def test_bessel_argument_structure(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        dual = results[1]
        ss = SumStructure.model_validate(dual.metadata["sum_structure"])
        bessel_wk = [wk for wk in ss.weight_kernels if wk.kind == "bessel_transform"][0]
        assert "4*pi*sqrt" in bessel_wk.argument_structure


class TestFormulaGating:
    """Formula mode must reject GENERIC/UNKNOWN arithmetic types."""

    def test_rejects_generic_type(self, generic_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([generic_eligible_term], ledger)
        # Should pass through unchanged (not applied)
        assert len(results) == 1
        assert results[0].kernel_state == KernelState.UNCOLLAPSED_DELTA

    def test_structural_accepts_generic(self, generic_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.STRUCTURAL_ONLY)
        ledger = TermLedger()
        results = v.apply([generic_eligible_term], ledger)
        # Structural mode should apply (no arithmetic type gating)
        assert len(results) == 1
        assert results[0].kernel_state == KernelState.VORONOI_APPLIED


class TestFormulaModeMainTerm:
    """Main term must have correct metadata."""

    def test_main_term_has_voronoi_main_kernel(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        main = results[0]
        assert "voronoi_main_kernel" in main.metadata

    def test_main_term_no_oscillatory_phases(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        main = results[0]
        assert len(main.phases) == 0

    def test_main_term_voronoi_applied(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        main = results[0]
        assert main.kernel_state == KernelState.VORONOI_APPLIED


class TestFormulaSerialization:
    """Formula mode terms must survive round-trip serialization."""

    def test_dual_term_round_trip(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        dual = results[1]
        dumped = dual.model_dump()
        restored = Term(**dumped)
        assert restored.kind == dual.kind
        assert restored.kernel_state == dual.kernel_state
        vm = get_voronoi_meta(restored)
        assert vm is not None
        assert vm.kind == VoronoiKind.FORMULA

    def test_main_term_round_trip(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        main = results[0]
        dumped = main.model_dump()
        restored = Term(**dumped)
        assert restored.status == TermStatus.MAIN_TERM
        assert restored.metadata.get("voronoi_main_term") is True


class TestNonMatchingPassthrough:
    """Non-matching terms should pass through unchanged in formula mode."""

    def test_non_matching_unchanged(self) -> None:
        term = Term(kind=TermKind.DIAGONAL, expression="diag")
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([term], ledger)
        assert len(results) == 1
        assert results[0].id == term.id


class TestTwistAuthority:
    """SumStructure.additive_twists is sole twist authority."""

    def test_dual_sum_twists_from_sum_structure(self, formula_eligible_term: Term) -> None:
        v = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        results = v.apply([formula_eligible_term], ledger)
        dual = results[1]
        ss = SumStructure.model_validate(dual.metadata["sum_structure"])
        # Must have dual twist on n*
        n_star_twists = [tw for tw in ss.additive_twists if tw.sum_variable == "n*"]
        assert len(n_star_twists) >= 1
        assert n_star_twists[0].invert_numerator is True
