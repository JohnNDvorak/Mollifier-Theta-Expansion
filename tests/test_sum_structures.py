"""Tests for SumIndex, CoeffSeq, AdditiveTwist, SumStructure IR types."""

from __future__ import annotations

from mollifier_theta.core.ir import Kernel, Phase, Range, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
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
from mollifier_theta.transforms.delta_method import DeltaMethodSetup


class TestSumStructureBasics:
    def test_sum_index_creation(self) -> None:
        idx = SumIndex(name="n", range_upper="T^theta")
        assert idx.name == "n"
        assert idx.range_upper == "T^theta"

    def test_coeff_seq_creation(self) -> None:
        cs = CoeffSeq(
            name="a_m", variable="m",
            arithmetic_type=ArithmeticType.MOLLIFIER,
            voronoi_eligible=VoronoiEligibility.ELIGIBLE,
        )
        assert cs.voronoi_eligible == VoronoiEligibility.ELIGIBLE
        assert cs.arithmetic_type == ArithmeticType.MOLLIFIER

    def test_additive_twist_creation(self) -> None:
        tw = AdditiveTwist(
            modulus="c", numerator="a", sum_variable="n", sign=1,
        )
        assert tw.sum_variable == "n"
        assert tw.sign == 1

    def test_weight_kernel_creation(self) -> None:
        wk = WeightKernel(kind="smooth", original_name="DeltaMethodKernel")
        assert wk.kind == "smooth"


class TestSumStructureLookup:
    def test_get_twist_for_variable(self) -> None:
        ss = SumStructure(
            additive_twists=[
                AdditiveTwist(modulus="c", numerator="a", sum_variable="m", sign=1),
                AdditiveTwist(modulus="c", numerator="b", sum_variable="n", sign=-1),
            ],
        )
        tw = ss.get_twist_for_variable("n")
        assert tw is not None
        assert tw.sign == -1

    def test_get_twist_missing(self) -> None:
        ss = SumStructure()
        assert ss.get_twist_for_variable("x") is None

    def test_get_coeff_for_variable(self) -> None:
        ss = SumStructure(
            coeff_seqs=[
                CoeffSeq(name="a_m", variable="m"),
                CoeffSeq(name="b_n", variable="n"),
            ],
        )
        cs = ss.get_coeff_for_variable("m")
        assert cs is not None
        assert cs.name == "a_m"

    def test_has_voronoi_eligible_twist(self) -> None:
        ss = SumStructure(
            coeff_seqs=[
                CoeffSeq(
                    name="a_m", variable="m",
                    voronoi_eligible=VoronoiEligibility.ELIGIBLE,
                ),
            ],
            additive_twists=[
                AdditiveTwist(modulus="c", numerator="a", sum_variable="m"),
            ],
        )
        assert ss.has_voronoi_eligible_twist()

    def test_no_voronoi_eligible_twist(self) -> None:
        ss = SumStructure(
            coeff_seqs=[
                CoeffSeq(
                    name="a_m", variable="m",
                    voronoi_eligible=VoronoiEligibility.INELIGIBLE,
                ),
            ],
            additive_twists=[
                AdditiveTwist(modulus="c", numerator="a", sum_variable="m"),
            ],
        )
        assert not ss.has_voronoi_eligible_twist()


class TestSumStructureFromPipeline:
    def test_delta_setup_produces_sum_structure(self) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            variables=["m", "n"],
            ranges=[
                Range(variable="m", lower="1", upper="T^theta"),
                Range(variable="n", lower="1", upper="T^theta"),
            ],
            kernels=[Kernel(name="W_AFE")],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = DeltaMethodSetup().apply([term], ledger)
        assert "sum_structure" in results[0].metadata

    def test_sum_structure_has_twists(self) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            variables=["m", "n"],
            kernels=[Kernel(name="W_AFE")],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = DeltaMethodSetup().apply([term], ledger)
        ss_data = results[0].metadata["sum_structure"]
        ss = SumStructure.model_validate(ss_data)
        assert len(ss.additive_twists) == 2

    def test_sum_structure_voronoi_eligible(self) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            variables=["m", "n"],
            kernels=[Kernel(name="W_AFE")],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = DeltaMethodSetup().apply([term], ledger)
        ss_data = results[0].metadata["sum_structure"]
        ss = SumStructure.model_validate(ss_data)
        assert ss.has_voronoi_eligible_twist()

    def test_sum_structure_serialization_roundtrip(self) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            variables=["m", "n"],
            kernels=[Kernel(name="W_AFE")],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = DeltaMethodSetup().apply([term], ledger)
        ss_data = results[0].metadata["sum_structure"]
        ss = SumStructure.model_validate(ss_data)
        re_dumped = ss.model_dump()
        ss2 = SumStructure.model_validate(re_dumped)
        assert len(ss2.sum_indices) == len(ss.sum_indices)
        assert len(ss2.additive_twists) == len(ss.additive_twists)


class TestBesselKernelFamily:
    def test_enum_values(self) -> None:
        assert BesselKernelFamily.J_BESSEL == "J_Bessel"
        assert BesselKernelFamily.Y_BESSEL == "Y_Bessel"
        assert BesselKernelFamily.K_BESSEL == "K_Bessel"
        assert BesselKernelFamily.J_PLUS_K == "J+K"
        assert BesselKernelFamily.UNSPECIFIED == "unspecified"

    def test_all_members(self) -> None:
        assert len(BesselKernelFamily) == 5


class TestVoronoiMainKernel:
    def test_creation(self) -> None:
        vmk = VoronoiMainKernel(
            arithmetic_type=ArithmeticType.DIVISOR,
            modulus="c",
            residue_structure="simple_pole",
            test_function="W(x)",
            polar_order=1,
            description="Polar residual from Estermann",
        )
        assert vmk.arithmetic_type == ArithmeticType.DIVISOR
        assert vmk.modulus == "c"
        assert vmk.polar_order == 1

    def test_round_trip_serialization(self) -> None:
        vmk = VoronoiMainKernel(
            arithmetic_type=ArithmeticType.HECKE,
            modulus="q",
            residue_structure="double_pole",
            polar_order=2,
        )
        dumped = vmk.model_dump()
        restored = VoronoiMainKernel.model_validate(dumped)
        assert restored.arithmetic_type == ArithmeticType.HECKE
        assert restored.polar_order == 2
        assert restored.model_dump() == dumped

    def test_defaults(self) -> None:
        vmk = VoronoiMainKernel(
            arithmetic_type=ArithmeticType.GENERIC,
            modulus="c",
        )
        assert vmk.residue_structure == ""
        assert vmk.test_function == ""
        assert vmk.polar_order == 1
        assert vmk.description == ""


class TestWeightKernelBessel:
    def test_default_bessel_family(self) -> None:
        wk = WeightKernel(kind="smooth")
        assert wk.bessel_family == BesselKernelFamily.UNSPECIFIED
        assert wk.argument_structure == ""

    def test_explicit_bessel_family(self) -> None:
        wk = WeightKernel(
            kind="bessel_transform",
            bessel_family=BesselKernelFamily.J_PLUS_K,
            argument_structure="4*pi*sqrt(m*n_star)/c",
        )
        assert wk.bessel_family == BesselKernelFamily.J_PLUS_K
        assert wk.argument_structure == "4*pi*sqrt(m*n_star)/c"

    def test_serialization_with_bessel(self) -> None:
        wk = WeightKernel(
            kind="bessel_transform",
            bessel_family=BesselKernelFamily.J_BESSEL,
            argument_structure="4*pi*sqrt(mn)/c",
        )
        dumped = wk.model_dump()
        restored = WeightKernel.model_validate(dumped)
        assert restored.bessel_family == BesselKernelFamily.J_BESSEL
