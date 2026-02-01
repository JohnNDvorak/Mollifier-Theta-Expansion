"""Tests for SumIndex, CoeffSeq, AdditiveTwist, SumStructure IR types."""

from __future__ import annotations

from mollifier_theta.core.ir import Kernel, Phase, Range, Term, TermKind
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.sum_structures import (
    AdditiveTwist,
    ArithmeticType,
    CoeffSeq,
    SumIndex,
    SumStructure,
    VoronoiEligibility,
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
