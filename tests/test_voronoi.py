"""Tests for VoronoiTransform and conrey89_voronoi pipeline."""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import (
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
)
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
from mollifier_theta.transforms.voronoi import VoronoiTransform


@pytest.fixture
def off_diagonal_term() -> Term:
    return Term(
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


@pytest.fixture
def setup_term(off_diagonal_term: Term) -> Term:
    """An off-diagonal term after DeltaMethodSetup."""
    ledger = TermLedger()
    ledger.add(off_diagonal_term)
    results = DeltaMethodSetup().apply([off_diagonal_term], ledger)
    return results[0]


class TestVoronoiGating:
    def test_applies_to_uncollapsed_delta(self, setup_term: Term) -> None:
        voronoi = VoronoiTransform(target_variable="n")
        assert voronoi._should_apply(setup_term)

    def test_rejects_none_kernel_state(self) -> None:
        term = Term(kind=TermKind.OFF_DIAGONAL, kernel_state=KernelState.NONE)
        voronoi = VoronoiTransform()
        assert not voronoi._should_apply(term)

    def test_rejects_collapsed(self) -> None:
        term = Term(kind=TermKind.OFF_DIAGONAL, kernel_state=KernelState.COLLAPSED)
        voronoi = VoronoiTransform()
        assert not voronoi._should_apply(term)

    def test_rejects_no_sum_structure(self) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            metadata={"delta_method_applied": True, "delta_method_collapsed": False},
        )
        voronoi = VoronoiTransform()
        assert not voronoi._should_apply(term)

    def test_rejects_ineligible_coefficients(self) -> None:
        ss = SumStructure(
            coeff_seqs=[
                CoeffSeq(name="a_n", variable="n",
                         voronoi_eligible=VoronoiEligibility.INELIGIBLE),
            ],
            additive_twists=[
                AdditiveTwist(modulus="c", numerator="a", sum_variable="n"),
            ],
        )
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            kernel_state=KernelState.UNCOLLAPSED_DELTA,
            metadata={
                "delta_method_applied": True,
                "delta_method_collapsed": False,
                "sum_structure": ss.model_dump(),
            },
        )
        voronoi = VoronoiTransform(target_variable="n")
        assert not voronoi._should_apply(term)


class TestVoronoiRewrite:
    @pytest.fixture(autouse=True)
    def _apply(self, setup_term: Term) -> None:
        ledger = TermLedger()
        ledger.add(setup_term)
        voronoi = VoronoiTransform(target_variable="n")
        results = voronoi.apply([setup_term], ledger)
        self.result = results[0]
        self.ledger = ledger

    def test_kernel_state_voronoi_applied(self) -> None:
        assert self.result.kernel_state == KernelState.VORONOI_APPLIED

    def test_voronoi_metadata_flag(self) -> None:
        assert self.result.metadata.get("voronoi_applied") is True
        assert self.result.metadata.get("voronoi_target_variable") == "n"
        assert self.result.metadata.get("voronoi_dual_variable") == "n*"

    def test_dual_variable_in_variables(self) -> None:
        assert "n*" in self.result.variables
        assert "n" not in self.result.variables

    def test_m_preserved(self) -> None:
        assert "m" in self.result.variables

    def test_dual_range_added(self) -> None:
        range_vars = {r.variable for r in self.result.ranges}
        assert "n*" in range_vars

    def test_original_kernels_preserved(self) -> None:
        kernel_names = {k.name for k in self.result.kernels}
        assert "W_AFE" in kernel_names
        assert "FourierKernel" in kernel_names

    def test_voronoi_dual_kernel_added(self) -> None:
        kernel_names = {k.name for k in self.result.kernels}
        assert "VoronoiDualKernel" in kernel_names

    def test_voronoi_kernel_properties(self) -> None:
        vk = [k for k in self.result.kernels if k.name == "VoronoiDualKernel"][0]
        assert vk.properties["is_voronoi_dual"] is True
        assert vk.properties["original_variable"] == "n"
        assert vk.properties["smooth"] is True

    def test_history_entry(self) -> None:
        assert self.result.history[-1].transform == "VoronoiTransform"

    def test_phases_renamed(self) -> None:
        """After Voronoi on n, phase expressions reference n* not n."""
        phase_exprs = {p.expression for p in self.result.phases}
        assert "(m/n*)^{it}" in phase_exprs

    def test_sum_structure_updated(self) -> None:
        ss = SumStructure.model_validate(self.result.metadata["sum_structure"])
        idx_names = {idx.name for idx in ss.sum_indices}
        assert "n*" in idx_names
        assert "n" not in idx_names

    def test_dual_coefficients(self) -> None:
        ss = SumStructure.model_validate(self.result.metadata["sum_structure"])
        cs = ss.get_coeff_for_variable("n*")
        assert cs is not None
        assert "Voronoi" in cs.description
        # Dual coefficients are no longer Voronoi-eligible (prevent double application)
        assert cs.voronoi_eligible == VoronoiEligibility.INELIGIBLE

    def test_dual_twist_sign_flipped(self) -> None:
        ss = SumStructure.model_validate(self.result.metadata["sum_structure"])
        tw = ss.get_twist_for_variable("n*")
        assert tw is not None
        assert tw.invert_numerator is True

    def test_bessel_weight_kernel(self) -> None:
        ss = SumStructure.model_validate(self.result.metadata["sum_structure"])
        bessel_kernels = [wk for wk in ss.weight_kernels if wk.kind == "bessel_transform"]
        assert len(bessel_kernels) >= 1


class TestVoronoiPassthrough:
    def test_non_matching_passes_through(self) -> None:
        term = Term(kind=TermKind.DIAGONAL)
        ledger = TermLedger()
        ledger.add(term)
        voronoi = VoronoiTransform()
        results = voronoi.apply([term], ledger)
        assert results[0].id == term.id

    def test_wrong_target_variable_passes_through(self, setup_term: Term) -> None:
        ledger = TermLedger()
        ledger.add(setup_term)
        voronoi = VoronoiTransform(target_variable="x")
        results = voronoi.apply([setup_term], ledger)
        assert results[0].id == setup_term.id


class TestVoronoiPipeline:
    def test_voronoi_pipeline_runs(self) -> None:
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        assert result is not None
        assert result.ledger.count() > 0

    def test_voronoi_pipeline_admissible(self) -> None:
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        assert result.theta_admissible is True

    def test_voronoi_pipeline_theta_max(self) -> None:
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        # PostVoronoi bound: E(theta) = 2*theta - 1/4, theta_max = 5/8
        assert result.theta_max is not None
        assert abs(result.theta_max - 5 / 8) < 1e-10

    def test_voronoi_in_transform_chain(self) -> None:
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        chain = result.report_data["transform_chain"]
        assert "VoronoiTransform(n)" in chain


class TestRenameVariableInString:
    """Tests for the token-aware variable renaming utility."""

    def test_standalone_variable(self) -> None:
        from mollifier_theta.transforms.voronoi import _rename_variable_in_string
        assert _rename_variable_in_string("n", "n", "n*") == "n*"

    def test_in_expression_slash(self) -> None:
        from mollifier_theta.transforms.voronoi import _rename_variable_in_string
        assert _rename_variable_in_string("(m/n)^{it}", "n", "n*") == "(m/n*)^{it}"

    def test_does_not_rename_inside_word(self) -> None:
        from mollifier_theta.transforms.voronoi import _rename_variable_in_string
        # Should NOT rename 'n' inside 'int' or 'sin'
        result = _rename_variable_in_string("int sin(n)", "n", "n*")
        assert result == "int sin(n*)"

    def test_multiple_occurrences(self) -> None:
        from mollifier_theta.transforms.voronoi import _rename_variable_in_string
        result = _rename_variable_in_string("n + n/c", "n", "n*")
        assert result == "n* + n*/c"

    def test_no_match(self) -> None:
        from mollifier_theta.transforms.voronoi import _rename_variable_in_string
        assert _rename_variable_in_string("m/c", "n", "n*") == "m/c"

    def test_voronoi_terms_in_ledger(self) -> None:
        from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
        result = conrey89_voronoi_pipeline(theta_val=0.56)
        all_terms = result.ledger.all_terms()
        voronoi_terms = [
            t for t in all_terms if t.metadata.get("voronoi_applied")
        ]
        assert len(voronoi_terms) > 0

    def test_conrey89_baseline_unchanged(self) -> None:
        """Regression: original conrey89 pipeline still gives theta_max = 4/7."""
        from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
        result = conrey89_pipeline(theta_val=0.56)
        assert abs(result.theta_max - 4 / 7) < 1e-10
        assert result.theta_admissible is True
