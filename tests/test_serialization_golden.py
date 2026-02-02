"""Golden serialization tests.

Ensures that IR model serialization is stable across code changes.
If a golden file doesn't match, either the model changed intentionally
(update the golden file) or a regression was introduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from mollifier_theta.core.stage_meta import KuznetsovMeta, VoronoiKind, VoronoiMeta
from mollifier_theta.core.sum_structures import (
    AdditiveTwist,
    ArithmeticType,
    BesselKernelFamily,
    CoeffSeq,
    SumIndex,
    SumStructure,
    VoronoiMainKernel,
    WeightKernel,
)

GOLDEN_DIR = Path(__file__).parent


def _load_golden(name: str) -> dict:
    path = GOLDEN_DIR / name
    return json.loads(path.read_text())


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, indent=2)


# ── Term golden ──────────────────────────────────────────────────────


@pytest.fixture
def golden_term() -> Term:
    return Term(
        id="golden_term_001",
        kind=TermKind.KLOOSTERMAN,
        expression="sum S(m,n;c)/c ...",
        variables=["m", "n", "c"],
        ranges=[
            Range(variable="m", lower="1", upper="N"),
            Range(variable="n", lower="1", upper="N"),
        ],
        kernels=[
            Kernel(name="W_AFE", support="(0,inf)", properties={"rapid_decay": True})
        ],
        phases=[
            Phase(
                expression="e(am/c)",
                depends_on=["m", "c"],
                is_separable=True,
                unit_modulus=True,
            ),
            Phase(
                expression="S(m,n;c)/c",
                depends_on=["m", "n", "c"],
                is_separable=False,
            ),
        ],
        history=[
            HistoryEntry(
                transform="TestSetup", parent_ids=[], description="Golden fixture"
            )
        ],
        status=TermStatus.ACTIVE,
        parents=[],
        multiplicity=2,
        metadata={"test_key": "test_value"},
        kernel_state=KernelState.KLOOSTERMANIZED,
    )


class TestTermGolden:
    def test_serialization_matches_golden(self, golden_term: Term) -> None:
        golden = _load_golden("golden_term.json")
        actual = golden_term.model_dump()
        assert _canonical(actual) == _canonical(golden)

    def test_round_trip_preserves_data(self, golden_term: Term) -> None:
        dumped = golden_term.model_dump()
        restored = Term(**dumped)
        assert restored.model_dump() == dumped

    def test_round_trip_preserves_types(self, golden_term: Term) -> None:
        dumped = golden_term.model_dump()
        restored = Term(**dumped)
        assert restored.kind == TermKind.KLOOSTERMAN
        assert restored.kernel_state == KernelState.KLOOSTERMANIZED
        assert restored.status == TermStatus.ACTIVE
        assert len(restored.phases) == 2
        assert restored.phases[0].unit_modulus is True
        assert restored.phases[1].unit_modulus is False


# ── SumStructure golden ─────────────────────────────────────────────


@pytest.fixture
def golden_ss() -> SumStructure:
    return SumStructure(
        sum_indices=[
            SumIndex(name="m", range_lower="1", range_upper="N"),
            SumIndex(name="n*", range_lower="1", range_upper="N_dual"),
        ],
        coeff_seqs=[
            CoeffSeq(name="a_m", variable="m", arithmetic_type=ArithmeticType.DIVISOR),
            CoeffSeq(name="b_n*", variable="n*"),
        ],
        additive_twists=[
            AdditiveTwist(numerator="a", modulus="c", sum_variable="m", sign=1),
            AdditiveTwist(
                numerator="b_bar",
                modulus="c",
                sum_variable="n*",
                sign=-1,
                invert_numerator=True,
            ),
        ],
        weight_kernels=[WeightKernel(kind="smooth", original_name="W_AFE")],
    )


class TestSumStructureGolden:
    def test_serialization_matches_golden(self, golden_ss: SumStructure) -> None:
        golden = _load_golden("golden_sum_structure.json")
        actual = golden_ss.model_dump()
        assert _canonical(actual) == _canonical(golden)

    def test_round_trip_preserves_data(self, golden_ss: SumStructure) -> None:
        dumped = golden_ss.model_dump()
        restored = SumStructure(**dumped)
        assert restored.model_dump() == dumped

    def test_round_trip_preserves_types(self, golden_ss: SumStructure) -> None:
        dumped = golden_ss.model_dump()
        restored = SumStructure(**dumped)
        assert restored.coeff_seqs[0].arithmetic_type == ArithmeticType.DIVISOR
        assert restored.additive_twists[1].invert_numerator is True
        assert len(restored.sum_indices) == 2


# ── Cross-format stability ──────────────────────────────────────────


class TestCrossFormatStability:
    def test_json_string_deterministic(self, golden_term: Term) -> None:
        """Two serializations of the same object produce identical JSON."""
        s1 = json.dumps(golden_term.model_dump(), sort_keys=True)
        s2 = json.dumps(golden_term.model_dump(), sort_keys=True)
        assert s1 == s2

    def test_golden_files_are_sorted(self) -> None:
        """Golden files use sort_keys=True."""
        for name in ["golden_term.json", "golden_sum_structure.json"]:
            data = _load_golden(name)
            reserialized = json.dumps(data, sort_keys=True, indent=2)
            original = (GOLDEN_DIR / name).read_text().strip()
            assert reserialized == original, f"{name} is not canonically sorted"


# ── KuznetsovMeta golden ──────────────────────────────────────────


class TestKuznetsovMetaGolden:
    def test_round_trip_preserves_data(self) -> None:
        meta = KuznetsovMeta(
            applied=True,
            sign_case="plus",
            bessel_transform="Phi_Kuznetsov",
            spectral_window_scale="K",
            spectral_components=["discrete_maass", "holomorphic", "eisenstein"],
            level="1",
        )
        dumped = meta.model_dump()
        restored = KuznetsovMeta.model_validate(dumped)
        assert restored.applied is True
        assert restored.sign_case == "plus"
        assert restored.spectral_components == ["discrete_maass", "holomorphic", "eisenstein"]
        assert restored.model_dump() == dumped

    def test_json_string_deterministic(self) -> None:
        meta = KuznetsovMeta(applied=True, sign_case="minus")
        s1 = json.dumps(meta.model_dump(), sort_keys=True)
        s2 = json.dumps(meta.model_dump(), sort_keys=True)
        assert s1 == s2


# ── VoronoiMainKernel golden ──────────────────────────────────────


class TestVoronoiMainKernelGolden:
    def test_round_trip_preserves_data(self) -> None:
        vmk = VoronoiMainKernel(
            arithmetic_type=ArithmeticType.DIVISOR,
            modulus="c",
            residue_structure="simple_pole",
            test_function="W(x)",
            polar_order=1,
            description="Test fixture",
        )
        dumped = vmk.model_dump()
        restored = VoronoiMainKernel.model_validate(dumped)
        assert restored.arithmetic_type == ArithmeticType.DIVISOR
        assert restored.polar_order == 1
        assert restored.model_dump() == dumped

    def test_json_string_deterministic(self) -> None:
        vmk = VoronoiMainKernel(
            arithmetic_type=ArithmeticType.HECKE, modulus="q",
        )
        s1 = json.dumps(vmk.model_dump(), sort_keys=True)
        s2 = json.dumps(vmk.model_dump(), sort_keys=True)
        assert s1 == s2


# ── SpectralKernel Term golden ────────────────────────────────────


class TestSpectralTermGolden:
    def test_round_trip_spectral_term(self) -> None:
        term = Term(
            id="golden_spectral_001",
            kind=TermKind.SPECTRAL,
            expression="Spectral expansion",
            variables=["m"],
            kernels=[
                Kernel(
                    name="SpectralKernel",
                    properties={
                        "spectral_types": ["discrete_maass", "holomorphic", "eisenstein"],
                        "spectral_parameter": "t_f",
                        "level": "1",
                    },
                ),
            ],
            phases=[
                Phase(
                    expression="spectral_expansion(lambda_f(m)*lambda_f(n), h(t_f))",
                    is_separable=True,
                    depends_on=["m"],
                    unit_modulus=False,
                ),
            ],
            kernel_state=KernelState.SPECTRALIZED,
            metadata={
                "_kuznetsov": KuznetsovMeta(
                    applied=True,
                    sign_case="plus",
                    spectral_components=["discrete_maass", "holomorphic", "eisenstein"],
                ).model_dump(),
            },
        )
        dumped = term.model_dump()
        restored = Term(**dumped)
        assert restored.kind == TermKind.SPECTRAL
        assert restored.kernel_state == KernelState.SPECTRALIZED
        assert restored.model_dump() == dumped

    def test_spectral_term_json_deterministic(self) -> None:
        term = Term(
            id="golden_spectral_002",
            kind=TermKind.SPECTRAL,
            kernel_state=KernelState.SPECTRALIZED,
            metadata={
                "_kuznetsov": KuznetsovMeta(applied=True).model_dump(),
            },
        )
        s1 = json.dumps(term.model_dump(), sort_keys=True)
        s2 = json.dumps(term.model_dump(), sort_keys=True)
        assert s1 == s2


# ── VoronoiMeta with kind golden ─────────────────────────────────


class TestVoronoiMetaKindGolden:
    def test_structural_kind_default(self) -> None:
        meta = VoronoiMeta(applied=True)
        assert meta.kind == VoronoiKind.STRUCTURAL_ONLY
        dumped = meta.model_dump()
        restored = VoronoiMeta.model_validate(dumped)
        assert restored.kind == VoronoiKind.STRUCTURAL_ONLY

    def test_formula_kind_round_trip(self) -> None:
        meta = VoronoiMeta(applied=True, kind=VoronoiKind.FORMULA)
        dumped = meta.model_dump()
        restored = VoronoiMeta.model_validate(dumped)
        assert restored.kind == VoronoiKind.FORMULA
        assert restored.model_dump() == dumped
