"""Tests for Kuznetsov trace formula transform (Wave 1B)."""

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
from mollifier_theta.core.stage_meta import get_kuznetsov_meta
from mollifier_theta.transforms.kuznetsov import KuznetsovTransform


@pytest.fixture
def kloosterman_term() -> Term:
    """A KLOOSTERMANIZED term ready for Kuznetsov."""
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="sum S(m,n;c)/c ...",
        variables=["m", "n*", "c"],
        ranges=[
            Range(variable="m", lower="1", upper="N"),
            Range(variable="n*", lower="1", upper="N_dual"),
            Range(variable="c", lower="1", upper="C"),
        ],
        kernels=[
            Kernel(name="W_AFE"),
            Kernel(name="VoronoiDualKernel"),
        ],
        phases=[
            Phase(
                expression="S(m,n;c)/c",
                depends_on=["m", "n*", "c"],
                is_separable=False,
                unit_modulus=False,
            ),
        ],
        kernel_state=KernelState.KLOOSTERMANIZED,
        metadata={
            "kloosterman_form": True,
            "voronoi_applied": True,
        },
    )


@pytest.fixture
def non_kloosterman_term() -> Term:
    """A term that is NOT KLOOSTERMANIZED — should be rejected."""
    return Term(
        kind=TermKind.OFF_DIAGONAL,
        expression="off-diagonal",
        variables=["m", "n"],
        kernel_state=KernelState.COLLAPSED,
    )


class TestKuznetsovStateTransition:
    def test_produces_spectralized(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        assert results[0].kernel_state == KernelState.SPECTRALIZED

    def test_output_kind_is_spectral(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        assert results[0].kind == TermKind.SPECTRAL


class TestKuznetsovGating:
    def test_rejects_non_kloosterman(self, non_kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([non_kloosterman_term], ledger)
        # Should pass through unchanged
        assert results[0].id == non_kloosterman_term.id
        assert results[0].kernel_state == KernelState.COLLAPSED

    def test_rejects_bound_only(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            kernel_state=KernelState.KLOOSTERMANIZED,
            status=TermStatus.BOUND_ONLY,
            lemma_citation="test",
        )
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([term], ledger)
        assert results[0].id == term.id  # unchanged


class TestKuznetsovKernels:
    def test_has_kuznetsov_kernel(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        kernel_names = [k.name for k in results[0].kernels]
        assert "KuznetsovKernel" in kernel_names

    def test_has_spectral_kernel(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        kernel_names = [k.name for k in results[0].kernels]
        assert "SpectralKernel" in kernel_names

    def test_spectral_kernel_properties(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        sk = [k for k in results[0].kernels if k.name == "SpectralKernel"][0]
        assert "discrete_maass" in sk.properties["spectral_types"]
        assert "holomorphic" in sk.properties["spectral_types"]
        assert "eisenstein" in sk.properties["spectral_types"]


class TestKuznetsovMeta:
    def test_kuznetsov_meta_populated(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform(sign_case="plus")
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        km = get_kuznetsov_meta(results[0])
        assert km is not None
        assert km.applied is True
        assert km.sign_case == "plus"
        assert "discrete_maass" in km.spectral_components

    def test_kuznetsov_meta_sign_minus(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform(sign_case="minus")
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        km = get_kuznetsov_meta(results[0])
        assert km.sign_case == "minus"


class TestKuznetsovPhases:
    def test_kloosterman_phase_consumed(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        phase_exprs = [p.expression for p in results[0].phases]
        # S(m,n;c)/c should be consumed
        assert not any("S(m,n;c)/c" in e for e in phase_exprs)

    def test_spectral_phase_added(self, kloosterman_term: Term) -> None:
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([kloosterman_term], ledger)
        phase_exprs = [p.expression for p in results[0].phases]
        assert any("spectral_expansion" in e for e in phase_exprs)


class TestKuznetsovPassthrough:
    def test_non_matching_pass_through(self) -> None:
        diag = Term(kind=TermKind.DIAGONAL, expression="diag")
        kuz = KuznetsovTransform()
        ledger = TermLedger()
        results = kuz.apply([diag], ledger)
        assert len(results) == 1
        assert results[0].id == diag.id


class TestKuznetsovStrictRunner:
    def test_strict_runner_validates(self, kloosterman_term: Term) -> None:
        from mollifier_theta.pipelines.strict_runner import StrictPipelineRunner
        ledger = TermLedger()
        ledger.add(kloosterman_term)
        runner = StrictPipelineRunner(ledger)
        kuz = KuznetsovTransform()
        results = runner.run_stage(
            kuz, [kloosterman_term], "KuznetsovTransform",
            allow_kernel_removal=False,
            _allow_phase_drop=True,  # Kloosterman phase is consumed
        )
        assert results[0].kernel_state == KernelState.SPECTRALIZED
