"""Property-based / fuzz tests for invariants across transformations.

Tests:
  1. Voronoi FORMULA mode with varied arithmetic types
  2. Kuznetsov phase consumption invariants
  3. Kernel state transitions exhaustive coverage
  4. Red Flag B invariant under mutation
"""

from __future__ import annotations

import pytest

from mollifier_theta.core.ir import (
    KERNEL_STATE_TRANSITIONS,
    HistoryEntry,
    Kernel,
    KernelState,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.invariants import (
    check_kernel_state_transition,
    check_phase_deps_subset,
    check_phases_tracked_with_context,
    check_spectral_bound_voronoi_kind,
    check_spectralized_has_kuznetsov_meta,
    validate_term,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    KuznetsovMeta,
    VoronoiKind,
    VoronoiMeta,
    _KUZNETSOV_KEY,
    _VORONOI_KEY,
)
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
from mollifier_theta.transforms.kuznetsov import KuznetsovTransform
from mollifier_theta.transforms.voronoi import VoronoiTransform, _FORMULA_ELIGIBLE_TYPES


# ── Helpers ─────────────────────────────────────────────────────────


def _make_uncollapsed_delta_term(
    arithmetic_type: ArithmeticType = ArithmeticType.DIVISOR,
    target_var: str = "n",
) -> Term:
    """Build a minimal uncollapsed-delta term suitable for Voronoi."""
    ss = SumStructure(
        sum_indices=[
            SumIndex(name="m", range_lower="1", range_upper="N"),
            SumIndex(name=target_var, range_lower="1", range_upper="N"),
        ],
        coeff_seqs=[
            CoeffSeq(name="a_m", variable="m", arithmetic_type=arithmetic_type),
            CoeffSeq(
                name=f"b_{target_var}", variable=target_var,
                arithmetic_type=arithmetic_type,
                voronoi_eligible=VoronoiEligibility.ELIGIBLE,
            ),
        ],
        additive_twists=[
            AdditiveTwist(
                numerator="a", modulus="c",
                sum_variable=target_var, sign=1,
            ),
        ],
        weight_kernels=[WeightKernel(kind="smooth", original_name="W_AFE")],
    )
    return Term(
        kind=TermKind.OFF_DIAGONAL,
        expression=f"sum_{{m,{target_var}}} a_m b_{target_var} e(a*{target_var}/c)",
        variables=["m", target_var, "c"],
        ranges=[
            Range(variable="m", lower="1", upper="N"),
            Range(variable=target_var, lower="1", upper="N"),
        ],
        kernels=[Kernel(name="W_AFE", support="(0,inf)", properties={"rapid_decay": True})],
        phases=[
            Phase(
                expression=f"e(a*{target_var}/c)",
                depends_on=[target_var, "c"],
                is_separable=False,
                unit_modulus=True,
            )
        ],
        kernel_state=KernelState.UNCOLLAPSED_DELTA,
        metadata={
            "delta_method_applied": True,
            "sum_structure": ss.model_dump(),
        },
    )


def _make_kloosterman_term(with_voronoi_formula: bool = True) -> Term:
    """Build a minimal Kloosterman term suitable for Kuznetsov."""
    voronoi_meta = VoronoiMeta(
        applied=True,
        target_variable="n",
        kind=VoronoiKind.FORMULA if with_voronoi_formula else VoronoiKind.STRUCTURAL_ONLY,
    )
    return Term(
        kind=TermKind.KLOOSTERMAN,
        expression="sum S(m,n;c)/c * V(...)",
        variables=["m", "n", "c"],
        ranges=[
            Range(variable="m", lower="1", upper="N"),
            Range(variable="n", lower="1", upper="N"),
        ],
        kernels=[Kernel(name="KloostermanKernel")],
        phases=[
            Phase(
                expression="S(m,n;c)/c",
                depends_on=["m", "n", "c"],
                is_separable=False,
            ),
            Phase(
                expression="e(absorbed_phase)",
                depends_on=["m"],
                absorbed=True,
            ),
        ],
        kernel_state=KernelState.KLOOSTERMANIZED,
        metadata={
            "kloosterman_form": True,
            "voronoi_applied": True,
            _VORONOI_KEY: voronoi_meta.model_dump(),
        },
    )


# ── 1. Voronoi FORMULA mode with varied arithmetic types ────────────


class TestVoronoiFormulaArithmeticTypes:
    """Fuzz Voronoi FORMULA mode across all arithmetic types."""

    @pytest.mark.parametrize("arith_type", list(ArithmeticType))
    def test_formula_mode_gating(self, arith_type: ArithmeticType) -> None:
        """Formula mode only applies to eligible types."""
        term = _make_uncollapsed_delta_term(arithmetic_type=arith_type)
        voronoi = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        ledger.add(term)

        result = voronoi.apply([term], ledger)

        if arith_type in _FORMULA_ELIGIBLE_TYPES:
            # Should produce 2 terms (main + dual)
            assert len(result) == 2, (
                f"Expected 2 terms for {arith_type}, got {len(result)}"
            )
            statuses = {t.status for t in result}
            assert TermStatus.MAIN_TERM in statuses
        else:
            # Ineligible types pass through unchanged
            assert len(result) == 1
            assert result[0].id == term.id

    @pytest.mark.parametrize("arith_type", list(_FORMULA_ELIGIBLE_TYPES))
    def test_formula_voronoi_metadata_present(self, arith_type: ArithmeticType) -> None:
        """Formula Voronoi terms have VoronoiMeta with kind=FORMULA."""
        term = _make_uncollapsed_delta_term(arithmetic_type=arith_type)
        voronoi = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        ledger.add(term)

        result = voronoi.apply([term], ledger)
        dual_terms = [t for t in result if t.status != TermStatus.MAIN_TERM]
        for dt in dual_terms:
            vm_data = dt.metadata.get(_VORONOI_KEY)
            assert vm_data is not None
            vm = VoronoiMeta.model_validate(vm_data)
            assert vm.kind == VoronoiKind.FORMULA

    @pytest.mark.parametrize("arith_type", list(_FORMULA_ELIGIBLE_TYPES))
    def test_formula_voronoi_passes_validate_term(self, arith_type: ArithmeticType) -> None:
        """All output terms pass single-term invariant checks."""
        term = _make_uncollapsed_delta_term(arithmetic_type=arith_type)
        voronoi = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        ledger.add(term)

        result = voronoi.apply([term], ledger)
        for t in result:
            violations = validate_term(t)
            assert violations == [], f"Invariant violations for {arith_type}: {violations}"


# ── 2. Kuznetsov phase consumption ──────────────────────────────────


class TestKuznetsovPhaseConsumption:
    """Fuzz Kuznetsov transform phase handling."""

    @pytest.mark.parametrize("sign_case", ["plus", "minus"])
    def test_kloosterman_phase_consumed(self, sign_case: str) -> None:
        """S(m,n;c)/c phase must be consumed, spectral phase added."""
        term = _make_kloosterman_term()
        kuznetsov = KuznetsovTransform(sign_case=sign_case)
        ledger = TermLedger()
        ledger.add(term)

        result = kuznetsov.apply([term], ledger)
        assert len(result) == 1
        out = result[0]

        # S(m,n;c)/c should not appear in non-absorbed output phases
        active_phase_exprs = [p.expression for p in out.phases if not p.absorbed]
        assert not any("S(m,n;c)/c" in expr for expr in active_phase_exprs)

        # Spectral expansion phase should be present
        assert any("spectral_expansion" in p.expression for p in out.phases)

    @pytest.mark.parametrize("sign_case", ["plus", "minus"])
    def test_kuznetsov_consumed_phases_metadata(self, sign_case: str) -> None:
        """Consumed phases recorded in _kuznetsov_consumed_phases metadata."""
        term = _make_kloosterman_term()
        kuznetsov = KuznetsovTransform(sign_case=sign_case)
        ledger = TermLedger()
        ledger.add(term)

        result = kuznetsov.apply([term], ledger)
        out = result[0]
        consumed = out.metadata.get("_kuznetsov_consumed_phases", [])
        assert len(consumed) > 0
        assert any("S(m,n;c)/c" in expr for expr in consumed)

    @pytest.mark.parametrize("sign_case", ["plus", "minus"])
    def test_kuznetsov_output_has_kuznetsov_meta(self, sign_case: str) -> None:
        """Output has _kuznetsov metadata with correct sign_case."""
        term = _make_kloosterman_term()
        kuznetsov = KuznetsovTransform(sign_case=sign_case)
        ledger = TermLedger()
        ledger.add(term)

        result = kuznetsov.apply([term], ledger)
        out = result[0]
        km_data = out.metadata.get(_KUZNETSOV_KEY)
        assert km_data is not None
        km = KuznetsovMeta.model_validate(km_data)
        assert km.applied is True
        assert km.sign_case == sign_case

    def test_kuznetsov_skips_non_kloosterman(self) -> None:
        """Non-KLOOSTERMAN terms pass through unchanged."""
        term = Term(
            kind=TermKind.DIAGONAL,
            expression="main term",
            kernel_state=KernelState.NONE,
        )
        kuznetsov = KuznetsovTransform()
        ledger = TermLedger()
        ledger.add(term)

        result = kuznetsov.apply([term], ledger)
        assert len(result) == 1
        assert result[0].id == term.id

    def test_kuznetsov_skips_inactive(self) -> None:
        """BOUND_ONLY Kloosterman terms pass through."""
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            expression="already bounded",
            kernel_state=KernelState.KLOOSTERMANIZED,
            status=TermStatus.BOUND_ONLY,
            lemma_citation="test",
        )
        kuznetsov = KuznetsovTransform()
        ledger = TermLedger()
        ledger.add(term)

        result = kuznetsov.apply([term], ledger)
        assert len(result) == 1
        assert result[0].id == term.id


# ── 3. Kernel state transitions exhaustive ──────────────────────────


class TestKernelStateTransitionsExhaustive:
    """Fuzz all pairs of kernel states to verify transition legality."""

    @pytest.mark.parametrize(
        "from_state",
        list(KernelState),
        ids=[s.value for s in KernelState],
    )
    @pytest.mark.parametrize(
        "to_state",
        list(KernelState),
        ids=[s.value for s in KernelState],
    )
    def test_transition_matches_table(
        self, from_state: KernelState, to_state: KernelState
    ) -> None:
        """Every state pair matches the declared transition table."""
        allowed = KERNEL_STATE_TRANSITIONS.get(from_state, set())
        violations = check_kernel_state_transition(from_state, to_state)

        if to_state in allowed:
            assert violations == [], (
                f"{from_state.value} → {to_state.value} should be legal "
                f"but got violations: {violations}"
            )
        else:
            assert len(violations) > 0, (
                f"{from_state.value} → {to_state.value} should be illegal "
                f"but no violations reported"
            )

    def test_no_self_transitions(self) -> None:
        """No state should transition to itself (that's a no-op, not tracked)."""
        for state in KernelState:
            allowed = KERNEL_STATE_TRANSITIONS.get(state, set())
            assert state not in allowed, (
                f"Self-transition {state.value} → {state.value} in table"
            )

    def test_spectralized_is_terminal(self) -> None:
        """SPECTRALIZED has no outgoing transitions."""
        allowed = KERNEL_STATE_TRANSITIONS.get(KernelState.SPECTRALIZED, set())
        assert len(allowed) == 0

    def test_none_has_single_transition(self) -> None:
        """NONE → UNCOLLAPSED_DELTA only."""
        allowed = KERNEL_STATE_TRANSITIONS.get(KernelState.NONE, set())
        assert allowed == {KernelState.UNCOLLAPSED_DELTA}


# ── 4. Red Flag B invariant under mutation ──────────────────────────


class TestRedFlagBFuzz:
    """Fuzz the Red Flag B invariant: SpectralLargeSieve requires formula Voronoi."""

    def _make_sls_bound_term(
        self,
        voronoi_kind: VoronoiKind | None = None,
        has_voronoi_meta: bool = True,
    ) -> Term:
        """Build a BoundOnly term claiming SpectralLargeSieve family."""
        metadata: dict = {
            "_bound": BoundMeta(
                strategy="SpectralLargeSieve",
                error_exponent="(3*theta+1)/2",
                citation="test",
                bound_family="SpectralLargeSieve",
                case_id="large_modulus",
            ).model_dump(),
        }
        if has_voronoi_meta and voronoi_kind is not None:
            metadata[_VORONOI_KEY] = VoronoiMeta(
                applied=True, kind=voronoi_kind,
            ).model_dump()
        return Term(
            kind=TermKind.SPECTRAL,
            expression="SLS bound test",
            status=TermStatus.BOUND_ONLY,
            kernel_state=KernelState.SPECTRALIZED,
            lemma_citation="test citation",
            metadata=metadata,
        )

    def test_formula_voronoi_passes(self) -> None:
        """SLS bound with formula Voronoi passes Red Flag B."""
        term = self._make_sls_bound_term(voronoi_kind=VoronoiKind.FORMULA)
        violations = check_spectral_bound_voronoi_kind(term)
        assert violations == []

    def test_structural_voronoi_fails(self) -> None:
        """SLS bound with structural Voronoi fails Red Flag B."""
        term = self._make_sls_bound_term(voronoi_kind=VoronoiKind.STRUCTURAL_ONLY)
        violations = check_spectral_bound_voronoi_kind(term)
        assert len(violations) > 0
        assert "VoronoiKind.FORMULA" in violations[0]

    def test_no_voronoi_meta_fails(self) -> None:
        """SLS bound without Voronoi metadata fails Red Flag B."""
        term = self._make_sls_bound_term(has_voronoi_meta=False)
        violations = check_spectral_bound_voronoi_kind(term)
        assert len(violations) > 0

    def test_non_sls_bound_passes(self) -> None:
        """Non-SLS BoundOnly terms skip Red Flag B."""
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            expression="DI bound",
            status=TermStatus.BOUND_ONLY,
            kernel_state=KernelState.KLOOSTERMANIZED,
            lemma_citation="test",
            metadata={
                "_bound": BoundMeta(
                    strategy="DI_Kloosterman",
                    error_exponent="7*theta/4",
                    citation="test",
                    bound_family="DI_Kloosterman",
                ).model_dump(),
            },
        )
        violations = check_spectral_bound_voronoi_kind(term)
        assert violations == []

    def test_active_term_skips_check(self) -> None:
        """Active terms are not checked by Red Flag B."""
        term = Term(
            kind=TermKind.SPECTRAL,
            expression="Active spectral",
            status=TermStatus.ACTIVE,
            kernel_state=KernelState.SPECTRALIZED,
            metadata={
                _KUZNETSOV_KEY: KuznetsovMeta(applied=True).model_dump(),
            },
        )
        violations = check_spectral_bound_voronoi_kind(term)
        assert violations == []

    @pytest.mark.parametrize("voronoi_kind", list(VoronoiKind))
    def test_validate_term_catches_red_flag_b(self, voronoi_kind: VoronoiKind) -> None:
        """validate_term includes Red Flag B check."""
        term = self._make_sls_bound_term(voronoi_kind=voronoi_kind)
        # Also need _kuznetsov meta for SPECTRALIZED check
        metadata = dict(term.metadata)
        metadata[_KUZNETSOV_KEY] = KuznetsovMeta(applied=True).model_dump()
        term = Term(**{**term.model_dump(), "metadata": metadata})

        violations = validate_term(term)
        if voronoi_kind == VoronoiKind.FORMULA:
            assert violations == []
        else:
            assert any("VoronoiKind.FORMULA" in v for v in violations)


# ── 5. SPECTRALIZED requires Kuznetsov meta ─────────────────────────


class TestSpectralizedKuznetsovMeta:
    """Fuzz the spectralized-requires-kuznetsov-meta invariant."""

    def test_spectralized_with_meta_passes(self) -> None:
        term = Term(
            kind=TermKind.SPECTRAL,
            kernel_state=KernelState.SPECTRALIZED,
            metadata={_KUZNETSOV_KEY: KuznetsovMeta(applied=True).model_dump()},
        )
        violations = check_spectralized_has_kuznetsov_meta(term)
        assert violations == []

    def test_spectralized_without_meta_fails(self) -> None:
        term = Term(
            kind=TermKind.SPECTRAL,
            kernel_state=KernelState.SPECTRALIZED,
        )
        violations = check_spectralized_has_kuznetsov_meta(term)
        assert len(violations) > 0

    @pytest.mark.parametrize(
        "state",
        [s for s in KernelState if s != KernelState.SPECTRALIZED],
    )
    def test_non_spectralized_skips_check(self, state: KernelState) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            kernel_state=state,
        )
        violations = check_spectralized_has_kuznetsov_meta(term)
        assert violations == []


# ── 6. Phase dependency subset under Voronoi ────────────────────────


class TestPhaseDepsUnderVoronoi:
    """After Voronoi transform, phase depends_on must be subset of variables."""

    @pytest.mark.parametrize("arith_type", list(_FORMULA_ELIGIBLE_TYPES))
    def test_formula_voronoi_phase_deps_valid(self, arith_type: ArithmeticType) -> None:
        term = _make_uncollapsed_delta_term(arithmetic_type=arith_type)
        voronoi = VoronoiTransform(target_variable="n", mode=VoronoiKind.FORMULA)
        ledger = TermLedger()
        ledger.add(term)

        result = voronoi.apply([term], ledger)
        violations = check_phase_deps_subset(result)
        assert violations == [], f"Phase deps violations for {arith_type}: {violations}"

    def test_structural_voronoi_phase_deps_valid(self) -> None:
        term = _make_uncollapsed_delta_term()
        voronoi = VoronoiTransform(target_variable="n", mode=VoronoiKind.STRUCTURAL_ONLY)
        ledger = TermLedger()
        ledger.add(term)

        result = voronoi.apply([term], ledger)
        violations = check_phase_deps_subset(result)
        assert violations == [], f"Phase deps violations: {violations}"
