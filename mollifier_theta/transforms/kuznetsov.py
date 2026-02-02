"""Kuznetsov trace formula transform: geometric Kloosterman side → spectral side.

Applies the Kuznetsov/Bruggeman trace formula to convert sums of
Kloosterman sums S(m,n;c)/c weighted by a test function into spectral
sums over automorphic forms (Maass, holomorphic, Eisenstein).

Mathematical operation:
  sum_c S(m,n;c)/c * h(c) → sum_f λ_f(m)λ_f(n) * ĥ(t_f) + (continuous spectrum)

Source: Iwaniec-Kowalski Ch. 16; sixth-moment draft prop:refined-kuznetsov.

State machine: KLOOSTERMANIZED → SPECTRALIZED
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    KernelState,
    Phase,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.core.stage_meta import KuznetsovMeta, _KUZNETSOV_KEY


class KuznetsovTransform:
    """Apply Kuznetsov trace formula: geometric Kloosterman side → spectral side.

    Requires kernel_state == KLOOSTERMANIZED.
    Source: Iwaniec-Kowalski Ch. 16; sixth-moment draft prop:refined-kuznetsov.
    """

    def __init__(self, sign_case: str = "plus") -> None:
        self.sign_case = sign_case

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
        """Gate: only apply to KLOOSTERMANIZED terms with Kloosterman kind."""
        return (
            term.kind == TermKind.KLOOSTERMAN
            and term.kernel_state == KernelState.KLOOSTERMANIZED
            and term.status == TermStatus.ACTIVE
        )

    def _apply_one(self, term: Term) -> Term:
        """Transform a single Kloosterman-side term to spectral side."""
        history = HistoryEntry(
            transform="KuznetsovTransform",
            parent_ids=[term.id],
            description=(
                f"Kuznetsov trace formula applied (sign_case={self.sign_case}). "
                f"Geometric Kloosterman sums S(m,n;c)/c replaced by spectral expansion "
                f"over Maass forms, holomorphic forms, and Eisenstein series. "
                f"Source: Iwaniec-Kowalski Ch. 16; prop:refined-kuznetsov."
            ),
        )

        # Build spectral kernels
        kuznetsov_kernel = Kernel(
            name="KuznetsovKernel",
            description=(
                f"Kuznetsov trace formula kernel (sign_case={self.sign_case}). "
                f"Encodes the Bessel integral transform Φ that maps the test function "
                f"on the geometric side to spectral weights."
            ),
            properties={
                "geometric_to_spectral": True,
                "sign_case": self.sign_case,
                "bessel_transform": "Phi_Kuznetsov",
            },
        )

        spectral_kernel = Kernel(
            name="SpectralKernel",
            description=(
                "Spectral decomposition kernel: discrete Maass + holomorphic + "
                "Eisenstein continuous spectrum contributions."
            ),
            properties={
                "spectral_types": ["discrete_maass", "holomorphic", "eisenstein"],
                "spectral_parameter": "t_f",
                "level": "1",
            },
        )

        new_kernels = list(term.kernels) + [kuznetsov_kernel, spectral_kernel]

        # Phase transformation: S(m,n;c)/c consumed, spectral expansion added
        new_phases: list[Phase] = []
        consumed_kloosterman = False
        for p in term.phases:
            if "S(m,n;c)/c" in p.expression and not p.absorbed:
                # Kloosterman phase consumed by trace formula
                consumed_kloosterman = True
                continue
            elif p.absorbed:
                # Keep absorbed phases
                new_phases.append(p)
            else:
                new_phases.append(p)

        # Add spectral expansion phase
        new_phases.append(Phase(
            expression="spectral_expansion(lambda_f(m)*lambda_f(n), h(t_f))",
            is_separable=True,
            absorbed=False,
            depends_on=["m"],
            unit_modulus=False,
        ))

        # Build Kuznetsov metadata
        kuznetsov_meta = KuznetsovMeta(
            applied=True,
            sign_case=self.sign_case,
            bessel_transform="Phi_Kuznetsov",
            spectral_window_scale="K",
            spectral_components=["discrete_maass", "holomorphic", "eisenstein"],
            level="1",
        )

        # Collect consumed phases for metadata
        consumed_phase_exprs = []
        for p in term.phases:
            if "S(m,n;c)/c" in p.expression and not p.absorbed:
                consumed_phase_exprs.append(p.expression)

        return Term(
            kind=TermKind.SPECTRAL,
            expression=(
                f"Spectral expansion: sum_f lambda_f(m)*lambda_f(n)*h(t_f) "
                f"+ Eisenstein continuous [Kuznetsov from {term.expression}]"
            ),
            variables=list(term.variables),
            ranges=list(term.ranges),
            kernels=new_kernels,
            phases=new_phases,
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            kernel_state=KernelState.SPECTRALIZED,
            metadata={
                **term.metadata,
                _KUZNETSOV_KEY: kuznetsov_meta.model_dump(),
                "_kuznetsov_consumed_phases": consumed_phase_exprs,
            },
        )

    def describe(self) -> str:
        return (
            f"Kuznetsov trace formula (sign_case={self.sign_case}): "
            f"converts geometric Kloosterman sums S(m,n;c)/c into spectral "
            f"expansion over Maass forms, holomorphic forms, and Eisenstein series. "
            f"Source: Iwaniec-Kowalski Ch. 16."
        )
