"""Phase absorption: detect separable phases, absorb into coefficients.

A phase is separable if it depends on a single summation variable.
Absorbing it means folding it into the coefficient sequence, which
preserves the L2 norm ||A||_2. The absorbed flag is set to True.

Correctness proof (structural):
  If phase p has unit_modulus=True and is_separable=True, then
  |p(n)| = 1 for all n. Absorbing p into coefficients a_n -> a_n * p(n)
  preserves ||a||_2 because |a_n * p(n)|^2 = |a_n|^2 * |p(n)|^2 = |a_n|^2.
  This is an exact isometry, not an approximation.
"""

from __future__ import annotations

import cmath
import math
import random

from mollifier_theta.core.ir import (
    HistoryEntry,
    Phase,
    Term,
)
from mollifier_theta.core.ledger import TermLedger


class PhaseAbsorb:
    """Absorb separable unit-modulus phases into coefficients."""

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        for term in terms:
            results.append(self._apply_one(term))
        ledger.add_many(results)
        return results

    def _apply_one(self, term: Term) -> Term:
        history = HistoryEntry(
            transform="PhaseAbsorb",
            parent_ids=[term.id],
            description="Absorbed separable unit-modulus phases into coefficients.",
        )

        new_phases: list[Phase] = []
        absorbed_any = False
        for phase in term.phases:
            if phase.is_separable and not phase.absorbed and phase.unit_modulus:
                # Structural correctness: |p(n)| = 1 => ||a*p||_2 = ||a||_2
                new_phases.append(
                    Phase(
                        expression=phase.expression,
                        depends_on=phase.depends_on,
                        is_separable=True,
                        absorbed=True,
                        unit_modulus=phase.unit_modulus,
                    )
                )
                absorbed_any = True
            else:
                new_phases.append(phase)

        if not absorbed_any:
            # Nothing to absorb — return term unchanged (but still new object)
            return term.with_updates(
                history=list(term.history) + [history],
                parents=[term.id],
            )

        return Term(
            kind=term.kind,
            expression=term.expression,
            variables=term.variables,
            ranges=list(term.ranges),
            kernels=list(term.kernels),
            phases=new_phases,
            history=list(term.history) + [history],
            parents=[term.id],
            multiplicity=term.multiplicity,
            kernel_state=term.kernel_state,
            metadata={
                **term.metadata,
                "phases_absorbed": True,
                "absorption_proof": "unit_modulus_isometry",
            },
        )

    def describe(self) -> str:
        return (
            "Phase absorption: separable unit-modulus phases (depending on "
            "single variable) absorbed into coefficient sequences. "
            "L2 norm ||A||_2 is preserved exactly (structural proof: "
            "|p(n)| = 1 => ||a*p||_2 = ||a||_2)."
        )


def verify_absorption_invariant(term: Term) -> list[str]:
    """Verify that all absorbed phases satisfy the structural proof conditions.

    Returns a list of violations (empty = all good).
    """
    violations: list[str] = []
    for phase in term.phases:
        if phase.absorbed and not phase.unit_modulus:
            violations.append(
                f"Phase '{phase.expression}' is absorbed but unit_modulus=False. "
                f"Absorption only preserves L2 norm for unit-modulus phases."
            )
        if phase.absorbed and not phase.is_separable:
            violations.append(
                f"Phase '{phase.expression}' is absorbed but is_separable=False. "
                f"Absorption requires separability."
            )
    return violations


def spot_check_norm_preservation(
    n_samples: int = 100,
    length: int = 50,
    seed: int = 42,
) -> tuple[bool, float, float]:
    """Numerical spot-check: random coefficients * unit phase preserves ||A||_2.

    Returns (passed, norm_before, norm_after).

    Note: This is a supplementary check. The primary correctness guarantee
    is the structural proof in verify_absorption_invariant().
    """
    rng = random.Random(seed)

    # Generate random complex coefficients
    coeffs = [
        complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(length)
    ]

    # Random unit-modulus phases
    phases = [cmath.exp(1j * rng.uniform(0, 2 * math.pi)) for _ in range(length)]

    norm_before = math.sqrt(sum(abs(c) ** 2 for c in coeffs))
    absorbed = [c * p for c, p in zip(coeffs, phases)]
    norm_after = math.sqrt(sum(abs(c) ** 2 for c in absorbed))

    passed = abs(norm_before - norm_after) < 1e-10
    return passed, norm_before, norm_after
