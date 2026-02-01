"""Phase absorption: detect separable phases, absorb into coefficients.

A phase is separable if it depends on a single summation variable.
Absorbing it means folding it into the coefficient sequence, which
preserves the L2 norm ||A||_2. The absorbed flag is set to True.
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
            if phase.is_separable and not phase.absorbed:
                # Mark as absorbed
                new_phases.append(
                    Phase(
                        expression=phase.expression,
                        depends_on=phase.depends_on,
                        is_separable=True,
                        absorbed=True,
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
            metadata={
                **term.metadata,
                "phases_absorbed": True,
            },
        )

    def describe(self) -> str:
        return (
            "Phase absorption: separable unit-modulus phases (depending on "
            "single variable) absorbed into coefficient sequences. "
            "L2 norm ||A||_2 is preserved."
        )


def spot_check_norm_preservation(
    n_samples: int = 100,
    length: int = 50,
    seed: int = 42,
) -> tuple[bool, float, float]:
    """Numerical spot-check: random coefficients * unit phase preserves ||A||_2.

    Returns (passed, norm_before, norm_after).
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
