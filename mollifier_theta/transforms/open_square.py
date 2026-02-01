"""Open the square |M*zeta|^2 -> cross-term families.

For mollifier M = sum_{ell=1}^{K} mu(ell)/ell^s * (mollifier coefficients),
opening |M*zeta|^2 produces K*(K+1)/2 cross-term families, one per
unordered pair (ell1, ell2). Off-diagonal pairs (ell1 != ell2) carry
multiplicity 2. Conjugation phases are tracked explicitly.
"""

from __future__ import annotations

from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    Phase,
    Range,
    Term,
    TermKind,
    TermStatus,
)
from mollifier_theta.core.ledger import TermLedger


class OpenSquare:
    """Open |M*zeta|^2 into cross-term families."""

    def __init__(self, K: int = 3) -> None:
        self.K = K

    def apply(self, terms: list[Term], ledger: TermLedger) -> list[Term]:
        results: list[Term] = []
        for term in terms:
            results.extend(self._apply_one(term))
        ledger.add_many(results)
        return results

    def _apply_one(self, term: Term) -> list[Term]:
        cross_terms: list[Term] = []

        for ell1 in range(1, self.K + 1):
            for ell2 in range(ell1, self.K + 1):
                is_diagonal_pair = ell1 == ell2
                multiplicity = 1 if is_diagonal_pair else 2

                history = HistoryEntry(
                    transform="OpenSquare",
                    parent_ids=[term.id],
                    description=f"Cross-term (ell1={ell1}, ell2={ell2}), mult={multiplicity}",
                )

                # Phase from conjugation: (m/n)^{it} when ell1 != ell2
                phases = list(term.phases)
                if not is_diagonal_pair:
                    phases.append(
                        Phase(
                            expression=f"(ell{ell1}_m / ell{ell2}_n)^{{it}}",
                            depends_on=["m", "n", "t"],
                            is_separable=False,
                            unit_modulus=True,
                        )
                    )

                cross = Term(
                    kind=TermKind.CROSS,
                    expression=(
                        f"sum_{{m,n}} a_{{ell{ell1},m}} conj(a_{{ell{ell2},n}}) "
                        f"(ell{ell1}*m)^{{-1/2-it}} (ell{ell2}*n)^{{-1/2+it}} W(m) W(n)"
                    ),
                    variables=["m", "n", "t"],
                    ranges=[
                        Range(variable="m", lower="1", upper="T^theta"),
                        Range(variable="n", lower="1", upper="T^theta"),
                        Range(variable="t", lower="0", upper="T"),
                    ],
                    kernels=list(term.kernels),
                    phases=phases,
                    history=list(term.history) + [history],
                    parents=[term.id],
                    multiplicity=multiplicity,
                    metadata={
                        "ell1": ell1,
                        "ell2": ell2,
                        "is_diagonal_pair": is_diagonal_pair,
                    },
                )
                cross_terms.append(cross)

        return cross_terms

    def describe(self) -> str:
        n_terms = self.K * (self.K + 1) // 2
        return (
            f"Open |M*zeta|^2 with K={self.K} mollifier terms into "
            f"{n_terms} cross-term families. Off-diagonal pairs carry "
            f"multiplicity 2. Conjugation phases tracked explicitly."
        )
