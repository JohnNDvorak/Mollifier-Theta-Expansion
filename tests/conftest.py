"""Shared fixtures for mollifier-theta tests."""

from __future__ import annotations

import pytest

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


@pytest.fixture
def integral_term() -> Term:
    """A basic integral term representing int_{0}^{T} |zeta(1/2+it)|^2 dt."""
    return Term(
        kind=TermKind.INTEGRAL,
        expression="int_0^T |zeta(1/2+it)|^2 dt",
        variables=["t"],
        ranges=[Range(variable="t", lower="0", upper="T")],
    )


@pytest.fixture
def mollified_integral_term() -> Term:
    """Integral term with mollifier: int |M*zeta|^2."""
    return Term(
        kind=TermKind.INTEGRAL,
        expression="int_0^T |M(1/2+it) zeta(1/2+it)|^2 dt",
        variables=["t"],
        ranges=[Range(variable="t", lower="0", upper="T")],
        metadata={"mollifier_length": 3},
    )


@pytest.fixture
def dirichlet_sum_term() -> Term:
    """A Dirichlet sum term with kernel."""
    return Term(
        kind=TermKind.DIRICHLET_SUM,
        expression="sum_{n<=x} a_n n^{-1/2-it} W(n/x)",
        variables=["n", "t"],
        ranges=[
            Range(variable="n", lower="1", upper="x"),
            Range(variable="t", lower="0", upper="T"),
        ],
        kernels=[
            Kernel(name="W_AFE", support="[0,inf)", argument="n/x")
        ],
    )


@pytest.fixture
def cross_term_with_phase() -> Term:
    """A cross-term with explicit phase."""
    return Term(
        kind=TermKind.CROSS,
        expression="sum a_{ell1,m} conj(a_{ell2,n}) (m/n)^{it}",
        variables=["m", "n", "t"],
        phases=[
            Phase(expression="(m/n)^{it}", depends_on=["m", "n", "t"])
        ],
    )


@pytest.fixture
def diagonal_term() -> Term:
    """A diagonal term (am=bn)."""
    return Term(
        kind=TermKind.DIAGONAL,
        expression="sum_{m} |a_m|^2 W(m)",
        variables=["m"],
    )


@pytest.fixture
def off_diagonal_term() -> Term:
    """An off-diagonal term with phase."""
    return Term(
        kind=TermKind.OFF_DIAGONAL,
        expression="sum_{m!=n} a_m conj(b_n) K(m,n)",
        variables=["m", "n"],
        phases=[
            Phase(expression="(m/n)^{it}", depends_on=["m", "n", "t"])
        ],
        kernels=[
            Kernel(name="FourierKernel", argument="log(m/n)")
        ],
    )


@pytest.fixture
def bound_term() -> Term:
    """A properly bounded term with citation."""
    return Term(
        kind=TermKind.OFF_DIAGONAL,
        expression="DI bound of off-diagonal",
        status=TermStatus.BOUND_ONLY,
        lemma_citation="Deshouillers-Iwaniec 1982, Theorem 1",
        scale_model="T^(3*theta - 1)",
    )


@pytest.fixture
def empty_ledger() -> TermLedger:
    return TermLedger()


@pytest.fixture
def populated_ledger(
    integral_term: Term, dirichlet_sum_term: Term, bound_term: Term
) -> TermLedger:
    ledger = TermLedger()
    ledger.add(integral_term)
    ledger.add(dirichlet_sum_term)
    ledger.add(bound_term)
    return ledger
