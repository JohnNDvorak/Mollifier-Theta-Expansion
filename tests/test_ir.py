"""Tests for core IR types."""

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


class TestTermConstruction:
    def test_basic_construction(self, integral_term: Term) -> None:
        assert integral_term.kind == TermKind.INTEGRAL
        assert integral_term.status == TermStatus.ACTIVE
        assert len(integral_term.id) == 12

    def test_auto_id_generation(self) -> None:
        t1 = Term(kind=TermKind.INTEGRAL)
        t2 = Term(kind=TermKind.INTEGRAL)
        assert t1.id != t2.id

    def test_default_fields(self) -> None:
        t = Term(kind=TermKind.INTEGRAL)
        assert t.variables == []
        assert t.ranges == []
        assert t.kernels == []
        assert t.phases == []
        assert t.history == []
        assert t.parents == []
        assert t.lemma_citation == ""
        assert t.multiplicity == 1

    def test_all_term_kinds_constructible(self) -> None:
        for kind in TermKind:
            t = Term(kind=kind)
            assert t.kind == kind


class TestTermImmutability:
    def test_frozen_model(self, integral_term: Term) -> None:
        with pytest.raises(Exception):
            integral_term.kind = TermKind.DIAGONAL  # type: ignore[misc]

    def test_frozen_status(self, integral_term: Term) -> None:
        with pytest.raises(Exception):
            integral_term.status = TermStatus.MAIN_TERM  # type: ignore[misc]

    def test_with_updates_creates_new(self, integral_term: Term) -> None:
        updated = integral_term.with_updates(kind=TermKind.DIAGONAL)
        assert updated.kind == TermKind.DIAGONAL
        assert integral_term.kind == TermKind.INTEGRAL
        assert updated.id != integral_term.id

    def test_with_updates_preserves_other_fields(
        self, dirichlet_sum_term: Term
    ) -> None:
        updated = dirichlet_sum_term.with_updates(
            status=TermStatus.MAIN_TERM
        )
        assert updated.variables == dirichlet_sum_term.variables
        assert len(updated.kernels) == len(dirichlet_sum_term.kernels)


class TestTermValidation:
    def test_bound_only_requires_citation(self) -> None:
        with pytest.raises(ValueError, match="lemma_citation"):
            Term(
                kind=TermKind.DIAGONAL,
                status=TermStatus.BOUND_ONLY,
            )

    def test_bound_only_with_citation_ok(self) -> None:
        t = Term(
            kind=TermKind.DIAGONAL,
            status=TermStatus.BOUND_ONLY,
            lemma_citation="Weil bound",
        )
        assert t.status == TermStatus.BOUND_ONLY

    def test_active_without_citation_ok(self) -> None:
        t = Term(kind=TermKind.INTEGRAL)
        assert t.lemma_citation == ""

    def test_error_without_citation_ok(self) -> None:
        t = Term(kind=TermKind.ERROR, status=TermStatus.ERROR)
        assert t.status == TermStatus.ERROR


class TestKernel:
    def test_construction(self) -> None:
        k = Kernel(name="W_AFE", support="[0,inf)", argument="n/x")
        assert k.name == "W_AFE"
        assert k.argument == "n/x"

    def test_mellin_transform(self) -> None:
        k = Kernel(
            name="W",
            properties={"mellin_transform": "Gamma(s)/Gamma(1/2)"},
        )
        assert "Gamma" in k.mellin_transform()

    def test_residue_structure(self) -> None:
        k = Kernel(name="W")
        assert "Res" in k.residue_structure()


class TestPhase:
    def test_construction(self) -> None:
        p = Phase(expression="(m/n)^{it}", depends_on=["m", "n", "t"])
        assert p.expression == "(m/n)^{it}"
        assert not p.absorbed

    def test_separable_flag(self) -> None:
        p = Phase(expression="m^{it}", is_separable=True, depends_on=["m", "t"])
        assert p.is_separable


class TestRange:
    def test_construction(self) -> None:
        r = Range(variable="n", lower="1", upper="T^theta")
        assert r.variable == "n"
        assert r.upper == "T^theta"


class TestHistoryEntry:
    def test_construction(self) -> None:
        h = HistoryEntry(
            transform="ApproxFunctionalEq",
            parent_ids=["abc123"],
            description="Applied AFE",
        )
        assert h.transform == "ApproxFunctionalEq"
        assert h.parent_ids == ["abc123"]


class TestTermSerialization:
    def test_model_dump_roundtrip(self, dirichlet_sum_term: Term) -> None:
        data = dirichlet_sum_term.model_dump()
        reconstructed = Term(**data)
        assert reconstructed.kind == dirichlet_sum_term.kind
        assert reconstructed.expression == dirichlet_sum_term.expression
        assert len(reconstructed.kernels) == len(dirichlet_sum_term.kernels)
