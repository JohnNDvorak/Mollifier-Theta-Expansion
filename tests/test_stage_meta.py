"""Tests for typed stage metadata (WI-2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mollifier_theta.core.ir import Term, TermKind
from mollifier_theta.core.stage_meta import (
    BoundMeta,
    DeltaMethodMeta,
    KloostermanMeta,
    VoronoiMeta,
    get_bound_meta,
    get_delta_meta,
    get_kloosterman_meta,
    get_voronoi_meta,
)


class TestDeltaMethodMeta:
    def test_construction(self) -> None:
        meta = DeltaMethodMeta(applied=True, collapsed=False, stage="setup")
        assert meta.applied is True
        assert meta.collapsed is False
        assert meta.stage == "setup"

    def test_typo_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            DeltaMethodMeta(appliedd=True)  # typo

    def test_roundtrip_json(self) -> None:
        meta = DeltaMethodMeta(applied=True, collapsed=True, stage="collapsed")
        data = meta.model_dump()
        restored = DeltaMethodMeta.model_validate(data)
        assert restored.applied == meta.applied
        assert restored.collapsed == meta.collapsed
        assert restored.stage == meta.stage


class TestVoronoiMeta:
    def test_construction(self) -> None:
        meta = VoronoiMeta(
            applied=True, target_variable="n",
            dual_variable="n*", dual_length="c^2/T^theta",
        )
        assert meta.applied is True
        assert meta.dual_variable == "n*"

    def test_typo_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            VoronoiMeta(appplied=True)  # typo


class TestKloostermanMeta:
    def test_construction(self) -> None:
        meta = KloostermanMeta(formed=True, variables=["m", "n", "c"])
        assert meta.formed is True
        assert meta.variables == ["m", "n", "c"]


class TestBoundMeta:
    def test_construction(self) -> None:
        meta = BoundMeta(
            strategy="DI_Kloosterman",
            error_exponent="7*theta/4",
            citation="DI 1982",
            bound_family="DI_Kloosterman",
        )
        assert meta.strategy == "DI_Kloosterman"
        assert meta.error_exponent == "7*theta/4"


class TestAccessors:
    def test_get_delta_meta_present(self) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            metadata={
                "_delta": DeltaMethodMeta(
                    applied=True, collapsed=False, stage="setup",
                ).model_dump(),
            },
        )
        meta = get_delta_meta(term)
        assert meta is not None
        assert meta.applied is True
        assert meta.collapsed is False

    def test_get_delta_meta_absent(self) -> None:
        term = Term(kind=TermKind.INTEGRAL)
        assert get_delta_meta(term) is None

    def test_get_voronoi_meta_present(self) -> None:
        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            metadata={
                "_voronoi": VoronoiMeta(
                    applied=True, target_variable="n",
                    dual_variable="n*", dual_length="c^2/T^theta",
                ).model_dump(),
            },
        )
        meta = get_voronoi_meta(term)
        assert meta is not None
        assert meta.target_variable == "n"

    def test_get_kloosterman_meta_present(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            metadata={
                "_kloosterman": KloostermanMeta(
                    formed=True, variables=["m", "n", "c"],
                ).model_dump(),
            },
        )
        meta = get_kloosterman_meta(term)
        assert meta is not None
        assert meta.formed is True

    def test_get_bound_meta_present(self) -> None:
        term = Term(
            kind=TermKind.KLOOSTERMAN,
            status="BoundOnly",
            lemma_citation="test",
            metadata={
                "_bound": BoundMeta(
                    strategy="DI_Kloosterman",
                    error_exponent="7*theta/4",
                    citation="DI 1982",
                    bound_family="DI_Kloosterman",
                ).model_dump(),
            },
        )
        meta = get_bound_meta(term)
        assert meta is not None
        assert meta.bound_family == "DI_Kloosterman"


class TestPipelineTermsCarryTypedMeta:
    def test_pipeline_terms_have_delta_meta(self) -> None:
        """After running delta method, terms carry both old and new keys."""
        from mollifier_theta.core.ledger import TermLedger
        from mollifier_theta.transforms.delta_method import DeltaMethodSetup

        term = Term(
            kind=TermKind.OFF_DIAGONAL,
            variables=["m", "n"],
        )
        ledger = TermLedger()
        ledger.add(term)
        results = DeltaMethodSetup().apply([term], ledger)

        for r in results:
            if r.kind == TermKind.OFF_DIAGONAL and r.metadata.get("delta_method_applied"):
                # Old key present
                assert r.metadata["delta_method_applied"] is True
                # New typed meta present
                meta = get_delta_meta(r)
                assert meta is not None
                assert meta.applied is True
                assert meta.collapsed is False
