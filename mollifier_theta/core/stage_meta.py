"""Typed stage metadata models for pipeline transforms.

Replaces ad-hoc string-key metadata with frozen Pydantic models
stored under reserved keys ("_delta", "_voronoi", "_kloosterman", "_bound").

Dual-write: transforms write BOTH old string keys AND new typed meta
during transition. Typed accessors provide safe retrieval.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mollifier_theta.core.frozen_collections import DeepFreezeModel


class DeltaMethodMeta(DeepFreezeModel):
    """Typed metadata for delta method stages."""

    model_config = {"frozen": True, "extra": "forbid"}

    applied: bool = False
    collapsed: bool = False
    stage: str = ""
    modulus_variable: str = "c"


class VoronoiMeta(DeepFreezeModel):
    """Typed metadata for Voronoi transform."""

    model_config = {"frozen": True, "extra": "forbid"}

    applied: bool = False
    target_variable: str = ""
    dual_variable: str = ""
    dual_length: str = ""


class KloostermanMeta(DeepFreezeModel):
    """Typed metadata for Kloosterman form."""

    model_config = {"frozen": True, "extra": "forbid"}

    formed: bool = False
    variables: list[str] = []
    consumed_phases: list[str] = []  # Phase expressions consumed into S(m,n;c)/c


class BoundMeta(DeepFreezeModel):
    """Typed metadata for bounding lemmas."""

    model_config = {"frozen": True, "extra": "forbid"}

    strategy: str = ""
    error_exponent: str = ""
    citation: str = ""
    bound_family: str = ""


# Reserved metadata keys
_DELTA_KEY = "_delta"
_VORONOI_KEY = "_voronoi"
_KLOOSTERMAN_KEY = "_kloosterman"
_BOUND_KEY = "_bound"


def get_delta_meta(term: Any) -> DeltaMethodMeta | None:
    """Extract typed delta method metadata from a term, if present."""
    raw = term.metadata.get(_DELTA_KEY)
    if raw is None:
        return None
    if isinstance(raw, DeltaMethodMeta):
        return raw
    return DeltaMethodMeta.model_validate(raw)


def get_voronoi_meta(term: Any) -> VoronoiMeta | None:
    """Extract typed Voronoi metadata from a term, if present."""
    raw = term.metadata.get(_VORONOI_KEY)
    if raw is None:
        return None
    if isinstance(raw, VoronoiMeta):
        return raw
    return VoronoiMeta.model_validate(raw)


def get_kloosterman_meta(term: Any) -> KloostermanMeta | None:
    """Extract typed Kloosterman metadata from a term, if present."""
    raw = term.metadata.get(_KLOOSTERMAN_KEY)
    if raw is None:
        return None
    if isinstance(raw, KloostermanMeta):
        return raw
    return KloostermanMeta.model_validate(raw)


def get_bound_meta(term: Any) -> BoundMeta | None:
    """Extract typed bound metadata from a term, if present."""
    raw = term.metadata.get(_BOUND_KEY)
    if raw is None:
        return None
    if isinstance(raw, BoundMeta):
        return raw
    return BoundMeta.model_validate(raw)
