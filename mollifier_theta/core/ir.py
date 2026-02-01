"""Core IR types for the mollifier theta expansion framework.

All objects are frozen Pydantic models — immutable after construction.
"""

from __future__ import annotations

import enum
import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator


class TermStatus(str, enum.Enum):
    ACTIVE = "Active"
    MAIN_TERM = "MainTerm"
    BOUND_ONLY = "BoundOnly"
    ERROR = "Error"


class TermKind(str, enum.Enum):
    INTEGRAL = "Integral"
    DIRICHLET_SUM = "DirichletSum"
    CROSS = "Cross"
    DIAGONAL = "Diagonal"
    OFF_DIAGONAL = "OffDiagonal"
    KLOOSTERMAN = "Kloosterman"
    ERROR = "Error"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Range(BaseModel):
    """Summation or integration range for a variable."""

    model_config = {"frozen": True}

    variable: str
    lower: str = "1"
    upper: str = "T"
    description: str = ""


class Kernel(BaseModel):
    """Smooth kernel attached to a term (W, Fourier, delta-method, etc.)."""

    model_config = {"frozen": True}

    name: str
    support: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    argument: str = ""
    description: str = ""

    def mellin_transform(self) -> str:
        """Symbolic description of Mellin transform (for documentation)."""
        return self.properties.get("mellin_transform", f"Mellin({self.name})")

    def residue_structure(self) -> str:
        """Symbolic description of residue structure (for documentation)."""
        return self.properties.get("residue_structure", f"Res({self.name})")


class Phase(BaseModel):
    """Phase factor tracked on a term."""

    model_config = {"frozen": True}

    expression: str
    is_separable: bool = False
    absorbed: bool = False
    depends_on: list[str] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    """Single step in a term's derivation history."""

    model_config = {"frozen": True}

    transform: str
    parent_ids: list[str] = Field(default_factory=list)
    description: str = ""


class Term(BaseModel):
    """Central IR node: one symbolic term in the expansion.

    Frozen after construction — transforms create new terms.
    """

    model_config = {"frozen": True}

    id: str = Field(default_factory=_new_id)
    kind: TermKind
    expression: str = ""
    variables: list[str] = Field(default_factory=list)
    ranges: list[Range] = Field(default_factory=list)
    kernels: list[Kernel] = Field(default_factory=list)
    phases: list[Phase] = Field(default_factory=list)
    scale_model: str = ""
    history: list[HistoryEntry] = Field(default_factory=list)
    status: TermStatus = TermStatus.ACTIVE
    parents: list[str] = Field(default_factory=list)
    lemma_citation: str = ""
    multiplicity: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_bound_only_has_citation(self) -> "Term":
        if self.status == TermStatus.BOUND_ONLY and not self.lemma_citation:
            raise ValueError(
                "BoundOnly terms must have a non-empty lemma_citation"
            )
        return self

    def with_updates(self, **kwargs: Any) -> "Term":
        """Return a new Term with specified fields replaced."""
        data = self.model_dump()
        data.update(kwargs)
        if "id" not in kwargs:
            data["id"] = _new_id()
        return Term(**data)
