"""First-class objects for summation structure, coefficient sequences, and additive twists.

These types allow transforms (especially Voronoi) to reason about the
structure of bilinear sums without resorting to string parsing.
All objects are frozen Pydantic models.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field

from mollifier_theta.core.frozen_collections import DeepFreezeModel


class ArithmeticType(str, enum.Enum):
    """Classification of an arithmetic coefficient sequence."""

    DIVISOR = "divisor"          # d_k(n) for some k
    MOEBIUS = "moebius"          # mu(n) related
    HECKE = "hecke"              # a_f(n) for some automorphic form f
    DIRICHLET = "dirichlet"      # Dirichlet series coefficients
    GENERIC = "generic"          # No special structure assumed
    MOLLIFIER = "mollifier"      # mu(n) * P(log n / log y) mollifier coefficients


class VoronoiEligibility(str, enum.Enum):
    """Whether a coefficient sequence is eligible for Voronoi summation."""

    ELIGIBLE = "eligible"        # Has known Voronoi formula
    INELIGIBLE = "ineligible"    # No known Voronoi
    UNKNOWN = "unknown"          # Not yet determined


class SumIndex(DeepFreezeModel):
    """A summation index with range and coprimality constraints."""

    model_config = {"frozen": True}

    name: str
    range_lower: str = "1"
    range_upper: str = "T^theta"
    range_description: str = ""
    coprime_to: list[str] = Field(default_factory=list)


class CoeffSeq(DeepFreezeModel):
    """A coefficient sequence attached to a summation variable.

    Tracks the arithmetic type, norm information, and Voronoi eligibility.
    """

    model_config = {"frozen": True}

    name: str                                    # e.g. "a_m", "b_n"
    variable: str                                # sum variable this attaches to
    arithmetic_type: ArithmeticType = ArithmeticType.GENERIC
    voronoi_eligible: VoronoiEligibility = VoronoiEligibility.UNKNOWN
    norm_bound: str = ""                         # e.g. "||a||_2 << T^{theta/2+eps}"
    description: str = ""
    citations: list[str] = Field(default_factory=list)


class AdditiveTwist(DeepFreezeModel):
    """An additive character twist e(±a*n/c) in a bilinear sum.

    This is the key structure that Voronoi summation acts on.
    """

    model_config = {"frozen": True}

    modulus: str                                 # modulus variable (e.g. "c")
    numerator: str                               # numerator variable (e.g. "a")
    sum_variable: str                            # which sum variable is twisted (e.g. "n")
    sign: int = 1                                # +1 or -1
    invert_numerator: bool = False               # True if twist is e(ā n/c) (modular inverse)
    description: str = ""

    def format_phase_expression(self) -> str:
        """Canonical phase expression string for this twist.

        Used by both DeltaMethodCollapse (to create phases) and
        KloostermanForm (to match and consume phases).
        """
        sign_str = "" if self.sign > 0 else "-"
        inv_str = "bar" if self.invert_numerator else ""
        return (
            f"e({sign_str}{self.numerator}{inv_str}"
            f"*{self.sum_variable}/{self.modulus})"
        )


class WeightKernel(DeepFreezeModel):
    """Classification of the smooth weight function in a sum.

    Different from the Kernel IR type — this is specifically about
    the weight attached to a summation, tracking what Voronoi does to it.
    """

    model_config = {"frozen": True}

    kind: str                                    # "smooth", "bessel_transform", "voronoi_dual"
    original_name: str = ""                      # Name of the kernel before transformation
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class SumStructure(DeepFreezeModel):
    """Complete structural description of a bilinear sum for transform matching.

    Contains the sum indices, coefficient sequences, additive twists,
    and weight kernels needed for Voronoi and other transforms to
    pattern-match and rewrite.
    """

    model_config = {"frozen": True}

    sum_indices: list[SumIndex] = Field(default_factory=list)
    coeff_seqs: list[CoeffSeq] = Field(default_factory=list)
    additive_twists: list[AdditiveTwist] = Field(default_factory=list)
    weight_kernels: list[WeightKernel] = Field(default_factory=list)

    def get_twist_for_variable(self, var_name: str) -> AdditiveTwist | None:
        """Find the additive twist acting on a given sum variable."""
        for twist in self.additive_twists:
            if twist.sum_variable == var_name:
                return twist
        return None

    def get_coeff_for_variable(self, var_name: str) -> CoeffSeq | None:
        """Find the coefficient sequence for a given sum variable."""
        for cs in self.coeff_seqs:
            if cs.variable == var_name:
                return cs
        return None

    def has_voronoi_eligible_twist(self) -> bool:
        """Check if any sum variable has both a twist and eligible coefficients."""
        for twist in self.additive_twists:
            cs = self.get_coeff_for_variable(twist.sum_variable)
            if cs and cs.voronoi_eligible == VoronoiEligibility.ELIGIBLE:
                return True
        return False
