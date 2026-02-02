"""Length-aware DI exponent model with independent sum-length exponents.

The standard DI bilinear Kloosterman bound (Theorem 12) for:
    sum_{m~M, n~N} a_m b_n S(m,n;c)/c
summed over c ~ C gives error bounded by:
    T^eps * [ (MN)^{1/2}*C + (MN)^{1/2}*(MNC)^{1/2} + MN*C^{1/2} ]^{1/2}

With M=T^alpha, N=T^beta, C=T^gamma, the three sub-terms have T-exponents:
    sub_A = (alpha+beta)/2 + gamma
    sub_B = alpha + beta + gamma/2     (terms 2 and 3 coincide)

So the error exponent from the raw DI inequality is:
    E_raw(alpha,beta,gamma) = max(sub_A, sub_B) / 2

IMPORTANT: In the symmetric case (alpha=beta=theta, gamma=1-theta), the raw
DI inequality gives E_raw = 3*theta/4 + 1/4, which is LESS restrictive than
the known E(theta) = 7*theta/4 from Conrey 1989.  The 7*theta/4 exponent
arises from the full second-moment bookkeeping (T-integral, delta-method
ranges, mean-value diagonal), not from a single application of the DI bound.

This module provides both:
  - The raw DI formula (for asymmetric exploration)
  - The known 7*theta/4 constraint (for symmetric cross-check)

SymPy containment: all symbolic operations go through ScaleModel.
"""

from __future__ import annotations

from dataclasses import dataclass

from mollifier_theta.analysis.exponent_model import ExponentConstraint
from mollifier_theta.core.scale_model import ScaleModel


@dataclass(frozen=True)
class LengthAwareDIModel:
    """Parametric DI exponent with independent length exponents.

    All exponents are stored as strings in theta, evaluated via ScaleModel.

    Attributes:
        alpha_str: T-exponent of M (first sum length), as string in theta.
        beta_str: T-exponent of N (second sum length), as string in theta.
        gamma_str: T-exponent of C (modulus range), as string in theta.
        label: Human-readable label for this configuration.
    """

    alpha_str: str
    beta_str: str
    gamma_str: str
    label: str = ""

    @classmethod
    def symmetric(cls) -> LengthAwareDIModel:
        """M = N = T^theta, C = T^{1-theta}."""
        return cls(
            alpha_str="theta",
            beta_str="theta",
            gamma_str="1 - theta",
            label="symmetric",
        )

    @classmethod
    def voronoi_dual(cls) -> LengthAwareDIModel:
        """M = T^theta, N* = T^{2-3*theta}, C = T^{1-theta}.

        After Voronoi summation on the n-variable, the dual length is
        N* ~ c^2/N ~ T^{2(1-theta)} / T^theta = T^{2-3*theta}.
        """
        return cls(
            alpha_str="theta",
            beta_str="2 - 3*theta",
            gamma_str="1 - theta",
            label="voronoi_dual",
        )

    @property
    def sub_A_str(self) -> str:
        """First sub-term exponent string: (alpha+beta)/2 + gamma."""
        return f"({self.alpha_str} + {self.beta_str})/2 + ({self.gamma_str})"

    @property
    def sub_B_str(self) -> str:
        """Second sub-term exponent string: alpha + beta + gamma/2."""
        return f"({self.alpha_str}) + ({self.beta_str}) + ({self.gamma_str})/2"

    @property
    def error_exponent_str(self) -> str:
        """Raw DI error exponent string: Max(sub_A, sub_B) / 2."""
        return f"Max({self.sub_A_str}, {self.sub_B_str}) / 2"

    @property
    def name(self) -> str:
        return f"LengthAwareDI_{self.label}" if self.label else "LengthAwareDI"

    def evaluate_error(self, theta_val: float) -> float:
        """Evaluate the error exponent at a specific theta value."""
        return ScaleModel.evaluate_expr(self.error_exponent_str, theta_val)

    def theta_max(self) -> float:
        """Solve E(theta) = 1 numerically.

        Uses ScaleModel's bisection fallback since Max expressions
        are not algebraically solvable by SymPy.
        """
        sm = ScaleModel(T_exponent=self.error_exponent_str)
        roots = sm.solve_all_roots(lo=0.0, hi=1.0)
        if not roots:
            # No root in (0,1) means E < 1 for all theta in (0,1)
            return 1.0
        return roots[0]

    def constraints(self) -> list[ExponentConstraint]:
        """Exponent constraints from this model.

        For the symmetric case, includes BOTH the raw DI formula AND
        the known 7*theta/4 constraint.  For asymmetric cases, only
        the raw DI formula is returned.
        """
        raw_expr = ScaleModel.simplify_expr(self.error_exponent_str)
        result = [
            ExponentConstraint(
                name=f"di_raw_{self.label}" if self.label else "di_raw",
                expression_str=raw_expr,
                description=(
                    f"Raw DI bilinear bound: alpha={self.alpha_str}, "
                    f"beta={self.beta_str}, gamma={self.gamma_str}"
                ),
                citation="Deshouillers-Iwaniec 1982/83, Theorem 12",
                bound_family=self.name,
            ),
        ]

        if self.label == "symmetric":
            result.append(
                ExponentConstraint(
                    name="di_conrey_7theta4",
                    expression_str="7*theta/4",
                    description=(
                        "Full second-moment DI exponent from Conrey 1989. "
                        "More restrictive than the raw DI inequality alone."
                    ),
                    citation="Conrey 1989, Section 4",
                    bound_family="DI_Kloosterman",
                ),
            )

        return result

    def sub_A_at(self, theta_val: float) -> float:
        """Evaluate sub_A at a specific theta."""
        return ScaleModel.evaluate_expr(self.sub_A_str, theta_val)

    def sub_B_at(self, theta_val: float) -> float:
        """Evaluate sub_B at a specific theta."""
        return ScaleModel.evaluate_expr(self.sub_B_str, theta_val)
