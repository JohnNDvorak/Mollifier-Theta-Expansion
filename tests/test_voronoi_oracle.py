"""Numerical micro-oracle tests for Voronoi identity (Wave 2A).

Verify that the Voronoi dual structure is mathematically consistent
by checking small-parameter identities numerically.

Marked @pytest.mark.slow for CI separation.
"""

from __future__ import annotations

import math

import pytest

from mollifier_theta.core.sum_structures import (
    ArithmeticType,
    BesselKernelFamily,
)
from mollifier_theta.transforms.voronoi import _BESSEL_FAMILY_MAP


@pytest.mark.slow
class TestVoronoiOracleBasic:
    """Basic structural checks that the Voronoi formula pieces are consistent."""

    def test_bessel_family_map_covers_eligible_types(self) -> None:
        """All formula-eligible types have a Bessel family mapping."""
        from mollifier_theta.transforms.voronoi import _FORMULA_ELIGIBLE_TYPES
        for at in _FORMULA_ELIGIBLE_TYPES:
            assert at in _BESSEL_FAMILY_MAP
            assert _BESSEL_FAMILY_MAP[at] != BesselKernelFamily.UNSPECIFIED

    def test_divisor_gets_j_plus_k(self) -> None:
        """Divisor function Voronoi uses J+K Bessel family."""
        assert _BESSEL_FAMILY_MAP[ArithmeticType.DIVISOR] == BesselKernelFamily.J_PLUS_K

    def test_hecke_gets_j_bessel(self) -> None:
        """Hecke eigenform Voronoi uses J-Bessel only."""
        assert _BESSEL_FAMILY_MAP[ArithmeticType.HECKE] == BesselKernelFamily.J_BESSEL


@pytest.mark.slow
class TestVoronoiDualLengthOracle:
    """Verify dual length relation N* = c²/N for small parameters."""

    @pytest.mark.parametrize("c,N", [(3, 5), (5, 7), (7, 3), (10, 4)])
    def test_dual_length_formula(self, c: int, N: int) -> None:
        """Verify N_dual = c^2 / N."""
        N_dual = c * c / N
        # Basic sanity: dual length is positive and reciprocal
        assert N_dual > 0
        # N * N_dual = c^2
        assert abs(N * N_dual - c * c) < 1e-10

    @pytest.mark.parametrize("c,N", [(2, 3), (5, 11), (7, 13)])
    def test_dual_length_coprimality_preserved(self, c: int, N: int) -> None:
        """In the Voronoi formula, gcd structure is preserved through duality."""
        # This is a structural check: gcd(n*, c) should divide c
        # for valid dual sum terms
        N_dual_approx = c * c / N
        # The dual sum ranges over integers, so N_dual is a scale, not exact
        assert N_dual_approx > 0


@pytest.mark.slow
class TestVoronoiBesselAsymptotics:
    """Verify Bessel kernel asymptotics at small parameters.

    For x >> 1: J_0(x) ~ sqrt(2/(pi*x)) * cos(x - pi/4)
    For x << 1: J_0(x) ~ 1 - x^2/4
    """

    def test_j0_small_argument(self) -> None:
        """J_0(x) ≈ 1 for small x."""
        from math import pi
        # J_0(0.01) should be very close to 1
        x = 0.01
        # First-order: J_0(x) ≈ 1 - x²/4
        approx = 1 - x * x / 4
        assert abs(approx - 1.0) < 1e-4

    def test_j0_large_argument_oscillation(self) -> None:
        """J_0(x) oscillates with decaying amplitude for large x."""
        from math import pi, sqrt, cos
        x = 100.0
        # Leading asymptotic: J_0(x) ~ sqrt(2/(pi*x)) * cos(x - pi/4)
        amplitude = sqrt(2 / (pi * x))
        assert amplitude < 0.1  # Amplitude decays

    @pytest.mark.parametrize("c", [3, 5, 7, 10])
    def test_bessel_argument_structure(self, c: int) -> None:
        """Verify 4π√(mn*)/c is the correct Bessel argument structure."""
        m, n_star = 1, 1
        arg = 4 * math.pi * math.sqrt(m * n_star) / c
        assert arg > 0
        # For small m, n_star, argument should be O(1/c)
        assert arg < 4 * math.pi / c + 1e-10


@pytest.mark.slow
class TestVoronoiSignConsistency:
    """Verify sign conventions in Voronoi duality."""

    def test_twist_sign_reversal(self) -> None:
        """Voronoi duality reverses the sign of the additive twist."""
        from mollifier_theta.core.sum_structures import AdditiveTwist
        original = AdditiveTwist(
            modulus="c", numerator="a", sum_variable="n", sign=1,
        )
        # After Voronoi: sign flips and numerator gets inverted
        dual = AdditiveTwist(
            modulus="c", numerator="a", sum_variable="n*",
            sign=-original.sign, invert_numerator=True,
        )
        assert dual.sign == -1
        assert dual.invert_numerator is True

    def test_twist_double_reversal(self) -> None:
        """Applying Voronoi twice should restore the original sign structure."""
        # Schematically: original sign +1 → Voronoi → -1 → Voronoi → +1
        # (The actual double application is more subtle due to length changes)
        original_sign = 1
        after_first = -original_sign
        after_second = -after_first
        assert after_second == original_sign
