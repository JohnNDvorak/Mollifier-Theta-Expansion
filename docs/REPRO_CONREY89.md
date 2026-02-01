# Reproduction: Conrey 1989

## What the Reproduction Proves

This pipeline reproduces the chain of analytic reductions in Conrey (1989) that establishes: the mollifier M of length T^theta with theta < 4/7 yields a mollified second moment with a computable main term, and all error terms are o(T).

Specifically, for theta = 4/7 - epsilon, the off-diagonal error has size T^{7*theta/4 + epsilon} < T, so the mollified second moment is asymptotic to T * P(theta) * (log T)^k.

## How the Reproduction Works

1. **Approximate Functional Equation:** The zeta integral becomes two Dirichlet sums + negligible error.
2. **Open the Square:** |M*zeta|^2 expands into K*(K+1)/2 cross-term families.
3. **Mean Value Integration:** t-integral -> Fourier kernel (NOT delta-approximated).
4. **Diagonal/Off-diagonal Split:** Structural decomposition.
5. **Diagonal Extraction:** Main term T * P(theta) * (log T)^k.
6. **Off-diagonal Reduction:** Delta method -> Kloosterman form -> Phase absorption.
7. **DI Bound:** Bilinear Kloosterman bound yields E(theta) = 7*theta/4.
8. **Theta Check:** E(theta) < 1 iff theta < 4/7.

## Two-Layer Verification

- **Layer 1:** DIExponentModel derives theta_max = 4/7 symbolically from the exponent algebra.
- **Layer 2:** Cross-check against KNOWN_THETA_MAX = 4/7. Mismatch = build failure.

## Critical Regression Tests

- `conrey89_pipeline(theta=0.56)` PASS
- `conrey89_pipeline(theta=0.58)` FAIL
- `DIExponentModel.theta_max()` returns 4/7
- `find_theta_max()` returns ~0.5714
