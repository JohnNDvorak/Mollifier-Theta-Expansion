# Lemma Catalog

## DI Bilinear Kloosterman Bound

**Citation:** Deshouillers-Iwaniec 1982/83, Theorem 12; Conrey 1989, Section 4

**Statement:** For bilinear sums of the form
  sum_{m,n} a_m b_n S(m,n;c)/c * V(...)
the DI bound provides square-root cancellation in both m and n variables simultaneously, via spectral theory of automorphic forms.

**Result:** Off-diagonal error = O(T^{7*theta/4 + epsilon}), admissible iff theta < 4/7.

**Implementation:** `lemmas/di_kloosterman.py` — two-layer verification (derived + cross-check).

## Weil Bound

**Citation:** Weil 1948

**Statement:** |S(m,n;c)| <= tau(c) * sqrt(gcd(m,n,c)) * c^{1/2}

**Result:** Individual Kloosterman sum bound. Gives theta < 1/3 (weaker than DI).

**Implementation:** `lemmas/trivial_bounds.py`

## Trivial Bound

**Citation:** Trivial bound (absolute convergence)

**Statement:** Use absolute values; no cancellation.

**Implementation:** `lemmas/trivial_bounds.py`
