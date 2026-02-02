# mollifier-theta

A formal symbolic bookkeeping framework that reproduces the Conrey/Levinson mollified second moment of the Riemann zeta function through the **theta < 4/7 barrier**.

This is not a numerical solver. It is a proof pipeline: an immutable intermediate representation tracks every term, kernel, phase, and exponent through each analytic reduction step, from the initial integral to the final DI Kloosterman bound, preserving the full derivation chain so that the theta_max = 4/7 result is machine-verifiable against the known constant from Conrey 1989.

**904 tests. Zero silent simplifications. Every bound carries a citation.**

---

## The Mathematics

The **mollifier theta problem** asks: for a Dirichlet polynomial mollifier of length T^theta, what is the largest theta for which the mollified second moment of zeta(1/2+it) can be asymptotically evaluated?

The answer, established by Conrey (1989) building on Deshouillers-Iwaniec (1982/83), is **theta < 4/7**. The proof proceeds through:

1. Approximate functional equation for zeta
2. Opening the square |M(s)zeta(s)|^2
3. Integration over t (extracting Fourier kernel structure)
4. Diagonal / off-diagonal split
5. Extraction of the main term from the diagonal
6. Delta method on off-diagonal terms
7. Formation of Kloosterman sums S(m,n;c)/c
8. Application of the Deshouillers-Iwaniec bilinear bound
9. Exponent algebra: E(theta) = 7*theta/4 < 1 gives theta < 4/7

This framework tracks every step symbolically, enforcing that no structure is lost, no phases are silently dropped, and no bounds are applied without literature citations.

---

## Install

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10. Dependencies: Pydantic v2, SymPy, Typer, Rich.

## Quick Start

```bash
# Reproduce Conrey89 at theta = 0.56 (should PASS)
mollifier repro conrey89 --theta 0.56

# Reproduce at theta = 0.58 (should FAIL — above 4/7)
mollifier repro conrey89 --theta 0.58

# Find the exact boundary via grid sweep
mollifier theta-sweep conrey89 --theta-min 0.45 --theta-max 0.65 --step 0.005

# Run the Voronoi summation variant
mollifier repro conrey89-voronoi --theta 0.56 --K 3

# Identify the binding constraint
mollifier diagnose slack --theta 0.56 --K 3

# Explore hypothetical improvements
mollifier diagnose what-if di_saving "-theta/3" --theta 0.56

# Compare baseline vs Voronoi pipeline
mollifier diagnose compare --theta 0.56 --K 3

# Theta-loss decomposition (pipeline vs raw DI overhead)
mollifier diagnose overhead --theta 0.56 --K 3 --pipeline conrey89 --json

# Export proof certificate (JSON + Markdown)
mollifier export proof-cert --theta 0.56 --K 3 --pipeline conrey89

# Export math parameters envelope
mollifier export math-params --theta 0.56 --K 3 --pipeline conrey89 --json

# Export diagonal main term to Mathematica
mollifier export mathematica diagonal-main-term
```

## Test

```bash
python -m pytest                  # All 904 tests
python -m pytest -x --tb=short    # Stop on first failure
python -m pytest -k "conrey89"    # Pipeline tests only
python -m pytest -k "golden"      # Golden regression fixtures
```

---

## Architecture

### Immutable IR

Every analytic object is a frozen Pydantic v2 model. Transforms never mutate input; they return new term lists. The full derivation history is preserved in every term.

```
Term
 ├── id           (unique, auto-generated)
 ├── kind         (INTEGRAL | DIRICHLET_SUM | CROSS | DIAGONAL | OFF_DIAGONAL | KLOOSTERMAN | SPECTRAL | ERROR)
 ├── status       (ACTIVE | MAIN_TERM | BOUND_ONLY | ERROR)
 ├── expression   (symbolic string)
 ├── variables    (["m", "n", "c", ...])
 ├── ranges       ([Range(variable, lower, upper)])
 ├── kernels      ([Kernel(name, support, properties)])
 ├── phases       ([Phase(expression, is_separable, depends_on)])
 ├── scale_model  (T-exponent as symbolic string in theta)
 ├── history      ([HistoryEntry(transform, parent_ids)])
 ├── metadata     (typed stage metadata: _bound, _voronoi, _kloosterman, _kuznetsov)
 └── kernel_state (NONE → UNCOLLAPSED_DELTA → VORONOI_APPLIED/COLLAPSED → KLOOSTERMANIZED → SPECTRALIZED)
```

### Transform Pipeline

Each transform is a pure function `list[Term] -> list[Term]`:

| Transform | What it does |
|-----------|-------------|
| `ApproxFE` | Approximate functional equation: integral to Dirichlet sums + error |
| `OpenSquare` | Open \|M*zeta\|^2 into K(K+1)/2 cross-term families |
| `IntegrateOverT` | Evaluate t-integral, extract Fourier kernel structure |
| `DiagonalSplit` | Structural split: am=bn (diagonal) vs am!=bn (off-diagonal) |
| `DiagonalExtract` | Extract polynomial main term T*P(theta)*(log T)^k |
| `DeltaMethod` | Apply Duke-Friedlander-Iwaniec delta method |
| `KloostermanForm` | Form Kloosterman sums S(m,n;c)/c |
| `PhaseAbsorb` | Explicit phase cancellation with norm tracking |
| `Voronoi` | Voronoi summation formula (671 lines; dual length tracking) |
| `Kuznetsov` | Kuznetsov trace formula (spectral variant) |

### Lemma Library

Every bound application requires a literature citation. This is enforced by a model validator on the `Term` class — a `BOUND_ONLY` term without `lemma_citation` is a build error.

| Bound | Citation | Result |
|-------|----------|--------|
| DI Bilinear Kloosterman | Deshouillers-Iwaniec 1982/83, Theorem 12 | E(theta) = 7*theta/4 |
| Conrey 7theta/4 | Conrey 1989, Section 4 | theta_max = 4/7 |
| Length-aware DI | DI Thm 12 (parametric) | E(alpha, beta, gamma) = max(sub_A, sub_B)/2 |
| Voronoi dual DI | DI Thm 12 + Voronoi | Dual length N* = T^{2-3*theta} |
| Weil bound | Weil 1948 | Weaker theta < 1/3 |
| Spectral large sieve | Spectral theory | Alternative spectral bound |

### Two-Layer Theta Verification

The theta_max = 4/7 result is verified through two independent mechanisms:

- **Layer 1**: Derived exponent algebra. The DI Kloosterman constraint `E(theta) = 7*theta/4` is solved symbolically via SymPy, yielding `theta_max = 4/7`.
- **Layer 2**: Cross-check against the known constant. The derived value is compared to the hardcoded `4/7` from Conrey 1989.

Disagreement between layers is a build-breaking error. This is not a unit test — it is a structural invariant.

### SymPy Containment

SymPy is imported in exactly two places: `core/scale_model.py` and `reports/`. All symbolic operations — expression parsing, evaluation, root-finding, simplification — go through `ScaleModel` class methods. Transforms and hot paths never touch SymPy directly. This is enforced by convention and tested.

---

## Project Layout

```
mollifier_theta/
    core/               Immutable IR, invariants, scale model, phase algebra
    transforms/         Pure term-to-term reductions (10 transforms)
    lemmas/             Bounds with mandatory citations (6 lemma types)
    pipelines/          End-to-end orchestration (3 pipeline variants + sweep + strict runner)
    analysis/           Post-pipeline diagnostics (slack, overhead, what-if, theta breakdown)
    reports/            Export: proof certificates, Mathematica, math params, envelope loader
    cli.py              Typer CLI with repro/export/diagnose sub-commands

tests/                  904 tests across 53 files
    fixtures/           Golden JSON regression fixtures

docs/
    IR_SPEC.md          IR object specification
    REPRO_CONREY89.md   Conrey89 reproduction details
    LEMMAS.md           Lemma catalog
    PLAN.md             Project milestones
    schemas/            v1.0 envelope contract documentation
```

### Versioned JSON Envelopes

Pipeline outputs are exported as versioned JSON envelopes with format contracts:

- **Math Parameters Envelope** (`v1.0`): Per-term bound family, error exponent, sum length exponents, citations. Sorted by `(bound_family, case_id, term_id)`.
- **Overhead Report Envelope** (`v1.0`): Per-term pipeline vs raw DI exponent comparison. Sorted by `(bound_family, term_id)`.
- **Theta Breakdown Envelope** (`v1.0`): Per-term binding analysis, slack, overhead. Sorted by slack ascending.

The envelope loader (`reports/envelope_loader.py`) enforces strict validation on import: version check, record count, required fields, and canonical sort order.

Golden regression fixtures in `tests/fixtures/` ensure byte-identical output stability.

---

## Non-Negotiable Rules

These are enforced by invariants, validators, and tests. See `CLAUDE.md`.

1. **Do not simplify.** Every transform preserves full symbolic structure. No silent delta-function approximations, no dropping error terms, no premature bounding.
2. **Immutability.** All IR objects are frozen Pydantic models. Transforms return new terms. Never mutate input.
3. **BoundOnly requires citation.** Enforced by model validator. Untested bounds cannot enter the proof.
4. **Phase tracking.** Phases are never silently dropped. Absorption is explicit and norm-preserving.
5. **Kernel preservation.** Smooth kernels survive all transforms. Removal requires documented justification.
6. **History chain.** Every term records its full derivation. Traceable from any leaf to the initial integral.
7. **SymPy containment.** SymPy only in `core/scale_model.py` and `reports/`. Never in transforms.
8. **Two-layer theta verification.** Derived algebra must agree with known constant. Disagreement breaks the build.

---

## License

This project is for research and educational purposes.
