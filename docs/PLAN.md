# Plan — Mollifier Theta Expansion Framework

## Milestones

### M0 — Scaffolding
Repo structure, pyproject.toml, Makefile, CLI skeleton.

### M1 — Term Ledger + IR
Core IR types (Term, Kernel, Phase, ScaleModel), TermLedger, invariant checks, serialization.

### M2 — Core Transforms
ApproxFunctionalEq, OpenSquare, IntegrateOverT.

### M3 — Diagonal/Off-diagonal Split
Split terms by diagonal (am=bn) vs off-diagonal (am≠bn).

### M4 — Diagonal Main Term Extraction
Extract polynomial main term from diagonal, Mathematica export.

### M5 — Off-diagonal Reduction
DeltaMethod, KloostermanForm, PhaseAbsorb transforms.

### M6 — Lemma Library + Pipeline + Reports
DI Kloosterman bound, theta constraints, full Conrey89 pipeline, theta sweep, reports.

## Acceptance Criteria
- theta=0.56 passes pipeline
- theta=0.58 fails pipeline
- theta_max = 4/7 derived symbolically and cross-checked
- ~155 tests passing
