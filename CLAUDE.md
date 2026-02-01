# CLAUDE.md — Non-negotiable rules for mollifier-theta

## DO NOT SIMPLIFY
Every transform must preserve full symbolic structure. No silent delta-function approximations, no dropping error terms, no premature bounding.

## Immutability
All IR objects (Term, Kernel, Phase, ScaleModel) are frozen Pydantic models. Transforms return new lists of new terms. Never mutate input.

## BoundOnly requires citation
A term may only have status=BoundOnly if it carries a non-empty lemma_citation field. This is enforced by invariants.py and tested.

## Phase tracking
Phases are never silently dropped. Every transform must account for all input phases in its outputs. Phase absorption is explicit and tested for norm preservation.

## Kernel preservation
Smooth kernels (W, Fourier, delta-method) survive all transforms. If a transform removes a kernel, it must document why and record in history.

## History chain
Every term records its full derivation history (parent term IDs + transform name). The chain must be traceable from any leaf back to the initial integral.

## SymPy containment
SymPy is used only in core/scale_model.py and reports/. Never import sympy in transforms or hot paths.

## Two-layer theta verification
The DI Kloosterman bound must produce theta_max via derived exponent algebra (Layer 1) AND cross-check against the known constant 4/7 (Layer 2). Disagreement is a build-breaking error.
