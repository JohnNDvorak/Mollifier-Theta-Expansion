# DEV NOTES — DO NOT SIMPLIFY

## Hard Invariant Rules

### 1. No premature delta approximation
The IntegrateOverT transform produces a Fourier kernel `K(log(am/bn))`. This kernel is **not** approximated as `delta(am - bn)`. The diagonal/off-diagonal split is a structural decomposition, not a kernel evaluation.

### 2. No silent error dropping
Error terms from the approximate functional equation carry explicit `status=Error` and `scale_model` with T^{-A}. They are never silently discarded.

### 3. No implicit phase cancellation
When opening the square |M*zeta|^2, conjugation phases are tracked explicitly on each cross-term. Phase absorption is a separate, tested transform.

### 4. No premature bounding
A term receives `status=BoundOnly` only when a specific lemma is applied, with the lemma name recorded in `lemma_citation`. The invariant `check_no_premature_bound()` enforces this globally.

### 5. Kernels are first-class objects
The approximate functional equation kernel W, the Fourier kernel from t-integration, and the smooth kernel from the delta method are all tracked as `Kernel` objects on every term. They are never "absorbed into the sum" implicitly.

### 6. Scale model arithmetic is symbolic
Exponents in theta are SymPy expressions. Products add exponents, sums take max. Numerical evaluation happens only at the final theta-check step.

### 7. Two-layer theta verification
The critical theta=4/7 result must be derived from the exponent algebra (Layer 1) and independently cross-checked against a hard-coded known constant (Layer 2). If these disagree, the build fails.
