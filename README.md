# mollifier-theta

Reproduction framework for the Conrey/Levinson mollified zeta second-moment pipeline through the theta < 4/7 barrier.

This is a symbolic bookkeeping system that tracks term families, kernels, phases, and exponent scales through the chain of analytic reductions. It is **not** a numerical solver.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Full Conrey89 reproduction
mollifier repro conrey89 --theta 0.56

# Theta sweep
mollifier theta-sweep conrey89 --theta-min 0.45 --theta-max 0.65 --step 0.005

# Export diagonal main term to Mathematica
mollifier export mathematica diagonal-main-term
```

## Test

```bash
make test
```

## Architecture

- **IR objects** (Pydantic v2): Term, Kernel, Phase, Range, ScaleModel
- **Transforms**: Immutable `list[Term] -> list[Term]` functions
- **Lemmas**: Bound application with mandatory citation tracking
- **Invariants**: No premature bounds, phase tracking, kernel preservation
