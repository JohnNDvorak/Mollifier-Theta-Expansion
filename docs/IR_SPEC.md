# IR Specification

## Term
Central IR node. Frozen Pydantic model.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique identifier (auto-generated) |
| kind | TermKind | Structural classification |
| expression | str | Symbolic expression description |
| variables | list[str] | Active summation/integration variables |
| ranges | list[Range] | Variable ranges |
| kernels | list[Kernel] | Attached smooth kernels |
| phases | list[Phase] | Phase factors |
| scale_model | str | ScaleModel key (serialized separately) |
| history | list[HistoryEntry] | Full derivation chain |
| status | TermStatus | Active / MainTerm / BoundOnly / Error |
| parents | list[str] | Parent term IDs |
| lemma_citation | str | Required for BoundOnly |
| multiplicity | int | Combinatorial multiplicity |
| metadata | dict | Extensible metadata |

## TermStatus
- **Active**: Under transformation, not yet classified
- **MainTerm**: Contributing to the main asymptotic
- **BoundOnly**: Bounded by a lemma (citation required)
- **Error**: Negligible error term

## TermKind
Integral, DirichletSum, Cross, Diagonal, OffDiagonal, Kloosterman, Error

## Kernel
Smooth weight function. Fields: name, support, properties, argument, description.
Methods: mellin_transform(), residue_structure().

## Phase
Oscillatory factor. Fields: expression, is_separable, absorbed, depends_on.

## Range
Variable range. Fields: variable, lower, upper, description.

## ScaleModel
T-exponent tracker (in core/scale_model.py, uses SymPy).
- T_exponent: symbolic expression in theta
- log_power: integer
- Arithmetic: product (add exponents), sum (max exponent)
- evaluate(theta_val) -> float
