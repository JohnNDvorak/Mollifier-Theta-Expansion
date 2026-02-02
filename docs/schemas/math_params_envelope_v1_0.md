# Math Parameters Envelope v1.0

**Format version:** `1.0`
**Producer:** `mollifier_theta.reports.math_parameter_export.export_math_parameters_envelope`
**Pydantic source of truth:** `MathParameterRecord` dataclass in same module.

## Envelope Structure

```json
{
  "format_version": "1.0",
  "record_count": <int>,
  "records": [ <MathParameterRecord>, ... ]
}
```

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format_version` | `string` | Yes | Must be `"1.0"`. |
| `record_count` | `int` | Yes | Must equal `len(records)`. |
| `records` | `list[object]` | Yes | Sorted by `(bound_family, case_id, term_id)`. |

### Record Fields (`MathParameterRecord`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `term_id` | `string` | Yes | Unique identifier of the BoundOnly term (UUID hex prefix). |
| `bound_family` | `string` | Yes | Family of the bounding lemma (e.g. `"DI_Kloosterman"`). |
| `case_id` | `string` | Yes | Case identifier within the family. |
| `error_exponent` | `string` | Yes | Symbolic T-exponent expression in `theta` (e.g. `"7*theta/4"`). |
| `m_length_exponent` | `string` | Yes | T-exponent of first sum length M. Default `"theta"`. |
| `n_length_exponent` | `string` | Yes | T-exponent of second sum length N. Changes for Voronoi dual terms. |
| `modulus_exponent` | `string` | Yes | T-exponent of modulus range C. Default `"1-theta"`. |
| `kernel_family_tags` | `list[string]` | Yes | Bessel kernel family classifications (may be empty). |
| `citation` | `string` | Yes | Literature reference for the bound. |

## Canonical Sort Key

Records are sorted by the tuple `(bound_family, case_id, term_id)` using Python's default string comparison. This ensures deterministic, byte-identical output for the same input terms.

## Forward-Compatibility Rules

Changes that **require** bumping `format_version`:

- Adding, removing, or renaming any top-level field.
- Adding, removing, or renaming any record field.
- Changing the type of any existing field.
- Changing the canonical sort key.

Changes that **do not** require a version bump:

- Adding new values to an enum-like string field (e.g. new `bound_family` values).
- Adding new entries to `kernel_family_tags`.
- Changes to the set of terms that produce records (pipeline changes).
