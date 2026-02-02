# Overhead Report Envelope v1.0

**Format version:** `1.0`
**Producer:** `mollifier_theta.analysis.overhead_report.OverheadReport.to_envelope`
**Pydantic source of truth:** `OverheadRecord` dataclass in same module.

## Envelope Structure

```json
{
  "format_version": "1.0",
  "theta_val": <float>,
  "record_count": <int>,
  "records": [ <OverheadRecord>, ... ]
}
```

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format_version` | `string` | Yes | Must be `"1.0"`. |
| `theta_val` | `float` | Yes | Theta value at which the overhead was evaluated. |
| `record_count` | `int` | Yes | Must equal `len(records)`. |
| `records` | `list[object]` | Yes | Sorted by `(bound_family, term_id)`. |

### Record Fields (`OverheadRecord`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `term_id` | `string` | Yes | Unique identifier of the BoundOnly term. |
| `bound_family` | `string` | Yes | Family of the bounding lemma. |
| `pipeline_exponent` | `string` | Yes | Error exponent currently used in the proof. |
| `pipeline_E_val` | `float` | Yes | `pipeline_exponent` evaluated at `theta_val`. |
| `raw_di_exponent` | `string` | Yes | Raw DI formula exponent string. |
| `raw_di_E_val` | `float` | Yes | Raw DI exponent evaluated at `theta_val`. |
| `overhead` | `float` | Yes | `pipeline_E_val - raw_di_E_val`. Positive means pipeline is worse. |
| `di_model_label` | `string` | Yes | `"symmetric"` or `"voronoi_dual"`. |
| `derivation_path` | `list[string]` | Yes | Transform chain from term history (may be empty). |

## Canonical Sort Key

Records are sorted by the tuple `(bound_family, term_id)` using Python's default string comparison.

## Forward-Compatibility Rules

Changes that **require** bumping `format_version`:

- Adding, removing, or renaming any top-level field.
- Adding, removing, or renaming any record field.
- Changing the type of any existing field.
- Changing the canonical sort key.
- Changing the semantics of `overhead` (e.g. sign convention).

Changes that **do not** require a version bump:

- Adding new `di_model_label` values (new DI model variants).
- Adding new `bound_family` values.
- Changes to which terms appear in the report (pipeline changes).
