# De-identification Module

**Version 0.6.0** | `privacy/de-identification/`

> **DISCLAIMER: RESEARCH USE ONLY.** This module provides research-grade de-identification
> tooling. It does not guarantee regulatory compliance on its own. Independent validation by
> a qualified expert, organizational privacy officer review, and IRB approval are required
> before use with real patient data in any clinical or production setting.

## Overview

The De-identification module implements a production-grade pipeline for removing or
transforming protected health information (PHI) in clinical trial data before it enters
the federated learning system. It supports multiple transformation strategies compliant
with the **HIPAA Safe Harbor** method (45 CFR 164.514(b)(2)) and provides audit trails
suitable for **Expert Determination** review (45 CFR 164.514(a)).

The pipeline uses cryptographically secure pseudonymization (HMAC-SHA256 with random salt),
consistent patient-level date shifting, geographic generalization, and full field redaction.
Every transformation is recorded in an auditable `TransformationRecord` so that compliance
officers can verify that de-identification was applied correctly.

## File Structure

```
de-identification/
├── README.md                       # This file
└── deidentification_pipeline.py    # Multi-method de-identification pipeline
```

## Regulatory References

- **45 CFR 164.514(b)(2)** -- Safe Harbor de-identification (18 identifiers)
- **45 CFR 164.514(a)** -- Expert Determination method
- **NIST SP 800-188** -- De-Identifying Government Datasets
- **HHS Guidance on De-identification** (2012, updated 2022)

## API Overview

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `DeidentificationMethod` | `REDACT`, `HASH`, `GENERALIZE`, `SUPPRESS`, `PSEUDONYMIZE`, `DATE_SHIFT` | Transformation strategies |
| `TransformationStatus` | `SUCCESS`, `PARTIAL`, `FAILED`, `SKIPPED` | Outcome of a transformation |

### Data Classes

| Class | Key Fields | Purpose |
|-------|-----------|---------|
| `TransformationRecord` | `field_name`, `method`, `original_type`, `status`, `timestamp` | Audit record for a single field transformation (never stores original values) |
| `DeidentificationResult` | `record_id`, `original_phi_count`, `transformed_count`, `clean_data`, `transformations` | Result for a single record with cleaned data and transformation audit trail |
| `PipelineStatistics` | `records_processed`, `phi_detected`, `phi_transformed`, `phi_failed`, `methods_used` | Cumulative pipeline statistics |

### DeidentificationPipeline

The primary class for de-identifying clinical trial data.

```python
from privacy.de_identification.deidentification_pipeline import (
    DeidentificationPipeline,
    DeidentificationMethod,
)

pipeline = DeidentificationPipeline(
    default_method=DeidentificationMethod.REDACT,
    salt=None,                    # Auto-generates os.urandom(32) if None
    date_shift_range_days=365,    # Max days for date shifting
    field_methods={               # Per-field method overrides
        "dob": DeidentificationMethod.DATE_SHIFT,
        "mrn": DeidentificationMethod.PSEUDONYMIZE,
        "zip_code": DeidentificationMethod.GENERALIZE,
    },
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `deidentify_record` | `(record: dict, record_id: str) -> DeidentificationResult` | `DeidentificationResult` | De-identify a single structured record |
| `deidentify_text` | `(text: str, record_id: str) -> DeidentificationResult` | `DeidentificationResult` | De-identify free text |
| `deidentify_dataset` | `(records: list[dict]) -> list[DeidentificationResult]` | `list[DeidentificationResult]` | Batch de-identification with cumulative statistics |
| `get_statistics` | `() -> PipelineStatistics` | `PipelineStatistics` | Cumulative pipeline statistics |

### Transformation Methods

| Method | Description | Data Utility | Reversible |
|--------|-------------|-------------|------------|
| `REDACT` | Replace PHI with `[REDACTED]` | Lowest | No |
| `HASH` | One-way SHA-256 hash | Low | No |
| `PSEUDONYMIZE` | HMAC-SHA256 with cryptographic salt | Medium | With salt |
| `GENERALIZE` | Replace with category label (e.g., age bracket, state) | Medium | No |
| `SUPPRESS` | Remove the field entirely | None | No |
| `DATE_SHIFT` | Shift dates by a consistent per-patient random offset | High | With mapping |

## Usage Example

```python
from privacy.de_identification.deidentification_pipeline import (
    DeidentificationPipeline,
    DeidentificationMethod,
)

pipeline = DeidentificationPipeline(
    default_method=DeidentificationMethod.REDACT,
    field_methods={
        "dob": DeidentificationMethod.DATE_SHIFT,
        "mrn": DeidentificationMethod.PSEUDONYMIZE,
    },
)

record = {
    "patient_name": "John Doe",
    "mrn": "MRN#12345678",
    "dob": "1965-08-21",
    "tumor_volume_cm3": 4.2,
    "stage": "IIIB",
    "pdl1_expression": 0.72,
}

result = pipeline.deidentify_record(record, record_id="rec_001")
print(result.clean_data)
# {
#   "patient_name": "[REDACTED]",
#   "mrn": "[PSEUDO:a3f8...]",
#   "dob": "1965-05-07",             # date-shifted
#   "tumor_volume_cm3": 4.2,         # clinical data preserved
#   "stage": "IIIB",                 # clinical data preserved
#   "pdl1_expression": 0.72,         # clinical data preserved
# }

print(f"PHI detected: {result.original_phi_count}")
print(f"PHI transformed: {result.transformed_count}")
for t in result.transformations:
    print(f"  {t.field_name}: {t.method.value} -> {t.status.value}")
```

## Security Considerations

- **Salt management**: The HMAC-SHA256 pseudonymization salt is generated via `os.urandom(32)`
  and must be stored securely. Loss of the salt means pseudonymized IDs cannot be reversed.
- **Date-shift consistency**: Date shifts are deterministic per `patient_id`, preserving
  temporal relationships within a patient's records while preventing calendar re-identification.
- **No PHI in logs**: The `TransformationRecord` stores only the field name and original type
  category -- never the original value.

## License

MIT License. See [LICENSE](../../LICENSE) for details.
