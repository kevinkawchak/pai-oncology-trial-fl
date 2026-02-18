# PHI/PII Management Module

**Version 0.6.0** | `privacy/phi-pii-management/`

> **DISCLAIMER: RESEARCH USE ONLY.** This module provides research-grade PHI/PII detection
> tooling. It is not a substitute for expert review, organizational privacy officer
> oversight, or validated clinical NLP systems. Independent validation is required
> before deployment in any clinical or production setting.

## Overview

The PHI/PII Management module implements comprehensive detection of all 18 HIPAA Safe Harbor
identifiers per **45 CFR 164.514(b)(2)**. It is designed as a pre-ingestion screening gate
for clinical trial data before it enters the federated learning pipeline, ensuring that
protected health information never crosses site boundaries in identifiable form.

The module provides regex-based pattern matching, structured field-name heuristics, optional
DICOM tag scanning, and an integration point for Microsoft Presidio NER-based detection.
Each finding is classified by HIPAA category, risk level, detection confidence, and detection
method.

## File Structure

```
phi-pii-management/
├── README.md           # This file
└── phi_detector.py     # Advanced PHI/PII detection engine
```

## Regulatory References

- **45 CFR 164.514(b)(2)** -- Safe Harbor de-identification standard (18 identifiers)
- **45 CFR 164.512** -- Uses and disclosures for research purposes
- **NIST SP 800-188** -- De-Identifying Government Datasets
- **HIPAA Privacy Rule** (45 CFR Parts 160, 164)

## API Overview

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `PHICategory` | `NAMES`, `GEOGRAPHIC`, `DATES`, `PHONE_NUMBERS`, `FAX_NUMBERS`, `EMAIL_ADDRESSES`, `SSN`, `MRN`, `HEALTH_PLAN_ID`, `ACCOUNT_NUMBERS`, `CERTIFICATE_LICENSE`, `VEHICLE_ID`, `DEVICE_ID`, `URLS`, `IP_ADDRESSES`, `BIOMETRIC_ID`, `PHOTOGRAPHS`, `OTHER_UNIQUE_ID` | All 18 HIPAA Safe Harbor identifiers |
| `RiskLevel` | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Risk classification for disclosed PHI |
| `DetectionMethod` | `REGEX`, `FIELD_NAME`, `DICOM_TAG`, `PRESIDIO`, `HEURISTIC` | How the PHI was detected |

### Data Classes

| Class | Key Fields | Purpose |
|-------|-----------|---------|
| `PHIFinding` | `phi_type`, `location`, `value_preview`, `confidence`, `risk_level`, `detection_method` | A single detected PHI element (value is always redacted in preview) |
| `ScanResult` | `total_items_scanned`, `total_findings`, `findings`, `risk_summary`, `category_summary` | Aggregated scan results with risk and category breakdowns |

### PHIDetectorAdvanced

The primary class for PHI/PII detection.

```python
from privacy.phi_pii_management.phi_detector import PHIDetectorAdvanced, ScanResult

detector = PHIDetectorAdvanced(
    custom_patterns=None,       # Optional additional regex patterns
    enable_dicom_scan=False,    # Enable DICOM tag scanning
    confidence_threshold=0.6,   # Minimum confidence to report a finding
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `scan_text` | `(text: str) -> ScanResult` | `ScanResult` | Scan free text for PHI patterns |
| `scan_record` | `(record: dict) -> ScanResult` | `ScanResult` | Scan a structured data record (field names + values) |
| `scan_dataset` | `(records: list[dict]) -> ScanResult` | `ScanResult` | Scan a batch of records |
| `scan_dicom_tags` | `(tags: dict) -> ScanResult` | `ScanResult` | Scan DICOM metadata tags for PHI |
| `has_phi` | `(text: str) -> bool` | `bool` | Quick boolean check for any PHI |
| `get_risk_summary` | `(result: ScanResult) -> dict` | `dict` | Aggregate risk-level counts |

### Risk Classification

Each PHI category is assigned a default risk level based on re-identification potential:

- **Critical**: SSN, MRN, Health Plan ID, Biometric ID
- **High**: Names, Email, Phone, Fax, Account Numbers, Certificate/License, Photographs
- **Medium**: Dates, Geographic, IP Addresses, URLs, Vehicle ID, Device ID, Other Unique ID

## Usage Example

```python
from privacy.phi_pii_management.phi_detector import PHIDetectorAdvanced

detector = PHIDetectorAdvanced(confidence_threshold=0.7)

# Scan free text
result = detector.scan_text(
    "Patient John Doe (SSN 123-45-6789) seen at 123 Main St. "
    "Contact: john.doe@example.com, phone (555) 123-4567. MRN#12345678."
)
print(f"Findings: {result.total_findings}")
print(f"Risk summary: {result.risk_summary}")
# Risk summary: {'critical': 2, 'high': 3, 'medium': 0}

for finding in result.findings:
    print(f"  [{finding.risk_level.value}] {finding.phi_type.value}: "
          f"{finding.value_preview} (confidence={finding.confidence:.2f})")

# Scan structured record
record_result = detector.scan_record({
    "patient_name": "Jane Smith",
    "dob": "1985-03-15",
    "mrn": "MRN#87654321",
    "tumor_type": "NSCLC",        # Not PHI
    "stage": "IIIA",              # Not PHI
})
print(f"Record PHI count: {record_result.total_findings}")
```

## Integration with Federated Learning Pipeline

The PHI detector is intended to run as a **pre-ingestion gate**:

1. Raw clinical data arrives at a site node.
2. `PHIDetectorAdvanced.scan_dataset()` scans all records.
3. If `ScanResult.has_critical` or `ScanResult.has_high` is `True`, the data is
   blocked from entering the FL pipeline until de-identification is applied.
4. The scan result is logged via `AuditLogger` for compliance audit trails.

## License

MIT License. See [LICENSE](../../LICENSE) for details.
