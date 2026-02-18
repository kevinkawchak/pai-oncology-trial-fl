# Privacy Framework

**Version 0.6.0** | PAI Oncology Trial FL

> **DISCLAIMER: RESEARCH USE ONLY.** This privacy framework provides research-grade tooling
> for engineers building HIPAA-compliant federated learning systems. It is not a substitute for
> legal counsel, IRB review, or organizational privacy officer oversight. Independent validation
> and regulatory clearance are required before deployment in any clinical setting.

## Overview

The `privacy/` module provides end-to-end privacy infrastructure for multi-site oncology
clinical trials operating under federated learning. It enforces data minimization, automated
PHI detection and removal, role-based access control, breach response workflows, consent
management, audit logging, and Data Use Agreement generation -- all aligned with U.S. and
international regulatory requirements.

The framework implements both the **Safe Harbor** method (45 CFR 164.514(b)(2)) and the
**Expert Determination** method (45 CFR 164.514(a)) for de-identification, ensuring that
protected health information never leaves a site boundary in identifiable form.

## Directory Structure

```
privacy/
├── README.md                          # This file — framework overview
├── __init__.py                        # Public API surface
├── phi_detector.py                    # Core PHI detection (18 HIPAA IDs)
├── deidentification.py                # Core de-identification pipeline
├── consent_manager.py                 # Consent & DUA compliance tracking
├── audit_logger.py                    # Audit logging with SHA-256 integrity
├── access_control.py                  # Core role-based access control
├── breach_response.py                 # Core breach detection & response
├── phi-pii-management/               # Extended PHI/PII detection engine
│   ├── README.md                      #   Module documentation
│   └── phi_detector.py               #   Advanced PHI scanner with DICOM/Presidio
├── de-identification/                 # Extended de-identification pipeline
│   ├── README.md                      #   Module documentation
│   └── deidentification_pipeline.py  #   Multi-method pipeline with audit trails
├── access-control/                    # Extended RBAC module
│   ├── README.md                      #   Module documentation
│   └── access_control_manager.py     #   21 CFR Part 11 compliant RBAC
├── breach-response/                   # Extended breach response module
│   ├── README.md                      #   Module documentation
│   └── breach_response_protocol.py   #   NIST SP 800-61r2 incident response
├── dua-templates-generator/           # DUA document generator
│   ├── README.md                      #   Module documentation
│   └── dua_generator.py             #   Multi-type DUA generation engine
└── dua_templates/                     # Pre-built DUA templates
    ├── standard_dua.md               #   Standard single-site DUA
    └── multi_site_dua.md             #   Multi-site federation DUA
```

## HIPAA Alignment

### Safe Harbor Method -- 45 CFR 164.514(b)(2)

The Safe Harbor method requires removal of the following **18 categories of identifiers**
from protected health information. The `PHIDetector` and `PHICategory` enum cover all 18:

| # | Identifier | PHICategory Enum | Risk Level |
|---|-----------|-----------------|------------|
| 1 | Names | `NAMES` | High |
| 2 | Geographic data smaller than state | `GEOGRAPHIC` | Medium |
| 3 | Dates (except year) related to an individual | `DATES` | Medium |
| 4 | Telephone numbers | `PHONE_NUMBERS` | High |
| 5 | Fax numbers | `FAX_NUMBERS` | High |
| 6 | Email addresses | `EMAIL_ADDRESSES` | High |
| 7 | Social Security numbers | `SSN` | Critical |
| 8 | Medical record numbers | `MRN` | Critical |
| 9 | Health plan beneficiary numbers | `HEALTH_PLAN_ID` | Critical |
| 10 | Account numbers | `ACCOUNT_NUMBERS` | High |
| 11 | Certificate/license numbers | `CERTIFICATE_LICENSE` | High |
| 12 | Vehicle identifiers and serial numbers | `VEHICLE_ID` | Medium |
| 13 | Device identifiers and serial numbers | `DEVICE_ID` | Medium |
| 14 | Web Universal Resource Locators (URLs) | `URLS` | Medium |
| 15 | Internet Protocol (IP) addresses | `IP_ADDRESSES` | Medium |
| 16 | Biometric identifiers | `BIOMETRIC_ID` | Critical |
| 17 | Full-face photographs and comparable images | `PHOTOGRAPHS` | High |
| 18 | Any other unique identifying number, characteristic, or code | `OTHER_UNIQUE_ID` | Medium |

The framework additionally requires that the covered entity has **no actual knowledge** that
the remaining information could be used alone or in combination to identify an individual
(45 CFR 164.514(b)(2)(ii)).

### Expert Determination Method -- 45 CFR 164.514(a)

The `DeidentificationPipeline` supports Expert Determination workflows where a qualified
statistical expert certifies that the risk of re-identification is "very small." The pipeline
produces transformation audit trails suitable for expert review.

## Regulatory Cross-References

The privacy framework is designed with alignment to the following regulations and standards:

### HIPAA (Health Insurance Portability and Accountability Act)

| Regulation | Scope | Module |
|-----------|-------|--------|
| 45 CFR 164.514(b)(2) | Safe Harbor de-identification (18 identifiers) | `phi-pii-management/`, `de-identification/` |
| 45 CFR 164.514(a) | Expert Determination method | `de-identification/` |
| 45 CFR 164.514(e) | Limited data set and DUA requirements | `dua-templates-generator/` |
| 45 CFR 164.312 | Security Rule -- access controls | `access-control/` |
| 45 CFR 164.312(b) | Audit controls | `audit_logger.py`, `access-control/` |
| 45 CFR 164.400-414 | Breach Notification Rule | `breach-response/` |
| 45 CFR 164.402 | Four-factor risk assessment | `breach-response/` |
| 45 CFR 164.508 | Authorization for uses and disclosures | `consent_manager.py` |
| 45 CFR 164.512 | Uses and disclosures for research | `phi-pii-management/` |

### 21 CFR Part 11 (Electronic Records; Electronic Signatures)

| Requirement | Implementation |
|------------|---------------|
| Audit trails (11.10(e)) | `AuditLogger` with SHA-256 integrity hashing; `AuditEntry` in access-control |
| Access controls (11.10(d)) | `AccessControlManager` with role-based permissions |
| Electronic signatures (11.50-11.100) | Signature fields in `DUADocument` lifecycle |
| Record integrity (11.10(c)) | SHA-256 tamper-detection on all audit events |

### GDPR (General Data Protection Regulation)

| Article | Scope | Module |
|---------|-------|--------|
| Art. 5(1)(c) | Data minimization | `de-identification/` (suppress, generalize) |
| Art. 6 | Lawful basis for processing | `consent_manager.py` |
| Art. 17 | Right to erasure | `consent_manager.py` (revoke_consent) |
| Art. 25 | Data protection by design and default | All modules |
| Art. 33 | Breach notification to supervisory authority (72 hrs) | `breach-response/` |
| Art. 34 | Communication to data subjects | `breach-response/` |
| Art. 35 | Data Protection Impact Assessment | Pipeline statistics and audit trails |

### EU AI Act (Regulation 2024/1689)

| Requirement | Implementation |
|------------|---------------|
| Art. 9 -- Risk management | Risk-level classification per PHI category |
| Art. 10 -- Data governance | De-identification pipeline with transformation audit trails |
| Art. 12 -- Record-keeping | `AuditLogger` immutable event log |
| Art. 13 -- Transparency | Scan results with per-finding confidence and method |
| Art. 14 -- Human oversight | Role-based access with PI/Safety Officer escalation |

### NIST Privacy Framework

| Function | Category | Module |
|----------|----------|--------|
| Identify-P | Inventory and Mapping (ID.IM-P) | `PHIDetector` scans across datasets |
| Govern-P | Risk Management (GV.RM-P) | Risk-level classification, four-factor assessment |
| Control-P | Data Processing Management (CT.DM-P) | De-identification pipeline, consent management |
| Communicate-P | Communication Practices (CM.AW-P) | Breach notification workflow |
| Protect-P | Data Protection (PR.DS-P) | RBAC, encryption requirements in DUAs |

### ICH E6(R3) -- Good Clinical Practice

| Principle | Implementation |
|-----------|---------------|
| 2.11 -- Confidentiality of records | PHI detection and de-identification before FL pipeline |
| 5.5 -- Data handling and record keeping | Audit trails with integrity hashing |
| 5.15 -- Record access | Role-based access control with minimum necessary principle |
| 5.18.6 -- Monitoring and audit | `AuditLogger.generate_report()`, compliance reports |
| Annex 2 -- Computerized systems | 21 CFR Part 11 aligned access control and audit |

## Quick Start

```python
from privacy import PHIDetector, Deidentifier, AccessControlManager, AuditLogger

# 1. Detect PHI in patient data
detector = PHIDetector()
matches = detector.scan_text("Patient John Doe, SSN 123-45-6789, MRN#12345678")
print(f"Found {len(matches)} PHI elements")

# 2. De-identify a record
deid = Deidentifier(method="redact")
result = deid.deidentify_record({
    "patient_name": "Jane Smith",
    "dob": "1985-03-15",
    "tumor_volume": 3.5,
})
print(result.clean_data)  # PHI fields replaced with [REDACTED]

# 3. Check access permissions
acm = AccessControlManager()
acm.register_user("u001", "Dr. Smith", role="coordinator")
allowed = acm.check_permission("u001", "start_federation")
print(f"Access granted: {allowed}")

# 4. Audit logging
logger = AuditLogger()
from privacy.audit_logger import EventType
logger.log(EventType.DATA_ACCESS, actor="u001", resource="site_01_data", action="training")
```

## Architecture Principles

1. **Defense in depth** -- PHI detection, de-identification, RBAC, and audit logging operate
   as independent layers. A failure in one layer does not compromise the others.

2. **Minimum necessary** -- The RBAC system enforces the HIPAA minimum necessary standard:
   each role receives only the permissions required for its function.

3. **Audit everything** -- Every access decision, consent change, de-identification operation,
   and breach indicator is recorded in tamper-evident audit logs.

4. **Data never leaves the site in identifiable form** -- De-identification is enforced before
   data enters the federated learning pipeline. Only model parameters and aggregated
   statistics cross site boundaries.

5. **Fail closed** -- Unknown users, expired consents, and unrecognized permissions default
   to denial.

## License

MIT License. See [LICENSE](../LICENSE) for details.
