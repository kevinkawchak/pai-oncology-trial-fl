# DUA Templates Generator Module

**Version 0.6.0** | `privacy/dua-templates-generator/`

> **DISCLAIMER: RESEARCH USE ONLY.** This module generates Data Use Agreement templates for
> research and development purposes. Generated documents are not legally binding and must
> be reviewed and approved by qualified legal counsel, IRB, and institutional officials
> before execution. Each institution's Office of Research Compliance should review all
> DUAs before signing.

## Overview

The DUA Templates Generator module produces structured Data Use Agreement documents for
oncology clinical trial data sharing in federated learning settings. It supports five
agreement types (intra-institutional, multi-site, commercial, academic collaboration,
government), with configurable security requirements, retention policies, and data category
classifications aligned with **45 CFR 164.514(e)** (limited data set and DUA requirements).

Each generated DUA includes party information, permitted data uses, security control
specifications, retention and destruction policies, breach notification obligations, and
amendment tracking -- providing a comprehensive starting point for legal review.

## File Structure

```
dua-templates-generator/
├── README.md              # This file
└── dua_generator.py       # DUA template generation engine
```

Related pre-built templates:

```
dua_templates/
├── standard_dua.md        # Standard single-site DUA template
└── multi_site_dua.md      # Multi-site federation DUA template
```

## Regulatory References

- **45 CFR 164.514(e)** -- Limited data set and DUA requirements
- **45 CFR 164.508** -- Authorization for uses and disclosures
- **NIH Data Sharing Policy** (2023)
- **HIPAA Privacy Rule** -- Minimum necessary standard

## API Overview

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `DUAType` | `INTRA_INSTITUTIONAL`, `MULTI_SITE`, `COMMERCIAL`, `ACADEMIC_COLLABORATION`, `GOVERNMENT` | Types of data use agreements |
| `DataCategory` | `LIMITED_DATA_SET`, `DE_IDENTIFIED`, `CODED`, `FULLY_IDENTIFIED`, `AGGREGATED` | Categories of data covered |
| `SecurityTier` | `STANDARD`, `ENHANCED`, `MAXIMUM` | Security requirement tiers |
| `DUAStatus` | `DRAFT`, `PENDING_REVIEW`, `UNDER_NEGOTIATION`, `APPROVED`, `ACTIVE`, `EXPIRED`, `TERMINATED` | Agreement lifecycle status |

### Data Classes

| Class | Key Fields | Purpose |
|-------|-----------|---------|
| `SecurityRequirements` | `encryption_at_rest`, `encryption_in_transit`, `encryption_standard`, `mfa_required`, `access_logging`, `data_loss_prevention`, `network_segmentation`, `incident_response_plan` | Security controls for data handling |
| `RetentionPolicy` | `retention_period_years`, `destruction_method`, `destruction_certification`, `extension_allowed`, `review_frequency_months` | Data retention and destruction rules |
| `DUAParty` | `organization_name`, `role`, `contact_name`, `contact_email`, `institution_type`, `irb_approval_number` | A party to the agreement |
| `DUADocument` | `dua_id`, `dua_type`, `status`, `parties`, `data_category`, `security_requirements`, `retention_policy`, `sections`, `amendments` | Complete generated DUA document |

### DUAGenerator

The primary class for generating Data Use Agreement documents.

```python
from privacy.dua_templates_generator.dua_generator import (
    DUAGenerator,
    DUAType,
    DataCategory,
)

generator = DUAGenerator()
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `generate` | `(dua_type, parties, data_category, purpose, ...) -> DUADocument` | `DUADocument` | Generate a complete DUA document |
| `generate_multi_site` | `(parties, study_id, ...) -> DUADocument` | `DUADocument` | Convenience method for multi-site federation DUAs |
| `render_markdown` | `(document: DUADocument) -> str` | `str` | Render a DUA document to Markdown |
| `update_status` | `(document, new_status) -> DUADocument` | `DUADocument` | Advance the DUA lifecycle status |
| `add_amendment` | `(document, amendment_text, author) -> DUADocument` | `DUADocument` | Add an amendment to an existing DUA |

### Default Security by Agreement Type

| DUA Type | Encryption | MFA | DLP | Network Segmentation | Annual Audit | Background Checks |
|----------|-----------|-----|-----|---------------------|-------------|-------------------|
| Intra-institutional | AES-256 (rest + transit) | Yes | No | No | No | No |
| Multi-site | AES-256 (rest + transit) | Yes | Yes | Yes | Yes | No |
| Commercial | AES-256 (rest + transit) | Yes | Yes | Yes | Yes | Yes |
| Academic Collaboration | AES-256 (rest + transit) | Yes | No | No | Yes | No |
| Government | AES-256 (rest + transit) | Yes | Yes | Yes | Yes | Yes |

### Default Retention by Agreement Type

| DUA Type | Retention Period | Destruction Method | Destruction Certification | Extension Allowed |
|----------|-----------------|-------------------|--------------------------|-------------------|
| Intra-institutional | 3 years | Cryptographic erasure | Yes | No |
| Multi-site | 5 years | Cryptographic erasure | Yes | Yes (with approval) |
| Commercial | 3 years | Cryptographic erasure | Yes | No |
| Academic Collaboration | 5 years | Cryptographic erasure | Yes | Yes (with approval) |
| Government | 7 years | Cryptographic erasure | Yes | Yes (with approval) |

## Usage Example

```python
from privacy.dua_templates_generator.dua_generator import (
    DUAGenerator,
    DUAType,
    DUAParty,
    DataCategory,
)

generator = DUAGenerator()

# Define parties
provider = DUAParty(
    organization_name="Memorial Cancer Center",
    role="provider",
    contact_name="Dr. Sarah Chen",
    contact_email="s.chen@memorial.example.org",
    institution_type="academic_medical_center",
    irb_approval_number="IRB-2026-0142",
)
recipient = DUAParty(
    organization_name="Regional Oncology Network",
    role="recipient",
    contact_name="Dr. James Park",
    contact_email="j.park@regional.example.org",
    institution_type="community_hospital",
    irb_approval_number="IRB-2026-0287",
)

# Generate multi-site DUA
document = generator.generate_multi_site(
    parties=[provider, recipient],
    study_id="ONCO-FL-2026-001",
    data_category=DataCategory.LIMITED_DATA_SET,
    purpose="Federated model training for NSCLC treatment response prediction",
)

# Render to markdown
markdown_text = generator.render_markdown(document)
print(f"DUA ID: {document.dua_id}")
print(f"Status: {document.status.value}")
print(f"Sections: {len(document.sections)}")

# Advance lifecycle
document = generator.update_status(document, "pending_review")
```

## Generated DUA Sections

A generated DUA document typically includes the following sections:

1. **Preamble** -- Agreement date, parties, and recitals
2. **Definitions** -- Key terms (PHI, limited data set, covered entity, etc.)
3. **Purpose and Scope** -- Permitted data uses and study description
4. **Data Description** -- Data elements, category, and volume
5. **Obligations of Data Provider** -- Delivery, quality, de-identification
6. **Obligations of Data Recipient** -- Use restrictions, safeguards, reporting
7. **Security Requirements** -- Technical and administrative safeguards
8. **Retention and Destruction** -- Retention period, destruction method, certification
9. **Breach Notification** -- Notification timelines and procedures
10. **Term and Termination** -- Effective dates, renewal, early termination
11. **Compliance and Audit** -- Right to audit, compliance monitoring
12. **Signatures** -- Authorized signatories for all parties

## License

MIT License. See [LICENSE](../../LICENSE) for details.
