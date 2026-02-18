# Regulatory Compliance Framework

[![Version](https://img.shields.io/badge/version-0.6.0-blue.svg)](../CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)

> **DISCLAIMER: RESEARCH USE ONLY** — This module provides research-grade regulatory tooling
> for federated learning oncology trials. It is not a substitute for qualified regulatory
> counsel. Independent validation, legal review, and regulatory authority engagement are
> required before any clinical deployment.

## Overview

The `regulatory/` package provides end-to-end compliance infrastructure for AI/ML-enabled
oncology clinical trials operating under federated learning architectures. It addresses the
unique regulatory challenges that arise when patient data remains distributed across
institutional boundaries while model parameters traverse jurisdictional lines.

The framework tracks five major health authority jurisdictions, implements ICH E6(R3) GCP
principles (published September 2025), manages IRB protocol lifecycles, and enforces
human oversight boundaries for AI-assisted clinical functions.

## Regulatory Landscape Context

As of early 2026, the FDA has authorized over 1,000 AI/ML-enabled medical devices,
with radiology and oncology representing the largest share. The regulatory environment
is shaped by several converging developments:

- **FDA PCCP Guidance (August 2025)** finalized predetermined change control plans for
  ML-enabled device software, establishing how manufacturers may implement algorithm
  updates without new submissions.
- **FDA QMSR (21 CFR Part 820)** effective February 2026, harmonizing US quality system
  requirements with ISO 13485:2016.
- **ICH E6(R3)** published September 2025, introducing a risk-proportionate approach to
  GCP and explicit provisions for technology-enabled clinical trials.
- **EU AI Act (Regulation 2024/1689)** classifying medical AI systems as high-risk,
  with conformity assessment obligations phasing in through 2026.

Federated learning systems add complexity because they process data across sites without
centralizing it, creating questions about which jurisdiction's rules govern the model
aggregation step, how audit trails span institutional boundaries, and how adverse event
detection responsibilities are allocated.

## Directory Structure

```
regulatory/
├── README.md                          # This file
├── __init__.py                        # Package exports
├── compliance_checker.py              # HIPAA/GDPR/FDA compliance validation
├── fda_submission.py                  # FDA submission lifecycle (510k/De Novo/PMA)
├── templates/                         # Document templates
│   ├── informed_consent.md            #   Informed consent template
│   └── irb_protocol.md               #   IRB protocol template
├── fda-compliance/                    # FDA device submission tracker
│   └── fda_submission_tracker.py      #   Full pathway lifecycle management
├── irb-management/                    # IRB protocol lifecycle
│   └── irb_protocol_manager.py        #   Multi-site protocol coordination
├── ich-gcp/                           # ICH E6(R3) GCP compliance
│   └── gcp_compliance_checker.py      #   Principle-level compliance scoring
├── regulatory-intelligence/           # Multi-jurisdiction tracker
│   └── regulatory_tracker.py          #   Guidance/regulation monitoring
└── human-oversight/                   # Human oversight & safety gates
    └── HUMAN_OVERSIGHT_QMS.md         #   Quality management system spec
```

## Standards Cross-Reference

The following table maps module capabilities to applicable standards across the five
monitored jurisdictions.

| Domain | FDA (US) | EMA (EU) | PMDA (Japan) | TGA (Australia) | Health Canada |
|--------|----------|----------|--------------|-----------------|---------------|
| **Device Classification** | 21 CFR 860; Product Codes (QAS, QDQ) | EU MDR 2017/745, Annex VIII | PMDA Act Art. 2, Class I-IV | TG Act s41BD, Class I-IIb/III | MDR SOR/98-282 |
| **Software as Medical Device** | FDA SaMD Framework; PCCP Guidance (2025) | MDCG 2019-11; EU AI Act 2024/1689 | MHLW SaMD Notification (2023) | TGA Regulatory Framework for Software (2023) | Guidance Doc: SaMD (2019) |
| **Clinical Evidence** | 21 CFR 812 (IDE); FDA Real-World Evidence | EU MDR Annex XIV, Annex XV (Clinical Eval) | PMDA Clinical Trial Act (2017) | TGA Clinical Evidence Guidelines | HC Clinical Trial Regulations (C.05) |
| **Quality System** | 21 CFR 820 (QMSR, eff. Feb 2026) | EU MDR Annex IX (QMS) | PMDA QMS Ministerial Ord. 169 | TGA Australian Manufacturing Principles | HC ISO 13485 Recognition (MDSAP) |
| **Data Integrity** | 21 CFR Part 11 (e-records, e-signatures) | EMA Annex 11 (Computerised Systems) | PMDA ER/ES Guidance | TGA Data Integrity (PIC/S PI 041) | HC Data Integrity Guidance (GUI-0069) |
| **GCP Compliance** | 21 CFR 50, 56, 312 | EU CTR 536/2014; ICH E6(R3) | PMDA GCP Ord. 28; ICH E6(R3) | TGA Note for Guidance on GCP | HC ICH E6(R3) Adoption |
| **AI/ML Specific** | FDA AI/ML Action Plan; PCCP Final Guidance | EU AI Act High-Risk (Annex III) | PMDA AI Medical Device Eval (2024) | TGA AI/ML Regulatory Reforms (2023) | HC Guidance on ML Devices (2024) |
| **Adverse Event Reporting** | 21 CFR 803 (MDR); MedWatch | EU MDR Art. 87 (Vigilance) | PMDA Adverse Event Reporting Ord. | TGA Adverse Event Reporting | HC Mandatory Problem Reporting |
| **Privacy / Data Protection** | HIPAA (45 CFR 160-164) | GDPR (Regulation 2016/679) | APPI (2022 amendments) | Privacy Act 1988; APPs | PIPEDA; Provincial health privacy |

## FDA AI/ML Device Statistics

The FDA maintains a public list of AI/ML-enabled devices that have received marketing
authorization. Key statistics relevant to this framework:

- **Total authorized AI/ML devices**: 1,000+ (as of late 2025)
- **Oncology-related authorizations**: A growing segment, including computer-aided
  detection/diagnosis for radiology, pathology image analysis, and treatment planning
  decision support.
- **Submission pathways used**: The majority of AI/ML devices have been cleared via
  510(k). De Novo classifications are used for novel algorithms without predicate
  devices. PMA is required for Class III devices.
- **PCCP adoption**: Following the August 2025 final guidance, an increasing number
  of submissions include predetermined change control plans allowing algorithm
  updates within pre-specified performance bounds.

These statistics inform the default pathway recommendations in
`fda-compliance/fda_submission_tracker.py` and the compliance checks in
`compliance_checker.py`.

## Quick Start

```python
from regulatory.compliance_checker import ComplianceChecker
from regulatory.fda_submission import FDASubmissionTracker

# Run compliance checks before federated training
checker = ComplianceChecker()
report = checker.check_all(config=training_config)

if report.overall_status == "pass":
    coordinator.start_training()
```

## Module Summaries

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `compliance_checker.py` | Pre-training HIPAA/GDPR/FDA validation | `ComplianceChecker`, `ComplianceReport` |
| `fda_submission.py` | Submission lifecycle tracking | `FDASubmissionTracker`, `SubmissionPathway` |
| `fda-compliance/` | Extended FDA pathway management with milestones | `FDASubmissionLifecycleTracker` |
| `irb-management/` | IRB protocol coordination across trial sites | `IRBProtocolManager` |
| `ich-gcp/` | E6(R3) principle-level compliance scoring | `GCPComplianceChecker` |
| `regulatory-intelligence/` | Multi-jurisdiction guidance monitoring | `RegulatoryTracker` |
| `human-oversight/` | Safety gates and automation boundaries | See `HUMAN_OVERSIGHT_QMS.md` |

## Related Modules

- [`privacy/`](../privacy/) — PHI detection, de-identification, consent management, RBAC
- [`federated/`](../federated/) — Secure aggregation, differential privacy, site enrollment
- [`q1-2026-standards/`](../q1-2026-standards/) — Cross-framework compliance objectives

## License

MIT. See [LICENSE](../LICENSE).
