# FDA Compliance Module

> **DISCLAIMER: RESEARCH USE ONLY** — This module provides research-grade tooling for
> understanding FDA submission workflows. It does not constitute regulatory advice and
> must not be used as the sole basis for a regulatory submission. Engage qualified
> regulatory professionals before interacting with the FDA.

## Overview

The `fda-compliance/` module implements a lifecycle tracker for FDA marketing submissions
of AI/ML-enabled medical devices used in oncology clinical trials. It covers all major
regulatory pathways and tracks documentation, milestones, review cycles, and PCCP
(Predetermined Change Control Plan) requirements per current FDA guidance.

Federated learning models present specific FDA challenges: the training data never leaves
institutional boundaries, which affects how clinical evidence and performance testing
documentation must be structured. This module encodes those considerations into the
submission workflow.

## Applicable Regulations and Guidance

| Document | Relevance |
|----------|-----------|
| 21 CFR Part 820 (QMSR, eff. Feb 2026) | Quality management system requirements |
| 21 CFR Part 11 | Electronic records and electronic signatures |
| 21 CFR Part 812 | Investigational device exemptions (IDE) |
| FDA PCCP Final Guidance (Aug 2025) | Predetermined change control plans for ML devices |
| FDA AI/ML SaMD Action Plan | Strategic framework for AI/ML-based SaMD oversight |
| IEC 62304:2006+AMD1:2015 | Software lifecycle processes for medical devices |
| ISO 14971:2019 | Risk management for medical devices |
| IEC 82304-1:2016 | Health software product safety |

## API Overview

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `SubmissionType` | `510k`, `de_novo`, `pma`, `breakthrough_device`, `pre_submission` | FDA submission pathways |
| `SubmissionStatus` | `planning` through `cleared`/`granted`/`approved`/`denied` | Lifecycle status |
| `DeviceClass` | `class_i`, `class_ii`, `class_iii` | FDA device classification |
| `DocumentCategory` | 15 categories including `ai_ml_documentation`, `pccp`, `cybersecurity` | Required document types |
| `DocumentStatus` | `not_started` through `finalized` | Document preparation status |
| `ReviewCycleType` | `initial_review`, `additional_info_request`, `deficiency_letter`, etc. | FDA review interactions |

### Core Dataclasses

**`SubmissionDocument`** — Tracks a single document within a submission package:
- `category`: Which `DocumentCategory` this document fulfills
- `status`: Current `DocumentStatus`
- `version`, `author`, `reviewer`: Document metadata
- `due_date`, `completed_date`: Timeline tracking

**`ReviewCycle`** — Records a single FDA review interaction:
- `cycle_type`: Type of review (initial, AI, deficiency)
- `submitted_date`, `response_due`, `response_date`: Cycle timeline
- `fda_questions`, `sponsor_responses`: Content exchanged

**`Milestone`** — Key submission milestones with completion tracking.

**`FDASubmission`** — Top-level submission record aggregating documents, milestones,
review cycles, and PCCP details for a single device submission.

### Primary Class: `FDASubmissionLifecycleTracker`

```python
from regulatory.fda_compliance.fda_submission_tracker import (
    FDASubmissionLifecycleTracker,
    SubmissionType,
    DeviceClass,
)

tracker = FDASubmissionLifecycleTracker()

# Create a new submission
submission = tracker.create_submission(
    device_name="OncoFL Tumor Response Predictor",
    submission_type=SubmissionType.DE_NOVO,
    device_class=DeviceClass.CLASS_II,
    intended_use="AI-assisted tumor response prediction from federated imaging data",
)

# Advance through lifecycle stages
tracker.advance_status(submission.submission_id)

# Check document completeness
gaps = tracker.get_documentation_gaps(submission.submission_id)

# Generate readiness report
report = tracker.generate_readiness_report(submission.submission_id)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `create_submission()` | Initialize a new FDA submission with auto-generated document requirements |
| `advance_status()` | Move submission to the next lifecycle stage with validation |
| `add_document()` | Add or update a document in the submission package |
| `update_document_status()` | Track document preparation progress |
| `add_review_cycle()` | Record an FDA review interaction |
| `add_milestone()` | Define and track submission milestones |
| `get_documentation_gaps()` | Identify missing or incomplete documents |
| `generate_readiness_report()` | Produce a submission readiness assessment |
| `get_submission_timeline()` | Calculate expected timeline based on pathway |

## Submission Pathway Decision Guide

```
Is there a legally marketed predicate device?
├── Yes → Is the device Class I or II?
│         ├── Yes → 510(k)
│         └── No  → PMA (Class III)
└── No  → Is the device low-to-moderate risk?
          ├── Yes → De Novo Classification
          └── No  → PMA
                    └── Does it qualify for Breakthrough Device?
                        └── Yes → Breakthrough Device Designation + PMA/De Novo
```

## PCCP Considerations for Federated Learning

Under the August 2025 final guidance, a PCCP allows manufacturers to describe
anticipated modifications to an AI/ML algorithm and the methodology for implementing
those changes without requiring a new marketing submission. For federated learning
systems, PCCP documentation should address:

1. **Retraining boundaries** — Which sites may contribute to retraining rounds and
   the minimum number of participating sites.
2. **Performance envelope** — Acceptable ranges for sensitivity, specificity, and
   AUC after model updates.
3. **Data drift monitoring** — How distributional shifts across federated sites
   are detected and reported.
4. **Aggregation algorithm changes** — Whether switching from FedAvg to FedProx
   (or similar) constitutes a modification requiring a new submission.

## File Reference

- **`fda_submission_tracker.py`** — Full implementation of the lifecycle tracker,
  enums, dataclasses, and submission management logic.

## License

MIT. See [LICENSE](../../LICENSE).
