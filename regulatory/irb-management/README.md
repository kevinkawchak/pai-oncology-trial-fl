# IRB Management Module

> **DISCLAIMER: RESEARCH USE ONLY** — This module provides research-grade tooling for
> IRB protocol management. It does not replace institutional IRB processes or constitute
> regulatory guidance. All protocols must undergo proper IRB review before any human
> subjects research is conducted.

## Overview

The `irb-management/` module manages Institutional Review Board protocol lifecycles for
AI/ML-enabled federated learning oncology trials. It coordinates initial submissions,
amendments, continuing reviews, adverse event reporting, and informed consent versioning
across multiple participating trial sites.

Federated learning trials introduce IRB considerations beyond traditional multi-site
studies: model parameters (not patient data) cross institutional boundaries, which
raises questions about what constitutes "data sharing" under the Common Rule. Each
participating site's IRB must evaluate whether the federated architecture adequately
protects its patient population, and amendments to the aggregation protocol may require
coordinated IRB review across all sites.

## Applicable Regulations

| Regulation | Scope |
|------------|-------|
| 45 CFR Part 46 (Common Rule) | Federal policy for protection of human subjects |
| 21 CFR Part 50 | FDA informed consent requirements |
| 21 CFR Part 56 | FDA IRB requirements |
| ICH E6(R3) (Sep 2025) | Good Clinical Practice principles for IRB/IEC oversight |
| OHRP Guidance | Office for Human Research Protections guidance documents |
| EU CTR 536/2014 | EU Clinical Trials Regulation (for multi-national trials) |

## Key Concepts

### Protocol Lifecycle

```
DRAFT → SUBMITTED → UNDER_REVIEW → APPROVED → ACTIVE
                                  → APPROVED_WITH_CONDITIONS → ACTIVE
                                  → DEFERRED → resubmit
                                  → DISAPPROVED → revise or withdraw
ACTIVE → CONTINUING_REVIEW → APPROVED (renewed)
       → AMENDMENT → UNDER_REVIEW → APPROVED
       → SUSPENDED (safety concern)
       → TERMINATED (early termination)
       → CLOSED (study complete)
       → EXPIRED (missed continuing review deadline)
```

### Review Types

| Type | Criteria | Typical Timeline |
|------|----------|-----------------|
| **Full Board** | Greater than minimal risk; vulnerable populations; novel AI/ML interventions | 4-8 weeks |
| **Expedited** | Minimal risk; falls under one of the nine expedited categories (45 CFR 46.110) | 1-3 weeks |
| **Exempt** | Meets exemption criteria; no identifiable data collected | Days to 2 weeks |
| **Continuing Review** | Annual review of approved protocols | 2-4 weeks |
| **Amendment** | Protocol modifications; consent form changes; site additions | 1-4 weeks |

### Federated Learning IRB Considerations

1. **Data residency attestation** — Each site's IRB may require documentation that
   patient data does not leave institutional control during federated training.
2. **Model parameter risk assessment** — IRBs should evaluate whether model gradients
   or parameters could enable reconstruction of individual patient data (gradient
   inversion attacks), and whether the differential privacy budget is adequate.
3. **Coordinated amendments** — Changes to the federated aggregation protocol (e.g.,
   number of training rounds, clipping thresholds, privacy budget) may require
   synchronized amendments across all participating IRBs.
4. **Informed consent language** — Consent forms must explain federated learning in
   lay terms, including what is and is not shared across sites.
5. **Single IRB (sIRB) coordination** — NIH-funded multi-site studies may use a
   single IRB of record, simplifying but not eliminating site-level review.

## API Summary

### Enums

| Enum | Purpose |
|------|---------|
| `ProtocolStatus` | Protocol lifecycle states (`draft` through `closed`) |
| `ReviewType` | IRB review categories (`full_board`, `expedited`, `exempt`, etc.) |
| `AmendmentType` | Types of protocol amendments |

### Core Dataclasses

**`IRBProtocol`** — Top-level protocol record including:
- Protocol number, title, PI, study phase
- Status, review type, approval dates, expiration
- Participating sites and their individual approval status
- Linked consent form versions

**`Amendment`** — Protocol amendment with rationale, affected sections, and
multi-site review tracking.

**`ContinuingReview`** — Annual continuing review record with enrollment
statistics, adverse event summary, and risk-benefit reassessment.

**`ConsentVersion`** — Informed consent document versioning with change
tracking and site-level approval status.

### Primary Class: `IRBProtocolManager`

```python
from regulatory.irb_management.irb_protocol_manager import (
    IRBProtocolManager,
    ReviewType,
)

manager = IRBProtocolManager()

# Create a new protocol
protocol = manager.create_protocol(
    title="Federated Learning for Tumor Response Prediction",
    pi_name="Dr. Jane Smith",
    study_phase="Phase II",
    review_type=ReviewType.FULL_BOARD,
    sites=["Memorial Hospital", "University Medical Center", "City Oncology"],
)

# Submit an amendment
amendment = manager.submit_amendment(
    protocol_id=protocol.protocol_id,
    amendment_type="protocol_change",
    description="Increase differential privacy epsilon from 1.0 to 2.0",
    affected_sections=["Section 8.3: Privacy Parameters"],
)

# Track continuing review
review = manager.initiate_continuing_review(protocol.protocol_id)
```

## File Reference

- **`irb_protocol_manager.py`** — Full implementation of the IRB protocol lifecycle
  manager, including multi-site coordination, amendment tracking, and consent versioning.

## License

MIT. See [LICENSE](../../LICENSE).
