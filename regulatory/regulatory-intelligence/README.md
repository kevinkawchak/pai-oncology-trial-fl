# Regulatory Intelligence Module

> **DISCLAIMER: RESEARCH USE ONLY** — This module provides research-grade tooling for
> tracking regulatory developments. It does not constitute legal or regulatory advice.
> Consult qualified regulatory affairs professionals for jurisdiction-specific compliance
> obligations.

## Overview

The `regulatory-intelligence/` module implements a multi-jurisdiction regulatory tracker
that monitors guidance documents, regulation changes, enforcement actions, recalls,
and safety communications across five major health authority jurisdictions. It provides
priority classification, deadline tracking, and impact assessment for AI/ML-enabled
oncology devices operating in federated learning environments.

Regulatory landscapes for AI/ML medical devices are evolving rapidly across all
jurisdictions. This module helps research teams maintain situational awareness of
developments that could affect trial design, data handling, or submission strategy.

## Monitored Jurisdictions

| Jurisdiction | Authority | Key AI/ML Frameworks |
|-------------|-----------|---------------------|
| **FDA** (US) | Food and Drug Administration | AI/ML SaMD Action Plan; PCCP Guidance (Aug 2025); QMSR (Feb 2026) |
| **EMA** (EU) | European Medicines Agency | EU MDR 2017/745; EU AI Act 2024/1689; MDCG guidance documents |
| **PMDA** (Japan) | Pharmaceuticals and Medical Devices Agency | AI-Based Medical Device Evaluation Guidance (2024); MHLW SaMD Notification |
| **TGA** (Australia) | Therapeutic Goods Administration | Regulatory Framework for AI/ML Medical Devices (2023); reform consultations |
| **Health Canada** | Health Products and Food Branch | ML-Enabled Medical Devices Guidance (2024); MDSAP participation |

## Update Categories

The tracker classifies regulatory developments by type and priority:

### Update Types

| Type | Description | Example |
|------|-------------|---------|
| `guidance_draft` | Proposed guidance open for comment | FDA draft guidance on AI transparency |
| `guidance_final` | Finalized guidance document | FDA PCCP Final Guidance (Aug 2025) |
| `regulation_proposed` | Proposed rule or regulation | QMSR proposed rule |
| `regulation_final` | Final rule with effective date | QMSR final rule (eff. Feb 2026) |
| `enforcement_action` | Warning letters, consent decrees | Device manufacturer warning letter |
| `advisory` | Safety advisories, field actions | Software vulnerability advisory |
| `standard_update` | ISO/IEC standard revisions | IEC 62304 amendment |
| `recall` | Device recall classification | Class I/II/III recall |
| `safety_communication` | Urgent safety notifications | Post-market safety signal |

### Priority Levels

| Priority | Criteria | Response Window |
|----------|----------|----------------|
| **Critical** | Immediate safety impact; mandatory recalls; enforcement actions | 24-48 hours |
| **High** | Final regulations with compliance deadlines; major guidance changes | 1-2 weeks |
| **Medium** | Draft guidances; proposed rules; standard updates | 30-90 days |
| **Low** | Informational advisories; conference presentations; industry trends | Quarterly review |

## API Summary

### Enums

| Enum | Purpose |
|------|---------|
| `Jurisdiction` | Health authority (`fda`, `ema`, `pmda`, `tga`, `health_canada`) |
| `UpdateType` | Regulatory development category |
| `Priority` | Impact priority level |
| `ImpactArea` | Affected domain (submission, quality, clinical, post-market, etc.) |

### Core Dataclasses

**`RegulatoryUpdate`** — A tracked regulatory development:
- `jurisdiction`: Originating health authority
- `update_type`: Category of the development
- `priority`: Assessed priority level
- `title`, `summary`: Description of the update
- `effective_date`, `comment_deadline`: Key dates
- `impact_areas`: Which operational domains are affected
- `action_items`: Required responses or assessments

**`ComplianceDeadline`** — An upcoming compliance obligation:
- `regulation`: Source regulation or guidance
- `jurisdiction`: Applicable jurisdiction
- `deadline`: Date by which compliance is required
- `requirements`: Specific compliance actions needed
- `status`: Tracking status (pending, in_progress, completed, waived)

**`ImpactAssessment`** — Analysis of how an update affects ongoing operations:
- `affected_submissions`: Submissions that may need revision
- `affected_protocols`: Clinical protocols requiring amendment
- `gap_analysis`: Identified compliance gaps
- `recommended_actions`: Prioritized response plan

### Primary Class: `RegulatoryTracker`

```python
from regulatory.regulatory_intelligence.regulatory_tracker import (
    RegulatoryTracker,
    Jurisdiction,
    Priority,
)

tracker = RegulatoryTracker()

# Add a regulatory update
tracker.add_update(
    jurisdiction=Jurisdiction.FDA,
    title="QMSR Final Rule Effective",
    summary="21 CFR Part 820 harmonized with ISO 13485:2016",
    effective_date="2026-02-02",
    priority=Priority.HIGH,
    impact_areas=["quality_system", "documentation"],
)

# Query upcoming deadlines
deadlines = tracker.get_upcoming_deadlines(
    jurisdictions=[Jurisdiction.FDA, Jurisdiction.EMA],
    days_ahead=90,
)

# Assess impact on active submissions
assessment = tracker.assess_impact(
    update_id=update.update_id,
    active_submissions=submission_list,
)
```

## Key Regulatory Timeline (2025-2026)

| Date | Event | Jurisdiction | Impact |
|------|-------|-------------|--------|
| Jan 2025 | AI/ML PCCP Draft Guidance | FDA | Comment period for PCCP framework |
| Aug 2025 | PCCP Final Guidance | FDA | Finalized ML update pathway |
| Sep 2025 | ICH E6(R3) Published | ICH (global) | New GCP standard for all jurisdictions |
| Feb 2026 | QMSR Effective | FDA | Quality system harmonization with ISO 13485 |
| 2026 Q1-Q2 | EU AI Act High-Risk Obligations Phase-In | EMA/EU | Conformity assessment for medical AI |
| 2026 | Health Canada ML Guidance Updates | Health Canada | Revised pre-market requirements |

## Federated Learning Regulatory Watch Items

The following areas warrant particular monitoring for federated learning oncology trials:

1. **Cross-border data governance** — Regulatory alignment (or divergence) between
   jurisdictions on whether federated model parameters constitute "data transfer."
2. **PCCP for federated retraining** — How FDA and other authorities will interpret
   predetermined change control plans for models retrained across distributed sites.
3. **EU AI Act conformity** — High-risk classification requirements for medical AI
   systems and whether federated training affects the conformity assessment process.
4. **Post-market surveillance** — Expectations for real-world performance monitoring
   of federated models across jurisdictions with different reporting requirements.

## File Reference

- **`regulatory_tracker.py`** — Full implementation of the multi-jurisdiction
  regulatory intelligence tracker with update management, deadline tracking,
  and impact assessment.

## License

MIT. See [LICENSE](../../LICENSE).
