# ICH-GCP Compliance Module

> **DISCLAIMER: RESEARCH USE ONLY** — This module provides research-grade tooling for
> assessing GCP compliance. It does not replace qualified clinical operations oversight
> or regulatory authority inspection. Independent review by GCP-trained personnel is
> required before relying on any compliance assessment.

## Overview

The `ich-gcp/` module implements a compliance checker against the thirteen fundamental
principles of ICH E6(R3) Good Clinical Practice, published in September 2025. E6(R3)
represents the first major revision of the ICH GCP guideline since E6(R2) in 2016, and
introduces a risk-proportionate, quality-by-design approach that is particularly
relevant to technology-enabled clinical trials such as those using federated learning.

This module assesses trial configurations and operational data against each GCP principle,
produces structured findings with severity classification (compliant, minor finding,
major finding, critical finding), and computes an overall compliance score.

## ICH E6(R3): What Changed

The September 2025 publication of E6(R3) introduced several changes that directly
affect AI/ML-enabled federated learning trials:

| Area | E6(R2) Approach | E6(R3) Approach |
|------|----------------|----------------|
| **Risk management** | General risk-based monitoring | Explicit quality-by-design with critical-to-quality (CtQ) factors |
| **Technology** | Limited guidance on electronic systems | Dedicated provisions for technology-enabled trials, eConsent, decentralized elements |
| **Data governance** | Document-centric | Data-centric; recognizes diverse data sources including real-world data |
| **Oversight model** | Prescriptive monitoring visits | Proportionate oversight based on identified risks |
| **Vendor management** | Basic outsourcing provisions | Enhanced requirements for technology vendors and platform providers |
| **Patient centricity** | Limited scope | Explicit principle for participant safety and well-being as paramount |

For federated learning specifically, E6(R3)'s risk-proportionate approach allows sites
to calibrate monitoring intensity based on actual risk rather than applying uniform
procedures. Its data governance provisions also clarify expectations when data remains
distributed across sites.

## The Thirteen GCP Principles

The `GCPPrinciple` enum implements the following E6(R3) principles, each assessed
independently by the compliance checker:

| # | Principle | Key Assessment Areas |
|---|-----------|---------------------|
| 1 | Ethical Conduct | Helsinki Declaration alignment, beneficence, justice |
| 2 | Scientific Soundness | Protocol design, statistical plan, endpoint validity |
| 3 | Participant Rights | Autonomy, privacy, voluntary participation |
| 4 | Informed Consent | Comprehension, voluntariness, ongoing consent, eConsent |
| 5 | IRB/IEC Review | Independent review, continuing oversight, composition |
| 6 | Qualified Personnel | Training, delegation, supervision |
| 7 | Data Integrity | Accuracy, completeness, traceability, 21 CFR Part 11 |
| 8 | Protocol Compliance | Adherence, deviation management, amendment procedures |
| 9 | Investigational Product | Storage, dispensing, accountability, blinding |
| 10 | Quality Management | CtQ factors, risk identification, CAPA |
| 11 | Record Keeping | Essential documents, retention, audit trail |
| 12 | Participant Safety | AE monitoring, DSMB oversight, stopping rules |
| 13 | Risk-Proportionate Approach | Monitoring plan calibration, centralized monitoring |

## API Summary

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `GCPPrinciple` | 13 principles as listed above | Principle being assessed |
| `ComplianceLevel` | `compliant`, `minor_finding`, `major_finding`, `critical_finding` | Severity classification |
| `FindingCategory` | Categories for structured findings | Finding categorization |

### Core Dataclasses

**`ComplianceFinding`** — A single compliance observation:
- `principle`: Which `GCPPrinciple` is affected
- `level`: Severity (`ComplianceLevel`)
- `description`: Finding narrative
- `reference`: Specific E6(R3) section reference
- `recommendation`: Suggested corrective action

**`PrincipleAssessment`** — Assessment of one principle:
- `principle`: The assessed principle
- `overall_level`: Aggregate compliance level
- `findings`: List of `ComplianceFinding` objects
- `score`: Numeric score (0.0-1.0)

**`GCPComplianceReport`** — Full compliance report:
- `assessments`: Per-principle assessment results
- `overall_score`: Weighted aggregate score
- `critical_findings_count`: Number of critical findings
- `recommendations`: Prioritized corrective actions

### Primary Class: `GCPComplianceChecker`

```python
from regulatory.ich_gcp.gcp_compliance_checker import (
    GCPComplianceChecker,
    GCPPrinciple,
)

checker = GCPComplianceChecker()

# Assess full GCP compliance
report = checker.assess_compliance(
    protocol_config=protocol,
    site_data=site_operational_data,
    training_config=federated_config,
)

# Check a specific principle
data_integrity = checker.assess_principle(
    GCPPrinciple.DATA_INTEGRITY,
    audit_trail=audit_log,
    part11_config=electronic_records_config,
)

print(f"Overall GCP score: {report.overall_score:.1%}")
print(f"Critical findings: {report.critical_findings_count}")
```

## Federated Learning GCP Considerations

E6(R3)'s provisions map to federated learning as follows:

- **Quality Management (Principle 10)** — CtQ factors for federated trials include
  model convergence metrics, site dropout handling, and aggregation integrity.
- **Data Integrity (Principle 7)** — Audit trails must span the federated coordinator
  and all participating sites, with cryptographic verification that no data left
  institutional boundaries.
- **Risk-Proportionate Approach (Principle 13)** — Sites with higher data heterogeneity
  or smaller patient populations may warrant more intensive centralized monitoring of
  their model update contributions.
- **Participant Safety (Principle 12)** — AE detection models trained via federated
  learning must meet safety gate thresholds (see `human-oversight/HUMAN_OVERSIGHT_QMS.md`).

## References

- ICH E6(R3): Good Clinical Practice, published September 2025
- ICH E8(R1): General Considerations for Clinical Studies
- ICH E9(R1): Statistical Principles for Clinical Trials (Estimands)
- 21 CFR Parts 11, 50, 56, 312
- EU Clinical Trials Regulation (EU) 536/2014

## File Reference

- **`gcp_compliance_checker.py`** — Full implementation of the E6(R3) compliance
  checker with principle-level assessment, scoring, and report generation.

## License

MIT. See [LICENSE](../../LICENSE).
