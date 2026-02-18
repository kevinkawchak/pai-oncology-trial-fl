# Breach Response Module

**Version 0.6.0** | `privacy/breach-response/`

> **DISCLAIMER: RESEARCH USE ONLY.** This module provides research-grade breach detection
> and incident response tooling. It does not replace an organization's formal Incident
> Response Plan, legal counsel, or regulatory notification obligations. Actual breach
> response must be conducted under the direction of qualified privacy and security officers.

## Overview

The Breach Response module implements a structured incident response workflow for PHI/PII
breaches in federated oncology trial systems. It is aligned with the **HIPAA Breach
Notification Rule** (45 CFR 164.400-414), **GDPR Articles 33-34**, and the **NIST SP
800-61r2** Computer Security Incident Handling Guide.

The module provides automated breach indicator monitoring, four-factor risk assessment per
45 CFR 164.402, incident lifecycle management (detection through closure), notification
deadline tracking (60-day HIPAA / 72-hour GDPR), remediation action management, and
post-incident lessons-learned capture.

## File Structure

```
breach-response/
├── README.md                       # This file
└── breach_response_protocol.py     # Breach detection, risk assessment, and response
```

## Regulatory References

- **45 CFR 164.400-414** -- HIPAA Breach Notification Rule
- **45 CFR 164.402** -- Four-factor risk assessment for breach determination
- **GDPR Article 33** -- Notification to supervisory authority (72 hours)
- **GDPR Article 34** -- Communication to data subjects
- **HHS Breach Notification Guidance** (2009, updated 2013)
- **NIST SP 800-61r2** -- Computer Security Incident Handling Guide

## API Overview

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `BreachSeverity` | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Severity classification aligned with NIST SP 800-61r2 |
| `IncidentPhase` | `DETECTED`, `TRIAGED`, `INVESTIGATING`, `CONTAINED`, `ERADICATING`, `RECOVERING`, `RESOLVED`, `NOTIFIED`, `CLOSED` | Incident lifecycle phases |
| `NotificationType` | `HHS_OCR`, `STATE_AG`, `INDIVIDUAL`, `MEDIA`, `GDPR_DPA`, `GDPR_INDIVIDUAL` | Regulatory notification types |
| `RemediationStatus` | `PLANNED`, `IN_PROGRESS`, `COMPLETED`, `VERIFIED`, `FAILED` | Remediation action status |

### Data Classes

| Class | Key Fields | Purpose |
|-------|-----------|---------|
| `RiskAssessment` | `phi_nature_score`, `unauthorized_access_score`, `acquisition_score`, `mitigation_score` | Four-factor risk assessment per 45 CFR 164.402. Composite score determines notification requirement. |
| `BreachNotification` | `notification_type`, `due_date`, `sent_date`, `recipient`, `status` | Tracks regulatory notification obligations and deadlines |
| `RemediationAction` | `action_id`, `description`, `assigned_to`, `status`, `verification_notes` | Individual remediation step with verification |
| `BreachIncident` | `incident_id`, `severity`, `phase`, `risk_assessment`, `notifications`, `remediation_actions`, `affected_sites`, `affected_individuals`, `root_cause`, `lessons_learned` | Full incident record with lifecycle tracking |

### BreachResponseManager

The primary class for breach detection and incident response.

```python
from privacy.breach_response.breach_response_protocol import (
    BreachResponseManager,
    BreachSeverity,
)

manager = BreachResponseManager(
    auto_escalate_threshold=3,       # Indicators before auto-escalation
    escalation_window_minutes=60,    # Time window for indicator counting
)
```

#### Indicator Reporting

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `report_indicator` | `(indicator_type, description, severity, source, metadata) -> str or None` | `str` or `None` | Report a breach indicator. Returns incident ID if auto-escalation triggers. |
| `get_all_indicators` | `() -> list` | `list` | Retrieve all reported indicators |

#### Incident Lifecycle

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `create_incident` | `(severity, description, affected_sites, phi_types) -> str` | `str` | Create a new breach incident, returns incident ID |
| `triage` | `(incident_id, notes) -> bool` | `bool` | Move incident to triaged phase |
| `investigate` | `(incident_id, notes) -> bool` | `bool` | Move incident to investigating phase |
| `contain` | `(incident_id, actions_taken) -> bool` | `bool` | Mark incident as contained |
| `eradicate` | `(incident_id, notes) -> bool` | `bool` | Mark root cause eradicated |
| `resolve` | `(incident_id, root_cause, lessons_learned) -> bool` | `bool` | Resolve the incident |
| `close` | `(incident_id) -> bool` | `bool` | Close the incident |

#### Risk Assessment

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `assess_risk` | `(incident_id, phi_nature, unauthorized_access, acquisition, mitigation) -> RiskAssessment` | `RiskAssessment` | Perform four-factor risk assessment. Composite score >= 0.5 triggers notification requirements. |

#### Notifications

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_notification` | `(incident_id, notification_type, recipient, due_date) -> bool` | `bool` | Schedule a regulatory notification |
| `send_notification` | `(incident_id, notification_type) -> bool` | `bool` | Record that a notification was sent |
| `get_overdue_notifications` | `() -> list` | `list` | List notifications past their deadline |

#### Remediation

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_remediation` | `(incident_id, description, assigned_to) -> str` | `str` | Add a remediation action, returns action ID |
| `complete_remediation` | `(incident_id, action_id, notes) -> bool` | `bool` | Mark a remediation action as completed |

### Four-Factor Risk Assessment

Per 45 CFR 164.402, breach determination uses four factors, each scored 0.0-1.0:

| Factor | Score Range | Description |
|--------|------------|-------------|
| Factor 1: PHI Nature | 0.0 (low sensitivity) - 1.0 (SSN, diagnosis + name) | Nature and extent of PHI involved |
| Factor 2: Unauthorized Access | 0.0 (internal, authorized adjacent) - 1.0 (external, malicious) | Who used or accessed the PHI |
| Factor 3: Acquisition | 0.0 (not acquired/viewed) - 1.0 (confirmed acquisition) | Whether PHI was actually acquired or viewed |
| Factor 4: Mitigation | 0.0 (fully mitigated) - 1.0 (no mitigation possible) | Extent to which risk has been mitigated |

Composite score = 0.30(F1) + 0.20(F2) + 0.30(F3) + 0.20(F4). Score >= 0.5 means notification is likely required.

## Usage Example

```python
from privacy.breach_response.breach_response_protocol import (
    BreachResponseManager,
    BreachSeverity,
)

manager = BreachResponseManager(auto_escalate_threshold=3)

# Report breach indicators
manager.report_indicator(
    indicator_type="anomalous_access",
    description="Bulk data export from site_03 outside business hours",
    severity=BreachSeverity.HIGH,
    source="access_control",
)

# Create incident manually
incident_id = manager.create_incident(
    severity=BreachSeverity.HIGH,
    description="Unauthorized bulk data export detected at site_03",
    affected_sites=["site_03"],
    phi_types=["names", "mrn", "dates"],
)

# Perform four-factor risk assessment
risk = manager.assess_risk(
    incident_id=incident_id,
    phi_nature=0.7,            # MRN + names = high sensitivity
    unauthorized_access=0.6,    # External contractor
    acquisition=0.8,            # Data confirmed downloaded
    mitigation=0.3,             # Encryption was in place
)
print(f"Composite risk: {risk.calculate_risk():.2f}")
print(f"Notification required: {risk.requires_notification}")

# Lifecycle progression
manager.investigate(incident_id, notes="Forensic analysis initiated")
manager.contain(incident_id, actions_taken="Account suspended, VPN revoked")
manager.resolve(incident_id, root_cause="Compromised contractor credentials")
```

## Notification Timeline Requirements

| Regulation | Deadline | Trigger | Notification Type |
|-----------|----------|---------|------------------|
| HIPAA | 60 days from discovery | Breach affecting any individuals | `HHS_OCR` |
| HIPAA | 60 days from end of calendar year | Breach affecting < 500 individuals | `HHS_OCR` (annual) |
| HIPAA | Without unreasonable delay | Breach affecting >= 500 individuals | `MEDIA`, `INDIVIDUAL` |
| State Laws | Varies (typically 30-60 days) | Per state breach notification law | `STATE_AG` |
| GDPR | 72 hours from awareness | Any personal data breach | `GDPR_DPA` |
| GDPR | Without undue delay | High risk to data subjects | `GDPR_INDIVIDUAL` |

## License

MIT License. See [LICENSE](../../LICENSE) for details.
