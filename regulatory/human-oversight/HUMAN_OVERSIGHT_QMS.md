# Human Oversight Quality Management System

**Document ID:** HO-QMS-001
**Version:** 0.6.0
**Effective Date:** 2026-02-18
**Classification:** Quality System Document

> **DISCLAIMER: RESEARCH USE ONLY** — This document defines research-grade human oversight
> boundaries for AI-assisted clinical trial functions. It must not be adopted as an
> institutional quality system without independent validation by qualified clinical
> operations, quality assurance, and regulatory affairs professionals. Patient safety
> is the highest priority.

## 1. Purpose

This document establishes the quality management framework for human oversight of
AI/ML-enabled functions within federated learning oncology clinical trials. It defines
automation boundaries, risk tiers, safety gates, and escalation procedures to ensure
that AI systems augment rather than replace clinical judgment in all safety-critical
decisions.

The framework is designed to comply with:
- ICH E6(R3) Principles 10 (Quality Management) and 12 (Participant Safety)
- 21 CFR Part 820 / QMSR (Quality Management System Regulation, eff. Feb 2026)
- ISO 14971:2019 (Risk Management for Medical Devices)
- EU AI Act 2024/1689, Article 14 (Human Oversight of High-Risk AI Systems)
- FDA AI/ML Action Plan — Good Machine Learning Practice (GMLP) principles

## 2. Scope

This QMS applies to all AI-assisted functions within the `pai-oncology-trial-fl`
framework, including but not limited to:
- Case Report Form (CRF) auto-fill from source data
- Adverse event (AE) detection, grading, and causality assessment
- Regulatory submission document generation
- Clinical data quality checks
- Treatment response prediction via federated models

## 3. Definitions

| Term | Definition |
|------|-----------|
| **Autonomous** | System acts without human intervention |
| **Auto (supervised)** | System acts automatically but human reviews output before downstream use |
| **Draft** | System produces a proposed output that requires human review and approval |
| **Advisory** | System provides a recommendation; human makes the decision independently |
| **Prohibited** | Function must not be performed by the AI system under any circumstances without specified human authorization |
| **e-Signature** | Electronic signature compliant with 21 CFR Part 11, requiring authentication and intent documentation |
| **CtQ Factor** | Critical-to-Quality factor per ICH E6(R3) quality-by-design framework |

## 4. CRF Auto-Fill Risk Tiers

CRF auto-fill functionality is stratified into three risk tiers based on the clinical
significance of the data field, the potential for patient harm from errors, and the
complexity of the source data mapping.

### 4.1 Tier Definitions

#### LOW Risk — Automated with Audit Trail

**Criteria:** Demographic and administrative fields with direct, unambiguous source-to-CRF
mapping. Errors are unlikely to affect clinical decisions or patient safety.

| Field Category | Examples | Automation Level | Human Review |
|---------------|----------|-----------------|--------------|
| Demographics | Date of birth, sex, ethnicity, country | Auto (supervised) | Batch review; statistical sampling at 5-10% |
| Administrative | Site ID, visit number, form completion date | Auto (supervised) | Exception-based review |
| Identifiers | Subject ID, randomization number | Auto (supervised) | 100% system-verified; human spot-check |

**Controls:**
- Full audit trail logging every auto-filled value with source reference
- Automated range checks and cross-field consistency validation
- Quarterly quality review of auto-fill accuracy (target: >99.5%)
- Any field failing validation is escalated to Medium tier review

#### MEDIUM Risk — Draft with Mandatory Human Review

**Criteria:** Clinical measurements and laboratory values where mapping requires
interpretation, unit conversion, or clinical context. Errors could affect data analysis
or trigger inappropriate clinical actions.

| Field Category | Examples | Automation Level | Human Review |
|---------------|----------|-----------------|--------------|
| Vital signs | Blood pressure, heart rate, temperature, weight | Draft | 100% review by trained data entry personnel |
| Laboratory values | CBC, metabolic panel, tumor markers | Draft | 100% review; flagging of out-of-range values |
| Medications | Concomitant medications, dosing | Draft | 100% review by clinical staff |
| Medical history | Coded diagnoses (MedDRA/ICD) | Draft | 100% review by medically qualified reviewer |

**Controls:**
- AI generates draft CRF entries with confidence scores
- All draft entries are visually distinguished from verified entries
- Entries with confidence below 80% are flagged for priority review
- Source document reference is displayed alongside each draft value
- Reviewer must confirm or correct each value with e-signature

#### HIGH Risk — Advisory Only with Clinician Adjudication

**Criteria:** Fields involving clinical judgment, outcome assessment, or efficacy
endpoints. Errors could directly affect patient safety or trial integrity.

| Field Category | Examples | Automation Level | Human Review |
|---------------|----------|-----------------|--------------|
| Tumor assessment | RECIST measurements, response classification | Advisory | Clinician adjudication; independent radiology review |
| Efficacy endpoints | Overall survival, progression-free survival | Advisory | Clinician determination; AI suggestion is reference only |
| Eligibility criteria | Inclusion/exclusion assessment | Advisory | PI or qualified delegate must independently verify |
| Protocol deviations | Deviation classification and impact | Advisory | Quality manager review and classification |

**Controls:**
- AI provides advisory suggestions only; no pre-population of CRF fields
- Advisory output is presented in a separate panel, not in the CRF form itself
- Clinician must independently enter the value; copy-paste from AI is audit-logged
- Discrepancies between AI advisory and clinician entry are flagged for quality review
- All High-risk fields require e-signature per 21 CFR Part 11

### 4.2 Tier Escalation

Any CRF field may be escalated to a higher risk tier based on:
- Repeated auto-fill errors exceeding quality thresholds
- Protocol amendments that change the clinical significance of a field
- Regulatory authority feedback or inspection findings
- Data Safety Monitoring Board (DSMB) recommendations

Tier de-escalation requires documented justification, quality trend analysis over a
minimum of 90 days, and approval by the trial Quality Manager.

## 5. Adverse Event Automation Boundaries

AI assistance in adverse event management is subject to strict functional boundaries.
Each step in the AE workflow has an explicitly defined automation ceiling.

### 5.1 AE Workflow Automation Matrix

| AE Function | Automation Level | AI Role | Human Role | Rationale |
|-------------|-----------------|---------|------------|-----------|
| **Detection** | Auto (supervised) | Scan clinical notes, labs, imaging, and patient-reported data for potential AEs | Review AI-flagged events; confirm or dismiss | Early detection benefits from AI pattern recognition across high-volume data |
| **Grading (CTCAE)** | Draft | Propose CTCAE grade based on clinical data | Review and confirm or modify grade; e-signature required | Grading requires clinical context that may not be captured in structured data |
| **Causality Assessment** | Advisory | Provide causality recommendation with supporting evidence | Independently assess causality; AI output is reference only | Causality determination is inherently a medical judgment |
| **Expectedness** | Advisory | Flag whether AE matches known safety profile | Investigator determines expectedness | Requires knowledge of individual patient context |
| **Seriousness** | Draft | Propose SAE classification based on criteria | Clinician confirms seriousness; SAE triggers immediate escalation | SAE determination has direct safety and reporting implications |
| **Narrative** | Draft | Generate draft AE narrative from structured data | Medical writer reviews and finalizes | Narratives must accurately reflect clinical nuance |
| **Regulatory Submission** | **Prohibited without e-signature** | Prepare submission-ready documents | Authorized person must review and apply 21 CFR Part 11 e-signature before any submission | Regulatory submissions carry legal and safety obligations |

### 5.2 Automation Boundaries — Absolute Prohibitions

The following actions are **prohibited** for autonomous AI execution under all circumstances:

1. **No autonomous regulatory submission** — The system must never submit an AE/SAE report
   to any regulatory authority (FDA MedWatch, EudraVigilance, PMDA, etc.) without an
   authorized human applying a 21 CFR Part 11 compliant e-signature.

2. **No autonomous causality determination** — The system must not record a final
   causality assessment. AI output is advisory only and must be clearly labeled as such.

3. **No autonomous unblinding** — The system must not unblind treatment assignment for
   causality or expectedness assessment without explicit authorization from the PI or
   designated unblinding officer.

4. **No autonomous dose modification** — The system must not recommend or implement
   dose changes based on AE data without clinician review and authorization.

5. **No suppression of AE signals** — The system must not filter out, downgrade, or
   suppress potential AE signals without human review. All detected signals must be
   presented to the reviewer.

## 6. Safety Gates

Safety gates are hard-coded thresholds and procedural requirements that cannot be
overridden by configuration. They represent the non-negotiable boundaries of AI
autonomy in clinical trial operations.

### 6.1 SAE Immediate Escalation Gate

**Trigger:** Any event classified as a Serious Adverse Event (SAE) by either the AI
system or a human reviewer.

**Required Actions (all mandatory, no exceptions):**

| Step | Timeline | Responsible Party | AI Role |
|------|----------|-------------------|---------|
| Flag and alert | Immediate (< 5 minutes) | System (automated) | Generate alert to PI, sponsor safety, and trial coordinator |
| Initial assessment | Within 24 hours | Investigator (PI or delegate) | Provide supporting clinical data summary |
| Regulatory timeline determination | Within 24 hours | Sponsor safety team | Draft timeline calculation (7-day or 15-day) |
| Report preparation | Per regulatory timeline | Sponsor safety team | Generate draft ICSRs; human review and e-signature required |
| IRB notification | Per institutional policy | Site coordinator | Prepare draft notification; human review required |
| DSMB notification | Per charter | DSMB coordinator | Alert only; no AI-generated recommendations to DSMB |

**Escalation overrides:** None. SAE escalation cannot be delayed, deferred, or suppressed
by the AI system. If the system is unable to deliver the alert (network failure, system
downtime), a secondary alerting pathway (email, SMS, pager) must be activated.

### 6.2 Regulatory Submission Gate

**Rule:** No regulatory submission of any kind may be transmitted to a health authority
without a 21 CFR Part 11 compliant e-signature from an authorized individual.

**Authorized roles:** Medical Monitor, Director of Drug Safety, VP of Regulatory Affairs,
or designated qualified person as defined in the trial delegation log.

**Implementation:**
- Submission function requires two-factor authentication + e-signature
- Submission queue holds prepared documents for human review
- System logs all submission attempts, approvals, and rejections
- Failed authentication locks the submission function after 3 attempts
- All submissions are archived with full audit trail

### 6.3 AE Detection Sensitivity Gate

**Requirement:** The AI adverse event detection system must maintain a sensitivity
(true positive rate) of **greater than 95%** for clinically significant adverse events,
validated against a human-reviewed reference standard.

**Validation protocol:**
- Baseline validation against a manually curated AE dataset of at least 500 events
  prior to deployment
- Ongoing monitoring: Monthly comparison of AI-detected AEs against investigator-reported
  AEs across all sites
- If sensitivity drops below 95% on any monthly assessment:
  1. Immediate notification to the trial Quality Manager and Data Management Lead
  2. Root cause analysis within 5 business days
  3. Model retraining or recalibration within 15 business days
  4. Re-validation before resuming automated detection
  5. If sensitivity cannot be restored, revert to fully manual AE detection
- Site-level sensitivity monitoring: If any individual site shows sensitivity below 90%,
  that site reverts to manual AE detection pending investigation

**Specificity target:** Greater than 80% to limit alert fatigue. If specificity drops
below 80%, the system continues operating but the Quality Manager is notified to assess
the impact on clinical workflow.

**Performance reporting:** Monthly sensitivity/specificity metrics are reported to the
DSMB and included in continuing review submissions to the IRB.

### 6.4 Model Update Safety Gate

**Rule:** No federated model update that affects AE detection, grading, or causality
advisory functions may be deployed without:

1. Validation against the reference AE dataset (sensitivity >95%, specificity >80%)
2. Comparison of pre-update vs. post-update performance on a held-out test set
3. Sign-off by the trial Medical Monitor
4. Documentation in the PCCP change log (if under an FDA PCCP)

### 6.5 Data Integrity Gate

**Rule:** All AI-generated or AI-modified clinical data entries must maintain a complete
audit trail that records:

- Original source data reference
- AI system version and model checkpoint used
- Confidence score for the generated output
- Human reviewer identity and e-signature (where required)
- Timestamp of each action in the workflow
- Any corrections or overrides with rationale

## 7. Roles and Responsibilities

| Role | Oversight Responsibilities |
|------|--------------------------|
| **Principal Investigator** | Final authority on all clinical decisions at the site; reviews SAE determinations; may delegate specific review functions |
| **Medical Monitor** | Sponsor-level clinical oversight; approves model updates affecting safety functions; reviews aggregate safety data |
| **Quality Manager** | Monitors CRF auto-fill accuracy; oversees tier escalation/de-escalation; manages CAPA for quality deviations |
| **Data Management Lead** | Validates AI data entry against source; manages data queries; ensures Part 11 compliance for electronic records |
| **Regulatory Affairs** | Authorizes regulatory submissions; manages PCCP documentation; tracks jurisdiction-specific requirements |
| **DSMB** | Independent safety oversight; receives monthly AI performance metrics; may recommend trial modification or termination |

## 8. Quality Metrics and Monitoring

| Metric | Target | Frequency | Escalation Trigger |
|--------|--------|-----------|-------------------|
| CRF auto-fill accuracy (Low tier) | >99.5% | Monthly | Below 99.0% |
| CRF draft accuracy (Medium tier) | >95.0% | Monthly | Below 90.0% |
| AE detection sensitivity | >95.0% | Monthly | Below 95.0% |
| AE detection specificity | >80.0% | Monthly | Below 80.0% |
| SAE alert delivery time | <5 minutes | Continuous | Any alert >15 minutes |
| Regulatory submission gate compliance | 100% | Continuous | Any unauthorized submission attempt |
| Audit trail completeness | 100% | Quarterly | Any gap detected |

## 9. Change Control

Modifications to this QMS document require:
1. Change request with rationale and risk assessment
2. Impact analysis on active trials
3. Review by Quality Manager, Medical Monitor, and Regulatory Affairs
4. Approval signatures from the trial Sponsor and PI
5. Communication to all participating sites
6. Training documentation for affected personnel

## 10. References

- ICH E6(R3): Good Clinical Practice (published September 2025)
- 21 CFR Part 820 / QMSR (effective February 2026)
- 21 CFR Part 11: Electronic Records; Electronic Signatures
- ISO 14971:2019: Medical devices — Application of risk management
- EU AI Act (Regulation 2024/1689), Article 14: Human Oversight
- CTCAE v5.0: Common Terminology Criteria for Adverse Events
- MedDRA v27.0: Medical Dictionary for Regulatory Activities
- FDA GMLP: Good Machine Learning Practice for Medical Device Development

## License

MIT. See [LICENSE](../../LICENSE).
