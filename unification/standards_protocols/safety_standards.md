# Safety Standards for Physical AI Oncology Surgical Robotics

## Overview

This document maps the safety requirements for the unification
framework to applicable international standards and regulatory
guidance. All components that influence surgical robot behavior,
patient data handling, or clinical decision-making must comply
with these requirements.

---

## 1. Applicable Standards

### 1.1 Medical Device Software

| Standard | Scope | Applicability |
|----------|-------|---------------|
| IEC 62304:2006+A1:2015 | Medical device software lifecycle | All software components |
| IEC 82304-1:2016 | Health software products | Platform-level requirements |
| ISO 14971:2019 | Application of risk management | Risk analysis for all features |
| ISO 13485:2016 | Quality management systems | Development process |

### 1.2 Surgical Robotics

| Standard | Scope | Applicability |
|----------|-------|---------------|
| IEC 80601-2-77:2019 | Medical robots for surgery | Robot control interfaces |
| ISO 13482:2014 | Personal care robots safety | Collaborative robot modes |
| IEC 60601-1:2020 | Medical electrical equipment | Hardware-connected systems |
| IEC 60601-1-2:2020 | EMC requirements | Wireless communication |

### 1.3 AI/ML Specific

| Guidance | Scope | Applicability |
|----------|-------|---------------|
| FDA AI/ML Action Plan (2021) | AI/ML-based SaMD | All ML components |
| FDA PCCP Draft Guidance (2023) | Predetermined change control | Federated model updates |
| EU AI Act (2024) | AI risk classification | High-risk classification |
| ISO/IEC 23894:2023 | AI risk management | AI system risk analysis |

### 1.4 Data Protection

| Regulation | Scope | Applicability |
|------------|-------|---------------|
| HIPAA (45 CFR 160-164) | US health data privacy | All patient data handling |
| GDPR (EU 2016/679) | EU data protection | EU site participation |
| FDA 21 CFR Part 11 | Electronic records/signatures | Audit trails, submissions |

---

## 2. Software Safety Classification

### 2.1 IEC 62304 Classification

The unification framework components are classified as follows:

| Component | Class | Rationale |
|-----------|-------|-----------|
| Isaac-MuJoCo bridge | Class B | Contributes to policy training for surgical devices |
| Policy exporter | Class B | Exported policies may influence device behavior |
| Validation suite | Class A | Tooling only; does not affect device behavior |
| Framework detector | Class A | Diagnostic tool only |
| Model converter | Class B | Converted models used in safety-critical simulation |
| Unified agent interface | Class B | Agents may influence clinical decisions |
| Federated coordinator | Class B | Aggregated models used in clinical context |

### 2.2 Class B Requirements

For Class B components, IEC 62304 requires:

- Software development plan
- Software requirements specification
- Software architecture documentation
- Detailed design for safety-critical units
- Unit verification (testing)
- Integration testing
- Software system testing
- Risk management per ISO 14971
- Configuration management
- Problem resolution process

---

## 3. Risk Management (ISO 14971)

### 3.1 Hazard Categories

| Hazard ID | Category | Description |
|-----------|----------|-------------|
| HAZ-001 | Mechanical | Excessive force applied to patient tissue |
| HAZ-002 | Mechanical | Robot motion outside safe workspace |
| HAZ-003 | Software | Policy produces unsafe joint commands |
| HAZ-004 | Software | Cross-engine state corruption during sync |
| HAZ-005 | Data | PHI leakage through federated model updates |
| HAZ-006 | Data | Model poisoning via compromised federation site |
| HAZ-007 | Clinical | Agent provides incorrect clinical recommendation |
| HAZ-008 | Clinical | Treatment plan based on hallucinated data |
| HAZ-009 | System | Loss of communication during procedure |
| HAZ-010 | System | Emergency stop failure |

### 3.2 Risk Controls

| Hazard | Control | Verification |
|--------|---------|--------------|
| HAZ-001 | Hardware force limiter + software safety envelope | Force sensor validation test |
| HAZ-002 | Joint limit enforcement + workspace boundary check | Kinematic validation suite |
| HAZ-003 | Action clipping + safety constraint layer | Policy validation test |
| HAZ-004 | State hash verification + rollback mechanism | Cross-engine calibration |
| HAZ-005 | Differential privacy + secure aggregation | Privacy budget audit |
| HAZ-006 | Byzantine-tolerant aggregation + anomaly detection | Federated robustness test |
| HAZ-007 | Human-in-the-loop review + scope restrictions | Agent audit trail review |
| HAZ-008 | RAG grounding + hallucination detection | Agent output validation |
| HAZ-009 | Heartbeat monitoring + failsafe to teleop | Communication failure test |
| HAZ-010 | Redundant E-stop (hardware + software) | E-stop response time test |

### 3.3 Residual Risk Acceptance

Residual risks are evaluated using the risk matrix:

| Severity \ Probability | Frequent | Probable | Occasional | Remote | Improbable |
|------------------------|----------|----------|------------|--------|------------|
| Catastrophic | UNACCEPTABLE | UNACCEPTABLE | UNACCEPTABLE | ALARP | ALARP |
| Critical | UNACCEPTABLE | UNACCEPTABLE | ALARP | ALARP | ACCEPTABLE |
| Serious | UNACCEPTABLE | ALARP | ALARP | ACCEPTABLE | ACCEPTABLE |
| Minor | ALARP | ALARP | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE |
| Negligible | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE | ACCEPTABLE |

ALARP = As Low As Reasonably Practicable

---

## 4. Surgical Robot Safety Requirements

### 4.1 Force Limits (IEC 80601-2-77)

| Parameter | Limit | Rationale |
|-----------|-------|-----------|
| Maximum tool force (normal) | 20 N | Soft tissue protection |
| Maximum tool force (bone) | 100 N | Cortical bone drilling |
| Emergency stop force | 5 N (sustained) | Immediate tissue release |
| Maximum joint velocity | 2*pi rad/s | Collision avoidance |
| Maximum tool velocity | 0.5 m/s | Procedural safety |

These limits are hard-coded in the safety envelope and cannot be
overridden by configuration or policy output.

### 4.2 Watchdog Requirements

- Control loop watchdog: 10 ms timeout
- Communication watchdog: 500 ms timeout
- Power watchdog: Hardware-level, independent of software
- Failsafe action: Gravity compensation + brakes engaged

### 4.3 Mode Transitions

```
POWERED_OFF -> INITIALIZATION -> READY
READY -> MANUAL_TELEOP -> ASSISTED_TELEOP -> SUPERVISED_AUTONOMY
SUPERVISED_AUTONOMY -> ASSISTED_TELEOP (downgrade on confidence drop)
ANY_STATE -> EMERGENCY_STOP (immediate, any trigger)
EMERGENCY_STOP -> POWERED_OFF (requires human reset)
```

Transitions from lower to higher autonomy require explicit
clinician authorization and system readiness verification.

---

## 5. Federated Learning Safety

### 5.1 Model Update Validation

Every federated model update must pass:

1. Gradient norm check (reject if > 10x median)
2. Weight divergence check (reject if > 3 sigma from global)
3. Safety constraint preservation (all limits maintained)
4. Performance regression check (accuracy within tolerance)

### 5.2 Privacy Safety

- Differential privacy budget: Tracked per site per trial
- Secure aggregation: Mandatory for updates touching patient data
- PHI detection: Automated scan on all model artifacts
- Membership inference resistance: Validated quarterly

---

## 6. Verification and Validation Plan

### 6.1 Unit Testing

- All safety-critical functions: 100% branch coverage
- Property-based testing for numerical routines
- Mutation testing to validate test effectiveness

### 6.2 Integration Testing

- Cross-engine state transfer: 10,000-step drift test
- Federated aggregation: Byzantine fault injection
- Agent workflow: Scope violation detection

### 6.3 System Testing

- Hardware-in-the-loop: Complete procedure on phantom
- Stress testing: 100 concurrent federation sites
- Failure mode testing: All HAZ-xxx failure scenarios

### 6.4 Clinical Validation

- Pre-clinical: Cadaver study for surgical procedures
- Clinical: Prospective trial with human oversight
- Post-market: Continuous monitoring per FDA requirements

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-17 | PAI Oncology Trial FL | Initial release |

Review cycle: Quarterly or upon standard update, whichever is sooner.
