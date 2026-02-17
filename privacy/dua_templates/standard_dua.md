# Standard Data Use Agreement (DUA)

## Federated Oncology Trial — Data Use Agreement Template

**Agreement Date:** _______________

**Data Provider (Site):** _______________

**Data Recipient (Coordinator):** _______________

**Study Title:** _______________

**Protocol Number:** _______________

---

## 1. Purpose

This Data Use Agreement establishes the terms and conditions under which de-identified clinical data may be used within a federated learning framework for oncology clinical trial research. Under this federated model, **no raw patient data leaves the originating institution**. Only model parameter updates (gradients) are shared with the central coordinator.

## 2. Scope of Data

- **Data Types:** De-identified clinical features, imaging biomarkers, treatment outcomes, and robotic procedure telemetry.
- **Data Volume:** Approximately _____ patient records per site.
- **Retention Period:** Data will be retained locally at each site for the duration of the study plus _____ years.

## 3. Privacy Protections

### 3.1 De-Identification
All data must be de-identified in accordance with HIPAA Safe Harbor (45 CFR 164.514(b)) before any model training occurs. The 18 HIPAA identifiers must be removed or replaced using the platform's automated de-identification pipeline.

### 3.2 Differential Privacy
Model updates shared with the coordinator will be protected by differential privacy with a minimum privacy budget of epsilon = _____ per training round.

### 3.3 Secure Aggregation
When enabled, client model updates are masked using pairwise cryptographic protocols so that the coordinator only observes the aggregate update.

## 4. Permitted Uses

- Training and validating federated AI models for oncology treatment prediction.
- Cross-site model aggregation for improved generalization.
- Publication of aggregate results (no individual patient data).

## 5. Prohibited Uses

- Re-identification of patients from model parameters.
- Sharing raw data, model parameters, or derived features with unauthorized parties.
- Commercial use of individual patient data beyond the scope of the study.

## 6. Security Requirements

- All communications between sites and the coordinator must use TLS 1.3 or equivalent encryption.
- Access to the federated learning platform must be authenticated and role-based.
- Audit logs must be maintained for all data access and model training events.

## 7. Breach Notification

In the event of a data breach or suspected re-identification, the affected site must notify the coordinator and the relevant IRB within 72 hours.

## 8. Termination

Either party may terminate this agreement with 30 days written notice. Upon termination, all local data remains with the originating site; shared model parameters will be deleted from the coordinator.

## 9. Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Data Provider PI | | | |
| Data Provider Privacy Officer | | | |
| Coordinator PI | | | |
| Coordinator Privacy Officer | | | |
