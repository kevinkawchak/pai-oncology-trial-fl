# Multi-Site Federated Learning Data Use Agreement

## Consortium Agreement for Federated Oncology AI Research

**Agreement Date:** _______________

**Consortium Name:** _______________

**Coordinating Institution:** _______________

---

## 1. Parties

This agreement governs federated learning collaboration among the following participating institutions:

| Site ID | Institution | Principal Investigator | IRB Approval # |
|---------|-------------|----------------------|----------------|
| Site 1 | | | |
| Site 2 | | | |
| Site 3 | | | |

## 2. Federated Learning Protocol

### 2.1 Architecture
- **Coordinator:** Central server hosted at the coordinating institution.
- **Clients:** Each participating site operates a local training node.
- **Data Flow:** Only encrypted model parameter updates are transmitted. Raw patient data never leaves the originating site.

### 2.2 Training Protocol
- **Algorithm:** Federated Averaging (FedAvg) with configurable local epochs.
- **Rounds:** Minimum _____ training rounds with all participating sites.
- **Minimum Participation:** At least _____ sites must participate in each round.

### 2.3 Privacy Configuration
| Parameter | Value |
|-----------|-------|
| Differential Privacy Epsilon | _____ per round |
| Differential Privacy Delta | _____ |
| Secure Aggregation | Enabled / Disabled |
| Maximum Gradient Norm | _____ |

## 3. Physical AI Components

### 3.1 Robotic Telemetry Data
Sites contributing surgical robot telemetry (e.g., da Vinci systems) must ensure all telemetry data is de-identified and stripped of location-specific identifiers before feature extraction.

### 3.2 Digital Twin Data
Patient digital twin simulations are executed locally. Only aggregate simulation statistics (not individual patient models) may be shared for cross-site validation.

## 4. Data Governance

### 4.1 Consent Requirements
- All patients must provide informed consent for federated learning participation.
- Consent must cover: model training, cross-site aggregation, and publication of aggregate results.
- Consent status must be tracked in the platform's consent management system.

### 4.2 Audit Requirements
- All sites must enable audit logging for data access and model training.
- Audit logs must be retained for a minimum of 7 years.
- Annual compliance audits will be coordinated by the lead institution.

## 5. Intellectual Property

- The federated model is jointly owned by all participating institutions.
- Individual site data remains the property of the originating institution.
- Publications must acknowledge all contributing sites.

## 6. Withdrawal

- A site may withdraw from the consortium with 60 days written notice.
- Upon withdrawal, the site's contribution to the global model cannot be individually removed (due to the aggregation process), but no further data from the site will be used.
- The withdrawing site must delete any global model parameters received from the coordinator.

## 7. Dispute Resolution

Disputes will be resolved through mediation by an independent committee appointed by the consortium steering board.

## 8. Signatures

Each participating institution must sign below to indicate agreement:

| Site ID | PI Signature | Privacy Officer Signature | Date |
|---------|-------------|--------------------------|------|
| Site 1 | | | |
| Site 2 | | | |
| Site 3 | | | |
