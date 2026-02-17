# Communication Protocols for Federated Oncology Trial Platform

## Overview

This document specifies the communication protocols used between
federated learning sites, simulation engines, agentic AI systems, and
clinical data stores. All protocols prioritize security, auditability,
and compliance with clinical trial regulations.

---

## 1. Inter-Site Communication (Federation)

### 1.1 Transport Layer

All inter-site communication uses **TLS 1.3** with mutual
authentication (mTLS). Certificate management follows:

- Each site generates a CSR signed by the federation CA
- Certificates are rotated every 90 days
- Certificate revocation via OCSP stapling
- Minimum cipher suite: TLS_AES_256_GCM_SHA384

### 1.2 Model Update Protocol

The federated learning coordination uses a request-response protocol
over gRPC (HTTP/2):

```
Coordinator -> Site: RoundStart(round_id, global_params_hash)
Site -> Coordinator: Acknowledge(round_id, site_id)
Coordinator -> Site: GlobalParameters(round_id, parameters)
Site -> Coordinator: LocalUpdate(round_id, site_id, delta_params, sample_count, metrics)
Coordinator -> Site: RoundResult(round_id, aggregated_params_hash, global_metrics)
```

**Message size limits:**
- Maximum message: 100 MB (sufficient for large model updates)
- Streaming for messages > 4 MB

**Retry policy:**
- Exponential backoff: 1s, 2s, 4s, 8s, 16s
- Maximum retries: 5
- Idempotency keys on all mutating operations

### 1.3 Secure Aggregation Protocol

When secure aggregation is enabled, the protocol extends to:

```
Site_i -> Site_j: PairwiseMask(round_id, mask_share_ij)
Site_i -> Coordinator: MaskedUpdate(round_id, masked_params)
Coordinator: Aggregate(sum of masked_params) = sum of true params
```

The coordinator never observes individual site updates.

---

## 2. Intra-Site Communication

### 2.1 Simulation Engine Bridge Protocol

The Isaac-MuJoCo bridge and other cross-engine bridges use a
shared-memory protocol for high-frequency state exchange:

```
Producer (Engine A):
  1. Write state to shared memory buffer
  2. Update sequence number (atomic)
  3. Signal consumer via eventfd

Consumer (Engine B):
  1. Wait on eventfd
  2. Read sequence number (atomic)
  3. Copy state from shared memory
  4. Apply coordinate transformation
```

**Timing constraints:**
- Maximum latency: 1 ms for 500 Hz control loops
- Buffer size: Double-buffered to avoid read-write contention
- Fallback: Socket-based IPC if shared memory unavailable

### 2.2 Agent Communication Protocol

Agentic AI systems communicate via a message bus:

```
Agent_A -> MessageBus: Publish(topic="tumor_board", message)
MessageBus -> Agent_B: Deliver(topic="tumor_board", message)
```

Message format follows the Model Context Protocol (MCP):

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "query_patient_record",
    "arguments": {
      "patient_id": "PAT-001-DEIDENT",
      "data_type": "imaging"
    }
  },
  "id": "msg-uuid"
}
```

### 2.3 ROS 2 Integration

For sites using ROS 2, the platform provides:

- Custom message types for federated learning updates
- Service interfaces for policy export/import
- Action interfaces for long-running validation tasks
- Parameter server integration for runtime configuration

QoS profiles:

| Topic | Reliability | Durability | History |
|-------|-------------|------------|---------|
| /joint_states | RELIABLE | VOLATILE | KEEP_LAST(1) |
| /force_torque | BEST_EFFORT | VOLATILE | KEEP_LAST(1) |
| /policy_update | RELIABLE | TRANSIENT_LOCAL | KEEP_ALL |
| /safety_status | RELIABLE | VOLATILE | KEEP_LAST(1) |

---

## 3. Data Store Access

### 3.1 FHIR API

Clinical data access uses HL7 FHIR R4 REST API:

```
GET /fhir/Patient?identifier=deident-id
GET /fhir/Observation?patient=deident-id&code=tumor-volume
POST /fhir/DiagnosticReport
```

- Authentication: OAuth 2.0 with SMART on FHIR scopes
- Authorization: Scope-based access per agent role
- Audit: Every access logged per HIPAA requirements

### 3.2 DICOM

Imaging data access uses DICOMweb:

```
GET /dicomweb/studies/{study_uid}/series/{series_uid}/instances
GET /dicomweb/studies/{study_uid}/series/{series_uid}/instances/{instance_uid}/rendered
```

- Transport: HTTPS with institutional certificate
- De-identification: Applied server-side before delivery
- Caching: Allowed for de-identified data only

---

## 4. Safety-Critical Communication

### 4.1 Emergency Stop Protocol

```
Any Component -> SafetyController: ESTOP(reason, source_id)
SafetyController -> All Actuators: HALT(immediate=true)
SafetyController -> Audit Logger: LOG(estop_event)
SafetyController -> Human Operator: ALERT(estop_notification)
```

**Requirements:**
- Maximum latency: 10 ms from signal to actuator halt
- Redundant channels: Hardware watchdog + software signal
- Cannot be overridden by software alone
- Logged with microsecond timestamps

### 4.2 Heartbeat Protocol

All safety-critical components send periodic heartbeats:

```
Component -> SafetyController: HEARTBEAT(component_id, status, timestamp)
```

- Interval: 100 ms
- Timeout: 500 ms (3 missed heartbeats triggers failsafe)
- Failsafe action: Transition to teleoperation mode

---

## 5. Authentication and Authorization

### 5.1 Inter-Site Authentication

- Mutual TLS with federation-issued certificates
- JWT tokens for API access (RS256 signing)
- Token lifetime: 1 hour, refresh token: 24 hours

### 5.2 Agent Authentication

- Each agent has a unique service account
- Permissions mapped from AgentRole to allowed operations
- All tool invocations include agent identity in audit trail

### 5.3 Human Authentication

- Multi-factor authentication for all human operators
- Session timeout: 30 minutes of inactivity
- Privileged operations require re-authentication

---

## Protocol Version: 1.0.0
