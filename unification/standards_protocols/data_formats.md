# Canonical Data Format Specifications

## Overview

This document defines the canonical data formats used across the unification
framework. All simulation engines, agentic AI frameworks, and federated
learning components exchange data through these standardized formats.

---

## 1. Simulation State Exchange Format

### 1.1 Joint State Record

All joint states are transmitted in the following canonical format,
regardless of the source simulation engine.

```json
{
  "joint_name": "shoulder_pan",
  "joint_type": "revolute",
  "position": 1.5708,
  "velocity": 0.0,
  "acceleration": 0.0,
  "torque": 0.0,
  "position_limits": [-6.2832, 6.2832],
  "velocity_limit": 6.2832,
  "torque_limit": 500.0,
  "damping": 0.1,
  "stiffness": 0.0,
  "friction": 0.0,
  "units": {
    "position": "radians",
    "velocity": "rad/s",
    "torque": "N*m"
  }
}
```

**Conventions:**
- Angles are always in radians (never degrees).
- Linear quantities are always in SI units (meters, N, kg).
- Quaternions use [w, x, y, z] ordering (scalar-first).
- Coordinate frames use Z-up as the canonical convention.
  Engines using Y-up (Isaac Sim) must convert before exchange.

### 1.2 Link State Record

```json
{
  "link_name": "tool0",
  "position": [0.5, 0.1, 0.8],
  "orientation": [1.0, 0.0, 0.0, 0.0],
  "linear_velocity": [0.0, 0.0, 0.0],
  "angular_velocity": [0.0, 0.0, 0.0],
  "mass": 0.5,
  "coordinate_frame": "world",
  "convention": "z_up"
}
```

### 1.3 Contact Record

```json
{
  "body_a": "needle_tip",
  "body_b": "tissue_surface",
  "position": [0.3, 0.0, 0.5],
  "normal": [0.0, 0.0, 1.0],
  "force": [0.0, 0.0, -2.5],
  "depth": 0.001,
  "friction_coefficient": 0.3
}
```

---

## 2. Policy Exchange Format

### 2.1 Policy Metadata (JSON)

Every exported policy must include a metadata file:

```json
{
  "policy_id": "uuid-v4",
  "name": "needle_biopsy_ppo",
  "version": "1.0.0",
  "policy_type": "rl",
  "algorithm": "PPO",
  "task_name": "NeedleBiopsy-v1",
  "observation_space": {
    "dimension": 48,
    "names": ["joint_pos_0", "..."],
    "normalization": {
      "mean": [0.0, "..."],
      "std": [1.0, "..."]
    }
  },
  "action_space": {
    "dimension": 6,
    "type": "continuous",
    "bounds": {
      "lower": [-1.0, "..."],
      "upper": [1.0, "..."]
    }
  },
  "safety_constraints": {
    "max_joint_velocity_rad_s": 6.2832,
    "max_end_effector_force_n": 50.0,
    "workspace_bounds": {
      "min": [-1.0, -1.0, 0.0],
      "max": [1.0, 1.0, 1.5]
    }
  },
  "training": {
    "framework": "isaac_lab",
    "steps": 5000000,
    "reward": 450.0,
    "success_rate": 0.92
  },
  "source_engine": "isaac_sim",
  "source_robot": "ur5e",
  "hash": "sha256_hex"
}
```

### 2.2 Policy Weight Formats

Supported formats in order of preference:

1. **SafeTensors** (.safetensors) - Recommended for security
2. **ONNX** (.onnx) - Recommended for cross-framework deployment
3. **PyTorch** (.pt) - Native for PyTorch-based training
4. **NumPy** (.npz) - Lightweight, no framework dependency

---

## 3. Telemetry and Audit Formats

### 3.1 Surgical Telemetry Record

```json
{
  "timestamp_utc": "2026-02-17T14:30:00.000Z",
  "session_id": "uuid-v4",
  "robot_model": "ur5e",
  "joint_positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
  "joint_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "joint_torques": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "end_effector_pose": {
    "position": [0.5, 0.1, 0.8],
    "orientation": [1.0, 0.0, 0.0, 0.0]
  },
  "force_torque": {
    "force": [0.0, 0.0, -2.5],
    "torque": [0.0, 0.0, 0.0]
  },
  "safety_status": "nominal",
  "control_mode": "position"
}
```

### 3.2 Audit Event Record

```json
{
  "event_id": "uuid-v4",
  "event_type": "tool_invoked",
  "timestamp": 1739800200.0,
  "agent_id": "uuid-v4",
  "user_identity": "system",
  "details": {},
  "previous_hash": "sha256_hex",
  "event_hash": "sha256_hex"
}
```

---

## 4. Federated Learning Exchange Formats

### 4.1 Model Update

```json
{
  "round_number": 5,
  "site_id": "hospital_a",
  "sample_count": 200,
  "parameter_format": "numpy_flat",
  "parameters": "base64_encoded_npz",
  "metrics": {
    "loss": 0.23,
    "accuracy": 0.87
  },
  "hash": "sha256_hex"
}
```

### 4.2 Aggregation Result

```json
{
  "round_number": 5,
  "strategy": "fedavg",
  "num_participants": 4,
  "global_metrics": {
    "accuracy": 0.89,
    "loss": 0.21
  },
  "converged": false,
  "parameter_hash": "sha256_hex"
}
```

---

## 5. DICOM and FHIR Interoperability

### 5.1 DICOM Tag Mapping

The platform maps DICOM tags to canonical field names for
simulation model construction:

| DICOM Tag | Canonical Field | Use Case |
|-----------|----------------|----------|
| (0010,0020) | patient_id | De-identified patient linkage |
| (0018,0050) | slice_thickness_mm | Mesh resolution |
| (0028,0030) | pixel_spacing_mm | Spatial calibration |
| (0020,0032) | image_position | Volume registration |
| (0008,0060) | modality | CT/MRI/PET selection |

### 5.2 FHIR Resource Mapping

| FHIR Resource | Canonical Field | Use Case |
|---------------|----------------|----------|
| Observation.valueQuantity | tumor_volume_cm3 | Digital twin initialization |
| Condition.code | diagnosis_code | Trial eligibility |
| MedicationRequest | treatment_regimen | Simulation parameters |
| Procedure | surgical_approach | Robot task selection |

---

## Versioning

All data format versions follow semantic versioning (MAJOR.MINOR.PATCH).
Breaking changes increment MAJOR. New fields with defaults increment MINOR.
Bug fixes increment PATCH.

Current version: **1.0.0**
