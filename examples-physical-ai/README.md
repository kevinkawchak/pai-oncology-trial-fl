# Physical AI Examples for Federated Oncology Trials

**Version:** 0.5.0
**Last Updated:** 2026-02-17
**License:** MIT
**Status:** Research Use Only

## Overview

Six production-quality example scripts demonstrating Physical AI capabilities
for federated learning in oncology clinical trials. Each script is self-contained,
runs without GPU hardware, and follows IEC/ISO regulatory standards for
medical robotic systems.

## Regulatory Standards Cross-Reference

| Example | IEC 80601-2-77 | ISO 14971 | IEC 62304 | 21 CFR Part 11 |
|---------|:--------------:|:---------:|:---------:|:--------------:|
| 01 Real-Time Safety Monitoring | Primary | Risk controls | SW lifecycle | -- |
| 02 Multi-Sensor Fusion | Sensor requirements | Hazard analysis | Class C SW | -- |
| 03 ROS 2 Surgical Deployment | System integration | Risk management | Architecture | -- |
| 04 Hand-Eye Calibration | Accuracy validation | Verification | Unit testing | -- |
| 05 Shared Autonomy Teleoperation | Operator controls | Use-related risk | State machines | -- |
| 06 Robotic Sample Handling | -- | Process risk | Traceability | Primary |

### Standard Descriptions

- **IEC 80601-2-77**: Medical electrical equipment -- Particular requirements for
  the basic safety and essential performance of robotically assisted surgical
  equipment (RASE).
- **ISO 14971**: Medical devices -- Application of risk management to medical
  devices. Covers hazard identification, risk estimation, risk evaluation, and
  risk control.
- **IEC 62304**: Medical device software -- Software life cycle processes.
  Defines software development and maintenance classes (A, B, C) based on
  safety classification.
- **21 CFR Part 11**: FDA regulation on electronic records and electronic
  signatures. Requires audit trails, access controls, and signature
  accountability.

## Quick Start

### Prerequisites

```bash
# Required (all examples)
pip install numpy>=1.24.0

# Optional (enhanced functionality)
pip install torch>=2.1.0          # GPU-accelerated inference
pip install mujoco>=3.0.0         # Physics simulation
pip install scipy>=1.11.0         # Statistical analysis
```

### Running Examples

```bash
# Run any example directly
python examples-physical-ai/01_realtime_safety_monitoring.py
python examples-physical-ai/02_multi_sensor_fusion.py
python examples-physical-ai/03_ros2_surgical_deployment.py
python examples-physical-ai/04_hand_eye_calibration.py
python examples-physical-ai/05_shared_autonomy_teleoperation.py
python examples-physical-ai/06_robotic_sample_handling.py
```

### Linting

```bash
ruff check examples-physical-ai/
ruff format --check examples-physical-ai/
```

## Examples

### 01 -- Real-Time Safety Monitoring (`01_realtime_safety_monitoring.py`)

Demonstrates IEC 80601-2-77 compliant real-time safety monitoring for surgical
robotic systems. Includes force/torque limit enforcement, workspace boundary
checking, emergency stop logic, and telemetry streaming with structured logging.

**Key classes:** `SafetyMonitor`, `WorkspaceBoundaryEnforcer`, `EmergencyStopController`,
`TelemetryStreamer`

### 02 -- Multi-Sensor Fusion (`02_multi_sensor_fusion.py`)

Multi-modal sensor fusion for surgical robotics combining force/torque sensors,
stereo cameras, joint encoders, and patient vital-sign monitors. Implements an
extended Kalman filter for state estimation and anomaly detection.

**Key classes:** `ExtendedKalmanFilter`, `SensorFusionPipeline`, `AnomalyDetector`,
`FusionOrchestrator`

### 03 -- ROS 2 Surgical Deployment (`03_ros2_surgical_deployment.py`)

ROS 2 deployment pattern for surgical robotic systems. Demonstrates node
lifecycle management, service discovery, action servers for procedure execution,
and health monitoring. All ROS 2 interfaces are mocked for portability.

**Key classes:** `MockNode`, `LifecycleManager`, `ProcedureActionServer`,
`SurgicalDeploymentOrchestrator`

### 04 -- Hand-Eye Calibration (`04_hand_eye_calibration.py`)

AX=XB hand-eye calibration for surgical robot systems. Includes transformation
chain validation, calibration error analysis with statistical hypothesis tests,
and multi-method comparison (Tsai-Lenz, Park-Martin, Dual Quaternion).

**Key classes:** `HandEyeCalibrator`, `TransformationChainValidator`,
`CalibrationErrorAnalyzer`, `CalibrationBenchmark`

### 05 -- Shared Autonomy Teleoperation (`05_shared_autonomy_teleoperation.py`)

Five levels of surgical autonomy from fully manual to fully autonomous, with
blending of operator input and AI guidance. Implements safety gate transitions,
authority arbitration, and performance monitoring.

**Key classes:** `AutonomyLevelManager`, `InputBlender`, `SafetyGateController`,
`SharedAutonomyController`

### 06 -- Robotic Sample Handling (`06_robotic_sample_handling.py`)

Automated specimen collection, transport, and processing workflow with
21 CFR Part 11 compliant audit trails, chain-of-custody tracking, and
electronic signature simulation.

**Key classes:** `AuditTrailManager`, `ChainOfCustody`, `ElectronicSignature`,
`SampleHandlingWorkflow`

## Hardware Requirements

All examples run in simulation mode by default. For physical deployment:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | Not required | NVIDIA with CUDA 12+ |
| Sensors | Simulated | Force/torque, stereo camera |
| Robot | Simulated | ROS 2-compatible manipulator |

## Disclaimer

**RESEARCH USE ONLY.** These examples are provided for research and educational
purposes. They are not validated for clinical use and must not be used in any
patient care setting without appropriate regulatory clearance and clinical
validation. See individual script headers for full disclaimers.
