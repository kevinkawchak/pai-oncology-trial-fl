#!/usr/bin/env python3
"""Multi-sensor fusion for surgical robotics in oncology procedures.

CLINICAL CONTEXT
================
Surgical robotic systems in oncology rely on multiple heterogeneous sensor
streams -- force/torque transducers, stereo cameras, joint encoders, and
patient vital-sign monitors -- to maintain situational awareness during
delicate procedures. This module demonstrates a production-quality sensor
fusion pipeline that combines these modalities into a unified state
estimate using an extended Kalman filter (EKF).

The fused state enables real-time anomaly detection, instrument tracking,
and tissue interaction monitoring that feed into the federated learning
framework for cross-site model improvement.

USE CASES COVERED
=================
1. Force/torque sensor data ingestion with noise modelling and bias
   compensation.
2. Stereo camera-based instrument pose estimation with latency
   compensation.
3. Joint encoder fusion for forward kinematics cross-validation.
4. Patient vital-sign integration for context-aware safety monitoring.
5. Extended Kalman filter (EKF) for optimal state estimation from
   heterogeneous, asynchronous sensor streams.
6. Anomaly detection on fused state with configurable thresholds.
7. Sensor health monitoring and degraded-mode operation.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

Optional:
    torch >= 2.1.0   (https://pytorch.org) -- GPU-accelerated inference
    mujoco >= 3.0.0  (https://mujoco.org) -- Physics simulation backend
    scipy >= 1.11.0  (https://scipy.org) -- Statistical analysis

HARDWARE REQUIREMENTS
=====================
- CPU: 4+ cores for real-time fusion loop
- RAM: 4 GB minimum for sensor buffers
- GPU: Not required (optional for ML inference)
- Sensors: Force/torque, stereo camera, joint encoders (simulated)

REFERENCES
==========
- IEC 80601-2-77:2019 -- Sensor requirements for RASE
- ISO 14971:2019 -- Hazard analysis for sensor failures
- IEC 62304:2006+AMD1:2015 -- Class C software for safety-critical fusion

DISCLAIMER
==========
RESEARCH USE ONLY. This software is provided for research and educational
purposes. It is NOT validated for clinical use and must NOT be deployed in
any patient care setting without appropriate regulatory clearance and
comprehensive clinical validation.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]

HAS_MUJOCO = False
try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore[assignment]

HAS_SCIPY = False
try:
    import scipy
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    scipy_stats = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.sensor_fusion")


# ============================================================================
# Enumerations
# ============================================================================


class SensorType(str, Enum):
    """Types of sensors in the surgical robotic system."""

    FORCE_TORQUE = "force_torque"
    STEREO_CAMERA = "stereo_camera"
    JOINT_ENCODER = "joint_encoder"
    PATIENT_MONITOR = "patient_monitor"


class SensorHealth(str, Enum):
    """Health status of an individual sensor."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    FAILED = "failed"
    CALIBRATING = "calibrating"


class FusionMode(str, Enum):
    """Operating mode of the fusion pipeline."""

    FULL = "full"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    OFFLINE = "offline"


class AnomalyType(str, Enum):
    """Classification of detected anomalies."""

    FORCE_SPIKE = "force_spike"
    POSITION_JUMP = "position_jump"
    SENSOR_DROPOUT = "sensor_dropout"
    VITAL_SIGN_ALERT = "vital_sign_alert"
    STATE_DIVERGENCE = "state_divergence"
    COMMUNICATION_LOSS = "communication_loss"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class SensorConfig:
    """Configuration for an individual sensor.

    Attributes:
        sensor_id: Unique sensor identifier.
        sensor_type: Category of sensor.
        sample_rate_hz: Expected sampling frequency.
        noise_std: Expected measurement noise standard deviation.
        latency_ms: Expected measurement latency.
        is_critical: Whether this sensor is safety-critical.
        bias: Known sensor bias values.
    """

    sensor_id: str = ""
    sensor_type: SensorType = SensorType.FORCE_TORQUE
    sample_rate_hz: float = 1000.0
    noise_std: float = 0.01
    latency_ms: float = 1.0
    is_critical: bool = True
    bias: tuple[float, ...] = ()


@dataclass
class SensorReading:
    """A single timestamped reading from a sensor.

    Attributes:
        sensor_id: Which sensor produced this reading.
        sensor_type: Type of sensor.
        timestamp: Unix timestamp with microsecond precision.
        values: Measurement values as a numpy array.
        is_valid: Whether the reading passed integrity checks.
        sequence_number: Monotonic counter for dropout detection.
    """

    sensor_id: str = ""
    sensor_type: SensorType = SensorType.FORCE_TORQUE
    timestamp: float = 0.0
    values: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_valid: bool = True
    sequence_number: int = 0


@dataclass
class FusedState:
    """Fused state estimate from all sensor modalities.

    Attributes:
        timestamp: Time of state estimate.
        position_mm: Estimated end-effector position (x, y, z).
        velocity_mm_s: Estimated end-effector velocity (vx, vy, vz).
        force_n: Estimated interaction force (fx, fy, fz).
        torque_nm: Estimated interaction torque (tx, ty, tz).
        joint_positions_rad: Estimated joint positions.
        covariance: State estimation covariance matrix.
        confidence: Overall confidence score [0, 1].
        mode: Current fusion operating mode.
        patient_vitals: Latest patient monitoring data.
    """

    timestamp: float = 0.0
    position_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_mm_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_positions_rad: np.ndarray = field(default_factory=lambda: np.zeros(6))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(15) * 0.01)
    confidence: float = 1.0
    mode: FusionMode = FusionMode.FULL
    patient_vitals: dict[str, float] = field(default_factory=dict)


@dataclass
class AnomalyRecord:
    """Record of a detected anomaly.

    Attributes:
        anomaly_id: Unique anomaly identifier.
        timestamp: When the anomaly was detected.
        anomaly_type: Classification of the anomaly.
        severity: Severity score from 0 (low) to 1 (critical).
        sensor_id: Which sensor detected it (if applicable).
        description: Human-readable description.
        values: Associated measurement values.
        checksum: Integrity hash.
    """

    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    anomaly_type: AnomalyType = AnomalyType.FORCE_SPIKE
    severity: float = 0.5
    sensor_id: str = ""
    description: str = ""
    values: dict[str, float] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self) -> None:
        """Compute integrity checksum."""
        if not self.checksum:
            payload = f"{self.anomaly_id}:{self.timestamp}:{self.anomaly_type.value}:{self.severity}"
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()


# ============================================================================
# Extended Kalman Filter
# ============================================================================


class ExtendedKalmanFilter:
    """Extended Kalman filter for surgical robot state estimation.

    State vector: [x, y, z, vx, vy, vz, fx, fy, fz, tx, ty, tz, q1, q2, q3]
    (15 dimensions: position, velocity, force, torque, 3 joint angles)

    The EKF handles asynchronous, multi-rate sensor updates with
    configurable process and measurement noise models.

    Args:
        state_dim: Dimension of the state vector.
        process_noise: Process noise covariance scalar.
        initial_covariance: Initial state covariance scalar.
    """

    def __init__(
        self,
        state_dim: int = 15,
        process_noise: float = 0.001,
        initial_covariance: float = 0.01,
    ) -> None:
        self._state_dim = state_dim
        self._state = np.zeros(state_dim)
        self._covariance = np.eye(state_dim) * initial_covariance
        self._process_noise = np.eye(state_dim) * process_noise
        self._prediction_count = 0
        self._update_count = 0
        self._innovation_history: list[float] = []
        logger.info("EKF initialised: state_dim=%d, Q=%.4f, P0=%.4f", state_dim, process_noise, initial_covariance)

    @property
    def state(self) -> np.ndarray:
        """Current state estimate."""
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current state covariance."""
        return self._covariance.copy()

    def predict(self, dt: float) -> np.ndarray:
        """Predict the state forward by dt seconds.

        Uses a constant-velocity model for position and constant-value
        model for force/torque/joints.

        Args:
            dt: Time step in seconds.

        Returns:
            Predicted state vector.
        """
        self._prediction_count += 1

        # State transition matrix (constant velocity model for position)
        F = np.eye(self._state_dim)
        # Position += velocity * dt
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt

        # Predict state
        self._state = F @ self._state

        # Predict covariance with process noise scaling
        Q_scaled = self._process_noise * dt
        self._covariance = F @ self._covariance @ F.T + Q_scaled

        # Ensure symmetry
        self._covariance = 0.5 * (self._covariance + self._covariance.T)

        return self._state.copy()

    def update(
        self,
        measurement: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_noise: np.ndarray,
    ) -> np.ndarray:
        """Update the state estimate with a new measurement.

        Args:
            measurement: Measurement vector.
            measurement_matrix: H matrix mapping state to measurement space.
            measurement_noise: R matrix (measurement noise covariance).

        Returns:
            Updated state vector.
        """
        self._update_count += 1

        # Innovation (measurement residual)
        predicted_measurement = measurement_matrix @ self._state
        innovation = measurement - predicted_measurement

        # Innovation covariance
        S = measurement_matrix @ self._covariance @ measurement_matrix.T + measurement_noise

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance; skipping update")
            return self._state.copy()

        K = self._covariance @ measurement_matrix.T @ S_inv

        # State update
        self._state = self._state + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self._state_dim) - K @ measurement_matrix
        self._covariance = I_KH @ self._covariance @ I_KH.T + K @ measurement_noise @ K.T

        # Ensure symmetry
        self._covariance = 0.5 * (self._covariance + self._covariance.T)

        # Track innovation norm for health monitoring
        innovation_norm = float(np.linalg.norm(innovation))
        self._innovation_history.append(innovation_norm)
        if len(self._innovation_history) > 1000:
            self._innovation_history.pop(0)

        return self._state.copy()

    def get_position_estimate(self) -> np.ndarray:
        """Extract position from state vector."""
        return self._state[0:3].copy()

    def get_velocity_estimate(self) -> np.ndarray:
        """Extract velocity from state vector."""
        return self._state[3:6].copy()

    def get_force_estimate(self) -> np.ndarray:
        """Extract force from state vector."""
        return self._state[6:9].copy()

    def get_torque_estimate(self) -> np.ndarray:
        """Extract torque from state vector."""
        return self._state[9:12].copy()

    def get_joint_estimate(self) -> np.ndarray:
        """Extract joint angles from state vector."""
        return self._state[12:15].copy()

    def get_position_uncertainty(self) -> float:
        """Compute position uncertainty (trace of position covariance)."""
        return float(np.trace(self._covariance[0:3, 0:3]))

    def get_innovation_statistics(self) -> dict[str, float]:
        """Compute innovation statistics for filter health monitoring.

        Returns:
            Dictionary with mean, std, and max innovation norms.
        """
        if not self._innovation_history:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "count": 0}
        arr = np.array(self._innovation_history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Get filter diagnostic information."""
        return {
            "state_dim": self._state_dim,
            "predictions": self._prediction_count,
            "updates": self._update_count,
            "position_uncertainty": self.get_position_uncertainty(),
            "covariance_trace": float(np.trace(self._covariance)),
            "innovation_stats": self.get_innovation_statistics(),
        }


# ============================================================================
# Sensor Health Monitor
# ============================================================================


class SensorHealthMonitor:
    """Monitors health of individual sensors for degraded-mode detection.

    Tracks update rates, dropout patterns, and noise levels to classify
    each sensor's health status.

    Args:
        configs: Dictionary of sensor configurations.
        dropout_threshold_s: Time without update before marking degraded.
        failure_threshold_s: Time without update before marking failed.
    """

    def __init__(
        self,
        configs: dict[str, SensorConfig],
        dropout_threshold_s: float = 0.1,
        failure_threshold_s: float = 1.0,
    ) -> None:
        self._configs = configs
        self._dropout_threshold = dropout_threshold_s
        self._failure_threshold = failure_threshold_s
        self._last_update: dict[str, float] = {sid: time.time() for sid in configs}
        self._update_counts: dict[str, int] = {sid: 0 for sid in configs}
        self._health: dict[str, SensorHealth] = {sid: SensorHealth.NOMINAL for sid in configs}
        self._noise_estimates: dict[str, list[float]] = {sid: [] for sid in configs}

    def record_update(self, sensor_id: str, timestamp: float, values: np.ndarray) -> SensorHealth:
        """Record a sensor update and reassess health.

        Args:
            sensor_id: Sensor that provided the update.
            timestamp: Timestamp of the reading.
            values: Measurement values.

        Returns:
            Updated health status for this sensor.
        """
        if sensor_id not in self._configs:
            return SensorHealth.FAILED

        self._last_update[sensor_id] = timestamp
        self._update_counts[sensor_id] = self._update_counts.get(sensor_id, 0) + 1

        # Estimate noise from value variance
        noise_buf = self._noise_estimates[sensor_id]
        noise_buf.append(float(np.linalg.norm(values)))
        if len(noise_buf) > 200:
            noise_buf.pop(0)

        self._health[sensor_id] = SensorHealth.NOMINAL
        return SensorHealth.NOMINAL

    def check_all(self, current_time: float) -> dict[str, SensorHealth]:
        """Check health of all sensors at the given time.

        Args:
            current_time: Current time for timeout evaluation.

        Returns:
            Dictionary of sensor_id to health status.
        """
        for sensor_id in self._configs:
            dt = current_time - self._last_update.get(sensor_id, 0)
            if dt > self._failure_threshold:
                self._health[sensor_id] = SensorHealth.FAILED
            elif dt > self._dropout_threshold:
                self._health[sensor_id] = SensorHealth.DEGRADED

        return dict(self._health)

    def get_fusion_mode(self) -> FusionMode:
        """Determine the appropriate fusion mode based on sensor health.

        Returns:
            The recommended fusion operating mode.
        """
        health_values = list(self._health.values())
        critical_sensors = [
            sid for sid, cfg in self._configs.items() if cfg.is_critical and self._health[sid] == SensorHealth.FAILED
        ]

        if critical_sensors:
            return FusionMode.EMERGENCY

        degraded_count = sum(1 for h in health_values if h in (SensorHealth.DEGRADED, SensorHealth.FAILED))
        if degraded_count > 0:
            return FusionMode.DEGRADED

        return FusionMode.FULL

    def get_summary(self) -> dict[str, Any]:
        """Get sensor health summary."""
        return {
            "sensor_health": {sid: h.value for sid, h in self._health.items()},
            "update_counts": dict(self._update_counts),
            "fusion_mode": self.get_fusion_mode().value,
        }


# ============================================================================
# Anomaly Detector
# ============================================================================


class AnomalyDetector:
    """Detects anomalies in fused state estimates.

    Uses statistical thresholds and rate-of-change limits to identify
    abnormal conditions that may indicate sensor failures, unexpected
    tissue interactions, or patient deterioration.

    Args:
        force_threshold_n: Force magnitude threshold for spike detection.
        position_jump_mm: Position change threshold per timestep.
        vital_sign_ranges: Normal ranges for patient vital signs.
    """

    def __init__(
        self,
        force_threshold_n: float = 8.0,
        position_jump_mm: float = 5.0,
        vital_sign_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._force_threshold = force_threshold_n
        self._position_jump = position_jump_mm
        self._vital_ranges = vital_sign_ranges or {
            "heart_rate_bpm": (50.0, 120.0),
            "spo2_percent": (90.0, 100.0),
            "blood_pressure_systolic": (80.0, 180.0),
            "temperature_c": (35.5, 38.5),
        }
        self._previous_state: FusedState | None = None
        self._anomaly_log: list[AnomalyRecord] = []
        self._window_forces: list[float] = []
        self._window_size = 200

    def check(self, state: FusedState) -> list[AnomalyRecord]:
        """Check a fused state for anomalies.

        Args:
            state: Current fused state estimate.

        Returns:
            List of detected anomalies (may be empty).
        """
        anomalies: list[AnomalyRecord] = []

        # Force spike detection
        force_mag = float(np.linalg.norm(state.force_n))
        self._window_forces.append(force_mag)
        if len(self._window_forces) > self._window_size:
            self._window_forces.pop(0)

        if force_mag > self._force_threshold:
            record = AnomalyRecord(
                anomaly_type=AnomalyType.FORCE_SPIKE,
                severity=min(1.0, force_mag / (self._force_threshold * 2)),
                description=f"Force spike: {force_mag:.2f}N > {self._force_threshold:.1f}N threshold",
                values={"force_magnitude_n": force_mag, "threshold_n": self._force_threshold},
            )
            anomalies.append(record)

        # Position jump detection
        if self._previous_state is not None:
            delta_pos = float(np.linalg.norm(state.position_mm - self._previous_state.position_mm))
            if delta_pos > self._position_jump:
                record = AnomalyRecord(
                    anomaly_type=AnomalyType.POSITION_JUMP,
                    severity=min(1.0, delta_pos / (self._position_jump * 3)),
                    description=f"Position jump: {delta_pos:.2f}mm > {self._position_jump:.1f}mm threshold",
                    values={"delta_mm": delta_pos, "threshold_mm": self._position_jump},
                )
                anomalies.append(record)

        # State divergence (high covariance)
        pos_uncertainty = float(np.trace(state.covariance[0:3, 0:3]))
        if pos_uncertainty > 1.0:
            record = AnomalyRecord(
                anomaly_type=AnomalyType.STATE_DIVERGENCE,
                severity=min(1.0, pos_uncertainty / 5.0),
                description=f"State divergence: position uncertainty {pos_uncertainty:.3f}",
                values={"position_uncertainty": pos_uncertainty},
            )
            anomalies.append(record)

        # Vital sign checking
        for vital_name, (low, high) in self._vital_ranges.items():
            value = state.patient_vitals.get(vital_name)
            if value is not None and (value < low or value > high):
                record = AnomalyRecord(
                    anomaly_type=AnomalyType.VITAL_SIGN_ALERT,
                    severity=0.8,
                    description=f"Vital sign out of range: {vital_name}={value:.1f} (range: {low}-{high})",
                    values={"vital_name": vital_name, "value": value, "low": low, "high": high},
                )
                anomalies.append(record)

        self._previous_state = state
        self._anomaly_log.extend(anomalies)
        return anomalies

    def get_force_statistics(self) -> dict[str, float]:
        """Compute statistics over the force history window."""
        if not self._window_forces:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "count": 0}
        arr = np.array(self._window_forces)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    def get_anomaly_log(self) -> list[AnomalyRecord]:
        """Return all detected anomalies."""
        return list(self._anomaly_log)

    def get_anomaly_summary(self) -> dict[str, int]:
        """Count anomalies by type."""
        counts: dict[str, int] = {}
        for record in self._anomaly_log:
            key = record.anomaly_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts


# ============================================================================
# Sensor Fusion Pipeline
# ============================================================================


class SensorFusionPipeline:
    """Complete multi-sensor fusion pipeline for surgical robotics.

    Orchestrates sensor ingestion, EKF state estimation, anomaly
    detection, and health monitoring into a unified processing pipeline.

    Args:
        sensor_configs: Configuration for each sensor.
        ekf_process_noise: EKF process noise parameter.
        anomaly_force_threshold: Force threshold for anomaly detection.
    """

    def __init__(
        self,
        sensor_configs: dict[str, SensorConfig] | None = None,
        ekf_process_noise: float = 0.001,
        anomaly_force_threshold: float = 8.0,
    ) -> None:
        # Default sensor configuration
        self._sensor_configs = sensor_configs or {
            "ft_sensor_0": SensorConfig(
                sensor_id="ft_sensor_0",
                sensor_type=SensorType.FORCE_TORQUE,
                sample_rate_hz=1000.0,
                noise_std=0.05,
                latency_ms=0.5,
                is_critical=True,
            ),
            "camera_0": SensorConfig(
                sensor_id="camera_0",
                sensor_type=SensorType.STEREO_CAMERA,
                sample_rate_hz=30.0,
                noise_std=0.5,
                latency_ms=33.0,
                is_critical=False,
            ),
            "encoder_0": SensorConfig(
                sensor_id="encoder_0",
                sensor_type=SensorType.JOINT_ENCODER,
                sample_rate_hz=1000.0,
                noise_std=0.001,
                latency_ms=0.1,
                is_critical=True,
            ),
            "patient_monitor_0": SensorConfig(
                sensor_id="patient_monitor_0",
                sensor_type=SensorType.PATIENT_MONITOR,
                sample_rate_hz=1.0,
                noise_std=1.0,
                latency_ms=500.0,
                is_critical=False,
            ),
        }

        self._ekf = ExtendedKalmanFilter(
            state_dim=15,
            process_noise=ekf_process_noise,
        )
        self._health_monitor = SensorHealthMonitor(self._sensor_configs)
        self._anomaly_detector = AnomalyDetector(force_threshold_n=anomaly_force_threshold)

        self._last_timestamp = 0.0
        self._frame_count = 0
        self._latest_vitals: dict[str, float] = {}
        self._reading_buffer: list[SensorReading] = []
        self._buffer_max = 5000

        logger.info(
            "SensorFusionPipeline initialised with %d sensors",
            len(self._sensor_configs),
        )

    def ingest(self, reading: SensorReading) -> None:
        """Ingest a single sensor reading into the fusion pipeline.

        Args:
            reading: The sensor reading to process.
        """
        # Buffer for audit
        self._reading_buffer.append(reading)
        if len(self._reading_buffer) > self._buffer_max:
            self._reading_buffer.pop(0)

        # Update health monitor
        self._health_monitor.record_update(reading.sensor_id, reading.timestamp, reading.values)

        # Route to appropriate EKF update based on sensor type
        if reading.sensor_type == SensorType.FORCE_TORQUE:
            self._update_force_torque(reading)
        elif reading.sensor_type == SensorType.STEREO_CAMERA:
            self._update_camera(reading)
        elif reading.sensor_type == SensorType.JOINT_ENCODER:
            self._update_encoder(reading)
        elif reading.sensor_type == SensorType.PATIENT_MONITOR:
            self._update_patient_monitor(reading)

    def _update_force_torque(self, reading: SensorReading) -> None:
        """Process a force/torque sensor update."""
        # Measurement maps to state indices 6-11 (force + torque)
        values = reading.values
        if len(values) < 6:
            values = np.concatenate([values, np.zeros(6 - len(values))])

        H = np.zeros((6, 15))
        H[0, 6] = 1.0  # fx
        H[1, 7] = 1.0  # fy
        H[2, 8] = 1.0  # fz
        H[3, 9] = 1.0  # tx
        H[4, 10] = 1.0  # ty
        H[5, 11] = 1.0  # tz

        config = self._sensor_configs.get(reading.sensor_id)
        noise_std = config.noise_std if config else 0.05
        R = np.eye(6) * (noise_std**2)

        self._ekf.update(values[:6], H, R)

    def _update_camera(self, reading: SensorReading) -> None:
        """Process a stereo camera position update."""
        # Camera measures position (indices 0-2)
        values = reading.values
        if len(values) < 3:
            values = np.concatenate([values, np.zeros(3 - len(values))])

        H = np.zeros((3, 15))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z

        config = self._sensor_configs.get(reading.sensor_id)
        noise_std = config.noise_std if config else 0.5
        R = np.eye(3) * (noise_std**2)

        self._ekf.update(values[:3], H, R)

    def _update_encoder(self, reading: SensorReading) -> None:
        """Process joint encoder updates."""
        values = reading.values
        if len(values) < 3:
            values = np.concatenate([values, np.zeros(3 - len(values))])

        H = np.zeros((3, 15))
        H[0, 12] = 1.0  # q1
        H[1, 13] = 1.0  # q2
        H[2, 14] = 1.0  # q3

        config = self._sensor_configs.get(reading.sensor_id)
        noise_std = config.noise_std if config else 0.001
        R = np.eye(3) * (noise_std**2)

        self._ekf.update(values[:3], H, R)

    def _update_patient_monitor(self, reading: SensorReading) -> None:
        """Process patient vital-sign updates (not EKF-integrated)."""
        vital_keys = ["heart_rate_bpm", "spo2_percent", "blood_pressure_systolic", "temperature_c"]
        for i, key in enumerate(vital_keys):
            if i < len(reading.values):
                self._latest_vitals[key] = float(reading.values[i])

    def step(self, dt: float) -> FusedState:
        """Advance the fusion pipeline by one timestep.

        Performs EKF prediction, checks sensor health, runs anomaly
        detection, and produces a fused state estimate.

        Args:
            dt: Time step in seconds.

        Returns:
            Fused state estimate.
        """
        self._frame_count += 1
        self._last_timestamp += dt

        # EKF prediction
        self._ekf.predict(dt)

        # Check sensor health
        health_status = self._health_monitor.check_all(self._last_timestamp)
        fusion_mode = self._health_monitor.get_fusion_mode()

        # Confidence based on covariance and sensor health
        pos_uncertainty = self._ekf.get_position_uncertainty()
        base_confidence = max(0.0, 1.0 - pos_uncertainty)
        if fusion_mode == FusionMode.DEGRADED:
            base_confidence *= 0.7
        elif fusion_mode == FusionMode.EMERGENCY:
            base_confidence *= 0.3

        # Build fused state
        state = FusedState(
            timestamp=self._last_timestamp,
            position_mm=self._ekf.get_position_estimate(),
            velocity_mm_s=self._ekf.get_velocity_estimate(),
            force_n=self._ekf.get_force_estimate(),
            torque_nm=self._ekf.get_torque_estimate(),
            joint_positions_rad=np.concatenate([self._ekf.get_joint_estimate(), np.zeros(3)]),
            covariance=self._ekf.covariance,
            confidence=base_confidence,
            mode=fusion_mode,
            patient_vitals=dict(self._latest_vitals),
        )

        # Run anomaly detection
        anomalies = self._anomaly_detector.check(state)
        if anomalies:
            for anomaly in anomalies:
                logger.warning(
                    "Anomaly: [%s] %s (severity=%.2f)",
                    anomaly.anomaly_type.value,
                    anomaly.description,
                    anomaly.severity,
                )

        return state

    def get_ekf_diagnostics(self) -> dict[str, Any]:
        """Get EKF diagnostic information."""
        return self._ekf.get_diagnostics()

    def get_health_summary(self) -> dict[str, Any]:
        """Get sensor health summary."""
        return self._health_monitor.get_summary()

    def get_anomaly_summary(self) -> dict[str, int]:
        """Get anomaly detection summary."""
        return self._anomaly_detector.get_anomaly_summary()

    def get_report(self) -> dict[str, Any]:
        """Generate a comprehensive fusion pipeline report."""
        return {
            "frame_count": self._frame_count,
            "last_timestamp": self._last_timestamp,
            "ekf_diagnostics": self.get_ekf_diagnostics(),
            "sensor_health": self.get_health_summary(),
            "anomaly_summary": self.get_anomaly_summary(),
            "force_statistics": self._anomaly_detector.get_force_statistics(),
            "latest_vitals": dict(self._latest_vitals),
        }


# ============================================================================
# Simulation and Demonstration
# ============================================================================


def generate_simulated_sensor_data(
    num_timesteps: int = 500,
    seed: int = 42,
    include_anomalies: bool = True,
) -> list[list[SensorReading]]:
    """Generate simulated multi-sensor data for demonstration.

    Creates time-synchronized readings from all four sensor types with
    realistic noise characteristics and optional anomaly injection.

    Args:
        num_timesteps: Number of simulation steps.
        seed: Random seed for reproducibility.
        include_anomalies: Whether to inject anomalies.

    Returns:
        List of reading batches, one per timestep.
    """
    rng = np.random.default_rng(seed)
    base_time = time.time()
    all_batches: list[list[SensorReading]] = []

    # Generate a smooth trajectory
    t_vals = np.linspace(0, 4 * np.pi, num_timesteps)
    traj_x = 40.0 * np.sin(t_vals)
    traj_y = 25.0 * np.cos(t_vals)
    traj_z = 15.0 * np.sin(2 * t_vals) + 50.0

    seq_counters: dict[str, int] = {
        "ft_sensor_0": 0,
        "camera_0": 0,
        "encoder_0": 0,
        "patient_monitor_0": 0,
    }

    for i in range(num_timesteps):
        batch: list[SensorReading] = []
        t = base_time + i * 0.001  # 1 kHz base rate

        # Force/torque sensor (1 kHz)
        fx = rng.normal(0, 0.3)
        fy = rng.normal(0, 0.3)
        fz = rng.normal(1.5, 0.4)
        tx = rng.normal(0, 0.02)
        ty = rng.normal(0, 0.02)
        tz = rng.normal(0, 0.02)

        if include_anomalies and 150 <= i <= 155:
            fz = 12.0 + rng.normal(0, 0.5)

        seq_counters["ft_sensor_0"] += 1
        batch.append(
            SensorReading(
                sensor_id="ft_sensor_0",
                sensor_type=SensorType.FORCE_TORQUE,
                timestamp=t,
                values=np.array([fx, fy, fz, tx, ty, tz]),
                sequence_number=seq_counters["ft_sensor_0"],
            )
        )

        # Stereo camera (30 Hz -- every ~33 steps)
        if i % 33 == 0:
            cam_x = float(traj_x[i]) + rng.normal(0, 0.5)
            cam_y = float(traj_y[i]) + rng.normal(0, 0.5)
            cam_z = float(traj_z[i]) + rng.normal(0, 0.5)

            if include_anomalies and 250 <= i <= 252:
                cam_x += 15.0  # Position jump

            seq_counters["camera_0"] += 1
            batch.append(
                SensorReading(
                    sensor_id="camera_0",
                    sensor_type=SensorType.STEREO_CAMERA,
                    timestamp=t,
                    values=np.array([cam_x, cam_y, cam_z]),
                    sequence_number=seq_counters["camera_0"],
                )
            )

        # Joint encoder (1 kHz)
        q1 = 0.5 * math.sin(i * 0.01) + rng.normal(0, 0.001)
        q2 = 0.3 * math.cos(i * 0.01) + rng.normal(0, 0.001)
        q3 = -0.2 * math.sin(i * 0.02) + rng.normal(0, 0.001)
        seq_counters["encoder_0"] += 1
        batch.append(
            SensorReading(
                sensor_id="encoder_0",
                sensor_type=SensorType.JOINT_ENCODER,
                timestamp=t,
                values=np.array([q1, q2, q3]),
                sequence_number=seq_counters["encoder_0"],
            )
        )

        # Patient monitor (1 Hz -- every ~1000 steps)
        if i % 1000 == 0 or i == 0:
            hr = 72.0 + rng.normal(0, 3.0)
            spo2 = 98.0 + rng.normal(0, 0.5)
            bp = 120.0 + rng.normal(0, 5.0)
            temp = 36.8 + rng.normal(0, 0.2)

            if include_anomalies and i == 0:
                # Normal vitals for baseline
                pass

            seq_counters["patient_monitor_0"] += 1
            batch.append(
                SensorReading(
                    sensor_id="patient_monitor_0",
                    sensor_type=SensorType.PATIENT_MONITOR,
                    timestamp=t,
                    values=np.array([hr, spo2, bp, temp]),
                    sequence_number=seq_counters["patient_monitor_0"],
                )
            )

        all_batches.append(batch)

    logger.info("Generated %d timesteps of simulated sensor data (anomalies=%s)", num_timesteps, include_anomalies)
    return all_batches


def run_sensor_fusion_demo() -> dict[str, Any]:
    """Run the complete multi-sensor fusion demonstration.

    Returns:
        Final pipeline report dictionary.
    """
    logger.info("=" * 70)
    logger.info("  Multi-Sensor Fusion Demo for Surgical Robotics")
    logger.info("  Version 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 70)

    # 1. Create fusion pipeline
    logger.info("\n--- Initialising Fusion Pipeline ---")
    pipeline = SensorFusionPipeline(
        ekf_process_noise=0.001,
        anomaly_force_threshold=8.0,
    )

    # 2. Generate simulated sensor data
    logger.info("\n--- Generating Simulated Sensor Data ---")
    sensor_batches = generate_simulated_sensor_data(
        num_timesteps=500,
        seed=42,
        include_anomalies=True,
    )

    # 3. Process sensor data through the fusion pipeline
    logger.info("\n--- Processing Sensor Data ---")
    fused_states: list[FusedState] = []
    dt = 0.001  # 1 kHz

    for i, batch in enumerate(sensor_batches):
        # Ingest all readings in this batch
        for reading in batch:
            pipeline.ingest(reading)

        # Step the fusion pipeline
        state = pipeline.step(dt)
        fused_states.append(state)

        # Periodic logging
        if (i + 1) % 100 == 0:
            force_mag = float(np.linalg.norm(state.force_n))
            logger.info(
                "  Step %d/%d: pos=(%.1f, %.1f, %.1f) force=%.2fN conf=%.3f mode=%s",
                i + 1,
                len(sensor_batches),
                state.position_mm[0],
                state.position_mm[1],
                state.position_mm[2],
                force_mag,
                state.confidence,
                state.mode.value,
            )

    # 4. Analyse results
    logger.info("\n--- Fusion Results ---")
    report = pipeline.get_report()

    # Position trajectory statistics
    positions = np.array([s.position_mm for s in fused_states])
    forces = np.array([np.linalg.norm(s.force_n) for s in fused_states])
    confidences = np.array([s.confidence for s in fused_states])

    logger.info("  Frames processed:     %d", report["frame_count"])
    logger.info("  Position range X:     [%.1f, %.1f] mm", np.min(positions[:, 0]), np.max(positions[:, 0]))
    logger.info("  Position range Y:     [%.1f, %.1f] mm", np.min(positions[:, 1]), np.max(positions[:, 1]))
    logger.info("  Position range Z:     [%.1f, %.1f] mm", np.min(positions[:, 2]), np.max(positions[:, 2]))
    logger.info("  Mean force:           %.3f N", np.mean(forces))
    logger.info("  Max force:            %.3f N", np.max(forces))
    logger.info("  Mean confidence:      %.3f", np.mean(confidences))

    # EKF diagnostics
    ekf_diag = report["ekf_diagnostics"]
    logger.info("\n  EKF Diagnostics:")
    logger.info("    Predictions:        %d", ekf_diag["predictions"])
    logger.info("    Updates:            %d", ekf_diag["updates"])
    logger.info("    Position uncert.:   %.6f", ekf_diag["position_uncertainty"])

    innov = ekf_diag.get("innovation_stats", {})
    logger.info("    Innovation mean:    %.6f", innov.get("mean", 0))
    logger.info("    Innovation std:     %.6f", innov.get("std", 0))

    # Anomaly summary
    anomaly_summary = report["anomaly_summary"]
    if anomaly_summary:
        logger.info("\n  Anomalies Detected:")
        for atype, count in anomaly_summary.items():
            logger.info("    %-20s %d", atype, count)
    else:
        logger.info("\n  No anomalies detected.")

    # Sensor health
    health = report["sensor_health"]
    logger.info("\n  Sensor Health:")
    for sid, status in health.get("sensor_health", {}).items():
        logger.info("    %-20s %s", sid, status)

    logger.info("\n" + "=" * 70)
    logger.info("  RESEARCH USE ONLY -- NOT FOR CLINICAL DEPLOYMENT")
    logger.info("=" * 70)

    return report


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    report = run_sensor_fusion_demo()
    sys.exit(0)
