"""Tests for physical AI examples.

Validates dataclass creation, configuration validation, enum values,
and sensor/safety types from examples-physical-ai/ modules.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module


# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
safety_mod = load_module(
    "realtime_safety_monitoring",
    "examples-physical-ai/01_realtime_safety_monitoring.py",
)

fusion_mod = load_module(
    "multi_sensor_fusion",
    "examples-physical-ai/02_multi_sensor_fusion.py",
)


class TestSafetyMonitoringEnums:
    """Verify safety monitoring enums from 01_realtime_safety_monitoring.py."""

    def test_safety_level_values(self):
        assert safety_mod.SafetyLevel.ADVISORY.value == "advisory"
        assert safety_mod.SafetyLevel.WARNING.value == "warning"
        assert safety_mod.SafetyLevel.CRITICAL.value == "critical"

    def test_monitoring_state_values(self):
        assert safety_mod.MonitoringState.IDLE.value == "idle"
        assert safety_mod.MonitoringState.MONITORING.value == "monitoring"
        assert safety_mod.MonitoringState.EMERGENCY_STOP.value == "emergency_stop"

    def test_surgical_phase_values(self):
        assert safety_mod.SurgicalPhase.APPROACH.value == "approach"
        assert safety_mod.SurgicalPhase.CONTACT.value == "contact"
        assert safety_mod.SurgicalPhase.CUTTING.value == "cutting"
        assert safety_mod.SurgicalPhase.IDLE.value == "idle"

    def test_estop_source_values(self):
        assert safety_mod.EStopSource.OPERATOR.value == "operator"
        assert safety_mod.EStopSource.FORCE_LIMIT.value == "force_limit"
        assert safety_mod.EStopSource.SOFTWARE_FAULT.value == "software_fault"


class TestSafetyConfig:
    """Verify SafetyConfig dataclass from safety monitoring module."""

    def test_safety_config_defaults(self):
        cfg = safety_mod.SafetyConfig()
        assert cfg.max_force_n > 0
        assert cfg.monitoring_frequency_hz > 0

    def test_safety_config_has_workspace_fields(self):
        cfg = safety_mod.SafetyConfig()
        assert hasattr(cfg, "workspace_min_mm")
        assert hasattr(cfg, "workspace_max_mm")


class TestTelemetryFrame:
    """Verify TelemetryFrame dataclass from safety monitoring module."""

    def test_telemetry_frame_creation(self):
        frame = safety_mod.TelemetryFrame()
        assert hasattr(frame, "timestamp")
        assert hasattr(frame, "frame_id")

    def test_telemetry_frame_with_custom_id(self):
        f1 = safety_mod.TelemetryFrame(frame_id=1)
        f2 = safety_mod.TelemetryFrame(frame_id=2)
        assert f1.frame_id != f2.frame_id


class TestSensorFusionEnums:
    """Verify sensor fusion enums from 02_multi_sensor_fusion.py."""

    def test_sensor_type_values(self):
        assert fusion_mod.SensorType.FORCE_TORQUE.value == "force_torque"
        assert fusion_mod.SensorType.STEREO_CAMERA.value == "stereo_camera"
        assert fusion_mod.SensorType.JOINT_ENCODER.value == "joint_encoder"

    def test_sensor_health_values(self):
        assert fusion_mod.SensorHealth.NOMINAL.value == "nominal"
        assert fusion_mod.SensorHealth.DEGRADED.value == "degraded"
        assert fusion_mod.SensorHealth.FAILED.value == "failed"

    def test_fusion_mode_values(self):
        assert fusion_mod.FusionMode.FULL.value == "full"
        assert fusion_mod.FusionMode.DEGRADED.value == "degraded"
        assert fusion_mod.FusionMode.EMERGENCY.value == "emergency"

    def test_anomaly_type_values(self):
        assert fusion_mod.AnomalyType.FORCE_SPIKE.value == "force_spike"
        assert fusion_mod.AnomalyType.VITAL_SIGN_ALERT.value == "vital_sign_alert"
        assert fusion_mod.AnomalyType.POSITION_JUMP.value == "position_jump"


class TestSensorConfig:
    """Verify SensorConfig dataclass from sensor fusion module."""

    def test_sensor_config_creation(self):
        cfg = fusion_mod.SensorConfig()
        assert hasattr(cfg, "sensor_id")
        assert hasattr(cfg, "sensor_type")

    def test_sensor_config_custom(self):
        cfg = fusion_mod.SensorConfig(
            sensor_id="sensor-01",
            sensor_type=fusion_mod.SensorType.FORCE_TORQUE,
        )
        assert cfg.sensor_id == "sensor-01"


class TestAnomalyRecord:
    """Verify AnomalyRecord dataclass from sensor fusion module."""

    def test_anomaly_record_creation(self):
        rec = fusion_mod.AnomalyRecord(
            anomaly_type=fusion_mod.AnomalyType.FORCE_SPIKE,
            severity=0.7,
            description="Test spike",
        )
        assert rec.anomaly_type == fusion_mod.AnomalyType.FORCE_SPIKE
        assert rec.severity == 0.7
        assert rec.description == "Test spike"
