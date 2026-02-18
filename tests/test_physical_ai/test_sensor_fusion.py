"""Tests for physical_ai/sensor_fusion.py.

Covers ClinicalSensorFusion creation, data ingestion, fuse() output,
anomaly detection, multi-modal integration, empty input handling,
buffer windowing, and reset behaviour.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("physical_ai.sensor_fusion", "physical_ai/sensor_fusion.py")

SensorReading = mod.SensorReading
ClinicalSensorFusion = mod.ClinicalSensorFusion


def _make_reading(sensor_id: str, sensor_type: str, timestamp: float, values: dict[str, float]) -> "SensorReading":
    """Helper to build a SensorReading."""
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        timestamp=timestamp,
        values=values,
    )


class TestClinicalSensorFusionCreation:
    """Tests for ClinicalSensorFusion initialization."""

    def test_default_sensor_configs(self):
        """Default configs include vitals, imaging, robot."""
        csf = ClinicalSensorFusion()
        assert "vitals" in csf.sensor_configs
        assert "imaging" in csf.sensor_configs
        assert "robot" in csf.sensor_configs

    def test_custom_sensor_configs(self):
        """Custom configs replace defaults."""
        custom = {"eeg": ["alpha", "beta"]}
        csf = ClinicalSensorFusion(sensor_configs=custom)
        assert "eeg" in csf.sensor_configs
        assert "vitals" not in csf.sensor_configs

    def test_default_anomaly_threshold(self):
        """Default anomaly threshold is 3.0."""
        csf = ClinicalSensorFusion()
        np.testing.assert_allclose(csf.anomaly_threshold, 3.0)

    def test_custom_anomaly_threshold(self):
        """Custom anomaly threshold is respected."""
        csf = ClinicalSensorFusion(anomaly_threshold=5.0)
        np.testing.assert_allclose(csf.anomaly_threshold, 5.0)


class TestIngestion:
    """Tests for sensor reading ingestion."""

    def test_ingest_known_sensor(self):
        """Ingesting a known sensor adds to its buffer."""
        csf = ClinicalSensorFusion()
        reading = _make_reading("vitals", "vital_signs", 1.0, {"heart_rate": 72.0})
        csf.ingest(reading)
        assert len(csf._buffers["vitals"]) == 1

    def test_ingest_unknown_sensor_creates_buffer(self):
        """Ingesting an unknown sensor_id creates a new buffer."""
        csf = ClinicalSensorFusion()
        reading = _make_reading("emg", "electromyography", 1.0, {"amplitude": 0.5})
        csf.ingest(reading)
        assert "emg" in csf._buffers
        assert len(csf._buffers["emg"]) == 1

    def test_window_size_limits_buffer(self):
        """Buffer does not exceed window_size."""
        csf = ClinicalSensorFusion(window_size=5)
        for i in range(10):
            reading = _make_reading("vitals", "vital_signs", float(i), {"heart_rate": 70.0 + i})
            csf.ingest(reading)
        assert len(csf._buffers["vitals"]) == 5


class TestFuse:
    """Tests for the fuse() method."""

    def test_fuse_empty_buffers(self):
        """Fuse with empty buffers returns zeros."""
        csf = ClinicalSensorFusion()
        fused = csf.fuse()
        np.testing.assert_allclose(fused, np.zeros(fused.shape))

    def test_fuse_output_dtype(self):
        """Fused vector is float64."""
        csf = ClinicalSensorFusion()
        fused = csf.fuse()
        assert fused.dtype == np.float64

    def test_fuse_with_data_produces_nonzero(self):
        """After ingesting data, fused vector has nonzero elements."""
        csf = ClinicalSensorFusion()
        reading = _make_reading("vitals", "vital_signs", 1.0, {"heart_rate": 80.0, "spo2": 98.0})
        csf.ingest(reading)
        fused = csf.fuse()
        assert np.any(fused != 0.0)

    def test_fuse_mean_and_std_pairs(self):
        """Fuse produces mean/std pairs: two values per key per sensor."""
        config = {"sensor_a": ["key1"]}
        csf = ClinicalSensorFusion(sensor_configs=config)
        csf.ingest(_make_reading("sensor_a", "test", 1.0, {"key1": 10.0}))
        csf.ingest(_make_reading("sensor_a", "test", 2.0, {"key1": 20.0}))
        fused = csf.fuse()
        assert fused.shape == (2,)
        np.testing.assert_allclose(fused[0], 15.0)  # mean of 10, 20
        np.testing.assert_allclose(fused[1], 5.0)  # std of 10, 20


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    def test_no_anomaly_with_few_readings(self):
        """Fewer than 5 readings means no anomaly detection."""
        csf = ClinicalSensorFusion()
        for i in range(3):
            csf.ingest(_make_reading("vitals", "vital_signs", float(i), {"heart_rate": 72.0}))
        anomalies = csf.detect_anomalies()
        assert anomalies == []

    def test_anomaly_detected_for_outlier(self):
        """A large outlier triggers anomaly detection."""
        config = {"sensor_a": ["val"]}
        csf = ClinicalSensorFusion(sensor_configs=config, anomaly_threshold=2.0)
        for i in range(10):
            csf.ingest(_make_reading("sensor_a", "test", float(i), {"val": 10.0}))
        # Inject extreme outlier
        csf.ingest(_make_reading("sensor_a", "test", 11.0, {"val": 1000.0}))
        anomalies = csf.detect_anomalies()
        assert len(anomalies) >= 1
        assert anomalies[0]["sensor_id"] == "sensor_a"

    def test_anomaly_log_accumulates(self):
        """Anomaly log grows with repeated detections."""
        config = {"sensor_a": ["val"]}
        csf = ClinicalSensorFusion(sensor_configs=config, anomaly_threshold=2.0)
        for i in range(10):
            csf.ingest(_make_reading("sensor_a", "test", float(i), {"val": 10.0}))
        csf.ingest(_make_reading("sensor_a", "test", 11.0, {"val": 1000.0}))
        csf.detect_anomalies()
        log = csf.get_anomaly_log()
        assert len(log) >= 1


class TestReset:
    """Tests for the reset() method."""

    def test_reset_clears_buffers_and_log(self):
        """Reset empties buffers and anomaly log."""
        csf = ClinicalSensorFusion()
        csf.ingest(_make_reading("vitals", "vital_signs", 1.0, {"heart_rate": 72.0}))
        csf.reset()
        assert all(len(buf) == 0 for buf in csf._buffers.values())
        assert csf.get_anomaly_log() == []
