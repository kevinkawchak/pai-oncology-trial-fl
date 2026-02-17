"""Clinical sensor fusion for real-time monitoring during trials.

Fuses data from multiple clinical sensors (vitals, imaging, robotic
telemetry) into unified feature representations that can be used by
federated models for anomaly detection and decision support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """A single reading from a clinical sensor.

    Attributes:
        sensor_id: Unique sensor identifier.
        sensor_type: Category (e.g. ``"vital_signs"``, ``"imaging"``).
        timestamp: Unix timestamp of the reading.
        values: Dictionary of measurement name to numeric value.
    """

    sensor_id: str
    sensor_type: str
    timestamp: float
    values: dict[str, float] = field(default_factory=dict)


class ClinicalSensorFusion:
    """Fuses multi-modal clinical sensor streams.

    Designed for real-time monitoring during oncology procedures,
    combining vital signs, imaging metrics, and robotic telemetry
    into a single feature vector for federated model inference.

    Args:
        sensor_configs: Mapping of sensor_id to expected value keys.
        anomaly_threshold: Z-score threshold for anomaly detection.
        window_size: Number of readings to keep per sensor for
            running statistics.
    """

    def __init__(
        self,
        sensor_configs: dict[str, list[str]] | None = None,
        anomaly_threshold: float = 3.0,
        window_size: int = 100,
    ):
        self.sensor_configs = sensor_configs or {
            "vitals": ["heart_rate", "blood_pressure_systolic", "spo2", "temperature"],
            "imaging": ["contrast_intensity", "tumor_margin_mm"],
            "robot": ["force_n", "velocity_mm_s"],
        }
        self.anomaly_threshold = anomaly_threshold
        self.window_size = window_size
        self._buffers: dict[str, list[SensorReading]] = {sid: [] for sid in self.sensor_configs}
        self._anomaly_log: list[dict] = []

    def ingest(self, reading: SensorReading) -> None:
        """Add a sensor reading to the appropriate buffer.

        Args:
            reading: The sensor reading to ingest.
        """
        sid = reading.sensor_id
        if sid not in self._buffers:
            self._buffers[sid] = []
        buf = self._buffers[sid]
        buf.append(reading)
        if len(buf) > self.window_size:
            buf.pop(0)

    def fuse(self) -> np.ndarray:
        """Produce a fused feature vector from all sensor streams.

        Returns a fixed-size vector by computing mean and std for each
        expected value key across all sensors.

        Returns:
            Numpy array of fused features.
        """
        features: list[float] = []
        for sid, keys in self.sensor_configs.items():
            buf = self._buffers.get(sid, [])
            for key in keys:
                vals = [r.values.get(key, 0.0) for r in buf]
                if vals:
                    features.append(float(np.mean(vals)))
                    features.append(float(np.std(vals)))
                else:
                    features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float64)

    def detect_anomalies(self) -> list[dict]:
        """Check recent readings for anomalous values.

        Uses a Z-score approach against the running window.

        Returns:
            List of anomaly records with sensor, key, value, and z-score.
        """
        anomalies: list[dict] = []
        for sid, keys in self.sensor_configs.items():
            buf = self._buffers.get(sid, [])
            if len(buf) < 5:
                continue
            for key in keys:
                vals = np.array([r.values.get(key, 0.0) for r in buf])
                mean = np.mean(vals)
                std = np.std(vals)
                if std < 1e-8:
                    continue
                latest = vals[-1]
                z = abs(latest - mean) / std
                if z > self.anomaly_threshold:
                    record = {
                        "sensor_id": sid,
                        "key": key,
                        "value": float(latest),
                        "z_score": float(z),
                        "mean": float(mean),
                        "std": float(std),
                    }
                    anomalies.append(record)
                    self._anomaly_log.append(record)
                    logger.warning(
                        "Anomaly detected: %s/%s = %.2f (z=%.2f)",
                        sid,
                        key,
                        latest,
                        z,
                    )
        return anomalies

    def get_anomaly_log(self) -> list[dict]:
        """Return all detected anomalies."""
        return list(self._anomaly_log)

    def reset(self) -> None:
        """Clear all sensor buffers and anomaly logs."""
        for sid in self._buffers:
            self._buffers[sid] = []
        self._anomaly_log = []
