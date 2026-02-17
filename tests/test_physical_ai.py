"""Tests for physical AI modules."""

import numpy as np
import pytest

from physical_ai.digital_twin import PatientDigitalTwin, TumorModel
from physical_ai.robotic_integration import (
    RobotType,
    SurgicalRobotInterface,
    TaskStatus,
)
from physical_ai.sensor_fusion import ClinicalSensorFusion, SensorReading
from physical_ai.simulation_bridge import (
    ModelFormat,
    RobotModel,
    SimulationBridge,
)


class TestTumorModel:
    def test_simulate_growth(self):
        tumor = TumorModel(volume_cm3=1.0, growth_rate=60.0)
        # After one doubling time, volume should approximately double
        vol = tumor.simulate_growth(60)
        assert abs(vol - 2.0) < 0.01

    def test_chemo_response(self):
        tumor = TumorModel(volume_cm3=5.0, chemo_sensitivity=0.5)
        vol = tumor.simulate_chemo_response(dose_mg=100, cycles=4)
        assert vol < 5.0  # Should shrink

    def test_radiation_response(self):
        tumor = TumorModel(volume_cm3=5.0, radio_sensitivity=0.6)
        vol = tumor.simulate_radiation_response(dose_gy=60, fractions=30)
        assert vol < 5.0  # Should shrink


class TestPatientDigitalTwin:
    def test_simulate_chemotherapy(self):
        twin = PatientDigitalTwin("patient_001")
        result = twin.simulate_treatment("chemotherapy", dose_mg=75, cycles=4)
        assert "predicted_volume_cm3" in result
        assert "response_category" in result

    def test_simulate_radiation(self):
        twin = PatientDigitalTwin("patient_002")
        result = twin.simulate_treatment("radiation", dose_gy=60, fractions=30)
        assert result["predicted_volume_cm3"] < result["initial_volume_cm3"]

    def test_simulate_immunotherapy(self):
        twin = PatientDigitalTwin("patient_003")
        result = twin.simulate_treatment("immunotherapy", response_rate=0.5)
        assert "response_category" in result

    def test_unknown_treatment_raises(self):
        twin = PatientDigitalTwin("patient_004")
        with pytest.raises(ValueError, match="Unknown treatment"):
            twin.simulate_treatment("unknown_therapy")

    def test_feature_vector(self):
        twin = PatientDigitalTwin(
            "patient_005",
            biomarkers={"pdl1": 0.7, "ki67": 0.3},
        )
        features = twin.generate_feature_vector()
        assert features.shape == (30,)
        assert features[0] == twin.tumor.volume_cm3

    def test_summary(self):
        twin = PatientDigitalTwin("patient_006")
        summary = twin.get_summary()
        assert summary["patient_id"] == "patient_006"
        assert "tumor_volume_cm3" in summary


class TestSurgicalRobotInterface:
    def test_connect_disconnect(self):
        robot = SurgicalRobotInterface("robot_01", RobotType.DA_VINCI)
        assert robot.connect() is True
        robot.disconnect()
        assert not robot._is_connected

    def test_plan_procedure(self):
        robot = SurgicalRobotInterface("robot_01")
        task = robot.plan_procedure("biopsy", np.array([1.0, 2.0, 3.0]))
        assert task.task_type == "biopsy"
        assert task.status == TaskStatus.PENDING

    def test_unsupported_task_raises(self):
        robot = SurgicalRobotInterface("robot_01", capabilities=["biopsy"])
        with pytest.raises(ValueError, match="not supported"):
            robot.plan_procedure("unsupported", np.array([0, 0, 0]))

    def test_execute_task(self):
        robot = SurgicalRobotInterface("robot_01")
        robot.connect()
        task = robot.plan_procedure("biopsy", np.array([1.0, 2.0, 3.0]))
        result = robot.execute_task(task, seed=42)
        assert result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        assert len(result.telemetry) == 10

    def test_execute_without_connect(self):
        robot = SurgicalRobotInterface("robot_01")
        task = robot.plan_procedure("biopsy", np.array([0, 0, 0]))
        result = robot.execute_task(task)
        assert result.status == TaskStatus.FAILED

    def test_telemetry_features(self):
        robot = SurgicalRobotInterface("robot_01")
        robot.connect()
        task = robot.plan_procedure("resection", np.array([1, 2, 3]), tumor_volume=3.5)
        task = robot.execute_task(task, seed=0)
        features = robot.get_telemetry_features(task)
        assert features.shape == (10,)


class TestClinicalSensorFusion:
    def test_ingest_and_fuse(self):
        fusion = ClinicalSensorFusion()
        for i in range(10):
            reading = SensorReading(
                sensor_id="vitals",
                sensor_type="vital_signs",
                timestamp=float(i),
                values={
                    "heart_rate": 70 + np.random.randn(),
                    "blood_pressure_systolic": 120 + np.random.randn(),
                    "spo2": 98 + np.random.randn() * 0.5,
                    "temperature": 37 + np.random.randn() * 0.1,
                },
            )
            fusion.ingest(reading)

        features = fusion.fuse()
        assert len(features) > 0
        assert not np.all(features == 0)

    def test_anomaly_detection(self):
        fusion = ClinicalSensorFusion(anomaly_threshold=2.0)
        # Normal readings
        for i in range(20):
            fusion.ingest(
                SensorReading(
                    sensor_id="vitals",
                    sensor_type="vital_signs",
                    timestamp=float(i),
                    values={"heart_rate": 70.0},
                )
            )
        # Anomalous reading
        fusion.ingest(
            SensorReading(
                sensor_id="vitals",
                sensor_type="vital_signs",
                timestamp=21.0,
                values={"heart_rate": 200.0},
            )
        )
        anomalies = fusion.detect_anomalies()
        assert len(anomalies) > 0

    def test_reset(self):
        fusion = ClinicalSensorFusion()
        fusion.ingest(SensorReading("vitals", "vital_signs", 0.0, {"heart_rate": 70}))
        fusion.reset()
        features = fusion.fuse()
        assert np.all(features == 0)


class TestSimulationBridge:
    def test_register_and_list(self):
        bridge = SimulationBridge()
        model = RobotModel(
            name="davinci_v1",
            source_format=ModelFormat.URDF,
            num_joints=7,
            num_links=8,
            mass_kg=15.0,
        )
        bridge.register_model(model)
        assert "davinci_v1" in bridge.list_models()

    def test_convert_urdf_to_mjcf(self):
        bridge = SimulationBridge()
        model = RobotModel(
            name="test_robot",
            source_format=ModelFormat.URDF,
            num_joints=6,
            num_links=7,
            mass_kg=10.0,
            joint_limits=[(-1.0, 1.0)] * 6,
        )
        converted = bridge.convert(model, ModelFormat.MJCF)
        assert converted.source_format == ModelFormat.MJCF

    def test_convert_same_format(self):
        bridge = SimulationBridge()
        model = RobotModel(name="test", source_format=ModelFormat.URDF)
        result = bridge.convert(model, ModelFormat.URDF)
        assert result is model

    def test_unsupported_conversion(self):
        bridge = SimulationBridge()
        model = RobotModel(name="test", source_format=ModelFormat.USD)
        with pytest.raises(ValueError, match="No conversion path"):
            bridge.convert(model, ModelFormat.SDF)

    def test_validate_conversion(self):
        bridge = SimulationBridge()
        original = RobotModel(
            name="orig",
            source_format=ModelFormat.URDF,
            num_joints=6,
            num_links=7,
            mass_kg=10.0,
            joint_limits=[(-1.0, 1.0)] * 6,
        )
        converted = bridge.convert(original, ModelFormat.MJCF)
        results = bridge.validate_conversion(original, converted)
        assert results["joints_match"]
        assert results["links_match"]
        assert results["mass_match"]
        assert results["joint_limits_match"]
