"""Tests for physical_ai/robotic_integration.py.

Covers SurgicalRobotInterface creation, connection lifecycle,
procedure planning, task execution, telemetry feature extraction,
task log reporting, and error paths.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("physical_ai.robotic_integration", "physical_ai/robotic_integration.py")

RobotType = mod.RobotType
TaskStatus = mod.TaskStatus
RoboticTask = mod.RoboticTask
SurgicalRobotInterface = mod.SurgicalRobotInterface


class TestRobotTypeEnum:
    """Tests for the RobotType enum."""

    def test_robot_type_values(self):
        """All expected robot types exist."""
        assert RobotType.DA_VINCI.value == "da_vinci"
        assert RobotType.ROBOTIC_ARM.value == "robotic_arm"
        assert RobotType.SPECIMEN_HANDLER.value == "specimen_handler"
        assert RobotType.GENERIC.value == "generic"


class TestTaskStatusEnum:
    """Tests for the TaskStatus enum."""

    def test_task_status_values(self):
        """All expected task statuses exist."""
        expected = {"pending", "running", "completed", "failed", "aborted"}
        actual = {s.value for s in TaskStatus}
        assert expected == actual


class TestSurgicalRobotInterfaceCreation:
    """Tests for SurgicalRobotInterface initialization."""

    def test_default_creation(self):
        """Default robot has generic type and standard capabilities."""
        robot = SurgicalRobotInterface(robot_id="R-001")
        assert robot.robot_id == "R-001"
        assert robot.robot_type == RobotType.GENERIC
        assert "biopsy" in robot.capabilities

    def test_custom_type_and_capabilities(self):
        """Custom robot type and capabilities are stored."""
        robot = SurgicalRobotInterface(
            robot_id="R-002",
            robot_type=RobotType.DA_VINCI,
            capabilities=["resection"],
        )
        assert robot.robot_type == RobotType.DA_VINCI
        assert robot.capabilities == ["resection"]

    def test_initially_disconnected(self):
        """Robot starts disconnected."""
        robot = SurgicalRobotInterface(robot_id="R-003")
        assert robot._is_connected is False


class TestConnectionLifecycle:
    """Tests for connect/disconnect methods."""

    def test_connect_returns_true(self):
        """Connect returns True."""
        robot = SurgicalRobotInterface(robot_id="R-004")
        assert robot.connect() is True
        assert robot._is_connected is True

    def test_disconnect(self):
        """Disconnect sets _is_connected to False."""
        robot = SurgicalRobotInterface(robot_id="R-005")
        robot.connect()
        robot.disconnect()
        assert robot._is_connected is False


class TestProcedurePlanning:
    """Tests for plan_procedure."""

    def test_plan_valid_procedure(self):
        """Planning a valid procedure returns a RoboticTask."""
        robot = SurgicalRobotInterface(robot_id="R-006")
        target = np.array([10.0, 20.0, 5.0])
        task = robot.plan_procedure("biopsy", target)
        assert isinstance(task, RoboticTask)
        assert task.task_type == "biopsy"
        assert task.status == TaskStatus.PENDING
        np.testing.assert_allclose(task.target_location, target)

    def test_plan_unsupported_procedure_raises(self):
        """Planning an unsupported procedure raises ValueError."""
        robot = SurgicalRobotInterface(robot_id="R-007")
        with pytest.raises(ValueError, match="not supported"):
            robot.plan_procedure("laser_ablation", np.zeros(3))

    def test_plan_includes_safety_constraints(self):
        """Planned task has safety constraints."""
        robot = SurgicalRobotInterface(robot_id="R-008")
        task = robot.plan_procedure("biopsy", np.zeros(3), tumor_volume=5.0, safety_margin_mm=10.0)
        assert task.constraints["safety_margin_mm"] == 10.0
        assert task.constraints["tumor_volume_cm3"] == 5.0


class TestTaskExecution:
    """Tests for execute_task."""

    def test_execute_without_connect_fails(self):
        """Executing without connection sets status to FAILED."""
        robot = SurgicalRobotInterface(robot_id="R-009")
        task = robot.plan_procedure("biopsy", np.zeros(3))
        result = robot.execute_task(task)
        assert result.status == TaskStatus.FAILED
        assert len(result.telemetry) == 0

    def test_execute_with_connection_generates_telemetry(self):
        """Executing while connected generates 10 telemetry records."""
        robot = SurgicalRobotInterface(robot_id="R-010")
        robot.connect()
        task = robot.plan_procedure("biopsy", np.array([5.0, 5.0, 5.0]))
        result = robot.execute_task(task, seed=42)
        assert len(result.telemetry) == 10
        assert result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    def test_telemetry_has_expected_keys(self):
        """Each telemetry record has step, position, force_n, velocity_mm_s."""
        robot = SurgicalRobotInterface(robot_id="R-011")
        robot.connect()
        task = robot.plan_procedure("biopsy", np.zeros(3))
        result = robot.execute_task(task, seed=0)
        for record in result.telemetry:
            assert "step" in record
            assert "position" in record
            assert "force_n" in record
            assert "velocity_mm_s" in record


class TestTelemetryFeatures:
    """Tests for get_telemetry_features."""

    def test_empty_telemetry_returns_zeros(self):
        """No telemetry yields a zero vector of length 10."""
        robot = SurgicalRobotInterface(robot_id="R-012")
        task = RoboticTask(task_id="T-001", task_type="biopsy")
        features = robot.get_telemetry_features(task)
        assert features.shape == (10,)
        np.testing.assert_allclose(features, np.zeros(10))

    def test_features_after_execution(self):
        """Feature vector has 10 elements and is float64."""
        robot = SurgicalRobotInterface(robot_id="R-013")
        robot.connect()
        task = robot.plan_procedure("biopsy", np.array([1.0, 2.0, 3.0]))
        robot.execute_task(task, seed=42)
        features = robot.get_telemetry_features(task)
        assert features.shape == (10,)
        assert features.dtype == np.float64


class TestTaskLog:
    """Tests for get_task_log."""

    def test_empty_log(self):
        """No tasks yields empty log."""
        robot = SurgicalRobotInterface(robot_id="R-014")
        assert robot.get_task_log() == []

    def test_log_after_plan(self):
        """Log reflects planned tasks."""
        robot = SurgicalRobotInterface(robot_id="R-015")
        robot.plan_procedure("biopsy", np.zeros(3))
        log = robot.get_task_log()
        assert len(log) == 1
        assert log[0]["task_type"] == "biopsy"
        assert log[0]["status"] == "pending"
