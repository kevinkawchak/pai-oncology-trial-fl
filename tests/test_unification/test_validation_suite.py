"""Tests for unification/cross_platform_tools/validation_suite.py.

Covers enums (ValidationResult, ValidationCategory, SeverityLevel,
ComplianceFramework), dataclasses (ValidationThresholds, ValidationCheck,
ValidationReport), KinematicsValidator, SafetyValidator,
PerformanceValidator, and ValidationSuite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("validation_suite", "unification/cross_platform_tools/validation_suite.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestValidationResult:
    """Tests for the ValidationResult enum."""

    def test_members(self):
        """ValidationResult has PASS, FAIL, WARN, SKIP, ERROR."""
        expected = {"PASS", "FAIL", "WARN", "SKIP", "ERROR"}
        assert set(mod.ValidationResult.__members__.keys()) == expected


class TestValidationCategory:
    """Tests for the ValidationCategory enum."""

    def test_members(self):
        """ValidationCategory has eight members."""
        expected = {
            "KINEMATICS",
            "DYNAMICS",
            "CONTACT",
            "SAFETY",
            "POLICY_CONSISTENCY",
            "STATE_TRANSFER",
            "PERFORMANCE",
            "REGRESSION",
        }
        assert set(mod.ValidationCategory.__members__.keys()) == expected


class TestSeverityLevel:
    """Tests for the SeverityLevel enum."""

    def test_members(self):
        """SeverityLevel has INFO, LOW, MEDIUM, HIGH, CRITICAL."""
        assert len(mod.SeverityLevel) == 5
        assert mod.SeverityLevel.INFO.value == "info"
        assert mod.SeverityLevel.CRITICAL.value == "critical"


class TestComplianceFramework:
    """Tests for the ComplianceFramework enum."""

    def test_count(self):
        """ComplianceFramework has four members."""
        assert len(mod.ComplianceFramework) == 4


# ---------------------------------------------------------------------------
# ValidationThresholds dataclass
# ---------------------------------------------------------------------------
class TestValidationThresholds:
    """Tests for the ValidationThresholds dataclass."""

    def test_defaults(self):
        """Default thresholds are reasonable."""
        t = mod.ValidationThresholds()
        assert t.max_position_error_m > 0
        assert t.max_velocity_error_rad_s > 0
        assert t.max_force_error_n > 0

    def test_custom_values(self):
        """Thresholds accept custom values."""
        t = mod.ValidationThresholds(max_position_error_m=0.05, max_force_error_n=2.0)
        assert t.max_position_error_m == 0.05
        assert t.max_force_error_n == 2.0

    def test_to_dict(self):
        """to_dict serializes all threshold fields."""
        t = mod.ValidationThresholds()
        d = t.to_dict()
        assert "max_position_error_m" in d
        assert "max_step_latency_ms" in d
        assert len(d) == 11


# ---------------------------------------------------------------------------
# ValidationCheck dataclass
# ---------------------------------------------------------------------------
class TestValidationCheck:
    """Tests for the ValidationCheck dataclass."""

    def test_defaults(self):
        """Default check has SKIP result."""
        vc = mod.ValidationCheck(
            name="Test check",
            category=mod.ValidationCategory.KINEMATICS,
        )
        assert vc.result == mod.ValidationResult.SKIP

    def test_measured_vs_threshold(self):
        """Check records measured and threshold values."""
        vc = mod.ValidationCheck(
            name="Force check",
            category=mod.ValidationCategory.SAFETY,
            result=mod.ValidationResult.PASS,
            measured_value=50.0,
            threshold_value=100.0,
        )
        assert vc.measured_value < vc.threshold_value


# ---------------------------------------------------------------------------
# ValidationReport dataclass
# ---------------------------------------------------------------------------
class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_empty_report_counts(self):
        """Empty report has zero counts."""
        report = mod.ValidationReport()
        assert report.pass_count == 0
        assert report.fail_count == 0
        assert report.warn_count == 0

    def test_pass_count(self):
        """pass_count counts PASS results."""
        checks = [
            mod.ValidationCheck(name="a", category=mod.ValidationCategory.KINEMATICS, result=mod.ValidationResult.PASS),
            mod.ValidationCheck(name="b", category=mod.ValidationCategory.KINEMATICS, result=mod.ValidationResult.FAIL),
            mod.ValidationCheck(name="c", category=mod.ValidationCategory.KINEMATICS, result=mod.ValidationResult.PASS),
        ]
        report = mod.ValidationReport(checks=checks)
        assert report.pass_count == 2
        assert report.fail_count == 1

    def test_overall_result_all_pass(self):
        """overall_result is PASS when all checks pass."""
        checks = [
            mod.ValidationCheck(name="a", category=mod.ValidationCategory.SAFETY, result=mod.ValidationResult.PASS),
        ]
        report = mod.ValidationReport(checks=checks)
        assert report.overall_result == mod.ValidationResult.PASS

    def test_overall_result_has_fail(self):
        """overall_result is FAIL when any check fails."""
        checks = [
            mod.ValidationCheck(name="a", category=mod.ValidationCategory.SAFETY, result=mod.ValidationResult.PASS),
            mod.ValidationCheck(name="b", category=mod.ValidationCategory.SAFETY, result=mod.ValidationResult.FAIL),
        ]
        report = mod.ValidationReport(checks=checks)
        assert report.overall_result == mod.ValidationResult.FAIL

    def test_overall_result_has_warn(self):
        """overall_result is WARN when has warnings but no fails."""
        checks = [
            mod.ValidationCheck(name="a", category=mod.ValidationCategory.SAFETY, result=mod.ValidationResult.PASS),
            mod.ValidationCheck(name="b", category=mod.ValidationCategory.SAFETY, result=mod.ValidationResult.WARN),
        ]
        report = mod.ValidationReport(checks=checks)
        assert report.overall_result == mod.ValidationResult.WARN

    def test_compute_hash(self):
        """compute_hash is deterministic."""
        checks = [
            mod.ValidationCheck(name="a", category=mod.ValidationCategory.SAFETY, result=mod.ValidationResult.PASS)
        ]
        report = mod.ValidationReport(checks=checks)
        h1 = report.compute_hash()
        h2 = report.compute_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_to_dict(self):
        """to_dict produces a JSON-serializable dictionary."""
        report = mod.ValidationReport()
        d = report.to_dict()
        assert "checks" in d
        assert "report_id" in d

    def test_to_json(self):
        """to_json returns valid JSON."""
        report = mod.ValidationReport()
        j = report.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# KinematicsValidator
# ---------------------------------------------------------------------------
class TestKinematicsValidator:
    """Tests for the KinematicsValidator class."""

    def test_validate_joint_positions_within_tolerance(self):
        """Matching positions produce PASS."""
        kv = mod.KinematicsValidator()
        thresholds = mod.ValidationThresholds()
        isaac = np.array([0.1, 0.2, 0.3])
        mujoco = np.array([0.1, 0.2, 0.3])
        names = ["j1", "j2", "j3"]
        checks = kv.validate_joint_positions(isaac, mujoco, names, thresholds)
        assert len(checks) > 0
        assert all(c.result == mod.ValidationResult.PASS for c in checks)

    def test_validate_joint_positions_outside_tolerance(self):
        """Divergent positions produce FAIL."""
        kv = mod.KinematicsValidator()
        thresholds = mod.ValidationThresholds()
        isaac = np.array([0.1, 0.2, 0.3])
        mujoco = np.array([0.5, 0.6, 0.7])
        names = ["j1", "j2", "j3"]
        checks = kv.validate_joint_positions(isaac, mujoco, names, thresholds)
        assert any(c.result == mod.ValidationResult.FAIL for c in checks)

    def test_validate_joint_velocities_within(self):
        """Matching velocities produce PASS."""
        kv = mod.KinematicsValidator()
        thresholds = mod.ValidationThresholds()
        isaac = np.array([1.0, 2.0])
        mujoco = np.array([1.0, 2.0])
        names = ["j1", "j2"]
        checks = kv.validate_joint_velocities(isaac, mujoco, names, thresholds)
        assert all(c.result == mod.ValidationResult.PASS for c in checks)

    def test_dimension_mismatch(self):
        """Dimension mismatch produces FAIL."""
        kv = mod.KinematicsValidator()
        thresholds = mod.ValidationThresholds()
        checks = kv.validate_joint_positions(np.array([1.0, 2.0]), np.array([1.0]), ["j1", "j2"], thresholds)
        assert any(c.result == mod.ValidationResult.FAIL for c in checks)


# ---------------------------------------------------------------------------
# SafetyValidator
# ---------------------------------------------------------------------------
class TestSafetyValidator:
    """Tests for the SafetyValidator class."""

    def test_validate_force_limits_within(self):
        """Forces within limits produce PASS."""
        sv = mod.SafetyValidator()
        forces = np.array([10.0, 20.0, 30.0])
        check = sv.validate_force_limits(forces, max_force_n=50.0)
        assert check.result == mod.ValidationResult.PASS

    def test_validate_force_limits_exceeded(self):
        """Forces exceeding limit produce FAIL."""
        sv = mod.SafetyValidator()
        forces = np.array([10.0, 20.0])
        check = sv.validate_force_limits(forces, max_force_n=5.0)
        assert check.result == mod.ValidationResult.FAIL

    def test_validate_velocity_limits_within(self):
        """Velocities within limits produce PASS."""
        sv = mod.SafetyValidator()
        velocities = np.array([0.1, 0.2])
        check = sv.validate_velocity_limits(velocities, max_velocity=10.0)
        assert check.result == mod.ValidationResult.PASS

    def test_validate_velocity_limits_exceeded(self):
        """Velocities exceeding limit produce FAIL."""
        sv = mod.SafetyValidator()
        velocities = np.array([15.0, 20.0])
        check = sv.validate_velocity_limits(velocities, max_velocity=10.0)
        assert check.result == mod.ValidationResult.FAIL

    def test_validate_workspace_bounds_within(self):
        """Positions within workspace produce PASS."""
        sv = mod.SafetyValidator()
        positions = np.array([0.5, 0.5, 0.5])
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([1.0, 1.0, 1.0])
        check = sv.validate_workspace_bounds(positions, bounds_min, bounds_max)
        assert check.result == mod.ValidationResult.PASS


# ---------------------------------------------------------------------------
# PerformanceValidator
# ---------------------------------------------------------------------------
class TestPerformanceValidator:
    """Tests for the PerformanceValidator class."""

    def test_validate_step_latency_within(self):
        """Low latency produces PASS."""
        pv = mod.PerformanceValidator()
        step_times = np.array([1.0, 1.5, 2.0, 1.2])  # milliseconds
        check = pv.validate_step_latency(step_times, max_latency_ms=10.0)
        assert check.result == mod.ValidationResult.PASS

    def test_validate_step_latency_exceeded(self):
        """High latency exceeding threshold produces FAIL."""
        pv = mod.PerformanceValidator()
        step_times = np.array([10.0, 15.0, 20.0, 25.0])  # milliseconds
        check = pv.validate_step_latency(step_times, max_latency_ms=5.0)
        assert check.result == mod.ValidationResult.FAIL


# ---------------------------------------------------------------------------
# ValidationSuite
# ---------------------------------------------------------------------------
class TestValidationSuite:
    """Tests for the ValidationSuite class."""

    def test_creation(self):
        """ValidationSuite can be created."""
        suite = mod.ValidationSuite()
        assert suite is not None

    def test_reports_empty_initially(self):
        """Reports list is empty before any validation."""
        suite = mod.ValidationSuite()
        assert suite.reports == []

    def test_run_kinematics_validation(self):
        """run_kinematics_validation produces a report."""
        suite = mod.ValidationSuite()
        names = ["j1", "j2", "j3"]
        isaac_pos = np.array([0.1, 0.2, 0.3])
        mujoco_pos = np.array([0.1, 0.2, 0.3])
        isaac_vel = np.array([0.5, 0.5, 0.5])
        mujoco_vel = np.array([0.5, 0.5, 0.5])
        report = suite.run_kinematics_validation(
            isaac_pos,
            mujoco_pos,
            isaac_vel,
            mujoco_vel,
            names,
        )
        assert isinstance(report, mod.ValidationReport)
        assert report.pass_count > 0

    def test_run_safety_validation(self):
        """run_safety_validation produces a report."""
        suite = mod.ValidationSuite()
        forces = np.array([5.0, 10.0])
        velocities = np.array([0.5, 1.0])
        ee_positions = np.array([0.3, 0.4, 0.5])
        report = suite.run_safety_validation(forces, velocities, ee_positions)
        assert isinstance(report, mod.ValidationReport)

    def test_run_safety_validation_with_workspace(self):
        """run_safety_validation with workspace bounds includes workspace check."""
        suite = mod.ValidationSuite()
        forces = np.array([5.0])
        velocities = np.array([0.5])
        ee_positions = np.array([0.5, 0.5, 0.5])
        report = suite.run_safety_validation(
            forces,
            velocities,
            ee_positions,
            workspace_min=np.array([0.0, 0.0, 0.0]),
            workspace_max=np.array([1.0, 1.0, 1.0]),
        )
        assert len(report.checks) == 3  # force, velocity, workspace

    def test_export_report(self):
        """export_report writes JSON to disk."""
        import os
        import tempfile

        suite = mod.ValidationSuite()
        names = ["j1"]
        report = suite.run_kinematics_validation(
            np.array([0.1]),
            np.array([0.1]),
            np.array([0.5]),
            np.array([0.5]),
            names,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            suite.export_report(report, path)
            with open(path) as fh:
                data = json.load(fh)
            assert "checks" in data
        finally:
            os.unlink(path)

    def test_reports_accumulate(self):
        """Reports list grows with each validation run."""
        suite = mod.ValidationSuite()
        names = ["j1"]
        suite.run_kinematics_validation(
            np.array([0.1]),
            np.array([0.1]),
            np.array([0.5]),
            np.array([0.5]),
            names,
        )
        suite.run_safety_validation(
            np.array([5.0]),
            np.array([0.5]),
            np.array([0.3, 0.4, 0.5]),
        )
        assert len(suite.reports) == 2
