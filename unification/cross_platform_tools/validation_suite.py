"""
Cross-Platform Validation Suite for Oncology Surgical Robotics Policies.

Provides automated validation of policies, models, and simulation states
across physics engines. Detects regressions, quantifies cross-engine drift,
and generates compliance reports for regulatory submissions.

DISCLAIMER: RESEARCH USE ONLY
This software is provided for research and educational purposes only.
It has NOT been validated for clinical use, is NOT approved by the FDA
or any other regulatory body, and MUST NOT be used to make clinical
decisions or control physical surgical robots in patient-facing settings.
All validation results must be reviewed by qualified engineers before
any clinical application. Use at your own risk.

Copyright (c) 2026 PAI Oncology Trial FL Contributors
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

try:
    import matplotlib

    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class ValidationResult(Enum):
    """Outcome of a validation check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    ERROR = "error"


class ValidationCategory(Enum):
    """Category of validation test."""

    KINEMATICS = "kinematics"
    DYNAMICS = "dynamics"
    CONTACT = "contact"
    SAFETY = "safety"
    POLICY_CONSISTENCY = "policy_consistency"
    STATE_TRANSFER = "state_transfer"
    PERFORMANCE = "performance"
    REGRESSION = "regression"


class SeverityLevel(Enum):
    """Severity of a validation finding."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks relevant to validation."""

    FDA_21_CFR_PART_11 = "fda_21cfr11"
    IEC_62304 = "iec_62304"
    ISO_14971 = "iso_14971"
    IEC_80601_2_77 = "iec_80601_2_77"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ValidationThresholds:
    """Configurable thresholds for validation checks."""

    max_position_error_m: float = 1e-3  # 1 mm
    max_velocity_error_rad_s: float = 0.01
    max_force_error_n: float = 0.5
    max_torque_error_nm: float = 0.1
    max_contact_position_error_m: float = 5e-3  # 5 mm
    max_contact_force_error_n: float = 2.0
    max_policy_action_divergence: float = 0.05
    max_state_transfer_error: float = 1e-6
    min_simulation_frequency_hz: float = 100.0
    max_step_latency_ms: float = 10.0
    regression_tolerance_pct: float = 5.0

    def to_dict(self) -> dict[str, float]:
        """Serialize thresholds to dictionary."""
        return {
            "max_position_error_m": self.max_position_error_m,
            "max_velocity_error_rad_s": self.max_velocity_error_rad_s,
            "max_force_error_n": self.max_force_error_n,
            "max_torque_error_nm": self.max_torque_error_nm,
            "max_contact_position_error_m": self.max_contact_position_error_m,
            "max_contact_force_error_n": self.max_contact_force_error_n,
            "max_policy_action_divergence": self.max_policy_action_divergence,
            "max_state_transfer_error": self.max_state_transfer_error,
            "min_simulation_frequency_hz": self.min_simulation_frequency_hz,
            "max_step_latency_ms": self.max_step_latency_ms,
            "regression_tolerance_pct": self.regression_tolerance_pct,
        }


@dataclass
class ValidationCheck:
    """A single validation check result."""

    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: ValidationCategory = ValidationCategory.KINEMATICS
    result: ValidationResult = ValidationResult.SKIP
    severity: SeverityLevel = SeverityLevel.INFO
    measured_value: float = 0.0
    threshold_value: float = 0.0
    unit: str = ""
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "check_id": self.check_id,
            "name": self.name,
            "category": self.category.value,
            "result": self.result.value,
            "severity": self.severity.value,
            "measured_value": self.measured_value,
            "threshold_value": self.threshold_value,
            "unit": self.unit,
            "description": self.description,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class ValidationReport:
    """Complete validation report for a cross-platform test session."""

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_name: str = ""
    timestamp: float = field(default_factory=time.time)
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    checks: list[ValidationCheck] = field(default_factory=list)
    engine_a: str = ""
    engine_b: str = ""
    robot_model: str = ""
    task_name: str = ""
    num_steps: int = 0
    total_duration_ms: float = 0.0
    compliance_frameworks: list[ComplianceFramework] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_count(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.result == ValidationResult.PASS)

    @property
    def fail_count(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if c.result == ValidationResult.FAIL)

    @property
    def warn_count(self) -> int:
        """Number of checks with warnings."""
        return sum(1 for c in self.checks if c.result == ValidationResult.WARN)

    @property
    def overall_result(self) -> ValidationResult:
        """Compute overall validation result."""
        if any(c.result == ValidationResult.ERROR for c in self.checks):
            return ValidationResult.ERROR
        if any(c.result == ValidationResult.FAIL for c in self.checks):
            return ValidationResult.FAIL
        if any(c.result == ValidationResult.WARN for c in self.checks):
            return ValidationResult.WARN
        if all(c.result == ValidationResult.PASS for c in self.checks):
            return ValidationResult.PASS
        return ValidationResult.SKIP

    def compute_hash(self) -> str:
        """Compute integrity hash of the report."""
        data = json.dumps(
            {
                "report_id": self.report_id,
                "session_name": self.session_name,
                "checks": [c.check_id for c in self.checks],
                "results": [c.result.value for c in self.checks],
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_id": self.report_id,
            "session_name": self.session_name,
            "timestamp": self.timestamp,
            "thresholds": self.thresholds.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "engine_a": self.engine_a,
            "engine_b": self.engine_b,
            "robot_model": self.robot_model,
            "task_name": self.task_name,
            "num_steps": self.num_steps,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "summary": {
                "total_checks": len(self.checks),
                "pass": self.pass_count,
                "fail": self.fail_count,
                "warn": self.warn_count,
                "overall": self.overall_result.value,
            },
            "compliance_frameworks": [cf.value for cf in self.compliance_frameworks],
            "report_hash": self.compute_hash(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
class KinematicsValidator:
    """Validates kinematic consistency between simulation engines."""

    def validate_joint_positions(
        self,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        joint_names: list[str],
        thresholds: ValidationThresholds,
    ) -> list[ValidationCheck]:
        """Compare joint positions between two engines."""
        checks: list[ValidationCheck] = []

        if len(positions_a) != len(positions_b):
            check = ValidationCheck(
                name="joint_position_dimension_match",
                category=ValidationCategory.KINEMATICS,
                result=ValidationResult.FAIL,
                severity=SeverityLevel.CRITICAL,
                description=f"Dimension mismatch: {len(positions_a)} vs {len(positions_b)}",
            )
            checks.append(check)
            return checks

        errors = np.abs(positions_a - positions_b)
        max_error = float(np.max(errors)) if len(errors) > 0 else 0.0
        mean_error = float(np.mean(errors)) if len(errors) > 0 else 0.0

        # Per-joint checks
        for i, (name, error) in enumerate(zip(joint_names, errors)):
            result = ValidationResult.PASS if error <= thresholds.max_position_error_m else ValidationResult.FAIL
            severity = SeverityLevel.INFO if result == ValidationResult.PASS else SeverityLevel.HIGH

            checks.append(
                ValidationCheck(
                    name=f"joint_position_{name}",
                    category=ValidationCategory.KINEMATICS,
                    result=result,
                    severity=severity,
                    measured_value=float(error),
                    threshold_value=thresholds.max_position_error_m,
                    unit="rad",
                    description=f"Joint '{name}' position error: {error:.6f} rad",
                )
            )

        # Aggregate check
        agg_result = ValidationResult.PASS if max_error <= thresholds.max_position_error_m else ValidationResult.FAIL
        checks.append(
            ValidationCheck(
                name="joint_positions_aggregate",
                category=ValidationCategory.KINEMATICS,
                result=agg_result,
                severity=SeverityLevel.HIGH if agg_result == ValidationResult.FAIL else SeverityLevel.INFO,
                measured_value=max_error,
                threshold_value=thresholds.max_position_error_m,
                unit="rad",
                description=f"Max joint position error: {max_error:.6f} rad (mean: {mean_error:.6f})",
                details={"max_error": max_error, "mean_error": mean_error, "num_joints": len(joint_names)},
            )
        )

        return checks

    def validate_joint_velocities(
        self,
        velocities_a: np.ndarray,
        velocities_b: np.ndarray,
        joint_names: list[str],
        thresholds: ValidationThresholds,
    ) -> list[ValidationCheck]:
        """Compare joint velocities between two engines."""
        checks: list[ValidationCheck] = []

        if len(velocities_a) != len(velocities_b):
            checks.append(
                ValidationCheck(
                    name="joint_velocity_dimension_match",
                    category=ValidationCategory.KINEMATICS,
                    result=ValidationResult.FAIL,
                    severity=SeverityLevel.CRITICAL,
                )
            )
            return checks

        errors = np.abs(velocities_a - velocities_b)
        max_error = float(np.max(errors)) if len(errors) > 0 else 0.0

        agg_result = (
            ValidationResult.PASS if max_error <= thresholds.max_velocity_error_rad_s else ValidationResult.FAIL
        )
        checks.append(
            ValidationCheck(
                name="joint_velocities_aggregate",
                category=ValidationCategory.KINEMATICS,
                result=agg_result,
                severity=SeverityLevel.HIGH if agg_result == ValidationResult.FAIL else SeverityLevel.INFO,
                measured_value=max_error,
                threshold_value=thresholds.max_velocity_error_rad_s,
                unit="rad/s",
                description=f"Max joint velocity error: {max_error:.6f} rad/s",
            )
        )

        return checks


class SafetyValidator:
    """Validates safety constraints in policy actions and states."""

    def validate_force_limits(
        self,
        forces: np.ndarray,
        max_force_n: float,
    ) -> ValidationCheck:
        """Check that all forces are within safety limits."""
        max_measured = float(np.max(np.abs(forces))) if len(forces) > 0 else 0.0
        result = ValidationResult.PASS if max_measured <= max_force_n else ValidationResult.FAIL
        severity = SeverityLevel.CRITICAL if result == ValidationResult.FAIL else SeverityLevel.INFO

        return ValidationCheck(
            name="force_safety_limit",
            category=ValidationCategory.SAFETY,
            result=result,
            severity=severity,
            measured_value=max_measured,
            threshold_value=max_force_n,
            unit="N",
            description=f"Max force: {max_measured:.2f} N (limit: {max_force_n:.2f} N)",
        )

    def validate_velocity_limits(
        self,
        velocities: np.ndarray,
        max_velocity: float,
    ) -> ValidationCheck:
        """Check that all velocities are within safety limits."""
        max_measured = float(np.max(np.abs(velocities))) if len(velocities) > 0 else 0.0
        result = ValidationResult.PASS if max_measured <= max_velocity else ValidationResult.FAIL

        return ValidationCheck(
            name="velocity_safety_limit",
            category=ValidationCategory.SAFETY,
            result=result,
            severity=SeverityLevel.CRITICAL if result == ValidationResult.FAIL else SeverityLevel.INFO,
            measured_value=max_measured,
            threshold_value=max_velocity,
            unit="rad/s",
            description=f"Max velocity: {max_measured:.4f} rad/s (limit: {max_velocity:.4f})",
        )

    def validate_workspace_bounds(
        self,
        positions: np.ndarray,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
    ) -> ValidationCheck:
        """Check that end-effector stays within workspace bounds."""
        in_bounds = np.all(positions >= bounds_min) and np.all(positions <= bounds_max)
        result = ValidationResult.PASS if in_bounds else ValidationResult.FAIL

        return ValidationCheck(
            name="workspace_bounds",
            category=ValidationCategory.SAFETY,
            result=result,
            severity=SeverityLevel.HIGH if result == ValidationResult.FAIL else SeverityLevel.INFO,
            description="End-effector workspace bounds check",
            details={
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "position": positions.tolist(),
            },
        )


class PerformanceValidator:
    """Validates simulation performance metrics."""

    def validate_step_latency(
        self,
        step_times_ms: np.ndarray,
        max_latency_ms: float,
    ) -> ValidationCheck:
        """Check that simulation step times are within acceptable limits."""
        max_measured = float(np.max(step_times_ms)) if len(step_times_ms) > 0 else 0.0
        mean_measured = float(np.mean(step_times_ms)) if len(step_times_ms) > 0 else 0.0
        p99 = float(np.percentile(step_times_ms, 99)) if len(step_times_ms) > 0 else 0.0

        result = ValidationResult.PASS if p99 <= max_latency_ms else ValidationResult.FAIL

        return ValidationCheck(
            name="step_latency",
            category=ValidationCategory.PERFORMANCE,
            result=result,
            severity=SeverityLevel.MEDIUM if result == ValidationResult.FAIL else SeverityLevel.INFO,
            measured_value=p99,
            threshold_value=max_latency_ms,
            unit="ms",
            description=f"Step latency p99: {p99:.2f} ms (mean: {mean_measured:.2f}, max: {max_measured:.2f})",
            details={"mean_ms": mean_measured, "max_ms": max_measured, "p99_ms": p99},
        )


# ---------------------------------------------------------------------------
# Validation suite orchestrator
# ---------------------------------------------------------------------------
class ValidationSuite:
    """Orchestrates cross-platform validation for oncology robotics.

    Runs a configurable set of validation checks comparing simulation
    behavior across physics engines. Generates reports suitable for
    regulatory submissions and regression detection.
    """

    def __init__(self, thresholds: Optional[ValidationThresholds] = None) -> None:
        self._thresholds = thresholds or ValidationThresholds()
        self._kinematics = KinematicsValidator()
        self._safety = SafetyValidator()
        self._performance = PerformanceValidator()
        self._reports: list[ValidationReport] = []

    def run_kinematics_validation(
        self,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        velocities_a: np.ndarray,
        velocities_b: np.ndarray,
        joint_names: list[str],
        engine_a: str = "isaac_sim",
        engine_b: str = "mujoco",
    ) -> ValidationReport:
        """Run kinematic comparison between two engines."""
        start = time.time()

        report = ValidationReport(
            session_name=f"kinematics_{engine_a}_vs_{engine_b}",
            thresholds=self._thresholds,
            engine_a=engine_a,
            engine_b=engine_b,
        )

        # Position checks
        pos_checks = self._kinematics.validate_joint_positions(positions_a, positions_b, joint_names, self._thresholds)
        report.checks.extend(pos_checks)

        # Velocity checks
        vel_checks = self._kinematics.validate_joint_velocities(
            velocities_a, velocities_b, joint_names, self._thresholds
        )
        report.checks.extend(vel_checks)

        report.total_duration_ms = (time.time() - start) * 1000.0
        self._reports.append(report)

        logger.info(
            "Kinematics validation: %s (pass=%d fail=%d warn=%d)",
            report.overall_result.value,
            report.pass_count,
            report.fail_count,
            report.warn_count,
        )
        return report

    def run_safety_validation(
        self,
        forces: np.ndarray,
        velocities: np.ndarray,
        ee_positions: np.ndarray,
        max_force_n: float = 50.0,
        max_velocity_rad_s: float = 6.2832,
        workspace_min: Optional[np.ndarray] = None,
        workspace_max: Optional[np.ndarray] = None,
    ) -> ValidationReport:
        """Run safety constraint validation."""
        start = time.time()

        report = ValidationReport(session_name="safety_validation", thresholds=self._thresholds)

        report.checks.append(self._safety.validate_force_limits(forces, max_force_n))
        report.checks.append(self._safety.validate_velocity_limits(velocities, max_velocity_rad_s))

        if workspace_min is not None and workspace_max is not None:
            report.checks.append(self._safety.validate_workspace_bounds(ee_positions, workspace_min, workspace_max))

        report.total_duration_ms = (time.time() - start) * 1000.0
        self._reports.append(report)

        logger.info(
            "Safety validation: %s (pass=%d fail=%d)",
            report.overall_result.value,
            report.pass_count,
            report.fail_count,
        )
        return report

    def export_report(self, report: ValidationReport, output_path: str) -> None:
        """Export a validation report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report.to_json())
        logger.info("Report exported: %s", output_path)

    @property
    def reports(self) -> list[ValidationReport]:
        """Return all generated reports."""
        return list(self._reports)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the validation suite."""
    logger.info("Validation Suite demonstration")

    suite = ValidationSuite()
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

    # Simulate data from two engines
    rng = np.random.default_rng(42)
    positions_a = rng.uniform(-3.14, 3.14, size=6)
    positions_b = positions_a + rng.normal(0, 0.0005, size=6)  # small drift

    velocities_a = rng.uniform(-1.0, 1.0, size=6)
    velocities_b = velocities_a + rng.normal(0, 0.005, size=6)

    # Kinematics validation
    kin_report = suite.run_kinematics_validation(
        positions_a,
        positions_b,
        velocities_a,
        velocities_b,
        joint_names,
        engine_a="isaac_sim",
        engine_b="mujoco",
    )

    # Safety validation
    forces = rng.uniform(0, 30, size=100)
    velocities = rng.uniform(0, 5, size=100)
    ee_pos = rng.uniform(-0.5, 0.5, size=3)

    safety_report = suite.run_safety_validation(
        forces,
        velocities,
        ee_pos,
        workspace_min=np.array([-1.0, -1.0, 0.0]),
        workspace_max=np.array([1.0, 1.0, 1.5]),
    )

    logger.info("Kinematics: %s | Safety: %s", kin_report.overall_result.value, safety_report.overall_result.value)
    logger.info("Total reports: %d", len(suite.reports))
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
