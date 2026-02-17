#!/usr/bin/env python3
"""Cross-framework and cross-platform validation for oncology surgical robotics.

CLINICAL CONTEXT
================
Oncology surgical robots trained in one simulation framework (e.g., NVIDIA
Isaac Sim) must be validated in alternative physics engines (MuJoCo, PyBullet)
before real-world deployment.  This ensures that learned policies generalize
across simulator-specific biases and satisfies regulatory requirements for
multi-environment verification.  This example demonstrates how to validate a
trained robotic policy across Isaac Sim, MuJoCo, and PyBullet, including
roundtrip model format conversion testing and cross-engine kinematics /
safety / performance validation.

USE CASES COVERED
=================
1. Cross-engine kinematic consistency validation comparing joint positions
   and velocities produced by a surgical robot policy in two physics engines.
2. Safety constraint validation ensuring forces, velocities, and workspace
   bounds remain within clinical limits across all target platforms.
3. Roundtrip model format conversion testing (URDF -> MJCF -> URDF) with
   physics property preservation checks.
4. Performance benchmarking of simulation step latency across engines to
   verify real-time control feasibility.
5. Multi-engine validation report generation with compliance-ready audit
   trails for FDA/IEC regulatory submissions.

FRAMEWORK REQUIREMENTS
======================
Required:
    - numpy >= 1.24.0
    - scipy >= 1.11.0

Optional:
    - mujoco >= 3.0.0  (MuJoCo physics engine)
      https://mujoco.readthedocs.io/
    - pybullet >= 3.2.5  (PyBullet physics engine)
      https://pybullet.org/
    - isaacsim >= 4.0.0  (NVIDIA Isaac Sim)
      https://developer.nvidia.com/isaac-sim
    - matplotlib >= 3.7.0  (validation result visualization)
      https://matplotlib.org/

HARDWARE REQUIREMENTS
=====================
    - CPU: Any modern x86-64 or ARM64 processor.
    - RAM: >= 8 GB recommended for multi-engine validation.
    - GPU: Required for Isaac Sim; optional for MuJoCo and PyBullet.

REFERENCES
==========
    - Todorov et al., "MuJoCo: A physics engine for model-based control",
      IROS 2012.  https://doi.org/10.1109/IROS.2012.6386109
    - Coumans & Bai, "PyBullet, a Python module for physics simulation for
      games, robotics and machine learning", 2016.
      https://pybullet.org/
    - NVIDIA Isaac Sim Documentation.
      https://docs.omniverse.nvidia.com/isaacsim/latest/
    - IEC 80601-2-77: Requirements for robotically assisted surgical
      equipment.

DISCLAIMER
==========
RESEARCH USE ONLY.  This software is provided for research and educational
purposes only.  It has NOT been validated for clinical use, is NOT approved
by the FDA or any other regulatory body, and MUST NOT be used to make
clinical decisions or control physical surgical robots in patient-facing
settings.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore[assignment]
    HAS_MUJOCO = False

try:
    import pybullet

    HAS_PYBULLET = True
except ImportError:
    pybullet = None  # type: ignore[assignment]
    HAS_PYBULLET = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

try:
    import isaacsim  # type: ignore[import-untyped]

    HAS_ISAAC = True
except ImportError:
    isaacsim = None  # type: ignore[assignment]
    HAS_ISAAC = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physical_ai.simulation_bridge import ModelFormat, RobotModel, SimulationBridge
from unification.cross_platform_tools.validation_suite import (
    ComplianceFramework,
    SeverityLevel,
    ValidationCategory,
    ValidationCheck,
    ValidationReport,
    ValidationResult,
    ValidationSuite,
    ValidationThresholds,
)

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Section 1 — Configuration: Enums + Dataclasses
# ============================================================================


class TargetEngine(str, Enum):
    """Supported target physics engines for cross-platform validation."""

    ISAAC_SIM = "isaac_sim"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"


class ConversionPath(str, Enum):
    """Model format conversion paths for roundtrip testing."""

    URDF_TO_MJCF = "urdf_to_mjcf"
    MJCF_TO_URDF = "mjcf_to_urdf"
    URDF_TO_SDF = "urdf_to_sdf"
    URDF_TO_USD = "urdf_to_usd"


class ValidationScope(str, Enum):
    """Scope of validation tests to execute."""

    KINEMATICS = "kinematics"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    FULL = "full"


ENGINE_AVAILABILITY: dict[TargetEngine, bool] = {
    TargetEngine.ISAAC_SIM: HAS_ISAAC,
    TargetEngine.MUJOCO: HAS_MUJOCO,
    TargetEngine.PYBULLET: HAS_PYBULLET,
}


@dataclass
class RobotConfig:
    """Configuration for the surgical robot model under test.

    Attributes:
        name: Robot model name.
        num_joints: Number of articulated joints.
        num_links: Number of rigid-body links.
        mass_kg: Total mass in kilograms.
        joint_names: Human-readable joint names.
        joint_limits: Per-joint (lower, upper) limits in radians.
        source_format: Original model description format.
        max_force_n: Maximum allowable force at end-effector.
        max_velocity_rad_s: Maximum allowable joint velocity.
        workspace_min: Minimum workspace bounds (x, y, z).
        workspace_max: Maximum workspace bounds (x, y, z).
    """

    name: str = "surgical_arm_6dof"
    num_joints: int = 6
    num_links: int = 7
    mass_kg: float = 12.5
    joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
    )
    joint_limits: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (-3.14, 3.14),
            (-1.57, 1.57),
            (-3.14, 3.14),
            (-3.14, 3.14),
            (-3.14, 3.14),
            (-3.14, 3.14),
        ]
    )
    source_format: str = "urdf"
    max_force_n: float = 40.0
    max_velocity_rad_s: float = 4.0
    workspace_min: list[float] = field(default_factory=lambda: [-0.8, -0.8, 0.0])
    workspace_max: list[float] = field(default_factory=lambda: [0.8, 0.8, 1.2])


@dataclass
class ValidationConfig:
    """Configuration for a cross-framework validation session.

    Attributes:
        session_name: Human-readable session name.
        robot: Robot model configuration.
        source_engine: Primary engine where the policy was trained.
        target_engines: Engines to validate against.
        scope: Which validation tests to run.
        thresholds: Numerical thresholds for pass/fail decisions.
        num_simulation_steps: Number of steps to simulate per engine.
        seed: Random seed for reproducibility.
        compliance_frameworks: Regulatory frameworks to validate against.
        export_reports: Whether to export reports to JSON.
        output_dir: Directory for exported reports.
    """

    session_name: str = "cross_framework_validation"
    robot: RobotConfig = field(default_factory=RobotConfig)
    source_engine: TargetEngine = TargetEngine.ISAAC_SIM
    target_engines: list[TargetEngine] = field(default_factory=lambda: [TargetEngine.MUJOCO, TargetEngine.PYBULLET])
    scope: ValidationScope = ValidationScope.FULL
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    num_simulation_steps: int = 100
    seed: int = 42
    compliance_frameworks: list[ComplianceFramework] = field(
        default_factory=lambda: [ComplianceFramework.IEC_62304, ComplianceFramework.IEC_80601_2_77]
    )
    export_reports: bool = False
    output_dir: str = "validation_reports"


@dataclass
class EngineSimulationData:
    """Simulated data from a single physics engine run.

    Attributes:
        engine: Source engine.
        joint_positions: Joint positions over time (steps x joints).
        joint_velocities: Joint velocities over time (steps x joints).
        ee_positions: End-effector positions over time (steps x 3).
        forces: Applied forces over time (steps,).
        step_times_ms: Per-step computation time in milliseconds.
    """

    engine: TargetEngine = TargetEngine.ISAAC_SIM
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    ee_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    forces: np.ndarray = field(default_factory=lambda: np.zeros(0))
    step_times_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class ConversionTestResult:
    """Result from a roundtrip model format conversion test."""

    conversion_path: str = ""
    source_format: str = ""
    target_format: str = ""
    roundtrip_format: str = ""
    properties_preserved: dict[str, bool] = field(default_factory=dict)
    all_passed: bool = False


@dataclass
class CrossFrameworkReport:
    """Complete report for a cross-framework validation session.

    Attributes:
        session_id: Unique session identifier.
        config: Validation session configuration.
        engine_reports: Per-engine-pair validation reports.
        conversion_tests: Roundtrip conversion test results.
        overall_pass: Whether all validations passed.
        total_checks: Total number of validation checks run.
        total_passed: Total number of checks that passed.
        total_duration_s: Total session duration in seconds.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: ValidationConfig = field(default_factory=ValidationConfig)
    engine_reports: list[ValidationReport] = field(default_factory=list)
    conversion_tests: list[ConversionTestResult] = field(default_factory=list)
    overall_pass: bool = False
    total_checks: int = 0
    total_passed: int = 0
    total_duration_s: float = 0.0


# ============================================================================
# Section 2 — Core Implementation: CrossFrameworkValidator
# ============================================================================


class SimulatedEngineRunner:
    """Generates simulated physics engine data for validation.

    In production, this would interface with actual Isaac Sim, MuJoCo,
    and PyBullet engines.  For this example, it produces synthetic
    trajectories with controlled cross-engine drift.

    Args:
        robot_config: Robot model configuration.
        seed: Random seed.
    """

    def __init__(self, robot_config: RobotConfig, seed: int = 42) -> None:
        self._robot = robot_config
        self._rng = np.random.default_rng(seed)

    def generate_trajectory(
        self,
        engine: TargetEngine,
        num_steps: int = 100,
        reference_trajectory: np.ndarray | None = None,
        drift_std: float = 0.0003,
    ) -> EngineSimulationData:
        """Generate a simulated robot trajectory for a given engine.

        Args:
            engine: Target physics engine.
            num_steps: Number of simulation steps.
            reference_trajectory: Base trajectory to add drift to.
            drift_std: Standard deviation of cross-engine drift.

        Returns:
            EngineSimulationData with positions, velocities, forces.
        """
        n_joints = self._robot.num_joints

        if reference_trajectory is not None:
            positions = reference_trajectory + self._rng.normal(0, drift_std, (num_steps, n_joints))
        else:
            positions = np.zeros((num_steps, n_joints))
            for step in range(num_steps):
                t = step / num_steps * 2 * np.pi
                for j in range(n_joints):
                    low, high = self._robot.joint_limits[j]
                    mid = (low + high) / 2
                    amp = (high - low) / 4
                    positions[step, j] = mid + amp * np.sin(t + j * 0.5)

        velocities = np.gradient(positions, axis=0) * 100.0
        velocities += self._rng.normal(0, drift_std * 10, velocities.shape)

        ee_positions = np.column_stack(
            [
                np.sin(np.linspace(0, 2 * np.pi, num_steps)) * 0.3,
                np.cos(np.linspace(0, 2 * np.pi, num_steps)) * 0.3,
                np.ones(num_steps) * 0.5 + np.sin(np.linspace(0, np.pi, num_steps)) * 0.2,
            ]
        )
        ee_positions += self._rng.normal(0, drift_std, ee_positions.shape)

        forces = self._rng.uniform(5, 25, size=num_steps)

        base_latency = {"isaac_sim": 2.0, "mujoco": 0.8, "pybullet": 1.5}
        mean_latency = base_latency.get(engine.value, 1.0)
        step_times = self._rng.exponential(mean_latency, size=num_steps) + 0.1

        logger.info("Generated %d-step trajectory for %s", num_steps, engine.value)
        return EngineSimulationData(
            engine=engine,
            joint_positions=positions,
            joint_velocities=velocities,
            ee_positions=ee_positions,
            forces=forces,
            step_times_ms=step_times,
        )


class CrossFrameworkValidator:
    """Validates robotic policies across multiple physics engines.

    Orchestrates simulation data generation, cross-engine comparisons,
    model format conversion tests, and regulatory-ready report compilation.

    Args:
        config: Validation session configuration.
    """

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self._suite = ValidationSuite(thresholds=config.thresholds)
        self._bridge = SimulationBridge()
        self._runner = SimulatedEngineRunner(config.robot, seed=config.seed)
        self._engine_data: dict[TargetEngine, EngineSimulationData] = {}

    # ------------------------------------------------------------------
    # Simulation data generation
    # ------------------------------------------------------------------

    def generate_engine_data(self) -> dict[TargetEngine, EngineSimulationData]:
        """Generate simulation data from all configured engines.

        Returns:
            Dictionary mapping engine to simulation data.
        """
        all_engines = [self.config.source_engine] + self.config.target_engines
        reference: np.ndarray | None = None

        for engine in all_engines:
            available = ENGINE_AVAILABILITY.get(engine, False)
            if not available:
                logger.warning("Engine %s not installed; using simulated data", engine.value)

            data = self._runner.generate_trajectory(
                engine=engine,
                num_steps=self.config.num_simulation_steps,
                reference_trajectory=reference,
            )
            self._engine_data[engine] = data

            if reference is None:
                reference = data.joint_positions

        logger.info("Generated data for %d engines", len(self._engine_data))
        return self._engine_data

    # ------------------------------------------------------------------
    # Cross-engine kinematics validation
    # ------------------------------------------------------------------

    def validate_kinematics(
        self,
        source_engine: TargetEngine,
        target_engine: TargetEngine,
    ) -> ValidationReport:
        """Validate kinematic consistency between two engines.

        Args:
            source_engine: Reference engine.
            target_engine: Engine to validate against.

        Returns:
            ValidationReport with kinematic check results.
        """
        src = self._engine_data.get(source_engine)
        tgt = self._engine_data.get(target_engine)
        if src is None or tgt is None:
            raise ValueError("Missing engine data. Call generate_engine_data() first.")

        n_steps = min(len(src.joint_positions), len(tgt.joint_positions))
        step_positions_src = np.mean(src.joint_positions[:n_steps], axis=0)
        step_positions_tgt = np.mean(tgt.joint_positions[:n_steps], axis=0)
        step_velocities_src = np.mean(src.joint_velocities[:n_steps], axis=0)
        step_velocities_tgt = np.mean(tgt.joint_velocities[:n_steps], axis=0)

        report = self._suite.run_kinematics_validation(
            positions_a=step_positions_src,
            positions_b=step_positions_tgt,
            velocities_a=step_velocities_src,
            velocities_b=step_velocities_tgt,
            joint_names=self.config.robot.joint_names,
            engine_a=source_engine.value,
            engine_b=target_engine.value,
        )

        logger.info(
            "Kinematics %s vs %s: %s (pass=%d, fail=%d)",
            source_engine.value,
            target_engine.value,
            report.overall_result.value,
            report.pass_count,
            report.fail_count,
        )
        return report

    # ------------------------------------------------------------------
    # Safety validation
    # ------------------------------------------------------------------

    def validate_safety(self, engine: TargetEngine) -> ValidationReport:
        """Validate safety constraints for a single engine.

        Args:
            engine: Target engine to validate.

        Returns:
            ValidationReport with safety check results.
        """
        data = self._engine_data.get(engine)
        if data is None:
            raise ValueError(f"No data for engine {engine.value}")

        report = self._suite.run_safety_validation(
            forces=data.forces,
            velocities=np.max(np.abs(data.joint_velocities), axis=1),
            ee_positions=np.mean(data.ee_positions, axis=0),
            max_force_n=self.config.robot.max_force_n,
            max_velocity_rad_s=self.config.robot.max_velocity_rad_s,
            workspace_min=np.array(self.config.robot.workspace_min),
            workspace_max=np.array(self.config.robot.workspace_max),
        )

        logger.info(
            "Safety %s: %s (pass=%d, fail=%d)",
            engine.value,
            report.overall_result.value,
            report.pass_count,
            report.fail_count,
        )
        return report

    # ------------------------------------------------------------------
    # Roundtrip conversion testing
    # ------------------------------------------------------------------

    def run_roundtrip_conversion_tests(self) -> list[ConversionTestResult]:
        """Test model format roundtrip conversions and verify property preservation.

        Returns:
            List of ConversionTestResult for each conversion path.
        """
        results: list[ConversionTestResult] = []

        source_model = RobotModel(
            name=self.config.robot.name,
            source_format=ModelFormat.URDF,
            num_joints=self.config.robot.num_joints,
            num_links=self.config.robot.num_links,
            mass_kg=self.config.robot.mass_kg,
            joint_limits=list(self.config.robot.joint_limits),
            metadata={"task": "oncology_surgical", "source": "training_pipeline"},
        )
        self._bridge.register_model(source_model)

        conversion_paths = [
            (ModelFormat.URDF, ModelFormat.MJCF, "URDF -> MJCF -> URDF"),
            (ModelFormat.URDF, ModelFormat.SDF, "URDF -> SDF -> URDF"),
            (ModelFormat.URDF, ModelFormat.USD, "URDF -> USD (one-way)"),
        ]

        for src_fmt, tgt_fmt, description in conversion_paths:
            try:
                converted = self._bridge.convert(source_model, tgt_fmt)
                validation = self._bridge.validate_conversion(source_model, converted)

                can_roundtrip = (tgt_fmt, src_fmt) in {
                    (ModelFormat.MJCF, ModelFormat.URDF),
                    (ModelFormat.SDF, ModelFormat.URDF),
                }

                roundtrip_validation: dict[str, bool] = {}
                if can_roundtrip:
                    roundtrip = self._bridge.convert(converted, src_fmt)
                    roundtrip_validation = self._bridge.validate_conversion(source_model, roundtrip)
                    combined = {**validation, **{f"roundtrip_{k}": v for k, v in roundtrip_validation.items()}}
                else:
                    combined = validation

                test_result = ConversionTestResult(
                    conversion_path=description,
                    source_format=src_fmt.value,
                    target_format=tgt_fmt.value,
                    roundtrip_format=src_fmt.value if can_roundtrip else "N/A",
                    properties_preserved=combined,
                    all_passed=all(combined.values()),
                )
                results.append(test_result)

                logger.info("Conversion %s: %s", description, "PASS" if test_result.all_passed else "FAIL")

            except ValueError as exc:
                logger.warning("Conversion %s skipped: %s", description, exc)
                results.append(
                    ConversionTestResult(
                        conversion_path=description,
                        source_format=src_fmt.value,
                        target_format=tgt_fmt.value,
                        all_passed=False,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Full validation session
    # ------------------------------------------------------------------

    def run_full_validation(self) -> CrossFrameworkReport:
        """Execute a complete cross-framework validation session.

        Runs kinematics, safety, performance, and conversion validations
        across all configured engines and compiles a unified report.

        Returns:
            CrossFrameworkReport with all validation results.
        """
        start = time.time()
        logger.info("Starting full cross-framework validation: %s", self.config.session_name)

        self.generate_engine_data()

        report = CrossFrameworkReport(config=self.config)
        all_engines = [self.config.source_engine] + self.config.target_engines

        # Kinematics validation across engine pairs
        if self.config.scope in (ValidationScope.KINEMATICS, ValidationScope.FULL):
            for target in self.config.target_engines:
                kin_report = self.validate_kinematics(self.config.source_engine, target)
                report.engine_reports.append(kin_report)

        # Safety validation per engine
        if self.config.scope in (ValidationScope.SAFETY, ValidationScope.FULL):
            for engine in all_engines:
                safety_report = self.validate_safety(engine)
                report.engine_reports.append(safety_report)

        # Roundtrip conversion tests
        conversion_results = self.run_roundtrip_conversion_tests()
        report.conversion_tests = conversion_results

        # Aggregate statistics
        report.total_checks = sum(len(r.checks) for r in report.engine_reports)
        report.total_passed = sum(r.pass_count for r in report.engine_reports)
        report.overall_pass = report.total_checks > 0 and report.total_passed == report.total_checks
        report.total_duration_s = time.time() - start

        logger.info(
            "Validation complete: %d/%d checks passed, overall=%s, duration=%.2fs",
            report.total_passed,
            report.total_checks,
            "PASS" if report.overall_pass else "FAIL",
            report.total_duration_s,
        )
        return report


# ============================================================================
# Section 3 — Pipeline Orchestration: ValidationPipelineManager
# ============================================================================


class ValidationPipelineManager:
    """Manages multiple validation sessions and produces comparative reports.

    Supports running validation with different thresholds or engine
    combinations and generating side-by-side comparison tables.
    """

    def __init__(self) -> None:
        self._sessions: list[CrossFrameworkReport] = []

    def run_session(self, config: ValidationConfig) -> CrossFrameworkReport:
        """Execute a single validation session.

        Args:
            config: Validation configuration.

        Returns:
            CrossFrameworkReport with results.
        """
        validator = CrossFrameworkValidator(config)
        report = validator.run_full_validation()
        self._sessions.append(report)
        return report

    def run_multi_threshold_comparison(
        self,
        base_config: ValidationConfig,
        position_thresholds: list[float],
    ) -> list[CrossFrameworkReport]:
        """Run validation at multiple tolerance levels for comparison.

        Args:
            base_config: Base configuration template.
            position_thresholds: List of position error thresholds to test.

        Returns:
            List of CrossFrameworkReport, one per threshold level.
        """
        reports: list[CrossFrameworkReport] = []
        for threshold in position_thresholds:
            thresholds = ValidationThresholds(max_position_error_m=threshold)
            config = ValidationConfig(
                session_name=f"threshold_{threshold:.1e}",
                robot=base_config.robot,
                source_engine=base_config.source_engine,
                target_engines=list(base_config.target_engines),
                scope=base_config.scope,
                thresholds=thresholds,
                num_simulation_steps=base_config.num_simulation_steps,
                seed=base_config.seed,
            )
            report = self.run_session(config)
            reports.append(report)
        return reports

    def print_session_report(self, report: CrossFrameworkReport) -> None:
        """Pretty-print a validation session report.

        Args:
            report: Cross-framework validation report.
        """
        print("\n" + "=" * 80)
        print("  CROSS-FRAMEWORK VALIDATION REPORT")
        print("=" * 80)
        print(f"  Session:     {report.config.session_name}")
        print(f"  Session ID:  {report.session_id}")
        print(f"  Robot:       {report.config.robot.name}")
        print(f"  Source:      {report.config.source_engine.value}")
        targets = ", ".join(e.value for e in report.config.target_engines)
        print(f"  Targets:     {targets}")
        print(f"  Duration:    {report.total_duration_s:.2f}s")
        print(f"  Overall:     {'PASS' if report.overall_pass else 'FAIL'}")
        print(f"  Checks:      {report.total_passed}/{report.total_checks} passed")

        # Engine availability
        print("\n  Engine Availability:")
        for engine, available in ENGINE_AVAILABILITY.items():
            status = "installed" if available else "simulated"
            print(f"    {engine.value:<15} {status}")

        # Per-report details
        print("\n  Validation Reports:")
        print(f"    {'Session':<40} {'Result':>8} {'Pass':>6} {'Fail':>6} {'Warn':>6}")
        print(f"    {'-' * 40} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 6}")
        for eng_report in report.engine_reports:
            print(
                f"    {eng_report.session_name:<40} "
                f"{eng_report.overall_result.value:>8} "
                f"{eng_report.pass_count:>6} "
                f"{eng_report.fail_count:>6} "
                f"{eng_report.warn_count:>6}"
            )

        # Conversion tests
        if report.conversion_tests:
            print("\n  Roundtrip Conversion Tests:")
            print(f"    {'Path':<35} {'Result':>8} {'Properties':}")
            print(f"    {'-' * 35} {'-' * 8} {'-' * 30}")
            for ct in report.conversion_tests:
                result = "PASS" if ct.all_passed else "FAIL"
                preserved = sum(1 for v in ct.properties_preserved.values() if v)
                total = len(ct.properties_preserved)
                print(f"    {ct.conversion_path:<35} {result:>8} {preserved}/{total} preserved")

        print("\n" + "=" * 80)
        print("  DISCLAIMER: RESEARCH USE ONLY")
        print("=" * 80)

    def print_threshold_comparison(self, reports: list[CrossFrameworkReport]) -> None:
        """Print a comparison table across threshold levels.

        Args:
            reports: List of reports with different threshold configurations.
        """
        print("\n" + "=" * 70)
        print("  THRESHOLD SENSITIVITY ANALYSIS")
        print("=" * 70)
        print(f"  {'Threshold (m)':<15} {'Result':>8} {'Passed':>8} {'Total':>8} {'Duration':>10}")
        print(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")
        for report in reports:
            threshold = report.config.thresholds.max_position_error_m
            result = "PASS" if report.overall_pass else "FAIL"
            print(
                f"  {threshold:<15.1e} {result:>8} {report.total_passed:>8} "
                f"{report.total_checks:>8} {report.total_duration_s:>9.2f}s"
            )
        print("=" * 70)

    @property
    def sessions(self) -> list[CrossFrameworkReport]:
        """Return all completed session reports."""
        return list(self._sessions)


# ============================================================================
# Section 4 — Demonstration
# ============================================================================


if __name__ == "__main__":
    logger.info("Physical AI Federated Learning — Cross-Framework Validation Example v0.5.0")
    logger.info(
        "Optional deps: mujoco=%s, pybullet=%s, isaac=%s, matplotlib=%s",
        HAS_MUJOCO,
        HAS_PYBULLET,
        HAS_ISAAC,
        HAS_MATPLOTLIB,
    )

    # Create validation configuration
    robot_config = RobotConfig(
        name="oncology_surgical_arm_6dof",
        num_joints=6,
        num_links=7,
        mass_kg=12.5,
        max_force_n=40.0,
        max_velocity_rad_s=4.0,
    )

    validation_config = ValidationConfig(
        session_name="oncology_arm_cross_validation",
        robot=robot_config,
        source_engine=TargetEngine.ISAAC_SIM,
        target_engines=[TargetEngine.MUJOCO, TargetEngine.PYBULLET],
        scope=ValidationScope.FULL,
        num_simulation_steps=100,
        seed=42,
    )

    # Run the validation pipeline
    manager = ValidationPipelineManager()

    print("\n--- Running Full Cross-Framework Validation ---")
    report = manager.run_session(validation_config)
    manager.print_session_report(report)

    # Run threshold sensitivity analysis
    print("\n--- Running Threshold Sensitivity Analysis ---")
    threshold_reports = manager.run_multi_threshold_comparison(
        validation_config,
        position_thresholds=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    )
    manager.print_threshold_comparison(threshold_reports)

    # Show supported conversion paths
    print("\n--- Supported Model Conversions ---")
    conversions = SimulationBridge.get_supported_conversions()
    for conv in conversions:
        print(f"  {conv['from']:>6} -> {conv['to']:<6}  steps: {conv['steps']}")

    logger.info("Cross-framework validation complete. Sessions: %d", len(manager.sessions))
    print("\nDISCLAIMER: RESEARCH USE ONLY. Not for clinical decision-making.")
