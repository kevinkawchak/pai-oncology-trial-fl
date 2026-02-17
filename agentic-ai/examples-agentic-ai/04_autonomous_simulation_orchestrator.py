"""
Autonomous Multi-Framework Simulation Orchestration Agent.

CLINICAL CONTEXT:
    This module implements an autonomous orchestration agent that manages
    simulation jobs across multiple physics engines (NVIDIA Isaac Sim,
    MuJoCo, PyBullet) for robotic-assisted oncology procedures. The agent
    autonomously submits simulation configurations, monitors job progress,
    compares results across frameworks, and selects optimal configurations
    for physical AI integration in clinical trials.

    The orchestrator supports multi-objective optimization of simulation
    parameters including trajectory accuracy, force compliance, collision
    avoidance, and procedure completion time, enabling cross-framework
    validation of robotic surgical plans.

USE CASES COVERED:
    1. Autonomous submission and monitoring of simulation jobs across
       Isaac Sim, MuJoCo, and PyBullet physics engines.
    2. Cross-framework result comparison with statistical analysis of
       trajectory accuracy, force profiles, and safety metrics.
    3. Multi-objective configuration optimization using Pareto front
       analysis to balance accuracy, speed, and safety.
    4. Automatic retry and fallback when simulations fail, with
       framework-specific error handling and recovery.
    5. Consolidated reporting with cross-framework validation scores
       and confidence intervals for surgical plan approval.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0        (https://numpy.org/)
    Optional:
        - anthropic >= 0.39.0    (https://docs.anthropic.com/)
        - openai >= 1.0.0        (https://platform.openai.com/)
        - isaacsim >= 4.0.0      (https://developer.nvidia.com/isaac-sim)
        - mujoco >= 3.0.0        (https://mujoco.org/)
        - pybullet >= 3.2.0      (https://pybullet.org/)

REFERENCES:
    - NVIDIA Isaac Sim Documentation (2025).
      URL: https://docs.omniverse.nvidia.com/isaacsim/
    - Todorov et al. (2012). MuJoCo: A physics engine for model-based
      control. IROS 2012. DOI: 10.1109/IROS.2012.6386109
    - Coumans & Bai (2016). PyBullet, a Python module for physics
      simulation. URL: http://pybullet.org
    - IEC 80601-2-77: Medical Electrical Equipment - Particular
      requirements for robotically assisted surgical equipment.

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or direct patient care. All simulation results must
    be independently validated before any clinical application.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports for optional dependencies
# ---------------------------------------------------------------------------
try:
    import anthropic  # type: ignore[import-untyped]

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

try:
    import openai  # type: ignore[import-untyped]

    HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[assignment]
    HAS_OPENAI = False

try:
    import mujoco  # type: ignore[import-untyped]

    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore[assignment]
    HAS_MUJOCO = False

try:
    import pybullet  # type: ignore[import-untyped]

    HAS_PYBULLET = True
except ImportError:
    pybullet = None  # type: ignore[assignment]
    HAS_PYBULLET = False

try:
    import isaacsim  # type: ignore[import-untyped]

    HAS_ISAACSIM = True
except ImportError:
    isaacsim = None  # type: ignore[assignment]
    HAS_ISAACSIM = False

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
class SimFramework(Enum):
    """Supported simulation physics frameworks."""

    ISAAC_SIM = "isaac_sim"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"


class JobStatus(Enum):
    """Simulation job lifecycle status."""

    QUEUED = "queued"
    CONFIGURING = "configuring"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ProcedureType(Enum):
    """Types of robotic-assisted oncology procedures."""

    BIOPSY = "biopsy"
    RESECTION = "resection"
    ABLATION = "ablation"
    NEEDLE_INSERTION = "needle_insertion"
    CATHETER_PLACEMENT = "catheter_placement"
    INSTRUMENT_EXCHANGE = "instrument_exchange"


class OptimizationObjective(Enum):
    """Multi-objective optimization targets."""

    TRAJECTORY_ACCURACY = "trajectory_accuracy"
    FORCE_COMPLIANCE = "force_compliance"
    COLLISION_AVOIDANCE = "collision_avoidance"
    COMPLETION_TIME = "completion_time"
    ENERGY_EFFICIENCY = "energy_efficiency"
    TISSUE_SAFETY = "tissue_safety"


class OrchestratorState(Enum):
    """State of the orchestration agent."""

    IDLE = "idle"
    PLANNING = "planning"
    SUBMITTING = "submitting"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETE = "complete"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Configuration for a single simulation job."""

    config_id: str = field(default_factory=lambda: f"CFG-{uuid.uuid4().hex[:10].upper()}")
    framework: SimFramework = SimFramework.MUJOCO
    procedure: ProcedureType = ProcedureType.BIOPSY
    robot_model: str = "kuka_iiwa_14"
    scene_file: str = ""
    time_step_s: float = 0.001
    max_duration_s: float = 60.0
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    target_position_mm: list[float] = field(default_factory=lambda: [100.0, 50.0, 30.0])
    max_force_n: float = 10.0
    max_velocity_mm_s: float = 50.0
    collision_margin_mm: float = 2.0
    tissue_stiffness_n_per_mm: float = 0.5
    seed: int = 42
    custom_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config_id": self.config_id,
            "framework": self.framework.value,
            "procedure": self.procedure.value,
            "robot_model": self.robot_model,
            "time_step_s": self.time_step_s,
            "max_duration_s": self.max_duration_s,
            "target_position_mm": self.target_position_mm,
            "max_force_n": self.max_force_n,
            "max_velocity_mm_s": self.max_velocity_mm_s,
            "collision_margin_mm": self.collision_margin_mm,
            "tissue_stiffness_n_per_mm": self.tissue_stiffness_n_per_mm,
            "seed": self.seed,
        }


@dataclass
class SimulationResult:
    """Result from a completed simulation job."""

    result_id: str = field(default_factory=lambda: f"RES-{uuid.uuid4().hex[:10].upper()}")
    job_id: str = ""
    config_id: str = ""
    framework: SimFramework = SimFramework.MUJOCO
    status: JobStatus = JobStatus.COMPLETED
    trajectory_error_mm: float = 0.0
    max_force_n: float = 0.0
    mean_force_n: float = 0.0
    collision_count: int = 0
    completion_time_s: float = 0.0
    energy_j: float = 0.0
    tissue_damage_score: float = 0.0
    safety_score: float = 0.0
    trajectory_points: list[list[float]] = field(default_factory=list)
    force_profile: list[float] = field(default_factory=list)
    wall_clock_time_s: float = 0.0
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "result_id": self.result_id,
            "job_id": self.job_id,
            "config_id": self.config_id,
            "framework": self.framework.value,
            "status": self.status.value,
            "trajectory_error_mm": self.trajectory_error_mm,
            "max_force_n": self.max_force_n,
            "mean_force_n": self.mean_force_n,
            "collision_count": self.collision_count,
            "completion_time_s": self.completion_time_s,
            "energy_j": self.energy_j,
            "tissue_damage_score": self.tissue_damage_score,
            "safety_score": self.safety_score,
            "wall_clock_time_s": self.wall_clock_time_s,
            "error_message": self.error_message,
        }


@dataclass
class SimulationJob:
    """A managed simulation job with lifecycle tracking."""

    job_id: str = field(default_factory=lambda: f"JOB-{uuid.uuid4().hex[:10].upper()}")
    config: SimulationConfig = field(default_factory=SimulationConfig)
    status: JobStatus = JobStatus.QUEUED
    result: Optional[SimulationResult] = None
    submitted_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error_log: list[str] = field(default_factory=list)


@dataclass
class CrossFrameworkComparison:
    """Comparison of simulation results across frameworks."""

    comparison_id: str = field(default_factory=lambda: f"CMP-{uuid.uuid4().hex[:8].upper()}")
    procedure: ProcedureType = ProcedureType.BIOPSY
    frameworks_compared: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    trajectory_agreement_pct: float = 0.0
    force_agreement_pct: float = 0.0
    safety_agreement_pct: float = 0.0
    best_framework: str = ""
    best_config_id: str = ""
    confidence_score: float = 0.0
    recommendation: str = ""
    pareto_optimal_configs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "comparison_id": self.comparison_id,
            "procedure": self.procedure.value,
            "frameworks_compared": self.frameworks_compared,
            "trajectory_agreement_pct": self.trajectory_agreement_pct,
            "force_agreement_pct": self.force_agreement_pct,
            "safety_agreement_pct": self.safety_agreement_pct,
            "best_framework": self.best_framework,
            "best_config_id": self.best_config_id,
            "confidence_score": self.confidence_score,
            "recommendation": self.recommendation,
            "pareto_optimal_configs": self.pareto_optimal_configs,
        }


@dataclass
class OptimizationResult:
    """Result of multi-objective configuration optimization."""

    optimization_id: str = field(default_factory=lambda: f"OPT-{uuid.uuid4().hex[:8].upper()}")
    objectives: list[str] = field(default_factory=list)
    pareto_front: list[dict[str, Any]] = field(default_factory=list)
    best_compromise: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    total_simulations: int = 0
    improvement_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "optimization_id": self.optimization_id,
            "objectives": self.objectives,
            "pareto_front_size": len(self.pareto_front),
            "best_compromise": self.best_compromise,
            "iterations": self.iterations,
            "total_simulations": self.total_simulations,
            "improvement_pct": self.improvement_pct,
        }


# ---------------------------------------------------------------------------
# Framework-specific simulation runners (synthetic)
# ---------------------------------------------------------------------------
class SimulationRunner:
    """Runs simulated physics simulations across frameworks.

    In production, each method would interface with the actual physics
    engine. Here we provide synthetic results for demonstration that
    capture realistic performance characteristics of each framework.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def run_isaac_sim(self, config: SimulationConfig) -> SimulationResult:
        """Run simulation in NVIDIA Isaac Sim (synthetic)."""
        start = time.time()
        # Isaac Sim characteristics: GPU-accelerated, high-fidelity, slower startup
        base_error = float(self._rng.uniform(0.3, 1.2))
        trajectory_error = base_error * (config.time_step_s / 0.001)

        n_points = int(config.max_duration_s / config.time_step_s / 100)
        n_points = min(max(n_points, 10), 200)
        trajectory = self._generate_trajectory(config, n_points, noise_std=0.1)
        force_profile = self._generate_force_profile(config, n_points, noise_std=0.2)

        max_force = float(np.max(force_profile))
        mean_force = float(np.mean(force_profile))
        collisions = int(self._rng.integers(0, 2))
        completion = float(self._rng.uniform(8.0, 15.0))
        energy = float(self._rng.uniform(5.0, 15.0))
        tissue_damage = float(np.clip(max_force / config.max_force_n * 0.3, 0, 1))
        safety = float(np.clip(1.0 - tissue_damage - collisions * 0.1, 0.3, 1.0))

        return SimulationResult(
            config_id=config.config_id,
            framework=SimFramework.ISAAC_SIM,
            status=JobStatus.COMPLETED,
            trajectory_error_mm=trajectory_error,
            max_force_n=max_force,
            mean_force_n=mean_force,
            collision_count=collisions,
            completion_time_s=completion,
            energy_j=energy,
            tissue_damage_score=tissue_damage,
            safety_score=safety,
            trajectory_points=trajectory,
            force_profile=force_profile,
            wall_clock_time_s=time.time() - start,
            metadata={"gpu_utilized": True, "render_mode": "path_tracing"},
        )

    def run_mujoco(self, config: SimulationConfig) -> SimulationResult:
        """Run simulation in MuJoCo (synthetic)."""
        start = time.time()
        # MuJoCo characteristics: fast, accurate contact dynamics
        base_error = float(self._rng.uniform(0.5, 1.5))
        trajectory_error = base_error * (config.time_step_s / 0.001)

        n_points = int(config.max_duration_s / config.time_step_s / 100)
        n_points = min(max(n_points, 10), 200)
        trajectory = self._generate_trajectory(config, n_points, noise_std=0.15)
        force_profile = self._generate_force_profile(config, n_points, noise_std=0.25)

        max_force = float(np.max(force_profile))
        mean_force = float(np.mean(force_profile))
        collisions = int(self._rng.integers(0, 3))
        completion = float(self._rng.uniform(7.0, 13.0))
        energy = float(self._rng.uniform(4.0, 12.0))
        tissue_damage = float(np.clip(max_force / config.max_force_n * 0.35, 0, 1))
        safety = float(np.clip(1.0 - tissue_damage - collisions * 0.1, 0.3, 1.0))

        return SimulationResult(
            config_id=config.config_id,
            framework=SimFramework.MUJOCO,
            status=JobStatus.COMPLETED,
            trajectory_error_mm=trajectory_error,
            max_force_n=max_force,
            mean_force_n=mean_force,
            collision_count=collisions,
            completion_time_s=completion,
            energy_j=energy,
            tissue_damage_score=tissue_damage,
            safety_score=safety,
            trajectory_points=trajectory,
            force_profile=force_profile,
            wall_clock_time_s=time.time() - start,
            metadata={"solver": "Newton", "integrator": "semi-implicit_euler"},
        )

    def run_pybullet(self, config: SimulationConfig) -> SimulationResult:
        """Run simulation in PyBullet (synthetic)."""
        start = time.time()
        # PyBullet characteristics: lightweight, moderate accuracy
        base_error = float(self._rng.uniform(0.8, 2.0))
        trajectory_error = base_error * (config.time_step_s / 0.001)

        n_points = int(config.max_duration_s / config.time_step_s / 100)
        n_points = min(max(n_points, 10), 200)
        trajectory = self._generate_trajectory(config, n_points, noise_std=0.2)
        force_profile = self._generate_force_profile(config, n_points, noise_std=0.3)

        max_force = float(np.max(force_profile))
        mean_force = float(np.mean(force_profile))
        collisions = int(self._rng.integers(0, 4))
        completion = float(self._rng.uniform(6.0, 12.0))
        energy = float(self._rng.uniform(3.5, 11.0))
        tissue_damage = float(np.clip(max_force / config.max_force_n * 0.4, 0, 1))
        safety = float(np.clip(1.0 - tissue_damage - collisions * 0.1, 0.2, 1.0))

        return SimulationResult(
            config_id=config.config_id,
            framework=SimFramework.PYBULLET,
            status=JobStatus.COMPLETED,
            trajectory_error_mm=trajectory_error,
            max_force_n=max_force,
            mean_force_n=mean_force,
            collision_count=collisions,
            completion_time_s=completion,
            energy_j=energy,
            tissue_damage_score=tissue_damage,
            safety_score=safety,
            trajectory_points=trajectory,
            force_profile=force_profile,
            wall_clock_time_s=time.time() - start,
            metadata={"physics_engine": "bullet3", "solver_iterations": 50},
        )

    def _generate_trajectory(
        self, config: SimulationConfig, n_points: int, noise_std: float = 0.1
    ) -> list[list[float]]:
        """Generate a synthetic trajectory toward the target position."""
        start = np.array([0.0, 0.0, 0.0])
        target = np.array(config.target_position_mm)
        points = []
        for i in range(n_points):
            t = (i + 1) / n_points
            smooth_t = 3 * t**2 - 2 * t**3
            point = start + (target - start) * smooth_t
            noise = self._rng.normal(0, noise_std, 3)
            point = point + noise
            points.append([float(p) for p in point])
        return points

    def _generate_force_profile(self, config: SimulationConfig, n_points: int, noise_std: float = 0.2) -> list[float]:
        """Generate a synthetic force profile during the procedure."""
        forces = []
        for i in range(n_points):
            t = i / max(n_points - 1, 1)
            # Force ramps up during approach, peaks at contact, decreases
            base_force = config.max_force_n * 0.5 * np.sin(np.pi * t) ** 2
            noise = abs(float(self._rng.normal(0, noise_std)))
            forces.append(float(np.clip(base_force + noise, 0, config.max_force_n * 1.2)))
        return forces


# ---------------------------------------------------------------------------
# Autonomous simulation orchestrator
# ---------------------------------------------------------------------------
class AutonomousSimulationOrchestrator:
    """Autonomous agent that orchestrates simulations across physics frameworks.

    Manages the full lifecycle of multi-framework simulation campaigns:
    planning, submission, monitoring, comparison, and optimization.
    """

    def __init__(self, seed: int = 42) -> None:
        self._runner = SimulationRunner(seed=seed)
        self._jobs: dict[str, SimulationJob] = {}
        self._results: list[SimulationResult] = []
        self._comparisons: list[CrossFrameworkComparison] = []
        self._state = OrchestratorState.IDLE
        self._rng = np.random.default_rng(seed)
        logger.info("AutonomousSimulationOrchestrator initialized")

    def plan_campaign(
        self,
        procedure: ProcedureType,
        frameworks: Optional[list[SimFramework]] = None,
        configurations_per_framework: int = 3,
        seed: int = 42,
    ) -> list[SimulationConfig]:
        """Plan a simulation campaign across frameworks.

        Args:
            procedure: Type of surgical procedure to simulate.
            frameworks: Physics frameworks to use (default: all three).
            configurations_per_framework: Number of config variants per framework.
            seed: Random seed for configuration generation.

        Returns:
            List of simulation configurations to execute.
        """
        self._state = OrchestratorState.PLANNING
        if frameworks is None:
            frameworks = [SimFramework.ISAAC_SIM, SimFramework.MUJOCO, SimFramework.PYBULLET]

        configs = []
        rng = np.random.default_rng(seed)

        for framework in frameworks:
            for i in range(configurations_per_framework):
                config = SimulationConfig(
                    framework=framework,
                    procedure=procedure,
                    time_step_s=float(rng.choice([0.0005, 0.001, 0.002])),
                    max_duration_s=float(rng.uniform(30.0, 90.0)),
                    target_position_mm=[
                        float(rng.uniform(80, 120)),
                        float(rng.uniform(40, 60)),
                        float(rng.uniform(20, 40)),
                    ],
                    max_force_n=float(rng.uniform(5.0, 15.0)),
                    max_velocity_mm_s=float(rng.uniform(30.0, 70.0)),
                    collision_margin_mm=float(rng.uniform(1.0, 3.0)),
                    tissue_stiffness_n_per_mm=float(rng.uniform(0.3, 0.8)),
                    seed=seed + i * 100 + list(SimFramework).index(framework) * 1000,
                )
                configs.append(config)

        logger.info(
            "Campaign planned: %d configurations across %d frameworks for %s",
            len(configs),
            len(frameworks),
            procedure.value,
        )
        return configs

    def submit_job(self, config: SimulationConfig) -> SimulationJob:
        """Submit a simulation job for execution.

        Args:
            config: Simulation configuration.

        Returns:
            The submitted SimulationJob with tracking ID.
        """
        self._state = OrchestratorState.SUBMITTING
        job = SimulationJob(config=config, status=JobStatus.QUEUED, submitted_at=time.time())
        self._jobs[job.job_id] = job
        logger.info("Job submitted: %s (framework=%s, config=%s)", job.job_id, config.framework.value, config.config_id)
        return job

    def execute_job(self, job: SimulationJob) -> SimulationResult:
        """Execute a simulation job and handle failures with retry.

        Args:
            job: The simulation job to execute.

        Returns:
            SimulationResult from the execution.
        """
        self._state = OrchestratorState.MONITORING
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        try:
            if job.config.framework == SimFramework.ISAAC_SIM:
                result = self._runner.run_isaac_sim(job.config)
            elif job.config.framework == SimFramework.MUJOCO:
                result = self._runner.run_mujoco(job.config)
            elif job.config.framework == SimFramework.PYBULLET:
                result = self._runner.run_pybullet(job.config)
            else:
                raise ValueError(f"Unsupported framework: {job.config.framework.value}")

            result.job_id = job.job_id
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            self._results.append(result)

            logger.info(
                "Job completed: %s (%s) error=%.2fmm, safety=%.2f, time=%.3fs",
                job.job_id,
                job.config.framework.value,
                result.trajectory_error_mm,
                result.safety_score,
                result.wall_clock_time_s,
            )
            return result

        except Exception as exc:
            job.error_log.append(str(exc))
            job.retry_count += 1

            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                logger.warning(
                    "Job %s failed (attempt %d/%d): %s. Retrying...",
                    job.job_id,
                    job.retry_count,
                    job.max_retries,
                    exc,
                )
                return self.execute_job(job)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                logger.error("Job %s failed permanently: %s", job.job_id, exc)
                return SimulationResult(
                    job_id=job.job_id,
                    config_id=job.config.config_id,
                    framework=job.config.framework,
                    status=JobStatus.FAILED,
                    error_message=str(exc),
                )

    def run_campaign(self, configs: list[SimulationConfig]) -> list[SimulationResult]:
        """Execute a full simulation campaign.

        Args:
            configs: List of simulation configurations.

        Returns:
            List of all simulation results.
        """
        results = []
        for config in configs:
            job = self.submit_job(config)
            result = self.execute_job(job)
            results.append(result)

        logger.info(
            "Campaign complete: %d/%d jobs succeeded",
            sum(1 for r in results if r.status == JobStatus.COMPLETED),
            len(results),
        )
        return results

    def compare_frameworks(self, results: Optional[list[SimulationResult]] = None) -> CrossFrameworkComparison:
        """Compare simulation results across frameworks.

        Args:
            results: Results to compare (default: all collected results).

        Returns:
            CrossFrameworkComparison with agreement metrics and recommendations.
        """
        self._state = OrchestratorState.ANALYZING
        if results is None:
            results = self._results

        completed = [r for r in results if r.status == JobStatus.COMPLETED]
        if not completed:
            return CrossFrameworkComparison(recommendation="No completed simulations to compare")

        # Group by framework
        by_framework: dict[str, list[SimulationResult]] = {}
        for r in completed:
            fw = r.framework.value
            if fw not in by_framework:
                by_framework[fw] = []
            by_framework[fw].append(r)

        # Compute per-framework statistics
        fw_stats: dict[str, dict[str, float]] = {}
        for fw, fw_results in by_framework.items():
            errors = [r.trajectory_error_mm for r in fw_results]
            forces = [r.max_force_n for r in fw_results]
            safety = [r.safety_score for r in fw_results]
            fw_stats[fw] = {
                "mean_error_mm": float(np.mean(errors)),
                "std_error_mm": float(np.std(errors)),
                "mean_max_force_n": float(np.mean(forces)),
                "mean_safety_score": float(np.mean(safety)),
                "n_simulations": len(fw_results),
            }

        # Compute agreement metrics
        all_errors = [r.trajectory_error_mm for r in completed]
        all_forces = [r.max_force_n for r in completed]
        all_safety = [r.safety_score for r in completed]

        error_cv = float(np.std(all_errors) / max(np.mean(all_errors), 0.001))
        force_cv = float(np.std(all_forces) / max(np.mean(all_forces), 0.001))
        safety_cv = float(np.std(all_safety) / max(np.mean(all_safety), 0.001))

        trajectory_agreement = float(np.clip((1.0 - error_cv) * 100, 0, 100))
        force_agreement = float(np.clip((1.0 - force_cv) * 100, 0, 100))
        safety_agreement = float(np.clip((1.0 - safety_cv) * 100, 0, 100))

        # Find best framework by composite score
        best_fw = ""
        best_score = -1.0
        for fw, stats in fw_stats.items():
            # Lower error is better, higher safety is better
            composite = stats["mean_safety_score"] - stats["mean_error_mm"] * 0.1
            if composite > best_score:
                best_score = composite
                best_fw = fw

        # Find best individual config
        best_result = max(completed, key=lambda r: r.safety_score - r.trajectory_error_mm * 0.1)
        confidence = float(np.clip((trajectory_agreement + force_agreement + safety_agreement) / 300.0, 0, 1))

        comparison = CrossFrameworkComparison(
            frameworks_compared=list(by_framework.keys()),
            results=[{"framework": fw, **stats} for fw, stats in fw_stats.items()],
            trajectory_agreement_pct=trajectory_agreement,
            force_agreement_pct=force_agreement,
            safety_agreement_pct=safety_agreement,
            best_framework=best_fw,
            best_config_id=best_result.config_id,
            confidence_score=confidence,
            recommendation=(
                f"Recommended framework: {best_fw} with trajectory agreement "
                f"{trajectory_agreement:.1f}% across frameworks. "
                f"Cross-framework validation confidence: {confidence:.2f}."
            ),
        )

        self._comparisons.append(comparison)
        logger.info(
            "Framework comparison: best=%s, agreement=(traj=%.1f%%, force=%.1f%%, safety=%.1f%%)",
            best_fw,
            trajectory_agreement,
            force_agreement,
            safety_agreement,
        )
        return comparison

    def optimize_configuration(
        self,
        base_config: SimulationConfig,
        objectives: Optional[list[OptimizationObjective]] = None,
        n_iterations: int = 5,
    ) -> OptimizationResult:
        """Optimize simulation configuration using multi-objective search.

        Performs a simplified Pareto optimization by exploring configuration
        variations and tracking the Pareto front of non-dominated solutions.

        Args:
            base_config: Starting configuration to optimize from.
            objectives: Optimization objectives.
            n_iterations: Number of optimization iterations.

        Returns:
            OptimizationResult with Pareto front and best compromise.
        """
        self._state = OrchestratorState.OPTIMIZING
        if objectives is None:
            objectives = [
                OptimizationObjective.TRAJECTORY_ACCURACY,
                OptimizationObjective.FORCE_COMPLIANCE,
                OptimizationObjective.TISSUE_SAFETY,
            ]

        pareto_front: list[dict[str, Any]] = []
        all_configs: list[tuple[SimulationConfig, SimulationResult]] = []
        initial_result: Optional[SimulationResult] = None

        for iteration in range(n_iterations):
            # Generate a variation of the base config
            varied_config = SimulationConfig(
                framework=base_config.framework,
                procedure=base_config.procedure,
                robot_model=base_config.robot_model,
                time_step_s=float(np.clip(base_config.time_step_s * (1 + self._rng.normal(0, 0.2)), 0.0001, 0.01)),
                max_duration_s=base_config.max_duration_s,
                target_position_mm=base_config.target_position_mm,
                max_force_n=float(np.clip(base_config.max_force_n * (1 + self._rng.normal(0, 0.1)), 1.0, 20.0)),
                max_velocity_mm_s=float(
                    np.clip(base_config.max_velocity_mm_s * (1 + self._rng.normal(0, 0.15)), 10.0, 100.0)
                ),
                collision_margin_mm=float(
                    np.clip(base_config.collision_margin_mm * (1 + self._rng.normal(0, 0.1)), 0.5, 5.0)
                ),
                tissue_stiffness_n_per_mm=base_config.tissue_stiffness_n_per_mm,
                seed=base_config.seed + iteration * 100,
            )

            job = self.submit_job(varied_config)
            result = self.execute_job(job)
            all_configs.append((varied_config, result))

            if iteration == 0:
                initial_result = result

            if result.status == JobStatus.COMPLETED:
                entry = {
                    "config_id": varied_config.config_id,
                    "trajectory_error_mm": result.trajectory_error_mm,
                    "max_force_n": result.max_force_n,
                    "safety_score": result.safety_score,
                    "tissue_damage_score": result.tissue_damage_score,
                    "completion_time_s": result.completion_time_s,
                }
                # Simple Pareto check: add if not dominated
                is_dominated = False
                for existing in pareto_front:
                    if (
                        existing["trajectory_error_mm"] <= entry["trajectory_error_mm"]
                        and existing["safety_score"] >= entry["safety_score"]
                        and existing["tissue_damage_score"] <= entry["tissue_damage_score"]
                        and existing != entry
                    ):
                        is_dominated = True
                        break

                if not is_dominated:
                    # Remove dominated entries
                    pareto_front = [
                        e
                        for e in pareto_front
                        if not (
                            entry["trajectory_error_mm"] <= e["trajectory_error_mm"]
                            and entry["safety_score"] >= e["safety_score"]
                            and entry["tissue_damage_score"] <= e["tissue_damage_score"]
                        )
                    ]
                    pareto_front.append(entry)

        # Find best compromise (closest to ideal point)
        best_compromise = {}
        if pareto_front:
            best_idx = 0
            best_dist = float("inf")
            for idx, entry in enumerate(pareto_front):
                dist = (
                    (entry["trajectory_error_mm"] / 2.0) ** 2
                    + (1.0 - entry["safety_score"]) ** 2
                    + entry["tissue_damage_score"] ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            best_compromise = pareto_front[best_idx]

        # Compute improvement over initial
        improvement = 0.0
        if initial_result is not None and best_compromise:
            initial_score = initial_result.safety_score - initial_result.trajectory_error_mm * 0.1
            best_score = best_compromise.get("safety_score", 0) - best_compromise.get("trajectory_error_mm", 0) * 0.1
            if initial_score > 0:
                improvement = (best_score - initial_score) / initial_score * 100.0

        opt_result = OptimizationResult(
            objectives=[o.value for o in objectives],
            pareto_front=pareto_front,
            best_compromise=best_compromise,
            iterations=n_iterations,
            total_simulations=len(all_configs),
            improvement_pct=improvement,
        )

        logger.info(
            "Optimization complete: %d Pareto-optimal configs, %.1f%% improvement",
            len(pareto_front),
            improvement,
        )
        return opt_result

    def get_orchestration_summary(self) -> dict[str, Any]:
        """Get a summary of the orchestration campaign."""
        completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)

        return {
            "state": self._state.value,
            "total_jobs": len(self._jobs),
            "completed_jobs": completed,
            "failed_jobs": failed,
            "total_results": len(self._results),
            "comparisons_performed": len(self._comparisons),
            "frameworks_used": list(set(r.framework.value for r in self._results)),
        }

    def export_audit_hash(self) -> str:
        """Compute an integrity hash over all results for audit."""
        data = json.dumps(
            [r.to_dict() for r in self._results],
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the autonomous simulation orchestrator."""
    logger.info("=" * 80)
    logger.info("Autonomous Simulation Orchestrator Demonstration")
    logger.info("Version: 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 80)

    logger.info(
        "Optional dependencies: HAS_ANTHROPIC=%s, HAS_OPENAI=%s, HAS_MUJOCO=%s, HAS_PYBULLET=%s, HAS_ISAACSIM=%s",
        HAS_ANTHROPIC,
        HAS_OPENAI,
        HAS_MUJOCO,
        HAS_PYBULLET,
        HAS_ISAACSIM,
    )

    # Initialize orchestrator
    orchestrator = AutonomousSimulationOrchestrator(seed=42)

    # Plan a simulation campaign for a biopsy procedure
    logger.info("-" * 60)
    logger.info("Planning simulation campaign for robotic biopsy...")
    configs = orchestrator.plan_campaign(
        procedure=ProcedureType.BIOPSY,
        frameworks=[SimFramework.ISAAC_SIM, SimFramework.MUJOCO, SimFramework.PYBULLET],
        configurations_per_framework=2,
        seed=42,
    )
    logger.info("Planned %d configurations", len(configs))

    # Execute the campaign
    logger.info("-" * 60)
    logger.info("Executing simulation campaign...")
    results = orchestrator.run_campaign(configs)

    for result in results:
        logger.info(
            "  %s [%s]: error=%.2fmm, force=%.1fN, safety=%.2f, time=%.3fs",
            result.framework.value,
            result.status.value,
            result.trajectory_error_mm,
            result.max_force_n,
            result.safety_score,
            result.wall_clock_time_s,
        )

    # Compare frameworks
    logger.info("-" * 60)
    logger.info("Comparing frameworks...")
    comparison = orchestrator.compare_frameworks()
    comp_dict = comparison.to_dict()
    for key, value in comp_dict.items():
        logger.info("  %s: %s", key, value)

    # Optimize best configuration
    logger.info("-" * 60)
    logger.info("Optimizing configuration...")
    best_config = configs[0]
    opt_result = orchestrator.optimize_configuration(
        base_config=best_config,
        n_iterations=4,
    )
    opt_dict = opt_result.to_dict()
    for key, value in opt_dict.items():
        logger.info("  %s: %s", key, value)

    # Summary
    logger.info("-" * 60)
    summary = orchestrator.get_orchestration_summary()
    logger.info("ORCHESTRATION SUMMARY:")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)

    # Audit hash
    audit_hash = orchestrator.export_audit_hash()
    logger.info("Audit hash: %s", audit_hash[:24])

    logger.info("=" * 80)
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
