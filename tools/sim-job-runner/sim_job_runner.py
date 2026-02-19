"""Cross-framework simulation job launcher for physical AI oncology research.

Provides a unified interface for launching, monitoring, and collecting
results from simulation jobs across multiple robotics and physics
frameworks (NVIDIA Isaac Lab, MuJoCo, PyBullet, Gazebo). Designed to
support surgical robotics policy training and digital-twin validation
in federated oncology trials.

DISCLAIMER: RESEARCH USE ONLY — This tool is intended for research and
educational purposes. It is NOT approved for controlling real robotic
systems or clinical devices. All simulation results must be validated
by qualified engineers before any physical deployment.

VERSION: 0.4.0
LICENSE: MIT
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Framework detection — check which simulation backends are available
# without requiring GPU hardware at import time.
# ---------------------------------------------------------------------------


def _check_importable(module_name: str) -> bool:
    """Return True if *module_name* can be found by importlib."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


# Detect frameworks at module level (cheap spec-only check, no GPU needed).
HAS_ISAAC = _check_importable("omni.isaac.lab") or _check_importable("isaacgym")
HAS_MUJOCO = _check_importable("mujoco")
HAS_PYBULLET = _check_importable("pybullet")
HAS_GAZEBO = _check_importable("gazebo")

# ---------------------------------------------------------------------------
# Reference constants — simulation configuration defaults.
# Source: NVIDIA Isaac Lab documentation, 2024.
# Source: MuJoCo documentation (DeepMind), 2024.
# Source: PyBullet Quickstart Guide, Erwin Coumans.
# Source: Gazebo (Open Robotics) documentation, 2024.
# ---------------------------------------------------------------------------

DEFAULT_SIM_PARAMS: dict[str, dict[str, Any]] = {
    "isaac": {
        "physics_dt": 1.0 / 120.0,
        "rendering_dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, -9.81],
        "gpu_required": True,
        "max_steps": 10000,
        "source": "NVIDIA Isaac Lab docs, 2024",
    },
    "mujoco": {
        "physics_dt": 0.002,
        "rendering_dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, -9.81],
        "gpu_required": False,
        "max_steps": 10000,
        "source": "MuJoCo documentation (DeepMind), 2024",
    },
    "pybullet": {
        "physics_dt": 1.0 / 240.0,
        "rendering_dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, -9.81],
        "gpu_required": False,
        "max_steps": 10000,
        "source": "PyBullet Quickstart Guide, Erwin Coumans",
    },
    "gazebo": {
        "physics_dt": 0.001,
        "rendering_dt": 1.0 / 30.0,
        "gravity": [0.0, 0.0, -9.81],
        "gpu_required": False,
        "max_steps": 10000,
        "source": "Gazebo (Open Robotics) documentation, 2024",
    },
}

# Supported oncology-specific task templates.
# Source: Derived from da Vinci Research Kit (dVRK) task taxonomy.
ONCOLOGY_TASK_TEMPLATES: dict[str, dict[str, Any]] = {
    "needle_insertion": {
        "description": "Simulated biopsy needle insertion with force feedback",
        "frameworks": ["isaac", "mujoco", "pybullet"],
        "max_force_n": 10.0,
        "target_accuracy_mm": 1.0,
        "source": "dVRK task taxonomy; Okamura 2009",
    },
    "tumour_resection": {
        "description": "Soft-tissue tumour resection with deformable body simulation",
        "frameworks": ["isaac", "mujoco"],
        "max_force_n": 20.0,
        "target_accuracy_mm": 2.0,
        "source": "dVRK task taxonomy; Shademan et al. 2016",
    },
    "catheter_navigation": {
        "description": "Endovascular catheter navigation through vasculature",
        "frameworks": ["isaac", "mujoco", "pybullet", "gazebo"],
        "max_force_n": 5.0,
        "target_accuracy_mm": 0.5,
        "source": "CathSim benchmark; Jianu et al. 2022",
    },
    "radiation_positioning": {
        "description": "Patient positioning for radiation therapy delivery",
        "frameworks": ["mujoco", "pybullet", "gazebo"],
        "max_force_n": 50.0,
        "target_accuracy_mm": 3.0,
        "source": "AAPM TG-176 report on patient positioning, 2014",
    },
}

# Bounded parameter ranges for job configuration.
MAX_SIM_STEPS: int = 1_000_000
MAX_NUM_ENVS: int = 16384
MAX_EPISODES: int = 100_000
PHYSICS_DT_BOUNDS: tuple[float, float] = (1e-5, 0.1)


# ── Enums ──────────────────────────────────────────────────────────────────


class SimFramework(str, Enum):
    """Supported simulation frameworks.

    Each framework offers different trade-offs between physics fidelity,
    GPU acceleration, and ecosystem integration.
    """

    ISAAC = "isaac"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    GAZEBO = "gazebo"


class JobStatus(str, Enum):
    """Lifecycle status of a simulation job.

    Jobs progress through these states sequentially; terminal states
    are COMPLETED, FAILED, and CANCELLED.
    """

    PENDING = "pending"
    CONFIGURING = "configuring"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of oncology-specific simulation tasks.

    Each task maps to a template in ONCOLOGY_TASK_TEMPLATES.
    """

    NEEDLE_INSERTION = "needle_insertion"
    TUMOUR_RESECTION = "tumour_resection"
    CATHETER_NAVIGATION = "catheter_navigation"
    RADIATION_POSITIONING = "radiation_positioning"
    CUSTOM = "custom"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class SimJobConfig:
    """Configuration for a simulation job.

    All numeric parameters bounded: envs 1-16384, steps 1-1M,
    episodes 1-100K, physics_dt 1e-5 to 0.1s.
    """

    job_id: str = ""
    framework: str = SimFramework.MUJOCO.value
    task_type: str = TaskType.NEEDLE_INSERTION.value
    num_envs: int = 1
    max_steps: int = 10000
    num_episodes: int = 1
    physics_dt: float = 0.002
    rendering_enabled: bool = False
    seed: int = 42
    custom_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Clamp numeric parameters to safe bounds and assign job ID."""
        if not self.job_id:
            self.job_id = f"sim-{uuid.uuid4().hex[:12]}"
        self.num_envs = max(1, min(self.num_envs, MAX_NUM_ENVS))
        self.max_steps = max(1, min(self.max_steps, MAX_SIM_STEPS))
        self.num_episodes = max(1, min(self.num_episodes, MAX_EPISODES))
        self.physics_dt = max(PHYSICS_DT_BOUNDS[0], min(self.physics_dt, PHYSICS_DT_BOUNDS[1]))


@dataclass
class SimJobResult:
    """Results from a completed simulation job (success_rate bounded 0-1)."""

    job_id: str = ""
    status: str = JobStatus.PENDING.value
    framework: str = ""
    task_type: str = ""
    episodes_completed: int = 0
    total_steps: int = 0
    wall_time_seconds: float = 0.0
    mean_reward: float = 0.0
    success_rate: float = 0.0
    max_force_n: float = 0.0
    positioning_error_mm: float = 0.0
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        """Clamp and set defaults."""
        self.success_rate = max(0.0, min(self.success_rate, 1.0))
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ── Core helpers ───────────────────────────────────────────────────────────


def detect_available_frameworks() -> dict[str, bool]:
    """Return framework name to availability mapping (no GPU init)."""
    return {
        SimFramework.ISAAC.value: HAS_ISAAC,
        SimFramework.MUJOCO.value: HAS_MUJOCO,
        SimFramework.PYBULLET.value: HAS_PYBULLET,
        SimFramework.GAZEBO.value: HAS_GAZEBO,
    }


def validate_job_config(config: SimJobConfig) -> list[str]:
    """Validate a job configuration. Returns list of issues (empty if valid)."""
    issues: list[str] = []
    available = detect_available_frameworks()

    # Check framework availability.
    if config.framework not in [f.value for f in SimFramework]:
        issues.append(f"Unknown framework '{config.framework}'. Supported: {[f.value for f in SimFramework]}.")
    elif not available.get(config.framework, False):
        issues.append(
            f"Framework '{config.framework}' is not installed in the current environment. "
            f"Job will be configured but cannot execute locally."
        )

    # Check GPU requirement.
    fw_defaults = DEFAULT_SIM_PARAMS.get(config.framework, {})
    if fw_defaults.get("gpu_required", False):
        issues.append(
            f"Framework '{config.framework}' requires GPU hardware. Ensure CUDA is available before launching."
        )

    # Check task compatibility.
    if config.task_type != TaskType.CUSTOM.value:
        template = ONCOLOGY_TASK_TEMPLATES.get(config.task_type)
        if template is None:
            issues.append(f"Unknown task type '{config.task_type}'. Supported: {list(ONCOLOGY_TASK_TEMPLATES.keys())}.")
        elif config.framework not in template.get("frameworks", []):
            issues.append(
                f"Task '{config.task_type}' is not supported by framework '{config.framework}'. "
                f"Compatible frameworks: {template['frameworks']}."
            )

    return issues


def launch_simulation(config: SimJobConfig) -> SimJobResult:
    """Launch a simulation job (stub — validates config and returns placeholder result)."""
    issues = validate_job_config(config)
    result = SimJobResult(
        job_id=config.job_id,
        framework=config.framework,
        task_type=config.task_type,
    )

    if issues:
        result.status = JobStatus.CONFIGURING.value
        result.warnings = issues
        logger.warning("Job %s has configuration issues: %s", config.job_id, issues)
    else:
        result.status = JobStatus.PENDING.value
        logger.info(
            "Job %s validated and queued for framework=%s task=%s",
            config.job_id,
            config.framework,
            config.task_type,
        )

    return result


def get_job_status(job_id: str) -> SimJobResult:
    """Retrieve job status (stub — returns PENDING)."""
    logger.info("Querying status for job %s (stub — returning PENDING).", job_id)
    return SimJobResult(job_id=job_id, status=JobStatus.PENDING.value)


def get_job_results(job_id: str) -> SimJobResult:
    """Retrieve completed job results (stub — returns demo metrics)."""
    logger.info("Retrieving results for job %s (stub — returning demo data).", job_id)
    return SimJobResult(
        job_id=job_id,
        status=JobStatus.COMPLETED.value,
        episodes_completed=100,
        total_steps=1_000_000,
        wall_time_seconds=3600.0,
        mean_reward=0.85,
        success_rate=0.92,
        max_force_n=8.5,
        positioning_error_mm=1.2,
    )


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="sim_job_runner",
        description=("Cross-framework simulation job launcher for physical AI oncology research. RESEARCH USE ONLY."),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.4.0")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands.")

    # launch ---
    sp_launch = subparsers.add_parser("launch", help="Launch a new simulation job.")
    sp_launch.add_argument(
        "--framework",
        choices=[f.value for f in SimFramework],
        default="mujoco",
        help="Simulation framework (default: mujoco).",
    )
    sp_launch.add_argument(
        "--task",
        choices=[t.value for t in TaskType],
        default="needle_insertion",
        help="Task type (default: needle_insertion).",
    )
    sp_launch.add_argument("--num-envs", type=int, default=1, help="Parallel environments (default: 1).")
    sp_launch.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode (default: 10000).")
    sp_launch.add_argument("--episodes", type=int, default=1, help="Number of episodes (default: 1).")
    sp_launch.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # status ---
    sp_status = subparsers.add_parser("status", help="Query the status of a running job.")
    sp_status.add_argument("job_id", type=str, help="Job identifier.")

    # results ---
    sp_results = subparsers.add_parser("results", help="Retrieve results from a completed job.")
    sp_results.add_argument("job_id", type=str, help="Job identifier.")

    return parser


def _output(data: Any, as_json: bool) -> None:
    """Print *data* as JSON or human-readable text."""
    if as_json:
        print(json.dumps(asdict(data) if hasattr(data, "__dataclass_fields__") else data, indent=2, default=str))
    else:
        if hasattr(data, "__dataclass_fields__"):
            for k, v in asdict(data).items():
                print(f"  {k}: {v}")
        else:
            print(data)


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the simulation job runner CLI. Returns 0 on success, 1 on failure."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "launch":
            config = SimJobConfig(
                framework=args.framework,
                task_type=args.task,
                num_envs=args.num_envs,
                max_steps=args.max_steps,
                num_episodes=args.episodes,
                seed=args.seed,
            )
            result = launch_simulation(config)
            print(f"  Job ID: {config.job_id}")
            _output(result, args.json)
            return 0

        if args.command == "status":
            result = get_job_status(args.job_id)
            _output(result, args.json)
            return 0

        if args.command == "results":
            result = get_job_results(args.job_id)
            _output(result, args.json)
            return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        logger.exception("Unexpected error (%s): %s", type(exc).__name__, exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
