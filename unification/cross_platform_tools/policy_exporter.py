"""
Cross-Platform Policy Exporter for Surgical Robotics RL/IL Policies.

Exports trained reinforcement learning and imitation learning policies
from one simulation framework's format to another. Supports weight
serialization, observation/action space normalization, and safety
constraint embedding for oncology surgical robotics.

DISCLAIMER: RESEARCH USE ONLY
This software is provided for research and educational purposes only.
It has NOT been validated for clinical use, is NOT approved by the FDA
or any other regulatory body, and MUST NOT be used to make clinical
decisions or control physical surgical robots in patient-facing settings.
All exported policies must be independently validated on the target
platform before any deployment. Use at your own risk.

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
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import onnx

    HAS_ONNX = True
except ImportError:
    onnx = None  # type: ignore[assignment]
    HAS_ONNX = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

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
class PolicyFormat(Enum):
    """Supported policy serialization formats."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    NUMPY = "numpy"
    SAFETENSORS = "safetensors"
    TFLITE = "tflite"


class PolicyType(Enum):
    """Type of learned policy."""

    REINFORCEMENT_LEARNING = "rl"
    IMITATION_LEARNING = "il"
    HYBRID = "hybrid"


class TrainingFramework(Enum):
    """Framework the policy was originally trained in."""

    ISAAC_LAB = "isaac_lab"
    STABLE_BASELINES3 = "stable_baselines3"
    RSL_RL = "rsl_rl"
    ROBOMIMIC = "robomimic"
    CUSTOM = "custom"


class SafetyLevel(Enum):
    """Safety constraint level embedded in the policy."""

    NONE = "none"
    SOFT_LIMITS = "soft_limits"
    HARD_LIMITS = "hard_limits"
    CLINICAL_VALIDATED = "clinical_validated"


class ExportStatus(Enum):
    """Status of a policy export operation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ObservationSpace:
    """Specification of the policy's observation space."""

    dimension: int = 0
    names: list[str] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    normalization_mean: list[float] = field(default_factory=list)
    normalization_std: list[float] = field(default_factory=list)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize an observation vector using stored statistics."""
        if not self.normalization_mean or not self.normalization_std:
            return obs
        mean = np.array(self.normalization_mean)
        std = np.array(self.normalization_std)
        std = np.where(std < 1e-8, 1.0, std)
        return (obs - mean) / std


@dataclass
class ActionSpace:
    """Specification of the policy's action space."""

    dimension: int = 0
    names: list[str] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    action_type: str = "continuous"

    def clip(self, action: np.ndarray) -> np.ndarray:
        """Clip an action vector to the defined bounds."""
        if not self.lower_bounds or not self.upper_bounds:
            return action
        lower = np.array(self.lower_bounds)
        upper = np.array(self.upper_bounds)
        return np.clip(action, lower, upper)


@dataclass
class SafetyConstraints:
    """Safety constraints to embed in the exported policy."""

    max_joint_velocity_rad_s: float = 6.2832
    max_joint_torque_nm: float = 500.0
    max_end_effector_force_n: float = 50.0
    max_end_effector_velocity_m_s: float = 0.5
    workspace_bounds_min: list[float] = field(default_factory=lambda: [-1.0, -1.0, 0.0])
    workspace_bounds_max: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.5])
    emergency_stop_force_n: float = 100.0
    safety_level: SafetyLevel = SafetyLevel.HARD_LIMITS

    def check_action(self, action: np.ndarray, action_space: ActionSpace) -> tuple[np.ndarray, list[str]]:
        """Apply safety constraints to an action. Returns (safe_action, violations)."""
        violations: list[str] = []
        safe_action = action.copy()

        for i, val in enumerate(safe_action):
            if abs(val) > self.max_joint_velocity_rad_s:
                violations.append(f"Joint {i} velocity {val:.3f} exceeds limit {self.max_joint_velocity_rad_s}")
                safe_action[i] = np.sign(val) * self.max_joint_velocity_rad_s

        return safe_action, violations


@dataclass
class PolicyMetadata:
    """Metadata for a trained policy."""

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    policy_type: PolicyType = PolicyType.REINFORCEMENT_LEARNING
    training_framework: TrainingFramework = TrainingFramework.CUSTOM
    algorithm: str = ""
    task_name: str = ""
    observation_space: ObservationSpace = field(default_factory=ObservationSpace)
    action_space: ActionSpace = field(default_factory=ActionSpace)
    safety_constraints: SafetyConstraints = field(default_factory=SafetyConstraints)
    training_steps: int = 0
    training_reward: float = 0.0
    training_success_rate: float = 0.0
    source_engine: str = ""
    source_robot: str = ""
    checkpoint_path: str = ""
    creation_timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute content hash for integrity verification."""
        data = json.dumps(
            {
                "policy_id": self.policy_id,
                "name": self.name,
                "algorithm": self.algorithm,
                "task_name": self.task_name,
                "obs_dim": self.observation_space.dimension,
                "act_dim": self.action_space.dimension,
                "training_steps": self.training_steps,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


@dataclass
class ExportReport:
    """Report generated after a policy export operation."""

    export_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = ""
    source_format: PolicyFormat = PolicyFormat.PYTORCH
    target_format: PolicyFormat = PolicyFormat.ONNX
    target_path: str = ""
    status: ExportStatus = ExportStatus.SUCCESS
    source_hash: str = ""
    target_hash: str = ""
    file_size_bytes: int = 0
    safety_level: SafetyLevel = SafetyLevel.NONE
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    export_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "export_id": self.export_id,
            "policy_id": self.policy_id,
            "source_format": self.source_format.value,
            "target_format": self.target_format.value,
            "target_path": self.target_path,
            "status": self.status.value,
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "file_size_bytes": self.file_size_bytes,
            "safety_level": self.safety_level.value,
            "warnings": self.warnings,
            "errors": self.errors,
            "export_time_ms": round(self.export_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Policy exporter
# ---------------------------------------------------------------------------
class PolicyExporter:
    """Exports trained policies across simulation frameworks.

    Handles weight serialization, observation/action space metadata,
    safety constraint embedding, and audit record generation.
    """

    def __init__(self) -> None:
        self._export_history: list[ExportReport] = []

    def export_to_numpy(
        self,
        weights: dict[str, np.ndarray],
        metadata: PolicyMetadata,
        output_path: str,
    ) -> ExportReport:
        """Export policy weights and metadata as a NumPy .npz archive."""
        report = ExportReport(
            policy_id=metadata.policy_id,
            source_format=PolicyFormat.PYTORCH,
            target_format=PolicyFormat.NUMPY,
            target_path=output_path,
            safety_level=metadata.safety_constraints.safety_level,
        )

        start = time.time()
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Store weights and metadata together
            save_dict: dict[str, Any] = {}
            for key, value in weights.items():
                save_dict[f"weight_{key}"] = value

            save_dict["metadata_json"] = np.array(
                [
                    json.dumps(
                        {
                            "policy_id": metadata.policy_id,
                            "name": metadata.name,
                            "algorithm": metadata.algorithm,
                            "task_name": metadata.task_name,
                            "obs_dim": metadata.observation_space.dimension,
                            "act_dim": metadata.action_space.dimension,
                            "training_steps": metadata.training_steps,
                            "source_engine": metadata.source_engine,
                            "source_robot": metadata.source_robot,
                        }
                    )
                ]
            )

            # Observation normalization stats
            if metadata.observation_space.normalization_mean:
                save_dict["obs_mean"] = np.array(metadata.observation_space.normalization_mean)
                save_dict["obs_std"] = np.array(metadata.observation_space.normalization_std)

            # Action bounds
            if metadata.action_space.lower_bounds:
                save_dict["action_lower"] = np.array(metadata.action_space.lower_bounds)
                save_dict["action_upper"] = np.array(metadata.action_space.upper_bounds)

            np.savez_compressed(str(path), **save_dict)

            report.status = ExportStatus.SUCCESS
            report.source_hash = metadata.compute_hash()

            if path.with_suffix(".npz").exists():
                report.file_size_bytes = path.with_suffix(".npz").stat().st_size
            elif path.exists():
                report.file_size_bytes = path.stat().st_size

        except Exception as exc:
            report.status = ExportStatus.FAILED
            report.errors.append(str(exc))
            logger.error("NumPy export failed: %s", exc)

        report.export_time_ms = (time.time() - start) * 1000.0
        self._export_history.append(report)

        logger.info(
            "Export %s: %s -> numpy (%d bytes) in %.1fms",
            report.status.value,
            metadata.name,
            report.file_size_bytes,
            report.export_time_ms,
        )
        return report

    def export_to_onnx(
        self,
        model: Any,
        metadata: PolicyMetadata,
        output_path: str,
    ) -> ExportReport:
        """Export a PyTorch model to ONNX format."""
        report = ExportReport(
            policy_id=metadata.policy_id,
            source_format=PolicyFormat.PYTORCH,
            target_format=PolicyFormat.ONNX,
            target_path=output_path,
            safety_level=metadata.safety_constraints.safety_level,
        )

        start = time.time()

        if not HAS_TORCH:
            report.status = ExportStatus.FAILED
            report.errors.append("PyTorch not installed; cannot export to ONNX")
            return report

        if not HAS_ONNX:
            report.status = ExportStatus.FAILED
            report.errors.append("ONNX not installed; cannot validate export")
            report.warnings.append("Install onnx with: pip install onnx")
            return report

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            dummy_input = torch.randn(1, metadata.observation_space.dimension)
            torch.onnx.export(
                model,
                dummy_input,
                str(path),
                input_names=["observation"],
                output_names=["action"],
                dynamic_axes={
                    "observation": {0: "batch_size"},
                    "action": {0: "batch_size"},
                },
                opset_version=17,
            )

            report.status = ExportStatus.SUCCESS
            report.source_hash = metadata.compute_hash()
            if path.exists():
                report.file_size_bytes = path.stat().st_size

        except Exception as exc:
            report.status = ExportStatus.FAILED
            report.errors.append(str(exc))

        report.export_time_ms = (time.time() - start) * 1000.0
        self._export_history.append(report)
        return report

    def export_metadata(self, metadata: PolicyMetadata, output_path: str) -> None:
        """Export policy metadata as a standalone JSON file."""
        data = {
            "policy_id": metadata.policy_id,
            "name": metadata.name,
            "policy_type": metadata.policy_type.value,
            "training_framework": metadata.training_framework.value,
            "algorithm": metadata.algorithm,
            "task_name": metadata.task_name,
            "observation_space": {
                "dimension": metadata.observation_space.dimension,
                "names": metadata.observation_space.names,
            },
            "action_space": {
                "dimension": metadata.action_space.dimension,
                "names": metadata.action_space.names,
                "lower_bounds": metadata.action_space.lower_bounds,
                "upper_bounds": metadata.action_space.upper_bounds,
            },
            "safety": {
                "level": metadata.safety_constraints.safety_level.value,
                "max_joint_velocity_rad_s": metadata.safety_constraints.max_joint_velocity_rad_s,
                "max_end_effector_force_n": metadata.safety_constraints.max_end_effector_force_n,
            },
            "training": {
                "steps": metadata.training_steps,
                "reward": metadata.training_reward,
                "success_rate": metadata.training_success_rate,
            },
            "source_engine": metadata.source_engine,
            "source_robot": metadata.source_robot,
            "hash": metadata.compute_hash(),
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

        logger.info("Metadata exported: %s", output_path)

    @property
    def history(self) -> list[ExportReport]:
        """Return export history."""
        return list(self._export_history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate policy export."""
    logger.info("Policy Exporter demonstration")
    logger.info("HAS_TORCH=%s  HAS_ONNX=%s", HAS_TORCH, HAS_ONNX)

    metadata = PolicyMetadata(
        name="needle_biopsy_ppo",
        policy_type=PolicyType.REINFORCEMENT_LEARNING,
        training_framework=TrainingFramework.ISAAC_LAB,
        algorithm="PPO",
        task_name="NeedleBiopsy-v1",
        observation_space=ObservationSpace(
            dimension=48,
            names=["joint_pos_" + str(i) for i in range(6)]
            + ["joint_vel_" + str(i) for i in range(6)]
            + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
            + ["target_x", "target_y", "target_z"]
            + ["force_x", "force_y", "force_z"]
            + ["remaining_" + str(i) for i in range(27)],
            normalization_mean=[0.0] * 48,
            normalization_std=[1.0] * 48,
        ),
        action_space=ActionSpace(
            dimension=6,
            names=[f"joint_vel_{i}" for i in range(6)],
            lower_bounds=[-1.0] * 6,
            upper_bounds=[1.0] * 6,
        ),
        training_steps=5_000_000,
        training_reward=450.0,
        training_success_rate=0.92,
        source_engine="isaac_sim",
        source_robot="ur5e_biopsy",
    )

    exporter = PolicyExporter()

    # Export as NumPy
    weights = {
        "policy.fc1.weight": np.random.randn(256, 48).astype(np.float32),
        "policy.fc1.bias": np.random.randn(256).astype(np.float32),
        "policy.fc2.weight": np.random.randn(128, 256).astype(np.float32),
        "policy.fc2.bias": np.random.randn(128).astype(np.float32),
        "policy.output.weight": np.random.randn(6, 128).astype(np.float32),
        "policy.output.bias": np.random.randn(6).astype(np.float32),
    }

    report = exporter.export_to_numpy(weights, metadata, "/tmp/needle_biopsy_policy")
    logger.info("NumPy export: %s", report.status.value)

    exporter.export_metadata(metadata, "/tmp/needle_biopsy_policy_metadata.json")

    logger.info("Demonstration complete. Exports: %d", len(exporter.history))


if __name__ == "__main__":
    main()
