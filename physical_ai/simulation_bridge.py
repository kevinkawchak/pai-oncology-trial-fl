"""Cross-platform simulation bridge for physical AI.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Enables interoperability between different simulation frameworks
(Isaac Lab, MuJoCo, Gazebo, PyBullet) by providing model format
conversion and physics consistency validation.

Inspired by the unification approach in physical-ai-oncology-trials,
which eliminates framework lock-in for surgical robot development.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SimulationFramework(str, Enum):
    """Supported simulation frameworks."""

    ISAAC_LAB = "isaac_lab"
    MUJOCO = "mujoco"
    GAZEBO = "gazebo"
    PYBULLET = "pybullet"


class ModelFormat(str, Enum):
    """Robot model description formats."""

    URDF = "urdf"
    MJCF = "mjcf"
    SDF = "sdf"
    USD = "usd"


# Supported conversion paths between model formats
CONVERSION_PATHS: dict[tuple[ModelFormat, ModelFormat], list[str]] = {
    (ModelFormat.URDF, ModelFormat.MJCF): ["parse_urdf", "emit_mjcf"],
    (ModelFormat.URDF, ModelFormat.SDF): ["parse_urdf", "emit_sdf"],
    (ModelFormat.URDF, ModelFormat.USD): ["parse_urdf", "emit_usd"],
    (ModelFormat.MJCF, ModelFormat.URDF): ["parse_mjcf", "emit_urdf"],
    (ModelFormat.SDF, ModelFormat.URDF): ["parse_sdf", "emit_urdf"],
}


@dataclass
class RobotModel:
    """Intermediate representation of a robot model.

    Attributes:
        name: Model name.
        source_format: Original file format.
        num_joints: Number of articulated joints.
        num_links: Number of rigid-body links.
        mass_kg: Total mass in kilograms.
        joint_limits: Per-joint (lower, upper) limits in radians.
        metadata: Arbitrary key-value metadata.
    """

    name: str
    source_format: ModelFormat
    num_joints: int = 0
    num_links: int = 0
    mass_kg: float = 0.0
    joint_limits: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SimulationBridge:
    """Converts robot models between simulation framework formats.

    Maintains an internal registry of validated models and provides
    conversion utilities that preserve physics properties.
    """

    def __init__(self) -> None:
        self._registry: dict[str, RobotModel] = {}

    def register_model(self, model: RobotModel) -> None:
        """Add a robot model to the internal registry."""
        self._registry[model.name] = model
        logger.info(
            "Registered model '%s' (%s, %d joints)",
            model.name,
            model.source_format.value,
            model.num_joints,
        )

    def get_model(self, name: str) -> RobotModel | None:
        """Retrieve a model by name."""
        return self._registry.get(name)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._registry.keys())

    def convert(
        self,
        model: RobotModel,
        target_format: ModelFormat,
    ) -> RobotModel:
        """Convert a robot model to a different format.

        Args:
            model: Source model to convert.
            target_format: Desired output format.

        Returns:
            A new RobotModel in the target format with preserved physics.

        Raises:
            ValueError: If the conversion path is unsupported.
        """
        if model.source_format == target_format:
            return model

        path_key = (model.source_format, target_format)
        if path_key not in CONVERSION_PATHS:
            raise ValueError(
                f"No conversion path from {model.source_format.value} "
                f"to {target_format.value}. "
                f"Supported: {list(CONVERSION_PATHS.keys())}"
            )

        steps = CONVERSION_PATHS[path_key]
        converted = RobotModel(
            name=f"{model.name}_{target_format.value}",
            source_format=target_format,
            num_joints=model.num_joints,
            num_links=model.num_links,
            mass_kg=model.mass_kg,
            joint_limits=list(model.joint_limits),
            metadata={
                **model.metadata,
                "converted_from": model.source_format.value,
                "conversion_steps": steps,
            },
        )

        logger.info(
            "Converted '%s' from %s to %s via %s",
            model.name,
            model.source_format.value,
            target_format.value,
            steps,
        )
        return converted

    def validate_conversion(
        self,
        original: RobotModel,
        converted: RobotModel,
        tolerance: float = 1e-6,
    ) -> dict[str, bool]:
        """Validate that a conversion preserved physical properties.

        Args:
            original: The source model.
            converted: The converted model.
            tolerance: Numerical tolerance for floating-point comparison.

        Returns:
            Dictionary of property names to pass/fail booleans.
        """
        results: dict[str, bool] = {}

        results["joints_match"] = original.num_joints == converted.num_joints
        results["links_match"] = original.num_links == converted.num_links
        results["mass_match"] = abs(original.mass_kg - converted.mass_kg) < tolerance

        if original.joint_limits and converted.joint_limits:
            orig = np.array(original.joint_limits)
            conv = np.array(converted.joint_limits)
            results["joint_limits_match"] = bool(np.allclose(orig, conv, atol=tolerance))
        else:
            results["joint_limits_match"] = len(original.joint_limits) == len(converted.joint_limits)

        all_pass = all(results.values())
        logger.info(
            "Validation %s: %s",
            "PASSED" if all_pass else "FAILED",
            results,
        )
        return results

    @staticmethod
    def get_supported_conversions() -> list[dict[str, str]]:
        """List all supported format conversion paths."""
        return [{"from": src.value, "to": dst.value, "steps": steps} for (src, dst), steps in CONVERSION_PATHS.items()]
