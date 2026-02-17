"""Simulation framework detection for physical AI integration.

Detects which robotics simulation frameworks are available in the
current environment and recommends a pipeline for training, validation,
and deployment of surgical robotics policies.

Supported frameworks:
- NVIDIA Isaac Lab (GPU-accelerated RL training)
- MuJoCo / MJX (precise physics validation)
- Gazebo (ROS 2 integration)
- PyBullet (rapid prototyping)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FrameworkInfo:
    """Information about a detected simulation framework.

    Attributes:
        name: Framework name.
        available: Whether the framework is importable.
        version: Detected version string (if available).
        gpu_capable: Whether the framework supports GPU acceleration.
        recommended_for: List of pipeline stages this framework is
            recommended for.
    """

    name: str
    available: bool = False
    version: str = ""
    gpu_capable: bool = False
    recommended_for: list[str] = field(default_factory=list)


# Map of framework name -> (import_path, version_attr, gpu_capable, recommended_for)
_FRAMEWORK_SPECS: dict[str, tuple[str, str, bool, list[str]]] = {
    "isaac_lab": (
        "omni.isaac.lab",
        "__version__",
        True,
        ["training", "large-scale-sim"],
    ),
    "mujoco": (
        "mujoco",
        "__version__",
        False,
        ["validation", "physics-accuracy"],
    ),
    "gazebo": (
        "gz.sim",
        "__version__",
        False,
        ["ros2-integration", "hardware-in-the-loop"],
    ),
    "pybullet": (
        "pybullet",
        "__version__",
        False,
        ["rapid-prototyping", "unit-testing"],
    ),
}


class FrameworkDetector:
    """Detects installed simulation frameworks and recommends a pipeline.

    Probes the Python environment for known robotics simulation
    libraries and records their availability and versions.
    """

    def __init__(self) -> None:
        self._frameworks: dict[str, FrameworkInfo] = {}
        self._detected = False

    def detect(self) -> dict[str, FrameworkInfo]:
        """Detect all known simulation frameworks.

        Returns:
            Mapping of framework name -> FrameworkInfo.
        """
        self._frameworks = {}

        for name, (import_path, ver_attr, gpu, recommended) in _FRAMEWORK_SPECS.items():
            info = FrameworkInfo(
                name=name,
                gpu_capable=gpu,
                recommended_for=list(recommended),
            )
            try:
                mod = importlib.import_module(import_path)
                info.available = True
                info.version = getattr(mod, ver_attr, "unknown")
            except (ImportError, ModuleNotFoundError):
                info.available = False

            self._frameworks[name] = info

        self._detected = True
        available = [n for n, f in self._frameworks.items() if f.available]
        logger.info("Detected simulation frameworks: %s", available)
        return dict(self._frameworks)

    def get_available(self) -> list[FrameworkInfo]:
        """Return only frameworks that are currently installed."""
        if not self._detected:
            self.detect()
        return [f for f in self._frameworks.values() if f.available]

    def recommend_pipeline(self) -> dict[str, str]:
        """Recommend a training/validation/deployment pipeline.

        Based on which frameworks are available, selects the best
        option for each pipeline stage.

        Returns:
            Mapping of pipeline stage -> recommended framework name.
        """
        if not self._detected:
            self.detect()

        pipeline: dict[str, str] = {}

        # Training: prefer Isaac Lab (GPU), then MuJoCo, then PyBullet
        for name in ["isaac_lab", "mujoco", "pybullet"]:
            if self._frameworks.get(name, FrameworkInfo(name=name)).available:
                pipeline["training"] = name
                break

        # Validation: prefer MuJoCo (precise), then PyBullet
        for name in ["mujoco", "pybullet"]:
            if self._frameworks.get(name, FrameworkInfo(name=name)).available:
                pipeline["validation"] = name
                break

        # ROS 2 integration: Gazebo only
        if self._frameworks.get("gazebo", FrameworkInfo(name="gazebo")).available:
            pipeline["ros2_integration"] = "gazebo"

        # Rapid prototyping: PyBullet preferred
        for name in ["pybullet", "mujoco"]:
            if self._frameworks.get(name, FrameworkInfo(name=name)).available:
                pipeline["prototyping"] = name
                break

        # Fallback: if nothing is installed, use simulated mode
        if not pipeline:
            pipeline["training"] = "simulated"
            pipeline["validation"] = "simulated"
            pipeline["prototyping"] = "simulated"

        logger.info("Recommended pipeline: %s", pipeline)
        return pipeline

    def get_framework(self, name: str) -> FrameworkInfo | None:
        """Retrieve info for a specific framework."""
        if not self._detected:
            self.detect()
        return self._frameworks.get(name)
