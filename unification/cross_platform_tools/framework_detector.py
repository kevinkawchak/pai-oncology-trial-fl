"""
Cross-Platform Framework Detector for Physical AI Oncology Trial Environments.

Detects installed simulation frameworks, agentic AI orchestrators, ML backends,
and hardware capabilities. Produces a structured inventory report suitable for
federation enrollment and compatibility checking.

DISCLAIMER: RESEARCH USE ONLY
This software is provided for research and educational purposes only.
It has NOT been validated for clinical use, is NOT approved by the FDA
or any other regulatory body, and MUST NOT be used to make clinical
decisions or control physical surgical robots in patient-facing settings.
Use at your own risk.

Copyright (c) 2026 PAI Oncology Trial FL Contributors
License: MIT
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

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
class FrameworkCategory(Enum):
    """Categories of frameworks relevant to the oncology trial platform."""

    SIMULATION = "simulation"
    AGENTIC_AI = "agentic_ai"
    ML_BACKEND = "ml_backend"
    ROBOTICS_MIDDLEWARE = "robotics_middleware"
    DATA_PROCESSING = "data_processing"
    PRIVACY = "privacy"
    VISUALIZATION = "visualization"


class DetectionStatus(Enum):
    """Result of a framework detection attempt."""

    DETECTED = "detected"
    NOT_FOUND = "not_found"
    PARTIAL = "partial"
    ERROR = "error"


class GPUVendor(Enum):
    """GPU hardware vendors."""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FrameworkInfo:
    """Information about a detected framework."""

    name: str
    category: FrameworkCategory
    status: DetectionStatus = DetectionStatus.NOT_FOUND
    version: str = ""
    import_path: str = ""
    location: str = ""
    gpu_required: bool = False
    gpu_available: bool = False
    notes: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    detection_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "version": self.version,
            "import_path": self.import_path,
            "location": self.location,
            "gpu_required": self.gpu_required,
            "gpu_available": self.gpu_available,
            "notes": self.notes,
            "capabilities": self.capabilities,
            "detection_time_ms": round(self.detection_time_ms, 2),
        }


@dataclass
class GPUInfo:
    """Information about available GPU hardware."""

    vendor: GPUVendor = GPUVendor.NONE
    name: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    compute_capability: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "vendor": self.vendor.value,
            "name": self.name,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "memory_total_mb": self.memory_total_mb,
            "memory_available_mb": self.memory_available_mb,
            "compute_capability": self.compute_capability,
        }


@dataclass
class ROSInfo:
    """Information about ROS installation."""

    detected: bool = False
    version: str = ""  # "1" or "2"
    distro: str = ""  # e.g., "humble", "iron", "jazzy"
    workspace: str = ""
    packages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "detected": self.detected,
            "version": self.version,
            "distro": self.distro,
            "workspace": self.workspace,
            "packages": self.packages,
        }


@dataclass
class SystemInfo:
    """Complete system information for federation enrollment."""

    hostname: str = ""
    platform: str = ""
    platform_version: str = ""
    architecture: str = ""
    python_version: str = ""
    python_executable: str = ""
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    ros: ROSInfo = field(default_factory=ROSInfo)
    frameworks: list[FrameworkInfo] = field(default_factory=list)
    detection_timestamp: float = 0.0
    total_detection_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "platform_version": self.platform_version,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "cpu_count": self.cpu_count,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "gpu": self.gpu.to_dict(),
            "ros": self.ros.to_dict(),
            "frameworks": [fw.to_dict() for fw in self.frameworks],
            "detection_timestamp": self.detection_timestamp,
            "total_detection_time_ms": round(self.total_detection_time_ms, 2),
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> dict[str, Any]:
        """Build a high-level summary of detected capabilities."""
        detected = [fw for fw in self.frameworks if fw.status == DetectionStatus.DETECTED]
        by_category: dict[str, list[str]] = {}
        for fw in detected:
            cat = fw.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(fw.name)

        return {
            "total_frameworks_detected": len(detected),
            "total_frameworks_checked": len(self.frameworks),
            "by_category": by_category,
            "gpu_available": self.gpu.vendor != GPUVendor.NONE,
            "ros_available": self.ros.detected,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------
def _detect_python_package(
    name: str,
    import_name: str,
    category: FrameworkCategory,
    gpu_required: bool = False,
    capabilities: Optional[list[str]] = None,
) -> FrameworkInfo:
    """Attempt to import a Python package and extract version info."""
    info = FrameworkInfo(
        name=name,
        category=category,
        import_path=import_name,
        gpu_required=gpu_required,
        capabilities=capabilities or [],
    )

    start = time.time()
    try:
        mod = importlib.import_module(import_name)
        info.status = DetectionStatus.DETECTED
        info.version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))

        # Try to find the package location
        spec = importlib.util.find_spec(import_name)
        if spec and spec.origin:
            info.location = str(Path(spec.origin).parent)

    except ImportError:
        info.status = DetectionStatus.NOT_FOUND
    except (AttributeError, TypeError, OSError) as exc:
        info.status = DetectionStatus.ERROR
        info.notes.append(f"Detection error: {exc}")

    info.detection_time_ms = (time.time() - start) * 1000.0
    return info


def detect_simulation_frameworks() -> list[FrameworkInfo]:
    """Detect installed simulation physics engines."""
    frameworks = []

    # MuJoCo
    mj = _detect_python_package(
        "MuJoCo",
        "mujoco",
        FrameworkCategory.SIMULATION,
        capabilities=["rigid_body", "soft_contact", "tendon", "muscle", "flex_body"],
    )
    frameworks.append(mj)

    # PyBullet
    pb = _detect_python_package(
        "PyBullet",
        "pybullet",
        FrameworkCategory.SIMULATION,
        capabilities=["rigid_body", "urdf", "sequential_impulse"],
    )
    frameworks.append(pb)

    # Isaac Sim (check for isaacsim or omni.isaac)
    isaac = _detect_python_package(
        "Isaac Sim",
        "isaacsim",
        FrameworkCategory.SIMULATION,
        gpu_required=True,
        capabilities=["gpu_parallel", "physx5", "deformable", "photorealistic", "usd"],
    )
    if isaac.status == DetectionStatus.NOT_FOUND:
        # Try alternative import path
        isaac_alt = _detect_python_package(
            "Isaac Sim",
            "omni.isaac.core",
            FrameworkCategory.SIMULATION,
            gpu_required=True,
            capabilities=["gpu_parallel", "physx5", "deformable", "photorealistic", "usd"],
        )
        if isaac_alt.status == DetectionStatus.DETECTED:
            isaac = isaac_alt
    frameworks.append(isaac)

    # Gazebo (check for gz or gazebo Python bindings)
    gz = _detect_python_package(
        "Gazebo",
        "gz.sim",
        FrameworkCategory.SIMULATION,
        capabilities=["ros_integration", "dart", "sdf"],
    )
    if gz.status == DetectionStatus.NOT_FOUND:
        # Check if gz command is available
        if shutil.which("gz") or shutil.which("gazebo"):
            gz.status = DetectionStatus.PARTIAL
            gz.notes.append("Gazebo binary found but Python bindings not available")
            try:
                result = subprocess.run(["gz", "sim", "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gz.version = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
    frameworks.append(gz)

    # SOFA (surgical simulation)
    sofa = _detect_python_package(
        "SOFA Framework",
        "Sofa",
        FrameworkCategory.SIMULATION,
        capabilities=["fem", "soft_tissue", "surgical_sim"],
    )
    frameworks.append(sofa)

    # Drake
    drake = _detect_python_package(
        "Drake",
        "pydrake",
        FrameworkCategory.SIMULATION,
        capabilities=["multibody", "optimization", "contact"],
    )
    frameworks.append(drake)

    return frameworks


def detect_agentic_frameworks() -> list[FrameworkInfo]:
    """Detect installed agentic AI orchestration frameworks."""
    frameworks = []

    crewai = _detect_python_package(
        "CrewAI",
        "crewai",
        FrameworkCategory.AGENTIC_AI,
        capabilities=["sequential", "hierarchical", "tool_use"],
    )
    frameworks.append(crewai)

    langgraph = _detect_python_package(
        "LangGraph",
        "langgraph",
        FrameworkCategory.AGENTIC_AI,
        capabilities=["graph", "stateful", "parallel", "conditional"],
    )
    frameworks.append(langgraph)

    autogen = _detect_python_package(
        "AutoGen",
        "autogen",
        FrameworkCategory.AGENTIC_AI,
        capabilities=["group_chat", "code_execution", "multi_agent"],
    )
    frameworks.append(autogen)

    langchain = _detect_python_package(
        "LangChain",
        "langchain",
        FrameworkCategory.AGENTIC_AI,
        capabilities=["chains", "agents", "retrievers", "memory"],
    )
    frameworks.append(langchain)

    return frameworks


def detect_ml_backends() -> list[FrameworkInfo]:
    """Detect installed machine learning backends."""
    frameworks = []

    pytorch = _detect_python_package(
        "PyTorch",
        "torch",
        FrameworkCategory.ML_BACKEND,
        capabilities=["autograd", "cuda", "distributed"],
    )
    if pytorch.status == DetectionStatus.DETECTED:
        try:
            import torch

            pytorch.gpu_available = torch.cuda.is_available()
            if pytorch.gpu_available:
                pytorch.notes.append(f"CUDA devices: {torch.cuda.device_count()}")
        except Exception:
            pass
    frameworks.append(pytorch)

    jax = _detect_python_package(
        "JAX",
        "jax",
        FrameworkCategory.ML_BACKEND,
        capabilities=["jit", "vmap", "pmap", "autograd"],
    )
    frameworks.append(jax)

    tensorflow = _detect_python_package(
        "TensorFlow",
        "tensorflow",
        FrameworkCategory.ML_BACKEND,
        capabilities=["keras", "distributed", "tflite"],
    )
    frameworks.append(tensorflow)

    sklearn = _detect_python_package(
        "scikit-learn",
        "sklearn",
        FrameworkCategory.ML_BACKEND,
        capabilities=["classification", "regression", "clustering"],
    )
    frameworks.append(sklearn)

    numpy = _detect_python_package(
        "NumPy",
        "numpy",
        FrameworkCategory.DATA_PROCESSING,
        capabilities=["ndarray", "linalg", "fft"],
    )
    frameworks.append(numpy)

    pandas = _detect_python_package(
        "pandas",
        "pandas",
        FrameworkCategory.DATA_PROCESSING,
        capabilities=["dataframe", "csv", "parquet"],
    )
    frameworks.append(pandas)

    return frameworks


def detect_robotics_middleware() -> list[FrameworkInfo]:
    """Detect installed robotics middleware (ROS, etc.)."""
    frameworks = []

    # ROS 2 Python client library
    rclpy = _detect_python_package(
        "ROS 2 (rclpy)",
        "rclpy",
        FrameworkCategory.ROBOTICS_MIDDLEWARE,
        capabilities=["pub_sub", "services", "actions", "parameters"],
    )
    frameworks.append(rclpy)

    # MoveIt
    moveit = _detect_python_package(
        "MoveIt 2",
        "moveit",
        FrameworkCategory.ROBOTICS_MIDDLEWARE,
        capabilities=["motion_planning", "kinematics", "collision"],
    )
    frameworks.append(moveit)

    return frameworks


def detect_gpu() -> GPUInfo:
    """Detect GPU hardware and driver information."""
    info = GPUInfo()

    # Try nvidia-smi for NVIDIA GPUs
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 4:
                    info.vendor = GPUVendor.NVIDIA
                    info.name = parts[0].strip()
                    info.driver_version = parts[1].strip()
                    info.memory_total_mb = int(float(parts[2].strip()))
                    info.memory_available_mb = int(float(parts[3].strip()))
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as exc:
            logger.debug("nvidia-smi detection failed: %s", exc)

    # Try to get CUDA version
    if info.vendor == GPUVendor.NVIDIA:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info.compute_capability = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            import torch

            if torch.cuda.is_available():
                info.cuda_version = torch.version.cuda or ""
        except ImportError:
            pass

    return info


def detect_ros() -> ROSInfo:
    """Detect ROS installation."""
    info = ROSInfo()

    # Check ROS 2 environment variables
    import os

    ros_distro = os.environ.get("ROS_DISTRO", "")
    if ros_distro:
        info.detected = True
        info.distro = ros_distro

        ros_version = os.environ.get("ROS_VERSION", "")
        info.version = ros_version

        ament_prefix = os.environ.get("AMENT_PREFIX_PATH", "")
        if ament_prefix:
            info.workspace = ament_prefix.split(":")[0] if ":" in ament_prefix else ament_prefix

    return info


def detect_system_info() -> SystemInfo:
    """Gather complete system information."""
    import multiprocessing

    info = SystemInfo(
        hostname=platform.node(),
        platform=platform.system(),
        platform_version=platform.version(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        cpu_count=multiprocessing.cpu_count() or 1,
        detection_timestamp=time.time(),
    )

    # Memory detection (Linux)
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info.memory_total_gb = kb / (1024 * 1024)
                    break
    except (FileNotFoundError, ValueError):
        pass

    return info


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------
class FrameworkDetector:
    """Comprehensive framework detection for federated oncology trial sites.

    Scans the environment for simulation engines, agentic AI frameworks,
    ML backends, robotics middleware, and hardware capabilities. Produces
    a structured report suitable for federation enrollment.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def detect_all(self) -> SystemInfo:
        """Run all detection routines and return a complete SystemInfo."""
        start = time.time()
        logger.info("Starting framework detection...")

        system = detect_system_info()
        system.gpu = detect_gpu()
        system.ros = detect_ros()

        # Detect all framework categories
        all_frameworks: list[FrameworkInfo] = []

        logger.info("Detecting simulation frameworks...")
        all_frameworks.extend(detect_simulation_frameworks())

        logger.info("Detecting agentic AI frameworks...")
        all_frameworks.extend(detect_agentic_frameworks())

        logger.info("Detecting ML backends...")
        all_frameworks.extend(detect_ml_backends())

        logger.info("Detecting robotics middleware...")
        all_frameworks.extend(detect_robotics_middleware())

        system.frameworks = all_frameworks
        system.total_detection_time_ms = (time.time() - start) * 1000.0

        detected_count = sum(1 for fw in all_frameworks if fw.status == DetectionStatus.DETECTED)
        logger.info(
            "Detection complete: %d/%d frameworks detected in %.1fms",
            detected_count,
            len(all_frameworks),
            system.total_detection_time_ms,
        )

        return system

    def print_report(self, system: SystemInfo) -> None:
        """Print a human-readable detection report to stdout."""
        print("\n" + "=" * 70)
        print("  Framework Detection Report")
        print("  Physical AI Oncology Trial Federated Learning Platform")
        print("=" * 70)

        print(f"\nSystem: {system.hostname} ({system.platform} {system.architecture})")
        print(f"Python: {system.python_version} ({system.python_executable})")
        print(f"CPUs: {system.cpu_count}  Memory: {system.memory_total_gb:.1f} GB")

        # GPU
        if system.gpu.vendor != GPUVendor.NONE:
            print(f"\nGPU: {system.gpu.name}")
            print(f"  Driver: {system.gpu.driver_version}  CUDA: {system.gpu.cuda_version}")
            print(f"  Memory: {system.gpu.memory_total_mb} MB total, {system.gpu.memory_available_mb} MB free")
        else:
            print("\nGPU: None detected")

        # ROS
        if system.ros.detected:
            print(f"\nROS: version {system.ros.version} ({system.ros.distro})")
        else:
            print("\nROS: Not detected")

        # Frameworks by category
        categories = {}
        for fw in system.frameworks:
            cat = fw.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(fw)

        for cat_name, fw_list in sorted(categories.items()):
            print(f"\n--- {cat_name.replace('_', ' ').title()} ---")
            for fw in fw_list:
                status_icon = {
                    DetectionStatus.DETECTED: "[OK]",
                    DetectionStatus.NOT_FOUND: "[--]",
                    DetectionStatus.PARTIAL: "[??]",
                    DetectionStatus.ERROR: "[!!]",
                }.get(fw.status, "[??]")

                version_str = f" v{fw.version}" if fw.version else ""
                print(f"  {status_icon} {fw.name}{version_str}")

                if self._verbose and fw.notes:
                    for note in fw.notes:
                        print(f"       Note: {note}")
                if self._verbose and fw.capabilities:
                    print(f"       Capabilities: {', '.join(fw.capabilities)}")

        print(f"\nDetection completed in {system.total_detection_time_ms:.1f}ms")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="framework_detector",
        description=(
            "Detect installed frameworks and hardware for the Physical AI Oncology Trial Federated Learning platform."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output with detailed capability information",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        default=False,
        help="Output results as JSON instead of human-readable report",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Write output to a file instead of stdout",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=["simulation", "agentic_ai", "ml_backend", "robotics_middleware", "all"],
        default="all",
        help="Detect only frameworks in the specified category",
    )
    return parser


def main() -> None:
    """CLI entry point for framework detection."""
    parser = build_parser()
    args = parser.parse_args()

    detector = FrameworkDetector(verbose=args.verbose)
    system = detector.detect_all()

    # Filter by category if requested
    if args.category != "all":
        system.frameworks = [fw for fw in system.frameworks if fw.category.value == args.category]

    if args.json:
        output = system.to_json(indent=2)
    else:
        detector.print_report(system)
        return

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        logger.info("Report written to %s", args.output)
    else:
        print(output)


if __name__ == "__main__":
    main()
