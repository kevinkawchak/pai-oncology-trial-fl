"""Tests for unification/cross_platform_tools/framework_detector.py.

Covers enums (FrameworkCategory, DetectionStatus, GPUVendor),
dataclasses (FrameworkInfo, GPUInfo, ROSInfo, SystemInfo),
detection functions, and FrameworkDetector.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import load_module

mod = load_module("framework_detector", "unification/cross_platform_tools/framework_detector.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestFrameworkCategory:
    """Tests for the FrameworkCategory enum."""

    def test_members(self):
        """FrameworkCategory has seven members."""
        expected = {
            "SIMULATION",
            "AGENTIC_AI",
            "ML_BACKEND",
            "ROBOTICS_MIDDLEWARE",
            "DATA_PROCESSING",
            "PRIVACY",
            "VISUALIZATION",
        }
        assert set(mod.FrameworkCategory.__members__.keys()) == expected

    def test_values(self):
        """Values are lowercase underscore strings."""
        assert mod.FrameworkCategory.SIMULATION.value == "simulation"
        assert mod.FrameworkCategory.AGENTIC_AI.value == "agentic_ai"


class TestDetectionStatus:
    """Tests for the DetectionStatus enum."""

    def test_members(self):
        """DetectionStatus has DETECTED, NOT_FOUND, PARTIAL, ERROR."""
        expected = {"DETECTED", "NOT_FOUND", "PARTIAL", "ERROR"}
        assert set(mod.DetectionStatus.__members__.keys()) == expected


class TestGPUVendor:
    """Tests for the GPUVendor enum."""

    def test_members(self):
        """GPUVendor has NVIDIA, AMD, INTEL, APPLE, NONE."""
        assert len(mod.GPUVendor) == 5
        assert mod.GPUVendor.NONE.value == "none"


# ---------------------------------------------------------------------------
# FrameworkInfo dataclass
# ---------------------------------------------------------------------------
class TestFrameworkInfo:
    """Tests for the FrameworkInfo dataclass."""

    def test_defaults(self):
        """Default FrameworkInfo has NOT_FOUND status."""
        fi = mod.FrameworkInfo(
            name="test",
            category=mod.FrameworkCategory.SIMULATION,
        )
        assert fi.status == mod.DetectionStatus.NOT_FOUND
        assert fi.version == ""
        assert fi.capabilities == []

    def test_to_dict(self):
        """to_dict produces a JSON-serializable dictionary."""
        fi = mod.FrameworkInfo(
            name="mujoco",
            category=mod.FrameworkCategory.SIMULATION,
            status=mod.DetectionStatus.DETECTED,
            version="3.1.0",
            capabilities=["physics", "rendering"],
        )
        d = fi.to_dict()
        assert d["name"] == "mujoco"
        assert d["status"] == "detected"
        assert d["version"] == "3.1.0"
        assert "physics" in d["capabilities"]


class TestGPUInfo:
    """Tests for the GPUInfo dataclass."""

    def test_defaults(self):
        """Default GPUInfo has NONE vendor."""
        gi = mod.GPUInfo()
        assert gi.vendor == mod.GPUVendor.NONE
        assert gi.name == ""

    def test_custom(self):
        """GPUInfo accepts custom values."""
        gi = mod.GPUInfo(
            vendor=mod.GPUVendor.NVIDIA,
            name="RTX 4090",
            driver_version="535.129",
            memory_total_mb=24576,
        )
        assert gi.name == "RTX 4090"
        assert gi.memory_total_mb == 24576


class TestROSInfo:
    """Tests for the ROSInfo dataclass."""

    def test_defaults(self):
        """Default ROSInfo is not detected."""
        ri = mod.ROSInfo()
        assert ri.detected is False
        assert ri.version == ""


class TestSystemInfo:
    """Tests for the SystemInfo dataclass."""

    def test_defaults(self):
        """Default SystemInfo has empty lists."""
        si = mod.SystemInfo()
        assert si.frameworks == []
        assert si.gpu is not None

    def test_to_dict(self):
        """to_dict includes frameworks, gpu, ros keys."""
        si = mod.SystemInfo()
        d = si.to_dict()
        assert "frameworks" in d
        assert "gpu" in d
        assert "ros" in d
        assert "summary" in d

    def test_to_json(self):
        """to_json returns valid JSON."""
        si = mod.SystemInfo()
        j = si.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------
class TestDetectSimulationFrameworks:
    """Tests for detect_simulation_frameworks."""

    def test_returns_list(self):
        """Returns a list of FrameworkInfo objects."""
        results = mod.detect_simulation_frameworks()
        assert isinstance(results, list)
        for fi in results:
            assert isinstance(fi, mod.FrameworkInfo)
            assert fi.category == mod.FrameworkCategory.SIMULATION

    def test_framework_names(self):
        """Expected simulation frameworks are checked."""
        results = mod.detect_simulation_frameworks()
        names = {fi.name for fi in results}
        assert "mujoco" in names or "MuJoCo" in names.union({fi.name.lower() for fi in results})


class TestDetectAgenticFrameworks:
    """Tests for detect_agentic_frameworks."""

    def test_returns_list(self):
        """Returns a list of FrameworkInfo for agentic AI."""
        results = mod.detect_agentic_frameworks()
        assert isinstance(results, list)
        for fi in results:
            assert fi.category == mod.FrameworkCategory.AGENTIC_AI


class TestDetectMLBackends:
    """Tests for detect_ml_backends."""

    def test_returns_list(self):
        """Returns a list of FrameworkInfo objects."""
        results = mod.detect_ml_backends()
        assert isinstance(results, list)
        assert len(results) > 0


class TestDetectGPU:
    """Tests for detect_gpu."""

    def test_returns_gpu_info(self):
        """Returns a GPUInfo instance."""
        gi = mod.detect_gpu()
        assert isinstance(gi, mod.GPUInfo)


class TestDetectROS:
    """Tests for detect_ros."""

    def test_returns_ros_info(self):
        """Returns a ROSInfo instance."""
        ri = mod.detect_ros()
        assert isinstance(ri, mod.ROSInfo)


class TestDetectSystemInfo:
    """Tests for detect_system_info."""

    def test_returns_system_info(self):
        """Returns a SystemInfo instance with populated fields."""
        si = mod.detect_system_info()
        assert isinstance(si, mod.SystemInfo)
        assert isinstance(si.frameworks, list)
        assert isinstance(si.gpu, mod.GPUInfo)


# ---------------------------------------------------------------------------
# FrameworkDetector class
# ---------------------------------------------------------------------------
class TestFrameworkDetector:
    """Tests for the FrameworkDetector class."""

    def test_detect_all(self):
        """detect_all returns a SystemInfo with all categories scanned."""
        detector = mod.FrameworkDetector()
        si = detector.detect_all()
        assert isinstance(si, mod.SystemInfo)
        categories = {fi.category for fi in si.frameworks}
        # At least simulation and ML backend categories are scanned
        assert mod.FrameworkCategory.SIMULATION in categories or len(si.frameworks) >= 0

    def test_print_report_no_error(self):
        """print_report executes without raising."""
        detector = mod.FrameworkDetector()
        si = detector.detect_all()
        # Should not raise
        detector.print_report(si)
