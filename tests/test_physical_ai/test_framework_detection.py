"""Tests for physical_ai/framework_detection.py.

Covers FrameworkInfo dataclass, FrameworkDetector creation, detect(),
get_available(), recommend_pipeline() fallback logic, and
get_framework() lookup.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("physical_ai.framework_detection", "physical_ai/framework_detection.py")

FrameworkInfo = mod.FrameworkInfo
FrameworkDetector = mod.FrameworkDetector
_FRAMEWORK_SPECS = mod._FRAMEWORK_SPECS


class TestFrameworkInfo:
    """Tests for the FrameworkInfo dataclass."""

    def test_default_values(self):
        """Default FrameworkInfo is unavailable with empty fields."""
        info = FrameworkInfo(name="test_framework")
        assert info.name == "test_framework"
        assert info.available is False
        assert info.version == ""
        assert info.gpu_capable is False
        assert info.recommended_for == []

    def test_custom_values(self):
        """FrameworkInfo accepts custom values."""
        info = FrameworkInfo(
            name="isaac",
            available=True,
            version="2.0",
            gpu_capable=True,
            recommended_for=["training"],
        )
        assert info.available is True
        assert info.version == "2.0"
        assert info.gpu_capable is True
        assert info.recommended_for == ["training"]


class TestFrameworkSpecs:
    """Tests for _FRAMEWORK_SPECS constant."""

    def test_isaac_lab_in_specs(self):
        """isaac_lab is in framework specs."""
        assert "isaac_lab" in _FRAMEWORK_SPECS

    def test_mujoco_in_specs(self):
        """mujoco is in framework specs."""
        assert "mujoco" in _FRAMEWORK_SPECS

    def test_pybullet_in_specs(self):
        """pybullet is in framework specs."""
        assert "pybullet" in _FRAMEWORK_SPECS

    def test_gazebo_in_specs(self):
        """gazebo is in framework specs."""
        assert "gazebo" in _FRAMEWORK_SPECS

    def test_spec_tuple_structure(self):
        """Each spec is a 4-tuple (import_path, version_attr, gpu, recommended)."""
        for name, spec in _FRAMEWORK_SPECS.items():
            assert len(spec) == 4
            import_path, ver_attr, gpu_capable, recommended = spec
            assert isinstance(import_path, str)
            assert isinstance(ver_attr, str)
            assert isinstance(gpu_capable, bool)
            assert isinstance(recommended, list)


class TestFrameworkDetectorCreation:
    """Tests for FrameworkDetector initialization."""

    def test_initial_state(self):
        """Detector starts with empty frameworks and not detected."""
        detector = FrameworkDetector()
        assert detector._detected is False
        assert detector._frameworks == {}


class TestFrameworkDetectorDetect:
    """Tests for FrameworkDetector.detect()."""

    def test_detect_returns_dict(self):
        """detect() returns a dict of framework names to FrameworkInfo."""
        detector = FrameworkDetector()
        result = detector.detect()
        assert isinstance(result, dict)
        for name in _FRAMEWORK_SPECS:
            assert name in result
            assert isinstance(result[name], FrameworkInfo)

    def test_detect_sets_detected_flag(self):
        """After detect(), _detected is True."""
        detector = FrameworkDetector()
        detector.detect()
        assert detector._detected is True

    def test_isaac_lab_likely_unavailable(self):
        """In test env, isaac_lab is likely not installed."""
        detector = FrameworkDetector()
        result = detector.detect()
        # omni.isaac.lab is typically not available in test environments
        assert result["isaac_lab"].available is False


class TestFrameworkDetectorGetAvailable:
    """Tests for get_available()."""

    def test_get_available_triggers_detect(self):
        """get_available auto-detects if not already done."""
        detector = FrameworkDetector()
        available = detector.get_available()
        assert detector._detected is True
        assert isinstance(available, list)

    def test_all_available_are_marked(self):
        """Every returned FrameworkInfo has available=True."""
        detector = FrameworkDetector()
        available = detector.get_available()
        for info in available:
            assert info.available is True


class TestFrameworkDetectorRecommendPipeline:
    """Tests for recommend_pipeline()."""

    def test_recommend_pipeline_returns_dict(self):
        """Pipeline recommendation is a dict of stage -> framework name."""
        detector = FrameworkDetector()
        pipeline = detector.recommend_pipeline()
        assert isinstance(pipeline, dict)
        assert len(pipeline) > 0

    def test_fallback_when_nothing_available(self):
        """When no frameworks are available, pipeline uses 'simulated'."""
        detector = FrameworkDetector()
        # Force empty detection
        detector._detected = True
        detector._frameworks = {name: FrameworkInfo(name=name, available=False) for name in _FRAMEWORK_SPECS}
        pipeline = detector.recommend_pipeline()
        assert pipeline.get("training") == "simulated"
        assert pipeline.get("validation") == "simulated"
        assert pipeline.get("prototyping") == "simulated"

    def test_recommend_pipeline_auto_detects(self):
        """recommend_pipeline auto-detects frameworks."""
        detector = FrameworkDetector()
        pipeline = detector.recommend_pipeline()
        assert detector._detected is True


class TestFrameworkDetectorGetFramework:
    """Tests for get_framework()."""

    def test_get_known_framework(self):
        """Getting a known framework name returns FrameworkInfo."""
        detector = FrameworkDetector()
        info = detector.get_framework("mujoco")
        assert info is not None
        assert info.name == "mujoco"

    def test_get_unknown_framework_returns_none(self):
        """Getting an unknown name returns None."""
        detector = FrameworkDetector()
        detector.detect()
        info = detector.get_framework("nonexistent_framework")
        assert info is None
