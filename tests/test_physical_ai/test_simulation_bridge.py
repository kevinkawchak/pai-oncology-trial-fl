"""Tests for physical_ai/simulation_bridge.py.

Covers SimulationFramework/ModelFormat enums, CONVERSION_PATHS,
RobotModel creation, SimulationBridge register/get/list/convert/validate,
unsupported conversion errors, and same-format identity conversion.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("physical_ai.simulation_bridge", "physical_ai/simulation_bridge.py")

SimulationFramework = mod.SimulationFramework
ModelFormat = mod.ModelFormat
CONVERSION_PATHS = mod.CONVERSION_PATHS
RobotModel = mod.RobotModel
SimulationBridge = mod.SimulationBridge


class TestSimulationFrameworkEnum:
    """Tests for SimulationFramework enum."""

    def test_expected_members(self):
        """All simulation frameworks exist."""
        expected = {"isaac_lab", "mujoco", "gazebo", "pybullet"}
        actual = {f.value for f in SimulationFramework}
        assert expected == actual


class TestModelFormatEnum:
    """Tests for ModelFormat enum."""

    def test_expected_members(self):
        """All model formats exist."""
        expected = {"urdf", "mjcf", "sdf", "usd"}
        actual = {f.value for f in ModelFormat}
        assert expected == actual


class TestConversionPaths:
    """Tests for CONVERSION_PATHS constant."""

    def test_urdf_to_mjcf_exists(self):
        """URDF to MJCF conversion path exists."""
        assert (ModelFormat.URDF, ModelFormat.MJCF) in CONVERSION_PATHS

    def test_urdf_to_sdf_exists(self):
        """URDF to SDF conversion path exists."""
        assert (ModelFormat.URDF, ModelFormat.SDF) in CONVERSION_PATHS

    def test_sdf_to_usd_not_supported(self):
        """SDF to USD conversion is not in CONVERSION_PATHS."""
        assert (ModelFormat.SDF, ModelFormat.USD) not in CONVERSION_PATHS


class TestRobotModel:
    """Tests for RobotModel dataclass."""

    def test_creation(self):
        """RobotModel stores attributes correctly."""
        model = RobotModel(
            name="test_arm",
            source_format=ModelFormat.URDF,
            num_joints=6,
            num_links=7,
            mass_kg=15.5,
            joint_limits=[(-3.14, 3.14)] * 6,
        )
        assert model.name == "test_arm"
        assert model.num_joints == 6
        assert model.num_links == 7
        np.testing.assert_allclose(model.mass_kg, 15.5)


class TestSimulationBridgeRegistry:
    """Tests for SimulationBridge model registration."""

    def test_register_and_get(self):
        """Registered model can be retrieved by name."""
        bridge = SimulationBridge()
        model = RobotModel(name="arm_a", source_format=ModelFormat.URDF)
        bridge.register_model(model)
        retrieved = bridge.get_model("arm_a")
        assert retrieved is not None
        assert retrieved.name == "arm_a"

    def test_get_nonexistent_returns_none(self):
        """Getting a non-registered name returns None."""
        bridge = SimulationBridge()
        assert bridge.get_model("nonexistent") is None

    def test_list_models(self):
        """list_models returns registered names."""
        bridge = SimulationBridge()
        bridge.register_model(RobotModel(name="a", source_format=ModelFormat.URDF))
        bridge.register_model(RobotModel(name="b", source_format=ModelFormat.MJCF))
        names = bridge.list_models()
        assert "a" in names
        assert "b" in names
        assert len(names) == 2


class TestSimulationBridgeConvert:
    """Tests for SimulationBridge.convert."""

    def test_same_format_returns_original(self):
        """Converting to the same format returns the original model."""
        bridge = SimulationBridge()
        model = RobotModel(name="arm", source_format=ModelFormat.URDF, num_joints=6)
        result = bridge.convert(model, ModelFormat.URDF)
        assert result is model

    def test_urdf_to_mjcf_conversion(self):
        """URDF to MJCF conversion produces correct format and name."""
        bridge = SimulationBridge()
        model = RobotModel(
            name="arm",
            source_format=ModelFormat.URDF,
            num_joints=6,
            num_links=7,
            mass_kg=10.0,
            joint_limits=[(-1.0, 1.0)] * 6,
        )
        converted = bridge.convert(model, ModelFormat.MJCF)
        assert converted.source_format == ModelFormat.MJCF
        assert converted.name == "arm_mjcf"
        assert converted.num_joints == 6
        assert converted.num_links == 7
        np.testing.assert_allclose(converted.mass_kg, 10.0)

    def test_unsupported_conversion_raises(self):
        """Unsupported conversion path raises ValueError."""
        bridge = SimulationBridge()
        model = RobotModel(name="arm", source_format=ModelFormat.USD)
        with pytest.raises(ValueError, match="No conversion path"):
            bridge.convert(model, ModelFormat.MJCF)

    def test_conversion_metadata_tracks_source(self):
        """Converted model metadata tracks the source format."""
        bridge = SimulationBridge()
        model = RobotModel(name="arm", source_format=ModelFormat.URDF)
        converted = bridge.convert(model, ModelFormat.SDF)
        assert converted.metadata["converted_from"] == "urdf"
        assert "conversion_steps" in converted.metadata


class TestSimulationBridgeValidation:
    """Tests for SimulationBridge.validate_conversion."""

    def test_valid_conversion(self):
        """Matching physics properties pass validation."""
        bridge = SimulationBridge()
        original = RobotModel(
            name="arm",
            source_format=ModelFormat.URDF,
            num_joints=6,
            num_links=7,
            mass_kg=10.0,
            joint_limits=[(-1.0, 1.0)] * 6,
        )
        converted = bridge.convert(original, ModelFormat.MJCF)
        results = bridge.validate_conversion(original, converted)
        assert results["joints_match"] is True
        assert results["links_match"] is True
        assert results["mass_match"] is True
        assert results["joint_limits_match"] is True

    def test_mismatched_joints_fail(self):
        """Different joint counts fail validation."""
        bridge = SimulationBridge()
        original = RobotModel(name="a", source_format=ModelFormat.URDF, num_joints=6)
        other = RobotModel(name="b", source_format=ModelFormat.MJCF, num_joints=4)
        results = bridge.validate_conversion(original, other)
        assert results["joints_match"] is False

    def test_get_supported_conversions(self):
        """get_supported_conversions returns a list of dicts."""
        conversions = SimulationBridge.get_supported_conversions()
        assert isinstance(conversions, list)
        assert len(conversions) == len(CONVERSION_PATHS)
        for entry in conversions:
            assert "from" in entry
            assert "to" in entry
            assert "steps" in entry
