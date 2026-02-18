"""Tests for unification/cross_platform_tools/model_converter.py.

Covers enums (ModelFormat, ConversionStatus, JointType, GeometryType,
MaterialType), dataclasses (Vector3, Quaternion, RobotModel, ConversionReport),
URDFParser, MJCFWriter, and ModelConverter.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module("model_converter", "unification/cross_platform_tools/model_converter.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestModelFormat:
    """Tests for the ModelFormat enum."""

    def test_members(self):
        """ModelFormat has URDF, MJCF, USD, SDF."""
        expected = {"URDF", "MJCF", "USD", "SDF"}
        assert set(mod.ModelFormat.__members__.keys()) == expected

    def test_values(self):
        """ModelFormat values are lowercase."""
        assert mod.ModelFormat.URDF.value == "urdf"
        assert mod.ModelFormat.MJCF.value == "mjcf"


class TestConversionStatus:
    """Tests for the ConversionStatus enum."""

    def test_members(self):
        """ConversionStatus has SUCCESS, PARTIAL, FAILED, SKIPPED."""
        expected = {"SUCCESS", "PARTIAL", "FAILED", "SKIPPED"}
        assert set(mod.ConversionStatus.__members__.keys()) == expected


class TestJointTypeConverter:
    """Tests for the JointType enum in model_converter."""

    def test_members(self):
        """JointType has seven members."""
        assert len(mod.JointType) == 7
        assert mod.JointType.REVOLUTE.value == "revolute"
        assert mod.JointType.BALL.value == "ball"


class TestGeometryType:
    """Tests for the GeometryType enum."""

    def test_members(self):
        """GeometryType has six members."""
        assert len(mod.GeometryType) == 6
        assert mod.GeometryType.BOX.value == "box"
        assert mod.GeometryType.MESH.value == "mesh"


class TestMaterialType:
    """Tests for the MaterialType enum."""

    def test_members(self):
        """MaterialType has three members."""
        expected = {"BASIC", "PBR", "PHONG"}
        assert set(mod.MaterialType.__members__.keys()) == expected


# ---------------------------------------------------------------------------
# Vector3 / Quaternion dataclass tests
# ---------------------------------------------------------------------------
class TestVector3:
    """Tests for the Vector3 dataclass."""

    def test_defaults(self):
        """Default Vector3 is origin."""
        v = mod.Vector3()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_custom(self):
        """Vector3 accepts custom values."""
        v = mod.Vector3(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0


class TestQuaternion:
    """Tests for the Quaternion dataclass."""

    def test_identity(self):
        """Default Quaternion is identity."""
        q = mod.Quaternion()
        assert q.w == 1.0
        assert q.x == 0.0 and q.y == 0.0 and q.z == 0.0


# ---------------------------------------------------------------------------
# RobotModel dataclass
# ---------------------------------------------------------------------------
class TestRobotModel:
    """Tests for the RobotModel dataclass."""

    def test_defaults(self):
        """Default RobotModel has 'robot' name and empty links/joints."""
        rm = mod.RobotModel()
        assert rm.name == "robot"
        assert rm.links == []
        assert rm.joints == []

    def test_get_link_found(self):
        """get_link returns the matching link."""
        link = mod.LinkSpec(name="base_link")
        rm = mod.RobotModel(links=[link])
        result = rm.get_link("base_link")
        assert result is not None
        assert result.name == "base_link"

    def test_get_link_not_found(self):
        """get_link returns None for missing link."""
        rm = mod.RobotModel()
        assert rm.get_link("nonexistent") is None

    def test_get_joint_found(self):
        """get_joint returns the matching joint."""
        joint = mod.JointSpec(name="joint1", joint_type=mod.JointType.REVOLUTE)
        rm = mod.RobotModel(joints=[joint])
        result = rm.get_joint("joint1")
        assert result is not None
        assert result.name == "joint1"

    def test_get_joint_not_found(self):
        """get_joint returns None for missing joint."""
        rm = mod.RobotModel()
        assert rm.get_joint("nonexistent") is None

    def test_compute_hash_deterministic(self):
        """compute_hash is deterministic for the same model."""
        link = mod.LinkSpec(name="base")
        rm = mod.RobotModel(name="robot", links=[link])
        h1 = rm.compute_hash()
        h2 = rm.compute_hash()
        assert h1 == h2
        assert len(h1) == 64


# ---------------------------------------------------------------------------
# ConversionReport dataclass
# ---------------------------------------------------------------------------
class TestConversionReport:
    """Tests for the ConversionReport dataclass."""

    def test_defaults(self):
        """Default report has SUCCESS status."""
        cr = mod.ConversionReport()
        assert cr.status == mod.ConversionStatus.SUCCESS

    def test_to_dict(self):
        """to_dict produces a serializable dictionary."""
        cr = mod.ConversionReport(
            source_format=mod.ModelFormat.URDF,
            target_format=mod.ModelFormat.MJCF,
            status=mod.ConversionStatus.SUCCESS,
        )
        d = cr.to_dict()
        assert d["status"] == "success"
        assert "source_format" in d
        assert "warnings" in d


# ---------------------------------------------------------------------------
# ModelConverter
# ---------------------------------------------------------------------------
class TestModelConverter:
    """Tests for the ModelConverter class."""

    def test_creation(self):
        """ModelConverter can be instantiated."""
        converter = mod.ModelConverter()
        assert converter is not None

    def test_get_supported_conversions(self):
        """Supported conversions returns a non-empty list."""
        converter = mod.ModelConverter()
        conversions = converter.get_supported_conversions()
        assert isinstance(conversions, list)
        assert len(conversions) > 0

    def test_history_empty_initially(self):
        """Conversion history starts empty."""
        converter = mod.ModelConverter()
        assert converter.history == []

    def test_supported_conversions_include_urdf_to_mjcf(self):
        """URDF -> MJCF is a supported conversion path."""
        converter = mod.ModelConverter()
        conversions = converter.get_supported_conversions()
        assert ("urdf", "mjcf") in conversions


# ---------------------------------------------------------------------------
# LinkSpec / JointSpec dataclasses
# ---------------------------------------------------------------------------
class TestLinkSpec:
    """Tests for the LinkSpec dataclass."""

    def test_defaults(self):
        """Default LinkSpec has empty name."""
        ls = mod.LinkSpec()
        assert ls.name == ""

    def test_custom_name(self):
        """LinkSpec accepts a custom name."""
        ls = mod.LinkSpec(name="arm_link")
        assert ls.name == "arm_link"


class TestJointSpec:
    """Tests for the JointSpec dataclass."""

    def test_defaults(self):
        """Default JointSpec has REVOLUTE type."""
        js = mod.JointSpec()
        assert js.joint_type == mod.JointType.REVOLUTE

    def test_custom_type(self):
        """JointSpec accepts different joint types."""
        js = mod.JointSpec(name="slider", joint_type=mod.JointType.PRISMATIC)
        assert js.joint_type == mod.JointType.PRISMATIC
