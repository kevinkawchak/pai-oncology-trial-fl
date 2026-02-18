"""Tests for examples/03_cross_framework_validation.py.

Validates validation configuration, target engines, conversion paths,
robot configuration, and scope definitions.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module(
    "cross_fw_validation",
    "examples/03_cross_framework_validation.py",
)


class TestCrossFrameworkEnums:
    """Verify all cross-framework validation enums."""

    def test_target_engine_values(self):
        assert mod.TargetEngine.ISAAC_SIM.value == "isaac_sim"
        assert mod.TargetEngine.MUJOCO.value == "mujoco"
        assert mod.TargetEngine.PYBULLET.value == "pybullet"

    def test_conversion_path_values(self):
        assert mod.ConversionPath.URDF_TO_MJCF.value == "urdf_to_mjcf"
        assert mod.ConversionPath.MJCF_TO_URDF.value == "mjcf_to_urdf"
        assert mod.ConversionPath.URDF_TO_SDF.value == "urdf_to_sdf"
        assert mod.ConversionPath.URDF_TO_USD.value == "urdf_to_usd"

    def test_validation_scope_values(self):
        assert mod.ValidationScope.KINEMATICS.value == "kinematics"
        assert mod.ValidationScope.SAFETY.value == "safety"
        assert mod.ValidationScope.PERFORMANCE.value == "performance"
        assert mod.ValidationScope.FULL.value == "full"


class TestEngineAvailability:
    """Verify ENGINE_AVAILABILITY dictionary structure."""

    def test_engine_availability_has_all_engines(self):
        for engine in mod.TargetEngine:
            assert engine in mod.ENGINE_AVAILABILITY

    def test_engine_availability_values_are_bool(self):
        for engine, avail in mod.ENGINE_AVAILABILITY.items():
            assert isinstance(avail, bool)


class TestRobotConfig:
    """Verify RobotConfig dataclass."""

    def test_robot_config_defaults(self):
        rc = mod.RobotConfig()
        assert rc.name == "surgical_arm_6dof"
        assert rc.num_joints == 6
        assert rc.num_links == 7
        assert rc.mass_kg == 12.5

    def test_robot_config_joint_names(self):
        rc = mod.RobotConfig()
        assert len(rc.joint_names) == 6
        assert "shoulder_pan" in rc.joint_names
        assert "elbow" in rc.joint_names

    def test_robot_config_joint_limits(self):
        rc = mod.RobotConfig()
        assert len(rc.joint_limits) == 6
        for low, high in rc.joint_limits:
            assert low < high

    def test_robot_config_safety_limits(self):
        rc = mod.RobotConfig()
        assert rc.max_force_n == 40.0
        assert rc.max_velocity_rad_s == 4.0

    def test_robot_config_workspace_bounds(self):
        rc = mod.RobotConfig()
        for i in range(3):
            assert rc.workspace_min[i] < rc.workspace_max[i]

    def test_robot_config_source_format(self):
        rc = mod.RobotConfig()
        assert rc.source_format == "urdf"


class TestValidationConfig:
    """Verify ValidationConfig dataclass."""

    def test_validation_config_defaults(self):
        vc = mod.ValidationConfig()
        assert vc.session_name == "cross_framework_validation"
        assert vc.source_engine == mod.TargetEngine.ISAAC_SIM
        assert isinstance(vc.robot, mod.RobotConfig)

    def test_validation_config_target_engines(self):
        vc = mod.ValidationConfig()
        assert mod.TargetEngine.MUJOCO in vc.target_engines
        assert mod.TargetEngine.PYBULLET in vc.target_engines


class TestModuleAttributes:
    """Verify the module exposes expected attributes."""

    def test_has_target_engine(self):
        assert hasattr(mod, "TargetEngine")

    def test_has_conversion_path(self):
        assert hasattr(mod, "ConversionPath")

    def test_has_validation_scope(self):
        assert hasattr(mod, "ValidationScope")

    def test_has_robot_config(self):
        assert hasattr(mod, "RobotConfig")

    def test_has_validation_config(self):
        assert hasattr(mod, "ValidationConfig")
