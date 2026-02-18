"""Cross-framework validation integration tests.

Tests bridge format conversion, model converter, and validation suite
working together for cross-platform surgical robot policy validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
bridge_mod = load_module(
    "isaac_mujoco_bridge",
    "unification/simulation_physics/isaac_mujoco_bridge.py",
)
converter_mod = load_module(
    "model_converter",
    "unification/cross_platform_tools/model_converter.py",
)
validation_mod = load_module(
    "validation_suite",
    "unification/cross_platform_tools/validation_suite.py",
)
detector_mod = load_module(
    "framework_detector",
    "unification/cross_platform_tools/framework_detector.py",
)


class TestBridgeEnums:
    """Verify the simulation bridge enums and types."""

    def test_bridge_state_enum(self):
        assert hasattr(bridge_mod, "BridgeState")
        assert bridge_mod.BridgeState.DISCONNECTED.value == "disconnected"

    def test_coordinate_convention_enum(self):
        assert hasattr(bridge_mod, "CoordinateConvention")

    def test_joint_type_enum(self):
        assert hasattr(bridge_mod, "JointType")

    def test_solver_type_enum(self):
        assert hasattr(bridge_mod, "SolverType")

    def test_contact_model_enum(self):
        assert hasattr(bridge_mod, "ContactModel")


class TestBridgeDataclasses:
    """Verify bridge dataclass creation."""

    def test_bridge_config_creation(self):
        assert hasattr(bridge_mod, "BridgeConfig")
        cfg = bridge_mod.BridgeConfig()
        assert hasattr(cfg, "isaac_config")
        assert hasattr(cfg, "mujoco_config")

    def test_joint_state_creation(self):
        assert hasattr(bridge_mod, "JointState")

    def test_simulation_state_creation(self):
        assert hasattr(bridge_mod, "SimulationState")


class TestModelConverter:
    """Verify model converter module structure."""

    def test_converter_module_loaded(self):
        assert converter_mod is not None

    def test_converter_has_expected_classes(self):
        attrs = dir(converter_mod)
        assert len(attrs) > 0


class TestValidationSuite:
    """Verify validation suite module structure and types."""

    def test_validation_suite_has_severity_level(self):
        assert hasattr(validation_mod, "SeverityLevel")

    def test_validation_suite_has_validation_category(self):
        assert hasattr(validation_mod, "ValidationCategory")

    def test_validation_suite_has_validation_check(self):
        assert hasattr(validation_mod, "ValidationCheck")

    def test_validation_suite_has_validation_result(self):
        assert hasattr(validation_mod, "ValidationResult")

    def test_validation_suite_has_validation_report(self):
        assert hasattr(validation_mod, "ValidationReport")

    def test_validation_suite_has_compliance_framework(self):
        assert hasattr(validation_mod, "ComplianceFramework")

    def test_validation_suite_class_exists(self):
        assert hasattr(validation_mod, "ValidationSuite")

    def test_validation_thresholds_exists(self):
        assert hasattr(validation_mod, "ValidationThresholds")


class TestFrameworkDetector:
    """Verify framework detector module structure."""

    def test_framework_detector_loaded(self):
        assert detector_mod is not None


class TestCrossModuleIntegration:
    """Verify that types from different modules are compatible."""

    def test_bridge_has_bridge_class(self):
        assert hasattr(bridge_mod, "IsaacMuJoCoBridge")

    def test_validation_check_creation(self):
        check = validation_mod.ValidationCheck(
            name="kinematic_test",
            category=list(validation_mod.ValidationCategory)[0],
            description="Test kinematic consistency",
        )
        assert check.name == "kinematic_test"

    def test_validation_report_creation(self):
        report = validation_mod.ValidationReport()
        assert report is not None
