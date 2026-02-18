"""Tests for examples/01_federated_training_workflow.py.

Validates workflow configuration, pipeline stages, parameter validation,
enum definitions, and dataclass construction.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module(
    "federated_workflow",
    "examples/01_federated_training_workflow.py",
)


class TestWorkflowEnums:
    """Verify all workflow enums."""

    def test_aggregation_mode_values(self):
        assert mod.AggregationMode.FEDAVG.value == "fedavg"
        assert mod.AggregationMode.FEDPROX.value == "fedprox"
        assert mod.AggregationMode.SCAFFOLD.value == "scaffold"

    def test_privacy_level_values(self):
        assert mod.PrivacyLevel.NONE.value == "none"
        assert mod.PrivacyLevel.LOW.value == "low"
        assert mod.PrivacyLevel.MEDIUM.value == "medium"
        assert mod.PrivacyLevel.HIGH.value == "high"

    def test_data_distribution_values(self):
        assert mod.DataDistribution.IID.value == "iid"
        assert mod.DataDistribution.NON_IID.value == "non_iid"

    def test_workflow_stage_values(self):
        assert mod.WorkflowStage.COMPLIANCE_CHECK.value == "compliance_check"
        assert mod.WorkflowStage.SITE_ENROLLMENT.value == "site_enrollment"
        assert mod.WorkflowStage.FEDERATED_TRAINING.value == "federated_training"
        assert mod.WorkflowStage.EVALUATION.value == "evaluation"
        assert mod.WorkflowStage.REPORTING.value == "reporting"


class TestPrivacyEpsilonMap:
    """Verify privacy epsilon mapping."""

    def test_none_has_no_epsilon(self):
        assert mod.PRIVACY_EPSILON_MAP[mod.PrivacyLevel.NONE] is None

    def test_low_epsilon_value(self):
        assert mod.PRIVACY_EPSILON_MAP[mod.PrivacyLevel.LOW] == 8.0

    def test_medium_epsilon_value(self):
        assert mod.PRIVACY_EPSILON_MAP[mod.PrivacyLevel.MEDIUM] == 2.0

    def test_high_epsilon_value(self):
        assert mod.PRIVACY_EPSILON_MAP[mod.PrivacyLevel.HIGH] == 0.5

    def test_all_privacy_levels_mapped(self):
        for level in mod.PrivacyLevel:
            assert level in mod.PRIVACY_EPSILON_MAP


class TestSiteConfig:
    """Verify SiteConfig dataclass."""

    def test_site_config_creation(self):
        sc = mod.SiteConfig(site_id="SITE-A", name="Hospital A")
        assert sc.site_id == "SITE-A"
        assert sc.name == "Hospital A"
        assert sc.patient_count == 200

    def test_site_config_default_capabilities(self):
        sc = mod.SiteConfig(site_id="S1", name="S1")
        assert "imaging" in sc.capabilities
        assert "biopsy" in sc.capabilities

    def test_site_config_custom_patient_count(self):
        sc = mod.SiteConfig(site_id="S2", name="S2", patient_count=500)
        assert sc.patient_count == 500


class TestWorkflowConfig:
    """Verify WorkflowConfig dataclass."""

    def test_workflow_config_has_expected_fields(self):
        cfg = mod.WorkflowConfig()
        assert hasattr(cfg, "trial_id")
        assert hasattr(cfg, "num_rounds")
        assert hasattr(cfg, "num_sites")

    def test_workflow_config_defaults(self):
        cfg = mod.WorkflowConfig()
        assert cfg.num_sites >= 1
        assert cfg.num_rounds >= 1


class TestModuleAttributes:
    """Verify the module exposes expected top-level attributes."""

    def test_has_aggregation_mode(self):
        assert hasattr(mod, "AggregationMode")

    def test_has_privacy_level(self):
        assert hasattr(mod, "PrivacyLevel")

    def test_has_workflow_stage(self):
        assert hasattr(mod, "WorkflowStage")

    def test_has_site_config(self):
        assert hasattr(mod, "SiteConfig")

    def test_has_workflow_config(self):
        assert hasattr(mod, "WorkflowConfig")
