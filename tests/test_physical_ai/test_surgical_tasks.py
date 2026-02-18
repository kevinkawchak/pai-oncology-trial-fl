"""Tests for physical_ai/surgical_tasks.py.

Covers ProcedureType/SafetyLevel enums, ClinicalThreshold defaults,
SurgicalTaskDefinition fields, STANDARD_TASKS registry,
SurgicalTaskEvaluator evaluation logic, and clinical readiness.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("physical_ai.surgical_tasks", "physical_ai/surgical_tasks.py")

ProcedureType = mod.ProcedureType
SafetyLevel = mod.SafetyLevel
ClinicalThreshold = mod.ClinicalThreshold
SurgicalTaskDefinition = mod.SurgicalTaskDefinition
SurgicalTaskEvaluator = mod.SurgicalTaskEvaluator
STANDARD_TASKS = mod.STANDARD_TASKS


class TestProcedureTypeEnum:
    """Tests for ProcedureType enum."""

    def test_expected_members(self):
        """All expected procedure types exist."""
        expected = {
            "needle_biopsy",
            "tissue_resection",
            "specimen_handling",
            "shunt_placement",
            "lymph_node_dissection",
            "ablation",
        }
        actual = {p.value for p in ProcedureType}
        assert expected == actual


class TestSafetyLevelEnum:
    """Tests for SafetyLevel enum."""

    def test_expected_members(self):
        """All safety levels exist."""
        expected = {"low_risk", "moderate_risk", "high_risk", "critical"}
        actual = {s.value for s in SafetyLevel}
        assert expected == actual


class TestClinicalThreshold:
    """Tests for ClinicalThreshold defaults and custom values."""

    def test_default_values(self):
        """Default thresholds are clinically reasonable."""
        ct = ClinicalThreshold()
        np.testing.assert_allclose(ct.max_position_error_mm, 2.0)
        np.testing.assert_allclose(ct.max_force_n, 5.0)
        np.testing.assert_allclose(ct.min_success_rate, 0.95)
        np.testing.assert_allclose(ct.max_collision_rate, 0.0)
        np.testing.assert_allclose(ct.max_procedure_time_s, 300.0)
        assert ct.safety_level == SafetyLevel.MODERATE_RISK

    def test_custom_values(self):
        """Custom threshold values are stored correctly."""
        ct = ClinicalThreshold(
            max_position_error_mm=1.0,
            max_force_n=3.0,
            safety_level=SafetyLevel.CRITICAL,
        )
        np.testing.assert_allclose(ct.max_position_error_mm, 1.0)
        np.testing.assert_allclose(ct.max_force_n, 3.0)
        assert ct.safety_level == SafetyLevel.CRITICAL


class TestSurgicalTaskDefinition:
    """Tests for SurgicalTaskDefinition dataclass."""

    def test_default_observation_keys(self):
        """Default observation keys include end effector and force."""
        task_def = SurgicalTaskDefinition(procedure=ProcedureType.NEEDLE_BIOPSY)
        assert "end_effector_pos" in task_def.observation_keys
        assert "force_torque" in task_def.observation_keys

    def test_default_reward_weights(self):
        """Default reward weights have expected keys."""
        task_def = SurgicalTaskDefinition(procedure=ProcedureType.ABLATION)
        assert "position_accuracy" in task_def.reward_weights
        assert "collision_penalty" in task_def.reward_weights
        np.testing.assert_allclose(task_def.reward_weights["collision_penalty"], -10.0)

    def test_default_domain_randomization(self):
        """Default domain randomization has tissue_stiffness."""
        task_def = SurgicalTaskDefinition(procedure=ProcedureType.TISSUE_RESECTION)
        assert "tissue_stiffness" in task_def.domain_randomization


class TestStandardTasks:
    """Tests for the STANDARD_TASKS registry."""

    def test_needle_biopsy_registered(self):
        """NEEDLE_BIOPSY is in STANDARD_TASKS."""
        assert ProcedureType.NEEDLE_BIOPSY in STANDARD_TASKS

    def test_needle_biopsy_thresholds(self):
        """Needle biopsy has tight position error and high success rate."""
        task_def = STANDARD_TASKS[ProcedureType.NEEDLE_BIOPSY]
        np.testing.assert_allclose(task_def.thresholds.max_position_error_mm, 1.5)
        np.testing.assert_allclose(task_def.thresholds.min_success_rate, 0.97)
        assert task_def.thresholds.safety_level == SafetyLevel.HIGH_RISK

    def test_specimen_handling_is_low_risk(self):
        """Specimen handling is classified as low risk."""
        task_def = STANDARD_TASKS[ProcedureType.SPECIMEN_HANDLING]
        assert task_def.thresholds.safety_level == SafetyLevel.LOW_RISK

    def test_tissue_resection_is_critical(self):
        """Tissue resection is classified as critical."""
        task_def = STANDARD_TASKS[ProcedureType.TISSUE_RESECTION]
        assert task_def.thresholds.safety_level == SafetyLevel.CRITICAL


class TestSurgicalTaskEvaluator:
    """Tests for SurgicalTaskEvaluator.evaluate."""

    def test_all_pass_scenario(self):
        """All metrics pass when within thresholds."""
        task_def = STANDARD_TASKS[ProcedureType.SPECIMEN_HANDLING]
        evaluator = SurgicalTaskEvaluator(task_def)
        result = evaluator.evaluate(
            position_errors_mm=[1.0, 2.0, 3.0],
            forces_n=[0.5, 1.0],
            collisions=0,
            total_attempts=100,
            procedure_time_s=30.0,
        )
        assert result["position_passed"] is True
        assert result["force_passed"] is True
        assert result["collision_passed"] is True
        assert result["time_passed"] is True
        assert result["clinical_ready"] is True

    def test_position_error_failure(self):
        """Exceeding max position error fails."""
        task_def = STANDARD_TASKS[ProcedureType.NEEDLE_BIOPSY]
        evaluator = SurgicalTaskEvaluator(task_def)
        result = evaluator.evaluate(
            position_errors_mm=[0.5, 0.8, 5.0],  # 5.0 exceeds 1.5
            forces_n=[1.0],
            collisions=0,
            total_attempts=100,
            procedure_time_s=60.0,
        )
        assert result["position_passed"] is False
        assert result["clinical_ready"] is False

    def test_force_failure(self):
        """Exceeding max force fails."""
        task_def = STANDARD_TASKS[ProcedureType.NEEDLE_BIOPSY]
        evaluator = SurgicalTaskEvaluator(task_def)
        result = evaluator.evaluate(
            position_errors_mm=[0.5],
            forces_n=[1.0, 10.0],  # 10.0 exceeds 3.0
            collisions=0,
            total_attempts=100,
            procedure_time_s=60.0,
        )
        assert result["force_passed"] is False
        assert result["clinical_ready"] is False

    def test_collision_rate_failure(self):
        """Any collision fails when max_collision_rate is 0."""
        task_def = STANDARD_TASKS[ProcedureType.NEEDLE_BIOPSY]
        evaluator = SurgicalTaskEvaluator(task_def)
        result = evaluator.evaluate(
            position_errors_mm=[0.5],
            forces_n=[1.0],
            collisions=1,
            total_attempts=100,
            procedure_time_s=60.0,
        )
        assert result["collision_passed"] is False

    def test_time_failure(self):
        """Exceeding max procedure time fails."""
        task_def = STANDARD_TASKS[ProcedureType.NEEDLE_BIOPSY]
        evaluator = SurgicalTaskEvaluator(task_def)
        result = evaluator.evaluate(
            position_errors_mm=[0.5],
            forces_n=[1.0],
            collisions=0,
            total_attempts=100,
            procedure_time_s=999.0,  # exceeds 120
        )
        assert result["time_passed"] is False
        assert result["clinical_ready"] is False

    def test_empty_position_and_force_lists(self):
        """Empty position/force lists default to 0.0, which passes."""
        task_def = STANDARD_TASKS[ProcedureType.SPECIMEN_HANDLING]
        evaluator = SurgicalTaskEvaluator(task_def)
        result = evaluator.evaluate(
            position_errors_mm=[],
            forces_n=[],
            collisions=0,
            total_attempts=1,
            procedure_time_s=10.0,
        )
        np.testing.assert_allclose(result["max_position_error_mm"], 0.0)
        np.testing.assert_allclose(result["max_force_n"], 0.0)
        assert result["clinical_ready"] is True
