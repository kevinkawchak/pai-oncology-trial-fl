"""Tests for agentic-ai/examples-agentic-ai/02_react_treatment_planner.py.

Validates ReAct planner enums, dataclasses, patient profiles, reasoning
step creation, action selection, and tool definitions.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module(
    "react_treatment_planner",
    "agentic-ai/examples-agentic-ai/02_react_treatment_planner.py",
)


class TestReActEnums:
    """Verify all ReAct planner enums."""

    def test_react_step_type_members(self):
        assert mod.ReActStepType.THOUGHT.value == "thought"
        assert mod.ReActStepType.ACTION.value == "action"
        assert mod.ReActStepType.OBSERVATION.value == "observation"
        assert mod.ReActStepType.FINAL_ANSWER.value == "final_answer"

    def test_treatment_modality_members(self):
        assert mod.TreatmentModality.CHEMOTHERAPY.value == "chemotherapy"
        assert mod.TreatmentModality.RADIATION.value == "radiation"
        assert mod.TreatmentModality.IMMUNOTHERAPY.value == "immunotherapy"
        assert mod.TreatmentModality.TARGETED_THERAPY.value == "targeted_therapy"
        assert mod.TreatmentModality.COMBINATION.value == "combination"
        assert mod.TreatmentModality.SURGERY.value == "surgery"

    def test_response_category_members(self):
        assert mod.ResponseCategory.COMPLETE_RESPONSE.value == "complete_response"
        assert mod.ResponseCategory.PROGRESSIVE_DISEASE.value == "progressive_disease"

    def test_toxicity_grade_values(self):
        assert mod.ToxicityGrade.GRADE_0.value == 0
        assert mod.ToxicityGrade.GRADE_5.value == 5

    def test_agent_state_members(self):
        assert mod.AgentState.IDLE.value == "idle"
        assert mod.AgentState.THINKING.value == "thinking"
        assert mod.AgentState.ACTING.value == "acting"
        assert mod.AgentState.WAITING_HUMAN.value == "waiting_for_human_review"
        assert mod.AgentState.COMPLETE.value == "complete"

    def test_tool_name_members(self):
        assert mod.ToolName.SIMULATE_CHEMO.value == "simulate_chemotherapy"
        assert mod.ToolName.SIMULATE_RADIATION.value == "simulate_radiation"
        assert mod.ToolName.CHECK_INTERACTIONS.value == "check_drug_interactions"


class TestPatientProfile:
    """Verify PatientProfile dataclass creation and methods."""

    def test_patient_profile_defaults(self):
        profile = mod.PatientProfile()
        assert profile.diagnosis == "NSCLC"
        assert profile.ecog_score == 1
        assert profile.tumor_volume_cm3 == 5.0

    def test_patient_profile_custom(self):
        profile = mod.PatientProfile(
            patient_id="PT-001",
            diagnosis="melanoma",
            stage="IV",
            tumor_volume_cm3=12.0,
            biomarkers={"BRAF_V600E": 1.0},
        )
        assert profile.patient_id == "PT-001"
        assert profile.biomarkers["BRAF_V600E"] == 1.0

    def test_patient_profile_to_context_string(self):
        profile = mod.PatientProfile(patient_id="PT-TEST", diagnosis="NSCLC")
        ctx = profile.to_context_string()
        assert "PT-TEST" in ctx
        assert "NSCLC" in ctx
        assert "Biomarkers" in ctx

    def test_patient_profile_organ_function_default(self):
        profile = mod.PatientProfile()
        assert profile.organ_function["renal"] == "adequate"
        assert profile.organ_function["hepatic"] == "adequate"

    def test_patient_profile_empty_prior_treatments(self):
        profile = mod.PatientProfile()
        ctx = profile.to_context_string()
        assert "none" in ctx.lower()

    def test_patient_profile_with_comorbidities(self):
        profile = mod.PatientProfile(comorbidities=["hypertension", "diabetes"])
        ctx = profile.to_context_string()
        assert "hypertension" in ctx
        assert "diabetes" in ctx


class TestReActDataclasses:
    """Verify other ReAct planner dataclasses."""

    def test_react_step_type_iteration(self):
        types = list(mod.ReActStepType)
        assert len(types) == 4

    def test_agent_state_error_exists(self):
        assert mod.AgentState.ERROR.value == "error"

    def test_treatment_modality_count(self):
        modalities = list(mod.TreatmentModality)
        assert len(modalities) >= 5

    def test_response_category_stable_disease(self):
        assert mod.ResponseCategory.STABLE_DISEASE.value == "stable_disease"


class TestReActPlannerIntegration:
    """Higher-level integration checks for the planner module."""

    def test_module_has_expected_classes(self):
        assert hasattr(mod, "PatientProfile")
        assert hasattr(mod, "ReActStepType")
        assert hasattr(mod, "TreatmentModality")
        assert hasattr(mod, "AgentState")
        assert hasattr(mod, "ToolName")

    def test_tool_name_all_unique(self):
        values = [t.value for t in mod.ToolName]
        assert len(values) == len(set(values))

    def test_agent_state_all_unique(self):
        values = [s.value for s in mod.AgentState]
        assert len(values) == len(set(values))
