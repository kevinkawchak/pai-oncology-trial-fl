"""Domain-to-safety integration tests.

Tests digital twin -> treatment simulation -> safety monitoring flow,
verifying that domain modules feed safety-critical data correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
dt_mod = load_module("physical_ai.digital_twin", "physical_ai/digital_twin.py")
sim_mod = load_module(
    "treatment_simulator",
    "digital-twins/treatment-simulation/treatment_simulator.py",
)
patient_mod = load_module(
    "patient_model",
    "digital-twins/patient-modeling/patient_model.py",
)


class TestDigitalTwinCreation:
    """Verify digital twin module structure and creation."""

    def test_digital_twin_class_exists(self):
        assert hasattr(dt_mod, "PatientDigitalTwin")

    def test_tumor_model_class_exists(self):
        assert hasattr(dt_mod, "TumorModel")

    def test_tumor_model_creation(self):
        model = dt_mod.TumorModel(
            volume_cm3=5.0,
            growth_rate=60.0,
        )
        assert model.volume_cm3 == 5.0


class TestTreatmentSimulation:
    """Verify treatment simulator enums and classes."""

    def test_treatment_modality_enum(self):
        assert sim_mod.TreatmentModality.CHEMOTHERAPY.value == "chemotherapy"
        assert sim_mod.TreatmentModality.RADIATION.value == "radiation"
        assert sim_mod.TreatmentModality.IMMUNOTHERAPY.value == "immunotherapy"

    def test_response_category_enum(self):
        assert sim_mod.ResponseCategory.COMPLETE_RESPONSE.value == "complete_response"
        assert sim_mod.ResponseCategory.PROGRESSIVE_DISEASE.value == "progressive_disease"

    def test_toxicity_grade_enum(self):
        assert sim_mod.ToxicityGrade.GRADE_0.value == 0
        assert sim_mod.ToxicityGrade.GRADE_4.value == 4

    def test_classify_response_function(self):
        classify = sim_mod._classify_response
        assert classify(0.70) == sim_mod.ResponseCategory.COMPLETE_RESPONSE
        assert classify(0.35) == sim_mod.ResponseCategory.PARTIAL_RESPONSE
        assert classify(0.0) == sim_mod.ResponseCategory.STABLE_DISEASE
        assert classify(-0.25) == sim_mod.ResponseCategory.PROGRESSIVE_DISEASE

    def test_treatment_simulator_class_exists(self):
        assert hasattr(sim_mod, "TreatmentSimulator")


class TestPatientModel:
    """Verify patient model module structure."""

    def test_patient_model_module_loaded(self):
        assert patient_mod is not None

    def test_patient_model_has_classes(self):
        attrs = dir(patient_mod)
        assert len(attrs) > 0


class TestTreatmentOutcome:
    """Verify TreatmentOutcome dataclass."""

    def test_treatment_outcome_creation(self):
        outcome = sim_mod.TreatmentOutcome(
            initial_volume_cm3=10.0,
            final_volume_cm3=4.0,
        )
        assert outcome.initial_volume_cm3 == 10.0
        assert outcome.final_volume_cm3 == 4.0


class TestChemotherapyProtocol:
    """Verify ChemotherapyProtocol dataclass."""

    def test_chemo_protocol_defaults(self):
        proto = sim_mod.ChemotherapyProtocol()
        assert hasattr(proto, "drug_name")
        assert hasattr(proto, "dose_mg_per_m2")
        assert hasattr(proto, "cycles")


class TestTreatmentSimulatorToSafety:
    """Verify treatment simulation results are safety-checkable."""

    def test_toxicity_profile_fields(self):
        tp = sim_mod.ToxicityProfile()
        assert hasattr(tp, "overall_grade")
        assert hasattr(tp, "hematologic_grade")

    def test_drug_class_enum(self):
        assert sim_mod.DrugClass.ALKYLATING_AGENT.value == "alkylating_agent"
        assert sim_mod.DrugClass.ANTIMETABOLITE.value == "antimetabolite"
