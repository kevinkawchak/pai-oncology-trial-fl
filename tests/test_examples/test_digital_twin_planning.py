"""Tests for examples/02_digital_twin_planning.py.

Validates treatment plan configuration, simulation parameters,
tumor configuration, patient configuration, and enum definitions.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module(
    "dt_planning",
    "examples/02_digital_twin_planning.py",
)


class TestDigitalTwinEnums:
    """Verify all digital twin planning enums."""

    def test_treatment_modality_values(self):
        assert mod.TreatmentModality.CHEMOTHERAPY.value == "chemotherapy"
        assert mod.TreatmentModality.RADIATION.value == "radiation"
        assert mod.TreatmentModality.IMMUNOTHERAPY.value == "immunotherapy"
        assert mod.TreatmentModality.COMBINATION.value == "combination"

    def test_growth_model_type_values(self):
        assert mod.GrowthModelType.EXPONENTIAL.value == "exponential"
        assert mod.GrowthModelType.LOGISTIC.value == "logistic"
        assert mod.GrowthModelType.GOMPERTZ.value == "gompertz"

    def test_response_category_values(self):
        assert mod.ResponseCategory.COMPLETE_RESPONSE.value == "complete_response"
        assert mod.ResponseCategory.PARTIAL_RESPONSE.value == "partial_response"
        assert mod.ResponseCategory.STABLE_DISEASE.value == "stable_disease"
        assert mod.ResponseCategory.PROGRESSIVE_DISEASE.value == "progressive_disease"

    def test_report_format_values(self):
        assert mod.ReportFormat.TEXT.value == "text"
        assert mod.ReportFormat.JSON.value == "json"
        assert mod.ReportFormat.DATAFRAME.value == "dataframe"

    def test_uncertainty_level_values(self):
        assert mod.UncertaintyLevel.LOW.value == "low"
        assert mod.UncertaintyLevel.MEDIUM.value == "medium"
        assert mod.UncertaintyLevel.HIGH.value == "high"


class TestUncertaintySamplesMap:
    """Verify the uncertainty-to-sample-count mapping."""

    def test_low_samples(self):
        assert mod.UNCERTAINTY_SAMPLES[mod.UncertaintyLevel.LOW] == 50

    def test_medium_samples(self):
        assert mod.UNCERTAINTY_SAMPLES[mod.UncertaintyLevel.MEDIUM] == 200

    def test_high_samples(self):
        assert mod.UNCERTAINTY_SAMPLES[mod.UncertaintyLevel.HIGH] == 1000

    def test_all_levels_mapped(self):
        for level in mod.UncertaintyLevel:
            assert level in mod.UNCERTAINTY_SAMPLES


class TestTumorConfig:
    """Verify TumorConfig dataclass."""

    def test_tumor_config_defaults(self):
        tc = mod.TumorConfig()
        assert tc.volume_cm3 == 3.5
        assert tc.growth_rate == 55.0
        assert tc.chemo_sensitivity == 0.55
        assert tc.radio_sensitivity == 0.65
        assert tc.immuno_sensitivity == 0.35
        assert tc.location == "lung"
        assert tc.histology == "adenocarcinoma"
        assert tc.growth_model == mod.GrowthModelType.GOMPERTZ

    def test_tumor_config_carrying_capacity(self):
        tc = mod.TumorConfig()
        assert tc.carrying_capacity == 100.0

    def test_tumor_config_custom(self):
        tc = mod.TumorConfig(
            volume_cm3=10.0,
            growth_model=mod.GrowthModelType.EXPONENTIAL,
            location="breast",
        )
        assert tc.volume_cm3 == 10.0
        assert tc.growth_model == mod.GrowthModelType.EXPONENTIAL
        assert tc.location == "breast"


class TestPatientConfig:
    """Verify PatientConfig dataclass."""

    def test_patient_config_has_expected_fields(self):
        pc = mod.PatientConfig()
        assert hasattr(pc, "patient_id")
        assert hasattr(pc, "tumor")

    def test_patient_config_tumor_default(self):
        pc = mod.PatientConfig()
        assert isinstance(pc.tumor, mod.TumorConfig)


class TestModuleAttributes:
    """Verify the module exposes expected top-level attributes."""

    def test_has_treatment_modality(self):
        assert hasattr(mod, "TreatmentModality")

    def test_has_growth_model_type(self):
        assert hasattr(mod, "GrowthModelType")

    def test_has_tumor_config(self):
        assert hasattr(mod, "TumorConfig")

    def test_has_patient_config(self):
        assert hasattr(mod, "PatientConfig")
