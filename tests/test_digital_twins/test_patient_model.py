"""Tests for digital-twins/patient-modeling/patient_model.py.

Covers TumorProfile creation and clamping, GrowthModel enum,
PatientPhysiologyEngine tumor growth (exponential, logistic, Gompertz,
von Bertalanffy), biomarker simulation, organ function assessment,
treatment tolerability, feature vector generation, factory, and cohort.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "patient_model",
    "digital-twins/patient-modeling/patient_model.py",
)

TumorType = mod.TumorType
GrowthModel = mod.GrowthModel
PatientStatus = mod.PatientStatus
BiomarkerType = mod.BiomarkerType
TumorProfile = mod.TumorProfile
PatientBiomarkers = mod.PatientBiomarkers
OrganFunction = mod.OrganFunction
PatientDigitalTwinConfig = mod.PatientDigitalTwinConfig
PatientPhysiologyEngine = mod.PatientPhysiologyEngine
PatientDigitalTwinFactory = mod.PatientDigitalTwinFactory
compute_tumor_trajectory = mod.compute_tumor_trajectory
MIN_TUMOR_VOLUME_CM3 = mod.MIN_TUMOR_VOLUME_CM3
MAX_TUMOR_VOLUME_CM3 = mod.MAX_TUMOR_VOLUME_CM3
DEFAULT_FEATURE_VECTOR_SIZE = mod.DEFAULT_FEATURE_VECTOR_SIZE


class TestTumorType:
    """Tests for TumorType enum."""

    def test_nsclc_exists(self):
        """NSCLC tumor type exists."""
        assert TumorType.NSCLC.value == "non_small_cell_lung_cancer"

    def test_all_types_present(self):
        """At least 10 tumor types are defined."""
        assert len(TumorType) >= 10


class TestGrowthModelEnum:
    """Tests for GrowthModel enum."""

    def test_expected_models(self):
        """All four growth models exist."""
        expected = {"exponential", "logistic", "gompertz", "von_bertalanffy"}
        actual = {m.value for m in GrowthModel}
        assert expected == actual


class TestTumorProfile:
    """Tests for TumorProfile dataclass and parameter clamping."""

    def test_default_values(self):
        """Defaults produce valid clamped values."""
        tp = TumorProfile()
        assert MIN_TUMOR_VOLUME_CM3 <= tp.volume_cm3 <= MAX_TUMOR_VOLUME_CM3
        assert 0.0 <= tp.chemo_sensitivity <= 1.0
        assert 0.0 <= tp.radio_sensitivity <= 1.0

    def test_negative_volume_clamped(self):
        """Negative volume is clamped to MIN_TUMOR_VOLUME_CM3."""
        tp = TumorProfile(volume_cm3=-5.0)
        np.testing.assert_allclose(tp.volume_cm3, MIN_TUMOR_VOLUME_CM3)

    def test_excessive_volume_clamped(self):
        """Excessive volume is clamped to MAX_TUMOR_VOLUME_CM3."""
        tp = TumorProfile(volume_cm3=99999.0)
        np.testing.assert_allclose(tp.volume_cm3, MAX_TUMOR_VOLUME_CM3)

    def test_sensitivity_clamped_to_unit(self):
        """Sensitivity values > 1.0 are clamped to 1.0."""
        tp = TumorProfile(chemo_sensitivity=2.0, radio_sensitivity=-0.5)
        np.testing.assert_allclose(tp.chemo_sensitivity, 1.0)
        np.testing.assert_allclose(tp.radio_sensitivity, 0.0)

    def test_grade_clamped(self):
        """Grade is clamped between 1 and 4."""
        tp = TumorProfile(grade=10)
        assert tp.grade == 4
        tp2 = TumorProfile(grade=0)
        assert tp2.grade == 1

    def test_negative_doubling_time_auto_computed(self):
        """Negative doubling time is auto-computed from growth rate."""
        tp = TumorProfile(doubling_time_days=-1.0, growth_rate_per_day=0.01)
        expected = math.log(2) / 0.01
        np.testing.assert_allclose(tp.doubling_time_days, expected, rtol=1e-6)


class TestPatientBiomarkers:
    """Tests for PatientBiomarkers."""

    def test_get_value_present(self):
        """get_value returns stored value."""
        bm = PatientBiomarkers(biomarker_values={"CEA": 5.0})
        np.testing.assert_allclose(bm.get_value("CEA"), 5.0)

    def test_get_value_missing(self):
        """get_value returns 0.0 for missing biomarker."""
        bm = PatientBiomarkers()
        np.testing.assert_allclose(bm.get_value("UNKNOWN"), 0.0)

    def test_is_within_normal_true(self):
        """Value within reference range returns True."""
        bm = PatientBiomarkers(
            biomarker_values={"CEA": 3.0},
            reference_ranges={"CEA": (0.0, 5.0)},
        )
        assert bm.is_within_normal("CEA") is True

    def test_is_within_normal_false(self):
        """Value outside reference range returns False."""
        bm = PatientBiomarkers(
            biomarker_values={"CEA": 10.0},
            reference_ranges={"CEA": (0.0, 5.0)},
        )
        assert bm.is_within_normal("CEA") is False

    def test_to_array(self):
        """to_array returns correctly ordered numpy array."""
        bm = PatientBiomarkers(biomarker_values={"A": 1.0, "B": 2.0, "C": 3.0})
        arr = bm.to_array(biomarker_order=["C", "A", "B"])
        np.testing.assert_allclose(arr, [3.0, 1.0, 2.0])


class TestComputeTumorTrajectory:
    """Tests for compute_tumor_trajectory function."""

    def test_exponential_trajectory_shape(self):
        """Exponential trajectory has (days+1) samples for step=1."""
        tp = TumorProfile(growth_model=GrowthModel.EXPONENTIAL, volume_cm3=2.0)
        traj = compute_tumor_trajectory(tp, days=30)
        assert traj.shape == (31,)

    def test_exponential_monotonic_increase(self):
        """Exponential trajectory is monotonically increasing."""
        tp = TumorProfile(growth_model=GrowthModel.EXPONENTIAL, volume_cm3=2.0)
        traj = compute_tumor_trajectory(tp, days=60)
        assert np.all(np.diff(traj) >= 0)

    def test_logistic_bounded_by_capacity(self):
        """Logistic trajectory never exceeds carrying capacity."""
        tp = TumorProfile(
            growth_model=GrowthModel.LOGISTIC,
            volume_cm3=2.0,
            carrying_capacity_cm3=100.0,
        )
        traj = compute_tumor_trajectory(tp, days=500)
        assert np.all(traj <= MAX_TUMOR_VOLUME_CM3)

    def test_gompertz_bounded_by_capacity(self):
        """Gompertz trajectory converges below carrying capacity."""
        tp = TumorProfile(
            growth_model=GrowthModel.GOMPERTZ,
            volume_cm3=2.0,
            carrying_capacity_cm3=50.0,
        )
        traj = compute_tumor_trajectory(tp, days=5000)
        # Final value should be near carrying capacity
        np.testing.assert_allclose(traj[-1], 50.0, atol=1.0)

    def test_von_bertalanffy_trajectory(self):
        """Von Bertalanffy trajectory starts at initial volume."""
        tp = TumorProfile(
            growth_model=GrowthModel.VON_BERTALANFFY,
            volume_cm3=5.0,
        )
        traj = compute_tumor_trajectory(tp, days=30)
        np.testing.assert_allclose(traj[0], 5.0, rtol=1e-6)

    def test_time_step_subsampling(self):
        """time_step_days > 1 reduces output length."""
        tp = TumorProfile(growth_model=GrowthModel.EXPONENTIAL, volume_cm3=2.0)
        traj = compute_tumor_trajectory(tp, days=100, time_step_days=10)
        assert len(traj) == 11  # 0,10,20,...,100


class TestPatientPhysiologyEngine:
    """Tests for PatientPhysiologyEngine."""

    def _make_engine(self, **kwargs):
        """Helper to create a configured engine."""
        config = PatientDigitalTwinConfig(**kwargs)
        return PatientPhysiologyEngine(config)

    def test_simulate_tumor_growth_advances_day(self):
        """simulate_tumor_growth advances current_day."""
        engine = self._make_engine(patient_id="PT-001")
        engine.simulate_tumor_growth(days=30)
        assert engine.current_day == 30

    def test_simulate_tumor_growth_returns_trajectory(self):
        """Returns array with shape matching days."""
        engine = self._make_engine(patient_id="PT-002")
        traj = engine.simulate_tumor_growth(days=10)
        assert traj.shape == (11,)

    def test_simulate_biomarker_evolution(self):
        """Biomarker evolution returns array of correct length."""
        config = PatientDigitalTwinConfig(
            patient_id="PT-003",
            biomarkers=PatientBiomarkers(biomarker_values={"CEA": 5.0}),
        )
        engine = PatientPhysiologyEngine(config)
        bio_traj = engine.simulate_biomarker_evolution("CEA", days=28, time_step_days=7, seed=42)
        assert len(bio_traj) > 0
        # Values should be non-negative
        assert np.all(bio_traj >= 0.0)

    def test_assess_organ_function(self):
        """Organ function assessment returns expected keys."""
        engine = self._make_engine(patient_id="PT-004")
        assessment = engine.assess_organ_function(treatment_day=0)
        assert "hepatic" in assessment
        assert "renal_gfr" in assessment
        assert "cardiac_lvef" in assessment

    def test_organ_function_decays_with_treatment(self):
        """Organ function decreases with longer treatment duration."""
        engine = self._make_engine(patient_id="PT-005")
        day0 = engine.assess_organ_function(treatment_day=0)
        day100 = engine.assess_organ_function(treatment_day=100)
        assert day100["hepatic"] <= day0["hepatic"]

    def test_compute_treatment_tolerability(self):
        """Tolerability is in [0, 1]."""
        engine = self._make_engine(patient_id="PT-006")
        tol = engine.compute_treatment_tolerability(dose_mg=100.0)
        assert 0.0 <= tol <= 1.0

    def test_tolerability_decreases_with_dose(self):
        """Higher dose produces lower tolerability."""
        engine = self._make_engine(patient_id="PT-007")
        tol_low = engine.compute_treatment_tolerability(dose_mg=10.0)
        tol_high = engine.compute_treatment_tolerability(dose_mg=200.0)
        assert tol_high <= tol_low

    def test_generate_feature_vector(self):
        """Feature vector has correct shape and is bounded [0, 1]."""
        engine = self._make_engine(patient_id="PT-008")
        fv = engine.generate_feature_vector()
        assert fv.shape == (DEFAULT_FEATURE_VECTOR_SIZE,)
        assert np.all(fv >= 0.0)
        assert np.all(fv <= 1.0)

    def test_get_summary(self):
        """Summary contains expected keys."""
        engine = self._make_engine(patient_id="PT-009")
        summary = engine.get_summary()
        assert summary["patient_id"] == "PT-009"
        assert "tumor_volume_cm3" in summary
        assert "growth_model" in summary


class TestPatientDigitalTwinFactory:
    """Tests for PatientDigitalTwinFactory."""

    def test_create_twin(self):
        """Factory creates a valid config."""
        factory = PatientDigitalTwinFactory()
        config = factory.create_twin(patient_id="TRIAL-001")
        assert config.patient_id == "TRIAL-001"
        assert factory.created_count == 1

    def test_create_cohort(self):
        """Factory creates a cohort of N patients."""
        factory = PatientDigitalTwinFactory()
        cohort = factory.create_cohort(n_patients=5, seed=42)
        assert len(cohort) == 5
        ids = [c.patient_id for c in cohort]
        assert len(set(ids)) == 5  # All unique

    def test_cohort_uses_tumor_type(self):
        """Cohort members have the specified tumor type."""
        factory = PatientDigitalTwinFactory()
        cohort = factory.create_cohort(
            n_patients=3,
            tumor_type=TumorType.BREAST_DUCTAL,
            seed=0,
        )
        for config in cohort:
            assert config.tumor_profile.tumor_type == TumorType.BREAST_DUCTAL
