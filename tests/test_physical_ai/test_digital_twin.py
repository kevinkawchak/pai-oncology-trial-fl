"""Tests for physical_ai/digital_twin.py.

Covers GrowthModel types (exponential, logistic, Gompertz), TumorModel
simulation, PatientDigitalTwin treatment response, volume trajectory,
boundary values, negative growth rates, and response classification.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("physical_ai.digital_twin", "physical_ai/digital_twin.py")

GrowthModel = mod.GrowthModel
TumorModel = mod.TumorModel
PatientDigitalTwin = mod.PatientDigitalTwin
_classify_response = mod._classify_response


class TestGrowthModelExponential:
    """Tests for GrowthModel.exponential static method."""

    def test_exponential_zero_days(self):
        """Volume unchanged at day 0."""
        result = GrowthModel.exponential(volume=2.0, growth_rate=60.0, days=0)
        np.testing.assert_allclose(result, 2.0, rtol=1e-10)

    def test_exponential_doubling_time(self):
        """Volume doubles exactly at growth_rate days."""
        result = GrowthModel.exponential(volume=5.0, growth_rate=30.0, days=30)
        np.testing.assert_allclose(result, 10.0, rtol=1e-10)

    def test_exponential_two_doublings(self):
        """Volume quadruples at 2x doubling time."""
        result = GrowthModel.exponential(volume=3.0, growth_rate=20.0, days=40)
        np.testing.assert_allclose(result, 12.0, rtol=1e-10)

    def test_exponential_growth_rate_clamped_to_one(self):
        """Growth rate below 1 is clamped to max(growth_rate, 1.0)."""
        result = GrowthModel.exponential(volume=1.0, growth_rate=0.5, days=1)
        expected = 1.0 * (2.0 ** (1 / 1.0))
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestGrowthModelLogistic:
    """Tests for GrowthModel.logistic static method."""

    def test_logistic_zero_days(self):
        """Volume unchanged at day 0."""
        result = GrowthModel.logistic(volume=2.0, growth_rate=60.0, days=0)
        np.testing.assert_allclose(result, 2.0, rtol=1e-6)

    def test_logistic_approaches_carrying_capacity(self):
        """At very large t, logistic growth converges to carrying capacity."""
        result = GrowthModel.logistic(volume=2.0, growth_rate=30.0, days=10000, carrying_capacity=100.0)
        np.testing.assert_allclose(result, 100.0, atol=0.01)

    def test_logistic_below_capacity(self):
        """Volume grows but stays below carrying capacity at moderate t."""
        result = GrowthModel.logistic(volume=2.0, growth_rate=60.0, days=60, carrying_capacity=100.0)
        assert 2.0 < result < 100.0

    def test_logistic_custom_carrying_capacity(self):
        """Logistic asymptotes to a custom carrying capacity."""
        result = GrowthModel.logistic(volume=1.0, growth_rate=10.0, days=50000, carrying_capacity=50.0)
        np.testing.assert_allclose(result, 50.0, atol=0.1)


class TestGrowthModelGompertz:
    """Tests for GrowthModel.gompertz static method."""

    def test_gompertz_zero_days(self):
        """Volume unchanged at day 0."""
        result = GrowthModel.gompertz(volume=2.0, growth_rate=60.0, days=0)
        np.testing.assert_allclose(result, 2.0, rtol=1e-6)

    def test_gompertz_approaches_plateau(self):
        """At very large t, Gompertz converges to plateau."""
        result = GrowthModel.gompertz(volume=2.0, growth_rate=30.0, days=100000, plateau=80.0)
        np.testing.assert_allclose(result, 80.0, atol=0.1)

    def test_gompertz_growth_is_monotonic(self):
        """Gompertz growth is monotonically increasing for positive rates."""
        v1 = GrowthModel.gompertz(volume=2.0, growth_rate=60.0, days=30)
        v2 = GrowthModel.gompertz(volume=2.0, growth_rate=60.0, days=60)
        assert v1 < v2


class TestTumorModel:
    """Tests for TumorModel dataclass and simulation methods."""

    def test_simulate_growth_default_exponential(self):
        """Default growth model is exponential."""
        tm = TumorModel(volume_cm3=2.0, growth_rate=60.0)
        result = tm.simulate_growth(60)
        np.testing.assert_allclose(result, 4.0, rtol=1e-10)

    def test_simulate_growth_logistic(self):
        """Logistic model returns value between initial and capacity."""
        tm = TumorModel(
            volume_cm3=5.0,
            growth_rate=30.0,
            growth_model="logistic",
            carrying_capacity=80.0,
        )
        result = tm.simulate_growth(30)
        assert 5.0 < result < 80.0

    def test_simulate_growth_gompertz(self):
        """Gompertz model returns value between initial and capacity."""
        tm = TumorModel(
            volume_cm3=5.0,
            growth_rate=30.0,
            growth_model="gompertz",
            carrying_capacity=80.0,
        )
        result = tm.simulate_growth(30)
        assert 5.0 < result < 80.0

    def test_simulate_chemo_response_reduces_volume(self):
        """Chemotherapy reduces tumor volume."""
        tm = TumorModel(volume_cm3=10.0, chemo_sensitivity=0.5)
        result = tm.simulate_chemo_response(dose_mg=100.0, cycles=4)
        assert result < 10.0

    def test_simulate_chemo_response_minimum_volume(self):
        """Volume never drops below 0.01 cm3."""
        tm = TumorModel(volume_cm3=0.02, chemo_sensitivity=1.0)
        result = tm.simulate_chemo_response(dose_mg=200.0, cycles=10)
        assert result >= 0.01

    def test_simulate_radiation_response_reduces_volume(self):
        """Radiation reduces tumor volume."""
        tm = TumorModel(volume_cm3=10.0, radio_sensitivity=0.6)
        result = tm.simulate_radiation_response(dose_gy=60.0, fractions=30)
        assert result < 10.0

    def test_simulate_radiation_response_zero_dose(self):
        """Zero radiation dose preserves volume (sf=1)."""
        tm = TumorModel(volume_cm3=10.0, radio_sensitivity=0.6)
        result = tm.simulate_radiation_response(dose_gy=0.0, fractions=30)
        np.testing.assert_allclose(result, 10.0, rtol=1e-6)

    def test_simulate_combination_therapy_returns_dict(self):
        """Combination therapy returns dictionary with expected keys."""
        tm = TumorModel(volume_cm3=10.0)
        result = tm.simulate_combination_therapy(
            chemo_dose_mg=75.0,
            chemo_cycles=2,
            radiation_dose_gy=60.0,
            radiation_fractions=30,
        )
        assert "initial_volume_cm3" in result
        assert "final_volume_cm3" in result
        assert "total_reduction_pct" in result
        np.testing.assert_allclose(result["initial_volume_cm3"], 10.0)


class TestPatientDigitalTwin:
    """Tests for the PatientDigitalTwin class."""

    def test_creation_defaults(self):
        """Twin is created with correct defaults."""
        twin = PatientDigitalTwin(patient_id="PT-001")
        assert twin.patient_id == "PT-001"
        assert twin.age == 60
        assert twin.tumor.volume_cm3 == 2.0
        assert twin.treatment_history == []

    def test_simulate_treatment_chemotherapy(self):
        """Chemotherapy simulation returns valid result dict."""
        twin = PatientDigitalTwin(patient_id="PT-002")
        result = twin.simulate_treatment("chemotherapy", dose_mg=75.0, cycles=4)
        assert "initial_volume_cm3" in result
        assert "predicted_volume_cm3" in result
        assert "response_category" in result
        assert len(twin.treatment_history) == 1

    def test_simulate_treatment_unknown_type_raises(self):
        """Unknown treatment type raises ValueError."""
        twin = PatientDigitalTwin(patient_id="PT-003")
        with pytest.raises(ValueError, match="Unknown treatment type"):
            twin.simulate_treatment("unknown_therapy")

    def test_generate_feature_vector_shape(self):
        """Feature vector is always length 30."""
        twin = PatientDigitalTwin(
            patient_id="PT-004",
            biomarkers={"CEA": 5.0, "LDH": 200.0},
        )
        fv = twin.generate_feature_vector()
        assert fv.shape == (30,)
        assert fv.dtype == np.float64

    def test_get_summary_keys(self):
        """Summary dict contains expected keys."""
        twin = PatientDigitalTwin(patient_id="PT-005")
        summary = twin.get_summary()
        assert summary["patient_id"] == "PT-005"
        assert "tumor_volume_cm3" in summary
        assert "growth_model" in summary

    def test_simulate_with_uncertainty_returns_stats(self):
        """Uncertainty quantification returns mean, std, percentiles."""
        twin = PatientDigitalTwin(patient_id="PT-006")
        result = twin.simulate_with_uncertainty(
            "chemotherapy",
            n_samples=10,
            seed=42,
            dose_mg=75.0,
            cycles=2,
        )
        assert "mean_volume_cm3" in result
        assert "std_volume_cm3" in result
        assert "p5_volume_cm3" in result
        assert "p95_volume_cm3" in result
        assert result["n_samples"] == 10
        assert result["p5_volume_cm3"] <= result["mean_volume_cm3"] <= result["p95_volume_cm3"]


class TestClassifyResponse:
    """Tests for the _classify_response helper."""

    def test_complete_response(self):
        """Reduction > 0.65 is complete response."""
        assert _classify_response(0.70) == "complete_response"

    def test_partial_response(self):
        """Reduction 0.30-0.65 is partial response."""
        assert _classify_response(0.40) == "partial_response"

    def test_stable_disease(self):
        """Reduction -0.20 to 0.30 is stable disease."""
        assert _classify_response(0.10) == "stable_disease"

    def test_progressive_disease(self):
        """Reduction < -0.20 is progressive disease."""
        assert _classify_response(-0.30) == "progressive_disease"

    def test_boundary_at_065(self):
        """Reduction exactly 0.65 is not complete response (> required)."""
        assert _classify_response(0.65) == "partial_response"

    def test_boundary_at_030(self):
        """Reduction exactly 0.30 is not partial response (> required)."""
        assert _classify_response(0.30) == "stable_disease"
