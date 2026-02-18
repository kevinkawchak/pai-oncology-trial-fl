"""Tests for digital-twins/treatment-simulation/treatment_simulator.py.

Covers TreatmentSimulator creation, chemotherapy PK/PD simulation,
radiation LQ model, immunotherapy stochastic response, combination
therapy, _compute_pk_concentration, _compute_auc, _classify_response,
ToxicityProfile, and RECIST 1.1 classification.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "treatment_simulator",
    "digital-twins/treatment-simulation/treatment_simulator.py",
)

TreatmentModality = mod.TreatmentModality
ResponseCategory = mod.ResponseCategory
ToxicityGrade = mod.ToxicityGrade
DrugClass = mod.DrugClass
ChemotherapyProtocol = mod.ChemotherapyProtocol
RadiationPlan = mod.RadiationPlan
ImmunotherapyRegimen = mod.ImmunotherapyRegimen
TreatmentOutcome = mod.TreatmentOutcome
ToxicityProfile = mod.ToxicityProfile
TreatmentSimulator = mod.TreatmentSimulator
_classify_response = mod._classify_response
_compute_pk_concentration = mod._compute_pk_concentration
_compute_auc = mod._compute_auc

MIN_VOLUME_CM3 = mod.MIN_VOLUME_CM3
MAX_VOLUME_CM3 = mod.MAX_VOLUME_CM3
MAX_DOSE_MG = mod.MAX_DOSE_MG
PK_ELIMINATION_HALF_LIFE_HOURS = mod.PK_ELIMINATION_HALF_LIFE_HOURS
PK_VOLUME_OF_DISTRIBUTION_L = mod.PK_VOLUME_OF_DISTRIBUTION_L
PK_BIOAVAILABILITY = mod.PK_BIOAVAILABILITY


class TestClassifyResponse:
    """Tests for _classify_response helper (RECIST 1.1 criteria)."""

    def test_complete_response(self):
        """Reduction >= 0.65 is complete response."""
        assert _classify_response(0.70) == ResponseCategory.COMPLETE_RESPONSE

    def test_partial_response(self):
        """Reduction 0.30-0.65 is partial response."""
        assert _classify_response(0.40) == ResponseCategory.PARTIAL_RESPONSE

    def test_stable_disease(self):
        """Reduction -0.20 to 0.30 is stable disease."""
        assert _classify_response(0.10) == ResponseCategory.STABLE_DISEASE

    def test_progressive_disease(self):
        """Reduction < -0.20 is progressive disease."""
        assert _classify_response(-0.30) == ResponseCategory.PROGRESSIVE_DISEASE

    def test_boundary_065(self):
        """Reduction exactly 0.65 is complete response (>= check)."""
        assert _classify_response(0.65) == ResponseCategory.COMPLETE_RESPONSE

    def test_boundary_030(self):
        """Reduction exactly 0.30 is partial response (>= check)."""
        assert _classify_response(0.30) == ResponseCategory.PARTIAL_RESPONSE


class TestComputePkConcentration:
    """Tests for _compute_pk_concentration (one-compartment PK model)."""

    def test_concentration_at_time_zero(self):
        """At t=0, concentration equals C0 = (dose * bioavailability) / Vd."""
        c0 = _compute_pk_concentration(dose_mg=100.0, time_hours=0.0)
        expected = (100.0 * PK_BIOAVAILABILITY) / PK_VOLUME_OF_DISTRIBUTION_L
        np.testing.assert_allclose(c0, expected, rtol=1e-10)

    def test_concentration_decays(self):
        """Concentration at later time is less than at time zero."""
        c0 = _compute_pk_concentration(dose_mg=100.0, time_hours=0.0)
        c6 = _compute_pk_concentration(dose_mg=100.0, time_hours=6.0)
        assert c6 < c0

    def test_concentration_at_half_life(self):
        """At one half-life, concentration is approximately C0/2."""
        c0 = _compute_pk_concentration(dose_mg=100.0, time_hours=0.0)
        c_half = _compute_pk_concentration(
            dose_mg=100.0,
            time_hours=PK_ELIMINATION_HALF_LIFE_HOURS,
        )
        np.testing.assert_allclose(c_half, c0 / 2.0, rtol=1e-6)

    def test_zero_dose(self):
        """Zero dose yields zero concentration."""
        c = _compute_pk_concentration(dose_mg=0.0, time_hours=0.0)
        np.testing.assert_allclose(c, 0.0)

    def test_negative_time_clamped(self):
        """Negative time is treated as max(time, 0)."""
        c = _compute_pk_concentration(dose_mg=100.0, time_hours=-5.0)
        c0 = _compute_pk_concentration(dose_mg=100.0, time_hours=0.0)
        np.testing.assert_allclose(c, c0)


class TestComputeAuc:
    """Tests for _compute_auc (trapezoidal AUC over 72 hours)."""

    def test_auc_positive_for_nonzero_dose(self):
        """AUC is positive for a nonzero dose."""
        auc = _compute_auc(dose_mg=100.0)
        assert auc > 0.0

    def test_auc_zero_for_zero_dose(self):
        """AUC is 0 for zero dose."""
        auc = _compute_auc(dose_mg=0.0)
        np.testing.assert_allclose(auc, 0.0)

    def test_auc_scales_with_dose(self):
        """Doubling dose approximately doubles AUC (linear PK)."""
        auc_50 = _compute_auc(dose_mg=50.0)
        auc_100 = _compute_auc(dose_mg=100.0)
        np.testing.assert_allclose(auc_100 / auc_50, 2.0, rtol=1e-6)

    def test_auc_mathematical_correctness(self):
        """AUC approximates analytical integral C0/ke for infinite time."""
        dose = 100.0
        ke = np.log(2) / PK_ELIMINATION_HALF_LIFE_HOURS
        c0 = (dose * PK_BIOAVAILABILITY) / PK_VOLUME_OF_DISTRIBUTION_L
        analytical_auc = c0 / ke  # integral from 0 to infinity
        numerical_auc = _compute_auc(dose_mg=dose)
        # 72-hour window captures most of the AUC
        np.testing.assert_allclose(numerical_auc, analytical_auc, rtol=0.01)


class TestChemotherapyProtocol:
    """Tests for ChemotherapyProtocol dataclass."""

    def test_defaults(self):
        """Default protocol has valid parameters."""
        proto = ChemotherapyProtocol()
        assert proto.drug_name == "carboplatin"
        assert proto.drug_class == DrugClass.PLATINUM
        assert 1 <= proto.cycles <= 12

    def test_dose_clamped(self):
        """Excessive dose is clamped."""
        proto = ChemotherapyProtocol(dose_mg_per_m2=500.0)
        np.testing.assert_allclose(proto.dose_mg_per_m2, MAX_DOSE_MG)

    def test_cycles_clamped(self):
        """Cycles beyond max are clamped."""
        proto = ChemotherapyProtocol(cycles=100)
        assert proto.cycles == 12


class TestRadiationPlan:
    """Tests for RadiationPlan dataclass."""

    def test_dose_per_fraction_computed(self):
        """dose_per_fraction_gy is auto-computed from total/fractions."""
        plan = RadiationPlan(total_dose_gy=60.0, fractions=30)
        np.testing.assert_allclose(plan.dose_per_fraction_gy, 2.0)


class TestTreatmentSimulatorCreation:
    """Tests for TreatmentSimulator initialization."""

    def test_default_creation(self):
        """Default simulator has valid clamped parameters."""
        sim = TreatmentSimulator()
        assert MIN_VOLUME_CM3 <= sim.initial_volume_cm3 <= MAX_VOLUME_CM3
        assert 0.0 <= sim.chemo_sensitivity <= 1.0
        assert sim.simulation_count == 0

    def test_volume_clamped(self):
        """Negative volume is clamped to MIN_VOLUME_CM3."""
        sim = TreatmentSimulator(initial_volume_cm3=-10.0)
        np.testing.assert_allclose(sim.initial_volume_cm3, MIN_VOLUME_CM3)


class TestTreatmentSimulatorChemotherapy:
    """Tests for simulate_chemotherapy."""

    def test_chemo_reduces_volume(self):
        """Chemotherapy with high sensitivity and dose reduces volume."""
        sim = TreatmentSimulator(
            initial_volume_cm3=10.0,
            chemo_sensitivity=0.9,
            growth_rate_per_day=0.001,
        )
        proto = ChemotherapyProtocol(
            dose_mg_per_m2=150.0,
            tumor_sensitivity=0.9,
            cycles=4,
        )
        outcome = sim.simulate_chemotherapy(protocol=proto, seed=42)
        assert outcome.final_volume_cm3 < sim.initial_volume_cm3

    def test_chemo_trajectory_length(self):
        """Trajectory has cycles+1 entries (initial + one per cycle)."""
        proto = ChemotherapyProtocol(cycles=4)
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_chemotherapy(protocol=proto, seed=42)
        assert len(outcome.volume_trajectory) == 5  # initial + 4 cycles

    def test_chemo_outcome_has_toxicity(self):
        """Chemotherapy outcome includes a ToxicityProfile."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_chemotherapy(seed=42)
        assert outcome.toxicity_profile is not None

    def test_chemo_simulation_count_incremented(self):
        """simulation_count increases after each simulation."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        sim.simulate_chemotherapy(seed=42)
        assert sim.simulation_count == 1


class TestTreatmentSimulatorRadiation:
    """Tests for simulate_radiation (LQ model)."""

    def test_radiation_reduces_volume(self):
        """Radiation reduces final volume."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0, radio_sensitivity=0.6)
        outcome = sim.simulate_radiation(seed=42)
        assert outcome.final_volume_cm3 < sim.initial_volume_cm3

    def test_radiation_lq_surviving_fraction(self):
        """Metadata contains the surviving fraction."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_radiation(seed=42)
        assert "surviving_fraction" in outcome.metadata
        sf = outcome.metadata["surviving_fraction"]
        assert 0.0 < sf <= 1.0

    def test_radiation_bed_in_metadata(self):
        """Biologically effective dose (BED) is in metadata."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_radiation(seed=42)
        assert "bed" in outcome.metadata
        assert outcome.metadata["bed"] > 0.0


class TestTreatmentSimulatorImmunotherapy:
    """Tests for simulate_immunotherapy."""

    def test_immunotherapy_returns_outcome(self):
        """Immunotherapy produces a valid TreatmentOutcome."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0, immuno_sensitivity=0.5)
        outcome = sim.simulate_immunotherapy(seed=42)
        assert isinstance(outcome, TreatmentOutcome)
        assert outcome.modality == TreatmentModality.IMMUNOTHERAPY

    def test_immunotherapy_metadata_has_response_info(self):
        """Metadata tracks responded, effective_rate, tmb, pdl1."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_immunotherapy(seed=42)
        assert "responded" in outcome.metadata
        assert "effective_rate" in outcome.metadata


class TestTreatmentSimulatorCombination:
    """Tests for simulate_combination."""

    def test_combination_chemo_radiation(self):
        """Chemo + radiation combination reduces volume."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_combination(
            chemo_protocol=ChemotherapyProtocol(cycles=2),
            radiation_plan=RadiationPlan(total_dose_gy=40.0, fractions=20),
            seed=42,
        )
        assert outcome.final_volume_cm3 < sim.initial_volume_cm3
        assert outcome.modality == TreatmentModality.COMBINATION

    def test_combination_no_args_defaults_to_chemo(self):
        """Combination with no protocols defaults to chemotherapy."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_combination(seed=42)
        assert outcome.modality == TreatmentModality.COMBINATION


class TestTreatmentSimulatorHistory:
    """Tests for simulation history tracking."""

    def test_history_accumulates(self):
        """History grows with each simulation."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        sim.simulate_chemotherapy(seed=0)
        sim.simulate_radiation(seed=1)
        history = sim.get_simulation_history()
        assert len(history) == 2
        assert history[0].modality == TreatmentModality.CHEMOTHERAPY
        assert history[1].modality == TreatmentModality.RADIATION

    def test_confidence_interval_bounds(self):
        """CI low <= final volume <= CI high."""
        sim = TreatmentSimulator(initial_volume_cm3=10.0)
        outcome = sim.simulate_chemotherapy(seed=42)
        assert outcome.confidence_interval_low <= outcome.final_volume_cm3
        assert outcome.final_volume_cm3 <= outcome.confidence_interval_high


class TestToxicityGradeEnum:
    """Tests for ToxicityGrade enum."""

    def test_grade_values(self):
        """Grades 0-5 exist."""
        assert ToxicityGrade.GRADE_0.value == 0
        assert ToxicityGrade.GRADE_5.value == 5

    def test_all_grades(self):
        """Six grade levels exist."""
        assert len(ToxicityGrade) == 6
