"""Tests for clinical-analytics/pkpd_engine.py.

Covers enums (PKModel, PDModel, EliminationRoute, DosingRegimen),
dataclasses (PKParameters, ConcentrationProfile), and the
PopulationPKPDEngine class: one-compartment profile (exponential decay),
two-compartment profile, AUC calculation, Cmax/Tmax extraction, Emax PD
model, covariate effects, division-by-zero guards, and therapeutic window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import load_module

mod = load_module("pkpd_engine", "clinical-analytics/pkpd_engine.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestPKModels:
    """Tests for the PKModel enum."""

    def test_pk_model_members(self):
        """PKModel has the expected members."""
        names = set(mod.PKModel.__members__.keys())
        assert "ONE_COMPARTMENT" in names
        assert "TWO_COMPARTMENT" in names

    def test_pk_model_values(self):
        """PKModel values are correct strings."""
        assert mod.PKModel.ONE_COMPARTMENT.value == "one_compartment"
        assert mod.PKModel.TWO_COMPARTMENT.value == "two_compartment"

    def test_pk_model_is_str_enum(self):
        """PKModel members are instances of str."""
        assert isinstance(mod.PKModel.ONE_COMPARTMENT, str)


class TestPDModels:
    """Tests for the PDModel enum."""

    def test_pd_model_members(self):
        """PDModel has at least Emax and sigmoid Emax."""
        names = set(mod.PDModel.__members__.keys())
        assert "EMAX" in names
        assert "SIGMOID_EMAX" in names
        assert "LINEAR" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestPKParameters:
    """Tests for the PKParameters dataclass."""

    def test_default_creation(self):
        """Default PKParameters are created with reasonable values."""
        pk = mod.PKParameters()
        assert pk.clearance_l_h > 0
        assert pk.volume_central_l > 0
        assert 0.0 <= pk.bioavailability <= 1.0

    def test_bioavailability_clamped(self):
        """Bioavailability is clamped to [0, 1]."""
        pk = mod.PKParameters(bioavailability=1.5)
        assert pk.bioavailability <= 1.0
        pk2 = mod.PKParameters(bioavailability=-0.5)
        assert pk2.bioavailability >= 0.0

    def test_ke_derived_from_cl_v(self):
        """ke_h is derived from clearance/volume if not explicitly set."""
        pk = mod.PKParameters(clearance_l_h=10.0, volume_central_l=50.0, ke_h=0.0)
        expected_ke = 10.0 / 50.0
        np.testing.assert_allclose(pk.ke_h, expected_ke, rtol=1e-4)


# ---------------------------------------------------------------------------
# PopulationPKPDEngine tests
# ---------------------------------------------------------------------------
class TestPopulationPKPDEngine:
    """Tests for the PopulationPKPDEngine simulation engine."""

    def test_one_compartment_iv_decay(self):
        """One-compartment IV bolus shows exponential decay."""
        engine = mod.PopulationPKPDEngine()
        pk = mod.PKParameters(clearance_l_h=10.0, volume_central_l=50.0)
        profile = engine.compute_one_compartment_profile(100.0, pk=pk, time_end_h=24.0)
        conc = profile.concentrations_mg_l
        # Concentration should start high and decay monotonically
        assert conc[0] > conc[-1]
        # Check approximate C0 = dose/V = 100/50 = 2.0 mg/L
        np.testing.assert_allclose(conc[0], 2.0, rtol=0.1)
        # Monotonically non-increasing after Cmax
        idx_max = np.argmax(conc)
        tail = conc[idx_max:]
        diffs = np.diff(tail)
        assert np.all(diffs <= 1e-10)

    def test_one_compartment_auc_positive(self):
        """One-compartment profile has a positive AUC."""
        engine = mod.PopulationPKPDEngine()
        profile = engine.compute_one_compartment_profile(100.0, time_end_h=48.0)
        assert profile.auc_mg_h_l > 0

    def test_one_compartment_cmax_positive(self):
        """One-compartment IV Cmax is positive."""
        engine = mod.PopulationPKPDEngine()
        profile = engine.compute_one_compartment_profile(100.0, time_end_h=24.0)
        assert profile.cmax_mg_l > 0

    def test_one_compartment_zero_dose(self):
        """Zero dose produces zero concentrations."""
        engine = mod.PopulationPKPDEngine()
        profile = engine.compute_one_compartment_profile(0.0, time_end_h=24.0)
        np.testing.assert_allclose(profile.concentrations_mg_l, 0.0, atol=1e-12)

    def test_two_compartment_profile(self):
        """Two-compartment profile has positive concentrations."""
        engine = mod.PopulationPKPDEngine()
        profile = engine.compute_two_compartment_profile(100.0, time_end_h=48.0)
        assert profile.cmax_mg_l > 0
        assert profile.auc_mg_h_l > 0

    def test_two_compartment_decay(self):
        """Two-compartment profile decays toward zero at later times."""
        engine = mod.PopulationPKPDEngine()
        profile = engine.compute_two_compartment_profile(100.0, time_end_h=72.0)
        conc = profile.concentrations_mg_l
        # Last concentration should be much lower than peak
        assert conc[-1] < profile.cmax_mg_l

    def test_emax_response(self):
        """Emax PD model produces effect between baseline and emax."""
        engine = mod.PopulationPKPDEngine()
        pd_params = mod.PDParameters(emax=100.0, ec50=5.0, baseline_effect=0.0)
        concentrations = np.array([0.0, 5.0, 50.0, 500.0])
        effects = engine.compute_pd_response(concentrations, pd=pd_params)
        # At C=0, effect ~ baseline (0)
        assert effects[0] < 1.0
        # At C=EC50, effect ~ Emax/2 = 50
        np.testing.assert_allclose(effects[1], 50.0, rtol=0.15)
        # At very high C, effect approaches Emax
        assert effects[-1] > 90.0

    def test_emax_response_monotonic(self):
        """Emax PD response is monotonically non-decreasing with concentration."""
        engine = mod.PopulationPKPDEngine()
        pd_params = mod.PDParameters(emax=100.0, ec50=5.0)
        conc = np.linspace(0.0, 100.0, 50)
        effects = engine.compute_pd_response(conc, pd=pd_params)
        diffs = np.diff(effects)
        assert np.all(diffs >= -1e-10)

    def test_covariate_weight_scaling(self):
        """Covariate adjustment scales clearance with body weight."""
        engine = mod.PopulationPKPDEngine()
        pk_base = mod.PKParameters(clearance_l_h=10.0, volume_central_l=50.0)
        # Heavier patient should have higher clearance
        pk_heavy = engine.apply_covariate_effects(pk_base, body_weight_kg=100.0)
        pk_light = engine.apply_covariate_effects(pk_base, body_weight_kg=50.0)
        assert pk_heavy.clearance_l_h > pk_light.clearance_l_h

    def test_therapeutic_window_check(self):
        """Therapeutic window check returns expected keys."""
        engine = mod.PopulationPKPDEngine()
        profile = engine.compute_one_compartment_profile(100.0, time_end_h=48.0)
        result = engine.check_therapeutic_window(
            profile, target_trough_mg_l=0.1, target_peak_mg_l=50.0,
        )
        assert "within_window" in result
        assert "fraction_time_in_window" in result

    def test_dose_response_curve(self):
        """Dose-response curve produces effects that increase with dose."""
        engine = mod.PopulationPKPDEngine()
        dr = engine.compute_dose_response_curve(n_doses=10)
        assert len(dr.doses) == 10
        assert len(dr.effects) == 10
        # Effects should generally increase with dose
        assert dr.effects[-1] > dr.effects[0]

    def test_run_count_increments(self):
        """run_count increments with each simulation."""
        engine = mod.PopulationPKPDEngine()
        initial = engine.run_count
        engine.compute_one_compartment_profile(50.0, time_end_h=12.0)
        assert engine.run_count > initial

    def test_audit_trail_populated(self):
        """Audit trail is populated after simulation runs."""
        engine = mod.PopulationPKPDEngine()
        engine.compute_one_compartment_profile(50.0, time_end_h=12.0)
        trail = engine.get_audit_trail()
        assert len(trail) >= 1

    def test_federated_aggregation_empty(self):
        """Federated aggregation with no sites returns error."""
        engine = mod.PopulationPKPDEngine()
        result = engine.federated_aggregate_site_statistics([])
        assert result.get("pooled_n", 0) == 0

    def test_federated_aggregation_two_sites(self):
        """Federated aggregation of two sites produces pooled statistics."""
        engine = mod.PopulationPKPDEngine()
        sites = [
            {"site_id": "S1", "n_subjects": 50,
             "mean_clearance": 10.0, "var_clearance": 1.0,
             "mean_volume": 50.0, "var_volume": 4.0,
             "mean_auc": 100.0, "var_auc": 25.0},
            {"site_id": "S2", "n_subjects": 50,
             "mean_clearance": 12.0, "var_clearance": 2.0,
             "mean_volume": 55.0, "var_volume": 5.0,
             "mean_auc": 90.0, "var_auc": 20.0},
        ]
        result = engine.federated_aggregate_site_statistics(sites)
        assert result["pooled_n"] == 100
        assert result["n_sites"] == 2
        # Mean clearance should be between site values
        assert 10.0 <= result["mean_clearance_l_h"] <= 12.0
