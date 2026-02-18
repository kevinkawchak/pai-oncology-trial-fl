"""Tests for tools/dose-calculator/dose_calculator.py.

Covers enums, dataclasses, BED/EQD2 computation, radiation dose calculation,
chemotherapy dose calculation, OAR validation, and CLI helpers.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest

from tests.conftest import load_module

mod = load_module("dose_calculator", "tools/dose-calculator/dose_calculator.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestDoseUnit:
    """Tests for the DoseUnit enum."""

    def test_dose_unit_members(self):
        """All expected DoseUnit members exist."""
        expected = {"GY", "CGY", "MG_M2", "MG", "MG_KG", "AUC"}
        assert set(mod.DoseUnit.__members__.keys()) == expected

    def test_dose_unit_values(self):
        """DoseUnit values are correct strings."""
        assert mod.DoseUnit.GY.value == "Gy"
        assert mod.DoseUnit.MG_M2.value == "mg/m2"
        assert mod.DoseUnit.AUC.value == "AUC"

    def test_dose_unit_is_str_enum(self):
        """DoseUnit members are strings."""
        assert isinstance(mod.DoseUnit.GY, str)


class TestChemoAgent:
    """Tests for the ChemoAgent enum."""

    def test_chemo_agent_count(self):
        """ChemoAgent has 8 members."""
        assert len(mod.ChemoAgent) == 8

    def test_cisplatin_value(self):
        """ChemoAgent.CISPLATIN has expected value."""
        assert mod.ChemoAgent.CISPLATIN.value == "cisplatin"


class TestRadiationTechnique:
    """Tests for the RadiationTechnique enum."""

    def test_technique_members(self):
        """RadiationTechnique has expected members."""
        names = set(mod.RadiationTechnique.__members__.keys())
        assert "IMRT" in names
        assert "SBRT" in names
        assert "SRS" in names
        assert "PROTON" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestPatientParameters:
    """Tests for the PatientParameters dataclass."""

    def test_defaults(self):
        """Default values are physiologically reasonable."""
        p = mod.PatientParameters()
        assert p.weight_kg == 70.0
        assert p.height_cm == 170.0
        assert p.bsa_m2 == 1.7

    def test_clamping_weight_low(self):
        """Weight below 20 kg is clamped to 20."""
        p = mod.PatientParameters(weight_kg=5.0)
        assert p.weight_kg == 20.0

    def test_clamping_weight_high(self):
        """Weight above 300 kg is clamped to 300."""
        p = mod.PatientParameters(weight_kg=500.0)
        assert p.weight_kg == 300.0

    def test_clamping_bsa_bounds(self):
        """BSA is clamped to [0.5, 3.0]."""
        p_low = mod.PatientParameters(bsa_m2=0.1)
        assert p_low.bsa_m2 == 0.5
        p_high = mod.PatientParameters(bsa_m2=5.0)
        assert p_high.bsa_m2 == 3.0

    def test_clamping_crcl(self):
        """Creatinine clearance is clamped to [10, 200]."""
        p = mod.PatientParameters(creatinine_clearance_ml_min=0.0)
        assert p.creatinine_clearance_ml_min == 10.0


class TestDoseCalculation:
    """Tests for the DoseCalculation dataclass."""

    def test_defaults(self):
        """Default DoseCalculation has zero values."""
        dc = mod.DoseCalculation()
        assert dc.total_dose == 0.0
        assert dc.is_within_bounds is False
        assert dc.warnings == []

    def test_asdict(self):
        """DoseCalculation can be serialized to dict."""
        dc = mod.DoseCalculation(calculation_type="radiation", total_dose=50.0)
        d = asdict(dc)
        assert d["calculation_type"] == "radiation"
        assert d["total_dose"] == 50.0


# ---------------------------------------------------------------------------
# BED / EQD2 computation
# ---------------------------------------------------------------------------
class TestComputeBED:
    """Tests for the compute_bed helper."""

    def test_known_bed(self):
        """BED for 60 Gy in 30 fractions with alpha/beta=10."""
        bed = mod.compute_bed(60.0, 2.0, 10.0)
        assert pytest.approx(bed, rel=1e-4) == 72.0

    def test_zero_dose(self):
        """BED is 0 when total dose is 0."""
        assert mod.compute_bed(0.0, 2.0, 10.0) == 0.0

    def test_zero_fraction_dose(self):
        """BED is 0 when dose per fraction is 0."""
        assert mod.compute_bed(60.0, 0.0, 10.0) == 0.0

    def test_zero_alpha_beta(self):
        """BED is 0 when alpha/beta is 0."""
        assert mod.compute_bed(60.0, 2.0, 0.0) == 0.0

    def test_negative_dose_clamped(self):
        """Negative total dose is clamped to 0."""
        assert mod.compute_bed(-10.0, 2.0, 10.0) == 0.0

    def test_exceeds_max_dose_clamped(self):
        """Total dose exceeding 100 Gy is clamped."""
        bed = mod.compute_bed(200.0, 2.0, 10.0)
        expected = 100.0 * (1.0 + 2.0 / 10.0)
        assert pytest.approx(bed, rel=1e-4) == expected


class TestComputeEQD2:
    """Tests for the compute_eqd2 helper."""

    def test_known_eqd2(self):
        """EQD2 for 60 Gy in 30 fractions with alpha/beta=10 equals 60 Gy."""
        eqd2 = mod.compute_eqd2(60.0, 2.0, 10.0)
        assert pytest.approx(eqd2, rel=1e-4) == 60.0

    def test_hypofractionated_eqd2(self):
        """EQD2 increases for hypofractionated schemes."""
        conventional = mod.compute_eqd2(60.0, 2.0, 10.0)
        hypo = mod.compute_eqd2(60.0, 4.0, 10.0)
        assert hypo > conventional

    def test_zero_alpha_beta_returns_zero(self):
        """EQD2 returns 0 when alpha/beta is 0."""
        assert mod.compute_eqd2(60.0, 2.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Radiation dose calculation
# ---------------------------------------------------------------------------
class TestCalculateRadiationDose:
    """Tests for calculate_radiation_dose."""

    def test_standard_lung(self):
        """Standard lung treatment: 2 Gy x 33 = 66 Gy, within bounds."""
        result = mod.calculate_radiation_dose("lung", "IMRT", 2.0, 33)
        assert result.total_dose == 66.0
        assert result.is_within_bounds is True
        assert result.calculation_type == "radiation"

    def test_exceeds_site_max(self):
        """Dose exceeding site max triggers warning."""
        result = mod.calculate_radiation_dose("brain", "IMRT", 2.5, 30)
        assert result.total_dose == 75.0
        assert result.is_within_bounds is False
        assert len(result.warnings) > 0
        assert "exceeds max" in result.warnings[0].lower()

    def test_fraction_clamping_low(self):
        """Dose per fraction below 0.1 is clamped to 0.1."""
        result = mod.calculate_radiation_dose("lung", "IMRT", -5.0, 10)
        assert result.dose_per_fraction == 0.1

    def test_fraction_count_clamped(self):
        """Fraction count is clamped to [1, 50]."""
        result = mod.calculate_radiation_dose("lung", "IMRT", 2.0, 0)
        assert result.fractions == 1

    def test_total_dose_capped_at_100(self):
        """Total dose cannot exceed 100 Gy absolute max."""
        result = mod.calculate_radiation_dose("lung", "SBRT", 34.0, 50)
        assert result.total_dose <= 100.0

    def test_unknown_site_no_bound_warning(self):
        """Unknown tumour site has no max dose constraint."""
        result = mod.calculate_radiation_dose("unknown_site", "IMRT", 2.0, 25)
        assert result.is_within_bounds is True

    def test_bed_and_eqd2_populated(self):
        """BED and EQD2 are non-zero for valid prescriptions."""
        result = mod.calculate_radiation_dose("prostate", "IMRT", 2.0, 39)
        assert result.bed > 0
        assert result.eqd2 > 0


# ---------------------------------------------------------------------------
# Chemotherapy dose calculation
# ---------------------------------------------------------------------------
class TestCalculateChemotherapyDose:
    """Tests for calculate_chemotherapy_dose."""

    def test_cisplatin_default(self):
        """Cisplatin with default dose computes correctly."""
        patient = mod.PatientParameters(bsa_m2=1.7)
        result = mod.calculate_chemotherapy_dose("cisplatin", patient)
        assert result.calculation_type == "chemotherapy"
        assert result.total_dose > 0

    def test_carboplatin_auc(self):
        """Carboplatin AUC-based dosing uses Calvert formula."""
        patient = mod.PatientParameters(creatinine_clearance_ml_min=100.0)
        result = mod.calculate_chemotherapy_dose("carboplatin", patient, target_auc=6.0)
        expected = 6.0 * (100.0 + 25.0)
        assert result.dose_unit == mod.DoseUnit.MG.value
        assert result.total_dose <= expected + 1.0  # within rounding

    def test_unknown_agent(self):
        """Unknown agent returns a warning."""
        patient = mod.PatientParameters()
        result = mod.calculate_chemotherapy_dose("unknownium", patient)
        assert len(result.warnings) > 0
        assert "unknown" in result.warnings[0].lower()

    def test_dose_out_of_range_warning(self):
        """Dose outside NCCN range triggers a warning."""
        patient = mod.PatientParameters(bsa_m2=1.7)
        result = mod.calculate_chemotherapy_dose("cisplatin", patient, dose_mg_m2=500.0)
        assert result.is_within_bounds is False
        assert len(result.warnings) > 0

    def test_carboplatin_auc_clamped(self):
        """AUC is clamped to [1, 10]."""
        patient = mod.PatientParameters(creatinine_clearance_ml_min=100.0)
        result = mod.calculate_chemotherapy_dose("carboplatin", patient, target_auc=50.0)
        # AUC clamped to 10, so dose = 10 * (100 + 25) = 1250, then capped at max_dose=800
        assert result.total_dose <= 800.0


# ---------------------------------------------------------------------------
# OAR validation
# ---------------------------------------------------------------------------
class TestValidateOARDose:
    """Tests for validate_oar_dose."""

    def test_spinal_cord_within(self):
        """Spinal cord dose within constraint passes."""
        result = mod.validate_oar_dose("spinal_cord", 45.0)
        assert result.is_within_bounds is True

    def test_spinal_cord_exceeds(self):
        """Spinal cord dose exceeding constraint fails."""
        result = mod.validate_oar_dose("spinal_cord", 55.0)
        assert result.is_within_bounds is False
        assert len(result.warnings) > 0

    def test_unknown_organ(self):
        """Unknown organ returns a warning about missing constraints."""
        result = mod.validate_oar_dose("unknown_organ", 30.0)
        assert len(result.warnings) > 0
        assert "no oar constraints" in result.warnings[0].lower()

    def test_dose_clamped_negative(self):
        """Negative dose is clamped to 0."""
        result = mod.validate_oar_dose("spinal_cord", -10.0)
        assert result.total_dose == 0.0
        assert result.is_within_bounds is True

    def test_lung_mean_constraint(self):
        """Lung mean dose uses mean_dose constraint (not max_dose)."""
        result = mod.validate_oar_dose("lung_mean", 25.0)
        assert result.is_within_bounds is False

    def test_max_dose_zero_fallback(self):
        """OAR with max_dose=0 should not falsely pass (truthiness fix)."""
        # heart_mean has mean_dose=26, no max_dose key
        result = mod.validate_oar_dose("heart_mean", 30.0)
        assert result.is_within_bounds is False
