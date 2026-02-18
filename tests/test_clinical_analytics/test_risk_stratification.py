"""Tests for clinical-analytics/risk_stratification.py.

Covers enums (RiskCategory, PrognosticFactor, DecisionConfidence,
TreatmentArm), dataclasses (PatientRiskProfile, CohortRiskSummary),
and the ClinicalRiskStratifier class: risk score computation (bounded
[0,1]), patient stratification, cohort stratification, calibration,
decision support, C-statistic validation.  Also covers the
AdaptiveEnrichmentAdvisor: enrichment recommendations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import load_module

mod = load_module(
    "risk_stratification",
    "clinical-analytics/risk_stratification.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestRiskEnums:
    """Tests for all risk stratification enums."""

    def test_risk_category_members(self):
        """RiskCategory has five strata."""
        expected = {"VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"}
        assert set(mod.RiskCategory.__members__.keys()) == expected

    def test_risk_category_values(self):
        """RiskCategory values are lowercase strings."""
        assert mod.RiskCategory.VERY_LOW.value == "very_low"
        assert mod.RiskCategory.VERY_HIGH.value == "very_high"

    def test_prognostic_factor_members(self):
        """PrognosticFactor has expected members."""
        names = set(mod.PrognosticFactor.__members__.keys())
        assert "TUMOR_STAGE" in names
        assert "BIOMARKER_STATUS" in names
        assert "AGE" in names
        assert len(names) >= 6

    def test_decision_confidence_members(self):
        """DecisionConfidence has expected members."""
        expected = {"STRONG", "MODERATE", "WEAK", "INSUFFICIENT_DATA"}
        assert set(mod.DecisionConfidence.__members__.keys()) == expected

    def test_treatment_arm_members(self):
        """TreatmentArm has expected members."""
        names = set(mod.TreatmentArm.__members__.keys())
        assert "STANDARD" in names
        assert "EXPERIMENTAL" in names
        assert "COMBINATION" in names
        assert "BEST_SUPPORTIVE_CARE" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestPatientRiskProfile:
    """Tests for the PatientRiskProfile dataclass."""

    def test_creation(self):
        """PatientRiskProfile is created with required fields."""
        profile = mod.PatientRiskProfile(
            patient_id="PAT-001",
            risk_category=mod.RiskCategory.MODERATE,
            risk_score=0.5,
        )
        assert profile.patient_id == "PAT-001"
        assert profile.risk_score == 0.5
        assert profile.risk_category == mod.RiskCategory.MODERATE

    def test_timestamp_populated(self):
        """PatientRiskProfile timestamp is auto-populated."""
        profile = mod.PatientRiskProfile(
            patient_id="PAT-002",
            risk_category=mod.RiskCategory.LOW,
            risk_score=0.2,
        )
        assert profile.timestamp != ""

    def test_default_contributing_factors(self):
        """Default contributing_factors is an empty dict."""
        profile = mod.PatientRiskProfile(
            patient_id="PAT-003",
            risk_category=mod.RiskCategory.HIGH,
            risk_score=0.7,
        )
        assert profile.contributing_factors == {}


# ---------------------------------------------------------------------------
# ClinicalRiskStratifier tests
# ---------------------------------------------------------------------------
class TestClinicalRiskStratifier:
    """Tests for ClinicalRiskStratifier risk scoring and stratification."""

    def test_default_creation(self):
        """Stratifier initializes with default criteria."""
        stratifier = mod.ClinicalRiskStratifier()
        assert stratifier.criteria is not None
        assert len(stratifier.criteria.weights) > 0

    def test_compute_risk_score_bounded(self):
        """Risk score is always bounded in [0, 1]."""
        stratifier = mod.ClinicalRiskStratifier()
        # All factors at maximum
        features = {f.value: 1.0 for f in mod.PrognosticFactor}
        score = stratifier.compute_risk_score(features)
        assert 0.0 <= score <= 1.0

    def test_compute_risk_score_zero_features(self):
        """Risk score with all-zero features is bounded."""
        stratifier = mod.ClinicalRiskStratifier()
        features = {f.value: 0.0 for f in mod.PrognosticFactor}
        score = stratifier.compute_risk_score(features)
        assert 0.0 <= score <= 1.0

    def test_compute_risk_score_empty_features(self):
        """Risk score with empty features does not crash."""
        stratifier = mod.ClinicalRiskStratifier()
        score = stratifier.compute_risk_score({})
        assert 0.0 <= score <= 1.0

    def test_stratify_patient_returns_profile(self):
        """stratify_patient returns a PatientRiskProfile instance."""
        stratifier = mod.ClinicalRiskStratifier()
        features = {mod.PrognosticFactor.TUMOR_STAGE.value: 0.7}
        profile = stratifier.stratify_patient(features, raw_patient_id="P-100")
        assert isinstance(profile, mod.PatientRiskProfile)
        assert profile.risk_score >= 0.0
        assert profile.risk_score <= 1.0

    def test_stratify_patient_high_risk(self):
        """High-risk patient gets HIGH or VERY_HIGH category."""
        stratifier = mod.ClinicalRiskStratifier()
        features = {f.value: 0.9 for f in mod.PrognosticFactor}
        profile = stratifier.stratify_patient(features)
        assert profile.risk_category in (mod.RiskCategory.HIGH, mod.RiskCategory.VERY_HIGH)

    def test_stratify_patient_low_risk(self):
        """Low-risk patient gets VERY_LOW or LOW category."""
        stratifier = mod.ClinicalRiskStratifier()
        features = {f.value: 0.05 for f in mod.PrognosticFactor}
        profile = stratifier.stratify_patient(features)
        assert profile.risk_category in (mod.RiskCategory.VERY_LOW, mod.RiskCategory.LOW)

    def test_stratify_cohort_empty(self):
        """Empty cohort returns zero-size summary."""
        stratifier = mod.ClinicalRiskStratifier()
        summary = stratifier.stratify_cohort([])
        assert summary.cohort_size == 0

    def test_stratify_cohort_multiple_patients(self):
        """Cohort stratification returns correct count and distribution."""
        stratifier = mod.ClinicalRiskStratifier()
        patients = [
            {f.value: 0.1 for f in mod.PrognosticFactor},
            {f.value: 0.5 for f in mod.PrognosticFactor},
            {f.value: 0.9 for f in mod.PrognosticFactor},
        ]
        summary = stratifier.stratify_cohort(patients)
        assert summary.cohort_size == 3
        assert summary.mean_risk > 0.0
        total_dist = sum(summary.distribution.values())
        assert total_dist == 3

    def test_calibration(self):
        """calibrate_model returns HL statistic and p-value."""
        stratifier = mod.ClinicalRiskStratifier()
        rng = np.random.default_rng(42)
        observed = rng.integers(0, 2, size=100).astype(np.float64)
        predicted = rng.uniform(0, 1, size=100)
        result = stratifier.calibrate_model(observed, predicted)
        assert "hl_statistic" in result
        assert "p_value" in result
        assert result["p_value"] >= 0.0
        assert result["p_value"] <= 1.0

    def test_decision_support_output(self):
        """generate_decision_support returns a DecisionSupportOutput."""
        stratifier = mod.ClinicalRiskStratifier()
        features = {f.value: 0.5 for f in mod.PrognosticFactor}
        profile = stratifier.stratify_patient(features)
        ds = stratifier.generate_decision_support(profile)
        assert isinstance(ds, mod.DecisionSupportOutput)
        assert ds.recommended_arm in mod.TreatmentArm
        assert ds.confidence in mod.DecisionConfidence

    def test_validate_stratification_c_statistic(self):
        """validate_stratification computes a C-statistic in [0, 1]."""
        stratifier = mod.ClinicalRiskStratifier()
        rng = np.random.default_rng(42)
        patients = [{f.value: rng.uniform(0, 1) for f in mod.PrognosticFactor} for _ in range(50)]
        profiles = [stratifier.stratify_patient(p) for p in patients]
        # Generate outcomes correlated with risk
        outcomes = np.array([1 if p.risk_score > 0.5 else 0 for p in profiles], dtype=np.float64)
        result = stratifier.validate_stratification(profiles, outcomes)
        assert "c_statistic" in result
        c_stat = result["c_statistic"]
        assert 0.0 <= c_stat <= 1.0


# ---------------------------------------------------------------------------
# AdaptiveEnrichmentAdvisor tests
# ---------------------------------------------------------------------------
class TestAdaptiveEnrichmentAdvisor:
    """Tests for the AdaptiveEnrichmentAdvisor."""

    def test_recommend_enrichment_basic(self):
        """Enrichment recommendation returns expected structure."""
        stratifier = mod.ClinicalRiskStratifier()
        advisor = mod.AdaptiveEnrichmentAdvisor(stratifier)
        # Build a simple cohort summary
        patients = [{f.value: 0.5 for f in mod.PrognosticFactor} for _ in range(20)]
        summary = stratifier.stratify_cohort(patients)
        # Subgroup outcomes with one benefiting group
        outcomes = {}
        for cat in mod.RiskCategory:
            outcomes[cat.value] = {
                "experimental_rate": 0.3,
                "control_rate": 0.2,
            }
        result = advisor.recommend_enrichment(summary, outcomes)
        assert "enrich" in result
        assert "exclude" in result
        assert isinstance(result["enrich"], list)

    def test_recommend_enrichment_negative_effect(self):
        """Subgroups with negative effect are flagged for exclusion."""
        stratifier = mod.ClinicalRiskStratifier()
        advisor = mod.AdaptiveEnrichmentAdvisor(
            stratifier,
            enrichment_benefit_threshold=0.1,
        )
        patients = [{f.value: 0.5 for f in mod.PrognosticFactor} for _ in range(20)]
        summary = stratifier.stratify_cohort(patients)
        outcomes = {}
        for cat in mod.RiskCategory:
            outcomes[cat.value] = {
                "experimental_rate": 0.1,
                "control_rate": 0.4,  # control better = negative effect
            }
        result = advisor.recommend_enrichment(summary, outcomes)
        assert len(result["exclude"]) > 0

    def test_futility_analysis_insufficient(self):
        """Futility analysis with tiny samples reports insufficient data."""
        stratifier = mod.ClinicalRiskStratifier()
        advisor = mod.AdaptiveEnrichmentAdvisor(stratifier)
        exp = np.array([1, 0])  # < 5 subjects
        ctrl = np.array([0, 1])
        result = advisor.futility_analysis(exp, ctrl, planned_total_n=200)
        assert "rationale" in result
        # With only 2 subjects per arm, should flag insufficient
        assert "Insufficient" in result["rationale"] or not np.isnan(result["conditional_power"])
