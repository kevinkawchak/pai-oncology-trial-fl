#!/usr/bin/env python3
"""Risk Stratification for Oncology Clinical Trial Patients.

CLINICAL CONTEXT
================
Risk stratification is a cornerstone of precision oncology. It enables
trialists to identify patients at higher risk of adverse outcomes, tailor
treatment intensity, and support adaptive trial designs. This example
demonstrates a composite risk scoring framework that integrates clinical,
laboratory, and genomic features into a unified risk model with calibration
assessment.

The approach mirrors validated prognostic indices used in oncology
(e.g., the International Prognostic Index for lymphoma, Heng criteria for
renal cell carcinoma) while remaining configurable for any tumour type.

USE CASES COVERED
=================
1. Generating synthetic patient feature sets with bounded clinical
   parameters (labs, performance status, tumour burden, genomic markers).
2. Computing weighted composite risk scores from heterogeneous features
   with configurable feature weights and normalisation.
3. Stratifying a patient cohort into risk categories (low, intermediate,
   high, very high) using percentile-based or absolute thresholds.
4. Generating clinical decision support output with risk-appropriate
   monitoring and treatment intensity recommendations.
5. Running calibration assessment to evaluate how well predicted risk
   categories align with observed event rates.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0   (https://numpy.org)
    scipy >= 1.11.0   (https://scipy.org)

REFERENCES
==========
- International Prognostic Index (IPI) -- NEJM 1993;329:987-94
- Heng Criteria -- JCO 2009;27:5794-9
- ECOG Performance Status Scale
- CTCAE v5.0 Adverse Event Grading

DISCLAIMER
==========
RESEARCH USE ONLY. This software is provided for research and educational
purposes only. It has NOT been validated for clinical use, is NOT approved
by the FDA or any other regulatory body, and MUST NOT be used to make
clinical decisions or direct patient care. All outputs must be reviewed
by qualified clinical professionals before any action is taken.

LICENSE: MIT
VERSION: 0.9.0
LAST UPDATED: 2026-02-18
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path so orchestrator modules are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.clinical_analytics.example_03")


# ============================================================================
# Configuration and Data Structures
# ============================================================================


@dataclass
class RiskFeatureConfig:
    """Configuration for risk feature generation and scoring.

    Attributes:
        n_patients: Number of patients to simulate.
        seed: RNG seed for reproducibility.
        feature_weights: Mapping of feature name to scoring weight.
        risk_thresholds: Percentile boundaries for risk categories.
        bounded_ranges: Physiologically plausible value ranges per feature.
    """

    n_patients: int = 300
    seed: int = 42
    feature_weights: dict[str, float] = field(default_factory=lambda: {
        "age_years": 0.10,
        "ecog_ps": 0.20,
        "ldh_ratio": 0.15,
        "neutrophil_lymphocyte_ratio": 0.12,
        "albumin_g_dl": -0.10,  # protective (negative weight)
        "tumour_burden_mm": 0.18,
        "ctdna_vaf": 0.15,
    })
    risk_thresholds: dict[str, float] = field(default_factory=lambda: {
        "low": 25.0,          # percentile
        "intermediate": 50.0,
        "high": 75.0,
        # above 75th percentile = very_high
    })
    bounded_ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "age_years": (18.0, 95.0),
        "ecog_ps": (0.0, 4.0),
        "ldh_ratio": (0.5, 5.0),
        "neutrophil_lymphocyte_ratio": (0.5, 25.0),
        "albumin_g_dl": (1.5, 5.5),
        "tumour_burden_mm": (5.0, 200.0),
        "ctdna_vaf": (0.0, 0.50),
    })


@dataclass
class PatientRiskProfile:
    """Individual patient risk assessment result.

    Attributes:
        patient_id: Unique identifier.
        features: Raw feature values.
        normalised_features: Features scaled to [0, 1].
        risk_score: Composite weighted risk score.
        risk_category: Assigned risk stratum.
        contributing_factors: Top features driving the score.
    """

    patient_id: str = ""
    features: dict[str, float] = field(default_factory=dict)
    normalised_features: dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0
    risk_category: str = "unknown"
    contributing_factors: list[tuple[str, float]] = field(default_factory=list)


# ============================================================================
# Synthetic Data Generation
# ============================================================================


def generate_patient_features(config: RiskFeatureConfig) -> list[dict[str, Any]]:
    """Generate synthetic patient feature sets with bounded clinical values.

    Args:
        config: Risk feature configuration.

    Returns:
        List of patient feature dictionaries.
    """
    rng = np.random.default_rng(config.seed)
    patients: list[dict[str, Any]] = []

    for i in range(config.n_patients):
        age = float(np.clip(rng.normal(63.0, 11.0), 18.0, 95.0))
        ecog = float(np.clip(rng.choice([0, 1, 2, 3, 4], p=[0.20, 0.35, 0.25, 0.15, 0.05]), 0, 4))
        ldh_ratio = float(np.clip(rng.lognormal(0.3, 0.5), 0.5, 5.0))
        nlr = float(np.clip(rng.lognormal(1.2, 0.7), 0.5, 25.0))
        albumin = float(np.clip(rng.normal(3.6, 0.7), 1.5, 5.5))
        tumour = float(np.clip(rng.lognormal(3.5, 0.8), 5.0, 200.0))
        ctdna = float(np.clip(rng.beta(2, 8), 0.0, 0.50))

        # Simulate whether an event occurred (correlated with risk)
        # Higher risk features -> higher event probability
        risk_proxy = 0.1 * (age / 95) + 0.2 * (ecog / 4) + 0.15 * (ldh_ratio / 5)
        risk_proxy += 0.15 * (nlr / 25) + 0.1 * (1 - albumin / 5.5)
        risk_proxy += 0.15 * (tumour / 200) + 0.15 * (ctdna / 0.5)
        event_prob = float(np.clip(risk_proxy, 0.05, 0.95))
        event = int(rng.binomial(1, event_prob))

        patient = {
            "patient_id": f"PAT-{i + 1:04d}",
            "age_years": round(age, 1),
            "ecog_ps": round(ecog, 0),
            "ldh_ratio": round(ldh_ratio, 2),
            "neutrophil_lymphocyte_ratio": round(nlr, 2),
            "albumin_g_dl": round(albumin, 2),
            "tumour_burden_mm": round(tumour, 1),
            "ctdna_vaf": round(ctdna, 4),
            "event_occurred": event,
        }
        patients.append(patient)

    logger.info("Generated %d patient feature sets", len(patients))
    return patients


# ============================================================================
# Risk Score Computation
# ============================================================================


def normalise_features(
    patients: list[dict[str, Any]],
    config: RiskFeatureConfig,
) -> list[dict[str, float]]:
    """Normalise patient features to [0, 1] using bounded ranges.

    Args:
        patients: List of patient feature dictionaries.
        config: Configuration with bounded ranges.

    Returns:
        List of normalised feature dictionaries.
    """
    normalised_list: list[dict[str, float]] = []

    for patient in patients:
        norm: dict[str, float] = {}
        for feat_name, (lo, hi) in config.bounded_ranges.items():
            raw = patient.get(feat_name, 0.0)
            if hi > lo:
                norm[feat_name] = float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))
            else:
                norm[feat_name] = 0.0
        normalised_list.append(norm)

    return normalised_list


def compute_risk_scores(
    patients: list[dict[str, Any]],
    config: RiskFeatureConfig,
) -> list[PatientRiskProfile]:
    """Compute composite weighted risk scores for all patients.

    The risk score is a weighted sum of normalised features. Features
    with negative weights (e.g., albumin) act as protective factors.
    Scores are rescaled to [0, 100] for interpretability.

    Args:
        patients: Patient feature dictionaries.
        config: Risk feature configuration.

    Returns:
        List of PatientRiskProfile with scores and categories.
    """
    normalised = normalise_features(patients, config)
    raw_scores: list[float] = []

    for norm_feats in normalised:
        score = 0.0
        for feat_name, weight in config.feature_weights.items():
            val = norm_feats.get(feat_name, 0.0)
            # For negative weights (protective), invert the value
            if weight < 0:
                score += abs(weight) * (1.0 - val)
            else:
                score += weight * val
        raw_scores.append(score)

    # Rescale to [0, 100]
    score_array = np.array(raw_scores)
    s_min, s_max = float(np.min(score_array)), float(np.max(score_array))
    if s_max > s_min:
        scaled = 100.0 * (score_array - s_min) / (s_max - s_min)
    else:
        scaled = np.full_like(score_array, 50.0)

    # Determine percentile thresholds
    p_low = float(np.percentile(scaled, config.risk_thresholds["low"]))
    p_int = float(np.percentile(scaled, config.risk_thresholds["intermediate"]))
    p_high = float(np.percentile(scaled, config.risk_thresholds["high"]))

    profiles: list[PatientRiskProfile] = []
    for idx, patient in enumerate(patients):
        s = float(scaled[idx])
        if s <= p_low:
            category = "low"
        elif s <= p_int:
            category = "intermediate"
        elif s <= p_high:
            category = "high"
        else:
            category = "very_high"

        # Identify top contributing factors
        contributions: list[tuple[str, float]] = []
        for feat_name, weight in config.feature_weights.items():
            val = normalised[idx].get(feat_name, 0.0)
            contrib = abs(weight) * val if weight >= 0 else abs(weight) * (1.0 - val)
            contributions.append((feat_name, round(contrib, 4)))
        contributions.sort(key=lambda x: x[1], reverse=True)

        profile = PatientRiskProfile(
            patient_id=patient["patient_id"],
            features={k: patient[k] for k in config.bounded_ranges},
            normalised_features=normalised[idx],
            risk_score=round(s, 2),
            risk_category=category,
            contributing_factors=contributions[:3],
        )
        profiles.append(profile)

    logger.info(
        "Risk scores computed | mean=%.1f | std=%.1f",
        float(np.mean(scaled)),
        float(np.std(scaled)),
    )
    return profiles


# ============================================================================
# Stratification Summary
# ============================================================================


def stratify_cohort(
    profiles: list[PatientRiskProfile],
) -> dict[str, dict[str, Any]]:
    """Stratify the cohort into risk categories and compute group summaries.

    Args:
        profiles: List of patient risk profiles.

    Returns:
        Dictionary mapping risk category to group summary statistics.
    """
    categories = ["low", "intermediate", "high", "very_high"]
    summary: dict[str, dict[str, Any]] = {}

    for cat in categories:
        group = [p for p in profiles if p.risk_category == cat]
        if not group:
            summary[cat] = {"n": 0, "mean_score": 0.0, "score_range": (0.0, 0.0)}
            continue
        scores = np.array([p.risk_score for p in group])
        summary[cat] = {
            "n": len(group),
            "pct": round(100.0 * len(group) / len(profiles), 1),
            "mean_score": round(float(np.mean(scores)), 2),
            "std_score": round(float(np.std(scores, ddof=1)) if len(group) > 1 else 0.0, 2),
            "score_range": (round(float(np.min(scores)), 2), round(float(np.max(scores)), 2)),
        }

    logger.info("Cohort stratified | %s", {c: s["n"] for c, s in summary.items()})
    return summary


# ============================================================================
# Decision Support Output
# ============================================================================


MONITORING_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "low": {
        "visit_frequency": "Every 12 weeks",
        "imaging_interval": "Every 16 weeks",
        "lab_frequency": "Every 6 weeks",
        "treatment_intensity": "Standard protocol dose",
    },
    "intermediate": {
        "visit_frequency": "Every 8 weeks",
        "imaging_interval": "Every 12 weeks",
        "lab_frequency": "Every 4 weeks",
        "treatment_intensity": "Standard protocol dose with close monitoring",
    },
    "high": {
        "visit_frequency": "Every 4 weeks",
        "imaging_interval": "Every 8 weeks",
        "lab_frequency": "Every 2 weeks",
        "treatment_intensity": "Consider dose modification if AE grade >= 3",
    },
    "very_high": {
        "visit_frequency": "Every 2 weeks",
        "imaging_interval": "Every 6 weeks",
        "lab_frequency": "Weekly",
        "treatment_intensity": "Consider reduced intensity or best supportive care",
    },
}


def generate_decision_support(
    profiles: list[PatientRiskProfile],
) -> list[dict[str, Any]]:
    """Generate clinical decision support recommendations per patient.

    Args:
        profiles: Patient risk profiles.

    Returns:
        List of decision support dictionaries.
    """
    decisions: list[dict[str, Any]] = []
    for p in profiles:
        recs = MONITORING_RECOMMENDATIONS.get(p.risk_category, {})
        decisions.append({
            "patient_id": p.patient_id,
            "risk_category": p.risk_category,
            "risk_score": p.risk_score,
            "top_factors": p.contributing_factors,
            "recommendations": recs,
        })
    logger.info("Decision support generated for %d patients", len(decisions))
    return decisions


# ============================================================================
# Calibration Assessment
# ============================================================================


def assess_calibration(
    profiles: list[PatientRiskProfile],
    patients: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assess calibration of risk scores against observed events.

    Computes observed event rates per risk category and evaluates whether
    higher risk strata truly correspond to higher event rates (monotonicity).
    Also computes the Hosmer-Lemeshow-style chi-squared statistic across
    deciles of predicted risk.

    Args:
        profiles: Patient risk profiles with predicted categories.
        patients: Patient data with observed event outcomes.

    Returns:
        Calibration assessment results.
    """
    # Build lookup
    event_map = {p["patient_id"]: p["event_occurred"] for p in patients}

    categories = ["low", "intermediate", "high", "very_high"]
    calibration: dict[str, dict[str, Any]] = {}

    for cat in categories:
        group = [p for p in profiles if p.risk_category == cat]
        if not group:
            calibration[cat] = {"n": 0, "event_rate": 0.0}
            continue
        events = [event_map.get(p.patient_id, 0) for p in group]
        event_rate = float(np.mean(events))
        calibration[cat] = {
            "n": len(group),
            "n_events": sum(events),
            "event_rate": round(event_rate, 4),
        }

    # Monotonicity check
    rates = [calibration[c]["event_rate"] for c in categories if calibration[c]["n"] > 0]
    is_monotonic = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    # Hosmer-Lemeshow across deciles
    scores = np.array([p.risk_score for p in profiles])
    events = np.array([event_map.get(p.patient_id, 0) for p in profiles])
    decile_edges = np.percentile(scores, np.arange(0, 101, 10))
    hl_chi2 = 0.0
    n_deciles = 0

    for i in range(len(decile_edges) - 1):
        lo, hi = decile_edges[i], decile_edges[i + 1]
        if i < len(decile_edges) - 2:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)
        n_g = int(np.sum(mask))
        if n_g == 0:
            continue
        observed = float(np.sum(events[mask]))
        # Expected events proportional to mean score in group
        mean_score = float(np.mean(scores[mask])) / 100.0
        expected = mean_score * n_g
        expected = max(expected, 0.5)  # avoid division by zero
        hl_chi2 += (observed - expected) ** 2 / expected
        n_deciles += 1

    # p-value from chi-squared distribution (df = n_deciles - 2)
    df = max(n_deciles - 2, 1)
    hl_pvalue = float(1.0 - sp_stats.chi2.cdf(hl_chi2, df))

    result = {
        "category_calibration": calibration,
        "is_monotonic": is_monotonic,
        "hosmer_lemeshow_chi2": round(hl_chi2, 3),
        "hosmer_lemeshow_df": df,
        "hosmer_lemeshow_pvalue": round(hl_pvalue, 4),
    }

    logger.info(
        "Calibration assessed | monotonic=%s | HL chi2=%.3f (p=%.4f)",
        is_monotonic, hl_chi2, hl_pvalue,
    )
    return result


# ============================================================================
# Printing Utilities
# ============================================================================


def _print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the risk stratification example."""
    logger.info("Starting Example 03: Risk Stratification")

    config = RiskFeatureConfig(n_patients=300, seed=42)

    print("\n" + "=" * 72)
    print("  RISK STRATIFICATION -- ONCOLOGY TRIAL COHORT")
    print("=" * 72)

    # Step 1: Generate synthetic patient features
    _print_section("1. SYNTHETIC PATIENT FEATURE GENERATION")
    patients = generate_patient_features(config)
    print(f"  Generated {len(patients)} patients with {len(config.feature_weights)} risk features")
    print(f"  Features: {', '.join(config.feature_weights.keys())}")
    print("\n  Sample patient (first):")
    for k, v in patients[0].items():
        if k != "patient_id":
            print(f"    {k:35s}  {v}")

    # Step 2: Compute risk scores
    _print_section("2. COMPOSITE RISK SCORE COMPUTATION")
    profiles = compute_risk_scores(patients, config)
    scores = np.array([p.risk_score for p in profiles])
    print("  Score distribution:")
    print(f"    Mean:   {float(np.mean(scores)):8.2f}")
    print(f"    Median: {float(np.median(scores)):8.2f}")
    print(f"    SD:     {float(np.std(scores, ddof=1)):8.2f}")
    print(f"    Range:  [{float(np.min(scores)):.2f}, {float(np.max(scores)):.2f}]")
    print("\n  Feature weights:")
    for feat, wt in config.feature_weights.items():
        direction = "risk" if wt > 0 else "protective"
        print(f"    {feat:35s}  weight={wt:+.2f}  ({direction})")

    # Step 3: Stratify cohort
    _print_section("3. COHORT STRATIFICATION")
    strat = stratify_cohort(profiles)
    print(f"  {'Category':>15s}  {'N':>6s}  {'%':>7s}  {'Mean Score':>12s}  "
          f"{'SD':>8s}  {'Range':>20s}")
    print(f"  {'-' * 72}")
    for cat in ["low", "intermediate", "high", "very_high"]:
        s = strat[cat]
        if s["n"] > 0:
            rng_str = f"[{s['score_range'][0]:.1f}, {s['score_range'][1]:.1f}]"
            print(f"  {cat:>15s}  {s['n']:6d}  {s['pct']:6.1f}%  "
                  f"{s['mean_score']:12.2f}  {s['std_score']:8.2f}  {rng_str:>20s}")

    # Step 4: Decision support
    _print_section("4. CLINICAL DECISION SUPPORT OUTPUT")
    decisions = generate_decision_support(profiles)
    # Show representative patient from each stratum
    for cat in ["low", "intermediate", "high", "very_high"]:
        rep = next((d for d in decisions if d["risk_category"] == cat), None)
        if rep:
            print(f"\n  [{cat.upper()}] Patient {rep['patient_id']} "
                  f"(score={rep['risk_score']:.1f})")
            for k, v in rep["recommendations"].items():
                print(f"    {k:25s}  {v}")
            print("    Top risk factors:")
            for feat, contrib in rep["top_factors"]:
                print(f"      - {feat}: contribution={contrib:.4f}")

    # Step 5: Calibration assessment
    _print_section("5. CALIBRATION ASSESSMENT")
    cal = assess_calibration(profiles, patients)
    print("\n  Observed event rates by risk category:")
    print(f"  {'Category':>15s}  {'N':>6s}  {'Events':>8s}  {'Event Rate':>12s}")
    print(f"  {'-' * 46}")
    for cat in ["low", "intermediate", "high", "very_high"]:
        c = cal["category_calibration"][cat]
        if c["n"] > 0:
            print(f"  {cat:>15s}  {c['n']:6d}  {c['n_events']:8d}  "
                  f"{c['event_rate']:12.4f}")

    print(f"\n  Monotonicity check:       {'PASS' if cal['is_monotonic'] else 'FAIL'}")
    print(f"  Hosmer-Lemeshow chi2:     {cal['hosmer_lemeshow_chi2']:.3f}")
    print(f"  Hosmer-Lemeshow df:       {cal['hosmer_lemeshow_df']}")
    print(f"  Hosmer-Lemeshow p-value:  {cal['hosmer_lemeshow_pvalue']:.4f}")

    print("\n" + "=" * 72)
    print("  RISK STRATIFICATION COMPLETE")
    print("  RESEARCH USE ONLY -- NOT FOR CLINICAL DECISION MAKING")
    print("=" * 72 + "\n")

    logger.info("Example 03 complete")


if __name__ == "__main__":
    main()
