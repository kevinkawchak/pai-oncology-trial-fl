#!/usr/bin/env python3
"""Full Clinical Analytics Workflow -- End-to-End Oncology Trial Pipeline.

CLINICAL CONTEXT
================
This example demonstrates a complete end-to-end clinical analytics workflow
that combines all major components of the PAI Clinical Analytics platform:
orchestration, data management, interoperability/harmonization, PK/PD
analysis, risk stratification, survival endpoints, and consortium (DSMB)
reporting. It mirrors the operational flow of a real multi-site oncology
clinical trial from data ingestion through to safety monitoring board output.

The workflow follows the typical lifecycle of a Phase II/III oncology trial
analysis: sites submit heterogeneous data, the analytics platform harmonizes
and validates it, runs pre-specified analyses, and produces structured reports
for the Data Safety Monitoring Board (DSMB) and regulatory submission.

USE CASES COVERED
=================
1. Initializing the analytics orchestrator with full trial configuration.
2. Registering and validating multi-site datasets via trial data manager.
3. Harmonizing data across sites via clinical interoperability layer
   (terminology mapping, unit conversion, schema alignment).
4. Running PK/PD analysis on harmonized concentration-time data.
5. Stratifying the patient cohort by composite risk score.
6. Computing survival endpoints (KM, log-rank, Cox PH, RMST).
7. Generating a structured DSMB safety report via consortium reporting.
8. Printing comprehensive results with all analysis sections.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0   (https://numpy.org)
    scipy >= 1.11.0   (https://scipy.org)

REFERENCES
==========
- ICH E6(R3) Good Clinical Practice
- ICH E9(R1) Estimands and Sensitivity Analysis
- CDISC SDTM / ADaM Implementation Guides
- FDA Guidance for Industry: Safety Reporting (2015)
- DAMOCLES Statement (2005) -- DSMB Charter Guidelines

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

import hashlib
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path so orchestrator modules are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy import stats as sp_stats
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.clinical_analytics.example_06")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrialConfig:
    """Full trial analytics configuration.

    Attributes:
        study_id: Unique study identifier.
        study_phase: Trial phase (e.g., 'Phase II').
        therapeutic_area: Therapeutic area label.
        n_sites: Number of participating sites.
        patients_per_site: Approximate patients per site.
        treatment_arms: Treatment arm labels.
        primary_endpoint: Primary efficacy endpoint.
        alpha: Significance level.
        seed: RNG seed for reproducibility.
    """

    study_id: str = "NCT-PAI-2026-001"
    study_phase: str = "Phase II"
    therapeutic_area: str = "Oncology -- Non-Small Cell Lung Cancer"
    n_sites: int = 4
    patients_per_site: int = 50
    treatment_arms: list[str] = field(
        default_factory=lambda: ["Placebo", "Experimental"]
    )
    primary_endpoint: str = "Progression-Free Survival (PFS)"
    alpha: float = 0.05
    seed: int = 42


# ============================================================================
# Trial Data Manager (Component 1: Data Registration & Validation)
# ============================================================================


class TrialDataManager:
    """Manages trial dataset registration, validation, and inventory.

    Simulates the trial_data_manager module for this end-to-end example.

    Args:
        config: Trial configuration.
    """

    def __init__(self, config: TrialConfig) -> None:
        self._config = config
        self._datasets: dict[str, list[dict[str, Any]]] = {}
        self._validation_log: list[dict[str, Any]] = []
        logger.info("TrialDataManager initialized | study=%s", config.study_id)

    def register_site_data(
        self, site_id: str, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Register and validate a site dataset.

        Args:
            site_id: Site identifier.
            records: Patient records from the site.

        Returns:
            Validation summary.
        """
        n_total = len(records)
        issues: list[str] = []

        # Validate required fields
        required = ["patient_id", "age", "sex", "treatment_arm"]
        for i, rec in enumerate(records):
            for fld in required:
                if fld not in rec or rec[fld] is None:
                    issues.append(f"Record {i}: missing '{fld}'")

            # Bounded checks
            if "age" in rec and (rec["age"] < 18 or rec["age"] > 95):
                issues.append(f"Record {i}: age={rec['age']} out of [18, 95]")

        self._datasets[site_id] = records
        summary = {
            "site_id": site_id,
            "n_records": n_total,
            "n_issues": len(issues),
            "issues": issues[:5],  # first 5 only
            "status": "VALID" if len(issues) == 0 else "VALID_WITH_WARNINGS",
        }
        self._validation_log.append(summary)
        logger.info(
            "Registered site %s | n=%d | issues=%d",
            site_id, n_total, len(issues),
        )
        return summary

    def get_all_records(self) -> list[dict[str, Any]]:
        """Return all registered records across sites."""
        all_records: list[dict[str, Any]] = []
        for records in self._datasets.values():
            all_records.extend(records)
        return all_records


# ============================================================================
# Clinical Interoperability (Component 2: Harmonization)
# ============================================================================


ICD10_TO_SNOMED: dict[str, str] = {
    "C34.1": "254637007", "C34.9": "93880001", "C50.9": "254838004",
    "C18.9": "363406005", "C61": "399068003", "C43.9": "372244006",
    "C64.9": "93849006", "C25.9": "363418001",
}


class ClinicalInteroperability:
    """Harmonizes data across sites with terminology mapping and unit conversion.

    Args:
        config: Trial configuration.
    """

    def __init__(self, config: TrialConfig) -> None:
        self._config = config
        self._harmonization_log: list[dict[str, Any]] = []
        logger.info("ClinicalInteroperability initialized")

    def harmonize(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Harmonize records: map terminology, convert units, validate.

        Args:
            records: Raw patient records from multiple sites.

        Returns:
            Harmonized records.
        """
        harmonized: list[dict[str, Any]] = []
        n_mapped = 0
        n_converted = 0

        for rec in records:
            h = dict(rec)

            # Map ICD-10 to SNOMED
            if "diagnosis_icd10" in h:
                snomed = ICD10_TO_SNOMED.get(h["diagnosis_icd10"], "UNMAPPED")
                h["diagnosis_snomed"] = snomed
                if snomed != "UNMAPPED":
                    n_mapped += 1

            # Convert weight if in pounds
            if h.get("weight_unit") == "lb" and "weight_kg" in h:
                h["weight_kg"] = round(h["weight_kg"] * 0.453592, 1)
                h["weight_unit"] = "kg"
                n_converted += 1

            harmonized.append(h)

        self._harmonization_log.append({
            "n_records": len(records),
            "n_terminology_mapped": n_mapped,
            "n_units_converted": n_converted,
        })
        logger.info(
            "Harmonized %d records | mapped=%d | converted=%d",
            len(records), n_mapped, n_converted,
        )
        return harmonized

    def get_alignment_summary(self) -> dict[str, Any]:
        """Return harmonization summary."""
        if not self._harmonization_log:
            return {"status": "no harmonization performed"}
        latest = self._harmonization_log[-1]
        return {
            "total_records": latest["n_records"],
            "terminology_mapped": latest["n_terminology_mapped"],
            "units_converted": latest["n_units_converted"],
            "mapping_rate": round(
                latest["n_terminology_mapped"] / max(latest["n_records"], 1), 4
            ),
        }


# ============================================================================
# PK/PD Analysis (Component 3)
# ============================================================================


def run_pkpd_analysis(
    dose_mg: float = 500.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run PK/PD analysis: one-compartment model with NCA.

    Args:
        dose_mg: Administered dose in milligrams.
        seed: RNG seed.

    Returns:
        PK/PD analysis results.
    """
    # One-compartment parameters
    ka = 1.2   # absorption rate (1/h)
    ke = 0.15  # elimination rate (1/h)
    vd = 50.0  # volume of distribution (L)
    f = 0.85   # bioavailability

    t = np.linspace(0.01, 48.0, 500)
    coeff = (f * dose_mg * ka) / (vd * (ka - ke))
    conc = np.clip(coeff * (np.exp(-ke * t) - np.exp(-ka * t)), 0, None)

    # NCA
    auc = float(np.trapezoid(conc, t))
    idx_max = int(np.argmax(conc))
    cmax = float(conc[idx_max])
    tmax = float(t[idx_max])

    # Terminal half-life
    n_term = max(int(0.3 * len(t)), 3)
    c_term = conc[-n_term:]
    t_term = t[-n_term:]
    pos = c_term > 0
    if np.sum(pos) >= 2:
        slope, _ = np.polyfit(t_term[pos], np.log(c_term[pos]), 1)
        t_half = float(np.log(2) / max(-slope, 1e-15))
    else:
        t_half = float("nan")

    # Emax dose-response
    emax, ec50, hill = 85.0, 12.0, 1.8
    effect_at_cmax = 5.0 + emax * cmax**hill / (ec50**hill + cmax**hill)

    result = {
        "dose_mg": dose_mg,
        "auc_mg_h_l": round(auc, 2),
        "cmax_mg_l": round(cmax, 3),
        "tmax_h": round(tmax, 2),
        "t_half_h": round(t_half, 2),
        "predicted_effect_at_cmax": round(float(effect_at_cmax), 2),
    }
    logger.info("PK/PD analysis | AUC=%.2f, Cmax=%.3f, Tmax=%.2f", auc, cmax, tmax)
    return result


# ============================================================================
# Risk Stratification (Component 4)
# ============================================================================


def stratify_patients(
    records: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """Stratify patients into risk categories based on clinical features.

    Args:
        records: Harmonized patient records.
        seed: RNG seed.

    Returns:
        Stratification results with category counts and summaries.
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []

    for rec in records:
        age_norm = np.clip((rec.get("age", 60) - 18) / 77.0, 0, 1)
        ecog_norm = np.clip(rec.get("ecog_ps", 1) / 4.0, 0, 1)
        ldh_norm = np.clip((rec.get("ldh_ratio", 1.0) - 0.5) / 4.5, 0, 1)
        alb_norm = 1.0 - np.clip((rec.get("albumin_g_dl", 3.5) - 1.5) / 4.0, 0, 1)

        score = 0.15 * age_norm + 0.25 * ecog_norm + 0.30 * ldh_norm + 0.30 * alb_norm
        scores.append(float(score))

    score_arr = np.array(scores)
    # Rescale to 0-100
    s_min, s_max = float(np.min(score_arr)), float(np.max(score_arr))
    if s_max > s_min:
        scaled = 100.0 * (score_arr - s_min) / (s_max - s_min)
    else:
        scaled = np.full_like(score_arr, 50.0)

    p25 = float(np.percentile(scaled, 25))
    p50 = float(np.percentile(scaled, 50))
    p75 = float(np.percentile(scaled, 75))

    categories: dict[str, int] = {"low": 0, "intermediate": 0, "high": 0, "very_high": 0}
    for s in scaled:
        if s <= p25:
            categories["low"] += 1
        elif s <= p50:
            categories["intermediate"] += 1
        elif s <= p75:
            categories["high"] += 1
        else:
            categories["very_high"] += 1

    result = {
        "n_patients": len(records),
        "score_mean": round(float(np.mean(scaled)), 2),
        "score_std": round(float(np.std(scaled, ddof=1)), 2),
        "categories": categories,
        "thresholds": {"p25": round(p25, 2), "p50": round(p50, 2), "p75": round(p75, 2)},
    }
    logger.info("Risk stratification | %s", categories)
    return result


# ============================================================================
# Survival Endpoints (Component 5)
# ============================================================================


def compute_survival_endpoints(
    records: list[dict[str, Any]],
    config: TrialConfig,
) -> dict[str, Any]:
    """Compute survival endpoints: KM medians, log-rank, Cox HR, RMST.

    Args:
        records: Harmonized patient records with time/event data.
        config: Trial configuration.

    Returns:
        Survival analysis results.
    """
    rng = np.random.default_rng(config.seed)
    arms = config.treatment_arms
    lambda_base = np.log(2) / 12.0  # 12-month median
    hr_true = 0.70

    times: list[float] = []
    events: list[int] = []
    arm_indices: list[int] = []

    for rec in records:
        arm_idx = arms.index(rec["treatment_arm"]) if rec["treatment_arm"] in arms else 0
        lam = lambda_base if arm_idx == 0 else lambda_base * hr_true
        t_event = float(rng.exponential(1.0 / lam))
        t_censor = float(rng.uniform(0, 36.0))
        apply_censor = rng.random() < 0.15

        if apply_censor and t_censor < t_event:
            obs_t = min(t_censor, 36.0)
            evt = 0
        elif t_event <= 36.0:
            obs_t = t_event
            evt = 1
        else:
            obs_t = 36.0
            evt = 0

        times.append(float(np.clip(obs_t, 0.01, 36.0)))
        events.append(evt)
        arm_indices.append(arm_idx)

    t_arr = np.array(times)
    e_arr = np.array(events)
    a_arr = np.array(arm_indices)

    # KM medians per arm
    medians: dict[str, float] = {}
    for arm_idx, arm_label in enumerate(arms):
        mask = a_arr == arm_idx
        t_arm = t_arr[mask]
        e_arm = e_arr[mask]
        order = np.argsort(t_arm)
        t_s = t_arm[order]
        e_s = e_arm[order]

        s = 1.0
        med = float("nan")
        event_times = np.unique(t_s[e_s == 1])
        for ti in event_times:
            di = int(np.sum((t_s == ti) & (e_s == 1)))
            ni = int(np.sum(t_s >= ti))
            if ni > 0:
                s *= (1 - di / ni)
            if s <= 0.5:
                med = float(ti)
                break
        medians[arm_label] = round(med, 2)

    # Log-rank test
    event_times_all = np.unique(t_arr[e_arr == 1])
    obs1, exp1, var_lr = 0.0, 0.0, 0.0
    for ti in event_times_all:
        r0 = int(np.sum((t_arr >= ti) & (a_arr == 0)))
        r1 = int(np.sum((t_arr >= ti) & (a_arr == 1)))
        n_r = r0 + r1
        d0 = int(np.sum((t_arr == ti) & (e_arr == 1) & (a_arr == 0)))
        d1 = int(np.sum((t_arr == ti) & (e_arr == 1) & (a_arr == 1)))
        di = d0 + d1
        if n_r < 2 or di == 0:
            continue
        obs1 += d1
        exp1 += di * r1 / n_r
        if n_r > 1:
            var_lr += di * r0 * r1 * (n_r - di) / (n_r**2 * (n_r - 1))

    lr_chi2 = (obs1 - exp1) ** 2 / max(var_lr, 1e-15)
    lr_pval = float(1 - sp_stats.chi2.cdf(lr_chi2, 1))

    # Cox HR (simple treatment effect)
    x = a_arr.reshape(-1, 1).astype(np.float64)
    order = np.argsort(-t_arr)
    t_s = t_arr[order]
    e_s = e_arr[order]
    x_s = x[order]
    n = len(t_arr)

    def neg_pll(beta: np.ndarray) -> float:
        eta = (x_s @ beta).flatten()
        exp_eta = np.exp(np.clip(eta, -500, 500))
        cum = np.cumsum(exp_eta)
        ll = sum(eta[i] - np.log(max(cum[i], 1e-15)) for i in range(n) if e_s[i] == 1)
        return -ll

    from scipy.optimize import minimize as sp_min
    res = sp_min(neg_pll, np.array([0.0]), method="BFGS")
    beta_hat = float(res.x[0])
    hr_est = float(np.exp(beta_hat))
    se_beta = float(np.sqrt(res.hess_inv[0, 0])) if res.hess_inv is not None else float("nan")
    hr_lower = float(np.exp(beta_hat - 1.96 * se_beta))
    hr_upper = float(np.exp(beta_hat + 1.96 * se_beta))

    # RMST at 24 months (simplified)
    tau = 24.0
    rmst_vals: dict[str, float] = {}
    for arm_idx, arm_label in enumerate(arms):
        mask = a_arr == arm_idx
        t_arm = t_arr[mask]
        e_arm = e_arr[mask]
        order_a = np.argsort(t_arm)
        t_sa = t_arm[order_a]
        e_sa = e_arm[order_a]
        s_curve = [1.0]
        t_curve = [0.0]
        s = 1.0
        for ti in np.unique(t_sa[e_sa == 1]):
            if ti > tau:
                break
            di = int(np.sum((t_sa == ti) & (e_sa == 1)))
            ni = int(np.sum(t_sa >= ti))
            if ni > 0:
                s *= (1 - di / ni)
            t_curve.append(float(ti))
            s_curve.append(s)
        t_curve.append(tau)
        s_curve.append(s)
        rmst_vals[arm_label] = round(float(np.trapezoid(s_curve, t_curve)), 2)

    result = {
        "n_total": len(t_arr),
        "n_events": int(np.sum(e_arr)),
        "median_survival": medians,
        "log_rank_chi2": round(lr_chi2, 4),
        "log_rank_pvalue": round(lr_pval, 6),
        "cox_hr": round(hr_est, 4),
        "cox_hr_ci": (round(hr_lower, 4), round(hr_upper, 4)),
        "cox_beta": round(beta_hat, 4),
        "rmst_24mo": rmst_vals,
    }
    logger.info(
        "Survival endpoints | HR=%.4f [%.4f, %.4f] | p=%.6f",
        hr_est, hr_lower, hr_upper, lr_pval,
    )
    return result


# ============================================================================
# Consortium Reporting / DSMB Report (Component 6)
# ============================================================================


@dataclass
class DSMBReport:
    """Structured Data Safety Monitoring Board report.

    Attributes:
        report_id: Unique report identifier.
        study_id: Study identifier.
        report_date: ISO-formatted date string.
        enrollment_summary: Enrollment metrics.
        safety_summary: Adverse event summary.
        efficacy_summary: Interim efficacy results.
        pkpd_summary: PK/PD analysis results.
        risk_summary: Risk stratification results.
        survival_summary: Survival endpoint results.
        recommendations: DSMB recommendation text.
        integrity_hash: SHA-256 hash for audit trail.
    """

    report_id: str = ""
    study_id: str = ""
    report_date: str = ""
    enrollment_summary: dict[str, Any] = field(default_factory=dict)
    safety_summary: dict[str, Any] = field(default_factory=dict)
    efficacy_summary: dict[str, Any] = field(default_factory=dict)
    pkpd_summary: dict[str, Any] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    survival_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: str = ""
    integrity_hash: str = ""


def generate_dsmb_report(
    config: TrialConfig,
    records: list[dict[str, Any]],
    pkpd: dict[str, Any],
    risk: dict[str, Any],
    survival: dict[str, Any],
) -> DSMBReport:
    """Generate a structured DSMB safety and efficacy report.

    Args:
        config: Trial configuration.
        records: All harmonized patient records.
        pkpd: PK/PD analysis results.
        risk: Risk stratification results.
        survival: Survival endpoint results.

    Returns:
        DSMBReport with all sections populated.
    """
    report_id = str(uuid.uuid4())

    # Enrollment
    arm_counts: dict[str, int] = {}
    for arm in config.treatment_arms:
        arm_counts[arm] = sum(1 for r in records if r.get("treatment_arm") == arm)

    site_counts: dict[str, int] = {}
    for r in records:
        sid = r.get("site_id", "UNKNOWN")
        site_counts[sid] = site_counts.get(sid, 0) + 1

    enrollment = {
        "total_enrolled": len(records),
        "by_arm": arm_counts,
        "by_site": site_counts,
        "n_sites_active": len(site_counts),
    }

    # Safety summary
    ae_grades: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    n_sae = 0
    for r in records:
        grade = r.get("max_ae_grade", 0)
        ae_grades[grade] = ae_grades.get(grade, 0) + 1
        if r.get("serious_ae", False):
            n_sae += 1

    safety = {
        "ae_grade_distribution": ae_grades,
        "n_serious_ae": n_sae,
        "grade_3_4_rate": round(
            (ae_grades.get(3, 0) + ae_grades.get(4, 0)) / max(len(records), 1), 4
        ),
    }

    # Efficacy
    efficacy = {
        "primary_endpoint": config.primary_endpoint,
        "hr_estimate": survival.get("cox_hr"),
        "hr_ci": survival.get("cox_hr_ci"),
        "log_rank_p": survival.get("log_rank_pvalue"),
        "median_survival": survival.get("median_survival"),
    }

    # Recommendation logic (simplified)
    p_val = survival.get("log_rank_pvalue", 1.0)
    grade_34_rate = safety["grade_3_4_rate"]
    if grade_34_rate > 0.30:
        recommendation = ("RECOMMEND SAFETY REVIEW: Grade 3/4 AE rate exceeds 30%. "
                          "Consider protocol amendment for dose modification.")
    elif p_val is not None and p_val < 0.001:
        recommendation = ("RECOMMEND EARLY REPORTING: Strong efficacy signal detected. "
                          "Consider interim analysis for early stopping.")
    else:
        recommendation = ("RECOMMEND CONTINUATION: Safety profile acceptable and "
                          "efficacy data maturing. Continue per protocol.")

    # Integrity hash
    hash_payload = f"{report_id}:{config.study_id}:{len(records)}:{survival.get('cox_hr')}"
    integrity = hashlib.sha256(hash_payload.encode()).hexdigest()

    report = DSMBReport(
        report_id=report_id,
        study_id=config.study_id,
        report_date="2026-02-18",
        enrollment_summary=enrollment,
        safety_summary=safety,
        efficacy_summary=efficacy,
        pkpd_summary=pkpd,
        risk_summary=risk,
        survival_summary=survival,
        recommendations=recommendation,
        integrity_hash=integrity,
    )

    logger.info("DSMB report generated | id=%s | hash=%s", report_id, integrity[:16])
    return report


# ============================================================================
# Synthetic Data Generation
# ============================================================================


def generate_multi_site_data(config: TrialConfig) -> dict[str, list[dict[str, Any]]]:
    """Generate synthetic multi-site trial data.

    Args:
        config: Trial configuration.

    Returns:
        Dictionary mapping site_id to list of patient records.
    """
    rng = np.random.default_rng(config.seed)
    diagnoses = list(ICD10_TO_SNOMED.keys())
    site_data: dict[str, list[dict[str, Any]]] = {}

    for site_idx in range(1, config.n_sites + 1):
        site_id = f"SITE-{site_idx:02d}"
        records: list[dict[str, Any]] = []

        for i in range(config.patients_per_site):
            rec = {
                "patient_id": f"PAT-{site_id}-{i + 1:04d}",
                "site_id": site_id,
                "age": int(np.clip(rng.normal(62, 11), 18, 95)),
                "sex": str(rng.choice(["M", "F"])),
                "weight_kg": round(float(np.clip(rng.normal(75, 15), 40, 160)), 1),
                "weight_unit": "kg",
                "ecog_ps": int(rng.choice([0, 1, 2, 3, 4], p=[0.2, 0.35, 0.25, 0.15, 0.05])),
                "diagnosis_icd10": str(rng.choice(diagnoses)),
                "treatment_arm": str(rng.choice(config.treatment_arms)),
                "hemoglobin_g_dl": round(float(np.clip(rng.normal(12.5, 2), 7, 18)), 1),
                "albumin_g_dl": round(float(np.clip(rng.normal(3.6, 0.7), 1.5, 5.5)), 2),
                "ldh_ratio": round(float(np.clip(rng.lognormal(0.3, 0.5), 0.5, 5.0)), 2),
                "creatinine_mg_dl": round(float(np.clip(rng.normal(1.0, 0.4), 0.3, 5.0)), 2),
                "max_ae_grade": int(rng.choice([0, 1, 2, 3, 4], p=[0.25, 0.30, 0.25, 0.15, 0.05])),
                "serious_ae": bool(rng.random() < 0.12),
            }
            records.append(rec)

        site_data[site_id] = records

    logger.info(
        "Generated multi-site data | sites=%d | total=%d",
        config.n_sites, config.n_sites * config.patients_per_site,
    )
    return site_data


# ============================================================================
# Printing Utilities
# ============================================================================


def _print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def _print_dict(d: dict[str, Any], indent: int = 4) -> None:
    """Print dictionary with formatted key-value pairs."""
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            for k2, v2 in v.items():
                print(f"{prefix}  {k2:>20s}: {v2}")
        elif isinstance(v, float):
            print(f"{prefix}{k:>30s}: {v:.4f}")
        else:
            print(f"{prefix}{k:>30s}: {v}")


# ============================================================================
# Main Workflow
# ============================================================================


def main() -> None:
    """Run the full clinical analytics workflow."""
    logger.info("Starting Example 06: Full Clinical Analytics Workflow")
    workflow_start = time.time()

    config = TrialConfig(
        study_id="NCT-PAI-2026-001",
        study_phase="Phase II",
        n_sites=4,
        patients_per_site=50,
        treatment_arms=["Placebo", "Experimental"],
        seed=42,
    )

    print("\n" + "=" * 72)
    print("  FULL CLINICAL ANALYTICS WORKFLOW -- END-TO-END PIPELINE")
    print("=" * 72)
    print(f"  Study:     {config.study_id}")
    print(f"  Phase:     {config.study_phase}")
    print(f"  Endpoint:  {config.primary_endpoint}")
    print(f"  Arms:      {config.treatment_arms}")

    # ── Step 1: Initialize orchestrator ──
    _print_section("STEP 1: INITIALIZE ORCHESTRATOR")
    data_manager = TrialDataManager(config)
    interop = ClinicalInteroperability(config)
    print(f"  Orchestrator ready | study={config.study_id}")

    # ── Step 2: Register datasets ──
    _print_section("STEP 2: REGISTER MULTI-SITE DATASETS")
    site_data = generate_multi_site_data(config)
    for site_id, records in site_data.items():
        summary = data_manager.register_site_data(site_id, records)
        print(f"  {site_id}: {summary['n_records']} records | "
              f"status={summary['status']} | issues={summary['n_issues']}")

    all_records = data_manager.get_all_records()
    print(f"\n  Total registered: {len(all_records)} records across "
          f"{config.n_sites} sites")

    # ── Step 3: Harmonize data ──
    _print_section("STEP 3: HARMONIZE DATA (CLINICAL INTEROPERABILITY)")
    harmonized = interop.harmonize(all_records)
    align_summary = interop.get_alignment_summary()
    print(f"  Records harmonized:     {align_summary['total_records']}")
    print(f"  Terminology mapped:     {align_summary['terminology_mapped']}")
    print(f"  Units converted:        {align_summary['units_converted']}")
    print(f"  Mapping rate:           {align_summary['mapping_rate']:.2%}")

    # ── Step 4: PK/PD analysis ──
    _print_section("STEP 4: PK/PD ANALYSIS")
    pkpd = run_pkpd_analysis(dose_mg=500.0, seed=config.seed)
    print(f"  Dose:             {pkpd['dose_mg']:.0f} mg")
    print(f"  AUC:              {pkpd['auc_mg_h_l']:.2f} mg*h/L")
    print(f"  Cmax:             {pkpd['cmax_mg_l']:.3f} mg/L")
    print(f"  Tmax:             {pkpd['tmax_h']:.2f} h")
    print(f"  t1/2:             {pkpd['t_half_h']:.2f} h")
    print(f"  Effect at Cmax:   {pkpd['predicted_effect_at_cmax']:.2f}")

    # ── Step 5: Risk stratification ──
    _print_section("STEP 5: PATIENT RISK STRATIFICATION")
    risk = stratify_patients(harmonized, seed=config.seed)
    print(f"  Patients scored:  {risk['n_patients']}")
    print(f"  Score mean (SD):  {risk['score_mean']:.2f} ({risk['score_std']:.2f})")
    print("  Category breakdown:")
    for cat, count in risk["categories"].items():
        pct = 100.0 * count / max(risk["n_patients"], 1)
        print(f"    {cat:>15s}: {count:4d} ({pct:5.1f}%)")

    # ── Step 6: Survival endpoints ──
    _print_section("STEP 6: SURVIVAL ENDPOINTS")
    survival = compute_survival_endpoints(harmonized, config)
    print(f"  N total:      {survival['n_total']}")
    print(f"  N events:     {survival['n_events']}")
    print("  Median survival:")
    for arm, med in survival["median_survival"].items():
        print(f"    {arm:>15s}: {med:.2f} months")
    print(f"  Log-rank:     chi2={survival['log_rank_chi2']:.4f}, "
          f"p={survival['log_rank_pvalue']:.6f}")
    print(f"  Cox HR:       {survival['cox_hr']:.4f} "
          f"({survival['cox_hr_ci'][0]:.4f}, {survival['cox_hr_ci'][1]:.4f})")
    print("  RMST (24 mo):")
    for arm, rmst in survival["rmst_24mo"].items():
        print(f"    {arm:>15s}: {rmst:.2f} months")

    # ── Step 7: DSMB report ──
    _print_section("STEP 7: GENERATE DSMB REPORT")
    report = generate_dsmb_report(config, harmonized, pkpd, risk, survival)
    print(f"  Report ID:    {report.report_id}")
    print(f"  Date:         {report.report_date}")
    print(f"  Integrity:    {report.integrity_hash[:32]}...")

    print("\n  [ENROLLMENT]")
    print(f"    Total: {report.enrollment_summary['total_enrolled']}")
    for arm, n in report.enrollment_summary["by_arm"].items():
        print(f"    {arm:>15s}: {n}")

    print("\n  [SAFETY]")
    print(f"    Grade 3/4 rate: {report.safety_summary['grade_3_4_rate']:.2%}")
    print(f"    Serious AEs:    {report.safety_summary['n_serious_ae']}")
    print(f"    AE distribution: {report.safety_summary['ae_grade_distribution']}")

    print("\n  [EFFICACY]")
    print(f"    Endpoint: {report.efficacy_summary['primary_endpoint']}")
    print(f"    HR:       {report.efficacy_summary['hr_estimate']}")
    print(f"    95% CI:   {report.efficacy_summary['hr_ci']}")
    print(f"    P-value:  {report.efficacy_summary['log_rank_p']}")

    print("\n  [RECOMMENDATION]")
    print(f"    {report.recommendations}")

    # ── Workflow complete ──
    elapsed = time.time() - workflow_start
    print("\n" + "=" * 72)
    print("  FULL CLINICAL ANALYTICS WORKFLOW COMPLETE")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print("  RESEARCH USE ONLY -- NOT FOR CLINICAL DECISION MAKING")
    print("=" * 72 + "\n")

    logger.info("Example 06 complete | elapsed=%.2f s", elapsed)


if __name__ == "__main__":
    main()
