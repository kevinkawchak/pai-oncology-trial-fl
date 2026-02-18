#!/usr/bin/env python3
"""Basic Clinical Analytics Pipeline -- Minimal Working Example.

CLINICAL CONTEXT
================
This example demonstrates the foundational workflow for running descriptive
clinical analytics on oncology trial data using the PAI Clinical Analytics
Orchestrator. It covers configuration, initialization, data ingestion,
descriptive statistics computation, and structured result reporting.

The pipeline computes standard clinical trial summary statistics including
enrollment metrics, demographic distributions, lab value summaries, and
adverse event frequencies -- the building blocks for any clinical study
report (CSR).

USE CASES COVERED
=================
1. Configuring the analytics orchestrator with basic trial parameters
   including study ID, therapeutic area, and analysis thresholds.
2. Generating synthetic patient enrollment and demographics data that
   mirrors real-world oncology trial structure.
3. Computing descriptive statistics (mean, median, SD, IQR) for
   continuous clinical variables (labs, vitals).
4. Tabulating categorical distributions (treatment arms, response
   categories, adverse event grades).
5. Producing structured analytic output suitable for downstream
   reporting and regulatory submission.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

REFERENCES
==========
- ICH E9(R1) Addendum on Estimands and Sensitivity Analysis
- ICH E6(R3) Good Clinical Practice
- CDISC Analysis Data Model (ADaM) Implementation Guide v1.3

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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path so orchestrator modules are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.clinical_analytics.example_01")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class AnalyticsConfig:
    """Configuration for the clinical analytics pipeline.

    Attributes:
        study_id: Unique study identifier (e.g., NCT number).
        study_title: Human-readable study title.
        therapeutic_area: ICD-11 therapeutic area code.
        n_patients: Number of patients to simulate.
        n_sites: Number of trial sites.
        treatment_arms: List of treatment arm labels.
        alpha: Statistical significance threshold.
        seed: Random number generator seed for reproducibility.
        bounded_lab_ranges: Physiologically plausible lab value ranges.
    """

    study_id: str = "NCT-PAI-2026-001"
    study_title: str = "PAI Oncology Phase II Basket Trial"
    therapeutic_area: str = "Oncology"
    n_patients: int = 200
    n_sites: int = 5
    treatment_arms: list[str] = field(default_factory=lambda: ["Control", "Arm_A", "Arm_B"])
    alpha: float = 0.05
    seed: int = 42
    bounded_lab_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "hemoglobin_g_dl": (7.0, 18.0),
            "wbc_k_ul": (1.0, 30.0),
            "platelets_k_ul": (50.0, 450.0),
            "creatinine_mg_dl": (0.3, 5.0),
            "alt_u_l": (5.0, 300.0),
            "albumin_g_dl": (1.5, 5.5),
        }
    )


# ============================================================================
# Clinical Analytics Orchestrator (self-contained example implementation)
# ============================================================================


class ClinicalAnalyticsOrchestrator:
    """Orchestrates clinical analytics workflows.

    This minimal orchestrator demonstrates the pattern used by the full
    ``analytics_orchestrator`` module. It manages data ingestion, validation,
    descriptive analytics, and structured result production.

    Args:
        config: Analytics pipeline configuration.
    """

    def __init__(self, config: AnalyticsConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._patients: list[dict[str, Any]] = []
        self._results: dict[str, Any] = {}
        self._run_id = str(uuid.uuid4())
        logger.info(
            "Orchestrator initialized | study=%s | run_id=%s",
            config.study_id,
            self._run_id,
        )

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_synthetic_cohort(self) -> list[dict[str, Any]]:
        """Generate a synthetic patient cohort for demonstration.

        Returns:
            List of patient dictionaries with demographics, labs, and
            treatment assignments.
        """
        arms = self._config.treatment_arms
        patients: list[dict[str, Any]] = []

        for i in range(self._config.n_patients):
            age = int(np.clip(self._rng.normal(62, 10), 18, 95))
            weight_kg = float(np.clip(self._rng.normal(75, 15), 40.0, 160.0))
            sex = self._rng.choice(["M", "F"])
            arm = str(self._rng.choice(arms))
            site_id = int(self._rng.integers(1, self._config.n_sites + 1))

            # Generate bounded lab values
            labs: dict[str, float] = {}
            for lab_name, (lo, hi) in self._config.bounded_lab_ranges.items():
                mu = (lo + hi) / 2.0
                sigma = (hi - lo) / 6.0  # 99.7% within bounds
                value = float(np.clip(self._rng.normal(mu, sigma), lo, hi))
                labs[lab_name] = round(value, 2)

            # RECIST response category (simulated)
            response = str(
                self._rng.choice(
                    ["CR", "PR", "SD", "PD"],
                    p=[0.10, 0.25, 0.40, 0.25],
                )
            )

            # Adverse event grade (CTCAE v5)
            ae_grade = int(self._rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.30, 0.20, 0.15, 0.05]))

            patient = {
                "patient_id": f"PAT-{i + 1:04d}",
                "site_id": f"SITE-{site_id:02d}",
                "age": age,
                "sex": sex,
                "weight_kg": weight_kg,
                "treatment_arm": arm,
                "labs": labs,
                "best_response": response,
                "max_ae_grade": ae_grade,
            }
            patients.append(patient)

        self._patients = patients
        logger.info("Generated synthetic cohort | n=%d", len(patients))
        return patients

    # ------------------------------------------------------------------
    # Descriptive analytics
    # ------------------------------------------------------------------

    def compute_descriptive_statistics(self) -> dict[str, Any]:
        """Compute descriptive statistics for the patient cohort.

        Computes demographic summaries, lab value distributions, treatment
        arm breakdowns, response rates, and AE frequency tables.

        Returns:
            Dictionary of structured analytic results.
        """
        if not self._patients:
            raise ValueError("No patients loaded. Call generate_synthetic_cohort first.")

        ages = np.array([p["age"] for p in self._patients], dtype=np.float64)
        weights = np.array([p["weight_kg"] for p in self._patients], dtype=np.float64)

        # Demographics
        demographics = {
            "age": {
                "mean": round(float(np.mean(ages)), 1),
                "median": round(float(np.median(ages)), 1),
                "std": round(float(np.std(ages, ddof=1)), 1),
                "min": int(np.min(ages)),
                "max": int(np.max(ages)),
                "iqr": round(float(np.percentile(ages, 75) - np.percentile(ages, 25)), 1),
            },
            "weight_kg": {
                "mean": round(float(np.mean(weights)), 1),
                "median": round(float(np.median(weights)), 1),
                "std": round(float(np.std(weights, ddof=1)), 1),
                "min": round(float(np.min(weights)), 1),
                "max": round(float(np.max(weights)), 1),
            },
            "sex_distribution": {
                "M": sum(1 for p in self._patients if p["sex"] == "M"),
                "F": sum(1 for p in self._patients if p["sex"] == "F"),
            },
        }

        # Lab value summaries
        lab_summaries: dict[str, dict[str, float]] = {}
        for lab_name in self._config.bounded_lab_ranges:
            values = np.array([p["labs"][lab_name] for p in self._patients], dtype=np.float64)
            lab_summaries[lab_name] = {
                "mean": round(float(np.mean(values)), 2),
                "median": round(float(np.median(values)), 2),
                "std": round(float(np.std(values, ddof=1)), 2),
                "p25": round(float(np.percentile(values, 25)), 2),
                "p75": round(float(np.percentile(values, 75)), 2),
            }

        # Treatment arm enrollment
        arm_counts: dict[str, int] = {}
        for arm in self._config.treatment_arms:
            arm_counts[arm] = sum(1 for p in self._patients if p["treatment_arm"] == arm)

        # Response distribution by arm
        response_by_arm: dict[str, dict[str, int]] = {}
        for arm in self._config.treatment_arms:
            arm_patients = [p for p in self._patients if p["treatment_arm"] == arm]
            counts: dict[str, int] = {}
            for resp in ["CR", "PR", "SD", "PD"]:
                counts[resp] = sum(1 for p in arm_patients if p["best_response"] == resp)
            response_by_arm[arm] = counts

        # Adverse event grade distribution
        ae_distribution: dict[str, int] = {}
        for grade in range(5):
            ae_distribution[f"Grade_{grade}"] = sum(1 for p in self._patients if p["max_ae_grade"] == grade)

        # Site enrollment
        site_enrollment: dict[str, int] = {}
        for i in range(1, self._config.n_sites + 1):
            site_key = f"SITE-{i:02d}"
            site_enrollment[site_key] = sum(1 for p in self._patients if p["site_id"] == site_key)

        self._results = {
            "run_id": self._run_id,
            "study_id": self._config.study_id,
            "n_patients": len(self._patients),
            "demographics": demographics,
            "lab_summaries": lab_summaries,
            "arm_enrollment": arm_counts,
            "response_by_arm": response_by_arm,
            "ae_distribution": ae_distribution,
            "site_enrollment": site_enrollment,
        }

        logger.info("Descriptive statistics computed | variables=%d", len(lab_summaries))
        return self._results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a structured summary report to stdout."""
        if not self._results:
            raise ValueError("No results. Call compute_descriptive_statistics first.")

        r = self._results
        print("\n" + "=" * 72)
        print("  CLINICAL ANALYTICS PIPELINE -- DESCRIPTIVE SUMMARY")
        print("=" * 72)
        print(f"  Study:      {r['study_id']}")
        print(f"  Run ID:     {r['run_id']}")
        print(f"  N Patients: {r['n_patients']}")
        print("-" * 72)

        # Demographics
        print("\n  [DEMOGRAPHICS]")
        demo = r["demographics"]
        age = demo["age"]
        print(
            f"    Age (years):  mean={age['mean']}, median={age['median']}, "
            f"SD={age['std']}, range=[{age['min']}, {age['max']}]"
        )
        wt = demo["weight_kg"]
        print(f"    Weight (kg):  mean={wt['mean']}, median={wt['median']}, SD={wt['std']}")
        sex = demo["sex_distribution"]
        print(f"    Sex:          M={sex['M']}, F={sex['F']}")

        # Lab summaries
        print("\n  [LAB VALUES]")
        for lab_name, stats in r["lab_summaries"].items():
            print(
                f"    {lab_name:25s}  mean={stats['mean']:8.2f}  "
                f"median={stats['median']:8.2f}  SD={stats['std']:8.2f}  "
                f"IQR=[{stats['p25']:.2f}, {stats['p75']:.2f}]"
            )

        # Arm enrollment
        print("\n  [TREATMENT ARM ENROLLMENT]")
        for arm, count in r["arm_enrollment"].items():
            pct = 100.0 * count / r["n_patients"]
            print(f"    {arm:15s}  n={count:4d}  ({pct:5.1f}%)")

        # Response by arm
        print("\n  [BEST OVERALL RESPONSE BY ARM]")
        for arm, responses in r["response_by_arm"].items():
            arm_n = sum(responses.values())
            parts = [f"{k}={v}({100 * v / max(arm_n, 1):.0f}%)" for k, v in responses.items()]
            print(f"    {arm:15s}  {', '.join(parts)}")

        # AE distribution
        print("\n  [ADVERSE EVENT GRADE DISTRIBUTION]")
        for grade, count in r["ae_distribution"].items():
            pct = 100.0 * count / r["n_patients"]
            bar = "#" * int(pct / 2)
            print(f"    {grade:10s}  n={count:4d}  ({pct:5.1f}%)  {bar}")

        # Site enrollment
        print("\n  [SITE ENROLLMENT]")
        for site, count in r["site_enrollment"].items():
            pct = 100.0 * count / r["n_patients"]
            print(f"    {site:10s}  n={count:4d}  ({pct:5.1f}%)")

        print("\n" + "=" * 72)
        print("  RESEARCH USE ONLY -- NOT FOR CLINICAL DECISION MAKING")
        print("=" * 72 + "\n")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the basic clinical analytics pipeline example."""
    logger.info("Starting Example 01: Basic Analytics Pipeline")

    # Step 1 -- Configure
    config = AnalyticsConfig(
        study_id="NCT-PAI-2026-001",
        study_title="PAI Oncology Phase II Basket Trial",
        n_patients=200,
        n_sites=5,
        treatment_arms=["Placebo", "Low_Dose", "High_Dose"],
        seed=42,
    )
    logger.info(
        "Configuration | study=%s | n=%d | arms=%s",
        config.study_id,
        config.n_patients,
        config.treatment_arms,
    )

    # Step 2 -- Initialize orchestrator
    orchestrator = ClinicalAnalyticsOrchestrator(config)

    # Step 3 -- Generate synthetic data
    patients = orchestrator.generate_synthetic_cohort()
    logger.info("Cohort generated | first_patient=%s", patients[0]["patient_id"])

    # Step 4 -- Run descriptive analytics
    results = orchestrator.compute_descriptive_statistics()
    logger.info(
        "Analytics complete | lab_variables=%d | arms=%d",
        len(results["lab_summaries"]),
        len(results["arm_enrollment"]),
    )

    # Step 5 -- Print structured report
    orchestrator.print_report()

    logger.info("Example 01 complete | run_id=%s", results["run_id"])


if __name__ == "__main__":
    main()
