#!/usr/bin/env python3
"""Digital twin treatment planning workflow for oncology clinical trials.

CLINICAL CONTEXT
================
A patient digital twin is a computational model that mirrors a real patient's
tumor characteristics, enabling clinicians to simulate treatment responses
before committing to a regimen.  In a federated trial setting, digital twins
allow each site to explore treatment options locally while sharing only
aggregate model improvements — never raw patient data.  This example
demonstrates twin creation, treatment protocol configuration, multi-modality
simulation, Monte-Carlo uncertainty quantification, and clinical report
generation.

USE CASES COVERED
=================
1. Patient digital twin creation with realistic tumor parameters (volume,
   growth model, histology, biomarkers) and de-identified demographics.
2. Treatment protocol setup including chemotherapy, radiation, immunotherapy,
   and multi-modality combination therapy plans.
3. Simulation execution with configurable growth models (exponential,
   logistic, Gompertz) and response estimation per RECIST-like criteria.
4. Monte-Carlo uncertainty quantification to produce confidence intervals
   on predicted treatment outcomes for clinical decision support.
5. Structured clinical report generation with treatment comparison tables
   and summary statistics suitable for tumor board review.

FRAMEWORK REQUIREMENTS
======================
Required:
    - numpy >= 1.24.0
    - scipy >= 1.11.0

Optional:
    - matplotlib >= 3.7.0  (outcome visualization)
      https://matplotlib.org/
    - pandas >= 2.0.0  (tabular report export)
      https://pandas.pydata.org/

HARDWARE REQUIREMENTS
=====================
    - CPU: Any modern x86-64 or ARM64 processor.
    - RAM: >= 2 GB (digital twin simulations are lightweight).
    - GPU: Not required.

REFERENCES
==========
    - Voigt et al., "Digital Twins for Personalized Oncology", Nature Reviews
      Clinical Oncology, 2023.
      https://www.nature.com/articles/s41571-023-00813-z
    - Eisenhauer et al., "New response evaluation criteria in solid tumours:
      Revised RECIST guideline (version 1.1)", European Journal of Cancer,
      2009.  https://doi.org/10.1016/j.ejca.2008.10.026
    - Rockne et al., "The 2019 mathematical oncology roadmap", Phys. Biol.
      2019.  https://doi.org/10.1088/1478-3975/ab1a09

DISCLAIMER
==========
RESEARCH USE ONLY.  This software is provided for research and educational
purposes only.  It has NOT been validated for clinical use, is NOT approved
by the FDA or any other regulatory body, and MUST NOT be used to make
clinical decisions or direct patient care.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physical_ai.digital_twin import PatientDigitalTwin, TumorModel

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Section 1 — Configuration: Enums + Dataclasses
# ============================================================================


class TreatmentModality(str, Enum):
    """Supported treatment modalities for digital twin simulation."""

    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    IMMUNOTHERAPY = "immunotherapy"
    COMBINATION = "combination"


class GrowthModelType(str, Enum):
    """Tumor growth model selection."""

    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"
    GOMPERTZ = "gompertz"


class ResponseCategory(str, Enum):
    """RECIST-like treatment response categories."""

    COMPLETE_RESPONSE = "complete_response"
    PARTIAL_RESPONSE = "partial_response"
    STABLE_DISEASE = "stable_disease"
    PROGRESSIVE_DISEASE = "progressive_disease"


class ReportFormat(str, Enum):
    """Output format for clinical reports."""

    TEXT = "text"
    JSON = "json"
    DATAFRAME = "dataframe"


class UncertaintyLevel(str, Enum):
    """Pre-configured uncertainty quantification sample counts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


UNCERTAINTY_SAMPLES: dict[UncertaintyLevel, int] = {
    UncertaintyLevel.LOW: 50,
    UncertaintyLevel.MEDIUM: 200,
    UncertaintyLevel.HIGH: 1000,
}


@dataclass
class TumorConfig:
    """Configuration for a patient's tumor model.

    Attributes:
        volume_cm3: Initial tumor volume in cubic centimeters.
        growth_rate: Doubling time in days.
        chemo_sensitivity: Chemotherapy sensitivity coefficient [0, 1].
        radio_sensitivity: Radiation sensitivity coefficient [0, 1].
        immuno_sensitivity: Immunotherapy sensitivity coefficient [0, 1].
        location: Anatomical site.
        histology: Histological classification.
        growth_model: Growth kinetics model.
        carrying_capacity: Maximum volume for logistic/Gompertz models.
    """

    volume_cm3: float = 3.5
    growth_rate: float = 55.0
    chemo_sensitivity: float = 0.55
    radio_sensitivity: float = 0.65
    immuno_sensitivity: float = 0.35
    location: str = "lung"
    histology: str = "adenocarcinoma"
    growth_model: GrowthModelType = GrowthModelType.GOMPERTZ
    carrying_capacity: float = 100.0


@dataclass
class PatientConfig:
    """De-identified patient configuration for digital twin creation.

    Attributes:
        patient_id: De-identified patient identifier.
        age: Patient age (bracket acceptable).
        tumor: Tumor model configuration.
        biomarkers: Dictionary of biomarker name to value.
    """

    patient_id: str = field(default_factory=lambda: f"PT-{uuid.uuid4().hex[:8].upper()}")
    age: int = 62
    tumor: TumorConfig = field(default_factory=TumorConfig)
    biomarkers: dict[str, float] = field(default_factory=dict)


@dataclass
class TreatmentProtocol:
    """A treatment protocol defining one or more modality configurations.

    Attributes:
        protocol_id: Unique protocol identifier.
        name: Human-readable protocol name.
        modality: Primary treatment modality.
        chemo_dose_mg: Chemotherapy dose in milligrams.
        chemo_cycles: Number of chemotherapy cycles.
        radiation_dose_gy: Total radiation dose in Gray.
        radiation_fractions: Number of radiation fractions.
        immunotherapy: Whether to include immunotherapy.
        immuno_response_rate: Population-level immunotherapy response rate.
        duration_weeks: Expected treatment duration in weeks.
        description: Free-text protocol description.
    """

    protocol_id: str = field(default_factory=lambda: f"PROTO-{uuid.uuid4().hex[:6].upper()}")
    name: str = "Standard Protocol"
    modality: TreatmentModality = TreatmentModality.CHEMOTHERAPY
    chemo_dose_mg: float = 75.0
    chemo_cycles: int = 4
    radiation_dose_gy: float = 60.0
    radiation_fractions: int = 30
    immunotherapy: bool = False
    immuno_response_rate: float = 0.3
    duration_weeks: int = 12
    description: str = ""


@dataclass
class SimulationResult:
    """Result from a single treatment simulation run.

    Attributes:
        protocol: The protocol that was simulated.
        initial_volume_cm3: Tumor volume before treatment.
        predicted_volume_cm3: Predicted tumor volume after treatment.
        volume_reduction_pct: Percentage reduction in tumor volume.
        response_category: RECIST-like response classification.
        details: Additional modality-specific result details.
    """

    protocol: TreatmentProtocol = field(default_factory=TreatmentProtocol)
    initial_volume_cm3: float = 0.0
    predicted_volume_cm3: float = 0.0
    volume_reduction_pct: float = 0.0
    response_category: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyResult:
    """Result from Monte-Carlo uncertainty quantification.

    Attributes:
        protocol: The protocol that was simulated.
        n_samples: Number of Monte-Carlo samples.
        mean_volume_cm3: Mean predicted volume.
        std_volume_cm3: Standard deviation of predicted volume.
        p5_volume_cm3: 5th percentile volume.
        p95_volume_cm3: 95th percentile volume.
        mean_reduction_pct: Mean volume reduction percentage.
        std_reduction_pct: Standard deviation of reduction percentage.
        response_distribution: Distribution of response categories.
    """

    protocol: TreatmentProtocol = field(default_factory=TreatmentProtocol)
    n_samples: int = 0
    mean_volume_cm3: float = 0.0
    std_volume_cm3: float = 0.0
    p5_volume_cm3: float = 0.0
    p95_volume_cm3: float = 0.0
    mean_reduction_pct: float = 0.0
    std_reduction_pct: float = 0.0
    response_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class ClinicalReport:
    """Structured clinical report for tumor board review.

    Attributes:
        report_id: Unique report identifier.
        patient_id: De-identified patient identifier.
        timestamp: Report generation timestamp.
        patient_summary: Overview of patient and tumor characteristics.
        simulation_results: Deterministic simulation outcomes.
        uncertainty_results: Monte-Carlo uncertainty analyses.
        recommendation: Summary recommendation text.
        disclaimer: Regulatory disclaimer text.
    """

    report_id: str = field(default_factory=lambda: f"RPT-{uuid.uuid4().hex[:8].upper()}")
    patient_id: str = ""
    timestamp: float = field(default_factory=time.time)
    patient_summary: dict[str, Any] = field(default_factory=dict)
    simulation_results: list[SimulationResult] = field(default_factory=list)
    uncertainty_results: list[UncertaintyResult] = field(default_factory=list)
    recommendation: str = ""
    disclaimer: str = "RESEARCH USE ONLY. Not validated for clinical decisions."


# ============================================================================
# Section 2 — Core Implementation: DigitalTwinPlanner
# ============================================================================


class DigitalTwinPlanner:
    """Treatment planning engine built on patient digital twins.

    Creates digital twins from patient configurations, runs deterministic
    and stochastic treatment simulations, and compiles structured clinical
    reports.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._twins: dict[str, PatientDigitalTwin] = {}
        self._protocols: dict[str, TreatmentProtocol] = {}
        self._simulation_results: list[SimulationResult] = []
        self._uncertainty_results: list[UncertaintyResult] = []

    # ------------------------------------------------------------------
    # Twin creation
    # ------------------------------------------------------------------

    def create_twin(self, patient_config: PatientConfig) -> PatientDigitalTwin:
        """Create a patient digital twin from configuration.

        Args:
            patient_config: Patient and tumor configuration.

        Returns:
            Initialized PatientDigitalTwin instance.
        """
        tumor = TumorModel(
            volume_cm3=patient_config.tumor.volume_cm3,
            growth_rate=patient_config.tumor.growth_rate,
            chemo_sensitivity=patient_config.tumor.chemo_sensitivity,
            radio_sensitivity=patient_config.tumor.radio_sensitivity,
            immuno_sensitivity=patient_config.tumor.immuno_sensitivity,
            location=patient_config.tumor.location,
            histology=patient_config.tumor.histology,
            growth_model=patient_config.tumor.growth_model.value,
            carrying_capacity=patient_config.tumor.carrying_capacity,
        )

        twin = PatientDigitalTwin(
            patient_id=patient_config.patient_id,
            tumor=tumor,
            age=patient_config.age,
            biomarkers=dict(patient_config.biomarkers),
        )

        self._twins[patient_config.patient_id] = twin
        logger.info(
            "Created digital twin: patient=%s, volume=%.2f cm3, growth=%s",
            patient_config.patient_id,
            tumor.volume_cm3,
            tumor.growth_model,
        )
        return twin

    # ------------------------------------------------------------------
    # Protocol management
    # ------------------------------------------------------------------

    def register_protocol(self, protocol: TreatmentProtocol) -> None:
        """Register a treatment protocol for simulation.

        Args:
            protocol: Treatment protocol to register.
        """
        self._protocols[protocol.protocol_id] = protocol
        logger.info("Registered protocol: %s (%s)", protocol.name, protocol.modality.value)

    def create_standard_protocols(self) -> list[TreatmentProtocol]:
        """Create a set of standard oncology treatment protocols.

        Returns:
            List of pre-configured treatment protocols.
        """
        protocols = [
            TreatmentProtocol(
                name="Standard Chemotherapy",
                modality=TreatmentModality.CHEMOTHERAPY,
                chemo_dose_mg=75.0,
                chemo_cycles=6,
                description="Standard-of-care chemotherapy: 6 cycles at 75 mg",
            ),
            TreatmentProtocol(
                name="Definitive Radiation",
                modality=TreatmentModality.RADIATION,
                radiation_dose_gy=60.0,
                radiation_fractions=30,
                description="Definitive radiation therapy: 60 Gy in 30 fractions",
            ),
            TreatmentProtocol(
                name="Immunotherapy (Pembrolizumab-like)",
                modality=TreatmentModality.IMMUNOTHERAPY,
                immunotherapy=True,
                immuno_response_rate=0.35,
                duration_weeks=16,
                description="Checkpoint inhibitor immunotherapy, 16-week course",
            ),
            TreatmentProtocol(
                name="Chemo + Radiation Combo",
                modality=TreatmentModality.COMBINATION,
                chemo_dose_mg=60.0,
                chemo_cycles=4,
                radiation_dose_gy=50.0,
                radiation_fractions=25,
                immunotherapy=False,
                description="Concurrent chemo-radiation: 4 cycles chemo + 50 Gy",
            ),
            TreatmentProtocol(
                name="Tri-modality (Chemo + RT + Immuno)",
                modality=TreatmentModality.COMBINATION,
                chemo_dose_mg=50.0,
                chemo_cycles=4,
                radiation_dose_gy=45.0,
                radiation_fractions=25,
                immunotherapy=True,
                immuno_response_rate=0.30,
                description="Tri-modality: chemo + radiation + immunotherapy consolidation",
            ),
        ]

        for proto in protocols:
            self.register_protocol(proto)

        return protocols

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_protocol(
        self,
        patient_id: str,
        protocol: TreatmentProtocol,
    ) -> SimulationResult:
        """Run a deterministic treatment simulation.

        Args:
            patient_id: De-identified patient identifier.
            protocol: Treatment protocol to simulate.

        Returns:
            SimulationResult with predicted outcomes.
        """
        twin = self._twins.get(patient_id)
        if twin is None:
            raise ValueError(f"No digital twin found for patient {patient_id}")

        kwargs: dict[str, Any] = {}
        treatment_type = protocol.modality.value

        if protocol.modality == TreatmentModality.CHEMOTHERAPY:
            kwargs = {"dose_mg": protocol.chemo_dose_mg, "cycles": protocol.chemo_cycles}
        elif protocol.modality == TreatmentModality.RADIATION:
            kwargs = {"dose_gy": protocol.radiation_dose_gy, "fractions": protocol.radiation_fractions}
        elif protocol.modality == TreatmentModality.IMMUNOTHERAPY:
            kwargs = {
                "response_rate": protocol.immuno_response_rate,
                "duration_weeks": protocol.duration_weeks,
                "seed": self._seed,
            }
        elif protocol.modality == TreatmentModality.COMBINATION:
            kwargs = {
                "chemo_dose_mg": protocol.chemo_dose_mg,
                "chemo_cycles": protocol.chemo_cycles,
                "radiation_dose_gy": protocol.radiation_dose_gy,
                "radiation_fractions": protocol.radiation_fractions,
                "immunotherapy": protocol.immunotherapy,
                "immuno_response_rate": protocol.immuno_response_rate,
                "seed": self._seed,
            }

        outcome = twin.simulate_treatment(treatment_type, **kwargs)

        result = SimulationResult(
            protocol=protocol,
            initial_volume_cm3=outcome["initial_volume_cm3"],
            predicted_volume_cm3=outcome["predicted_volume_cm3"],
            volume_reduction_pct=outcome["volume_reduction_pct"],
            response_category=outcome["response_category"],
            details=outcome,
        )

        self._simulation_results.append(result)
        logger.info(
            "Simulation: %s -> %s (%.1f%% reduction)",
            protocol.name,
            result.response_category,
            result.volume_reduction_pct,
        )
        return result

    # ------------------------------------------------------------------
    # Uncertainty quantification
    # ------------------------------------------------------------------

    def run_uncertainty_analysis(
        self,
        patient_id: str,
        protocol: TreatmentProtocol,
        uncertainty_level: UncertaintyLevel = UncertaintyLevel.MEDIUM,
        param_noise_std: float = 0.1,
    ) -> UncertaintyResult:
        """Run Monte-Carlo uncertainty quantification on a treatment protocol.

        Args:
            patient_id: De-identified patient identifier.
            protocol: Treatment protocol to simulate.
            uncertainty_level: Controls the number of MC samples.
            param_noise_std: Std-dev of perturbation on sensitivity params.

        Returns:
            UncertaintyResult with distributional statistics.
        """
        twin = self._twins.get(patient_id)
        if twin is None:
            raise ValueError(f"No digital twin found for patient {patient_id}")

        n_samples = UNCERTAINTY_SAMPLES[uncertainty_level]
        treatment_type = protocol.modality.value

        kwargs: dict[str, Any] = {}
        if protocol.modality == TreatmentModality.CHEMOTHERAPY:
            kwargs = {"dose_mg": protocol.chemo_dose_mg, "cycles": protocol.chemo_cycles}
        elif protocol.modality == TreatmentModality.RADIATION:
            kwargs = {"dose_gy": protocol.radiation_dose_gy, "fractions": protocol.radiation_fractions}
        elif protocol.modality == TreatmentModality.IMMUNOTHERAPY:
            kwargs = {"response_rate": protocol.immuno_response_rate, "duration_weeks": protocol.duration_weeks}
        elif protocol.modality == TreatmentModality.COMBINATION:
            treatment_type = "combination"
            kwargs = {
                "chemo_dose_mg": protocol.chemo_dose_mg,
                "chemo_cycles": protocol.chemo_cycles,
                "radiation_dose_gy": protocol.radiation_dose_gy,
                "radiation_fractions": protocol.radiation_fractions,
                "immunotherapy": protocol.immunotherapy,
                "immuno_response_rate": protocol.immuno_response_rate,
            }

        mc_result = twin.simulate_with_uncertainty(
            treatment_type=treatment_type,
            n_samples=n_samples,
            param_noise_std=param_noise_std,
            seed=self._seed,
            **kwargs,
        )

        # Count response categories from the simulation log
        response_dist: dict[str, int] = {}
        recent_sims = twin.treatment_history[-n_samples:]
        for sim in recent_sims:
            cat = sim.get("result", {}).get("response_category", "unknown")
            response_dist[cat] = response_dist.get(cat, 0) + 1

        uq_result = UncertaintyResult(
            protocol=protocol,
            n_samples=n_samples,
            mean_volume_cm3=mc_result["mean_volume_cm3"],
            std_volume_cm3=mc_result["std_volume_cm3"],
            p5_volume_cm3=mc_result["p5_volume_cm3"],
            p95_volume_cm3=mc_result["p95_volume_cm3"],
            mean_reduction_pct=mc_result["mean_reduction_pct"],
            std_reduction_pct=mc_result["std_reduction_pct"],
            response_distribution=response_dist,
        )

        self._uncertainty_results.append(uq_result)
        logger.info(
            "UQ analysis: %s — mean reduction=%.1f%% +/- %.1f%%, %d samples",
            protocol.name,
            uq_result.mean_reduction_pct,
            uq_result.std_reduction_pct,
            n_samples,
        )
        return uq_result

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        patient_id: str,
        protocols: list[TreatmentProtocol],
        include_uncertainty: bool = True,
        uncertainty_level: UncertaintyLevel = UncertaintyLevel.LOW,
    ) -> ClinicalReport:
        """Generate a comprehensive clinical report for tumor board review.

        Args:
            patient_id: De-identified patient identifier.
            protocols: Treatment protocols to simulate and compare.
            include_uncertainty: Whether to run MC uncertainty analysis.
            uncertainty_level: MC sample count level.

        Returns:
            ClinicalReport with simulation and UQ results.
        """
        twin = self._twins.get(patient_id)
        if twin is None:
            raise ValueError(f"No digital twin found for patient {patient_id}")

        report = ClinicalReport(patient_id=patient_id, patient_summary=twin.get_summary())

        # Run deterministic simulations
        for protocol in protocols:
            sim_result = self.simulate_protocol(patient_id, protocol)
            report.simulation_results.append(sim_result)

        # Run uncertainty analyses
        if include_uncertainty:
            for protocol in protocols:
                uq_result = self.run_uncertainty_analysis(
                    patient_id,
                    protocol,
                    uncertainty_level=uncertainty_level,
                )
                report.uncertainty_results.append(uq_result)

        # Generate recommendation
        report.recommendation = self._generate_recommendation(report)

        logger.info("Generated clinical report %s for patient %s", report.report_id, patient_id)
        return report

    def _generate_recommendation(self, report: ClinicalReport) -> str:
        """Generate a textual recommendation based on simulation results.

        Args:
            report: Clinical report with completed simulations.

        Returns:
            Summary recommendation string.
        """
        if not report.simulation_results:
            return "Insufficient data for recommendation."

        best_result = max(report.simulation_results, key=lambda r: r.volume_reduction_pct)
        best_name = best_result.protocol.name
        best_reduction = best_result.volume_reduction_pct
        best_response = best_result.response_category

        lines = [
            f"Based on digital twin simulation of {len(report.simulation_results)} protocols:",
            f"  Best predicted outcome: {best_name}",
            f"  Predicted volume reduction: {best_reduction:.1f}%",
            f"  Response category: {best_response}",
        ]

        if report.uncertainty_results:
            best_uq = next((u for u in report.uncertainty_results if u.protocol.name == best_name), None)
            if best_uq is not None:
                lines.append(f"  90% CI volume: [{best_uq.p5_volume_cm3:.2f}, {best_uq.p95_volume_cm3:.2f}] cm3")
                lines.append(
                    f"  Mean reduction: {best_uq.mean_reduction_pct:.1f}% +/- {best_uq.std_reduction_pct:.1f}%"
                )

        lines.append("")
        lines.append("NOTE: This recommendation is generated by computational simulation")
        lines.append("and must be reviewed by a qualified oncologist before clinical action.")

        return "\n".join(lines)


# ============================================================================
# Section 3 — Pipeline Orchestration: TreatmentPlanningPipeline
# ============================================================================


class TreatmentPlanningPipeline:
    """End-to-end pipeline for digital twin-based treatment planning.

    Orchestrates twin creation for multiple patients, runs a battery of
    treatment simulations, and generates comparative reports.
    """

    def __init__(self, seed: int = 42) -> None:
        self._planner = DigitalTwinPlanner(seed=seed)
        self._reports: list[ClinicalReport] = []
        self._seed = seed

    def create_patient_cohort(self, patient_configs: list[PatientConfig]) -> list[PatientDigitalTwin]:
        """Create digital twins for a cohort of patients.

        Args:
            patient_configs: List of patient configurations.

        Returns:
            List of created digital twin instances.
        """
        twins = []
        for config in patient_configs:
            twin = self._planner.create_twin(config)
            twins.append(twin)
        logger.info("Created cohort of %d digital twins", len(twins))
        return twins

    def run_planning_session(
        self,
        patient_ids: list[str],
        protocols: list[TreatmentProtocol] | None = None,
        include_uncertainty: bool = True,
    ) -> list[ClinicalReport]:
        """Run treatment planning for a set of patients.

        Args:
            patient_ids: List of patient identifiers.
            protocols: Treatment protocols to evaluate (creates defaults if None).
            include_uncertainty: Whether to include MC uncertainty analysis.

        Returns:
            List of clinical reports, one per patient.
        """
        if protocols is None:
            protocols = self._planner.create_standard_protocols()

        reports = []
        for pid in patient_ids:
            report = self._planner.generate_report(
                pid,
                protocols,
                include_uncertainty=include_uncertainty,
                uncertainty_level=UncertaintyLevel.LOW,
            )
            reports.append(report)
            self._reports.append(report)

        logger.info("Completed planning session for %d patients with %d protocols", len(patient_ids), len(protocols))
        return reports

    def print_cohort_summary(self, reports: list[ClinicalReport]) -> None:
        """Print a summary table of cohort treatment planning results.

        Args:
            reports: List of clinical reports to summarise.
        """
        print("\n" + "=" * 90)
        print("  COHORT TREATMENT PLANNING SUMMARY")
        print("=" * 90)

        for report in reports:
            print(f"\n  Patient: {report.patient_id}")
            vol = report.patient_summary.get("tumor_volume_cm3", 0.0)
            loc = report.patient_summary.get("tumor_location", "unknown")
            print(f"  Tumor: {vol:.1f} cm3, {loc}")
            print(f"  {'Protocol':<35} {'Reduction':>10} {'Response':<25} {'Volume':>10}")
            print(f"  {'-' * 35} {'-' * 10} {'-' * 25} {'-' * 10}")

            for sim in report.simulation_results:
                print(
                    f"  {sim.protocol.name:<35} "
                    f"{sim.volume_reduction_pct:>9.1f}% "
                    f"{sim.response_category:<25} "
                    f"{sim.predicted_volume_cm3:>9.2f}"
                )

            if report.uncertainty_results:
                print(f"\n  {'Uncertainty Analysis':}")
                print(f"  {'Protocol':<35} {'Mean Red.':>10} {'Std Dev':>10} {'90% CI Volume':>20}")
                print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 20}")
                for uq in report.uncertainty_results:
                    ci_str = f"[{uq.p5_volume_cm3:.2f}, {uq.p95_volume_cm3:.2f}]"
                    print(
                        f"  {uq.protocol.name:<35} "
                        f"{uq.mean_reduction_pct:>9.1f}% "
                        f"{uq.std_reduction_pct:>9.1f}% "
                        f"{ci_str:>20}"
                    )

        print("\n" + "=" * 90)
        print("  DISCLAIMER: RESEARCH USE ONLY")
        print("=" * 90)

    @property
    def reports(self) -> list[ClinicalReport]:
        """Return all generated clinical reports."""
        return list(self._reports)


# ============================================================================
# Section 4 — Demonstration
# ============================================================================


def _create_example_patients() -> list[PatientConfig]:
    """Create a set of example patient configurations."""
    patients = [
        PatientConfig(
            patient_id="PT-LUNG-001",
            age=58,
            tumor=TumorConfig(
                volume_cm3=4.2,
                growth_rate=50.0,
                chemo_sensitivity=0.6,
                radio_sensitivity=0.7,
                immuno_sensitivity=0.4,
                location="lung",
                histology="adenocarcinoma",
                growth_model=GrowthModelType.GOMPERTZ,
            ),
            biomarkers={
                "pdl1_expression": 0.65,
                "ki67_index": 0.35,
                "her2_status": 0.0,
                "er_status": 0.0,
                "egfr_mutation": 1.0,
            },
        ),
        PatientConfig(
            patient_id="PT-BREAST-002",
            age=45,
            tumor=TumorConfig(
                volume_cm3=2.8,
                growth_rate=70.0,
                chemo_sensitivity=0.5,
                radio_sensitivity=0.55,
                immuno_sensitivity=0.25,
                location="breast",
                histology="invasive_ductal",
                growth_model=GrowthModelType.LOGISTIC,
            ),
            biomarkers={
                "pdl1_expression": 0.15,
                "ki67_index": 0.45,
                "her2_status": 1.0,
                "er_status": 1.0,
                "pr_status": 0.8,
            },
        ),
        PatientConfig(
            patient_id="PT-COLON-003",
            age=67,
            tumor=TumorConfig(
                volume_cm3=5.1,
                growth_rate=40.0,
                chemo_sensitivity=0.45,
                radio_sensitivity=0.5,
                immuno_sensitivity=0.3,
                location="colon",
                histology="adenocarcinoma",
                growth_model=GrowthModelType.EXPONENTIAL,
            ),
            biomarkers={
                "pdl1_expression": 0.20,
                "ki67_index": 0.55,
                "msi_status": 0.0,
                "kras_mutation": 1.0,
                "cea_level": 12.5,
            },
        ),
    ]
    return patients


if __name__ == "__main__":
    logger.info("Physical AI Federated Learning — Digital Twin Planning Example v0.5.0")
    logger.info("Optional deps: matplotlib=%s, pandas=%s", HAS_MATPLOTLIB, HAS_PANDAS)

    # Create the pipeline
    pipeline = TreatmentPlanningPipeline(seed=42)

    # Create a patient cohort
    patient_configs = _create_example_patients()
    twins = pipeline.create_patient_cohort(patient_configs)
    patient_ids = [pc.patient_id for pc in patient_configs]

    # Display twin summaries
    print("\n--- Digital Twin Summaries ---")
    for twin in twins:
        summary = twin.get_summary()
        print(
            f"  {summary['patient_id']}: "
            f"volume={summary['tumor_volume_cm3']:.1f} cm3, "
            f"location={summary['tumor_location']}, "
            f"histology={summary['tumor_histology']}, "
            f"growth={summary['growth_model']}"
        )

    # Display feature vectors (for federated model input)
    print("\n--- Feature Vectors (for federated model input) ---")
    for twin in twins:
        features = twin.generate_feature_vector()
        print(f"  {twin.patient_id}: shape={features.shape}, range=[{features.min():.3f}, {features.max():.3f}]")

    # Run planning session with standard protocols
    print("\n--- Running Treatment Planning Session ---")
    reports = pipeline.run_planning_session(patient_ids, include_uncertainty=True)

    # Print cohort summary
    pipeline.print_cohort_summary(reports)

    # Print detailed recommendation for first patient
    print("\n--- Detailed Recommendation (Patient 1) ---")
    print(reports[0].recommendation)

    logger.info("Digital twin planning example complete. Reports generated: %d", len(pipeline.reports))
    print("\nDISCLAIMER: RESEARCH USE ONLY. Not for clinical decision-making.")
