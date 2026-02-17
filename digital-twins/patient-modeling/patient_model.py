"""
Patient Digital Twin Modeling Engine for Oncology Clinical Trials.

CLINICAL CONTEXT:
    Patient digital twins provide computational representations of individual
    oncology patients, capturing tumor growth kinetics, biomarker dynamics,
    organ function, and physiological state. These models enable treatment
    response prediction and federated learning feature generation without
    exposing protected health information (PHI) across institutional boundaries.

    The modeling engine supports multiple tumor histologies (NSCLC, SCLC,
    breast ductal/lobular, colorectal, pancreatic) and growth dynamics
    (exponential, logistic, Gompertz, von Bertalanffy). All clinical
    parameters are bounded to clinically plausible ranges to prevent
    numerical instability and biologically implausible predictions.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0  (https://numpy.org/)
        - scipy >= 1.11.0  (https://scipy.org/)
    Optional:
        - torch >= 2.5.0   (https://pytorch.org/)
        - monai >= 1.3.0   (https://monai.io/)

REFERENCES:
    - Benzekry et al. (2014). Classical mathematical models for description
      and prediction of experimental tumor growth. PLoS Comput Biol, 10(8).
      DOI: 10.1371/journal.pcbi.1003800
    - Eisenhauer et al. (2009). New response evaluation criteria in solid
      tumours: Revised RECIST guideline (version 1.1). Eur J Cancer, 45(2).
      DOI: 10.1016/j.ejca.2008.10.026
    - Murphy et al. (2016). Differences in predictions of ODE models of
      tumor growth. J Math Biol, 73(6-7). DOI: 10.1007/s00285-016-1009-5
    - Ribba et al. (2012). A tumor growth inhibition model for low-grade
      glioma treated with chemotherapy or radiotherapy. Clinical Cancer
      Research, 18(18). DOI: 10.1158/1078-0432.CCR-12-0084

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or guide patient treatment. All simulation results
    must be independently validated before any clinical application.

VERSION: 0.4.0
LICENSE: MIT
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports for optional dependencies
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    from scipy import integrate, optimize

    HAS_SCIPY = True
except ImportError:
    integrate = None  # type: ignore[assignment]
    optimize = None  # type: ignore[assignment]
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants  bounded clinical parameter ranges
# ---------------------------------------------------------------------------
MIN_TUMOR_VOLUME_CM3: float = 0.001
MAX_TUMOR_VOLUME_CM3: float = 1000.0
MIN_GROWTH_RATE_PER_DAY: float = 0.0001
MAX_GROWTH_RATE_PER_DAY: float = 0.5
MIN_SENSITIVITY: float = 0.0
MAX_SENSITIVITY: float = 1.0
MIN_DOSE_MG: float = 0.0
MAX_DOSE_MG: float = 200.0
MIN_BIOMARKER_VALUE: float = 0.0
MAX_BIOMARKER_VALUE: float = 10000.0
DEFAULT_FEATURE_VECTOR_SIZE: int = 64
MAX_SIMULATION_DAYS: int = 3650  # 10 years


# ---------------------------------------------------------------------------
# Enum classes
# ---------------------------------------------------------------------------
class TumorType(Enum):
    """Supported tumor histology types for digital twin modeling.

    Each tumor type carries default growth parameters calibrated from
    published clinical literature. These defaults can be overridden on
    a per-patient basis during twin configuration.
    """

    NSCLC = "non_small_cell_lung_cancer"
    SCLC = "small_cell_lung_cancer"
    BREAST_DUCTAL = "breast_invasive_ductal_carcinoma"
    BREAST_LOBULAR = "breast_invasive_lobular_carcinoma"
    COLORECTAL = "colorectal_adenocarcinoma"
    PANCREATIC = "pancreatic_ductal_adenocarcinoma"
    HEPATOCELLULAR = "hepatocellular_carcinoma"
    RENAL_CLEAR_CELL = "renal_clear_cell_carcinoma"
    MELANOMA = "cutaneous_melanoma"
    GLIOBLASTOMA = "glioblastoma_multiforme"
    OVARIAN = "ovarian_serous_carcinoma"
    PROSTATE = "prostate_adenocarcinoma"


class GrowthModel(Enum):
    """Mathematical models for tumor volume growth dynamics.

    Each model captures different aspects of tumor biology:
    - EXPONENTIAL: Constant doubling time, suitable for early-stage tumors
    - LOGISTIC: Carrying-capacity-limited growth with sigmoidal trajectory
    - GOMPERTZ: Decelerating growth rate, common in solid tumors
    - VON_BERTALANFFY: Energy-balance growth model for metabolic modeling
    """

    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"
    GOMPERTZ = "gompertz"
    VON_BERTALANFFY = "von_bertalanffy"


class PatientStatus(Enum):
    """Patient clinical status within the trial lifecycle.

    Tracks the patient's current phase in treatment to ensure
    appropriate simulation parameters and constraints are applied.
    """

    BASELINE = "baseline"
    ON_TREATMENT = "on_treatment"
    FOLLOW_UP = "follow_up"
    PROGRESSION = "progression"
    REMISSION = "remission"


class BiomarkerType(Enum):
    """Clinically relevant biomarker categories for oncology monitoring."""

    CTDNA = "circulating_tumor_dna"
    CEA = "carcinoembryonic_antigen"
    CA125 = "cancer_antigen_125"
    PSA = "prostate_specific_antigen"
    AFP = "alpha_fetoprotein"
    LDH = "lactate_dehydrogenase"
    NEUTROPHIL_COUNT = "absolute_neutrophil_count"
    PLATELET_COUNT = "platelet_count"
    HEMOGLOBIN = "hemoglobin"
    CREATININE = "serum_creatinine"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TumorProfile:
    """Tumor characteristics for digital twin modeling.

    All numeric parameters are bounded to clinically plausible ranges
    during post-initialization validation.
    """

    tumor_type: TumorType = TumorType.NSCLC
    volume_cm3: float = 2.0
    growth_rate_per_day: float = 0.01
    doubling_time_days: float = 70.0
    chemo_sensitivity: float = 0.5
    radio_sensitivity: float = 0.6
    immuno_sensitivity: float = 0.3
    alpha_beta_ratio: float = 10.0
    carrying_capacity_cm3: float = 500.0
    growth_model: GrowthModel = GrowthModel.GOMPERTZ
    anatomical_location: str = "right_upper_lobe"
    grade: int = 2
    stage: str = "IIB"
    mutation_profile: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and clamp all clinical parameters to safe bounds."""
        self.volume_cm3 = float(np.clip(self.volume_cm3, MIN_TUMOR_VOLUME_CM3, MAX_TUMOR_VOLUME_CM3))
        self.growth_rate_per_day = float(
            np.clip(self.growth_rate_per_day, MIN_GROWTH_RATE_PER_DAY, MAX_GROWTH_RATE_PER_DAY)
        )
        self.chemo_sensitivity = float(np.clip(self.chemo_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.radio_sensitivity = float(np.clip(self.radio_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.immuno_sensitivity = float(np.clip(self.immuno_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.alpha_beta_ratio = float(np.clip(self.alpha_beta_ratio, 1.0, 25.0))
        self.carrying_capacity_cm3 = float(np.clip(self.carrying_capacity_cm3, 1.0, MAX_TUMOR_VOLUME_CM3))
        self.grade = int(np.clip(self.grade, 1, 4))
        if self.doubling_time_days <= 0:
            self.doubling_time_days = math.log(2) / max(self.growth_rate_per_day, MIN_GROWTH_RATE_PER_DAY)
        logger.debug(
            "TumorProfile validated: type=%s, volume=%.3f cm3, growth_rate=%.4f/day",
            self.tumor_type.value,
            self.volume_cm3,
            self.growth_rate_per_day,
        )


@dataclass
class PatientBiomarkers:
    """Collection of clinically relevant biomarker measurements."""

    patient_id: str = ""
    biomarker_values: dict[str, float] = field(default_factory=dict)
    reference_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    measurement_day: int = 0
    unit_map: dict[str, str] = field(default_factory=dict)

    def get_value(self, biomarker_name: str) -> float:
        """Retrieve a biomarker value, returning 0.0 if not measured."""
        return self.biomarker_values.get(biomarker_name, 0.0)

    def is_within_normal(self, biomarker_name: str) -> bool:
        """Check if a biomarker value falls within its reference range."""
        value = self.get_value(biomarker_name)
        if biomarker_name not in self.reference_ranges:
            logger.warning("No reference range for biomarker: %s", biomarker_name)
            return True
        low, high = self.reference_ranges[biomarker_name]
        return low <= value <= high

    def to_array(self, biomarker_order: Sequence[str] | None = None) -> np.ndarray:
        """Convert biomarker values to a NumPy array in specified order.

        Args:
            biomarker_order: Ordered list of biomarker names. If None,
                uses sorted keys from biomarker_values.

        Returns:
            1-D float64 array of biomarker values.
        """
        if biomarker_order is None:
            biomarker_order = sorted(self.biomarker_values.keys())
        values = [self.biomarker_values.get(name, 0.0) for name in biomarker_order]
        return np.array(values, dtype=np.float64)


@dataclass
class OrganFunction:
    """Organ function parameters for treatment tolerability assessment.

    All values are bounded to physiologically plausible ranges.
    """

    hepatic_function_score: float = 0.9
    renal_gfr_ml_min: float = 90.0
    cardiac_lvef_pct: float = 60.0
    pulmonary_fev1_pct: float = 85.0
    bone_marrow_reserve: float = 0.85
    performance_status: int = 1

    def __post_init__(self) -> None:
        """Validate and clamp organ function parameters."""
        self.hepatic_function_score = float(np.clip(self.hepatic_function_score, 0.0, 1.0))
        self.renal_gfr_ml_min = float(np.clip(self.renal_gfr_ml_min, 5.0, 200.0))
        self.cardiac_lvef_pct = float(np.clip(self.cardiac_lvef_pct, 10.0, 80.0))
        self.pulmonary_fev1_pct = float(np.clip(self.pulmonary_fev1_pct, 10.0, 120.0))
        self.bone_marrow_reserve = float(np.clip(self.bone_marrow_reserve, 0.0, 1.0))
        self.performance_status = int(np.clip(self.performance_status, 0, 4))


@dataclass
class PatientDigitalTwinConfig:
    """Complete configuration for instantiating a patient digital twin.

    Aggregates tumor profile, biomarkers, organ function, and metadata
    into a single configuration object for the physiology engine.
    """

    patient_id: str = ""
    tumor_profile: TumorProfile = field(default_factory=TumorProfile)
    biomarkers: PatientBiomarkers = field(default_factory=PatientBiomarkers)
    organ_function: OrganFunction = field(default_factory=OrganFunction)
    status: PatientStatus = PatientStatus.BASELINE
    age_years: int = 60
    weight_kg: float = 70.0
    height_cm: float = 170.0
    sex: str = "unknown"
    enrollment_day: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate patient-level parameters."""
        if not self.patient_id:
            self.patient_id = f"DT-{uuid.uuid4().hex[:12].upper()}"
        self.age_years = int(np.clip(self.age_years, 18, 110))
        self.weight_kg = float(np.clip(self.weight_kg, 20.0, 300.0))
        self.height_cm = float(np.clip(self.height_cm, 100.0, 250.0))
        if self.sex not in ("M", "F", "unknown"):
            self.sex = "unknown"
        self.biomarkers.patient_id = self.patient_id


# ---------------------------------------------------------------------------
# Tumor growth kinetics (NumPy-based computations)
# ---------------------------------------------------------------------------
def _exponential_growth(volume: float, growth_rate: float, days: int) -> np.ndarray:
    """Compute exponential tumor growth trajectory.

    Args:
        volume: Initial tumor volume in cm3.
        growth_rate: Daily fractional growth rate.
        days: Number of days to simulate.

    Returns:
        Array of daily volume values with shape (days + 1,).
    """
    t = np.arange(days + 1, dtype=np.float64)
    trajectory = volume * np.exp(growth_rate * t)
    return np.clip(trajectory, MIN_TUMOR_VOLUME_CM3, MAX_TUMOR_VOLUME_CM3)


def _logistic_growth(volume: float, growth_rate: float, days: int, carrying_capacity: float) -> np.ndarray:
    """Compute logistic tumor growth trajectory.

    Args:
        volume: Initial tumor volume in cm3.
        growth_rate: Daily fractional growth rate.
        days: Number of days to simulate.
        carrying_capacity: Maximum tumor volume in cm3.

    Returns:
        Array of daily volume values with shape (days + 1,).
    """
    t = np.arange(days + 1, dtype=np.float64)
    safe_volume = max(volume, MIN_TUMOR_VOLUME_CM3)
    safe_capacity = max(carrying_capacity, safe_volume + 1.0)
    ratio = safe_volume / safe_capacity
    denominator = 1.0 + ((1.0 / ratio) - 1.0) * np.exp(-growth_rate * t)
    trajectory = safe_capacity / denominator
    return np.clip(trajectory, MIN_TUMOR_VOLUME_CM3, MAX_TUMOR_VOLUME_CM3)


def _gompertz_growth(volume: float, growth_rate: float, days: int, carrying_capacity: float) -> np.ndarray:
    """Compute Gompertz tumor growth trajectory.

    The Gompertz model captures decelerating growth commonly observed
    in solid tumors as they outgrow their vascular supply.

    Args:
        volume: Initial tumor volume in cm3.
        growth_rate: Daily fractional growth rate.
        days: Number of days to simulate.
        carrying_capacity: Asymptotic maximum volume in cm3.

    Returns:
        Array of daily volume values with shape (days + 1,).
    """
    t = np.arange(days + 1, dtype=np.float64)
    safe_volume = max(volume, MIN_TUMOR_VOLUME_CM3)
    safe_capacity = max(carrying_capacity, safe_volume + 1.0)
    beta = np.log(safe_capacity / safe_volume)
    trajectory = safe_capacity * np.exp(-beta * np.exp(-growth_rate * t))
    return np.clip(trajectory, MIN_TUMOR_VOLUME_CM3, MAX_TUMOR_VOLUME_CM3)


def _von_bertalanffy_growth(volume: float, growth_rate: float, days: int, carrying_capacity: float) -> np.ndarray:
    """Compute von Bertalanffy tumor growth trajectory.

    Based on metabolic scaling: growth proportional to surface area
    (V^(2/3)) and decay proportional to volume (V).

    Args:
        volume: Initial tumor volume in cm3.
        growth_rate: Daily fractional growth rate.
        days: Number of days to simulate.
        carrying_capacity: Equilibrium volume in cm3.

    Returns:
        Array of daily volume values with shape (days + 1,).
    """
    t = np.arange(days + 1, dtype=np.float64)
    safe_volume = max(volume, MIN_TUMOR_VOLUME_CM3)
    safe_capacity = max(carrying_capacity, safe_volume + 1.0)
    v0_ratio = (safe_volume / safe_capacity) ** (1.0 / 3.0)
    decay_term = np.exp(-growth_rate * t / 3.0)
    trajectory = safe_capacity * (1.0 - (1.0 - v0_ratio) * decay_term) ** 3.0
    return np.clip(trajectory, MIN_TUMOR_VOLUME_CM3, MAX_TUMOR_VOLUME_CM3)


# ---------------------------------------------------------------------------
# Growth model dispatch
# ---------------------------------------------------------------------------
_GROWTH_DISPATCH = {
    GrowthModel.EXPONENTIAL: _exponential_growth,
    GrowthModel.LOGISTIC: _logistic_growth,
    GrowthModel.GOMPERTZ: _gompertz_growth,
    GrowthModel.VON_BERTALANFFY: _von_bertalanffy_growth,
}


def compute_tumor_trajectory(
    profile: TumorProfile,
    days: int,
    time_step_days: int = 1,
) -> np.ndarray:
    """Compute tumor growth trajectory using the configured growth model.

    Args:
        profile: Tumor profile with growth parameters.
        days: Total simulation duration in days.
        time_step_days: Sampling interval in days.

    Returns:
        Array of volume values sampled at the specified time step.
    """
    clamped_days = int(np.clip(days, 1, MAX_SIMULATION_DAYS))
    growth_fn = _GROWTH_DISPATCH.get(profile.growth_model)
    if growth_fn is None:
        logger.warning(
            "Unknown growth model %s, falling back to GOMPERTZ",
            profile.growth_model,
        )
        growth_fn = _gompertz_growth

    if profile.growth_model == GrowthModel.EXPONENTIAL:
        trajectory = growth_fn(profile.volume_cm3, profile.growth_rate_per_day, clamped_days)
    else:
        trajectory = growth_fn(
            profile.volume_cm3,
            profile.growth_rate_per_day,
            clamped_days,
            profile.carrying_capacity_cm3,
        )

    step = max(1, int(time_step_days))
    return trajectory[::step]


# ---------------------------------------------------------------------------
# PatientPhysiologyEngine
# ---------------------------------------------------------------------------
class PatientPhysiologyEngine:
    """Core simulation engine for patient digital twin physiology.

    Integrates tumor growth kinetics, biomarker evolution, and organ
    function tracking into a unified simulation framework. All outputs
    are de-identified and suitable for federated learning feature
    generation.

    Args:
        config: Patient digital twin configuration.
    """

    def __init__(self, config: PatientDigitalTwinConfig) -> None:
        self.config = config
        self._simulation_history: list[dict[str, Any]] = []
        self._current_day: int = config.enrollment_day
        logger.info(
            "PatientPhysiologyEngine initialized for patient %s (tumor=%s, model=%s)",
            config.patient_id,
            config.tumor_profile.tumor_type.value,
            config.tumor_profile.growth_model.value,
        )

    @property
    def patient_id(self) -> str:
        """Return the de-identified patient identifier."""
        return self.config.patient_id

    @property
    def current_day(self) -> int:
        """Return the current simulation day."""
        return self._current_day

    def simulate_tumor_growth(
        self,
        days: int,
        time_step_days: int = 1,
    ) -> np.ndarray:
        """Simulate tumor volume trajectory over the specified period.

        Args:
            days: Number of days to simulate forward.
            time_step_days: Sampling interval in days.

        Returns:
            Array of tumor volumes (cm3) sampled at the given time step.
        """
        trajectory = compute_tumor_trajectory(
            self.config.tumor_profile,
            days=days,
            time_step_days=time_step_days,
        )
        self._current_day += days
        self._simulation_history.append(
            {
                "event": "tumor_growth",
                "start_day": self._current_day - days,
                "end_day": self._current_day,
                "initial_volume_cm3": float(trajectory[0]),
                "final_volume_cm3": float(trajectory[-1]),
            }
        )
        logger.info(
            "Simulated %d days of tumor growth for %s: %.3f -> %.3f cm3",
            days,
            self.patient_id,
            trajectory[0],
            trajectory[-1],
        )
        return trajectory

    def simulate_biomarker_evolution(
        self,
        biomarker_name: str,
        days: int,
        time_step_days: int = 7,
        correlation_with_tumor: float = 0.8,
        noise_std: float = 0.05,
        seed: int | None = None,
    ) -> np.ndarray:
        """Simulate biomarker time-series correlated with tumor dynamics."""
        rng = np.random.default_rng(seed)
        tumor_trajectory = compute_tumor_trajectory(
            self.config.tumor_profile,
            days=days,
            time_step_days=time_step_days,
        )
        baseline_value = self.config.biomarkers.get_value(biomarker_name)
        if baseline_value <= 0.0:
            baseline_value = 10.0

        # Scale biomarker relative to tumor volume change
        volume_ratio = tumor_trajectory / max(self.config.tumor_profile.volume_cm3, MIN_TUMOR_VOLUME_CM3)
        biomarker_trajectory = baseline_value * (correlation_with_tumor * volume_ratio + (1.0 - correlation_with_tumor))
        # Add measurement noise
        noise = rng.normal(0.0, noise_std * baseline_value, size=biomarker_trajectory.shape)
        biomarker_trajectory = biomarker_trajectory + noise
        biomarker_trajectory = np.clip(biomarker_trajectory, MIN_BIOMARKER_VALUE, MAX_BIOMARKER_VALUE)

        logger.info(
            "Simulated %s evolution for %s over %d days: baseline=%.2f, final=%.2f",
            biomarker_name,
            self.patient_id,
            days,
            biomarker_trajectory[0],
            biomarker_trajectory[-1],
        )
        return biomarker_trajectory

    def assess_organ_function(self, treatment_day: int = 0) -> dict[str, float]:
        """Assess current organ function accounting for treatment effects."""
        organs = self.config.organ_function
        decay_factor = 1.0 - 0.001 * min(treatment_day, 365)
        decay_factor = max(decay_factor, 0.5)

        assessment = {
            "hepatic": float(np.clip(organs.hepatic_function_score * decay_factor, 0.1, 1.0)),
            "renal_gfr": float(np.clip(organs.renal_gfr_ml_min * decay_factor, 10.0, 200.0)),
            "cardiac_lvef": float(np.clip(organs.cardiac_lvef_pct * decay_factor, 15.0, 80.0)),
            "pulmonary_fev1": float(np.clip(organs.pulmonary_fev1_pct * decay_factor, 15.0, 120.0)),
            "bone_marrow": float(np.clip(organs.bone_marrow_reserve * decay_factor, 0.1, 1.0)),
            "performance_status": organs.performance_status,
            "treatment_day": treatment_day,
        }
        logger.debug(
            "Organ function assessment for %s at day %d: %s",
            self.patient_id,
            treatment_day,
            assessment,
        )
        return assessment

    def compute_treatment_tolerability(self, dose_mg: float) -> float:
        """Estimate treatment tolerability score (0-1) based on organ function and dose (0-200 mg)."""
        clamped_dose = float(np.clip(dose_mg, MIN_DOSE_MG, MAX_DOSE_MG))
        organs = self.config.organ_function
        dose_intensity = clamped_dose / MAX_DOSE_MG

        # Weighted organ function composite
        organ_score = (
            0.30 * organs.hepatic_function_score
            + 0.25 * min(organs.renal_gfr_ml_min / 90.0, 1.0)
            + 0.15 * (organs.cardiac_lvef_pct / 60.0)
            + 0.15 * (organs.pulmonary_fev1_pct / 85.0)
            + 0.15 * organs.bone_marrow_reserve
        )
        organ_score = float(np.clip(organ_score, 0.0, 1.0))

        # Performance status penalty
        ps_penalty = 1.0 - (organs.performance_status * 0.15)
        ps_penalty = max(ps_penalty, 0.3)

        tolerability = organ_score * ps_penalty * (1.0 - 0.3 * dose_intensity)
        tolerability = float(np.clip(tolerability, 0.0, 1.0))

        logger.info(
            "Tolerability for %s at dose %.1f mg: %.3f (organ=%.3f, ps_penalty=%.3f)",
            self.patient_id,
            clamped_dose,
            tolerability,
            organ_score,
            ps_penalty,
        )
        return tolerability

    def generate_feature_vector(self) -> np.ndarray:
        """Generate a de-identified feature vector (no PHI) for federated model input."""
        tp = self.config.tumor_profile
        of = self.config.organ_function

        features = [
            tp.volume_cm3 / MAX_TUMOR_VOLUME_CM3,
            tp.growth_rate_per_day / MAX_GROWTH_RATE_PER_DAY,
            tp.chemo_sensitivity,
            tp.radio_sensitivity,
            tp.immuno_sensitivity,
            tp.alpha_beta_ratio / 25.0,
            tp.carrying_capacity_cm3 / MAX_TUMOR_VOLUME_CM3,
            float(tp.grade) / 4.0,
            of.hepatic_function_score,
            min(of.renal_gfr_ml_min / 200.0, 1.0),
            of.cardiac_lvef_pct / 80.0,
            of.pulmonary_fev1_pct / 120.0,
            of.bone_marrow_reserve,
            float(of.performance_status) / 4.0,
            self.config.age_years / 110.0,
            self.config.weight_kg / 300.0,
            self.config.height_cm / 250.0,
            1.0 if self.config.sex == "M" else (0.0 if self.config.sex == "F" else 0.5),
        ]

        # Append biomarker values (normalized)
        biomarker_values = list(self.config.biomarkers.biomarker_values.values())
        for val in biomarker_values[:20]:
            features.append(min(float(val) / MAX_BIOMARKER_VALUE, 1.0))

        # Pad or truncate to fixed size
        if len(features) < DEFAULT_FEATURE_VECTOR_SIZE:
            features.extend([0.0] * (DEFAULT_FEATURE_VECTOR_SIZE - len(features)))
        features = features[:DEFAULT_FEATURE_VECTOR_SIZE]

        feature_array = np.array(features, dtype=np.float64)
        feature_array = np.clip(feature_array, 0.0, 1.0)

        logger.debug(
            "Generated feature vector for %s: shape=%s, mean=%.4f",
            self.patient_id,
            feature_array.shape,
            float(np.mean(feature_array)),
        )
        return feature_array

    def get_simulation_history(self) -> list[dict[str, Any]]:
        """Return the complete simulation history for audit purposes."""
        return list(self._simulation_history)

    def get_summary(self) -> dict[str, Any]:
        """Return a de-identified summary of the digital twin state."""
        return {
            "patient_id": self.patient_id,
            "status": self.config.status.value,
            "tumor_type": self.config.tumor_profile.tumor_type.value,
            "tumor_volume_cm3": self.config.tumor_profile.volume_cm3,
            "growth_model": self.config.tumor_profile.growth_model.value,
            "current_simulation_day": self._current_day,
            "num_simulations": len(self._simulation_history),
            "performance_status": self.config.organ_function.performance_status,
        }


# ---------------------------------------------------------------------------
# PatientDigitalTwinFactory
# ---------------------------------------------------------------------------

# Default growth parameters per tumor type (growth_rate_per_day, doubling_time_days)
_DEFAULT_TUMOR_PARAMS: dict[TumorType, dict[str, float]] = {
    TumorType.NSCLC: {"growth_rate": 0.008, "doubling_time": 86.6, "chemo_sens": 0.45, "radio_sens": 0.55},
    TumorType.SCLC: {"growth_rate": 0.025, "doubling_time": 27.7, "chemo_sens": 0.70, "radio_sens": 0.65},
    TumorType.BREAST_DUCTAL: {"growth_rate": 0.006, "doubling_time": 115.5, "chemo_sens": 0.55, "radio_sens": 0.50},
    TumorType.BREAST_LOBULAR: {"growth_rate": 0.005, "doubling_time": 138.6, "chemo_sens": 0.45, "radio_sens": 0.45},
    TumorType.COLORECTAL: {"growth_rate": 0.010, "doubling_time": 69.3, "chemo_sens": 0.50, "radio_sens": 0.40},
    TumorType.PANCREATIC: {"growth_rate": 0.015, "doubling_time": 46.2, "chemo_sens": 0.35, "radio_sens": 0.30},
    TumorType.HEPATOCELLULAR: {"growth_rate": 0.012, "doubling_time": 57.8, "chemo_sens": 0.30, "radio_sens": 0.35},
    TumorType.RENAL_CLEAR_CELL: {"growth_rate": 0.007, "doubling_time": 99.0, "chemo_sens": 0.25, "radio_sens": 0.30},
    TumorType.MELANOMA: {"growth_rate": 0.018, "doubling_time": 38.5, "chemo_sens": 0.30, "radio_sens": 0.35},
    TumorType.GLIOBLASTOMA: {"growth_rate": 0.020, "doubling_time": 34.7, "chemo_sens": 0.40, "radio_sens": 0.60},
    TumorType.OVARIAN: {"growth_rate": 0.011, "doubling_time": 63.0, "chemo_sens": 0.60, "radio_sens": 0.40},
    TumorType.PROSTATE: {"growth_rate": 0.003, "doubling_time": 231.0, "chemo_sens": 0.35, "radio_sens": 0.55},
}


class PatientDigitalTwinFactory:
    """Factory for creating configured patient digital twins.

    Provides convenient methods to create digital twin configurations
    with tumor-type-specific defaults calibrated from published
    clinical literature. Custom parameter overrides are supported.

    Example::

        factory = PatientDigitalTwinFactory()
        config = factory.create_twin(
            patient_id="TRIAL-001-PT-042",
            tumor_type=TumorType.NSCLC,
            growth_model=GrowthModel.GOMPERTZ,
        )
        engine = PatientPhysiologyEngine(config)
    """

    def __init__(self) -> None:
        self._created_count: int = 0
        logger.info("PatientDigitalTwinFactory initialized")

    def create_twin(
        self,
        patient_id: str = "",
        tumor_type: TumorType = TumorType.NSCLC,
        growth_model: GrowthModel = GrowthModel.GOMPERTZ,
        initial_volume_cm3: float = 2.0,
        age_years: int = 60,
        sex: str = "unknown",
        weight_kg: float = 70.0,
        height_cm: float = 170.0,
        performance_status: int = 1,
        biomarker_values: dict[str, float] | None = None,
        mutation_profile: dict[str, bool] | None = None,
        **kwargs: Any,
    ) -> PatientDigitalTwinConfig:
        """Create a fully configured patient digital twin with tumor-type defaults."""
        defaults = _DEFAULT_TUMOR_PARAMS.get(tumor_type, _DEFAULT_TUMOR_PARAMS[TumorType.NSCLC])

        tumor_profile = TumorProfile(
            tumor_type=tumor_type,
            volume_cm3=initial_volume_cm3,
            growth_rate_per_day=defaults["growth_rate"],
            doubling_time_days=defaults["doubling_time"],
            chemo_sensitivity=defaults["chemo_sens"],
            radio_sensitivity=defaults.get("radio_sens", 0.5),
            immuno_sensitivity=kwargs.get("immuno_sensitivity", 0.3),
            growth_model=growth_model,
            mutation_profile=mutation_profile or {},
        )

        biomarkers = PatientBiomarkers(
            patient_id=patient_id,
            biomarker_values=biomarker_values or {},
        )

        organ_function = OrganFunction(
            performance_status=performance_status,
        )

        config = PatientDigitalTwinConfig(
            patient_id=patient_id,
            tumor_profile=tumor_profile,
            biomarkers=biomarkers,
            organ_function=organ_function,
            age_years=age_years,
            sex=sex,
            weight_kg=weight_kg,
            height_cm=height_cm,
            metadata=kwargs,
        )

        self._created_count += 1
        logger.info(
            "Created digital twin #%d: patient=%s, tumor=%s, model=%s",
            self._created_count,
            config.patient_id,
            tumor_type.value,
            growth_model.value,
        )
        return config

    def create_cohort(
        self,
        n_patients: int,
        tumor_type: TumorType = TumorType.NSCLC,
        growth_model: GrowthModel = GrowthModel.GOMPERTZ,
        seed: int = 42,
        volume_mean: float = 5.0,
        volume_std: float = 3.0,
        age_mean: int = 62,
        age_std: int = 10,
    ) -> list[PatientDigitalTwinConfig]:
        """Create a synthetic cohort of patient digital twins with stochastic variation."""
        rng = np.random.default_rng(seed)
        cohort: list[PatientDigitalTwinConfig] = []

        for i in range(n_patients):
            volume = float(np.clip(rng.normal(volume_mean, volume_std), MIN_TUMOR_VOLUME_CM3, MAX_TUMOR_VOLUME_CM3))
            age = int(np.clip(rng.normal(age_mean, age_std), 18, 110))
            sex = rng.choice(["M", "F"])

            config = self.create_twin(
                patient_id=f"COHORT-{i + 1:04d}",
                tumor_type=tumor_type,
                growth_model=growth_model,
                initial_volume_cm3=volume,
                age_years=age,
                sex=sex,
            )
            cohort.append(config)

        logger.info(
            "Created synthetic cohort: n=%d, tumor=%s, model=%s",
            n_patients,
            tumor_type.value,
            growth_model.value,
        )
        return cohort

    @property
    def created_count(self) -> int:
        """Return the total number of twins created by this factory."""
        return self._created_count
