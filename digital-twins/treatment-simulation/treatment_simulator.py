"""
Treatment Simulation Engine for Oncology Digital Twins.

CLINICAL CONTEXT:
    Multi-modality treatment simulation for oncology clinical trials.
    Supports chemotherapy PK/PD, radiation LQ model, immunotherapy stochastic
    response, targeted therapy, and combination therapy. All doses bounded
    to clinically plausible ranges. RECIST 1.1 response, CTCAE v5.0 toxicity.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0  (https://numpy.org/)
        - scipy >= 1.11.0  (https://scipy.org/)
    Optional:
        - torch >= 2.5.0   (https://pytorch.org/)
        - monai >= 1.3.0   (https://monai.io/)

REFERENCES:
    - Eisenhauer et al. (2009). RECIST 1.1. DOI: 10.1016/j.ejca.2008.10.026
    - NCI (2017). CTCAE v5.0. URL: https://ctep.cancer.gov/
    - Joiner & van der Kogel (2009). Basic Clinical Radiobiology. DOI: 10.1201/b13224
    - Fang et al. (2022). PBPK in oncology. DOI: 10.1002/psp4.12845

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
    from scipy import integrate, optimize, stats

    HAS_SCIPY = True
except ImportError:
    integrate = None  # type: ignore[assignment]
    optimize = None  # type: ignore[assignment]
    stats = None  # type: ignore[assignment]
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
MIN_DOSE_MG: float = 0.0
MAX_DOSE_MG: float = 200.0
MIN_DOSE_GY: float = 0.0
MAX_DOSE_GY: float = 80.0
MIN_FRACTIONS: int = 1
MAX_FRACTIONS: int = 45
MIN_CYCLES: int = 1
MAX_CYCLES: int = 12
MIN_VOLUME_CM3: float = 0.001
MAX_VOLUME_CM3: float = 1000.0
MIN_SENSITIVITY: float = 0.0
MAX_SENSITIVITY: float = 1.0
PK_ELIMINATION_HALF_LIFE_HOURS: float = 6.0
PK_VOLUME_OF_DISTRIBUTION_L: float = 50.0
PK_BIOAVAILABILITY: float = 0.85


# ---------------------------------------------------------------------------
# Enum classes
# ---------------------------------------------------------------------------
class TreatmentModality(Enum):
    """Supported treatment modalities for simulation."""

    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    IMMUNOTHERAPY = "immunotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    COMBINATION = "combination"


class ResponseCategory(Enum):
    """RECIST 1.1 treatment response classification."""

    COMPLETE_RESPONSE = "complete_response"
    PARTIAL_RESPONSE = "partial_response"
    STABLE_DISEASE = "stable_disease"
    PROGRESSIVE_DISEASE = "progressive_disease"


class ToxicityGrade(Enum):
    """CTCAE v5.0 toxicity grading scale (Grade 0-5)."""

    GRADE_0 = 0
    GRADE_1 = 1
    GRADE_2 = 2
    GRADE_3 = 3
    GRADE_4 = 4
    GRADE_5 = 5


class DrugClass(Enum):
    """Chemotherapy drug classification for PK/PD modeling."""

    ALKYLATING = "alkylating_agent"
    ANTIMETABOLITE = "antimetabolite"
    ANTHRACYCLINE = "anthracycline"
    TAXANE = "taxane"
    PLATINUM = "platinum_compound"
    VINCA_ALKALOID = "vinca_alkaloid"
    TOPOISOMERASE_INHIBITOR = "topoisomerase_inhibitor"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ChemotherapyProtocol:
    """Chemotherapy regimen parameters for simulation."""

    drug_name: str = "carboplatin"
    drug_class: DrugClass = DrugClass.PLATINUM
    dose_mg_per_m2: float = 75.0
    cycles: int = 4
    cycle_length_days: int = 21
    dose_reduction_factor: float = 1.0
    infusion_duration_hours: float = 1.0
    tumor_sensitivity: float = 0.5

    def __post_init__(self) -> None:
        """Validate and clamp chemotherapy parameters."""
        self.dose_mg_per_m2 = float(np.clip(self.dose_mg_per_m2, MIN_DOSE_MG, MAX_DOSE_MG))
        self.cycles = int(np.clip(self.cycles, MIN_CYCLES, MAX_CYCLES))
        self.cycle_length_days = int(np.clip(self.cycle_length_days, 7, 42))
        self.dose_reduction_factor = float(np.clip(self.dose_reduction_factor, 0.0, 1.0))
        self.infusion_duration_hours = float(np.clip(self.infusion_duration_hours, 0.1, 24.0))
        self.tumor_sensitivity = float(np.clip(self.tumor_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))


@dataclass
class RadiationPlan:
    """Radiation therapy plan parameters for LQ model simulation."""

    total_dose_gy: float = 60.0
    fractions: int = 30
    dose_per_fraction_gy: float = 0.0
    alpha_beta_ratio: float = 10.0
    tumor_sensitivity: float = 0.6
    organ_at_risk_dose_gy: float = 45.0
    treatment_days_per_week: int = 5
    technique: str = "IMRT"

    def __post_init__(self) -> None:
        """Validate and compute derived radiation parameters."""
        self.total_dose_gy = float(np.clip(self.total_dose_gy, MIN_DOSE_GY, MAX_DOSE_GY))
        self.fractions = int(np.clip(self.fractions, MIN_FRACTIONS, MAX_FRACTIONS))
        self.dose_per_fraction_gy = self.total_dose_gy / max(self.fractions, 1)
        self.alpha_beta_ratio = float(np.clip(self.alpha_beta_ratio, 1.0, 25.0))
        self.tumor_sensitivity = float(np.clip(self.tumor_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.organ_at_risk_dose_gy = float(np.clip(self.organ_at_risk_dose_gy, 0.0, self.total_dose_gy))
        self.treatment_days_per_week = int(np.clip(self.treatment_days_per_week, 1, 7))


@dataclass
class ImmunotherapyRegimen:
    """Immunotherapy protocol parameters for simulation."""

    agent_name: str = "pembrolizumab"
    dose_mg_per_kg: float = 2.0
    cycle_length_days: int = 21
    total_cycles: int = 8
    expected_response_rate: float = 0.35
    tumor_sensitivity: float = 0.3
    tumor_mutational_burden: float = 10.0
    pdl1_expression_pct: float = 50.0

    def __post_init__(self) -> None:
        """Validate immunotherapy parameters."""
        self.dose_mg_per_kg = float(np.clip(self.dose_mg_per_kg, 0.1, 20.0))
        self.cycle_length_days = int(np.clip(self.cycle_length_days, 7, 42))
        self.total_cycles = int(np.clip(self.total_cycles, 1, 24))
        self.expected_response_rate = float(np.clip(self.expected_response_rate, 0.0, 1.0))
        self.tumor_sensitivity = float(np.clip(self.tumor_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.tumor_mutational_burden = float(np.clip(self.tumor_mutational_burden, 0.0, 100.0))
        self.pdl1_expression_pct = float(np.clip(self.pdl1_expression_pct, 0.0, 100.0))


@dataclass
class TreatmentOutcome:
    """Result of a treatment simulation with RECIST 1.1 response classification."""

    simulation_id: str = ""
    modality: TreatmentModality = TreatmentModality.CHEMOTHERAPY
    response_category: ResponseCategory = ResponseCategory.STABLE_DISEASE
    initial_volume_cm3: float = 0.0
    final_volume_cm3: float = 0.0
    volume_reduction_pct: float = 0.0
    volume_trajectory: list[float] = field(default_factory=list)
    confidence_interval_low: float = 0.0
    confidence_interval_high: float = 0.0
    treatment_duration_days: int = 0
    toxicity_profile: Optional[ToxicityProfile] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate simulation ID if not provided."""
        if not self.simulation_id:
            self.simulation_id = f"SIM-{uuid.uuid4().hex[:12].upper()}"


@dataclass
class ToxicityProfile:
    """Multi-organ CTCAE v5.0 toxicity assessment from treatment simulation."""

    overall_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    hematologic_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    hepatic_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    renal_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    cardiac_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    neurologic_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    gastrointestinal_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    dermatologic_grade: ToxicityGrade = ToxicityGrade.GRADE_0
    dose_limiting: bool = False
    requires_dose_reduction: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _classify_response(volume_reduction_fraction: float) -> ResponseCategory:
    """Classify treatment response using RECIST 1.1 criteria."""
    if volume_reduction_fraction >= 0.65:
        return ResponseCategory.COMPLETE_RESPONSE
    elif volume_reduction_fraction >= 0.30:
        return ResponseCategory.PARTIAL_RESPONSE
    elif volume_reduction_fraction >= -0.20:
        return ResponseCategory.STABLE_DISEASE
    else:
        return ResponseCategory.PROGRESSIVE_DISEASE


def _compute_pk_concentration(
    dose_mg: float,
    time_hours: float,
    elimination_half_life_hours: float = PK_ELIMINATION_HALF_LIFE_HOURS,
    volume_of_distribution_l: float = PK_VOLUME_OF_DISTRIBUTION_L,
    bioavailability: float = PK_BIOAVAILABILITY,
) -> float:
    """Compute plasma concentration (mg/L) using one-compartment PK model."""
    clamped_dose = float(np.clip(dose_mg, MIN_DOSE_MG, MAX_DOSE_MG))
    ke = np.log(2) / max(elimination_half_life_hours, 0.1)
    c0 = (clamped_dose * bioavailability) / max(volume_of_distribution_l, 1.0)
    concentration = c0 * np.exp(-ke * max(time_hours, 0.0))
    return float(max(concentration, 0.0))


def _compute_auc(
    dose_mg: float,
    elimination_half_life_hours: float = PK_ELIMINATION_HALF_LIFE_HOURS,
    volume_of_distribution_l: float = PK_VOLUME_OF_DISTRIBUTION_L,
    bioavailability: float = PK_BIOAVAILABILITY,
) -> float:
    """Compute AUC (mg*h/L) using trapezoidal approximation over 72 hours."""
    time_points = np.linspace(0.0, 72.0, 361)
    concentrations = np.array(
        [
            _compute_pk_concentration(
                dose_mg, t, elimination_half_life_hours, volume_of_distribution_l, bioavailability
            )
            for t in time_points
        ]
    )
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise RuntimeError("NumPy has no 'trapezoid' or 'trapz' function; cannot compute AUC.")
    auc = float(_trapz(concentrations, time_points))
    return max(auc, 0.0)


def _estimate_toxicity(
    modality: TreatmentModality,
    dose_intensity: float,
    cumulative_cycles: int,
    organ_sensitivity: float = 0.5,
    seed: int | None = None,
) -> ToxicityProfile:
    """Estimate CTCAE v5.0 toxicity profile using stochastic dose-response model."""
    rng = np.random.default_rng(seed)
    dose_intensity = float(np.clip(dose_intensity, 0.0, 1.0))
    organ_sensitivity = float(np.clip(organ_sensitivity, 0.0, 1.0))

    cumulative_factor = min(cumulative_cycles / 8.0, 1.5)
    base_risk = dose_intensity * organ_sensitivity * cumulative_factor

    def _sample_grade(risk: float) -> ToxicityGrade:
        """Sample a toxicity grade from risk probability."""
        risk = float(np.clip(risk, 0.0, 1.0))
        rand_val = rng.random()
        if rand_val > risk:
            return ToxicityGrade.GRADE_0
        elif rand_val > risk * 0.6:
            return ToxicityGrade.GRADE_1
        elif rand_val > risk * 0.3:
            return ToxicityGrade.GRADE_2
        elif rand_val > risk * 0.1:
            return ToxicityGrade.GRADE_3
        else:
            return ToxicityGrade.GRADE_4

    # Modality-specific risk multipliers for each organ system
    risk_multipliers = {
        TreatmentModality.CHEMOTHERAPY: {
            "hematologic": 1.2,
            "hepatic": 0.8,
            "renal": 0.7,
            "cardiac": 0.5,
            "neurologic": 0.6,
            "gi": 1.0,
            "dermatologic": 0.4,
        },
        TreatmentModality.RADIATION: {
            "hematologic": 0.5,
            "hepatic": 0.4,
            "renal": 0.3,
            "cardiac": 0.6,
            "neurologic": 0.4,
            "gi": 0.8,
            "dermatologic": 0.9,
        },
        TreatmentModality.IMMUNOTHERAPY: {
            "hematologic": 0.3,
            "hepatic": 0.7,
            "renal": 0.5,
            "cardiac": 0.3,
            "neurologic": 0.4,
            "gi": 0.8,
            "dermatologic": 0.9,
        },
        TreatmentModality.TARGETED_THERAPY: {
            "hematologic": 0.6,
            "hepatic": 0.8,
            "renal": 0.5,
            "cardiac": 0.7,
            "neurologic": 0.3,
            "gi": 0.7,
            "dermatologic": 1.0,
        },
    }
    multipliers = risk_multipliers.get(modality, risk_multipliers[TreatmentModality.CHEMOTHERAPY])

    hemat = _sample_grade(base_risk * multipliers["hematologic"])
    hepat = _sample_grade(base_risk * multipliers["hepatic"])
    renal = _sample_grade(base_risk * multipliers["renal"])
    cardiac = _sample_grade(base_risk * multipliers["cardiac"])
    neuro = _sample_grade(base_risk * multipliers["neurologic"])
    gi_tox = _sample_grade(base_risk * multipliers["gi"])
    derm = _sample_grade(base_risk * multipliers["dermatologic"])

    all_grades = [hemat, hepat, renal, cardiac, neuro, gi_tox, derm]
    max_grade_val = max(g.value for g in all_grades)
    overall = ToxicityGrade(min(max_grade_val, 4))
    dose_limiting = max_grade_val >= 3
    requires_reduction = max_grade_val >= 2

    return ToxicityProfile(
        overall_grade=overall,
        hematologic_grade=hemat,
        hepatic_grade=hepat,
        renal_grade=renal,
        cardiac_grade=cardiac,
        neurologic_grade=neuro,
        gastrointestinal_grade=gi_tox,
        dermatologic_grade=derm,
        dose_limiting=dose_limiting,
        requires_dose_reduction=requires_reduction,
        details={"modality": modality.value, "dose_intensity": dose_intensity},
    )


# ---------------------------------------------------------------------------
# TreatmentSimulator
# ---------------------------------------------------------------------------
class TreatmentSimulator:
    """Core treatment simulation engine for oncology digital twins.

    Simulates treatment response across multiple modalities using PK/PD
    models for chemotherapy, the linear-quadratic model for radiation,
    and stochastic immune response models for immunotherapy. All doses
    and volumes are bounded; response follows RECIST 1.1 criteria.
    """

    def __init__(
        self,
        initial_volume_cm3: float = 5.0,
        chemo_sensitivity: float = 0.5,
        radio_sensitivity: float = 0.6,
        immuno_sensitivity: float = 0.3,
        alpha_beta_ratio: float = 10.0,
        growth_rate_per_day: float = 0.01,
    ) -> None:
        self.initial_volume_cm3 = float(np.clip(initial_volume_cm3, MIN_VOLUME_CM3, MAX_VOLUME_CM3))
        self.chemo_sensitivity = float(np.clip(chemo_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.radio_sensitivity = float(np.clip(radio_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.immuno_sensitivity = float(np.clip(immuno_sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY))
        self.alpha_beta_ratio = float(np.clip(alpha_beta_ratio, 1.0, 25.0))
        self.growth_rate_per_day = float(np.clip(growth_rate_per_day, 0.0001, 0.5))
        self._simulation_count: int = 0
        self._history: list[TreatmentOutcome] = []
        logger.info(
            "TreatmentSimulator initialized: volume=%.3f cm3, chemo_sens=%.2f, radio_sens=%.2f",
            self.initial_volume_cm3,
            self.chemo_sensitivity,
            self.radio_sensitivity,
        )

    def simulate_chemotherapy(
        self,
        protocol: ChemotherapyProtocol | None = None,
        seed: int | None = None,
    ) -> TreatmentOutcome:
        """Simulate chemotherapy response using PK/PD modeling with cycle-by-cycle tracking."""
        if protocol is None:
            protocol = ChemotherapyProtocol()

        effective_dose = protocol.dose_mg_per_m2 * protocol.dose_reduction_factor
        effective_dose = float(np.clip(effective_dose, MIN_DOSE_MG, MAX_DOSE_MG))

        auc = _compute_auc(effective_dose)
        volume = self.initial_volume_cm3
        trajectory = [volume]

        for cycle_idx in range(protocol.cycles):
            # Drug effect: kill fraction proportional to AUC and sensitivity
            kill_fraction = protocol.tumor_sensitivity * self.chemo_sensitivity * min(auc / 100.0, 0.8)
            kill_fraction = float(np.clip(kill_fraction, 0.0, 0.8))
            volume = volume * (1.0 - kill_fraction)
            volume = float(np.clip(volume, MIN_VOLUME_CM3, MAX_VOLUME_CM3))

            # Inter-cycle regrowth
            regrowth_days = protocol.cycle_length_days
            volume = volume * np.exp(self.growth_rate_per_day * regrowth_days)
            volume = float(np.clip(volume, MIN_VOLUME_CM3, MAX_VOLUME_CM3))
            trajectory.append(volume)

            logger.debug(
                "Chemo cycle %d/%d: kill=%.3f, volume=%.3f cm3",
                cycle_idx + 1,
                protocol.cycles,
                kill_fraction,
                volume,
            )

        reduction = (self.initial_volume_cm3 - volume) / max(self.initial_volume_cm3, MIN_VOLUME_CM3)
        response = _classify_response(reduction)
        duration_days = protocol.cycles * protocol.cycle_length_days

        # Confidence interval via parameter perturbation
        ci_low, ci_high = self._compute_confidence_interval(volume, reduction, 0.1, seed=seed)

        toxicity = _estimate_toxicity(
            TreatmentModality.CHEMOTHERAPY,
            dose_intensity=effective_dose / MAX_DOSE_MG,
            cumulative_cycles=protocol.cycles,
            seed=seed,
        )

        outcome = TreatmentOutcome(
            modality=TreatmentModality.CHEMOTHERAPY,
            response_category=response,
            initial_volume_cm3=self.initial_volume_cm3,
            final_volume_cm3=volume,
            volume_reduction_pct=reduction * 100.0,
            volume_trajectory=trajectory,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            treatment_duration_days=duration_days,
            toxicity_profile=toxicity,
            metadata={"protocol": protocol.drug_name, "auc": auc},
        )

        self._simulation_count += 1
        self._history.append(outcome)
        logger.info(
            "Chemotherapy simulation complete: %s (%.1f%% reduction, %s toxicity)",
            response.value,
            reduction * 100.0,
            toxicity.overall_grade.name,
        )
        return outcome

    def simulate_radiation(
        self,
        plan: RadiationPlan | None = None,
        seed: int | None = None,
    ) -> TreatmentOutcome:
        """Simulate radiation response using LQ model: SF = exp(-n*(alpha*d + beta*d^2))."""
        if plan is None:
            plan = RadiationPlan()

        dose_per_fraction = plan.dose_per_fraction_gy
        alpha = 0.3 * plan.tumor_sensitivity * self.radio_sensitivity
        beta = alpha / max(plan.alpha_beta_ratio, 1.0)

        # Linear-quadratic surviving fraction
        sf = np.exp(-plan.fractions * (alpha * dose_per_fraction + beta * dose_per_fraction**2))
        sf = float(np.clip(sf, 0.001, 1.0))

        # Compute volume trajectory (fraction by fraction)
        volume = self.initial_volume_cm3
        trajectory = [volume]
        fraction_sf = np.exp(-(alpha * dose_per_fraction + beta * dose_per_fraction**2))
        fraction_sf = float(np.clip(fraction_sf, 0.001, 1.0))

        treatment_day = 0
        for frac_idx in range(plan.fractions):
            volume = volume * fraction_sf
            volume = float(np.clip(volume, MIN_VOLUME_CM3, MAX_VOLUME_CM3))

            # Model inter-fraction repopulation (weekday treatments only)
            treatment_day += 1
            if treatment_day % plan.treatment_days_per_week == 0:
                weekend_days = 7 - plan.treatment_days_per_week
                volume = volume * np.exp(self.growth_rate_per_day * weekend_days)
                volume = float(np.clip(volume, MIN_VOLUME_CM3, MAX_VOLUME_CM3))

            if frac_idx % max(plan.fractions // 10, 1) == 0:
                trajectory.append(volume)

        trajectory.append(volume)
        final_volume = volume

        reduction = (self.initial_volume_cm3 - final_volume) / max(self.initial_volume_cm3, MIN_VOLUME_CM3)
        response = _classify_response(reduction)
        total_days = int(np.ceil(plan.fractions / plan.treatment_days_per_week) * 7)

        ci_low, ci_high = self._compute_confidence_interval(final_volume, reduction, 0.08, seed=seed)

        toxicity = _estimate_toxicity(
            TreatmentModality.RADIATION,
            dose_intensity=plan.total_dose_gy / MAX_DOSE_GY,
            cumulative_cycles=plan.fractions,
            seed=seed,
        )

        outcome = TreatmentOutcome(
            modality=TreatmentModality.RADIATION,
            response_category=response,
            initial_volume_cm3=self.initial_volume_cm3,
            final_volume_cm3=final_volume,
            volume_reduction_pct=reduction * 100.0,
            volume_trajectory=trajectory,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            treatment_duration_days=total_days,
            toxicity_profile=toxicity,
            metadata={
                "technique": plan.technique,
                "surviving_fraction": sf,
                "bed": float(plan.total_dose_gy * (1 + dose_per_fraction / plan.alpha_beta_ratio)),
            },
        )

        self._simulation_count += 1
        self._history.append(outcome)
        logger.info(
            "Radiation simulation complete: %s (%.1f%% reduction, SF=%.4f)",
            response.value,
            reduction * 100.0,
            sf,
        )
        return outcome

    def simulate_immunotherapy(
        self,
        regimen: ImmunotherapyRegimen | None = None,
        seed: int | None = None,
    ) -> TreatmentOutcome:
        """Simulate immunotherapy response with stochastic immune dynamics based on TMB and PD-L1."""
        if regimen is None:
            regimen = ImmunotherapyRegimen()

        rng = np.random.default_rng(seed)

        # Compute effective response probability
        tmb_factor = min(regimen.tumor_mutational_burden / 20.0, 1.5)
        pdl1_factor = regimen.pdl1_expression_pct / 100.0
        sensitivity_factor = 0.5 + self.immuno_sensitivity
        effective_rate = regimen.expected_response_rate * tmb_factor * pdl1_factor * sensitivity_factor
        effective_rate = float(np.clip(effective_rate, 0.0, 0.95))

        responds = rng.random() < effective_rate
        total_days = regimen.total_cycles * regimen.cycle_length_days

        volume = self.initial_volume_cm3
        trajectory = [volume]
        days_per_step = regimen.cycle_length_days

        for cycle_idx in range(regimen.total_cycles):
            if responds:
                # Immune-mediated tumor decay
                decay_rate = 0.03 * self.immuno_sensitivity * regimen.tumor_sensitivity
                decay_rate = float(np.clip(decay_rate, 0.001, 0.1))
                volume = volume * np.exp(-decay_rate * days_per_step)
            else:
                # Progressive disease with possible pseudoprogression
                if cycle_idx < 2:
                    # Initial pseudoprogression phase
                    volume = volume * np.exp(self.growth_rate_per_day * 0.5 * days_per_step)
                else:
                    # Continued progression
                    volume = volume * np.exp(self.growth_rate_per_day * days_per_step)

            volume = float(np.clip(volume, MIN_VOLUME_CM3, MAX_VOLUME_CM3))
            trajectory.append(volume)

        reduction = (self.initial_volume_cm3 - volume) / max(self.initial_volume_cm3, MIN_VOLUME_CM3)
        response = _classify_response(reduction)

        ci_low, ci_high = self._compute_confidence_interval(volume, reduction, 0.15, seed=seed)

        toxicity = _estimate_toxicity(
            TreatmentModality.IMMUNOTHERAPY,
            dose_intensity=regimen.dose_mg_per_kg / 10.0,
            cumulative_cycles=regimen.total_cycles,
            seed=seed,
        )

        outcome = TreatmentOutcome(
            modality=TreatmentModality.IMMUNOTHERAPY,
            response_category=response,
            initial_volume_cm3=self.initial_volume_cm3,
            final_volume_cm3=volume,
            volume_reduction_pct=reduction * 100.0,
            volume_trajectory=trajectory,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            treatment_duration_days=total_days,
            toxicity_profile=toxicity,
            metadata={
                "agent": regimen.agent_name,
                "responded": responds,
                "effective_rate": effective_rate,
                "tmb": regimen.tumor_mutational_burden,
                "pdl1": regimen.pdl1_expression_pct,
            },
        )

        self._simulation_count += 1
        self._history.append(outcome)
        logger.info(
            "Immunotherapy simulation complete: %s (responded=%s, %.1f%% reduction)",
            response.value,
            responds,
            reduction * 100.0,
        )
        return outcome

    def simulate_combination(
        self,
        chemo_protocol: ChemotherapyProtocol | None = None,
        radiation_plan: RadiationPlan | None = None,
        immuno_regimen: ImmunotherapyRegimen | None = None,
        sequence: Sequence[TreatmentModality] | None = None,
        seed: int | None = None,
    ) -> TreatmentOutcome:
        """Simulate sequential combination therapy with synergistic interaction effects."""
        if sequence is None:
            sequence = []
            if chemo_protocol is not None:
                sequence.append(TreatmentModality.CHEMOTHERAPY)
            if radiation_plan is not None:
                sequence.append(TreatmentModality.RADIATION)
            if immuno_regimen is not None:
                sequence.append(TreatmentModality.IMMUNOTHERAPY)
            if not sequence:
                sequence = [TreatmentModality.CHEMOTHERAPY]
                chemo_protocol = ChemotherapyProtocol()

        volume = self.initial_volume_cm3
        combined_trajectory = [volume]
        total_duration = 0
        modality_results: dict[str, Any] = {}
        max_toxicity_grade = 0

        for step_idx, modality in enumerate(sequence):
            step_seed = (seed + step_idx * 1000) if seed is not None else None

            # Create a temporary simulator with current volume
            step_sim = TreatmentSimulator(
                initial_volume_cm3=volume,
                chemo_sensitivity=self.chemo_sensitivity,
                radio_sensitivity=self.radio_sensitivity,
                immuno_sensitivity=self.immuno_sensitivity,
                alpha_beta_ratio=self.alpha_beta_ratio,
                growth_rate_per_day=self.growth_rate_per_day,
            )

            if modality == TreatmentModality.CHEMOTHERAPY:
                result = step_sim.simulate_chemotherapy(chemo_protocol, seed=step_seed)
            elif modality == TreatmentModality.RADIATION:
                result = step_sim.simulate_radiation(radiation_plan, seed=step_seed)
            elif modality == TreatmentModality.IMMUNOTHERAPY:
                result = step_sim.simulate_immunotherapy(immuno_regimen, seed=step_seed)
            else:
                logger.warning("Unsupported modality in combination: %s", modality.value)
                continue

            # Apply synergy bonus for combination (5% additional kill per prior modality)
            synergy_factor = 1.0 + (0.05 * step_idx)
            synergy_volume = result.final_volume_cm3 / synergy_factor
            synergy_volume = float(np.clip(synergy_volume, MIN_VOLUME_CM3, MAX_VOLUME_CM3))

            volume = synergy_volume
            combined_trajectory.extend(result.volume_trajectory[1:])
            total_duration += result.treatment_duration_days
            modality_results[modality.value] = {
                "response": result.response_category.value,
                "volume_after": volume,
                "reduction_pct": result.volume_reduction_pct,
            }

            if result.toxicity_profile is not None:
                max_toxicity_grade = max(max_toxicity_grade, result.toxicity_profile.overall_grade.value)

        overall_reduction = (self.initial_volume_cm3 - volume) / max(self.initial_volume_cm3, MIN_VOLUME_CM3)
        overall_response = _classify_response(overall_reduction)

        ci_low, ci_high = self._compute_confidence_interval(volume, overall_reduction, 0.12, seed=seed)

        outcome = TreatmentOutcome(
            modality=TreatmentModality.COMBINATION,
            response_category=overall_response,
            initial_volume_cm3=self.initial_volume_cm3,
            final_volume_cm3=volume,
            volume_reduction_pct=overall_reduction * 100.0,
            volume_trajectory=combined_trajectory,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            treatment_duration_days=total_duration,
            metadata={"modality_results": modality_results, "sequence": [m.value for m in sequence]},
        )

        self._simulation_count += 1
        self._history.append(outcome)
        logger.info(
            "Combination simulation complete: %s (%.1f%% reduction, %d modalities)",
            overall_response.value,
            overall_reduction * 100.0,
            len(sequence),
        )
        return outcome

    def _compute_confidence_interval(
        self,
        final_volume: float,
        reduction: float,
        uncertainty_std: float,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Compute 95% confidence interval for volume prediction."""
        margin = final_volume * uncertainty_std * 1.96
        ci_low = float(np.clip(final_volume - margin, MIN_VOLUME_CM3, MAX_VOLUME_CM3))
        ci_high = float(np.clip(final_volume + margin, MIN_VOLUME_CM3, MAX_VOLUME_CM3))
        return ci_low, ci_high

    def get_simulation_history(self) -> list[TreatmentOutcome]:
        """Return the complete simulation history."""
        return list(self._history)

    @property
    def simulation_count(self) -> int:
        """Return the total number of simulations performed."""
        return self._simulation_count
