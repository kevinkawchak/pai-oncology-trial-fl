"""Patient digital twin for oncology treatment simulation.

A digital twin is a patient-specific computational model that mirrors
a real patient's tumor characteristics and can simulate treatment
responses — enabling safer planning and federated model validation
without sharing raw patient data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TumorModel:
    """Simplified tumor growth and treatment-response model.

    Attributes:
        volume_cm3: Current tumor volume in cubic centimeters.
        growth_rate: Doubling-time parameter (days).
        chemo_sensitivity: Sensitivity to chemotherapy [0, 1].
        radio_sensitivity: Sensitivity to radiation [0, 1].
        location: Anatomical region identifier.
        histology: Tumor histology type.
    """

    volume_cm3: float = 2.0
    growth_rate: float = 60.0
    chemo_sensitivity: float = 0.5
    radio_sensitivity: float = 0.6
    location: str = "lung"
    histology: str = "adenocarcinoma"

    def simulate_growth(self, days: int) -> float:
        """Simulate untreated tumor growth over a number of days."""
        return self.volume_cm3 * (2.0 ** (days / self.growth_rate))

    def simulate_chemo_response(self, dose_mg: float, cycles: int) -> float:
        """Estimate tumor volume after chemotherapy cycles."""
        volume = self.volume_cm3
        for _ in range(cycles):
            kill_fraction = self.chemo_sensitivity * min(dose_mg / 100.0, 1.0)
            volume *= 1.0 - kill_fraction
            # Regrowth between cycles (21-day cycle)
            volume *= 2.0 ** (21.0 / self.growth_rate)
        return max(volume, 0.01)

    def simulate_radiation_response(self, dose_gy: float, fractions: int) -> float:
        """Estimate tumor volume after radiation therapy using the linear-quadratic model."""
        alpha = 0.3 * self.radio_sensitivity
        beta = 0.03 * self.radio_sensitivity
        dose_per_fraction = dose_gy / max(fractions, 1)
        surviving_fraction = np.exp(
            -fractions * (alpha * dose_per_fraction + beta * dose_per_fraction**2)
        )
        return float(self.volume_cm3 * surviving_fraction)


class PatientDigitalTwin:
    """Patient-specific digital twin for treatment simulation.

    Integrates tumor modelling, treatment history tracking, and
    federated-model-informed predictions without exposing PHI.

    Args:
        patient_id: De-identified patient identifier.
        tumor: Tumor model parameters.
        age: Patient age (de-identified age bracket is acceptable).
        biomarkers: Dictionary of biomarker name -> value.
    """

    def __init__(
        self,
        patient_id: str,
        tumor: TumorModel | None = None,
        age: int = 60,
        biomarkers: dict[str, float] | None = None,
    ):
        self.patient_id = patient_id
        self.tumor = tumor or TumorModel()
        self.age = age
        self.biomarkers = biomarkers or {}
        self.treatment_history: list[dict] = []
        self._simulation_log: list[dict] = []

    def simulate_treatment(
        self,
        treatment_type: str,
        **kwargs,
    ) -> dict[str, float]:
        """Simulate a treatment plan and return predicted outcomes.

        Args:
            treatment_type: One of ``"chemotherapy"``, ``"radiation"``,
                ``"immunotherapy"``.
            **kwargs: Treatment-specific parameters
                (e.g. ``dose_mg``, ``cycles``, ``dose_gy``, ``fractions``).

        Returns:
            Dictionary with predicted post-treatment volume and
            response classification.
        """
        initial_volume = self.tumor.volume_cm3

        if treatment_type == "chemotherapy":
            final_volume = self.tumor.simulate_chemo_response(
                dose_mg=kwargs.get("dose_mg", 75.0),
                cycles=kwargs.get("cycles", 4),
            )
        elif treatment_type == "radiation":
            final_volume = self.tumor.simulate_radiation_response(
                dose_gy=kwargs.get("dose_gy", 60.0),
                fractions=kwargs.get("fractions", 30),
            )
        elif treatment_type == "immunotherapy":
            response_rate = kwargs.get("response_rate", 0.3)
            rng = np.random.default_rng(hash(self.patient_id) % (2**31))
            responds = rng.random() < response_rate
            final_volume = initial_volume * (0.3 if responds else 1.1)
        else:
            raise ValueError(f"Unknown treatment type: {treatment_type}")

        reduction = (initial_volume - final_volume) / initial_volume
        if reduction > 0.65:
            response = "complete_response"
        elif reduction > 0.30:
            response = "partial_response"
        elif reduction > -0.20:
            response = "stable_disease"
        else:
            response = "progressive_disease"

        result = {
            "initial_volume_cm3": initial_volume,
            "predicted_volume_cm3": final_volume,
            "volume_reduction_pct": reduction * 100,
            "response_category": response,
        }

        self.treatment_history.append({"type": treatment_type, "params": kwargs, "result": result})
        self._simulation_log.append(result)

        logger.info(
            "Patient %s — %s simulation: %.1f%% reduction (%s)",
            self.patient_id,
            treatment_type,
            reduction * 100,
            response,
        )
        return result

    def generate_feature_vector(self) -> np.ndarray:
        """Generate a numeric feature vector for federated model input.

        Combines tumor parameters, age, and biomarkers into a fixed-size
        array suitable for the federated MLP model. This vector never
        contains PHI.
        """
        features = [
            self.tumor.volume_cm3,
            self.tumor.growth_rate,
            self.tumor.chemo_sensitivity,
            self.tumor.radio_sensitivity,
            float(self.age),
        ]
        # Pad biomarkers to fixed size
        biomarker_values = list(self.biomarkers.values())[:25]
        features.extend(biomarker_values)
        features.extend([0.0] * (30 - len(features)))
        return np.array(features[:30], dtype=np.float64)

    def get_summary(self) -> dict:
        """Return a de-identified summary of the digital twin state."""
        return {
            "patient_id": self.patient_id,
            "tumor_volume_cm3": self.tumor.volume_cm3,
            "tumor_location": self.tumor.location,
            "tumor_histology": self.tumor.histology,
            "num_treatments_simulated": len(self.treatment_history),
            "biomarkers": self.biomarkers,
        }
