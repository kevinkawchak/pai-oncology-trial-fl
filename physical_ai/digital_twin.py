"""Patient digital twin for oncology treatment simulation.

A digital twin is a patient-specific computational model that mirrors
a real patient's tumor characteristics and can simulate treatment
responses — enabling safer planning and federated model validation
without sharing raw patient data.

v0.2.0 enhancements:
- Multiple tumor growth models (exponential, logistic, Gompertz).
- Combination therapy simulation (chemo + radiation, chemo + immuno).
- Uncertainty quantification via Monte-Carlo sampling.
- Treatment scheduling with multi-cycle response tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---- Growth Models -------------------------------------------------------


class GrowthModel:
    """Pluggable tumor growth model interface."""

    @staticmethod
    def exponential(volume: float, growth_rate: float, days: int) -> float:
        """Exponential (doubling-time) growth."""
        return volume * (2.0 ** (days / max(growth_rate, 1.0)))

    @staticmethod
    def logistic(
        volume: float, growth_rate: float, days: int, carrying_capacity: float = 100.0
    ) -> float:
        """Logistic growth with carrying capacity (cm3)."""
        k = np.log(2) / max(growth_rate, 1.0)
        ratio = volume / carrying_capacity
        new_vol = carrying_capacity / (1.0 + ((1.0 / ratio) - 1.0) * np.exp(-k * days))
        return float(new_vol)

    @staticmethod
    def gompertz(volume: float, growth_rate: float, days: int, plateau: float = 80.0) -> float:
        """Gompertz growth — rapid early growth that decelerates."""
        alpha = np.log(2) / max(growth_rate, 1.0)
        beta = np.log(plateau / max(volume, 0.01))
        new_vol = plateau * np.exp(-beta * np.exp(-alpha * days))
        return float(new_vol)


# ---- Tumor Model ----------------------------------------------------------


@dataclass
class TumorModel:
    """Tumor growth and treatment-response model.

    Attributes:
        volume_cm3: Current tumor volume in cubic centimeters.
        growth_rate: Doubling-time parameter (days).
        chemo_sensitivity: Sensitivity to chemotherapy [0, 1].
        radio_sensitivity: Sensitivity to radiation [0, 1].
        immuno_sensitivity: Sensitivity to immunotherapy [0, 1].
        location: Anatomical region identifier.
        histology: Tumor histology type.
        growth_model: ``"exponential"``, ``"logistic"``, or
            ``"gompertz"``.
        carrying_capacity: Maximum volume for logistic/Gompertz (cm3).
    """

    volume_cm3: float = 2.0
    growth_rate: float = 60.0
    chemo_sensitivity: float = 0.5
    radio_sensitivity: float = 0.6
    immuno_sensitivity: float = 0.3
    location: str = "lung"
    histology: str = "adenocarcinoma"
    growth_model: str = "exponential"
    carrying_capacity: float = 100.0

    def simulate_growth(self, days: int) -> float:
        """Simulate untreated tumor growth over a number of days."""
        if self.growth_model == "logistic":
            return GrowthModel.logistic(
                self.volume_cm3, self.growth_rate, days, self.carrying_capacity
            )
        if self.growth_model == "gompertz":
            return GrowthModel.gompertz(
                self.volume_cm3, self.growth_rate, days, self.carrying_capacity
            )
        return GrowthModel.exponential(self.volume_cm3, self.growth_rate, days)

    def simulate_chemo_response(self, dose_mg: float, cycles: int) -> float:
        """Estimate tumor volume after chemotherapy cycles."""
        volume = self.volume_cm3
        for _ in range(cycles):
            kill_fraction = self.chemo_sensitivity * min(dose_mg / 100.0, 1.0)
            volume *= 1.0 - kill_fraction
            # Regrowth between cycles (21-day cycle)
            volume = self._regrow(volume, 21)
        return max(volume, 0.01)

    def simulate_radiation_response(self, dose_gy: float, fractions: int) -> float:
        """Estimate tumor volume after radiation (linear-quadratic model)."""
        alpha = 0.3 * self.radio_sensitivity
        beta = 0.03 * self.radio_sensitivity
        dose_per_fraction = dose_gy / max(fractions, 1)
        surviving_fraction = np.exp(
            -fractions * (alpha * dose_per_fraction + beta * dose_per_fraction**2)
        )
        return float(max(self.volume_cm3 * surviving_fraction, 0.01))

    def simulate_immunotherapy_response(
        self,
        response_rate: float = 0.3,
        duration_weeks: int = 12,
        seed: int | None = None,
    ) -> float:
        """Simulate immunotherapy with stochastic response.

        Immunotherapy response depends on ``immuno_sensitivity`` and
        the population-level response rate.
        """
        rng = np.random.default_rng(seed)
        effective_rate = response_rate * (0.5 + self.immuno_sensitivity)
        effective_rate = min(effective_rate, 1.0)
        responds = rng.random() < effective_rate

        if responds:
            # Exponential decay during treatment
            decay_rate = 0.05 * self.immuno_sensitivity
            volume = self.volume_cm3 * np.exp(-decay_rate * duration_weeks)
        else:
            # Progressive disease: mild growth
            volume = self._regrow(self.volume_cm3, duration_weeks * 7)

        return float(max(volume, 0.01))

    def simulate_combination_therapy(
        self,
        chemo_dose_mg: float = 75.0,
        chemo_cycles: int = 4,
        radiation_dose_gy: float = 0.0,
        radiation_fractions: int = 0,
        immunotherapy: bool = False,
        immuno_response_rate: float = 0.3,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Simulate a combination treatment plan.

        Applies therapies sequentially: chemotherapy first, then
        radiation, then immunotherapy (if enabled).

        Returns:
            Dictionary with per-modality and final volumes.
        """
        results: dict[str, float] = {"initial_volume_cm3": self.volume_cm3}
        volume = self.volume_cm3

        # Chemotherapy
        if chemo_cycles > 0:
            chemo_vol = self.simulate_chemo_response(chemo_dose_mg, chemo_cycles)
            results["post_chemo_cm3"] = chemo_vol
            volume = chemo_vol

        # Radiation
        if radiation_dose_gy > 0 and radiation_fractions > 0:
            # Apply radiation to the post-chemo volume
            saved = self.volume_cm3
            self.volume_cm3 = volume
            rad_vol = self.simulate_radiation_response(radiation_dose_gy, radiation_fractions)
            self.volume_cm3 = saved
            results["post_radiation_cm3"] = rad_vol
            volume = rad_vol

        # Immunotherapy
        if immunotherapy:
            saved = self.volume_cm3
            self.volume_cm3 = volume
            immuno_vol = self.simulate_immunotherapy_response(immuno_response_rate, seed=seed)
            self.volume_cm3 = saved
            results["post_immunotherapy_cm3"] = immuno_vol
            volume = immuno_vol

        results["final_volume_cm3"] = volume
        reduction = (self.volume_cm3 - volume) / max(self.volume_cm3, 0.01)
        results["total_reduction_pct"] = reduction * 100
        return results

    def _regrow(self, volume: float, days: int) -> float:
        """Simulate regrowth using the configured growth model."""
        saved = self.volume_cm3
        self.volume_cm3 = volume
        grown = self.simulate_growth(days)
        self.volume_cm3 = saved
        return grown


# ---- Patient Digital Twin -------------------------------------------------


class PatientDigitalTwin:
    """Patient-specific digital twin for treatment simulation.

    Integrates tumor modelling, treatment history tracking, and
    federated-model-informed predictions without exposing PHI.

    Args:
        patient_id: De-identified patient identifier.
        tumor: Tumor model parameters.
        age: Patient age (de-identified bracket acceptable).
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

    # ------------------------------------------------------------------
    # Single-modality simulation
    # ------------------------------------------------------------------

    def simulate_treatment(
        self,
        treatment_type: str,
        **kwargs,
    ) -> dict[str, float]:
        """Simulate a treatment plan and return predicted outcomes.

        Args:
            treatment_type: One of ``"chemotherapy"``, ``"radiation"``,
                ``"immunotherapy"``, ``"combination"``.
            **kwargs: Treatment-specific parameters.

        Returns:
            Dictionary with predicted volumes and response category.
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
            final_volume = self.tumor.simulate_immunotherapy_response(
                response_rate=kwargs.get("response_rate", 0.3),
                duration_weeks=kwargs.get("duration_weeks", 12),
                seed=kwargs.get("seed"),
            )
        elif treatment_type == "combination":
            combo = self.tumor.simulate_combination_therapy(**kwargs)
            final_volume = combo["final_volume_cm3"]
        else:
            raise ValueError(f"Unknown treatment type: {treatment_type}")

        reduction = (initial_volume - final_volume) / max(initial_volume, 0.01)
        response = _classify_response(reduction)

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

    # ------------------------------------------------------------------
    # Uncertainty quantification
    # ------------------------------------------------------------------

    def simulate_with_uncertainty(
        self,
        treatment_type: str,
        n_samples: int = 100,
        param_noise_std: float = 0.1,
        seed: int = 42,
        **kwargs,
    ) -> dict[str, float]:
        """Monte-Carlo simulation with parameter uncertainty.

        Perturbs tumor sensitivity parameters to estimate the
        distribution of treatment outcomes.

        Args:
            treatment_type: Treatment modality.
            n_samples: Number of MC samples.
            param_noise_std: Standard deviation of parameter noise.
            seed: Random seed.
            **kwargs: Treatment parameters.

        Returns:
            Dictionary with mean, std, p5, p95 of final volume and
            reduction.
        """
        rng = np.random.default_rng(seed)
        volumes: list[float] = []

        original_chemo = self.tumor.chemo_sensitivity
        original_radio = self.tumor.radio_sensitivity
        original_immuno = self.tumor.immuno_sensitivity

        for i in range(n_samples):
            # Perturb sensitivities
            self.tumor.chemo_sensitivity = float(
                np.clip(original_chemo + rng.normal(0, param_noise_std), 0.0, 1.0)
            )
            self.tumor.radio_sensitivity = float(
                np.clip(original_radio + rng.normal(0, param_noise_std), 0.0, 1.0)
            )
            self.tumor.immuno_sensitivity = float(
                np.clip(original_immuno + rng.normal(0, param_noise_std), 0.0, 1.0)
            )

            result = self.simulate_treatment(treatment_type, seed=seed + i, **kwargs)
            volumes.append(result["predicted_volume_cm3"])

        # Restore originals
        self.tumor.chemo_sensitivity = original_chemo
        self.tumor.radio_sensitivity = original_radio
        self.tumor.immuno_sensitivity = original_immuno

        arr = np.array(volumes)
        initial = self.tumor.volume_cm3
        reductions = (initial - arr) / max(initial, 0.01) * 100

        return {
            "mean_volume_cm3": float(np.mean(arr)),
            "std_volume_cm3": float(np.std(arr)),
            "p5_volume_cm3": float(np.percentile(arr, 5)),
            "p95_volume_cm3": float(np.percentile(arr, 95)),
            "mean_reduction_pct": float(np.mean(reductions)),
            "std_reduction_pct": float(np.std(reductions)),
            "n_samples": n_samples,
        }

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def generate_feature_vector(self) -> np.ndarray:
        """Generate a numeric feature vector for federated model input.

        Combines tumor parameters, age, and biomarkers into a fixed-size
        array suitable for the federated MLP model.  This vector never
        contains PHI.
        """
        features = [
            self.tumor.volume_cm3,
            self.tumor.growth_rate,
            self.tumor.chemo_sensitivity,
            self.tumor.radio_sensitivity,
            float(self.age),
        ]
        biomarker_values = list(self.biomarkers.values())[:25]
        features.extend(biomarker_values)
        features.extend([0.0] * (30 - len(features)))
        return np.array(features[:30], dtype=np.float64)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a de-identified summary of the digital twin state."""
        return {
            "patient_id": self.patient_id,
            "tumor_volume_cm3": self.tumor.volume_cm3,
            "tumor_location": self.tumor.location,
            "tumor_histology": self.tumor.histology,
            "growth_model": self.tumor.growth_model,
            "num_treatments_simulated": len(self.treatment_history),
            "biomarkers": self.biomarkers,
        }


# ---- Helpers --------------------------------------------------------------


def _classify_response(reduction: float) -> str:
    """Classify treatment response using RECIST-like criteria."""
    if reduction > 0.65:
        return "complete_response"
    if reduction > 0.30:
        return "partial_response"
    if reduction > -0.20:
        return "stable_disease"
    return "progressive_disease"
