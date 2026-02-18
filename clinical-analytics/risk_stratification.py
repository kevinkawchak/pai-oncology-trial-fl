"""Clinical risk stratification — federated risk scoring for oncology trials.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Implements patient-level and cohort-level risk stratification for multi-site
oncology clinical trials: weighted prognostic scoring bounded to [0, 1],
Hosmer-Lemeshow calibration, concordance (C-statistic) validation, decision-
support mapping to treatment arms, federated aggregation of site-level risk
models, and adaptive enrichment with conditional-power futility analysis.
Patient identifiers are hashed with HMAC-SHA256 before storage.  Only
stdlib, numpy, and scipy are required.

DISCLAIMER: RESEARCH USE ONLY — Not for individual treatment decisions
without independent clinical validation and regulatory approval.

LICENSE: MIT
VERSION: 0.9.0
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_HMAC_KEY = b"pai-oncology-risk-stratification-key"
_DEFAULT_SEED = 42
_RISK_SCORE_MIN = 0.0
_RISK_SCORE_MAX = 1.0
_CALIBRATION_BINS = 10
_MIN_COHORT_SIZE = 5

# ===================================================================== #
#  Enums                                                                  #
# ===================================================================== #

class RiskCategory(str, Enum):
    """Discrete risk strata for oncology trial participants."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class PrognosticFactor(str, Enum):
    """Clinically recognised prognostic factors in oncology."""
    TUMOR_STAGE = "tumor_stage"
    HISTOLOGY = "histology"
    BIOMARKER_STATUS = "biomarker_status"
    PERFORMANCE_STATUS = "performance_status"
    AGE = "age"
    COMORBIDITY = "comorbidity"
    GENOMIC_SIGNATURE = "genomic_signature"

class DecisionConfidence(str, Enum):
    """Confidence level attached to a decision-support recommendation."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT_DATA = "insufficient_data"

class TreatmentArm(str, Enum):
    """Treatment arms available in the trial."""
    STANDARD = "standard"
    EXPERIMENTAL = "experimental"
    COMBINATION = "combination"
    BEST_SUPPORTIVE_CARE = "best_supportive_care"

# ===================================================================== #
#  Dataclasses                                                            #
# ===================================================================== #

@dataclass
class PatientRiskProfile:
    """Risk assessment for a single patient (hashed ID, score in [0,1])."""
    patient_id: str
    risk_category: RiskCategory
    risk_score: float
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    confidence: DecisionConfidence = DecisionConfidence.MODERATE
    assessment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class CohortRiskSummary:
    """Aggregate risk summary for a patient cohort."""
    cohort_size: int = 0
    distribution: Dict[str, int] = field(default_factory=dict)
    median_risk: float = 0.0
    iqr: Tuple[float, float] = (0.0, 0.0)
    mean_risk: float = 0.0
    std_risk: float = 0.0
    category_proportions: Dict[str, float] = field(default_factory=dict)

@dataclass
class StratificationCriteria:
    """Factors, weights, and thresholds for patient stratification."""
    factors: List[PrognosticFactor] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    thresholds: List[float] = field(default_factory=lambda: [0.15, 0.35, 0.55, 0.75])

@dataclass
class RiskModelConfig:
    """Configuration for the underlying risk model."""
    model_type: str = "weighted_linear"
    features: List[str] = field(default_factory=list)
    calibration_bins: int = _CALIBRATION_BINS
    regularisation_lambda: float = 0.01
    seed: int = _DEFAULT_SEED

@dataclass
class DecisionSupportOutput:
    """Output of the decision-support engine."""
    recommended_arm: TreatmentArm
    confidence: DecisionConfidence
    supporting_evidence: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    alternative_arms: List[Dict[str, Any]] = field(default_factory=list)
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ===================================================================== #
#  Helpers                                                                #
# ===================================================================== #

def _audit_log(action: str, details: Dict[str, Any]) -> None:
    """Emit a structured audit-trail entry."""
    entry = {
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }
    logger.info("AUDIT | %s | %s", action, entry)

def _hash_patient_id(raw_id: str) -> str:
    """Return an HMAC-SHA256 hex digest of *raw_id*."""
    return hmac.new(
        _HMAC_KEY, raw_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()

# ===================================================================== #
#  ClinicalRiskStratifier                                                 #
# ===================================================================== #

class ClinicalRiskStratifier:
    """Federated clinical risk stratification engine.

    Computes risk scores from prognostic feature vectors, assigns patients
    to discrete risk categories, produces cohort-level summaries, and
    generates decision-support recommendations.
    """

    def __init__(
        self,
        criteria: Optional[StratificationCriteria] = None,
        config: Optional[RiskModelConfig] = None,
    ) -> None:
        self.criteria = criteria or self._default_criteria()
        self.config = config or RiskModelConfig()
        self._calibration_slope: float = 1.0
        self._calibration_intercept: float = 0.0
        self._rng = np.random.default_rng(self.config.seed)
        logger.info(
            "ClinicalRiskStratifier initialised — model_type=%s, %d factors",
            self.config.model_type, len(self.criteria.factors),
        )

    @staticmethod
    def _default_criteria() -> StratificationCriteria:
        """Return sensible defaults when no criteria are supplied."""
        return StratificationCriteria(
            factors=list(PrognosticFactor),
            weights={
                PrognosticFactor.TUMOR_STAGE.value: 0.25,
                PrognosticFactor.HISTOLOGY.value: 0.10,
                PrognosticFactor.BIOMARKER_STATUS.value: 0.15,
                PrognosticFactor.PERFORMANCE_STATUS.value: 0.15,
                PrognosticFactor.AGE.value: 0.10,
                PrognosticFactor.COMORBIDITY.value: 0.10,
                PrognosticFactor.GENOMIC_SIGNATURE.value: 0.15,
            },
            thresholds=[0.15, 0.35, 0.55, 0.75],
        )

    # -- Risk scoring -------------------------------------------------- #

    def compute_risk_score(self, features: Dict[str, float]) -> float:
        """Weighted risk score from normalised [0,1] features, clamped output."""
        weights = self.criteria.weights
        score = 0.0
        total_weight = 0.0
        for factor in self.criteria.factors:
            key = factor.value
            if key in features:
                w = weights.get(key, 0.0)
                score += w * float(features[key])
                total_weight += w
        # Guard against division by zero
        score = (score / total_weight) if total_weight > 0.0 else 0.5
        # Apply calibration
        score = self._calibration_slope * score + self._calibration_intercept
        return float(np.clip(score, _RISK_SCORE_MIN, _RISK_SCORE_MAX))

    def _score_to_category(self, score: float) -> RiskCategory:
        """Map a continuous score to a discrete risk category."""
        t = sorted(self.criteria.thresholds)
        if score < t[0]:
            return RiskCategory.VERY_LOW
        if score < t[1]:
            return RiskCategory.LOW
        if score < t[2]:
            return RiskCategory.MODERATE
        if score < t[3]:
            return RiskCategory.HIGH
        return RiskCategory.VERY_HIGH

    def _assess_confidence(self, features: Dict[str, float]) -> DecisionConfidence:
        """Determine confidence based on feature completeness."""
        total = len(self.criteria.factors)
        if total == 0:
            return DecisionConfidence.INSUFFICIENT_DATA
        ratio = sum(1 for f in self.criteria.factors if f.value in features) / total
        if ratio >= 0.85:
            return DecisionConfidence.STRONG
        if ratio >= 0.60:
            return DecisionConfidence.MODERATE
        if ratio >= 0.35:
            return DecisionConfidence.WEAK
        return DecisionConfidence.INSUFFICIENT_DATA

    def _compute_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Return the weighted contribution of each available factor."""
        w = self.criteria.weights
        return {
            f.value: w.get(f.value, 0.0) * float(features[f.value])
            for f in self.criteria.factors
            if f.value in features
        }

    # -- Patient stratification ---------------------------------------- #

    def stratify_patient(
        self,
        features: Dict[str, float],
        raw_patient_id: Optional[str] = None,
    ) -> PatientRiskProfile:
        """Stratify a single patient into a risk category.

        Args:
            features: Prognostic feature vector (normalised values).
            raw_patient_id: Unhashed identifier — hashed before storage.
        """
        hashed_id = _hash_patient_id(raw_patient_id or uuid.uuid4().hex)
        score = self.compute_risk_score(features)
        profile = PatientRiskProfile(
            patient_id=hashed_id,
            risk_category=self._score_to_category(score),
            risk_score=score,
            contributing_factors=self._compute_contributions(features),
            confidence=self._assess_confidence(features),
        )
        _audit_log("stratify_patient", {
            "patient_id": hashed_id,
            "risk_category": profile.risk_category.value,
            "risk_score": round(score, 4),
        })
        return profile

    # -- Cohort stratification ----------------------------------------- #

    def stratify_cohort(
        self,
        patients: Sequence[Dict[str, float]],
        patient_ids: Optional[Sequence[str]] = None,
    ) -> CohortRiskSummary:
        """Stratify a cohort and return an aggregate summary."""
        n = len(patients)
        if n == 0:
            logger.warning("stratify_cohort called with empty patient list")
            return CohortRiskSummary()
        profiles = [
            self.stratify_patient(
                feat,
                raw_patient_id=(patient_ids[i] if patient_ids and i < len(patient_ids) else None),
            )
            for i, feat in enumerate(patients)
        ]
        scores = np.array([p.risk_score for p in profiles])
        distribution: Dict[str, int] = {cat.value: 0 for cat in RiskCategory}
        for p in profiles:
            distribution[p.risk_category.value] += 1
        proportions = {k: v / n for k, v in distribution.items()}
        q1, q3 = float(np.percentile(scores, 25)), float(np.percentile(scores, 75))
        summary = CohortRiskSummary(
            cohort_size=n,
            distribution=distribution,
            median_risk=float(np.median(scores)),
            iqr=(q1, q3),
            mean_risk=float(np.mean(scores)),
            std_risk=float(np.std(scores, ddof=1)) if n > 1 else 0.0,
            category_proportions=proportions,
        )
        _audit_log("stratify_cohort", {
            "cohort_size": n, "median_risk": round(summary.median_risk, 4),
        })
        return summary

    # -- Calibration (Hosmer-Lemeshow) --------------------------------- #

    def calibrate_model(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, Any]:
        """Hosmer-Lemeshow calibration with Platt-style linear recalibration."""
        observed = np.asarray(observed, dtype=np.float64).ravel()
        predicted = np.asarray(predicted, dtype=np.float64).ravel()
        if observed.shape[0] != predicted.shape[0]:
            raise ValueError(
                f"Length mismatch: observed={observed.shape[0]}, "
                f"predicted={predicted.shape[0]}"
            )
        n = observed.shape[0]
        n_bins = min(self.config.calibration_bins, n)
        order = np.argsort(predicted)
        sorted_obs, sorted_pred = observed[order], predicted[order]
        bin_edges = np.array_split(np.arange(n), n_bins)

        hl_stat = 0.0
        bins_detail: List[Dict[str, float]] = []
        for indices in bin_edges:
            if len(indices) == 0:
                continue
            n_g = float(len(indices))
            obs_rate = float(np.mean(sorted_obs[indices]))
            pred_rate = float(np.mean(sorted_pred[indices]))
            exp_ev = pred_rate * n_g
            exp_non = (1.0 - pred_rate) * n_g
            obs_ev = float(np.sum(sorted_obs[indices]))
            obs_non = n_g - obs_ev
            if exp_ev > 0:
                hl_stat += (obs_ev - exp_ev) ** 2 / exp_ev
            if exp_non > 0:
                hl_stat += (obs_non - exp_non) ** 2 / exp_non
            bins_detail.append({
                "n": n_g,
                "observed_rate": round(obs_rate, 4),
                "predicted_rate": round(pred_rate, 4),
            })

        dof = max(n_bins - 2, 1)
        p_value = float(1.0 - sp_stats.chi2.cdf(hl_stat, dof))

        # Platt scaling (linear recalibration via least-squares)
        slope, intercept = self._fit_platt_scaling(observed, predicted)
        self._calibration_slope = slope
        self._calibration_intercept = intercept

        result = {
            "hl_statistic": round(hl_stat, 4),
            "p_value": round(p_value, 4),
            "bins": bins_detail,
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "n_bins": n_bins,
            "dof": dof,
        }
        _audit_log("calibrate_model", {
            "hl_statistic": result["hl_statistic"],
            "p_value": result["p_value"],
            "n_samples": n,
        })
        logger.info(
            "Calibration — HL=%.4f, p=%.4f, slope=%.4f, intercept=%.4f",
            hl_stat, p_value, slope, intercept,
        )
        return result

    @staticmethod
    def _fit_platt_scaling(
        observed: np.ndarray, predicted: np.ndarray
    ) -> Tuple[float, float]:
        """Fit linear recalibration via least-squares normal equations."""
        if len(predicted) < 2:
            return 1.0, 0.0
        x_mean = float(np.mean(predicted))
        y_mean = float(np.mean(observed))
        ss_xx = float(np.sum((predicted - x_mean) ** 2))
        if ss_xx < 1e-12:
            return 1.0, 0.0
        ss_xy = float(np.sum((predicted - x_mean) * (observed - y_mean)))
        slope = float(np.clip(ss_xy / ss_xx, 0.1, 10.0))
        intercept = float(np.clip(y_mean - slope * x_mean, -0.5, 0.5))
        return slope, intercept

    # -- Decision support ---------------------------------------------- #

    def generate_decision_support(
        self, profile: PatientRiskProfile
    ) -> DecisionSupportOutput:
        """Generate a treatment-arm recommendation based on risk profile."""
        evidence: List[str] = []
        contraindications: List[str] = []
        alternatives: List[Dict[str, Any]] = []
        cat, score = profile.risk_category, profile.risk_score
        factors = profile.contributing_factors

        # Primary arm selection
        if cat in (RiskCategory.VERY_LOW, RiskCategory.LOW):
            arm = TreatmentArm.STANDARD
            evidence.append(f"Low risk score ({score:.2f}) supports standard therapy.")
            alternatives.append({"arm": TreatmentArm.EXPERIMENTAL.value,
                                 "rationale": "Consider if biomarker-positive."})
        elif cat == RiskCategory.MODERATE:
            arm = TreatmentArm.EXPERIMENTAL
            evidence.append(f"Moderate risk ({score:.2f}) suggests experimental benefit.")
            alternatives.append({"arm": TreatmentArm.COMBINATION.value,
                                 "rationale": "Combination may add efficacy."})
            alternatives.append({"arm": TreatmentArm.STANDARD.value,
                                 "rationale": "Standard remains a safe alternative."})
        elif cat == RiskCategory.HIGH:
            arm = TreatmentArm.COMBINATION
            evidence.append(f"High risk ({score:.2f}) warrants combination approach.")
            alternatives.append({"arm": TreatmentArm.EXPERIMENTAL.value,
                                 "rationale": "Single agent if comorbidities limit combo."})
        else:
            arm = TreatmentArm.BEST_SUPPORTIVE_CARE
            evidence.append(f"Very high risk ({score:.2f}) — best supportive care.")
            alternatives.append({"arm": TreatmentArm.COMBINATION.value,
                                 "rationale": "Combination if performance status adequate."})

        # Contraindication checks
        comorbidity_c = factors.get(PrognosticFactor.COMORBIDITY.value, 0.0)
        perf_c = factors.get(PrognosticFactor.PERFORMANCE_STATUS.value, 0.0)
        if comorbidity_c > 0.08 and arm in (TreatmentArm.COMBINATION, TreatmentArm.EXPERIMENTAL):
            contraindications.append(
                f"Elevated comorbidity may limit {arm.value} tolerability "
                f"(contribution={comorbidity_c:.3f})."
            )
        if perf_c > 0.10:
            contraindications.append(
                f"Poor performance status may reduce benefit (contribution={perf_c:.3f})."
            )

        # Confidence
        if profile.confidence == DecisionConfidence.INSUFFICIENT_DATA:
            dec_conf = DecisionConfidence.INSUFFICIENT_DATA
        elif contraindications:
            dec_conf = DecisionConfidence.WEAK
        elif profile.confidence == DecisionConfidence.STRONG:
            dec_conf = DecisionConfidence.STRONG
        else:
            dec_conf = DecisionConfidence.MODERATE

        output = DecisionSupportOutput(
            recommended_arm=arm, confidence=dec_conf,
            supporting_evidence=evidence, contraindications=contraindications,
            alternative_arms=alternatives,
        )
        _audit_log("generate_decision_support", {
            "patient_id": profile.patient_id, "arm": arm.value,
            "confidence": dec_conf.value,
        })
        return output

    # -- Validation (C-statistic / AUC) -------------------------------- #

    def validate_stratification(
        self,
        profiles: Sequence[PatientRiskProfile],
        outcomes: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate stratification via C-statistic with Hanley-McNeil SE."""
        outcomes = np.asarray(outcomes, dtype=np.float64).ravel()
        scores = np.array([p.risk_score for p in profiles])
        if len(scores) != len(outcomes):
            raise ValueError(
                f"Profile/outcome length mismatch: {len(scores)} vs {len(outcomes)}"
            )
        event_idx = np.where(outcomes == 1)[0]
        nonevent_idx = np.where(outcomes == 0)[0]
        n_conc = n_disc = n_tied = 0
        for ei in event_idx:
            diffs = scores[ei] - scores[nonevent_idx]
            n_conc += int(np.sum(diffs > 0))
            n_disc += int(np.sum(diffs < 0))
            n_tied += int(np.sum(diffs == 0))
        n_pairs = len(event_idx) * len(nonevent_idx)
        if n_pairs == 0:
            logger.warning("No valid pairs for C-statistic")
            nan = float("nan")
            return {"c_statistic": nan, "n_concordant": 0, "n_discordant": 0,
                    "n_tied": 0, "n_pairs": 0, "se": nan, "ci_95": (nan, nan)}

        c_stat = (n_conc + 0.5 * n_tied) / n_pairs
        n_ev, n_ne = len(event_idx), len(nonevent_idx)
        q1 = c_stat / (2.0 - c_stat)
        q2 = 2.0 * c_stat ** 2 / (1.0 + c_stat)
        numerator = (c_stat * (1 - c_stat) + (n_ev - 1) * (q1 - c_stat ** 2)
                     + (n_ne - 1) * (q2 - c_stat ** 2))
        se = float(np.sqrt(max(numerator / (n_ev * n_ne), 0.0)))
        ci_lo = float(np.clip(c_stat - 1.96 * se, 0, 1))
        ci_hi = float(np.clip(c_stat + 1.96 * se, 0, 1))
        result = {
            "c_statistic": round(c_stat, 4), "n_concordant": n_conc,
            "n_discordant": n_disc, "n_tied": n_tied, "n_pairs": n_pairs,
            "se": round(se, 4), "ci_95": (round(ci_lo, 4), round(ci_hi, 4)),
        }
        _audit_log("validate_stratification", {
            "c_statistic": result["c_statistic"], "n": len(profiles),
        })
        logger.info("Validation — C=%.4f (95%% CI %.4f–%.4f)", c_stat, ci_lo, ci_hi)
        return result

    # -- Federated model aggregation ----------------------------------- #

    def aggregate_site_risk_models(
        self, site_models: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate site risk-model weights via sample-size-weighted averaging."""
        if not site_models:
            raise ValueError("Cannot aggregate: no site models provided")
        total_n = sum(m.get("n_patients", 1) for m in site_models) or len(site_models)
        aggregated: Dict[str, float] = {}
        for model in site_models:
            contrib = model.get("n_patients", 1) / total_n
            for factor, w in model.get("weights", {}).items():
                aggregated[factor] = aggregated.get(factor, 0.0) + contrib * w
        w_sum = sum(aggregated.values())
        if w_sum > 0:
            aggregated = {k: v / w_sum for k, v in aggregated.items()}
        self.criteria.weights = aggregated
        _audit_log("aggregate_site_risk_models", {
            "n_sites": len(site_models), "total_patients": total_n,
        })
        logger.info("Aggregated %d sites, %d patients", len(site_models), total_n)
        return aggregated

# ===================================================================== #
#  AdaptiveEnrichmentAdvisor                                              #
# ===================================================================== #

class AdaptiveEnrichmentAdvisor:
    """Recommends enrichment strategies and performs futility analysis.

    Given risk-stratified cohort data and interim outcomes, identifies
    subgroups for preferential enrolment and determines whether interim
    data support continuing the trial.
    """

    def __init__(
        self,
        stratifier: ClinicalRiskStratifier,
        futility_threshold: float = 0.20,
        enrichment_benefit_threshold: float = 0.10,
    ) -> None:
        self.stratifier = stratifier
        self.futility_threshold = futility_threshold
        self.enrichment_benefit_threshold = enrichment_benefit_threshold
        logger.info(
            "AdaptiveEnrichmentAdvisor — futility_threshold=%.2f", futility_threshold
        )

    def recommend_enrichment(
        self,
        cohort_summary: CohortRiskSummary,
        subgroup_outcomes: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Recommend enrichment based on per-subgroup treatment effects."""
        enrich: List[str] = []
        exclude: List[str] = []
        effects: Dict[str, float] = {}
        rationale: List[str] = []
        for cat in RiskCategory:
            if cat.value not in subgroup_outcomes:
                continue
            sub = subgroup_outcomes[cat.value]
            effect = sub.get("experimental_rate", 0.0) - sub.get("control_rate", 0.0)
            effects[cat.value] = round(effect, 4)
            prop = cohort_summary.category_proportions.get(cat.value, 0.0)
            if effect >= self.enrichment_benefit_threshold:
                enrich.append(cat.value)
                rationale.append(
                    f"{cat.value}: effect={effect:.3f} >= threshold "
                    f"({self.enrichment_benefit_threshold:.2f}), prop={prop:.2%} — ENRICH."
                )
            elif effect <= -self.enrichment_benefit_threshold:
                exclude.append(cat.value)
                rationale.append(
                    f"{cat.value}: negative effect={effect:.3f} — EXCLUDE."
                )
            else:
                rationale.append(
                    f"{cat.value}: effect={effect:.3f} below threshold — maintain."
                )
        result = {"enrich": enrich, "exclude": exclude,
                  "effects": effects, "rationale": rationale}
        _audit_log("recommend_enrichment", {
            "n_enrich": len(enrich), "n_exclude": len(exclude),
        })
        return result

    def futility_analysis(
        self,
        interim_experimental: np.ndarray,
        interim_control: np.ndarray,
        planned_total_n: int,
        alpha: float = 0.025,
    ) -> Dict[str, Any]:
        """Conditional-power futility analysis using interim Z-statistic."""
        exp = np.asarray(interim_experimental, dtype=np.float64).ravel()
        ctrl = np.asarray(interim_control, dtype=np.float64).ravel()
        n_exp, n_ctrl = len(exp), len(ctrl)
        n_interim = n_exp + n_ctrl
        if n_exp < _MIN_COHORT_SIZE or n_ctrl < _MIN_COHORT_SIZE:
            return {
                "conditional_power": float("nan"),
                "z_interim": float("nan"),
                "information_fraction": n_interim / max(planned_total_n, 1),
                "recommend_stop": False,
                "rationale": "Insufficient interim data for futility assessment.",
            }
        p_exp = float(np.mean(exp))
        p_ctrl = float(np.mean(ctrl))
        delta = p_exp - p_ctrl
        p_pool = (np.sum(exp) + np.sum(ctrl)) / (n_exp + n_ctrl)
        se_pool = np.sqrt(
            max(p_pool * (1.0 - p_pool) * (1.0 / n_exp + 1.0 / n_ctrl), 1e-12)
        )
        z_interim = delta / se_pool
        info_frac = float(np.clip(n_interim / max(planned_total_n, 1), 0.01, 0.99))
        z_alpha = float(sp_stats.norm.ppf(1.0 - alpha))

        # Conditional power under current trend
        sqrt_info = np.sqrt(info_frac)
        sqrt_rem = np.sqrt(1.0 - info_frac)
        if sqrt_rem < 1e-12:
            cp = 0.0
        else:
            cp = float(sp_stats.norm.cdf(
                (z_interim * sqrt_info - z_alpha * sqrt_rem) / sqrt_rem
            ))
        cp = float(np.clip(cp, 0.0, 1.0))
        recommend_stop = cp < self.futility_threshold
        if recommend_stop:
            rationale_text = (
                f"Conditional power ({cp:.3f}) < futility threshold "
                f"({self.futility_threshold:.2f}). Recommend stopping."
            )
        else:
            rationale_text = (
                f"Conditional power ({cp:.3f}) >= futility threshold "
                f"({self.futility_threshold:.2f}). Continue enrolment."
            )
        result = {
            "conditional_power": round(cp, 4),
            "z_interim": round(float(z_interim), 4),
            "information_fraction": round(info_frac, 4),
            "observed_delta": round(delta, 4),
            "recommend_stop": recommend_stop,
            "rationale": rationale_text,
        }
        _audit_log("futility_analysis", {
            "conditional_power": result["conditional_power"],
            "z_interim": result["z_interim"],
            "recommend_stop": recommend_stop,
        })
        logger.info(
            "Futility — CP=%.4f, z=%.4f, info=%.2f, stop=%s",
            cp, z_interim, info_frac, recommend_stop,
        )
        return result

    def subgroup_futility_sweep(
        self,
        subgroup_data: Dict[str, Dict[str, np.ndarray]],
        planned_total_n: int,
        alpha: float = 0.025,
    ) -> Dict[str, Dict[str, Any]]:
        """Run futility analysis independently for each risk subgroup."""
        results: Dict[str, Dict[str, Any]] = {}
        for cat_key, arms in subgroup_data.items():
            results[cat_key] = self.futility_analysis(
                arms.get("experimental", np.array([])),
                arms.get("control", np.array([])),
                planned_total_n, alpha=alpha,
            )
        stop_groups = [k for k, v in results.items() if v.get("recommend_stop")]
        if stop_groups:
            logger.warning("Futility in subgroups: %s", ", ".join(stop_groups))
        _audit_log("subgroup_futility_sweep", {
            "n_subgroups": len(results), "stop_recommended": stop_groups,
        })
        return results

    def generate_advisory(
        self,
        cohort_summary: CohortRiskSummary,
        subgroup_outcomes: Dict[str, Dict[str, float]],
        subgroup_data: Dict[str, Dict[str, np.ndarray]],
        planned_total_n: int,
        alpha: float = 0.025,
    ) -> Dict[str, Any]:
        """Produce a combined enrichment and futility advisory report."""
        enrichment = self.recommend_enrichment(cohort_summary, subgroup_outcomes)
        futility = self.subgroup_futility_sweep(
            subgroup_data, planned_total_n, alpha=alpha,
        )
        return {
            "enrichment": enrichment,
            "futility": futility,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": self._synthesise_advisory(enrichment, futility),
        }

    @staticmethod
    def _synthesise_advisory(
        enrichment: Dict[str, Any],
        futility: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Synthesise human-readable advisory statements."""
        stmts: List[str] = []
        enrich = enrichment.get("enrich", [])
        exclude = enrichment.get("exclude", [])
        futile = [k for k, v in futility.items() if v.get("recommend_stop")]
        if enrich:
            stmts.append(f"Enrich enrolment in: {', '.join(enrich)}.")
        if exclude:
            stmts.append(f"Consider excluding: {', '.join(exclude)}.")
        if futile:
            stmts.append(
                f"Futility crossed in: {', '.join(futile)}. Consider stopping."
            )
        if not stmts:
            stmts.append("No enrichment or futility actions recommended at this interim.")
        return stmts
