"""Survival analysis — privacy-preserving Kaplan-Meier and Cox PH for federated trials.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

This module implements core survival analysis methods for multi-site oncology
clinical trials operating under a federated learning paradigm.  All patient
identifiers are hashed with HMAC-SHA256 before any computation, ensuring that
raw identifiers never leave the local site boundary.

Supported estimators and tests:
    * Kaplan-Meier product-limit estimator with Greenwood variance and
      pointwise confidence intervals.
    * Log-rank, Wilcoxon (Gehan-Breslow), Tarone-Ware, and
      Fleming-Harrington two-sample hypothesis tests.
    * Cox proportional-hazards regression fitted via Newton-Raphson
      optimisation of the partial likelihood.
    * Restricted mean survival time (RMST) via trapezoidal integration.
    * Harrell's concordance index for model discrimination.

Federated workflow:
    Each site computes local at-risk/event tables and gradient/Hessian
    contributions.  The coordinator aggregates these summary statistics
    — never individual patient records — to produce global estimates.

DISCLAIMER: RESEARCH USE ONLY.  Not validated for clinical use.
LICENSE: MIT
VERSION: 0.9.0
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_LEVEL: float = 0.95
MAX_NEWTON_RAPHSON_ITER: int = 50
NEWTON_RAPHSON_TOL: float = 1e-9
MIN_AT_RISK: int = 1
DEFAULT_HMAC_KEY: bytes = b"pai-oncology-trial-fl-default-key"


# ===================================================================
# Enums  &  Data classes
# ===================================================================
class SurvivalEndpoint(str, Enum):
    """Clinical survival endpoint definitions."""

    OVERALL_SURVIVAL = "overall_survival"
    PROGRESSION_FREE_SURVIVAL = "progression_free_survival"
    DISEASE_FREE_SURVIVAL = "disease_free_survival"
    TIME_TO_PROGRESSION = "time_to_progression"
    DURATION_OF_RESPONSE = "duration_of_response"


class CensoringType(str, Enum):
    """Censoring mechanism."""

    RIGHT = "right"
    LEFT = "left"
    INTERVAL = "interval"


class TestType(str, Enum):
    """Two-sample survival test variants."""

    LOG_RANK = "log_rank"
    WILCOXON = "wilcoxon"
    TARONE_WARE = "tarone_ware"
    FLEMING_HARRINGTON = "fleming_harrington"


class HazardModel(str, Enum):
    """Parametric and semi-parametric hazard model families."""

    COX_PH = "cox_ph"
    WEIBULL = "weibull"
    EXPONENTIAL = "exponential"
    LOG_NORMAL = "log_normal"
    LOG_LOGISTIC = "log_logistic"


@dataclass
class SurvivalData:
    """One patient's survival record (time, event, covariates)."""

    patient_id_hash: str
    time: float
    event: int  # 1 = event observed, 0 = censored
    group: str = "control"
    covariates: Dict[str, float] = field(default_factory=dict)
    endpoint: SurvivalEndpoint = SurvivalEndpoint.OVERALL_SURVIVAL
    censoring_type: CensoringType = CensoringType.RIGHT

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"Survival time must be non-negative, got {self.time}")
        if self.event not in (0, 1):
            raise ValueError(f"Event indicator must be 0 or 1, got {self.event}")


@dataclass
class KaplanMeierEstimate:
    """Kaplan-Meier product-limit survival estimate with Greenwood CIs."""

    time_points: np.ndarray
    survival_probabilities: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL


@dataclass
class CoxPHResult:
    """Cox proportional-hazards model fit result."""

    coefficients: np.ndarray
    hazard_ratios: np.ndarray
    confidence_intervals: np.ndarray  # shape (p, 2)
    p_values: np.ndarray
    concordance_index: float
    standard_errors: np.ndarray
    covariate_names: List[str] = field(default_factory=list)
    iterations: int = 0
    log_partial_likelihood: float = 0.0


@dataclass
class LogRankResult:
    """Two-sample weighted log-rank test result."""

    chi_squared: float
    p_value: float
    degrees_of_freedom: int = 1
    test_type: TestType = TestType.LOG_RANK
    observed_a: float = 0.0
    expected_a: float = 0.0
    observed_b: float = 0.0
    expected_b: float = 0.0


@dataclass
class SiteContribution:
    """Aggregate at-risk / event table from one federated site."""

    site_id: str
    time_points: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    total_patients: int


@dataclass
class FederatedSurvivalSummary:
    """Aggregated survival analysis across federated sites."""

    site_contributions: List[SiteContribution]
    aggregated_km: KaplanMeierEstimate
    aggregated_cox: Optional[CoxPHResult] = None
    total_events: int = 0
    total_patients: int = 0
    num_sites: int = 0


# ===================================================================
# Helpers: audit trail, Greenwood CI, product-limit core
# ===================================================================
def _audit_log(action: str, details: Dict[str, Any]) -> str:
    """Emit a structured audit log entry and return its unique ID."""
    entry_id = uuid.uuid4().hex[:12]
    logger.info(
        "AUDIT [%s] action=%s | %s",
        entry_id,
        action,
        " | ".join(f"{k}={v}" for k, v in details.items()),
    )
    return entry_id


def _greenwood_ci(
    survival: np.ndarray,
    greenwood_sum: np.ndarray,
    z_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise CIs on the log(-log S) scale, back-transformed to [0,1]."""
    n = len(survival)
    ci_lo = np.zeros(n, dtype=np.float64)
    ci_hi = np.zeros(n, dtype=np.float64)
    for j in range(n):
        s_j = survival[j]
        if s_j <= 0.0 or s_j >= 1.0 or greenwood_sum[j] <= 0.0:
            ci_lo[j] = s_j
            ci_hi[j] = s_j
            continue
        log_log_s = math.log(-math.log(s_j))
        se_ll = math.sqrt(greenwood_sum[j]) / abs(math.log(s_j))
        ci_lo[j] = max(0.0, math.exp(-math.exp(log_log_s + z_alpha * se_ll)))
        ci_hi[j] = min(1.0, math.exp(-math.exp(log_log_s - z_alpha * se_ll)))
    return ci_lo, ci_hi


def _product_limit(
    at_risk: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (survival, greenwood_cumsum) from at-risk and event counts."""
    n = len(at_risk)
    survival = np.ones(n, dtype=np.float64)
    gw = np.zeros(n, dtype=np.float64)
    for j in range(n):
        n_j = max(float(at_risk[j]), 1.0)
        d_j = float(events[j])
        ratio = 1.0 - d_j / n_j
        survival[j] = ratio if j == 0 else survival[j - 1] * ratio
        denom = n_j * (n_j - d_j)
        summand = d_j / denom if denom > 0 else 0.0
        gw[j] = summand if j == 0 else gw[j - 1] + summand
    return survival, gw


# ===================================================================
# Main analyser class
# ===================================================================
class PrivacyPreservingSurvivalAnalyzer:
    """Privacy-preserving survival analysis for federated oncology trials.

    Provides Kaplan-Meier estimation, log-rank testing, and Cox
    proportional-hazards regression.  Patient identifiers are
    HMAC-SHA256-hashed before any computation.
    """

    def __init__(
        self,
        hmac_key: bytes | None = None,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        max_iterations: int = MAX_NEWTON_RAPHSON_ITER,
        convergence_tol: float = NEWTON_RAPHSON_TOL,
    ) -> None:
        self._hmac_key = hmac_key if hmac_key is not None else DEFAULT_HMAC_KEY
        self.confidence_level = confidence_level
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        alpha = 1.0 - self.confidence_level
        self._z_alpha = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))
        _audit_log(
            "analyser_init",
            {
                "confidence_level": self.confidence_level,
                "max_iterations": self.max_iterations,
            },
        )

    # ----- Patient-ID hashing -----

    def hash_patient_id(self, raw_id: str) -> str:
        """Return the HMAC-SHA256 hex digest of *raw_id*."""
        return hmac.new(
            self._hmac_key,
            raw_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ----- Kaplan-Meier estimator -----

    def compute_kaplan_meier(
        self,
        data: Sequence[SurvivalData],
    ) -> KaplanMeierEstimate:
        """Kaplan-Meier product-limit estimator with Greenwood CIs.

        Confidence intervals use the log(-log S(t)) transform so that
        bounds are guaranteed to lie in [0, 1].
        """
        if not data:
            raise ValueError("Cannot compute Kaplan-Meier on empty data")
        _audit_log("km_start", {"n_records": len(data)})

        times = np.array([d.time for d in data], dtype=np.float64)
        events = np.array([d.event for d in data], dtype=np.int32)
        # Sort by time; break ties by putting events before censored
        order = np.lexsort((-events, times))
        times, events = times[order], events[order]

        unique_times = np.unique(times[events == 1])
        if len(unique_times) == 0:
            return KaplanMeierEstimate(
                time_points=np.array([0.0]),
                survival_probabilities=np.ones(1),
                ci_lower=np.ones(1),
                ci_upper=np.ones(1),
                at_risk=np.array([len(data)]),
                events=np.zeros(1, dtype=np.int32),
                confidence_level=self.confidence_level,
            )

        n_t = len(unique_times)
        at_risk_arr = np.array([max(int(np.sum(times >= t_j)), MIN_AT_RISK) for t_j in unique_times], dtype=np.int64)
        events_arr = np.array([int(np.sum((times == t_j) & (events == 1))) for t_j in unique_times], dtype=np.int64)

        survival, gw = _product_limit(at_risk_arr, events_arr)
        ci_lo, ci_hi = _greenwood_ci(survival, gw, self._z_alpha)

        _audit_log(
            "km_complete",
            {
                "n_time_points": n_t,
                "n_events": int(np.sum(events_arr)),
                "final_survival": round(float(survival[-1]), 4),
            },
        )
        return KaplanMeierEstimate(
            time_points=unique_times,
            survival_probabilities=survival,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            at_risk=at_risk_arr,
            events=events_arr,
            confidence_level=self.confidence_level,
        )

    # ----- Two-sample survival tests -----

    def compute_log_rank_test(
        self,
        group_a: Sequence[SurvivalData],
        group_b: Sequence[SurvivalData],
        test_type: TestType = TestType.LOG_RANK,
    ) -> LogRankResult:
        """Weighted two-sample log-rank test (log-rank / Wilcoxon / Tarone-Ware / FH)."""
        if not group_a or not group_b:
            raise ValueError("Both groups must be non-empty")
        _audit_log(
            "log_rank_start",
            {
                "n_a": len(group_a),
                "n_b": len(group_b),
                "test_type": test_type.value,
            },
        )

        times_a = np.array([d.time for d in group_a], dtype=np.float64)
        ev_a = np.array([d.event for d in group_a], dtype=np.int32)
        times_b = np.array([d.time for d in group_b], dtype=np.float64)
        ev_b = np.array([d.event for d in group_b], dtype=np.int32)

        all_event_times = np.unique(
            np.concatenate(
                [
                    times_a[ev_a == 1],
                    times_b[ev_b == 1],
                ]
            )
        )
        if len(all_event_times) == 0:
            return LogRankResult(chi_squared=0.0, p_value=1.0, test_type=test_type)

        # Preliminary pooled KM for Fleming-Harrington weights
        pooled_km = (
            self.compute_kaplan_meier(list(group_a) + list(group_b))
            if test_type == TestType.FLEMING_HARRINGTON
            else None
        )

        numerator = denominator = obs_a_total = exp_a_total = 0.0
        for k, t_k in enumerate(all_event_times):
            n_a_k = float(np.sum(times_a >= t_k))
            n_b_k = float(np.sum(times_b >= t_k))
            n_k = n_a_k + n_b_k
            if n_k < MIN_AT_RISK:
                continue

            d_a_k = float(np.sum((times_a == t_k) & (ev_a == 1)))
            d_b_k = float(np.sum((times_b == t_k) & (ev_b == 1)))
            d_k = d_a_k + d_b_k
            e_a_k = d_k * (n_a_k / n_k)

            # Weight selection
            if test_type == TestType.WILCOXON:
                w_k = n_k
            elif test_type == TestType.TARONE_WARE:
                w_k = math.sqrt(n_k)
            elif test_type == TestType.FLEMING_HARRINGTON:
                if pooled_km is not None and len(pooled_km.time_points) > 0:
                    idx = np.searchsorted(pooled_km.time_points, t_k, side="left")
                    w_k = float(pooled_km.survival_probabilities[idx - 1]) if idx > 0 else 1.0
                else:
                    w_k = 1.0
            else:  # LOG_RANK
                w_k = 1.0

            numerator += w_k * (d_a_k - e_a_k)
            # Hypergeometric variance: d * n_a * n_b * (n - d) / (n^2 * (n-1))
            var_k = (d_k * n_a_k * n_b_k * (n_k - d_k)) / (n_k**2 * (n_k - 1)) if n_k > 1 else 0.0
            denominator += w_k**2 * var_k
            obs_a_total += d_a_k
            exp_a_total += e_a_k

        chi_sq = (numerator**2) / denominator if denominator > 0 else 0.0
        p_value = float(1.0 - sp_stats.chi2.cdf(chi_sq, df=1))
        obs_b_total = float(np.sum(ev_b))
        exp_b_total = obs_b_total + obs_a_total - exp_a_total

        _audit_log(
            "log_rank_complete",
            {
                "chi_squared": round(chi_sq, 4),
                "p_value": round(p_value, 6),
            },
        )
        return LogRankResult(
            chi_squared=chi_sq,
            p_value=p_value,
            degrees_of_freedom=1,
            test_type=test_type,
            observed_a=obs_a_total,
            expected_a=exp_a_total,
            observed_b=obs_b_total,
            expected_b=exp_b_total,
        )

    # ----- Cox PH regression -----

    def fit_cox_ph(
        self,
        data: Sequence[SurvivalData],
        covariate_names: Sequence[str],
    ) -> CoxPHResult:
        """Cox PH via Newton-Raphson (Breslow partial likelihood)."""
        if not data:
            raise ValueError("Cannot fit Cox PH on empty data")
        if not covariate_names:
            raise ValueError("At least one covariate is required")

        n, p = len(data), len(covariate_names)
        _audit_log("cox_ph_start", {"n_subjects": n, "n_covariates": p})

        times = np.array([d.time for d in data], dtype=np.float64)
        events = np.array([d.event for d in data], dtype=np.int32)
        X = np.zeros((n, p), dtype=np.float64)
        for i, rec in enumerate(data):
            for j, name in enumerate(covariate_names):
                if name not in rec.covariates:
                    raise ValueError(f"Covariate '{name}' missing from record {rec.patient_id_hash[:8]}...")
                X[i, j] = rec.covariates[name]

        # Sort descending by time for efficient risk-set computation
        order = np.argsort(-times)
        times, events, X = times[order], events[order], X[order]

        beta = np.zeros(p, dtype=np.float64)
        log_pl = -np.inf
        converged = False
        n_iter = 0
        delta = np.inf

        for iteration in range(self.max_iterations):
            n_iter = iteration + 1
            grad, hessian, log_pl = self._cox_gradient_hessian(beta, X, times, events)
            try:
                h_inv = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                logger.warning("Singular Hessian at iter %d — adding ridge", n_iter)
                h_inv = np.linalg.inv(hessian + 1e-6 * np.eye(p))

            beta_new = beta - h_inv @ grad
            delta = float(np.max(np.abs(beta_new - beta)))
            beta = beta_new
            logger.debug("NR iter %d: delta=%.2e, log_pl=%.6f", n_iter, delta, log_pl)
            if delta < self.convergence_tol:
                converged = True
                break

        if not converged:
            logger.warning("Cox PH did not converge in %d iters (delta=%.2e)", self.max_iterations, delta)

        # Standard errors from observed information
        se = self._safe_se_from_hessian(hessian, p)
        hr = np.exp(beta)
        ci = np.column_stack(
            [
                np.exp(beta - self._z_alpha * se),
                np.exp(beta + self._z_alpha * se),
            ]
        )
        p_vals = np.array(
            [2.0 * (1.0 - float(sp_stats.norm.cdf(abs(beta[j] / se[j])))) if se[j] > 0 else 1.0 for j in range(p)]
        )
        c_idx = self.compute_concordance_index(X @ beta, times, events)

        _audit_log(
            "cox_ph_complete",
            {
                "iterations": n_iter,
                "converged": converged,
                "log_pl": round(log_pl, 4),
                "c_index": round(c_idx, 4),
            },
        )
        return CoxPHResult(
            coefficients=beta,
            hazard_ratios=hr,
            confidence_intervals=ci,
            p_values=p_vals,
            concordance_index=c_idx,
            standard_errors=se,
            covariate_names=list(covariate_names),
            iterations=n_iter,
            log_partial_likelihood=log_pl,
        )

    def _cox_gradient_hessian(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Gradient, Hessian, log-PL (Breslow).  Data sorted *descending* by time."""
        n, p = X.shape
        eta = X @ beta
        # Numerical stability: subtract max before exp
        exp_eta = np.exp(eta - np.max(eta))

        # Prefix sums give risk-set totals (data sorted descending)
        cum_exp = np.cumsum(exp_eta)
        cum_exp_x = np.cumsum(exp_eta[:, np.newaxis] * X, axis=0)
        # For Hessian: cumulative weighted outer product
        cum_exp_xx = np.zeros((n, p, p), dtype=np.float64)
        for i in range(n):
            outer_i = exp_eta[i] * np.outer(X[i], X[i])
            cum_exp_xx[i] = outer_i if i == 0 else cum_exp_xx[i - 1] + outer_i

        grad = np.zeros(p, dtype=np.float64)
        hessian = np.zeros((p, p), dtype=np.float64)
        log_pl = 0.0

        for i in range(n):
            if events[i] == 0:
                continue
            denom = cum_exp[i]
            if denom <= 0:
                continue
            w_x = cum_exp_x[i] / denom  # E[X | risk set]
            w_xx = cum_exp_xx[i] / denom  # E[XX' | risk set]

            grad += X[i] - w_x
            hessian -= w_xx - np.outer(w_x, w_x)
            log_pl += eta[i] - math.log(denom)

        return grad, hessian, log_pl

    @staticmethod
    def _safe_se_from_hessian(hessian: np.ndarray, p: int) -> np.ndarray:
        """SE from observed information, with ridge fallback."""
        try:
            var_b = np.linalg.inv(-hessian)
        except np.linalg.LinAlgError:
            logger.warning("Singular information matrix — ridge fallback")
            var_b = np.linalg.inv(-hessian + 1e-6 * np.eye(p))
        return np.sqrt(np.maximum(np.diag(var_b), 0.0))

    # ----- Median survival -----

    def compute_median_survival(self, km: KaplanMeierEstimate) -> Optional[float]:
        """Smallest time t where S(t) <= 0.5, or None if never reached."""
        below = np.where(km.survival_probabilities <= 0.5)[0]
        if len(below) == 0:
            logger.info("Median survival undefined — curve does not cross 0.5")
            return None
        median = float(km.time_points[below[0]])
        _audit_log("median_survival", {"median": median})
        return median

    # ----- Restricted mean survival time -----

    def compute_restricted_mean(self, km: KaplanMeierEstimate, tau: float) -> float:
        """RMST = area under the KM curve from 0 to *tau* (trapezoidal rule)."""
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        t_ext = np.concatenate([[0.0], km.time_points])
        s_ext = np.concatenate([[1.0], km.survival_probabilities])
        mask = t_ext <= tau
        t_tr, s_tr = t_ext[mask], s_ext[mask]
        if t_tr[-1] < tau:
            t_tr = np.append(t_tr, tau)
            s_tr = np.append(s_tr, s_tr[-1])
        # np.trapezoid (numpy >= 2.0) replaces the deprecated np.trapz
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        rmst = float(_trapz(s_tr, t_tr))
        _audit_log("rmst", {"tau": tau, "rmst": round(rmst, 4)})
        return rmst

    # ----- Harrell's concordance index -----

    def compute_concordance_index(
        self,
        predicted: np.ndarray,
        observed_times: np.ndarray,
        observed_events: np.ndarray,
    ) -> float:
        """Harrell's C-index.  0.5 = random, 1.0 = perfect discrimination."""
        n = len(predicted)
        if n < 2:
            return 0.5
        concordant = discordant = tied = 0.0
        for i in range(n):
            if observed_events[i] == 0:
                continue
            for j in range(n):
                if i == j or observed_times[j] < observed_times[i]:
                    continue
                # Skip non-admissible pair: j censored at same time
                if observed_times[j] == observed_times[i] and observed_events[j] == 0:
                    continue
                if predicted[i] > predicted[j]:
                    concordant += 1.0
                elif predicted[i] < predicted[j]:
                    discordant += 1.0
                else:
                    tied += 0.5
        total = concordant + discordant + tied
        return float((concordant + 0.5 * tied) / total) if total > 0 else 0.5

    # ----- Federated Kaplan-Meier -----

    def federated_kaplan_meier(
        self,
        site_contributions: Sequence[SiteContribution],
    ) -> FederatedSurvivalSummary:
        """Aggregate site-level at-risk/event tables into a global KM curve."""
        if not site_contributions:
            raise ValueError("At least one site contribution is required")
        _audit_log("fed_km_start", {"n_sites": len(site_contributions)})

        all_times = np.unique(np.concatenate([sc.time_points for sc in site_contributions]))
        n_t = len(all_times)
        g_risk = np.zeros(n_t, dtype=np.int64)
        g_events = np.zeros(n_t, dtype=np.int64)

        for sc in site_contributions:
            for k, t_k in enumerate(all_times):
                match = np.where(np.isclose(sc.time_points, t_k))[0]
                if len(match) > 0:
                    g_risk[k] += sc.at_risk[match[0]]
                    g_events[k] += sc.events[match[0]]
                else:
                    # Carry forward at-risk from closest earlier time
                    pos = np.searchsorted(sc.time_points, t_k, side="right") - 1
                    if 0 <= pos < len(sc.at_risk):
                        g_risk[k] += sc.at_risk[pos]

        survival, gw = _product_limit(g_risk, g_events)
        ci_lo, ci_hi = _greenwood_ci(survival, gw, self._z_alpha)

        agg_km = KaplanMeierEstimate(
            time_points=all_times,
            survival_probabilities=survival,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            at_risk=g_risk,
            events=g_events,
            confidence_level=self.confidence_level,
        )
        tot_ev = int(np.sum(g_events))
        tot_pt = sum(sc.total_patients for sc in site_contributions)

        _audit_log(
            "fed_km_complete",
            {
                "n_sites": len(site_contributions),
                "total_events": tot_ev,
                "total_patients": tot_pt,
            },
        )
        return FederatedSurvivalSummary(
            site_contributions=list(site_contributions),
            aggregated_km=agg_km,
            total_events=tot_ev,
            total_patients=tot_pt,
            num_sites=len(site_contributions),
        )

    # ----- Federated Cox aggregation -----

    def federated_cox_aggregation(
        self,
        site_gradients: Sequence[np.ndarray],
        site_hessians: Sequence[np.ndarray],
        covariate_names: Optional[Sequence[str]] = None,
        current_beta: Optional[np.ndarray] = None,
    ) -> CoxPHResult:
        """One Newton-Raphson step from aggregated site gradients/Hessians."""
        if not site_gradients or not site_hessians:
            raise ValueError("At least one site gradient and Hessian required")
        if len(site_gradients) != len(site_hessians):
            raise ValueError("Number of gradients and Hessians must match")

        p = len(site_gradients[0])
        _audit_log("fed_cox_start", {"n_sites": len(site_gradients), "p": p})

        g_grad = sum(np.asarray(g, dtype=np.float64) for g in site_gradients)
        g_hess = sum(np.asarray(h, dtype=np.float64) for h in site_hessians)
        beta = (
            np.asarray(current_beta, dtype=np.float64).copy()
            if current_beta is not None
            else np.zeros(p, dtype=np.float64)
        )

        try:
            h_inv = np.linalg.inv(g_hess)
        except np.linalg.LinAlgError:
            logger.warning("Singular global Hessian — ridge fallback")
            h_inv = np.linalg.inv(g_hess + 1e-6 * np.eye(p))

        beta_new = beta - h_inv @ g_grad
        se = self._safe_se_from_hessian(g_hess, p)
        hr = np.exp(beta_new)
        ci = np.column_stack(
            [
                np.exp(beta_new - self._z_alpha * se),
                np.exp(beta_new + self._z_alpha * se),
            ]
        )
        p_vals = np.array(
            [2.0 * (1.0 - float(sp_stats.norm.cdf(abs(beta_new[j] / se[j])))) if se[j] > 0 else 1.0 for j in range(p)]
        )
        names = list(covariate_names) if covariate_names else [f"x{j}" for j in range(p)]

        _audit_log(
            "fed_cox_complete",
            {
                "n_sites": len(site_gradients),
                "max_step": float(np.max(np.abs(beta_new - beta))),
            },
        )
        return CoxPHResult(
            coefficients=beta_new,
            hazard_ratios=hr,
            confidence_intervals=ci,
            p_values=p_vals,
            concordance_index=0.0,
            standard_errors=se,
            covariate_names=names,
            iterations=1,
            log_partial_likelihood=0.0,
        )

    # ----- Utilities -----

    def prepare_patient_data(
        self,
        raw_id: str,
        time: float,
        event: int,
        group: str = "control",
        covariates: Optional[Dict[str, float]] = None,
        endpoint: SurvivalEndpoint = SurvivalEndpoint.OVERALL_SURVIVAL,
        censoring_type: CensoringType = CensoringType.RIGHT,
    ) -> SurvivalData:
        """Create a SurvivalData record with HMAC-SHA256 hashed patient ID."""
        return SurvivalData(
            patient_id_hash=self.hash_patient_id(raw_id),
            time=time,
            event=event,
            group=group,
            covariates=covariates or {},
            endpoint=endpoint,
            censoring_type=censoring_type,
        )

    def extract_site_contribution(
        self,
        data: Sequence[SurvivalData],
        site_id: str,
    ) -> SiteContribution:
        """Summarise local patient data into aggregate counts for federation."""
        if not data:
            raise ValueError("Cannot extract contribution from empty data")
        times = np.array([d.time for d in data], dtype=np.float64)
        ev = np.array([d.event for d in data], dtype=np.int32)
        uniq = np.sort(np.unique(times[ev == 1]))
        if len(uniq) == 0:
            uniq = np.array([0.0])
        at_risk = np.array([int(np.sum(times >= t)) for t in uniq], dtype=np.int64)
        ev_counts = np.array([int(np.sum((times == t) & (ev == 1))) for t in uniq], dtype=np.int64)
        _audit_log("site_contribution", {"site_id": site_id, "n_patients": len(data), "n_events": int(np.sum(ev))})
        return SiteContribution(
            site_id=site_id, time_points=uniq, at_risk=at_risk, events=ev_counts, total_patients=len(data)
        )

    def compute_site_cox_contributions(
        self,
        data: Sequence[SurvivalData],
        covariate_names: Sequence[str],
        beta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Local Cox gradient and Hessian at *beta* for federated aggregation."""
        if not data:
            raise ValueError("Cannot compute contributions from empty data")
        n, p = len(data), len(covariate_names)
        times = np.array([d.time for d in data], dtype=np.float64)
        events = np.array([d.event for d in data], dtype=np.int32)
        X = np.zeros((n, p), dtype=np.float64)
        for i, rec in enumerate(data):
            for j, name in enumerate(covariate_names):
                if name not in rec.covariates:
                    raise ValueError(f"Covariate '{name}' missing in record {rec.patient_id_hash[:8]}...")
                X[i, j] = rec.covariates[name]
        order = np.argsort(-times)
        grad, hess, _ = self._cox_gradient_hessian(beta, X[order], times[order], events[order])
        _audit_log(
            "site_cox_contribution",
            {"n_subjects": n, "n_covariates": p, "grad_norm": round(float(np.linalg.norm(grad)), 6)},
        )
        return grad, hess
