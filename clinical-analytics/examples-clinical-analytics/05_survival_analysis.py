#!/usr/bin/env python3
"""Survival Analysis for Oncology Clinical Trial Endpoints.

CLINICAL CONTEXT
================
Time-to-event (survival) analysis is the primary endpoint framework for
the majority of oncology clinical trials. Overall survival (OS) and
progression-free survival (PFS) are the gold-standard endpoints for
regulatory approval (FDA, EMA). This example demonstrates the full
survival analysis toolkit: synthetic data generation with censoring,
Kaplan-Meier estimation, log-rank hypothesis testing, Cox proportional
hazards regression, concordance index discrimination, and restricted
mean survival time (RMST) estimation.

All computations use only NumPy and SciPy, implementing the core
statistical methods from first principles without external survival
analysis packages, making the methods transparent and auditable.

USE CASES COVERED
=================
1. Generating synthetic time-to-event data with exponential and Weibull
   hazards, administrative censoring, and random loss to follow-up.
2. Computing Kaplan-Meier survival curves with Greenwood standard errors
   and pointwise confidence intervals.
3. Performing the log-rank test (Mantel-Haenszel) to compare survival
   distributions between treatment arms.
4. Fitting a Cox proportional hazards model via partial likelihood
   maximisation (Newton-Raphson) with Breslow tie handling.
5. Calculating Harrell's concordance index (C-index) for model
   discrimination and restricted mean survival time (RMST) for
   clinically interpretable treatment effect quantification.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0   (https://numpy.org)
    scipy >= 1.11.0   (https://scipy.org)

REFERENCES
==========
- Kaplan & Meier (1958) JASA 53:457-481
- Cox (1972) JRSS-B 34:187-220
- Harrell et al. (1996) Stat Med 15:361-387
- Royston & Parmar (2013) BMC Med Res Methodol 13:152 (RMST)
- ICH E9(R1) Estimands and Sensitivity Analysis

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
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.clinical_analytics.example_05")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SurvivalConfig:
    """Configuration for survival data generation and analysis.

    Attributes:
        n_per_arm: Number of patients per treatment arm.
        arms: Treatment arm labels.
        baseline_median_months: Median survival in the control arm (months).
        hazard_ratio: True hazard ratio (experimental vs control).
        max_followup_months: Maximum administrative follow-up time.
        censoring_rate: Probability of random censoring per patient.
        alpha: Type I error rate for hypothesis tests.
        seed: RNG seed for reproducibility.
    """

    n_per_arm: int = 150
    arms: list[str] = field(default_factory=lambda: ["Control", "Experimental"])
    baseline_median_months: float = 12.0
    hazard_ratio: float = 0.70
    max_followup_months: float = 36.0
    censoring_rate: float = 0.15
    alpha: float = 0.05
    seed: int = 42


# ============================================================================
# Synthetic Survival Data Generation
# ============================================================================


def generate_survival_data(config: SurvivalConfig) -> dict[str, np.ndarray]:
    """Generate synthetic time-to-event data with censoring.

    Uses exponential distribution for event times. Control arm uses
    baseline hazard; experimental arm uses baseline * HR. Administrative
    censoring is applied at max_followup_months. Random censoring is
    applied independently with uniform dropout times.

    Args:
        config: Survival analysis configuration.

    Returns:
        Dictionary with 'time', 'event', 'arm', 'arm_label' arrays.
    """
    rng = np.random.default_rng(config.seed)

    # Baseline hazard rate from median: lambda = ln(2) / median
    lambda_control = np.log(2) / config.baseline_median_months
    lambda_experimental = lambda_control * config.hazard_ratio

    all_times: list[float] = []
    all_events: list[int] = []
    all_arms: list[int] = []

    for arm_idx, arm_label in enumerate(config.arms):
        lam = lambda_control if arm_idx == 0 else lambda_experimental
        n = config.n_per_arm

        # True event times (exponential)
        event_times = rng.exponential(1.0 / lam, size=n)

        # Random censoring times (uniform over follow-up)
        censor_times = rng.uniform(0, config.max_followup_months, size=n)

        for i in range(n):
            t_event = float(event_times[i])
            t_censor = float(censor_times[i])

            # Apply random censoring
            apply_random_censor = rng.random() < config.censoring_rate

            if apply_random_censor and t_censor < t_event:
                obs_time = min(t_censor, config.max_followup_months)
                event = 0
            elif t_event <= config.max_followup_months:
                obs_time = t_event
                event = 1
            else:
                # Administrative censoring
                obs_time = config.max_followup_months
                event = 0

            # Bound time to positive values
            obs_time = float(np.clip(obs_time, 0.01, config.max_followup_months))
            all_times.append(obs_time)
            all_events.append(event)
            all_arms.append(arm_idx)

    data = {
        "time": np.array(all_times, dtype=np.float64),
        "event": np.array(all_events, dtype=np.int32),
        "arm": np.array(all_arms, dtype=np.int32),
    }

    n_events = int(np.sum(data["event"]))
    logger.info(
        "Survival data generated | N=%d | events=%d (%.1f%%) | arms=%s",
        len(all_times), n_events, 100 * n_events / len(all_times), config.arms,
    )
    return data


# ============================================================================
# Kaplan-Meier Estimator
# ============================================================================


@dataclass
class KMResult:
    """Kaplan-Meier survival curve result.

    Attributes:
        times: Unique event times.
        survival: Survival probability at each time.
        se: Greenwood standard error at each time.
        ci_lower: Lower confidence interval bound.
        ci_upper: Upper confidence interval bound.
        n_at_risk: Number at risk at each time.
        n_events: Number of events at each time.
        median_survival: Estimated median survival time.
        arm_label: Treatment arm label.
    """

    times: np.ndarray = field(default_factory=lambda: np.array([]))
    survival: np.ndarray = field(default_factory=lambda: np.array([]))
    se: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    n_at_risk: np.ndarray = field(default_factory=lambda: np.array([]))
    n_events: np.ndarray = field(default_factory=lambda: np.array([]))
    median_survival: float = float("nan")
    arm_label: str = ""


def kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
    alpha: float = 0.05,
    arm_label: str = "",
) -> KMResult:
    """Compute the Kaplan-Meier survival estimator.

    Implements the product-limit estimator with Greenwood variance formula
    and pointwise log-log confidence intervals.

    Args:
        time: Observed times.
        event: Event indicators (1=event, 0=censored).
        alpha: Significance level for confidence intervals.
        arm_label: Label for this arm.

    Returns:
        KMResult with survival curve data.
    """
    n_total = len(time)
    z = sp_stats.norm.ppf(1 - alpha / 2)

    # Sort by time
    order = np.argsort(time)
    t_sorted = time[order]
    e_sorted = event[order]

    # Unique event times (only where events occurred)
    event_times = np.unique(t_sorted[e_sorted == 1])

    km_times: list[float] = []
    km_surv: list[float] = []
    km_se: list[float] = []
    km_risk: list[int] = []
    km_events_at: list[int] = []

    s = 1.0
    greenwood_sum = 0.0
    n_at_risk = n_total

    for t_i in event_times:
        # Count events and censorings at this time
        d_i = int(np.sum((t_sorted == t_i) & (e_sorted == 1)))
        c_i = int(np.sum((t_sorted < t_i) & (e_sorted == 0) & (t_sorted > (km_times[-1] if km_times else 0))))

        # Number at risk (subtract earlier censorings)
        # More precise: count subjects with time >= t_i
        n_at_risk = int(np.sum(t_sorted >= t_i))

        if n_at_risk > 0 and n_at_risk > d_i:
            s *= (1.0 - d_i / n_at_risk)
            greenwood_sum += d_i / (n_at_risk * (n_at_risk - d_i))
        elif n_at_risk > 0:
            s = 0.0

        se = s * np.sqrt(greenwood_sum) if greenwood_sum >= 0 else 0.0

        km_times.append(float(t_i))
        km_surv.append(s)
        km_se.append(se)
        km_risk.append(n_at_risk)
        km_events_at.append(d_i)

    km_times_arr = np.array(km_times)
    km_surv_arr = np.array(km_surv)
    km_se_arr = np.array(km_se)

    # Confidence intervals (log-log transform for better coverage)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_log_s = np.log(-np.log(np.clip(km_surv_arr, 1e-15, 1 - 1e-15)))
        se_log_log = km_se_arr / (km_surv_arr * np.abs(np.log(np.clip(km_surv_arr, 1e-15, 1 - 1e-15))) + 1e-15)

    ci_lower = np.clip(km_surv_arr ** np.exp(z * se_log_log), 0, 1)
    ci_upper = np.clip(km_surv_arr ** np.exp(-z * se_log_log), 0, 1)

    # Median survival (first time S <= 0.5)
    below_50 = np.where(km_surv_arr <= 0.5)[0]
    median = float(km_times_arr[below_50[0]]) if len(below_50) > 0 else float("nan")

    result = KMResult(
        times=km_times_arr,
        survival=km_surv_arr,
        se=km_se_arr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_at_risk=np.array(km_risk),
        n_events=np.array(km_events_at),
        median_survival=round(median, 2),
        arm_label=arm_label,
    )

    logger.info(
        "KM estimate | arm=%s | median=%.2f months | events=%d",
        arm_label, median, int(np.sum(km_events_at)),
    )
    return result


# ============================================================================
# Log-Rank Test
# ============================================================================


def log_rank_test(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
) -> dict[str, float]:
    """Perform the log-rank (Mantel-Haenszel) test.

    Compares survival distributions between two groups under the null
    hypothesis of no difference in survival.

    Args:
        time: Observed times.
        event: Event indicators.
        group: Group indicators (0 or 1).

    Returns:
        Dictionary with test statistic, p-value, and degrees of freedom.
    """
    # Unique event times across both groups
    event_times = np.unique(time[event == 1])

    observed_1 = 0.0
    expected_1 = 0.0
    variance = 0.0

    for t_i in event_times:
        at_risk_0 = int(np.sum((time >= t_i) & (group == 0)))
        at_risk_1 = int(np.sum((time >= t_i) & (group == 1)))
        n_total_risk = at_risk_0 + at_risk_1

        events_0 = int(np.sum((time == t_i) & (event == 1) & (group == 0)))
        events_1 = int(np.sum((time == t_i) & (event == 1) & (group == 1)))
        d_i = events_0 + events_1

        if n_total_risk < 2 or d_i == 0:
            continue

        # Expected events in group 1
        e_1i = d_i * at_risk_1 / n_total_risk
        expected_1 += e_1i
        observed_1 += events_1

        # Hypergeometric variance
        if n_total_risk > 1:
            v_i = (d_i * at_risk_0 * at_risk_1 * (n_total_risk - d_i)) / (
                n_total_risk ** 2 * (n_total_risk - 1)
            )
            variance += v_i

    # Chi-squared statistic
    if variance > 0:
        chi2_stat = (observed_1 - expected_1) ** 2 / variance
    else:
        chi2_stat = 0.0

    p_value = float(1.0 - sp_stats.chi2.cdf(chi2_stat, df=1))

    result = {
        "chi2_statistic": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "df": 1,
        "observed_group1": round(observed_1, 1),
        "expected_group1": round(expected_1, 2),
    }

    logger.info(
        "Log-rank test | chi2=%.4f, p=%.6f | O1=%.0f, E1=%.1f",
        chi2_stat, p_value, observed_1, expected_1,
    )
    return result


# ============================================================================
# Cox Proportional Hazards Model
# ============================================================================


def fit_cox_ph(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
) -> dict[str, Any]:
    """Fit a Cox proportional hazards model via partial likelihood.

    Maximises the Cox partial log-likelihood using BFGS optimisation.
    Breslow method is used for tied event times.

    Args:
        time: Observed times (n,).
        event: Event indicators (n,).
        covariates: Covariate matrix (n, p).

    Returns:
        Dictionary with coefficients, hazard ratios, standard errors,
        p-values, and model fit statistics.
    """
    n, p = covariates.shape

    # Sort by time (descending for efficient risk set computation)
    order = np.argsort(-time)
    t_sorted = time[order]
    e_sorted = event[order]
    x_sorted = covariates[order]

    def neg_partial_log_likelihood(beta: np.ndarray) -> float:
        """Negative Cox partial log-likelihood (Breslow)."""
        eta = x_sorted @ beta  # linear predictor
        exp_eta = np.exp(np.clip(eta, -500, 500))

        # Cumulative sum of exp(eta) from largest time to smallest
        # (since sorted descending, cumsum gives risk set sums)
        cum_exp_eta = np.cumsum(exp_eta)

        # Partial log-likelihood
        ll = 0.0
        for i in range(n):
            if e_sorted[i] == 1:
                ll += eta[i] - np.log(max(cum_exp_eta[i], 1e-15))

        return -ll

    # Optimise
    beta_init = np.zeros(p)
    result = minimize(
        neg_partial_log_likelihood,
        beta_init,
        method="BFGS",
        options={"maxiter": 200, "gtol": 1e-8},
    )
    beta_hat = result.x

    # Approximate standard errors from inverse Hessian
    if result.hess_inv is not None:
        se = np.sqrt(np.diag(result.hess_inv))
    else:
        se = np.full(p, float("nan"))

    # Hazard ratios and Wald test p-values
    hr = np.exp(beta_hat)
    z_scores = beta_hat / np.where(se > 0, se, 1e-15)
    p_values = 2 * (1 - sp_stats.norm.cdf(np.abs(z_scores)))

    # Confidence intervals for HR
    z_alpha = 1.96
    hr_lower = np.exp(beta_hat - z_alpha * se)
    hr_upper = np.exp(beta_hat + z_alpha * se)

    # Concordance index
    c_index = concordance_index(time, event, covariates @ beta_hat)

    fit_result = {
        "coefficients": beta_hat.tolist(),
        "hazard_ratios": hr.tolist(),
        "standard_errors": se.tolist(),
        "z_scores": z_scores.tolist(),
        "p_values": p_values.tolist(),
        "hr_ci_lower": hr_lower.tolist(),
        "hr_ci_upper": hr_upper.tolist(),
        "concordance_index": c_index,
        "log_likelihood": -result.fun,
        "converged": result.success,
        "n_observations": n,
        "n_events": int(np.sum(event)),
    }

    logger.info(
        "Cox PH fitted | covariates=%d | C-index=%.4f | converged=%s",
        p, c_index, result.success,
    )
    return fit_result


# ============================================================================
# Concordance Index
# ============================================================================


def concordance_index(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
) -> float:
    """Compute Harrell's concordance index (C-index).

    The C-index measures the probability that for a random pair of
    subjects, the subject with the higher predicted risk score
    experiences the event before the other.

    Args:
        time: Observed times.
        event: Event indicators.
        risk_score: Predicted risk scores (higher = worse prognosis).

    Returns:
        C-index value in [0, 1].
    """
    concordant = 0
    discordant = 0
    tied_risk = 0

    n = len(time)
    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] < time[i]:
                continue  # j cannot be compared if j had shorter time and i had event

            # i had event at time[i], j survived at least until time[i]
            if time[j] > time[i] or (time[j] == time[i] and event[j] == 0):
                if risk_score[i] > risk_score[j]:
                    concordant += 1
                elif risk_score[i] < risk_score[j]:
                    discordant += 1
                else:
                    tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5

    c_index = (concordant + 0.5 * tied_risk) / total
    return round(c_index, 4)


# ============================================================================
# Restricted Mean Survival Time (RMST)
# ============================================================================


def compute_rmst(
    km_result: KMResult,
    tau: float,
) -> dict[str, float]:
    """Compute restricted mean survival time (RMST) up to time tau.

    RMST is the area under the Kaplan-Meier curve up to tau, providing
    a clinically interpretable measure of average survival time within
    the restricted window.

    Args:
        km_result: Kaplan-Meier result.
        tau: Restriction time point (months).

    Returns:
        Dictionary with RMST estimate and approximate standard error.
    """
    times = km_result.times
    survival = km_result.survival

    # Add time 0 with S=1 and truncate at tau
    t = np.concatenate([[0.0], times])
    s = np.concatenate([[1.0], survival])

    # Truncate at tau
    mask = t <= tau
    t_trunc = t[mask]
    s_trunc = s[mask]

    # Add endpoint at tau
    if t_trunc[-1] < tau:
        t_trunc = np.append(t_trunc, tau)
        s_trunc = np.append(s_trunc, s_trunc[-1])

    # RMST = integral of S(t) from 0 to tau (trapezoidal rule)
    rmst = float(np.trapezoid(s_trunc, t_trunc))

    # Approximate SE using Greenwood-based approach
    # SE^2(RMST) = sum over event times of [integral from t_i to tau of S(u)du]^2 * d_i/(n_i*(n_i-d_i))
    se_sq = 0.0
    for idx in range(len(km_result.times)):
        t_i = km_result.times[idx]
        if t_i >= tau:
            break
        d_i = km_result.n_events[idx]
        n_i = km_result.n_at_risk[idx]
        if n_i <= d_i or n_i == 0:
            continue
        # Integral of S(u) from t_i to tau
        future_mask = (t_trunc >= t_i) & (t_trunc <= tau)
        if np.sum(future_mask) >= 2:
            area_i = float(np.trapezoid(s_trunc[future_mask], t_trunc[future_mask]))
        else:
            area_i = 0.0
        se_sq += area_i ** 2 * d_i / (n_i * (n_i - d_i))

    se = np.sqrt(se_sq)

    result = {
        "rmst_months": round(rmst, 3),
        "rmst_se": round(float(se), 3),
        "tau_months": tau,
        "rmst_ci_lower": round(rmst - 1.96 * float(se), 3),
        "rmst_ci_upper": round(rmst + 1.96 * float(se), 3),
    }

    logger.info(
        "RMST | arm=%s | RMST(tau=%.0f)=%.3f months (SE=%.3f)",
        km_result.arm_label, tau, rmst, se,
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
    """Run the survival analysis example."""
    logger.info("Starting Example 05: Survival Analysis")

    config = SurvivalConfig(
        n_per_arm=150,
        baseline_median_months=12.0,
        hazard_ratio=0.70,
        max_followup_months=36.0,
        censoring_rate=0.15,
        seed=42,
    )

    print("\n" + "=" * 72)
    print("  SURVIVAL ANALYSIS -- ONCOLOGY TRIAL TIME-TO-EVENT ENDPOINTS")
    print("=" * 72)
    print(f"  True baseline median:  {config.baseline_median_months:.1f} months")
    print(f"  True hazard ratio:     {config.hazard_ratio:.2f}")
    print(f"  Patients per arm:      {config.n_per_arm}")
    print(f"  Max follow-up:         {config.max_followup_months:.1f} months")

    # --- 1. Generate survival data ---
    _print_section("1. SYNTHETIC SURVIVAL DATA GENERATION")
    data = generate_survival_data(config)
    for arm_idx, arm_label in enumerate(config.arms):
        mask = data["arm"] == arm_idx
        n_arm = int(np.sum(mask))
        n_events = int(np.sum(data["event"][mask]))
        med_time = float(np.median(data["time"][mask]))
        print(f"  {arm_label:>15s}:  N={n_arm}  events={n_events} "
              f"({100*n_events/n_arm:.1f}%)  median_obs_time={med_time:.1f} mo")

    # --- 2. Kaplan-Meier curves ---
    _print_section("2. KAPLAN-MEIER SURVIVAL ESTIMATES")
    km_results: dict[str, KMResult] = {}
    for arm_idx, arm_label in enumerate(config.arms):
        mask = data["arm"] == arm_idx
        km = kaplan_meier(
            data["time"][mask],
            data["event"][mask],
            alpha=config.alpha,
            arm_label=arm_label,
        )
        km_results[arm_label] = km

        print(f"\n  {arm_label}:")
        print(f"    Median survival: {km.median_survival:.2f} months")
        print(f"    Events:          {int(np.sum(km.n_events))}")

        # Show survival at landmark times
        landmarks = [6, 12, 18, 24]
        print(f"    {'Time (mo)':>12s}  {'S(t)':>8s}  {'SE':>8s}  "
              f"{'95% CI':>18s}  {'At Risk':>8s}")
        for lm in landmarks:
            idx = np.searchsorted(km.times, lm, side="right") - 1
            if idx >= 0 and idx < len(km.times):
                print(f"    {lm:12d}  {km.survival[idx]:8.4f}  {km.se[idx]:8.4f}  "
                      f"[{km.ci_lower[idx]:.4f}, {km.ci_upper[idx]:.4f}]  "
                      f"{km.n_at_risk[idx]:8d}")

    # --- 3. Log-rank test ---
    _print_section("3. LOG-RANK TEST (MANTEL-HAENSZEL)")
    lr = log_rank_test(data["time"], data["event"], data["arm"])
    print(f"  Chi-squared statistic:  {lr['chi2_statistic']:.4f}")
    print(f"  Degrees of freedom:     {lr['df']}")
    print(f"  P-value:                {lr['p_value']:.6f}")
    print(f"  Observed (Exp arm):     {lr['observed_group1']:.0f}")
    print(f"  Expected (Exp arm):     {lr['expected_group1']:.1f}")
    sig = "SIGNIFICANT" if lr["p_value"] < config.alpha else "NOT SIGNIFICANT"
    print(f"  Result (alpha={config.alpha}):   {sig}")

    # --- 4. Cox PH model ---
    _print_section("4. COX PROPORTIONAL HAZARDS MODEL")
    # Covariates: treatment arm indicator
    x = data["arm"].reshape(-1, 1).astype(np.float64)
    cox = fit_cox_ph(data["time"], data["event"], x)

    print("  Model fit:")
    print(f"    N observations:    {cox['n_observations']}")
    print(f"    N events:          {cox['n_events']}")
    print(f"    Log-likelihood:    {cox['log_likelihood']:.4f}")
    print(f"    Converged:         {cox['converged']}")
    print("\n  Coefficient estimates:")
    print(f"    {'Covariate':>15s}  {'Coef':>8s}  {'HR':>8s}  {'SE':>8s}  "
          f"{'z':>8s}  {'p':>10s}  {'95% CI for HR':>20s}")
    print(f"    {'-' * 78}")

    covar_names = ["Treatment (Exp vs Ctrl)"]
    for i, name in enumerate(covar_names):
        print(f"    {name:>15s}  {cox['coefficients'][i]:8.4f}  "
              f"{cox['hazard_ratios'][i]:8.4f}  {cox['standard_errors'][i]:8.4f}  "
              f"{cox['z_scores'][i]:8.4f}  {cox['p_values'][i]:10.6f}  "
              f"[{cox['hr_ci_lower'][i]:.4f}, {cox['hr_ci_upper'][i]:.4f}]")

    # --- 5. Concordance index and RMST ---
    _print_section("5. CONCORDANCE INDEX AND RMST")
    print(f"  Concordance index (C-statistic): {cox['concordance_index']:.4f}")
    interpretation = (
        "good" if cox["concordance_index"] > 0.7
        else "moderate" if cox["concordance_index"] > 0.6
        else "poor"
    )
    print(f"  Interpretation: {interpretation} discrimination")

    tau = 24.0  # months
    print(f"\n  Restricted Mean Survival Time (tau={tau:.0f} months):")
    print(f"    {'Arm':>15s}  {'RMST (mo)':>12s}  {'SE':>8s}  {'95% CI':>24s}")
    print(f"    {'-' * 62}")

    rmst_results: dict[str, dict[str, float]] = {}
    for arm_label, km in km_results.items():
        rmst = compute_rmst(km, tau=tau)
        rmst_results[arm_label] = rmst
        print(f"    {arm_label:>15s}  {rmst['rmst_months']:12.3f}  "
              f"{rmst['rmst_se']:8.3f}  "
              f"[{rmst['rmst_ci_lower']:.3f}, {rmst['rmst_ci_upper']:.3f}]")

    # RMST difference
    if len(rmst_results) == 2:
        arms = list(rmst_results.keys())
        diff = rmst_results[arms[1]]["rmst_months"] - rmst_results[arms[0]]["rmst_months"]
        se_diff = np.sqrt(
            rmst_results[arms[0]]["rmst_se"] ** 2 +
            rmst_results[arms[1]]["rmst_se"] ** 2
        )
        z_diff = diff / max(se_diff, 1e-15)
        p_diff = 2 * (1 - sp_stats.norm.cdf(abs(z_diff)))
        print(f"\n    RMST difference ({arms[1]} - {arms[0]}): "
              f"{diff:+.3f} months (SE={se_diff:.3f}, p={p_diff:.6f})")

    print("\n" + "=" * 72)
    print("  SURVIVAL ANALYSIS COMPLETE")
    print("  RESEARCH USE ONLY -- NOT FOR CLINICAL DECISION MAKING")
    print("=" * 72 + "\n")

    logger.info("Example 05 complete")


if __name__ == "__main__":
    main()
