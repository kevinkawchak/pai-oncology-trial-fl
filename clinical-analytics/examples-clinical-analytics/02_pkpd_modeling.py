#!/usr/bin/env python3
"""Pharmacokinetic / Pharmacodynamic (PK/PD) Modeling for Oncology Trials.

CLINICAL CONTEXT
================
This example demonstrates core PK/PD modeling workflows used in oncology
clinical trial analytics. Pharmacokinetic models describe how a drug is
absorbed, distributed, metabolised, and eliminated (ADME), while
pharmacodynamic models characterise the relationship between drug
concentration and therapeutic effect.

Accurate PK/PD modeling is essential for dose selection, therapeutic drug
monitoring, and understanding inter-patient variability -- all critical
for oncology dose-finding studies (ICH E4, ICH E9).

USE CASES COVERED
=================
1. One-compartment PK model with first-order absorption and elimination,
   simulating plasma concentration-time profiles.
2. Two-compartment PK model with distribution to a peripheral compartment,
   capturing biphasic elimination kinetics.
3. Non-compartmental analysis (NCA): computation of AUC (trapezoidal),
   Cmax, and Tmax from simulated concentration-time data.
4. Emax dose-response model demonstrating sigmoidal relationship between
   dose and pharmacodynamic effect with Hill coefficient.
5. Covariate effects on PK parameters: allometric scaling of clearance
   by body weight and age-dependent renal function adjustment.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0   (https://numpy.org)
    scipy >= 1.11.0   (https://scipy.org)

REFERENCES
==========
- Rowland & Tozer, Clinical Pharmacokinetics and Pharmacodynamics, 5th Ed
- ICH E4: Dose-Response Information to Support Drug Registration
- ICH E9(R1): Addendum on Estimands and Sensitivity Analysis
- Gabrielsson & Weiner, Pharmacokinetic and Pharmacodynamic Data Analysis

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
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.clinical_analytics.example_02")


# ============================================================================
# PK Parameter Dataclasses
# ============================================================================


@dataclass
class OneCompartmentParams:
    """Parameters for a one-compartment PK model with first-order absorption.

    Attributes:
        dose_mg: Administered dose in milligrams.
        ka_hr: Absorption rate constant (1/h).
        ke_hr: Elimination rate constant (1/h).
        vd_l: Volume of distribution in litres.
        f_bioavail: Bioavailability fraction [0, 1].
    """

    dose_mg: float = 500.0
    ka_hr: float = 1.2
    ke_hr: float = 0.15
    vd_l: float = 50.0
    f_bioavail: float = 0.85


@dataclass
class TwoCompartmentParams:
    """Parameters for a two-compartment PK model.

    Attributes:
        dose_mg: Administered dose in milligrams.
        ka_hr: Absorption rate constant (1/h).
        k10_hr: Elimination rate from central compartment (1/h).
        k12_hr: Distribution rate central -> peripheral (1/h).
        k21_hr: Distribution rate peripheral -> central (1/h).
        vc_l: Volume of central compartment in litres.
        f_bioavail: Bioavailability fraction [0, 1].
    """

    dose_mg: float = 500.0
    ka_hr: float = 1.2
    k10_hr: float = 0.10
    k12_hr: float = 0.20
    k21_hr: float = 0.08
    vc_l: float = 30.0
    f_bioavail: float = 0.85


@dataclass
class EmaxParams:
    """Parameters for a sigmoidal Emax dose-response model.

    Attributes:
        emax: Maximum achievable effect (arbitrary units).
        ec50: Concentration at half-maximal effect (mg/L).
        hill: Hill coefficient (sigmoidicity factor).
        e0: Baseline effect (placebo response).
    """

    emax: float = 100.0
    ec50: float = 10.0
    hill: float = 1.5
    e0: float = 5.0


@dataclass
class CovariateProfile:
    """Patient covariate profile for population PK adjustments.

    Attributes:
        patient_id: Unique patient identifier.
        weight_kg: Body weight in kilograms [40, 160].
        age_years: Age in years [18, 95].
        sex: Biological sex ('M' or 'F').
        creatinine_clearance_ml_min: Estimated CrCl (Cockcroft-Gault).
        albumin_g_dl: Serum albumin [1.5, 5.5].
    """

    patient_id: str = "PAT-0001"
    weight_kg: float = 70.0
    age_years: int = 60
    sex: str = "M"
    creatinine_clearance_ml_min: float = 90.0
    albumin_g_dl: float = 4.0


# ============================================================================
# One-Compartment PK Model
# ============================================================================


class OneCompartmentModel:
    """One-compartment PK model with first-order absorption and elimination.

    The oral dose enters an absorption compartment and is transferred to
    the systemic (central) compartment at rate ka, with elimination at rate ke.

    C(t) = (F * Dose * ka) / (Vd * (ka - ke)) * [exp(-ke*t) - exp(-ka*t)]

    Args:
        params: One-compartment model parameters.
    """

    def __init__(self, params: OneCompartmentParams) -> None:
        self._p = params
        logger.info(
            "OneCompartmentModel | dose=%.1f mg, ka=%.3f/h, ke=%.3f/h, Vd=%.1f L",
            params.dose_mg, params.ka_hr, params.ke_hr, params.vd_l,
        )

    def concentration(self, t: np.ndarray) -> np.ndarray:
        """Compute plasma concentration at time points t (hours).

        Args:
            t: Array of time points in hours.

        Returns:
            Plasma concentration in mg/L at each time point.
        """
        p = self._p
        coeff = (p.f_bioavail * p.dose_mg * p.ka_hr) / (p.vd_l * (p.ka_hr - p.ke_hr))
        conc = coeff * (np.exp(-p.ke_hr * t) - np.exp(-p.ka_hr * t))
        return np.clip(conc, 0.0, None)


# ============================================================================
# Two-Compartment PK Model
# ============================================================================


class TwoCompartmentModel:
    """Two-compartment PK model solved via ODE integration.

    Three ODEs describe absorption site (A_gut), central compartment (A1),
    and peripheral compartment (A2):
        dA_gut/dt = -ka * A_gut
        dA1/dt    = ka * A_gut - (k10 + k12) * A1 + k21 * A2
        dA2/dt    = k12 * A1 - k21 * A2

    Args:
        params: Two-compartment model parameters.
    """

    def __init__(self, params: TwoCompartmentParams) -> None:
        self._p = params
        logger.info(
            "TwoCompartmentModel | dose=%.1f mg, ka=%.3f/h, k10=%.3f/h, "
            "k12=%.3f/h, k21=%.3f/h, Vc=%.1f L",
            params.dose_mg, params.ka_hr, params.k10_hr,
            params.k12_hr, params.k21_hr, params.vc_l,
        )

    def _ode_system(self, t: float, y: list[float]) -> list[float]:
        """ODE right-hand side for the two-compartment model."""
        a_gut, a1, a2 = y
        p = self._p
        da_gut = -p.ka_hr * a_gut
        da1 = p.ka_hr * a_gut - (p.k10_hr + p.k12_hr) * a1 + p.k21_hr * a2
        da2 = p.k12_hr * a1 - p.k21_hr * a2
        return [da_gut, da1, da2]

    def concentration(self, t: np.ndarray) -> np.ndarray:
        """Compute plasma concentration in the central compartment.

        Args:
            t: Array of time points in hours.

        Returns:
            Plasma concentration in mg/L at each time point.
        """
        p = self._p
        y0 = [p.f_bioavail * p.dose_mg, 0.0, 0.0]
        sol = solve_ivp(
            self._ode_system,
            t_span=(float(t[0]), float(t[-1])),
            y0=y0,
            t_eval=t,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        conc = np.clip(sol.y[1] / p.vc_l, 0.0, None)
        return conc


# ============================================================================
# Non-Compartmental Analysis (NCA)
# ============================================================================


def compute_nca(t: np.ndarray, conc: np.ndarray) -> dict[str, float]:
    """Perform non-compartmental analysis on concentration-time data.

    Computes AUC (linear trapezoidal), Cmax, Tmax, and half-life
    estimate from the terminal phase.

    Args:
        t: Time points in hours.
        conc: Corresponding concentrations in mg/L.

    Returns:
        Dictionary with NCA parameters.
    """
    # AUC by linear trapezoidal rule
    auc = float(np.trapezoid(conc, t))

    # Cmax and Tmax
    idx_max = int(np.argmax(conc))
    cmax = float(conc[idx_max])
    tmax = float(t[idx_max])

    # Terminal half-life estimate (last 30% of data in log-linear phase)
    n_terminal = max(int(0.3 * len(t)), 3)
    t_term = t[-n_terminal:]
    c_term = conc[-n_terminal:]
    positive_mask = c_term > 0
    if np.sum(positive_mask) >= 2:
        log_c = np.log(c_term[positive_mask])
        t_pos = t_term[positive_mask]
        # Linear regression on log-concentration
        slope, _ = np.polyfit(t_pos, log_c, 1)
        ke_terminal = -slope
        t_half = float(np.log(2) / ke_terminal) if ke_terminal > 0 else float("nan")
    else:
        t_half = float("nan")

    results = {
        "auc_mg_h_l": round(auc, 2),
        "cmax_mg_l": round(cmax, 3),
        "tmax_h": round(tmax, 2),
        "t_half_h": round(t_half, 2),
    }
    logger.info(
        "NCA results | AUC=%.2f mg*h/L, Cmax=%.3f mg/L, Tmax=%.2f h, t1/2=%.2f h",
        auc, cmax, tmax, t_half,
    )
    return results


# ============================================================================
# Emax Dose-Response Model
# ============================================================================


class EmaxModel:
    """Sigmoidal Emax dose-response model.

    E(C) = E0 + Emax * C^hill / (EC50^hill + C^hill)

    Args:
        params: Emax model parameters.
    """

    def __init__(self, params: EmaxParams) -> None:
        self._p = params
        logger.info(
            "EmaxModel | Emax=%.1f, EC50=%.1f, Hill=%.2f, E0=%.1f",
            params.emax, params.ec50, params.hill, params.e0,
        )

    def effect(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute pharmacodynamic effect for given concentrations.

        Args:
            concentrations: Drug concentrations in mg/L.

        Returns:
            Effect values (arbitrary units).
        """
        p = self._p
        c_pow = np.power(np.clip(concentrations, 0, None), p.hill)
        ec50_pow = p.ec50 ** p.hill
        effect = p.e0 + p.emax * c_pow / (ec50_pow + c_pow)
        return effect

    def compute_ec_values(self) -> dict[str, float]:
        """Compute key effective concentration landmarks.

        Returns:
            Dictionary with EC10, EC20, EC50, EC80, EC90 values.
        """
        p = self._p
        results: dict[str, float] = {}
        for pct in [10, 20, 50, 80, 90]:
            frac = pct / 100.0
            # C = EC50 * (frac / (1 - frac))^(1/hill)
            ec = p.ec50 * (frac / (1.0 - frac)) ** (1.0 / p.hill)
            results[f"ec{pct}_mg_l"] = round(ec, 3)
        return results


# ============================================================================
# Covariate Effects on Clearance
# ============================================================================


def compute_adjusted_clearance(
    base_cl_l_h: float,
    profile: CovariateProfile,
    ref_weight_kg: float = 70.0,
    ref_age_years: int = 50,
    weight_exponent: float = 0.75,
    age_coefficient: float = -0.005,
) -> dict[str, Any]:
    """Compute covariate-adjusted clearance using allometric scaling.

    CL_adj = CL_base * (WT/WT_ref)^theta_WT * exp(theta_AGE * (AGE - AGE_ref))

    Allometric scaling with body weight follows the well-established
    3/4 power law (West et al., 1997). Age effect is modelled as an
    exponential decline reflecting reduced renal/hepatic function.

    Args:
        base_cl_l_h: Population typical clearance (L/h).
        profile: Patient covariate profile.
        ref_weight_kg: Reference body weight for allometric scaling.
        ref_age_years: Reference age.
        weight_exponent: Allometric exponent for weight.
        age_coefficient: Coefficient for age effect.

    Returns:
        Dictionary with adjusted clearance and component effects.
    """
    # Allometric weight scaling
    weight_factor = (profile.weight_kg / ref_weight_kg) ** weight_exponent

    # Age-dependent decline
    age_factor = np.exp(age_coefficient * (profile.age_years - ref_age_years))

    # Renal function adjustment (proportional to CrCl)
    renal_factor = profile.creatinine_clearance_ml_min / 90.0

    # Combined adjusted clearance
    cl_adjusted = base_cl_l_h * weight_factor * float(age_factor) * renal_factor
    cl_adjusted = float(np.clip(cl_adjusted, 0.5, 100.0))  # bounded

    result = {
        "patient_id": profile.patient_id,
        "base_cl_l_h": round(base_cl_l_h, 3),
        "weight_factor": round(weight_factor, 4),
        "age_factor": round(float(age_factor), 4),
        "renal_factor": round(renal_factor, 4),
        "adjusted_cl_l_h": round(cl_adjusted, 3),
        "percent_change": round(100.0 * (cl_adjusted - base_cl_l_h) / base_cl_l_h, 1),
    }
    logger.info(
        "Covariate-adjusted CL | patient=%s | base=%.3f -> adj=%.3f L/h (%.1f%%)",
        profile.patient_id, base_cl_l_h, cl_adjusted, result["percent_change"],
    )
    return result


# ============================================================================
# Printing utilities
# ============================================================================


def _print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def _print_dict(d: dict[str, Any], indent: int = 4) -> None:
    """Print a dictionary with formatted key-value pairs."""
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{prefix}{k:30s}  {v:>12.4f}")
        else:
            print(f"{prefix}{k:30s}  {str(v):>12s}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the PK/PD modeling example."""
    logger.info("Starting Example 02: PK/PD Modeling")

    print("\n" + "=" * 72)
    print("  PK/PD MODELING EXAMPLE -- ONCOLOGY DOSE CHARACTERISATION")
    print("=" * 72)

    # --- One-compartment model ---
    _print_section("1. ONE-COMPARTMENT PK MODEL")
    pk1_params = OneCompartmentParams(dose_mg=500.0, ka_hr=1.2, ke_hr=0.15, vd_l=50.0)
    pk1 = OneCompartmentModel(pk1_params)
    t = np.linspace(0.01, 48.0, 500)
    conc_1c = pk1.concentration(t)
    nca_1c = compute_nca(t, conc_1c)
    print("  One-compartment NCA results:")
    _print_dict(nca_1c)

    # --- Two-compartment model ---
    _print_section("2. TWO-COMPARTMENT PK MODEL")
    pk2_params = TwoCompartmentParams(
        dose_mg=500.0, ka_hr=1.2, k10_hr=0.10, k12_hr=0.20, k21_hr=0.08, vc_l=30.0,
    )
    pk2 = TwoCompartmentModel(pk2_params)
    conc_2c = pk2.concentration(t)
    nca_2c = compute_nca(t, conc_2c)
    print("  Two-compartment NCA results:")
    _print_dict(nca_2c)

    # --- Comparison ---
    _print_section("3. MODEL COMPARISON (1C vs 2C)")
    for key in nca_1c:
        v1 = nca_1c[key]
        v2 = nca_2c[key]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v2 - v1
            print(f"    {key:25s}  1C={v1:>10.3f}  2C={v2:>10.3f}  diff={diff:>+10.3f}")

    # --- Emax dose-response ---
    _print_section("4. EMAX DOSE-RESPONSE MODEL")
    emax_params = EmaxParams(emax=85.0, ec50=12.0, hill=1.8, e0=5.0)
    emax_model = EmaxModel(emax_params)
    doses = np.array([0, 1, 2, 5, 10, 15, 20, 30, 50, 100], dtype=np.float64)
    effects = emax_model.effect(doses)
    print("  Dose-Response Table:")
    print(f"    {'Dose (mg/L)':>15s}  {'Effect':>12s}  {'% of Emax':>12s}")
    for d, e in zip(doses, effects):
        pct = 100.0 * (e - emax_params.e0) / emax_params.emax
        print(f"    {d:15.1f}  {e:12.2f}  {pct:11.1f}%")

    ec_values = emax_model.compute_ec_values()
    print("\n  Effective Concentration Landmarks:")
    _print_dict(ec_values)

    # --- Covariate effects ---
    _print_section("5. COVARIATE EFFECTS ON CLEARANCE")
    base_clearance = 5.0  # L/h typical population value
    rng = np.random.default_rng(42)

    profiles = []
    for i in range(8):
        wt = float(np.clip(rng.normal(75, 18), 40.0, 160.0))
        age = int(np.clip(rng.normal(62, 12), 18, 95))
        sex = str(rng.choice(["M", "F"]))
        crcl = float(np.clip(rng.normal(85, 25), 20.0, 160.0))
        alb = float(np.clip(rng.normal(3.8, 0.6), 1.5, 5.5))
        profiles.append(CovariateProfile(
            patient_id=f"PAT-{i + 1:04d}",
            weight_kg=round(wt, 1),
            age_years=age,
            sex=sex,
            creatinine_clearance_ml_min=round(crcl, 1),
            albumin_g_dl=round(alb, 1),
        ))

    print(f"  Base population CL: {base_clearance:.1f} L/h")
    print(f"  {'Patient':>10s}  {'Wt(kg)':>8s}  {'Age':>5s}  {'CrCl':>8s}  "
          f"{'CL_adj':>8s}  {'%Change':>8s}")
    print(f"  {'-' * 58}")

    for profile in profiles:
        adj = compute_adjusted_clearance(base_clearance, profile)
        print(
            f"  {profile.patient_id:>10s}  {profile.weight_kg:8.1f}  "
            f"{profile.age_years:5d}  {profile.creatinine_clearance_ml_min:8.1f}  "
            f"{adj['adjusted_cl_l_h']:8.3f}  {adj['percent_change']:>+7.1f}%"
        )

    # --- Summary ---
    print("\n" + "=" * 72)
    print("  PK/PD MODELING COMPLETE")
    print("  RESEARCH USE ONLY -- NOT FOR CLINICAL DECISION MAKING")
    print("=" * 72 + "\n")

    logger.info("Example 02 complete")


if __name__ == "__main__":
    main()
