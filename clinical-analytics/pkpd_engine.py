"""Population PK/PD modeling engine — federated pharmacokinetic analysis.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Implements population pharmacokinetic/pharmacodynamic (PopPK/PD) modeling for
oncology clinical trials conducted across federated sites.  Supports one- and
two-compartment PK models with first-order absorption/elimination, multiple PD
response surfaces (Emax, sigmoid Emax, linear, log-linear, turnover), covariate
adjustment (body weight, age, renal function), and federated aggregation of
site-level summary statistics without sharing patient-level data.

DISCLAIMER: RESEARCH USE ONLY. This software is provided for research and educational
purposes only. It has NOT been validated for clinical use and must NOT be used for
patient care decisions.

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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate, optimize

# ---------------------------------------------------------------------------
# Structured logging — 21 CFR Part 11 audit trail support
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_HMAC_KEY = b"pai-oncology-pkpd-audit-key"

# ---------------------------------------------------------------------------
# Constants — clinically plausible parameter bounds
# ---------------------------------------------------------------------------
_EPS: float = 1e-12
_MAX_CONCENTRATION_MG_L: float = 1000.0
_MAX_TIME_HOURS: float = 720.0  # 30 days
_DEFAULT_TIME_POINTS: int = 500
_AUC_INTEGRATION_HOURS: float = 168.0  # 7-day integration window


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class PKModel(str, Enum):
    """Pharmacokinetic compartmental model type."""

    ONE_COMPARTMENT = "one_compartment"
    TWO_COMPARTMENT = "two_compartment"
    THREE_COMPARTMENT = "three_compartment"
    PHYSIOLOGICAL = "physiological"


class PDModel(str, Enum):
    """Pharmacodynamic response model type."""

    EMAX = "emax"
    SIGMOID_EMAX = "sigmoid_emax"
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    TURNOVER = "turnover"


class EliminationRoute(str, Enum):
    """Primary drug elimination pathway."""

    HEPATIC = "hepatic"
    RENAL = "renal"
    BILIARY = "biliary"
    MIXED = "mixed"


class DosingRegimen(str, Enum):
    """Route and method of drug administration."""

    BOLUS_IV = "bolus_iv"
    INFUSION = "infusion"
    ORAL = "oral"
    SUBCUTANEOUS = "subcutaneous"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PKParameters:
    """Pharmacokinetic parameters for compartmental modeling.

    RESEARCH USE ONLY.

    Attributes:
        clearance_l_h: Systemic clearance (L/h).
        volume_central_l: Volume of distribution, central compartment (L).
        volume_peripheral_l: Volume of distribution, peripheral compartment (L).
        half_life_h: Terminal elimination half-life (h).
        bioavailability: Fraction of dose reaching systemic circulation [0-1].
        ka_h: First-order absorption rate constant (1/h).
        ke_h: First-order elimination rate constant (1/h).
        q_l_h: Inter-compartmental clearance (L/h) for two-compartment models.
        infusion_duration_h: Duration of IV infusion (h), 0 for bolus.
    """

    clearance_l_h: float = 10.0
    volume_central_l: float = 50.0
    volume_peripheral_l: float = 80.0
    half_life_h: float = 6.0
    bioavailability: float = 1.0
    ka_h: float = 1.5
    ke_h: float = 0.0
    q_l_h: float = 5.0
    infusion_duration_h: float = 0.0

    def __post_init__(self) -> None:
        self.clearance_l_h = max(self.clearance_l_h, _EPS)
        self.volume_central_l = max(self.volume_central_l, _EPS)
        self.volume_peripheral_l = max(self.volume_peripheral_l, _EPS)
        self.half_life_h = max(self.half_life_h, _EPS)
        self.bioavailability = float(np.clip(self.bioavailability, 0.0, 1.0))
        self.ka_h = max(self.ka_h, 0.0)
        # Derive ke from CL/V if not explicitly provided
        if self.ke_h <= 0.0:
            self.ke_h = self.clearance_l_h / self.volume_central_l
        self.q_l_h = max(self.q_l_h, 0.0)
        self.infusion_duration_h = max(self.infusion_duration_h, 0.0)


@dataclass
class PDParameters:
    """Pharmacodynamic parameters for dose-response modeling.

    RESEARCH USE ONLY.

    Attributes:
        emax: Maximum achievable drug effect (arbitrary units).
        ec50: Concentration producing 50 percent of Emax (mg/L).
        hill_coefficient: Hill coefficient governing steepness of sigmoid.
        baseline_effect: Baseline effect in absence of drug.
        kin: Zero-order production rate constant for turnover models (1/h).
        kout: First-order loss rate constant for turnover models (1/h).
    """

    emax: float = 100.0
    ec50: float = 5.0
    hill_coefficient: float = 1.0
    baseline_effect: float = 0.0
    kin: float = 1.0
    kout: float = 0.1

    def __post_init__(self) -> None:
        self.emax = max(self.emax, _EPS)
        self.ec50 = max(self.ec50, _EPS)
        self.hill_coefficient = max(self.hill_coefficient, _EPS)
        self.kin = max(self.kin, _EPS)
        self.kout = max(self.kout, _EPS)


@dataclass
class PopulationPKPDConfig:
    """Configuration for a population PK/PD analysis run.

    RESEARCH USE ONLY.

    Attributes:
        pk_model: Compartmental PK model type.
        pd_model: PD response model type.
        elimination_route: Primary elimination pathway.
        dosing_regimen: Route of administration.
        covariates: List of covariate names to include.
        random_effects_variance: Between-subject variability variances (omega^2).
        residual_error_variance: Residual unexplained variability (sigma^2).
        error_model: Residual error model type (additive, proportional, combined).
    """

    pk_model: PKModel = PKModel.ONE_COMPARTMENT
    pd_model: PDModel = PDModel.EMAX
    elimination_route: EliminationRoute = EliminationRoute.HEPATIC
    dosing_regimen: DosingRegimen = DosingRegimen.BOLUS_IV
    covariates: List[str] = field(
        default_factory=lambda: ["weight", "age", "creatinine_clearance"],
    )
    random_effects_variance: Dict[str, float] = field(
        default_factory=lambda: {"clearance": 0.09, "volume": 0.04},
    )
    residual_error_variance: float = 0.04
    error_model: str = "proportional"


@dataclass
class ConcentrationProfile:
    """Time-concentration profile resulting from a PK simulation.

    RESEARCH USE ONLY.

    Attributes:
        time_points_h: Array of time points (hours).
        concentrations_mg_l: Array of plasma concentrations (mg/L).
        auc_mg_h_l: Area under the concentration-time curve (mg*h/L).
        cmax_mg_l: Peak plasma concentration (mg/L).
        tmax_h: Time to peak concentration (h).
        half_life_h: Observed terminal half-life (h).
        metadata: Additional run metadata.
    """

    time_points_h: np.ndarray = field(default_factory=lambda: np.array([]))
    concentrations_mg_l: np.ndarray = field(default_factory=lambda: np.array([]))
    auc_mg_h_l: float = 0.0
    cmax_mg_l: float = 0.0
    tmax_h: float = 0.0
    half_life_h: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DoseResponse:
    """Dose-response relationship result from a PD simulation.

    RESEARCH USE ONLY.

    Attributes:
        doses: Array of evaluated dose levels.
        effects: Array of predicted effects at each dose.
        therapeutic_window_low: Lower bound of the therapeutic window (mg/L).
        therapeutic_window_high: Upper bound of the therapeutic window (mg/L).
        ec50_estimated: Estimated EC50 from the curve data.
        emax_estimated: Estimated Emax from the curve data.
        metadata: Additional run metadata.
    """

    doses: np.ndarray = field(default_factory=lambda: np.array([]))
    effects: np.ndarray = field(default_factory=lambda: np.array([]))
    therapeutic_window_low: float = 0.0
    therapeutic_window_high: float = 0.0
    ec50_estimated: float = 0.0
    emax_estimated: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audit helper — 21 CFR Part 11 compliant logging
# ---------------------------------------------------------------------------
def _audit_log(
    action: str,
    actor: str,
    resource: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a 21 CFR Part 11 compliant audit record with HMAC integrity."""
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = f"{action}:{actor}:{resource}:{timestamp}"
    integrity_hash = hmac.new(
        _HMAC_KEY, payload.encode(), hashlib.sha256,
    ).hexdigest()
    record: Dict[str, Any] = {
        "event_id": f"PKPD-{uuid.uuid4().hex[:12].upper()}",
        "timestamp": timestamp,
        "action": action,
        "actor": actor,
        "resource": resource,
        "details": details or {},
        "integrity_hash": integrity_hash,
    }
    logger.info("AUDIT [%s] actor=%s resource=%s", action, actor, resource)
    return record


def _hash_patient_id(patient_id: str) -> str:
    """Hash a patient identifier using HMAC-SHA256 for de-identification.

    Returns a hex-encoded pseudonym that cannot be reversed without the key.
    """
    return hmac.new(
        _HMAC_KEY, patient_id.encode("utf-8"), hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Trapezoidal AUC helper (numpy version-safe)
# ---------------------------------------------------------------------------
def _trapezoidal_auc(y: np.ndarray, x: np.ndarray) -> float:
    """Compute AUC via the trapezoidal rule, compatible with numpy >= 1.24."""
    _trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz_fn is not None:
        return float(_trapz_fn(y, x))
    # Manual fallback when neither name is available
    dx = np.diff(x)
    avg_y = 0.5 * (y[:-1] + y[1:])
    return float(np.sum(dx * avg_y))


# ---------------------------------------------------------------------------
# PopulationPKPDEngine
# ---------------------------------------------------------------------------
class PopulationPKPDEngine:
    """Population PK/PD modeling engine for federated oncology trials.

    RESEARCH USE ONLY.

    Provides compartmental PK simulation (one- and two-compartment),
    pharmacodynamic response surface modeling, covariate-adjusted parameter
    estimation, and federated aggregation of site-level summary statistics.

    Args:
        config: Population PK/PD configuration.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        config: Optional[PopulationPKPDConfig] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or PopulationPKPDConfig()
        self._rng = np.random.default_rng(seed)
        self._audit_trail: List[Dict[str, Any]] = []
        self._run_count: int = 0
        logger.info(
            "PopulationPKPDEngine initialised: pk=%s pd=%s route=%s dosing=%s",
            self.config.pk_model.value,
            self.config.pd_model.value,
            self.config.elimination_route.value,
            self.config.dosing_regimen.value,
        )

    # ------------------------------------------------------------------
    # One-compartment PK model
    # ------------------------------------------------------------------
    def compute_one_compartment_profile(
        self,
        dose_mg: float,
        pk: Optional[PKParameters] = None,
        time_end_h: float = 48.0,
        n_points: int = _DEFAULT_TIME_POINTS,
    ) -> ConcentrationProfile:
        """Simulate a one-compartment PK concentration-time profile.

        RESEARCH USE ONLY.

        IV bolus:
            C(t) = (Dose * F / Vc) * exp(-ke * t)

        First-order oral absorption:
            C(t) = (Dose * F * ka) / (Vc * (ka - ke))
                   * (exp(-ke * t) - exp(-ka * t))

        Zero-order IV infusion (duration T_inf):
            C(t <= T_inf) = R0 / (Vc * ke) * (1 - exp(-ke * t))
            C(t >  T_inf) = C(T_inf) * exp(-ke * (t - T_inf))

        Args:
            dose_mg: Administered dose in mg.
            pk: Pharmacokinetic parameters (defaults used if None).
            time_end_h: End of simulation window in hours.
            n_points: Number of time points to evaluate.

        Returns:
            ConcentrationProfile with time, concentration, AUC, Cmax, Tmax.
        """
        pk = pk or PKParameters()
        time_end_h = float(np.clip(time_end_h, 0.1, _MAX_TIME_HOURS))
        dose_mg = max(dose_mg, 0.0)
        t = np.linspace(0.0, time_end_h, max(n_points, 10))

        ke = pk.ke_h
        ka = pk.ka_h
        vc = pk.volume_central_l
        f = pk.bioavailability
        is_oral = self.config.dosing_regimen in (
            DosingRegimen.ORAL, DosingRegimen.SUBCUTANEOUS,
        )

        if is_oral and ka > _EPS:
            # First-order absorption, one-compartment elimination
            denom = ka - ke
            if abs(denom) < _EPS:
                # L'Hopital limit when ka approaches ke
                c = (dose_mg * f * ka / max(vc, _EPS)) * t * np.exp(-ke * t)
            else:
                c = ((dose_mg * f * ka) / (max(vc, _EPS) * denom)) * (
                    np.exp(-ke * t) - np.exp(-ka * t)
                )
        elif pk.infusion_duration_h > _EPS:
            # Zero-order IV infusion
            t_inf = pk.infusion_duration_h
            r0 = dose_mg / t_inf  # infusion rate mg/h
            vc_ke = max(vc, _EPS) * ke + _EPS
            c = np.where(
                t <= t_inf,
                (r0 / vc_ke) * (1.0 - np.exp(-ke * t)),
                (r0 / vc_ke)
                * (1.0 - np.exp(-ke * t_inf))
                * np.exp(-ke * (t - t_inf)),
            )
        else:
            # IV bolus
            c0 = (dose_mg * f) / max(vc, _EPS)
            c = c0 * np.exp(-ke * t)

        c = np.clip(c, 0.0, _MAX_CONCENTRATION_MG_L)
        auc = _trapezoidal_auc(c, t)
        cmax, tmax = self._extract_cmax_tmax(c, t)
        obs_hl = self._estimate_terminal_half_life(c, t)

        self._run_count += 1
        self._audit_trail.append(
            _audit_log(
                "one_compartment_pk", "engine",
                f"run_{self._run_count}", {"dose_mg": dose_mg},
            )
        )
        return ConcentrationProfile(
            time_points_h=t,
            concentrations_mg_l=c,
            auc_mg_h_l=auc,
            cmax_mg_l=cmax,
            tmax_h=tmax,
            half_life_h=obs_hl,
            metadata={
                "pk_model": PKModel.ONE_COMPARTMENT.value,
                "dose_mg": dose_mg,
            },
        )

    # ------------------------------------------------------------------
    # Two-compartment PK model (ODE-based)
    # ------------------------------------------------------------------
    def compute_two_compartment_profile(
        self,
        dose_mg: float,
        pk: Optional[PKParameters] = None,
        time_end_h: float = 72.0,
        n_points: int = _DEFAULT_TIME_POINTS,
    ) -> ConcentrationProfile:
        """Simulate a two-compartment PK profile by solving the ODE system.

        RESEARCH USE ONLY.

        ODE system (IV bolus initial condition in central compartment):
            dA_c/dt = -(ke + k12) * A_c + k21 * A_p + Input(t)
            dA_p/dt =  k12 * A_c - k21 * A_p

        With oral absorption an additional gut compartment is prepended:
            dA_gut/dt = -ka * A_gut
            dA_c/dt   =  ka * A_gut - (ke + k12)*A_c + k21*A_p
            dA_p/dt   =  k12 * A_c - k21 * A_p

        Args:
            dose_mg: Administered dose in mg.
            pk: Pharmacokinetic parameters.
            time_end_h: Simulation end time in hours.
            n_points: Number of output time points.

        Returns:
            ConcentrationProfile with the full two-compartment solution.
        """
        pk = pk or PKParameters()
        time_end_h = float(np.clip(time_end_h, 0.1, _MAX_TIME_HOURS))
        dose_mg = max(dose_mg, 0.0)

        ke = pk.clearance_l_h / max(pk.volume_central_l, _EPS)
        k12 = pk.q_l_h / max(pk.volume_central_l, _EPS)
        k21 = pk.q_l_h / max(pk.volume_peripheral_l, _EPS)
        vc = pk.volume_central_l
        f = pk.bioavailability
        ka = pk.ka_h
        is_oral = self.config.dosing_regimen in (
            DosingRegimen.ORAL, DosingRegimen.SUBCUTANEOUS,
        )

        def _ode_iv(_t: float, y: np.ndarray) -> List[float]:
            a_c, a_p = float(y[0]), float(y[1])
            da_c = -(ke + k12) * a_c + k21 * a_p
            da_p = k12 * a_c - k21 * a_p
            return [da_c, da_p]

        def _ode_oral(_t: float, y: np.ndarray) -> List[float]:
            a_gut, a_c, a_p = float(y[0]), float(y[1]), float(y[2])
            da_gut = -ka * a_gut
            da_c = ka * a_gut - (ke + k12) * a_c + k21 * a_p
            da_p = k12 * a_c - k21 * a_p
            return [da_gut, da_c, da_p]

        t_span = (0.0, time_end_h)
        t_eval = np.linspace(0.0, time_end_h, max(n_points, 10))

        if is_oral and ka > _EPS:
            y0 = [dose_mg * f, 0.0, 0.0]
            sol = integrate.solve_ivp(
                _ode_oral, t_span, y0, t_eval=t_eval,
                method="RK45", rtol=1e-8, atol=1e-10,
            )
            amounts_central = sol.y[1]
        else:
            y0 = [dose_mg * f, 0.0]
            sol = integrate.solve_ivp(
                _ode_iv, t_span, y0, t_eval=t_eval,
                method="RK45", rtol=1e-8, atol=1e-10,
            )
            amounts_central = sol.y[0]

        c = amounts_central / max(vc, _EPS)
        c = np.clip(c, 0.0, _MAX_CONCENTRATION_MG_L)
        t_out = sol.t

        auc = _trapezoidal_auc(c, t_out)
        cmax, tmax = self._extract_cmax_tmax(c, t_out)
        obs_hl = self._estimate_terminal_half_life(c, t_out)

        self._run_count += 1
        self._audit_trail.append(
            _audit_log(
                "two_compartment_pk", "engine",
                f"run_{self._run_count}", {"dose_mg": dose_mg},
            )
        )
        return ConcentrationProfile(
            time_points_h=t_out,
            concentrations_mg_l=c,
            auc_mg_h_l=auc,
            cmax_mg_l=cmax,
            tmax_h=tmax,
            half_life_h=obs_hl,
            metadata={
                "pk_model": PKModel.TWO_COMPARTMENT.value,
                "dose_mg": dose_mg,
            },
        )

    # ------------------------------------------------------------------
    # PD response models
    # ------------------------------------------------------------------
    def compute_pd_response(
        self,
        concentrations: np.ndarray,
        pd: Optional[PDParameters] = None,
        pd_model: Optional[PDModel] = None,
    ) -> np.ndarray:
        """Compute pharmacodynamic effect for an array of concentrations.

        RESEARCH USE ONLY.

        Supported models:
            EMAX:          E = E0 + Emax * C / (EC50 + C)
            SIGMOID_EMAX:  E = E0 + Emax * C^n / (EC50^n + C^n)
            LINEAR:        E = E0 + slope * C
            LOG_LINEAR:    E = E0 + slope * ln(C + 1)
            TURNOVER:      R_ss = (kin / kout) * (1 + Emax*C / (EC50 + C))

        Args:
            concentrations: Array of drug concentrations (mg/L).
            pd: PD parameters.
            pd_model: Override the configured PD model type.

        Returns:
            Array of pharmacodynamic effects.
        """
        pd = pd or PDParameters()
        model = pd_model or self.config.pd_model
        c = np.asarray(concentrations, dtype=np.float64)
        c = np.clip(c, 0.0, _MAX_CONCENTRATION_MG_L)

        e0 = pd.baseline_effect
        emax = pd.emax
        ec50 = pd.ec50
        n = pd.hill_coefficient

        if model == PDModel.EMAX:
            effect = e0 + emax * c / (ec50 + c + _EPS)

        elif model == PDModel.SIGMOID_EMAX:
            cn = np.power(c + _EPS, n)
            ec50n = math.pow(ec50, n)
            effect = e0 + emax * cn / (ec50n + cn + _EPS)

        elif model == PDModel.LINEAR:
            slope = emax / max(ec50, _EPS)
            effect = e0 + slope * c

        elif model == PDModel.LOG_LINEAR:
            slope = emax / max(math.log(ec50 + 1.0), _EPS)
            effect = e0 + slope * np.log(c + 1.0)

        elif model == PDModel.TURNOVER:
            # Steady-state indirect response (stimulation of production)
            stimulation = 1.0 + emax * c / (ec50 + c + _EPS)
            effect = (pd.kin / max(pd.kout, _EPS)) * stimulation

        else:
            logger.warning("Unknown PD model %s — falling back to Emax", model)
            effect = e0 + emax * c / (ec50 + c + _EPS)

        return np.asarray(effect, dtype=np.float64)

    # ------------------------------------------------------------------
    # Dose-response curve
    # ------------------------------------------------------------------
    def compute_dose_response_curve(
        self,
        dose_range: Optional[np.ndarray] = None,
        pk: Optional[PKParameters] = None,
        pd: Optional[PDParameters] = None,
        n_doses: int = 50,
    ) -> DoseResponse:
        """Generate a full dose-response curve: PK at each dose then PD.

        RESEARCH USE ONLY.

        Args:
            dose_range: Explicit dose array (mg).  Log-spaced default if None.
            pk: PK parameters.
            pd: PD parameters.
            n_doses: Number of dose levels for the default range.

        Returns:
            DoseResponse with doses, effects, and therapeutic window bounds.
        """
        pk = pk or PKParameters()
        pd = pd or PDParameters()
        if dose_range is None:
            dose_range = np.logspace(-1, 3, max(n_doses, 5))

        effects = np.zeros(len(dose_range), dtype=np.float64)
        cmax_vals = np.zeros(len(dose_range), dtype=np.float64)

        for i, dose in enumerate(dose_range):
            profile = self.compute_one_compartment_profile(
                dose, pk=pk, time_end_h=48.0, n_points=200,
            )
            cmax_vals[i] = profile.cmax_mg_l
            pd_out = self.compute_pd_response(
                np.array([profile.cmax_mg_l]), pd=pd,
            )
            effects[i] = pd_out[0]

        tw_low, tw_high = self._find_therapeutic_window(
            dose_range, effects, pd.emax, pd.baseline_effect,
        )
        ec50_est = self._estimate_ec50_from_curve(
            cmax_vals, effects, pd.baseline_effect,
        )

        self._run_count += 1
        self._audit_trail.append(
            _audit_log(
                "dose_response_curve", "engine",
                f"run_{self._run_count}",
                {"n_doses": len(dose_range)},
            )
        )
        return DoseResponse(
            doses=dose_range,
            effects=effects,
            therapeutic_window_low=tw_low,
            therapeutic_window_high=tw_high,
            ec50_estimated=ec50_est,
            emax_estimated=float(np.max(effects)) if len(effects) > 0 else 0.0,
            metadata={"pd_model": self.config.pd_model.value},
        )

    # ------------------------------------------------------------------
    # Covariate effects
    # ------------------------------------------------------------------
    def apply_covariate_effects(
        self,
        pk: PKParameters,
        body_weight_kg: float = 70.0,
        age_years: float = 50.0,
        creatinine_clearance_ml_min: float = 100.0,
    ) -> PKParameters:
        """Adjust PK parameters for patient-level covariates.

        RESEARCH USE ONLY.

        Allometric scaling:
            CL_i = CL_pop * (WT/70)^0.75 * renal_effect * age_effect
            V_i  = V_pop  * (WT/70)^1.00

        Args:
            pk: Population-typical PK parameters.
            body_weight_kg: Patient body weight in kg.
            age_years: Patient age in years.
            creatinine_clearance_ml_min: Estimated creatinine clearance (mL/min).

        Returns:
            New PKParameters adjusted for the individual covariates.
        """
        wt = max(body_weight_kg, 1.0)
        age = max(age_years, 0.1)
        crcl = max(creatinine_clearance_ml_min, 1.0)
        wt_ratio = wt / 70.0

        # Renal function effect on clearance
        renal_effect = 1.0
        if self.config.elimination_route in (
            EliminationRoute.RENAL, EliminationRoute.MIXED,
        ):
            frac = 0.7 if self.config.elimination_route == EliminationRoute.RENAL else 0.35
            renal_effect = 1.0 - frac + frac * (crcl / 100.0)
            renal_effect = float(np.clip(renal_effect, 0.2, 2.0))

        # Age effect — mild reduction after 65
        age_effect = 1.0
        if age > 65.0:
            age_effect = max(1.0 - 0.005 * (age - 65.0), 0.7)

        cl_adj = pk.clearance_l_h * math.pow(wt_ratio, 0.75) * renal_effect * age_effect
        vc_adj = pk.volume_central_l * math.pow(wt_ratio, 1.0)
        vp_adj = pk.volume_peripheral_l * math.pow(wt_ratio, 1.0)
        q_adj = pk.q_l_h * math.pow(wt_ratio, 0.75)
        ke_adj = cl_adj / max(vc_adj, _EPS)
        hl_adj = math.log(2.0) / max(ke_adj, _EPS)

        logger.debug(
            "Covariate adjustment: WT=%.1fkg AGE=%.0fy CrCL=%.0f "
            "-> CL=%.2f Vc=%.2f ke=%.4f",
            wt, age, crcl, cl_adj, vc_adj, ke_adj,
        )
        return PKParameters(
            clearance_l_h=cl_adj,
            volume_central_l=vc_adj,
            volume_peripheral_l=vp_adj,
            half_life_h=hl_adj,
            bioavailability=pk.bioavailability,
            ka_h=pk.ka_h,
            ke_h=ke_adj,
            q_l_h=q_adj,
            infusion_duration_h=pk.infusion_duration_h,
        )

    # ------------------------------------------------------------------
    # Population parameter estimation (simplified NLME)
    # ------------------------------------------------------------------
    def estimate_population_parameters(
        self,
        observed_times: np.ndarray,
        observed_concentrations: np.ndarray,
        dose_mg: float,
        initial_pk: Optional[PKParameters] = None,
    ) -> Tuple[PKParameters, Dict[str, Any]]:
        """Estimate population PK parameters from observed concentration data.

        RESEARCH USE ONLY.

        Uses scipy.optimize.minimize (Nelder-Mead) to find CL and Vc that
        minimise weighted least squares on log-transformed concentrations,
        analogous to the FO method in NONMEM.

        Args:
            observed_times: Observation time points (hours).
            observed_concentrations: Measured plasma concentrations (mg/L).
            dose_mg: Administered dose (mg).
            initial_pk: Starting parameter estimates.

        Returns:
            Tuple of (optimised PKParameters, diagnostics dict).
        """
        initial_pk = initial_pk or PKParameters()
        obs_t = np.asarray(observed_times, dtype=np.float64)
        obs_c = np.asarray(observed_concentrations, dtype=np.float64)

        valid = obs_c > _EPS
        obs_t_v = obs_t[valid]
        obs_c_v = obs_c[valid]

        if len(obs_t_v) < 2:
            logger.warning("Insufficient valid observations for estimation")
            return initial_pk, {"converged": False, "message": "insufficient data"}

        log_obs = np.log(obs_c_v)

        def _objective(params: np.ndarray) -> float:
            cl_e = max(float(params[0]), _EPS)
            vc_e = max(float(params[1]), _EPS)
            ke_e = cl_e / vc_e
            f_val = initial_pk.bioavailability
            is_oral = self.config.dosing_regimen in (
                DosingRegimen.ORAL, DosingRegimen.SUBCUTANEOUS,
            )
            if is_oral and initial_pk.ka_h > _EPS:
                ka_val = initial_pk.ka_h
                diff = ka_val - ke_e
                if abs(diff) < _EPS:
                    pred = (dose_mg * f_val * ka_val / vc_e) * obs_t_v * np.exp(-ke_e * obs_t_v)
                else:
                    pred = (dose_mg * f_val * ka_val / (vc_e * diff)) * (
                        np.exp(-ke_e * obs_t_v) - np.exp(-ka_val * obs_t_v)
                    )
            else:
                c0 = (dose_mg * f_val) / vc_e
                pred = c0 * np.exp(-ke_e * obs_t_v)

            pred = np.clip(pred, _EPS, _MAX_CONCENTRATION_MG_L)
            residuals = log_obs - np.log(pred)
            return float(np.sum(residuals ** 2))

        x0 = np.array([initial_pk.clearance_l_h, initial_pk.volume_central_l])
        result = optimize.minimize(
            _objective, x0, method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
        )

        cl_opt = max(float(result.x[0]), _EPS)
        vc_opt = max(float(result.x[1]), _EPS)
        ke_opt = cl_opt / vc_opt
        hl_opt = math.log(2.0) / max(ke_opt, _EPS)

        optimised = PKParameters(
            clearance_l_h=cl_opt,
            volume_central_l=vc_opt,
            volume_peripheral_l=initial_pk.volume_peripheral_l,
            half_life_h=hl_opt,
            bioavailability=initial_pk.bioavailability,
            ka_h=initial_pk.ka_h,
            ke_h=ke_opt,
            q_l_h=initial_pk.q_l_h,
            infusion_duration_h=initial_pk.infusion_duration_h,
        )
        diagnostics: Dict[str, Any] = {
            "converged": bool(result.success),
            "objective_value": float(result.fun),
            "n_iterations": int(result.nit),
            "message": str(result.message),
            "cl_estimated": cl_opt,
            "vc_estimated": vc_opt,
            "ke_estimated": ke_opt,
        }

        self._run_count += 1
        self._audit_trail.append(
            _audit_log(
                "population_estimation", "engine",
                f"run_{self._run_count}",
                {"converged": result.success, "obj": float(result.fun)},
            )
        )
        logger.info(
            "Population estimation: CL=%.3f L/h, Vc=%.3f L, converged=%s",
            cl_opt, vc_opt, result.success,
        )
        return optimised, diagnostics

    # ------------------------------------------------------------------
    # Therapeutic window checking
    # ------------------------------------------------------------------
    def check_therapeutic_window(
        self,
        profile: ConcentrationProfile,
        target_trough_mg_l: float = 1.0,
        target_peak_mg_l: float = 20.0,
    ) -> Dict[str, Any]:
        """Evaluate whether a profile stays within the therapeutic window.

        RESEARCH USE ONLY.

        Args:
            profile: A computed ConcentrationProfile.
            target_trough_mg_l: Minimum effective concentration (mg/L).
            target_peak_mg_l: Maximum tolerated concentration (mg/L).

        Returns:
            Dictionary with within_window flag, time fractions, etc.
        """
        c = profile.concentrations_mg_l
        t = profile.time_points_h

        if len(c) == 0 or len(t) == 0:
            return {"within_window": False, "error": "empty profile"}

        above_trough = c >= target_trough_mg_l
        below_peak = c <= target_peak_mg_l
        within = above_trough & below_peak

        total_time = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        dt = np.diff(t)

        time_within = float(np.sum(dt[within[:-1]])) if len(dt) > 0 else 0.0
        time_below = float(np.sum(dt[~above_trough[:-1]])) if len(dt) > 0 else 0.0
        time_above = float(np.sum(dt[~below_peak[:-1]])) if len(dt) > 0 else 0.0
        frac_within = time_within / max(total_time, _EPS)

        return {
            "within_window": bool(frac_within > 0.8),
            "fraction_time_in_window": round(frac_within, 4),
            "time_within_h": round(time_within, 2),
            "time_below_trough_h": round(time_below, 2),
            "time_above_peak_h": round(time_above, 2),
            "target_trough_mg_l": target_trough_mg_l,
            "target_peak_mg_l": target_peak_mg_l,
            "observed_cmax_mg_l": round(float(profile.cmax_mg_l), 4),
            "observed_trough_mg_l": round(float(np.min(c)), 4),
        }

    # ------------------------------------------------------------------
    # Federated aggregation of site-level PK/PD statistics
    # ------------------------------------------------------------------
    def federated_aggregate_site_statistics(
        self,
        site_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate PK/PD summary statistics from multiple federated sites.

        RESEARCH USE ONLY.

        Each site summary is expected to contain:
            site_id, n_subjects, mean_clearance, var_clearance,
            mean_volume, var_volume, mean_auc, var_auc.

        Computes pooled weighted means and combined variances using
        standard federated averaging without sharing patient-level data.

        Args:
            site_summaries: List of per-site summary dictionaries.

        Returns:
            Dictionary of pooled population-level statistics.
        """
        if not site_summaries:
            logger.warning("No site summaries provided for aggregation")
            return {"error": "no sites", "pooled_n": 0}

        total_n = 0
        w_cl = 0.0
        w_vc = 0.0
        w_auc = 0.0
        pv_cl = 0.0
        pv_vc = 0.0
        pv_auc = 0.0
        site_hashes: List[str] = []

        for site in site_summaries:
            n = max(int(site.get("n_subjects", 0)), 0)
            if n == 0:
                continue
            total_n += n
            m_cl = float(site.get("mean_clearance", 0.0))
            v_cl = float(site.get("var_clearance", 0.0))
            m_vc = float(site.get("mean_volume", 0.0))
            v_vc = float(site.get("var_volume", 0.0))
            m_auc = float(site.get("mean_auc", 0.0))
            v_auc = float(site.get("var_auc", 0.0))

            w_cl += n * m_cl
            w_vc += n * m_vc
            w_auc += n * m_auc
            # Within-site variance component
            pv_cl += (n - 1) * v_cl
            pv_vc += (n - 1) * v_vc
            pv_auc += (n - 1) * v_auc

            sid = str(site.get("site_id", "unknown"))
            site_hashes.append(_hash_patient_id(sid)[:16])

        if total_n == 0:
            return {"error": "all sites zero subjects", "pooled_n": 0}

        g_cl = w_cl / total_n
        g_vc = w_vc / total_n
        g_auc = w_auc / total_n

        # Between-site variance component
        for site in site_summaries:
            n = max(int(site.get("n_subjects", 0)), 0)
            if n == 0:
                continue
            pv_cl += n * (float(site.get("mean_clearance", 0.0)) - g_cl) ** 2
            pv_vc += n * (float(site.get("mean_volume", 0.0)) - g_vc) ** 2
            pv_auc += n * (float(site.get("mean_auc", 0.0)) - g_auc) ** 2

        denom = max(total_n - 1, 1)
        pv_cl /= denom
        pv_vc /= denom
        pv_auc /= denom

        pooled: Dict[str, Any] = {
            "pooled_n": total_n,
            "n_sites": len(site_summaries),
            "mean_clearance_l_h": round(g_cl, 4),
            "var_clearance": round(pv_cl, 6),
            "cv_clearance_pct": round(
                100.0 * math.sqrt(max(pv_cl, 0.0)) / max(g_cl, _EPS), 2,
            ),
            "mean_volume_l": round(g_vc, 4),
            "var_volume": round(pv_vc, 6),
            "cv_volume_pct": round(
                100.0 * math.sqrt(max(pv_vc, 0.0)) / max(g_vc, _EPS), 2,
            ),
            "mean_auc_mg_h_l": round(g_auc, 4),
            "var_auc": round(pv_auc, 6),
            "site_ids_hashed": site_hashes,
        }

        self._audit_trail.append(
            _audit_log(
                "federated_aggregation", "engine", "population_stats",
                {"n_sites": len(site_summaries), "pooled_n": total_n},
            )
        )
        logger.info(
            "Federated aggregation: %d sites, N=%d, mean_CL=%.3f, mean_Vc=%.3f",
            len(site_summaries), total_n, g_cl, g_vc,
        )
        return pooled

    # ------------------------------------------------------------------
    # Population simulation with BSV
    # ------------------------------------------------------------------
    def simulate_population(
        self,
        dose_mg: float,
        n_subjects: int = 100,
        pk: Optional[PKParameters] = None,
        pd: Optional[PDParameters] = None,
        time_end_h: float = 48.0,
    ) -> Dict[str, Any]:
        """Simulate a virtual population with between-subject variability.

        RESEARCH USE ONLY.

        Adds log-normal random effects to CL and Vc:
            CL_i = CL_pop * exp(eta_CL),  eta_CL ~ N(0, omega_CL^2)

        Then summarises population-level exposure and effect metrics.

        Args:
            dose_mg: Dose administered to each virtual subject (mg).
            n_subjects: Number of virtual subjects.
            pk: Population-typical PK parameters.
            pd: PD parameters for effect computation.
            time_end_h: Simulation window (hours).

        Returns:
            Dictionary with population summary statistics.
        """
        pk = pk or PKParameters()
        pd = pd or PDParameters()
        n_subjects = max(n_subjects, 1)

        omega_cl = self.config.random_effects_variance.get("clearance", 0.09)
        omega_v = self.config.random_effects_variance.get("volume", 0.04)

        aucs: List[float] = []
        cmaxs: List[float] = []
        tmaxs: List[float] = []
        peak_effects: List[float] = []

        for _ in range(n_subjects):
            eta_cl = self._rng.normal(0.0, math.sqrt(max(omega_cl, 0.0)))
            eta_v = self._rng.normal(0.0, math.sqrt(max(omega_v, 0.0)))

            ind_pk = PKParameters(
                clearance_l_h=pk.clearance_l_h * math.exp(eta_cl),
                volume_central_l=pk.volume_central_l * math.exp(eta_v),
                volume_peripheral_l=pk.volume_peripheral_l * math.exp(eta_v),
                half_life_h=pk.half_life_h,
                bioavailability=pk.bioavailability,
                ka_h=pk.ka_h,
                ke_h=0.0,  # re-derived in __post_init__
                q_l_h=pk.q_l_h,
                infusion_duration_h=pk.infusion_duration_h,
            )

            if self.config.pk_model == PKModel.TWO_COMPARTMENT:
                prof = self.compute_two_compartment_profile(
                    dose_mg, pk=ind_pk, time_end_h=time_end_h, n_points=200,
                )
            else:
                prof = self.compute_one_compartment_profile(
                    dose_mg, pk=ind_pk, time_end_h=time_end_h, n_points=200,
                )

            aucs.append(prof.auc_mg_h_l)
            cmaxs.append(prof.cmax_mg_l)
            tmaxs.append(prof.tmax_h)
            eff = self.compute_pd_response(np.array([prof.cmax_mg_l]), pd=pd)
            peak_effects.append(float(eff[0]))

        arr_auc = np.array(aucs)
        arr_cmax = np.array(cmaxs)
        arr_eff = np.array(peak_effects)

        summary: Dict[str, Any] = {
            "n_subjects": n_subjects,
            "dose_mg": dose_mg,
            "auc_mean": round(float(np.mean(arr_auc)), 4),
            "auc_sd": round(float(np.std(arr_auc, ddof=1)), 4) if n_subjects > 1 else 0.0,
            "auc_cv_pct": round(
                100.0 * float(np.std(arr_auc, ddof=1)) / max(float(np.mean(arr_auc)), _EPS), 2,
            ) if n_subjects > 1 else 0.0,
            "cmax_mean": round(float(np.mean(arr_cmax)), 4),
            "cmax_sd": round(float(np.std(arr_cmax, ddof=1)), 4) if n_subjects > 1 else 0.0,
            "tmax_median": round(float(np.median(tmaxs)), 4),
            "effect_mean": round(float(np.mean(arr_eff)), 4),
            "effect_sd": round(float(np.std(arr_eff, ddof=1)), 4) if n_subjects > 1 else 0.0,
            "auc_percentiles": {
                "p5": round(float(np.percentile(arr_auc, 5)), 4),
                "p25": round(float(np.percentile(arr_auc, 25)), 4),
                "p50": round(float(np.percentile(arr_auc, 50)), 4),
                "p75": round(float(np.percentile(arr_auc, 75)), 4),
                "p95": round(float(np.percentile(arr_auc, 95)), 4),
            },
        }

        self._run_count += 1
        self._audit_trail.append(
            _audit_log(
                "population_simulation", "engine",
                f"run_{self._run_count}", {"n": n_subjects},
            )
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_cmax_tmax(
        c: np.ndarray, t: np.ndarray,
    ) -> Tuple[float, float]:
        """Extract Cmax and Tmax from concentration-time arrays."""
        if len(c) == 0:
            return 0.0, 0.0
        idx = int(np.argmax(c))
        return float(c[idx]), float(t[idx])

    @staticmethod
    def _estimate_terminal_half_life(
        c: np.ndarray, t: np.ndarray,
    ) -> float:
        """Estimate terminal half-life via log-linear regression on the tail.

        Uses the last 40 percent of the profile where concentrations are
        declining.  Returns 0 when estimation is not possible.
        """
        if len(c) < 4:
            return 0.0

        n_tail = max(int(len(c) * 0.4), 3)
        c_tail = c[-n_tail:]
        t_tail = t[-n_tail:]

        pos = c_tail > _EPS
        if np.sum(pos) < 2:
            return 0.0

        log_c = np.log(c_tail[pos])
        t_pos = t_tail[pos]

        # Linear regression: log(C) = intercept - ke * t
        n_pts = len(t_pos)
        s_t = np.sum(t_pos)
        s_lc = np.sum(log_c)
        s_t2 = np.sum(t_pos ** 2)
        s_tlc = np.sum(t_pos * log_c)

        denom = n_pts * s_t2 - s_t ** 2
        if abs(denom) < _EPS:
            return 0.0

        slope = (n_pts * s_tlc - s_t * s_lc) / denom
        if slope >= 0:
            return 0.0  # not declining

        ke_est = -slope
        hl = math.log(2.0) / max(ke_est, _EPS)
        return round(float(np.clip(hl, 0.0, _MAX_TIME_HOURS)), 4)

    def _find_therapeutic_window(
        self,
        doses: np.ndarray,
        effects: np.ndarray,
        emax: float,
        baseline: float,
    ) -> Tuple[float, float]:
        """Find doses at 20 percent and 80 percent of maximum effect."""
        if len(doses) == 0 or len(effects) == 0:
            return 0.0, 0.0

        eff_range = float(np.max(effects)) - baseline
        if eff_range <= _EPS:
            return float(doses[0]), float(doses[-1])

        tgt_lo = baseline + 0.2 * eff_range
        tgt_hi = baseline + 0.8 * eff_range

        tw_lo = self._interpolate_dose_for_effect(doses, effects, tgt_lo)
        tw_hi = self._interpolate_dose_for_effect(doses, effects, tgt_hi)
        return round(tw_lo, 4), round(tw_hi, 4)

    @staticmethod
    def _interpolate_dose_for_effect(
        doses: np.ndarray,
        effects: np.ndarray,
        target_effect: float,
    ) -> float:
        """Linear interpolation to find the dose producing a target effect."""
        if len(doses) < 2:
            return 0.0
        for i in range(len(effects) - 1):
            e1, e2 = effects[i], effects[i + 1]
            if (e1 <= target_effect <= e2) or (e2 <= target_effect <= e1):
                denom = e2 - e1
                if abs(denom) < _EPS:
                    return float(doses[i])
                frac = (target_effect - e1) / denom
                return float(doses[i] + frac * (doses[i + 1] - doses[i]))
        if target_effect <= effects[0]:
            return float(doses[0])
        return float(doses[-1])

    @staticmethod
    def _estimate_ec50_from_curve(
        concentrations: np.ndarray,
        effects: np.ndarray,
        baseline: float,
    ) -> float:
        """Estimate EC50 as the concentration at 50 percent of max effect."""
        if len(concentrations) < 2 or len(effects) < 2:
            return 0.0
        emax_obs = float(np.max(effects))
        target = baseline + 0.5 * (emax_obs - baseline)
        for i in range(len(effects) - 1):
            e1, e2 = effects[i], effects[i + 1]
            if (e1 <= target <= e2) or (e2 <= target <= e1):
                denom = e2 - e1
                if abs(denom) < _EPS:
                    return float(concentrations[i])
                frac = (target - e1) / denom
                return float(
                    concentrations[i]
                    + frac * (concentrations[i + 1] - concentrations[i])
                )
        return float(concentrations[len(concentrations) // 2])

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the full 21 CFR Part 11 audit trail."""
        return list(self._audit_trail)

    @property
    def run_count(self) -> int:
        """Total number of simulation / estimation runs performed."""
        return self._run_count

    def hash_patient_id(self, patient_id: str) -> str:
        """Public interface to HMAC-SHA256 patient ID hashing."""
        return _hash_patient_id(patient_id)
