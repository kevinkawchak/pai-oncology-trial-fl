"""Radiation and chemotherapy dose calculator for oncology clinical trials.

Computes and validates dose parameters for radiation therapy and
chemotherapy regimens used in federated oncology research. Implements
standard dose-limiting constraints and provides bounded calculations
suitable for treatment planning verification in simulation environments.

DISCLAIMER: RESEARCH USE ONLY — This tool is intended for research and
educational purposes. It is NOT approved for clinical use. All dose
calculations must be independently verified by qualified medical
physicists and oncologists before any treatment decision.

VERSION: 0.4.0
LICENSE: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference constants — all values cite their authoritative source.
# ---------------------------------------------------------------------------

# Maximum total radiation doses by site (Gy).
# Source: NCCN Clinical Practice Guidelines in Oncology, v2.2024.
MAX_TOTAL_DOSE_GY: dict[str, float] = {
    "brain": 60.0,
    "head_and_neck": 70.0,
    "lung": 66.0,
    "breast": 50.0,
    "prostate": 79.2,
    "cervix": 85.0,
    "rectum": 50.4,
    "esophagus": 50.4,
    "pancreas": 54.0,
    "liver": 60.0,
}

# Standard fraction sizes (Gy per fraction).
# Source: RTOG Protocol reference tables, Radiation Therapy Oncology Group.
STANDARD_FRACTION_SIZES_GY: dict[str, tuple[float, float]] = {
    "conventional": (1.8, 2.0),
    "hypofractionated": (2.5, 15.0),
    "sbrt": (6.0, 34.0),
    "srs": (12.0, 24.0),
}

# Organ-at-risk dose constraints (Gy).
# Source: Emami et al. 1991 (TD 5/5) + QUANTEC updates (2010).
OAR_CONSTRAINTS_GY: dict[str, dict[str, float]] = {
    "spinal_cord": {"max_dose": 50.0, "td_5_5": 50.0},
    "brainstem": {"max_dose": 54.0, "td_5_5": 60.0},
    "optic_nerve": {"max_dose": 55.0, "td_5_5": 55.0},
    "optic_chiasm": {"max_dose": 55.0, "td_5_5": 56.0},
    "lung_mean": {"mean_dose": 20.0, "v20_pct": 37.0},
    "heart_mean": {"mean_dose": 26.0, "v25_pct": 10.0},
    "liver_mean": {"mean_dose": 30.0, "td_5_5": 30.0},
    "kidney_mean": {"mean_dose": 18.0, "td_5_5": 23.0},
    "parotid_mean": {"mean_dose": 26.0, "td_5_5": 32.0},
}

# Alpha/beta ratios for the linear-quadratic model.
# Source: Fowler JF. The linear-quadratic formula and progress in
# fractionated radiotherapy. Br J Radiol. 1989;62(740):679-694.
ALPHA_BETA_RATIOS: dict[str, float] = {
    "tumour_high": 10.0,
    "tumour_low": 3.0,
    "late_responding_tissue": 3.0,
    "acute_responding_tissue": 10.0,
    "prostate_tumour": 1.5,
    "breast_tumour": 4.0,
    "melanoma": 0.6,
}

# Chemotherapy agent dose ranges (mg/m^2 unless stated).
# Source: NCCN Chemotherapy Order Templates, v2024.
CHEMO_DOSE_RANGES: dict[str, dict[str, Any]] = {
    "cisplatin": {"min_dose": 20.0, "max_dose": 100.0, "unit": "mg/m2", "cycle_days": [1], "source": "NCCN v2024"},
    "carboplatin": {
        "min_dose": 200.0,
        "max_dose": 800.0,
        "unit": "mg (AUC-based)",
        "cycle_days": [1],
        "source": "NCCN v2024 (AUC 4-7)",
    },
    "paclitaxel": {"min_dose": 80.0, "max_dose": 225.0, "unit": "mg/m2", "cycle_days": [1], "source": "NCCN v2024"},
    "docetaxel": {"min_dose": 60.0, "max_dose": 100.0, "unit": "mg/m2", "cycle_days": [1], "source": "NCCN v2024"},
    "doxorubicin": {"min_dose": 40.0, "max_dose": 75.0, "unit": "mg/m2", "cycle_days": [1], "source": "NCCN v2024"},
    "fluorouracil": {
        "min_dose": 200.0,
        "max_dose": 1000.0,
        "unit": "mg/m2",
        "cycle_days": [1, 2, 3, 4, 5],
        "source": "NCCN v2024",
    },
    "gemcitabine": {
        "min_dose": 800.0,
        "max_dose": 1250.0,
        "unit": "mg/m2",
        "cycle_days": [1, 8],
        "source": "NCCN v2024",
    },
    "temozolomide": {
        "min_dose": 75.0,
        "max_dose": 200.0,
        "unit": "mg/m2",
        "cycle_days": list(range(1, 6)),
        "source": "NCCN v2024 (Stupp protocol)",
    },
}

# BSA bounds (m^2) — physiologically plausible range.
# Source: Mosteller RD. Simplified Calculation of Body-Surface Area.
# N Engl J Med. 1987;317(17):1098.
BSA_BOUNDS_M2: tuple[float, float] = (0.5, 3.0)

# Absolute dose bounds to prevent runaway calculations.
RADIATION_DOSE_ABS_MIN_GY: float = 0.0
RADIATION_DOSE_ABS_MAX_GY: float = 100.0
CHEMO_DOSE_ABS_MAX_MG_M2: float = 2000.0


# ── Enums ──────────────────────────────────────────────────────────────────


class DoseUnit(str, Enum):
    """Units of measure for radiation and chemotherapy dosing.

    Follows the conventions of ICRU Reports 50 and 83.
    """

    GY = "Gy"
    CGY = "cGy"
    MG_M2 = "mg/m2"
    MG = "mg"
    MG_KG = "mg/kg"
    AUC = "AUC"


class ChemoAgent(str, Enum):
    """Chemotherapy agents supported by the calculator.

    Agents are limited to those with well-established dose-response
    curves published in NCCN Clinical Practice Guidelines.
    """

    CISPLATIN = "cisplatin"
    CARBOPLATIN = "carboplatin"
    PACLITAXEL = "paclitaxel"
    DOCETAXEL = "docetaxel"
    DOXORUBICIN = "doxorubicin"
    FLUOROURACIL = "fluorouracil"
    GEMCITABINE = "gemcitabine"
    TEMOZOLOMIDE = "temozolomide"


class RadiationTechnique(str, Enum):
    """Radiation delivery techniques.

    Source: ASTRO/ACR practice parameters for radiation oncology, 2023.
    """

    THREE_D_CRT = "3D-CRT"
    IMRT = "IMRT"
    VMAT = "VMAT"
    SBRT = "SBRT"
    SRS = "SRS"
    PROTON = "proton"
    ELECTRON = "electron"
    BRACHYTHERAPY = "brachytherapy"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class PatientParameters:
    """Patient-specific parameters for dose computation.

    All numerics are bounded to physiologically plausible ranges:
    weight 20-300 kg, height 50-250 cm, BSA 0.5-3.0 m^2, CrCl 10-200 mL/min.
    """

    patient_id: str = ""
    weight_kg: float = 70.0
    height_cm: float = 170.0
    bsa_m2: float = 1.7
    creatinine_clearance_ml_min: float = 100.0
    tumour_site: str = "lung"

    def __post_init__(self) -> None:
        """Clamp values to safe physiological bounds."""
        self.weight_kg = max(20.0, min(self.weight_kg, 300.0))
        self.height_cm = max(50.0, min(self.height_cm, 250.0))
        self.bsa_m2 = max(BSA_BOUNDS_M2[0], min(self.bsa_m2, BSA_BOUNDS_M2[1]))
        self.creatinine_clearance_ml_min = max(10.0, min(self.creatinine_clearance_ml_min, 200.0))


@dataclass
class DoseCalculation:
    """Result of a single dose calculation (radiation or chemotherapy).

    Includes BED/EQD2 for radiation, bounded total dose, and reference citations.
    """

    calculation_type: str = ""
    agent_or_technique: str = ""
    prescribed_dose: float = 0.0
    dose_unit: str = DoseUnit.GY.value
    fractions: int = 0
    dose_per_fraction: float = 0.0
    total_dose: float = 0.0
    bed: float = 0.0
    eqd2: float = 0.0
    is_within_bounds: bool = False
    warnings: list[str] = field(default_factory=list)
    reference: str = ""


# ── Core calculation helpers ───────────────────────────────────────────────


def compute_bed(total_dose_gy: float, dose_per_fraction_gy: float, alpha_beta: float) -> float:
    """Compute biologically effective dose: BED = D * (1 + d / (a/b)).

    Source: Fowler JF, Br J Radiol 1989. All inputs bounded.
    """
    total_dose_gy = max(RADIATION_DOSE_ABS_MIN_GY, min(total_dose_gy, RADIATION_DOSE_ABS_MAX_GY))
    if dose_per_fraction_gy <= 0 or alpha_beta <= 0:
        return 0.0
    return total_dose_gy * (1.0 + dose_per_fraction_gy / alpha_beta)


def compute_eqd2(total_dose_gy: float, dose_per_fraction_gy: float, alpha_beta: float) -> float:
    """Compute EQD2 = BED / (1 + 2/(a/b)). Source: ICRU Report 83."""
    bed = compute_bed(total_dose_gy, dose_per_fraction_gy, alpha_beta)
    if alpha_beta <= 0:
        return 0.0
    return bed / (1.0 + 2.0 / alpha_beta)


def calculate_radiation_dose(
    tumour_site: str,
    technique: str,
    dose_per_fraction_gy: float,
    num_fractions: int,
    alpha_beta: float = 10.0,
) -> DoseCalculation:
    """Calculate and validate a radiation dose prescription.

    All parameters are bounded: dose_per_fraction 0.1-34 Gy,
    fractions 1-50, alpha_beta 0.1-20.
    """
    dose_per_fraction_gy = max(0.1, min(dose_per_fraction_gy, 34.0))
    num_fractions = max(1, min(num_fractions, 50))
    alpha_beta = max(0.1, min(alpha_beta, 20.0))

    total = dose_per_fraction_gy * num_fractions
    total = min(total, RADIATION_DOSE_ABS_MAX_GY)
    bed = compute_bed(total, dose_per_fraction_gy, alpha_beta)
    eqd2 = compute_eqd2(total, dose_per_fraction_gy, alpha_beta)

    result = DoseCalculation(
        calculation_type="radiation",
        agent_or_technique=technique,
        prescribed_dose=dose_per_fraction_gy,
        dose_unit=DoseUnit.GY.value,
        fractions=num_fractions,
        dose_per_fraction=dose_per_fraction_gy,
        total_dose=round(total, 2),
        bed=round(bed, 2),
        eqd2=round(eqd2, 2),
        reference="NCCN Clinical Practice Guidelines v2024; RTOG protocols; QUANTEC 2010",
    )

    site_key = tumour_site.lower().replace(" ", "_")
    max_dose = MAX_TOTAL_DOSE_GY.get(site_key)
    if max_dose is not None and total > max_dose:
        result.warnings.append(
            f"Total dose {total:.1f} Gy exceeds max for {site_key} ({max_dose} Gy). Source: NCCN v2024."
        )
    result.is_within_bounds = len(result.warnings) == 0
    return result


def calculate_chemotherapy_dose(
    agent_name: str,
    patient: PatientParameters,
    dose_mg_m2: float | None = None,
    target_auc: float | None = None,
) -> DoseCalculation:
    """Calculate and validate a chemotherapy dose.

    Standard agents use ``dose_mg_m2 * BSA``. Carboplatin uses the
    Calvert formula: Dose (mg) = AUC * (CrCl + 25).
    Source: Calvert AH, et al. J Clin Oncol. 1989;7(11):1748-1756.
    """
    agent_key = agent_name.lower()
    agent_info = CHEMO_DOSE_RANGES.get(agent_key)

    if agent_info is None:
        calc = DoseCalculation(calculation_type="chemotherapy", agent_or_technique=agent_name)
        calc.warnings.append(f"Unknown agent '{agent_name}'. Supported: {list(CHEMO_DOSE_RANGES.keys())}.")
        return calc

    # Carboplatin AUC-based dosing.
    if agent_key == "carboplatin" and target_auc is not None:
        target_auc = max(1.0, min(target_auc, 10.0))
        absolute_dose_mg = target_auc * (patient.creatinine_clearance_ml_min + 25.0)
        absolute_dose_mg = max(0.0, min(absolute_dose_mg, float(agent_info["max_dose"])))
        calc = DoseCalculation(
            calculation_type="chemotherapy",
            agent_or_technique=agent_name,
            prescribed_dose=round(absolute_dose_mg, 1),
            dose_unit=DoseUnit.MG.value,
            total_dose=round(absolute_dose_mg, 1),
            reference=f"Calvert formula (AUC {target_auc}); {agent_info['source']}",
        )
        calc.is_within_bounds = absolute_dose_mg <= float(agent_info["max_dose"])
        if not calc.is_within_bounds:
            calc.warnings.append(
                f"Calculated dose {absolute_dose_mg:.1f} mg exceeds upper bound "
                f"{agent_info['max_dose']} mg. Source: {agent_info['source']}."
            )
        return calc

    # Standard mg/m^2 dosing.
    min_d = float(agent_info["min_dose"])
    max_d = float(agent_info["max_dose"])
    if dose_mg_m2 is None:
        dose_mg_m2 = (min_d + max_d) / 2.0
    dose_mg_m2 = max(0.0, min(dose_mg_m2, CHEMO_DOSE_ABS_MAX_MG_M2))

    absolute_dose = dose_mg_m2 * patient.bsa_m2
    within = min_d <= dose_mg_m2 <= max_d

    calc = DoseCalculation(
        calculation_type="chemotherapy",
        agent_or_technique=agent_name,
        prescribed_dose=round(dose_mg_m2, 1),
        dose_unit=DoseUnit.MG_M2.value,
        total_dose=round(absolute_dose, 1),
        is_within_bounds=within,
        reference=agent_info["source"],
    )
    if not within:
        calc.warnings.append(
            f"Dose {dose_mg_m2:.1f} mg/m2 outside range [{min_d}, {max_d}] mg/m2. Source: {agent_info['source']}."
        )
    return calc


def validate_oar_dose(organ: str, dose_gy: float) -> DoseCalculation:
    """Validate an organ-at-risk dose against QUANTEC constraints (bounded 0-100 Gy)."""
    dose_gy = max(RADIATION_DOSE_ABS_MIN_GY, min(dose_gy, RADIATION_DOSE_ABS_MAX_GY))
    organ_key = organ.lower().replace(" ", "_")
    constraints = OAR_CONSTRAINTS_GY.get(organ_key)

    calc = DoseCalculation(
        calculation_type="oar_validation",
        agent_or_technique=organ,
        prescribed_dose=round(dose_gy, 2),
        dose_unit=DoseUnit.GY.value,
        total_dose=round(dose_gy, 2),
        reference="QUANTEC 2010; Emami et al. 1991",
    )

    if constraints is None:
        calc.warnings.append(f"No OAR constraints found for '{organ}'. Available: {list(OAR_CONSTRAINTS_GY.keys())}.")
        return calc

    max_allowed = constraints.get("max_dose") or constraints.get("mean_dose", math.inf)
    calc.is_within_bounds = dose_gy <= max_allowed
    if not calc.is_within_bounds:
        calc.warnings.append(
            f"Dose {dose_gy:.1f} Gy exceeds constraint {max_allowed:.1f} Gy for {organ_key}. Source: QUANTEC 2010."
        )
    return calc


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dose_calculator",
        description=("Radiation & chemotherapy dose calculator for oncology trials. RESEARCH USE ONLY."),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.4.0")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands.")

    # radiation ---
    sp_rad = subparsers.add_parser("radiation", help="Calculate a radiation dose prescription.")
    sp_rad.add_argument("--site", required=True, help="Tumour site (e.g. lung, prostate).")
    sp_rad.add_argument("--technique", default="IMRT", help="Radiation technique (default: IMRT).")
    sp_rad.add_argument("--dose-per-fraction", type=float, required=True, help="Dose per fraction in Gy.")
    sp_rad.add_argument("--fractions", type=int, required=True, help="Number of fractions.")
    sp_rad.add_argument("--alpha-beta", type=float, default=10.0, help="Alpha/beta ratio (default: 10).")

    # chemotherapy ---
    sp_chemo = subparsers.add_parser("chemotherapy", help="Calculate a chemotherapy dose.")
    sp_chemo.add_argument("--agent", required=True, help="Chemotherapy agent name.")
    sp_chemo.add_argument("--bsa", type=float, default=1.7, help="Body surface area in m^2.")
    sp_chemo.add_argument("--dose", type=float, default=None, help="Dose in mg/m^2 (optional).")
    sp_chemo.add_argument("--auc", type=float, default=None, help="Target AUC for carboplatin.")
    sp_chemo.add_argument("--crcl", type=float, default=100.0, help="Creatinine clearance mL/min.")

    # validate ---
    sp_val = subparsers.add_parser("validate", help="Validate an organ-at-risk dose.")
    sp_val.add_argument("--organ", required=True, help="Organ-at-risk name.")
    sp_val.add_argument("--dose", type=float, required=True, help="Observed dose in Gy.")

    return parser


def _output(data: Any, as_json: bool) -> None:
    """Print *data* as JSON or human-readable text."""
    if as_json:
        print(json.dumps(asdict(data) if hasattr(data, "__dataclass_fields__") else data, indent=2, default=str))
    else:
        if hasattr(data, "__dataclass_fields__"):
            for k, v in asdict(data).items():
                print(f"  {k}: {v}")
        else:
            print(data)


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the dose calculator CLI. Returns 0 on success, 1 on failure."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "radiation":
            result = calculate_radiation_dose(
                tumour_site=args.site,
                technique=args.technique,
                dose_per_fraction_gy=args.dose_per_fraction,
                num_fractions=args.fractions,
                alpha_beta=args.alpha_beta,
            )
            _output(result, args.json)
            return 0 if result.is_within_bounds else 1

        if args.command == "chemotherapy":
            patient = PatientParameters(bsa_m2=args.bsa, creatinine_clearance_ml_min=args.crcl)
            result = calculate_chemotherapy_dose(
                agent_name=args.agent,
                patient=patient,
                dose_mg_m2=args.dose,
                target_auc=args.auc,
            )
            _output(result, args.json)
            return 0 if result.is_within_bounds else 1

        if args.command == "validate":
            result = validate_oar_dose(organ=args.organ, dose_gy=args.dose)
            _output(result, args.json)
            return 0 if result.is_within_bounds else 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
