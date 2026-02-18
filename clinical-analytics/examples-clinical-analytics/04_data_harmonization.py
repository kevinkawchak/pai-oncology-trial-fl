#!/usr/bin/env python3
"""Data Harmonization and Interoperability for Multi-Site Oncology Trials.

CLINICAL CONTEXT
================
Federated multi-site oncology trials must reconcile heterogeneous data
representations across institutions. Differences in coding systems (ICD-10
vs SNOMED CT), schema conventions, variable naming, and unit systems
create significant barriers to pooled analysis. This example demonstrates
a comprehensive data harmonization pipeline including terminology mapping,
schema alignment, dataset harmonization, cross-site reporting, and CDISC
SDTM export.

Proper data harmonization is mandated by ICH E6(R3) for multi-centre
trials and is essential for regulatory submissions requiring CDISC
standards (FDA, PMDA, NMPA).

USE CASES COVERED
=================
1. Mapping ICD-10 diagnosis codes to SNOMED CT concept identifiers
   using a curated oncology-specific crosswalk table.
2. Aligning heterogeneous schemas between two trial sites by detecting
   column name mismatches, type conflicts, and missing variables.
3. Harmonizing a synthetic multi-site dataset by applying terminology
   mapping, unit conversions, and schema normalization.
4. Generating a cross-site data alignment report with completeness
   metrics, concordance rates, and discrepancy flags.
5. Exporting harmonized data to CDISC SDTM domain format (DM, LB, AE)
   with required variables and controlled terminology.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0   (https://numpy.org)

REFERENCES
==========
- CDISC SDTM Implementation Guide v3.4
- CDISC Controlled Terminology (2025-12-19)
- ICD-10-CM 2026 (WHO / CMS)
- SNOMED CT International Edition 2025-07-01
- ICH E6(R3) Good Clinical Practice

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

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.clinical_analytics.example_04")


# ============================================================================
# Terminology Crosswalk: ICD-10 <-> SNOMED CT
# ============================================================================

ICD10_TO_SNOMED: dict[str, dict[str, str]] = {
    "C34.1": {"snomed": "254637007", "display": "Non-small cell lung cancer"},
    "C34.9": {"snomed": "93880001", "display": "Primary malignant neoplasm of lung"},
    "C50.9": {"snomed": "254838004", "display": "Carcinoma of breast"},
    "C18.9": {"snomed": "363406005", "display": "Malignant neoplasm of colon"},
    "C61":   {"snomed": "399068003", "display": "Malignant neoplasm of prostate"},
    "C43.9": {"snomed": "372244006", "display": "Malignant melanoma"},
    "C64.9": {"snomed": "93849006", "display": "Renal cell carcinoma"},
    "C25.9": {"snomed": "363418001", "display": "Malignant neoplasm of pancreas"},
    "C56.9": {"snomed": "363443007", "display": "Malignant neoplasm of ovary"},
    "C71.9": {"snomed": "428061005", "display": "Malignant neoplasm of brain"},
    "C16.9": {"snomed": "363349007", "display": "Malignant neoplasm of stomach"},
    "C22.0": {"snomed": "95214007", "display": "Hepatocellular carcinoma"},
}


def map_icd10_to_snomed(icd10_code: str) -> dict[str, str]:
    """Map an ICD-10-CM code to its SNOMED CT equivalent.

    Args:
        icd10_code: ICD-10-CM diagnosis code (e.g., 'C34.1').

    Returns:
        Dictionary with 'icd10', 'snomed', 'display', and 'mapped' flag.
    """
    entry = ICD10_TO_SNOMED.get(icd10_code)
    if entry:
        return {
            "icd10": icd10_code,
            "snomed": entry["snomed"],
            "display": entry["display"],
            "mapped": "true",
        }
    return {
        "icd10": icd10_code,
        "snomed": "UNMAPPED",
        "display": f"No SNOMED mapping for {icd10_code}",
        "mapped": "false",
    }


# ============================================================================
# Schema Alignment
# ============================================================================


@dataclass
class SiteSchema:
    """Schema definition for a trial site dataset.

    Attributes:
        site_id: Unique site identifier.
        columns: Mapping of column name to data type string.
        units: Mapping of column name to measurement unit.
    """

    site_id: str = ""
    columns: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)


@dataclass
class SchemaAlignmentReport:
    """Report of schema alignment between two sites.

    Attributes:
        site_a: First site ID.
        site_b: Second site ID.
        matched_columns: Columns present in both with compatible types.
        mismatched_types: Columns with type conflicts.
        only_in_a: Columns only in site A.
        only_in_b: Columns only in site B.
        unit_conflicts: Columns with different measurement units.
        alignment_score: Fraction of columns successfully aligned [0, 1].
    """

    site_a: str = ""
    site_b: str = ""
    matched_columns: list[str] = field(default_factory=list)
    mismatched_types: list[dict[str, str]] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    unit_conflicts: list[dict[str, str]] = field(default_factory=list)
    alignment_score: float = 0.0


def align_schemas(schema_a: SiteSchema, schema_b: SiteSchema) -> SchemaAlignmentReport:
    """Align schemas from two trial sites and identify discrepancies.

    Args:
        schema_a: Schema from site A.
        schema_b: Schema from site B.

    Returns:
        SchemaAlignmentReport with alignment details.
    """
    cols_a = set(schema_a.columns.keys())
    cols_b = set(schema_b.columns.keys())
    common = cols_a & cols_b
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    matched: list[str] = []
    mismatched: list[dict[str, str]] = []
    unit_conflicts: list[dict[str, str]] = []

    for col in sorted(common):
        type_a = schema_a.columns[col]
        type_b = schema_b.columns[col]
        if type_a == type_b:
            matched.append(col)
        else:
            mismatched.append({"column": col, "type_a": type_a, "type_b": type_b})
            # Still consider it partially matched
            matched.append(col)

        # Check units
        unit_a = schema_a.units.get(col, "")
        unit_b = schema_b.units.get(col, "")
        if unit_a and unit_b and unit_a != unit_b:
            unit_conflicts.append({"column": col, "unit_a": unit_a, "unit_b": unit_b})

    total_cols = len(cols_a | cols_b)
    alignment_score = len(matched) / total_cols if total_cols > 0 else 0.0

    report = SchemaAlignmentReport(
        site_a=schema_a.site_id,
        site_b=schema_b.site_id,
        matched_columns=matched,
        mismatched_types=mismatched,
        only_in_a=only_a,
        only_in_b=only_b,
        unit_conflicts=unit_conflicts,
        alignment_score=round(alignment_score, 4),
    )

    logger.info(
        "Schema alignment | %s vs %s | matched=%d, type_mismatches=%d, "
        "only_a=%d, only_b=%d, score=%.2f",
        schema_a.site_id, schema_b.site_id, len(matched),
        len(mismatched), len(only_a), len(only_b), alignment_score,
    )
    return report


# ============================================================================
# Dataset Harmonization
# ============================================================================


# Unit conversion factors to standard units
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("lb", "kg"): 0.453592,
    ("kg", "kg"): 1.0,
    ("in", "cm"): 2.54,
    ("cm", "cm"): 1.0,
    ("g/L", "g/dL"): 0.1,
    ("g/dL", "g/dL"): 1.0,
    ("umol/L", "mg/dL"): 0.0113,   # creatinine
    ("mg/dL", "mg/dL"): 1.0,
    ("mmol/L", "mg/dL"): 18.0,     # glucose
    ("U/L", "U/L"): 1.0,
    ("10^9/L", "K/uL"): 1.0,
    ("K/uL", "K/uL"): 1.0,
}

STANDARD_UNITS: dict[str, str] = {
    "weight": "kg",
    "height": "cm",
    "albumin": "g/dL",
    "creatinine": "mg/dL",
    "glucose": "mg/dL",
    "alt": "U/L",
    "wbc": "K/uL",
    "hemoglobin": "g/dL",
}


def harmonize_dataset(
    records: list[dict[str, Any]],
    source_units: dict[str, str],
) -> list[dict[str, Any]]:
    """Harmonize a dataset by applying terminology mapping and unit conversions.

    Args:
        records: List of patient record dictionaries.
        source_units: Mapping of variable name to source unit.

    Returns:
        List of harmonized records with standardised units and codes.
    """
    harmonized: list[dict[str, Any]] = []

    for record in records:
        h_record = dict(record)

        # Map ICD-10 to SNOMED if diagnosis code present
        if "diagnosis_icd10" in h_record:
            mapping = map_icd10_to_snomed(h_record["diagnosis_icd10"])
            h_record["diagnosis_snomed"] = mapping["snomed"]
            h_record["diagnosis_display"] = mapping["display"]
            h_record["diagnosis_mapped"] = mapping["mapped"]

        # Apply unit conversions
        for var_name, src_unit in source_units.items():
            if var_name in h_record and isinstance(h_record[var_name], (int, float)):
                std_unit = STANDARD_UNITS.get(var_name, src_unit)
                conv_key = (src_unit, std_unit)
                factor = UNIT_CONVERSIONS.get(conv_key, 1.0)
                h_record[var_name] = round(float(h_record[var_name]) * factor, 4)
                h_record[f"{var_name}_unit"] = std_unit

        harmonized.append(h_record)

    logger.info("Harmonized %d records", len(harmonized))
    return harmonized


# ============================================================================
# Cross-Site Alignment Report
# ============================================================================


def generate_alignment_report(
    site_a_records: list[dict[str, Any]],
    site_b_records: list[dict[str, Any]],
    key_variables: list[str],
) -> dict[str, Any]:
    """Generate a cross-site data alignment report.

    Assesses completeness, value distributions, and concordance for
    key variables across two site datasets.

    Args:
        site_a_records: Records from site A.
        site_b_records: Records from site B.
        key_variables: Variables to assess for alignment.

    Returns:
        Alignment report dictionary.
    """
    report: dict[str, Any] = {
        "site_a_n": len(site_a_records),
        "site_b_n": len(site_b_records),
        "variables_assessed": len(key_variables),
        "variable_reports": {},
    }

    for var in key_variables:
        vals_a = [r[var] for r in site_a_records if var in r and r[var] is not None]
        vals_b = [r[var] for r in site_b_records if var in r and r[var] is not None]

        completeness_a = len(vals_a) / max(len(site_a_records), 1)
        completeness_b = len(vals_b) / max(len(site_b_records), 1)

        var_report: dict[str, Any] = {
            "completeness_a": round(completeness_a, 4),
            "completeness_b": round(completeness_b, 4),
        }

        # Numeric variable statistics
        if vals_a and isinstance(vals_a[0], (int, float)):
            arr_a = np.array(vals_a, dtype=np.float64)
            arr_b = np.array(vals_b, dtype=np.float64) if vals_b else np.array([])
            var_report["site_a_mean"] = round(float(np.mean(arr_a)), 3)
            var_report["site_a_std"] = round(float(np.std(arr_a, ddof=1)), 3) if len(arr_a) > 1 else 0.0
            if len(arr_b) > 0:
                var_report["site_b_mean"] = round(float(np.mean(arr_b)), 3)
                var_report["site_b_std"] = round(float(np.std(arr_b, ddof=1)), 3) if len(arr_b) > 1 else 0.0
                # Standardised mean difference (Cohen's d)
                pooled_std = float(np.sqrt(
                    ((len(arr_a) - 1) * np.var(arr_a, ddof=1) +
                     (len(arr_b) - 1) * np.var(arr_b, ddof=1)) /
                    max(len(arr_a) + len(arr_b) - 2, 1)
                ))
                smd = abs(float(np.mean(arr_a)) - float(np.mean(arr_b))) / max(pooled_std, 1e-9)
                var_report["standardised_mean_diff"] = round(smd, 4)
                var_report["aligned"] = smd < 0.1  # <0.1 = negligible difference
        else:
            # Categorical variable
            unique_a = set(str(v) for v in vals_a)
            unique_b = set(str(v) for v in vals_b)
            overlap = unique_a & unique_b
            concordance = len(overlap) / max(len(unique_a | unique_b), 1)
            var_report["unique_a"] = len(unique_a)
            var_report["unique_b"] = len(unique_b)
            var_report["overlap"] = len(overlap)
            var_report["concordance"] = round(concordance, 4)
            var_report["aligned"] = concordance >= 0.8

        report["variable_reports"][var] = var_report

    n_aligned = sum(
        1 for vr in report["variable_reports"].values() if vr.get("aligned", False)
    )
    report["overall_alignment"] = round(n_aligned / max(len(key_variables), 1), 4)

    logger.info(
        "Alignment report | vars=%d | aligned=%d | overall=%.2f",
        len(key_variables), n_aligned, report["overall_alignment"],
    )
    return report


# ============================================================================
# CDISC SDTM Export
# ============================================================================


@dataclass
class SDTMDomain:
    """A CDISC SDTM domain representation.

    Attributes:
        domain: Two-letter SDTM domain code (e.g., 'DM', 'LB', 'AE').
        records: List of SDTM-compliant record dictionaries.
        metadata: Domain-level metadata.
    """

    domain: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def export_to_sdtm(
    harmonized_records: list[dict[str, Any]],
    study_id: str = "NCT-PAI-2026-001",
) -> dict[str, SDTMDomain]:
    """Export harmonized records to CDISC SDTM domain format.

    Generates DM (Demographics), LB (Laboratory), and AE (Adverse Events)
    domains with required SDTM variables and controlled terminology.

    Args:
        harmonized_records: Harmonized patient records.
        study_id: Clinical study identifier.

    Returns:
        Dictionary mapping domain code to SDTMDomain object.
    """
    domains: dict[str, SDTMDomain] = {}

    # --- DM (Demographics) Domain ---
    dm_records: list[dict[str, Any]] = []
    for rec in harmonized_records:
        dm_rec = {
            "STUDYID": study_id,
            "DOMAIN": "DM",
            "USUBJID": f"{study_id}-{rec.get('patient_id', 'UNKNOWN')}",
            "SUBJID": rec.get("patient_id", ""),
            "SITEID": rec.get("site_id", ""),
            "AGE": rec.get("age", None),
            "AGEU": "YEARS",
            "SEX": rec.get("sex", ""),
            "RACE": rec.get("race", "UNKNOWN"),
            "ARM": rec.get("treatment_arm", ""),
            "ARMCD": rec.get("treatment_arm", "")[:8].upper().replace(" ", ""),
            "ACTARM": rec.get("treatment_arm", ""),
            "COUNTRY": rec.get("country", "USA"),
        }
        dm_records.append(dm_rec)

    domains["DM"] = SDTMDomain(
        domain="DM",
        records=dm_records,
        metadata={
            "description": "Demographics",
            "structure": "One record per subject",
            "standard": "CDISC SDTM v3.4",
        },
    )

    # --- LB (Laboratory) Domain ---
    lab_mapping = {
        "hemoglobin": ("HGB", "Hemoglobin", "g/dL"),
        "wbc": ("WBC", "Leukocytes", "K/uL"),
        "albumin": ("ALB", "Albumin", "g/dL"),
        "creatinine": ("CREAT", "Creatinine", "mg/dL"),
        "alt": ("ALT", "Alanine Aminotransferase", "U/L"),
    }

    lb_records: list[dict[str, Any]] = []
    seq = 0
    for rec in harmonized_records:
        for var_name, (testcd, test_display, unit) in lab_mapping.items():
            if var_name in rec and rec[var_name] is not None:
                seq += 1
                lb_rec = {
                    "STUDYID": study_id,
                    "DOMAIN": "LB",
                    "USUBJID": f"{study_id}-{rec.get('patient_id', 'UNKNOWN')}",
                    "LBSEQ": seq,
                    "LBTESTCD": testcd,
                    "LBTEST": test_display,
                    "LBORRES": str(rec[var_name]),
                    "LBORRESU": unit,
                    "LBSTRESN": float(rec[var_name]),
                    "LBSTRESU": unit,
                    "LBSTAT": "",
                    "LBNRIND": "",  # Normal range indicator would be set by reference ranges
                }
                lb_records.append(lb_rec)

    domains["LB"] = SDTMDomain(
        domain="LB",
        records=lb_records,
        metadata={
            "description": "Laboratory Test Results",
            "structure": "One record per subject per test per visit",
            "standard": "CDISC SDTM v3.4",
        },
    )

    # --- AE (Adverse Events) Domain ---
    ae_records: list[dict[str, Any]] = []
    seq = 0
    for rec in harmonized_records:
        if rec.get("ae_term"):
            seq += 1
            ae_rec = {
                "STUDYID": study_id,
                "DOMAIN": "AE",
                "USUBJID": f"{study_id}-{rec.get('patient_id', 'UNKNOWN')}",
                "AESEQ": seq,
                "AETERM": rec.get("ae_term", ""),
                "AEDECOD": rec.get("ae_preferred_term", rec.get("ae_term", "")),
                "AEBODSYS": rec.get("ae_soc", ""),
                "AESEV": rec.get("ae_severity", ""),
                "AESER": "Y" if rec.get("ae_serious", False) else "N",
                "AEREL": rec.get("ae_relationship", "NOT RELATED"),
                "AETOXGR": str(rec.get("ae_grade", "")),
            }
            ae_records.append(ae_rec)

    domains["AE"] = SDTMDomain(
        domain="AE",
        records=ae_records,
        metadata={
            "description": "Adverse Events",
            "structure": "One record per subject per adverse event",
            "standard": "CDISC SDTM v3.4",
        },
    )

    logger.info(
        "SDTM export | DM=%d, LB=%d, AE=%d records",
        len(dm_records), len(lb_records), len(ae_records),
    )
    return domains


# ============================================================================
# Synthetic Data Helpers
# ============================================================================


def _generate_site_data(
    site_id: str,
    n: int,
    seed: int,
    use_imperial: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Generate synthetic site data with optional imperial units.

    Args:
        site_id: Site identifier.
        n: Number of records.
        seed: RNG seed.
        use_imperial: If True, use pounds/inches instead of kg/cm.

    Returns:
        Tuple of (records, source_units).
    """
    rng = np.random.default_rng(seed)
    diagnoses = list(ICD10_TO_SNOMED.keys())
    ae_terms = [
        "Nausea", "Fatigue", "Neutropenia", "Thrombocytopenia",
        "Diarrhea", "Rash", "Peripheral neuropathy", "Anemia",
    ]

    records: list[dict[str, Any]] = []
    for i in range(n):
        weight = float(np.clip(rng.normal(75, 15), 40.0, 160.0))
        height = float(np.clip(rng.normal(170, 10), 140.0, 210.0))
        if use_imperial:
            weight = weight * 2.20462  # kg -> lb
            height = height / 2.54     # cm -> in

        rec: dict[str, Any] = {
            "patient_id": f"PAT-{site_id}-{i + 1:04d}",
            "site_id": site_id,
            "age": int(np.clip(rng.normal(62, 11), 18, 95)),
            "sex": str(rng.choice(["M", "F"])),
            "weight": round(weight, 1),
            "height": round(height, 1),
            "diagnosis_icd10": str(rng.choice(diagnoses)),
            "treatment_arm": str(rng.choice(["Control", "Experimental"])),
            "hemoglobin": round(float(np.clip(rng.normal(12.5, 2.0), 7.0, 18.0)), 1),
            "wbc": round(float(np.clip(rng.normal(7.0, 3.0), 1.0, 30.0)), 1),
            "albumin": round(float(np.clip(rng.normal(3.8, 0.6), 1.5, 5.5)), 1),
            "creatinine": round(float(np.clip(rng.normal(1.0, 0.4), 0.3, 5.0)), 2),
            "alt": round(float(np.clip(rng.normal(30, 15), 5.0, 300.0)), 0),
            "ae_term": str(rng.choice(ae_terms)) if rng.random() > 0.3 else "",
            "ae_grade": int(rng.choice([1, 2, 3, 4], p=[0.40, 0.30, 0.20, 0.10])),
            "ae_serious": bool(rng.random() < 0.15),
            "country": "USA",
        }
        records.append(rec)

    units = {
        "weight": "lb" if use_imperial else "kg",
        "height": "in" if use_imperial else "cm",
        "hemoglobin": "g/dL",
        "wbc": "K/uL",
        "albumin": "g/dL",
        "creatinine": "mg/dL",
        "alt": "U/L",
    }

    return records, units


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
    """Run the data harmonization example."""
    logger.info("Starting Example 04: Data Harmonization")

    print("\n" + "=" * 72)
    print("  DATA HARMONIZATION -- MULTI-SITE ONCOLOGY TRIAL")
    print("=" * 72)

    # --- 1. ICD-10 to SNOMED mapping ---
    _print_section("1. ICD-10 TO SNOMED CT TERMINOLOGY MAPPING")
    test_codes = ["C34.1", "C50.9", "C18.9", "C61", "C99.9"]
    print(f"  {'ICD-10':>10s}  {'SNOMED':>12s}  {'Mapped':>8s}  {'Display'}")
    print(f"  {'-' * 65}")
    for code in test_codes:
        result = map_icd10_to_snomed(code)
        print(f"  {result['icd10']:>10s}  {result['snomed']:>12s}  "
              f"{result['mapped']:>8s}  {result['display']}")
    mapped_count = sum(1 for c in test_codes if map_icd10_to_snomed(c)["mapped"] == "true")
    print(f"\n  Mapping rate: {mapped_count}/{len(test_codes)} "
          f"({100*mapped_count/len(test_codes):.0f}%)")

    # --- 2. Schema alignment ---
    _print_section("2. CROSS-SITE SCHEMA ALIGNMENT")
    schema_a = SiteSchema(
        site_id="SITE-01",
        columns={
            "patient_id": "str", "age": "int", "sex": "str",
            "weight": "float", "height": "float", "hemoglobin": "float",
            "wbc": "float", "albumin": "float", "creatinine": "float",
            "alt": "float", "diagnosis_icd10": "str", "treatment_arm": "str",
        },
        units={"weight": "kg", "height": "cm", "hemoglobin": "g/dL",
               "albumin": "g/dL", "creatinine": "mg/dL"},
    )
    schema_b = SiteSchema(
        site_id="SITE-02",
        columns={
            "patient_id": "str", "age": "float", "sex": "str",  # type mismatch: int vs float
            "weight": "float", "height": "float", "hemoglobin": "float",
            "wbc": "float", "albumin": "float",
            "diagnosis_icd10": "str", "treatment_arm": "str",
            "ecog_ps": "int",  # only in site B
        },
        units={"weight": "lb", "height": "in", "hemoglobin": "g/dL",
               "albumin": "g/dL"},
    )

    alignment = align_schemas(schema_a, schema_b)
    print(f"  Sites: {alignment.site_a} vs {alignment.site_b}")
    print(f"  Matched columns:   {len(alignment.matched_columns)}")
    print(f"  Type mismatches:   {len(alignment.mismatched_types)}")
    for m in alignment.mismatched_types:
        print(f"    - {m['column']}: {m['type_a']} vs {m['type_b']}")
    print(f"  Only in {alignment.site_a}: {alignment.only_in_a}")
    print(f"  Only in {alignment.site_b}: {alignment.only_in_b}")
    print(f"  Unit conflicts:    {len(alignment.unit_conflicts)}")
    for uc in alignment.unit_conflicts:
        print(f"    - {uc['column']}: {uc['unit_a']} vs {uc['unit_b']}")
    print(f"  Alignment score:   {alignment.alignment_score:.2%}")

    # --- 3. Dataset harmonization ---
    _print_section("3. SYNTHETIC DATASET HARMONIZATION")
    site_a_data, units_a = _generate_site_data("SITE-01", 80, seed=42, use_imperial=False)
    site_b_data, units_b = _generate_site_data("SITE-02", 60, seed=99, use_imperial=True)

    print(f"  Site A: {len(site_a_data)} records (metric units)")
    print(f"  Site B: {len(site_b_data)} records (imperial units)")

    harmonized_a = harmonize_dataset(site_a_data, units_a)
    harmonized_b = harmonize_dataset(site_b_data, units_b)

    # Show conversion example
    print("\n  Unit conversion example (Site B, first patient):")
    orig = site_b_data[0]
    harm = harmonized_b[0]
    print(f"    Weight: {orig['weight']:.1f} lb -> {harm['weight']:.1f} kg")
    print(f"    Height: {orig['height']:.1f} in -> {harm['height']:.1f} cm")
    print(f"    Diagnosis: {orig['diagnosis_icd10']} -> SNOMED {harm.get('diagnosis_snomed', 'N/A')}")
    print(f"      ({harm.get('diagnosis_display', 'N/A')})")

    # --- 4. Cross-site alignment report ---
    _print_section("4. CROSS-SITE ALIGNMENT REPORT")
    key_vars = ["age", "weight", "hemoglobin", "albumin", "creatinine", "treatment_arm"]
    report = generate_alignment_report(harmonized_a, harmonized_b, key_vars)

    print(f"  Site A: {report['site_a_n']} records | Site B: {report['site_b_n']} records")
    print(f"  Variables assessed: {report['variables_assessed']}")
    print(f"\n  {'Variable':>20s}  {'Compl A':>9s}  {'Compl B':>9s}  "
          f"{'SMD/Conc':>10s}  {'Aligned':>8s}")
    print(f"  {'-' * 62}")

    for var, vr in report["variable_reports"].items():
        metric = vr.get("standardised_mean_diff", vr.get("concordance", "N/A"))
        metric_str = f"{metric:.4f}" if isinstance(metric, float) else str(metric)
        aligned_str = "YES" if vr.get("aligned", False) else "NO"
        print(f"  {var:>20s}  {vr['completeness_a']:9.2%}  {vr['completeness_b']:9.2%}  "
              f"{metric_str:>10s}  {aligned_str:>8s}")

    print(f"\n  Overall alignment score: {report['overall_alignment']:.2%}")

    # --- 5. CDISC SDTM export ---
    _print_section("5. CDISC SDTM EXPORT")
    all_harmonized = harmonized_a + harmonized_b
    sdtm_domains = export_to_sdtm(all_harmonized)

    for domain_code, domain in sdtm_domains.items():
        print(f"\n  Domain: {domain_code} ({domain.metadata.get('description', '')})")
        print(f"    Standard:  {domain.metadata.get('standard', '')}")
        print(f"    Structure: {domain.metadata.get('structure', '')}")
        print(f"    Records:   {len(domain.records)}")
        if domain.records:
            print(f"    Variables:  {', '.join(domain.records[0].keys())}")

    print("\n" + "=" * 72)
    print("  DATA HARMONIZATION COMPLETE")
    print("  RESEARCH USE ONLY -- NOT FOR CLINICAL DECISION MAKING")
    print("=" * 72 + "\n")

    logger.info("Example 04 complete")


if __name__ == "__main__":
    main()
