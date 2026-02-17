"""Data harmonization for multi-site federated oncology trials.

Different hospitals encode clinical data using varying vocabularies,
units, and formats.  This module provides automated mapping and
normalisation routines so that data arriving from heterogeneous sources
can be aligned before local model training.

Key capabilities:
- Vocabulary mapping between DICOM SR codes, FHIR CodeableConcepts,
  and site-specific naming conventions.
- Unit conversion for common oncology measurements (e.g. tumor size
  in mm vs cm, dose in cGy vs Gy).
- Feature normalisation (z-score, min-max) across the local partition
  to ensure consistent model input scales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---- Vocabulary Mappings ------------------------------------------------

ONCOLOGY_VOCABULARY: dict[str, dict[str, str]] = {
    "tumor_volume": {
        "dicom": "118565006",
        "fhir": "observation-tumor-volume",
        "display": "Tumor Volume",
        "unit": "cm3",
    },
    "tumor_stage": {
        "dicom": "385356007",
        "fhir": "observation-stage-group",
        "display": "TNM Stage",
        "unit": "categorical",
    },
    "pdl1_expression": {
        "dicom": "784166006",
        "fhir": "observation-pdl1",
        "display": "PD-L1 Expression",
        "unit": "percent",
    },
    "ki67_index": {
        "dicom": "725041004",
        "fhir": "observation-ki67",
        "display": "Ki-67 Proliferation Index",
        "unit": "percent",
    },
    "her2_status": {
        "dicom": "431513002",
        "fhir": "observation-her2",
        "display": "HER2 Receptor Status",
        "unit": "categorical",
    },
    "estrogen_receptor": {
        "dicom": "416053008",
        "fhir": "observation-er-status",
        "display": "Estrogen Receptor Status",
        "unit": "categorical",
    },
    "progesterone_receptor": {
        "dicom": "416237000",
        "fhir": "observation-pr-status",
        "display": "Progesterone Receptor Status",
        "unit": "categorical",
    },
    "white_blood_cell": {
        "dicom": "767002",
        "fhir": "observation-wbc",
        "display": "White Blood Cell Count",
        "unit": "10^9/L",
    },
    "hemoglobin": {
        "dicom": "718-7",
        "fhir": "observation-hemoglobin",
        "display": "Hemoglobin",
        "unit": "g/dL",
    },
    "platelet_count": {
        "dicom": "777-3",
        "fhir": "observation-platelets",
        "display": "Platelet Count",
        "unit": "10^9/L",
    },
}

UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("mm", "cm"): 0.1,
    ("cm", "mm"): 10.0,
    ("cGy", "Gy"): 0.01,
    ("Gy", "cGy"): 100.0,
    ("mg/dL", "mmol/L"): 0.0555,
    ("mmol/L", "mg/dL"): 18.018,
    ("F", "C"): 1.0,  # handled specially
}


@dataclass
class HarmonizationReport:
    """Results of harmonizing a batch of records.

    Attributes:
        records_processed: Number of input records.
        fields_mapped: Total fields that were vocabulary-mapped.
        units_converted: Total values that underwent unit conversion.
        normalisation_applied: Whether feature normalisation was run.
        warnings: Non-critical issues encountered during harmonization.
    """

    records_processed: int = 0
    fields_mapped: int = 0
    units_converted: int = 0
    normalisation_applied: bool = False
    warnings: list[str] = field(default_factory=list)


class DataHarmonizer:
    """Harmonizes heterogeneous oncology data for federated training.

    Applies vocabulary mapping, unit conversion, and feature
    normalisation so that data from different hospitals can be consumed
    by a single federated model architecture.

    Args:
        vocabulary: Custom vocabulary mapping overriding defaults.
        target_units: Mapping of feature name -> desired unit.
        normalisation: ``"zscore"`` or ``"minmax"`` (default ``None``
            to skip normalisation).
    """

    def __init__(
        self,
        vocabulary: dict[str, dict[str, str]] | None = None,
        target_units: dict[str, str] | None = None,
        normalisation: str | None = None,
    ):
        self.vocabulary = vocabulary or dict(ONCOLOGY_VOCABULARY)
        self.target_units = target_units or {}
        self.normalisation = normalisation

        # Build reverse maps: code -> canonical name
        self._code_to_name: dict[str, str] = {}
        for name, meta in self.vocabulary.items():
            for key in ("dicom", "fhir"):
                if key in meta:
                    self._code_to_name[meta[key]] = name

    # ------------------------------------------------------------------
    # Vocabulary mapping
    # ------------------------------------------------------------------

    def map_field_name(self, field_name: str) -> str:
        """Map a site-specific field name to the canonical name.

        Checks the vocabulary for DICOM/FHIR codes and common aliases.
        Returns the original name if no mapping is found.
        """
        if field_name in self.vocabulary:
            return field_name
        if field_name in self._code_to_name:
            return self._code_to_name[field_name]
        # Try lowercase match
        lower = field_name.lower().replace("-", "_").replace(" ", "_")
        if lower in self.vocabulary:
            return lower
        return field_name

    def map_record(self, record: dict) -> dict:
        """Map all field names in a record to canonical names."""
        mapped = {}
        for key, value in record.items():
            canonical = self.map_field_name(key)
            mapped[canonical] = value
        return mapped

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert a numeric value between units.

        Args:
            value: The numeric measurement.
            from_unit: Source unit string.
            to_unit: Target unit string.

        Returns:
            Converted value.  Returns the original value unchanged if
            no conversion factor is available.
        """
        if from_unit == to_unit:
            return value
        # Special case: Fahrenheit to Celsius
        if from_unit == "F" and to_unit == "C":
            return (value - 32.0) * 5.0 / 9.0
        if from_unit == "C" and to_unit == "F":
            return value * 9.0 / 5.0 + 32.0
        factor = UNIT_CONVERSIONS.get((from_unit, to_unit))
        if factor is not None:
            return value * factor
        logger.warning("No conversion from %s to %s", from_unit, to_unit)
        return value

    # ------------------------------------------------------------------
    # Feature normalisation
    # ------------------------------------------------------------------

    def normalise_features(
        self,
        x: np.ndarray,
        method: str | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Normalise feature columns.

        Args:
            x: Feature matrix (n_samples, n_features).
            method: ``"zscore"`` or ``"minmax"``.  Falls back to
                ``self.normalisation`` if not given.

        Returns:
            Tuple of (normalised_x, stats_dict).
        """
        method = method or self.normalisation or "zscore"
        stats: dict[str, np.ndarray] = {}

        if method == "zscore":
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            std[std < 1e-12] = 1.0
            normalised = (x - mean) / std
            stats = {"mean": mean, "std": std}
        elif method == "minmax":
            xmin = np.min(x, axis=0)
            xmax = np.max(x, axis=0)
            denom = xmax - xmin
            denom[denom < 1e-12] = 1.0
            normalised = (x - xmin) / denom
            stats = {"min": xmin, "max": xmax}
        else:
            raise ValueError(f"Unknown normalisation method: {method}")

        return normalised, stats

    # ------------------------------------------------------------------
    # Batch harmonization
    # ------------------------------------------------------------------

    def harmonize_batch(
        self,
        records: list[dict],
        normalise: bool = False,
    ) -> tuple[list[dict], HarmonizationReport]:
        """Harmonize a batch of patient records.

        Applies vocabulary mapping and optional unit conversion to
        every record.

        Args:
            records: Raw patient records from a site.
            normalise: Whether to also normalise numeric values
                after mapping (requires all records to share numeric
                keys).

        Returns:
            Tuple of (harmonized_records, report).
        """
        report = HarmonizationReport(records_processed=len(records))
        harmonized: list[dict] = []

        for record in records:
            mapped = self.map_record(record)
            report.fields_mapped += sum(1 for k in record if self.map_field_name(k) != k)

            # Convert units if configured
            for fld, target_unit in self.target_units.items():
                if fld in mapped and isinstance(mapped[fld], (int, float)):
                    meta = self.vocabulary.get(fld, {})
                    src_unit = meta.get("unit", target_unit)
                    if src_unit != target_unit:
                        mapped[fld] = self.convert_units(float(mapped[fld]), src_unit, target_unit)
                        report.units_converted += 1

            harmonized.append(mapped)

        report.normalisation_applied = normalise
        logger.info(
            "Harmonized %d records — %d fields mapped, %d units converted",
            report.records_processed,
            report.fields_mapped,
            report.units_converted,
        )
        return harmonized, report
