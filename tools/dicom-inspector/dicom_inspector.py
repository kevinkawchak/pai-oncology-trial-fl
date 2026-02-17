"""DICOM medical image inspection tool for oncology clinical trials.

Provides command-line utilities for inspecting, validating, and summarizing
DICOM files used in federated learning pipelines for radiation oncology.
Supports CT, MR, PET, and ultrasound modality images commonly used in
treatment planning and response assessment.

DISCLAIMER: RESEARCH USE ONLY — This tool is intended for research and
educational purposes. It is NOT approved for clinical diagnostic use. All
outputs must be reviewed by qualified medical professionals before any
clinical decision-making.

VERSION: 0.4.0
LICENSE: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports — pydicom is optional so the tool can still be imported
# in environments that lack the library (e.g., CI without imaging deps).
# ---------------------------------------------------------------------------
try:
    import pydicom
    from pydicom.errors import InvalidDicomError

    HAS_PYDICOM = True
except ImportError:  # pragma: no cover
    HAS_PYDICOM = False
    pydicom = None  # type: ignore[assignment]
    InvalidDicomError = Exception  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# DICOM tag reference — common tags inspected for oncology imaging QA.
# Source: DICOM PS3.6 2024b — Data Dictionary, NEMA.
# ---------------------------------------------------------------------------
REQUIRED_ONCOLOGY_TAGS: dict[str, str] = {
    "PatientID": "0010,0020",
    "PatientName": "0010,0010",
    "Modality": "0008,0060",
    "StudyDate": "0008,0020",
    "StudyDescription": "0008,1030",
    "SeriesDescription": "0008,103E",
    "Manufacturer": "0008,0070",
    "SliceThickness": "0018,0050",
    "PixelSpacing": "0028,0030",
    "ImagePositionPatient": "0020,0032",
    "ImageOrientationPatient": "0020,0037",
    "Rows": "0028,0010",
    "Columns": "0028,0011",
    "BitsAllocated": "0028,0100",
}

# Oncology-specific validation limits.
# Source: ACR–AAPM Technical Standard for Medical Physics, 2020.
SLICE_THICKNESS_BOUNDS_MM: dict[str, tuple[float, float]] = {
    "CT": (0.5, 10.0),
    "MR": (0.5, 10.0),
    "PT": (1.0, 5.0),
}

PIXEL_SPACING_BOUNDS_MM: tuple[float, float] = (0.1, 5.0)


# ── Enums ──────────────────────────────────────────────────────────────────


class DicomModality(str, Enum):
    """Supported DICOM imaging modalities for oncology trials.

    Values follow the standard DICOM Modality code definitions
    (DICOM PS3.3, Section C.7.3.1.1.1).
    """

    CT = "CT"
    MR = "MR"
    PT = "PT"
    US = "US"
    RTDOSE = "RTDOSE"
    RTSTRUCT = "RTSTRUCT"
    RTPLAN = "RTPLAN"
    RTIMAGE = "RTIMAGE"
    DX = "DX"
    CR = "CR"
    NM = "NM"
    XA = "XA"
    MG = "MG"
    OTHER = "OTHER"

    @classmethod
    def from_string(cls, value: str) -> "DicomModality":
        """Resolve a modality string to the corresponding enum member."""
        try:
            return cls(value.upper())
        except ValueError:
            logger.warning("Unrecognised modality '%s'; mapping to OTHER.", value)
            return cls.OTHER


class ValidationSeverity(str, Enum):
    """Severity level for DICOM validation findings.

    Used to classify issues found during automated QA checks.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class ValidationFinding:
    """Single finding from a DICOM validation pass.

    Attributes:
        tag_name: Human-readable DICOM tag name.
        severity: Severity classification of the finding.
        message: Description of the issue.
        expected: Expected value or range (if applicable).
        actual: Actual value found in the file.
    """

    tag_name: str
    severity: ValidationSeverity
    message: str
    expected: str = ""
    actual: str = ""


@dataclass
class DicomInspectionResult:
    """Aggregated result of inspecting one or more DICOM files.

    Attributes:
        file_path: Path to the inspected DICOM file.
        modality: Detected imaging modality.
        patient_id: De-identified patient identifier.
        study_date: Study date string (YYYYMMDD).
        dimensions: Image dimensions as (rows, cols).
        pixel_spacing_mm: In-plane pixel spacing in millimetres.
        slice_thickness_mm: Slice thickness in millimetres.
        manufacturer: Scanner manufacturer string.
        tags_present: List of required tags that are present.
        tags_missing: List of required tags that are absent.
        validation_findings: Detailed validation findings.
        is_valid: Overall pass / fail flag.
        metadata: Additional key-value metadata extracted.
    """

    file_path: str = ""
    modality: str = ""
    patient_id: str = ""
    study_date: str = ""
    dimensions: tuple[int, int] = (0, 0)
    pixel_spacing_mm: tuple[float, float] = (0.0, 0.0)
    slice_thickness_mm: float = 0.0
    manufacturer: str = ""
    tags_present: list[str] = field(default_factory=list)
    tags_missing: list[str] = field(default_factory=list)
    validation_findings: list[dict[str, str]] = field(default_factory=list)
    is_valid: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DicomSummary:
    """Summary statistics for a collection of DICOM files.

    Attributes:
        total_files: Number of files processed.
        modalities: Mapping of modality to count.
        patients: Unique patient identifiers found.
        valid_count: Files that passed validation.
        invalid_count: Files that failed validation.
        findings_by_severity: Count of findings grouped by severity.
    """

    total_files: int = 0
    modalities: dict[str, int] = field(default_factory=dict)
    patients: list[str] = field(default_factory=list)
    valid_count: int = 0
    invalid_count: int = 0
    findings_by_severity: dict[str, int] = field(default_factory=dict)


# ── Core helpers ───────────────────────────────────────────────────────────


def _require_pydicom() -> None:
    """Raise a clear error when pydicom is not installed."""
    if not HAS_PYDICOM:
        raise RuntimeError(
            "pydicom is required for DICOM inspection but is not installed. Install it with: pip install pydicom"
        )


def _safe_getattr(ds: Any, attr: str, default: Any = "") -> Any:
    """Retrieve a DICOM attribute with a fallback default."""
    try:
        return getattr(ds, attr, default)
    except Exception:
        return default


def inspect_dicom_file(file_path: str | Path) -> DicomInspectionResult:
    """Inspect a single DICOM file and return structured metadata.

    Parameters
    ----------
    file_path:
        Path to a DICOM (.dcm) file on disk.

    Returns
    -------
    DicomInspectionResult:
        Populated inspection result dataclass.
    """
    _require_pydicom()
    path = Path(file_path)
    result = DicomInspectionResult(file_path=str(path))

    if not path.exists():
        result.validation_findings.append(
            asdict(ValidationFinding("File", ValidationSeverity.ERROR, f"File not found: {path}"))
        )
        return result

    try:
        ds = pydicom.dcmread(str(path), force=True)
    except (InvalidDicomError, Exception) as exc:
        result.validation_findings.append(
            asdict(ValidationFinding("File", ValidationSeverity.ERROR, f"Cannot read DICOM: {exc}"))
        )
        return result

    # Extract core attributes.
    result.modality = str(_safe_getattr(ds, "Modality", "UNKNOWN"))
    result.patient_id = str(_safe_getattr(ds, "PatientID", ""))
    result.study_date = str(_safe_getattr(ds, "StudyDate", ""))
    result.manufacturer = str(_safe_getattr(ds, "Manufacturer", ""))

    rows = int(_safe_getattr(ds, "Rows", 0) or 0)
    cols = int(_safe_getattr(ds, "Columns", 0) or 0)
    result.dimensions = (rows, cols)

    ps = _safe_getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) == 2:
        result.pixel_spacing_mm = (float(ps[0]), float(ps[1]))

    st = _safe_getattr(ds, "SliceThickness", None)
    if st is not None:
        result.slice_thickness_mm = float(st)

    # Check required tags.
    for tag_name in REQUIRED_ONCOLOGY_TAGS:
        if hasattr(ds, tag_name) and _safe_getattr(ds, tag_name) not in (None, ""):
            result.tags_present.append(tag_name)
        else:
            result.tags_missing.append(tag_name)
            result.validation_findings.append(
                asdict(
                    ValidationFinding(tag_name, ValidationSeverity.WARNING, f"Required tag '{tag_name}' is missing.")
                )
            )

    result.is_valid = (
        len([f for f in result.validation_findings if f.get("severity") == ValidationSeverity.ERROR.value]) == 0
    )

    logger.info("Inspected %s — modality=%s valid=%s", path.name, result.modality, result.is_valid)
    return result


def validate_dicom_file(file_path: str | Path) -> DicomInspectionResult:
    """Run extended validation on a DICOM file.

    In addition to the basic inspection, this checks that numeric
    parameters fall within accepted oncology imaging bounds.

    Parameters
    ----------
    file_path:
        Path to a DICOM (.dcm) file.

    Returns
    -------
    DicomInspectionResult:
        Inspection result with additional validation findings.
    """
    result = inspect_dicom_file(file_path)

    # Validate slice thickness bounds.
    modality_key = result.modality.upper()
    if modality_key in SLICE_THICKNESS_BOUNDS_MM and result.slice_thickness_mm > 0:
        lo, hi = SLICE_THICKNESS_BOUNDS_MM[modality_key]
        if not (lo <= result.slice_thickness_mm <= hi):
            result.validation_findings.append(
                asdict(
                    ValidationFinding(
                        "SliceThickness",
                        ValidationSeverity.WARNING,
                        f"Slice thickness {result.slice_thickness_mm:.2f} mm outside recommended range [{lo}, {hi}] mm.",
                        expected=f"[{lo}, {hi}]",
                        actual=f"{result.slice_thickness_mm:.2f}",
                    )
                )
            )

    # Validate pixel spacing bounds.
    lo_ps, hi_ps = PIXEL_SPACING_BOUNDS_MM
    for idx, ps_val in enumerate(result.pixel_spacing_mm):
        if ps_val > 0 and not (lo_ps <= ps_val <= hi_ps):
            result.validation_findings.append(
                asdict(
                    ValidationFinding(
                        "PixelSpacing",
                        ValidationSeverity.WARNING,
                        f"Pixel spacing[{idx}] {ps_val:.3f} mm outside bounds [{lo_ps}, {hi_ps}] mm.",
                        expected=f"[{lo_ps}, {hi_ps}]",
                        actual=f"{ps_val:.3f}",
                    )
                )
            )

    error_count = len([f for f in result.validation_findings if f.get("severity") == ValidationSeverity.ERROR.value])
    result.is_valid = error_count == 0
    return result


def summarize_dicom_directory(directory: str | Path) -> DicomSummary:
    """Summarize all DICOM files found under *directory*.

    Parameters
    ----------
    directory:
        Root directory to scan for ``*.dcm`` files.

    Returns
    -------
    DicomSummary:
        Aggregate statistics across all discovered DICOM files.
    """
    _require_pydicom()
    root = Path(directory)
    summary = DicomSummary()
    patients_set: set[str] = set()

    dcm_files = list(root.rglob("*.dcm"))
    summary.total_files = len(dcm_files)
    logger.info("Found %d DICOM files under %s", summary.total_files, root)

    for dcm_path in dcm_files:
        result = inspect_dicom_file(dcm_path)
        modality = result.modality or "UNKNOWN"
        summary.modalities[modality] = summary.modalities.get(modality, 0) + 1
        if result.patient_id:
            patients_set.add(result.patient_id)
        if result.is_valid:
            summary.valid_count += 1
        else:
            summary.invalid_count += 1
        for finding in result.validation_findings:
            sev = finding.get("severity", "info")
            summary.findings_by_severity[sev] = summary.findings_by_severity.get(sev, 0) + 1

    summary.patients = sorted(patients_set)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dicom_inspector",
        description=("DICOM medical image inspection tool for oncology clinical trials. RESEARCH USE ONLY."),
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

    # inspect ---
    sp_inspect = subparsers.add_parser("inspect", help="Inspect a single DICOM file.")
    sp_inspect.add_argument("file", type=str, help="Path to a DICOM (.dcm) file.")

    # validate ---
    sp_validate = subparsers.add_parser("validate", help="Validate a DICOM file against oncology QA standards.")
    sp_validate.add_argument("file", type=str, help="Path to a DICOM (.dcm) file.")

    # summarize ---
    sp_summarize = subparsers.add_parser("summarize", help="Summarize DICOM files in a directory.")
    sp_summarize.add_argument("directory", type=str, help="Root directory to scan for .dcm files.")

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
    """Entry-point for the DICOM inspector CLI.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int:
        Exit code — 0 on success, 1 on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "inspect":
            result = inspect_dicom_file(args.file)
            _output(result, args.json)
            return 0 if result.is_valid else 1

        if args.command == "validate":
            result = validate_dicom_file(args.file)
            _output(result, args.json)
            return 0 if result.is_valid else 1

        if args.command == "summarize":
            summary = summarize_dicom_directory(args.directory)
            _output(summary, args.json)
            return 0
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
