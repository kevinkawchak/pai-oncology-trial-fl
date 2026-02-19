"""Advanced PHI/PII detection engine for federated oncology trial data.

Implements comprehensive detection of all 18 HIPAA Safe Harbor identifiers
per 45 CFR 164.514(b)(2), with optional DICOM tag scanning and Presidio
integration. Designed for pre-ingestion screening of clinical trial data
before it enters federated learning pipelines.

References:
    - 45 CFR 164.514(b)(2): Safe Harbor de-identification standard
    - 45 CFR 164.512: Uses and disclosures for research
    - NIST SP 800-188: De-Identifying Government Datasets
    - HIPAA Privacy Rule (45 CFR Parts 160, 164)

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils.log_sanitizer import sanitize_log_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PHICategory(str, Enum):
    """All 18 HIPAA Safe Harbor identifiers per 45 CFR 164.514(b)(2).

    Each value maps to a specific category of protected health information
    that must be removed or transformed before data qualifies as
    de-identified under the Safe Harbor method.
    """

    # 1. Names
    NAMES = "names"
    # 2. Geographic data smaller than state
    GEOGRAPHIC = "geographic"
    # 3. Dates (except year) related to an individual
    DATES = "dates"
    # 4. Telephone numbers
    PHONE_NUMBERS = "phone_numbers"
    # 5. Fax numbers
    FAX_NUMBERS = "fax_numbers"
    # 6. Email addresses
    EMAIL_ADDRESSES = "email_addresses"
    # 7. Social Security numbers
    SSN = "ssn"
    # 8. Medical record numbers
    MRN = "mrn"
    # 9. Health plan beneficiary numbers
    HEALTH_PLAN_ID = "health_plan_id"
    # 10. Account numbers
    ACCOUNT_NUMBERS = "account_numbers"
    # 11. Certificate/license numbers
    CERTIFICATE_LICENSE = "certificate_license"
    # 12. Vehicle identifiers and serial numbers
    VEHICLE_ID = "vehicle_id"
    # 13. Device identifiers and serial numbers
    DEVICE_ID = "device_id"
    # 14. Web Universal Resource Locators (URLs)
    URLS = "urls"
    # 15. Internet Protocol (IP) addresses
    IP_ADDRESSES = "ip_addresses"
    # 16. Biometric identifiers
    BIOMETRIC_ID = "biometric_id"
    # 17. Full-face photographs and comparable images
    PHOTOGRAPHS = "photographs"
    # 18. Any other unique identifying number, characteristic, or code
    OTHER_UNIQUE_ID = "other_unique_id"


class RiskLevel(str, Enum):
    """Risk classification for detected PHI/PII elements."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    """Method used to detect PHI."""

    REGEX = "regex"
    FIELD_NAME = "field_name"
    DICOM_TAG = "dicom_tag"
    PRESIDIO = "presidio"
    HEURISTIC = "heuristic"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PHIFinding:
    """A single detected PHI element.

    Attributes:
        phi_type: The HIPAA identifier category.
        location: Where the finding was detected (field name, offset, tag).
        value_preview: Redacted preview (first 2 chars + '***') — never the full value.
        confidence: Detection confidence in [0.0, 1.0].
        risk_level: Assessed risk if this PHI were disclosed.
        detection_method: How the PHI was detected.
        context: Additional context (e.g., surrounding field name).
    """

    phi_type: PHICategory
    location: str
    value_preview: str
    confidence: float
    risk_level: RiskLevel
    detection_method: DetectionMethod = DetectionMethod.REGEX
    context: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ScanResult:
    """Aggregated result of a PHI scan across one or more data items.

    Attributes:
        total_items_scanned: Number of records/texts scanned.
        total_findings: Number of PHI findings detected.
        findings: List of individual PHI findings.
        risk_summary: Count of findings by risk level.
        category_summary: Count of findings by PHI category.
        scan_metadata: Additional metadata about the scan.
    """

    total_items_scanned: int = 0
    total_findings: int = 0
    findings: list[PHIFinding] = field(default_factory=list)
    risk_summary: dict[str, int] = field(default_factory=dict)
    category_summary: dict[str, int] = field(default_factory=dict)
    scan_metadata: dict[str, Any] = field(default_factory=dict)

    def add_finding(self, finding: PHIFinding) -> None:
        """Add a finding and update summaries."""
        self.findings.append(finding)
        self.total_findings += 1
        risk_key = finding.risk_level.value
        self.risk_summary[risk_key] = self.risk_summary.get(risk_key, 0) + 1
        cat_key = finding.phi_type.value
        self.category_summary[cat_key] = self.category_summary.get(cat_key, 0) + 1

    @property
    def has_critical(self) -> bool:
        """Whether any critical-risk findings were detected."""
        return self.risk_summary.get("critical", 0) > 0

    @property
    def has_high(self) -> bool:
        """Whether any high-risk findings were detected."""
        return self.risk_summary.get("high", 0) > 0


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

_CATEGORY_RISK: dict[PHICategory, RiskLevel] = {
    PHICategory.SSN: RiskLevel.CRITICAL,
    PHICategory.MRN: RiskLevel.CRITICAL,
    PHICategory.HEALTH_PLAN_ID: RiskLevel.CRITICAL,
    PHICategory.ACCOUNT_NUMBERS: RiskLevel.HIGH,
    PHICategory.NAMES: RiskLevel.HIGH,
    PHICategory.EMAIL_ADDRESSES: RiskLevel.HIGH,
    PHICategory.PHONE_NUMBERS: RiskLevel.HIGH,
    PHICategory.FAX_NUMBERS: RiskLevel.HIGH,
    PHICategory.DATES: RiskLevel.MEDIUM,
    PHICategory.GEOGRAPHIC: RiskLevel.MEDIUM,
    PHICategory.IP_ADDRESSES: RiskLevel.MEDIUM,
    PHICategory.URLS: RiskLevel.MEDIUM,
    PHICategory.CERTIFICATE_LICENSE: RiskLevel.HIGH,
    PHICategory.VEHICLE_ID: RiskLevel.MEDIUM,
    PHICategory.DEVICE_ID: RiskLevel.MEDIUM,
    PHICategory.BIOMETRIC_ID: RiskLevel.CRITICAL,
    PHICategory.PHOTOGRAPHS: RiskLevel.HIGH,
    PHICategory.OTHER_UNIQUE_ID: RiskLevel.MEDIUM,
}


def _redact_preview(value: str) -> str:
    """Create a redacted preview: show first 2 chars + '***'.

    Never logs or stores the full PHI value.
    """
    if len(value) <= 2:
        return "***"
    return value[:2] + "***"


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: dict[PHICategory, list[re.Pattern[str]]] = {
    PHICategory.SSN: [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        re.compile(r"\b\d{9}\b"),  # Unformatted SSN
    ],
    PHICategory.MRN: [
        re.compile(r"\bMRN[:\s#]*\d{6,10}\b", re.IGNORECASE),
        re.compile(r"\bmedical[\s_-]?record[\s_-]?(?:number|num|no|#)[:\s]*\d{6,10}\b", re.IGNORECASE),
    ],
    PHICategory.PHONE_NUMBERS: [
        re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ],
    PHICategory.FAX_NUMBERS: [
        re.compile(r"\bfax[:\s]*(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", re.IGNORECASE),
    ],
    PHICategory.EMAIL_ADDRESSES: [
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ],
    PHICategory.DATES: [
        re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
        re.compile(r"\b(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
            r"\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ],
    PHICategory.GEOGRAPHIC: [
        # ZIP codes — 5 or 9 digit
        re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    ],
    PHICategory.IP_ADDRESSES: [
        re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),  # IPv6
    ],
    PHICategory.URLS: [
        re.compile(r"https?://[^\s<>\"]+"),
    ],
    PHICategory.ACCOUNT_NUMBERS: [
        re.compile(r"\baccount[\s_-]?(?:number|num|no|#)[:\s]*\d{8,12}\b", re.IGNORECASE),
    ],
    PHICategory.CERTIFICATE_LICENSE: [
        re.compile(r"\b(?:license|licence|cert(?:ificate)?)\s*#?\s*[:]\s*[A-Z0-9]{6,15}\b", re.IGNORECASE),
    ],
    PHICategory.VEHICLE_ID: [
        # VIN: 17 alphanumeric characters (excluding I, O, Q)
        re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    ],
    PHICategory.HEALTH_PLAN_ID: [
        re.compile(r"\bhealth[\s_-]?plan[\s_-]?(?:id|number|num|no|#)[:\s]*\w{6,15}\b", re.IGNORECASE),
    ],
}

# Field names that indicate PHI in structured data
_PHI_FIELD_NAMES: dict[str, PHICategory] = {
    "name": PHICategory.NAMES,
    "first_name": PHICategory.NAMES,
    "last_name": PHICategory.NAMES,
    "patient_name": PHICategory.NAMES,
    "full_name": PHICategory.NAMES,
    "address": PHICategory.GEOGRAPHIC,
    "street": PHICategory.GEOGRAPHIC,
    "street_address": PHICategory.GEOGRAPHIC,
    "city": PHICategory.GEOGRAPHIC,
    "zip": PHICategory.GEOGRAPHIC,
    "zipcode": PHICategory.GEOGRAPHIC,
    "zip_code": PHICategory.GEOGRAPHIC,
    "postal_code": PHICategory.GEOGRAPHIC,
    "phone": PHICategory.PHONE_NUMBERS,
    "telephone": PHICategory.PHONE_NUMBERS,
    "phone_number": PHICategory.PHONE_NUMBERS,
    "fax": PHICategory.FAX_NUMBERS,
    "fax_number": PHICategory.FAX_NUMBERS,
    "email": PHICategory.EMAIL_ADDRESSES,
    "email_address": PHICategory.EMAIL_ADDRESSES,
    "ssn": PHICategory.SSN,
    "social_security": PHICategory.SSN,
    "social_security_number": PHICategory.SSN,
    "mrn": PHICategory.MRN,
    "medical_record": PHICategory.MRN,
    "medical_record_number": PHICategory.MRN,
    "dob": PHICategory.DATES,
    "date_of_birth": PHICategory.DATES,
    "birth_date": PHICategory.DATES,
    "date_of_death": PHICategory.DATES,
    "admission_date": PHICategory.DATES,
    "discharge_date": PHICategory.DATES,
    "health_plan_id": PHICategory.HEALTH_PLAN_ID,
    "insurance_id": PHICategory.HEALTH_PLAN_ID,
    "account_number": PHICategory.ACCOUNT_NUMBERS,
    "license_number": PHICategory.CERTIFICATE_LICENSE,
    "certificate_number": PHICategory.CERTIFICATE_LICENSE,
    "vehicle_id": PHICategory.VEHICLE_ID,
    "vin": PHICategory.VEHICLE_ID,
    "device_serial": PHICategory.DEVICE_ID,
    "device_id": PHICategory.DEVICE_ID,
    "url": PHICategory.URLS,
    "website": PHICategory.URLS,
    "ip_address": PHICategory.IP_ADDRESSES,
    "ip": PHICategory.IP_ADDRESSES,
    "biometric_id": PHICategory.BIOMETRIC_ID,
    "fingerprint": PHICategory.BIOMETRIC_ID,
    "photo": PHICategory.PHOTOGRAPHS,
    "photograph": PHICategory.PHOTOGRAPHS,
    "image_path": PHICategory.PHOTOGRAPHS,
}

# DICOM tags that contain or may contain PHI
# Reference: DICOM PS3.15 Annex E — Attribute Confidentiality Profiles
_DICOM_PHI_TAGS: dict[str, PHICategory] = {
    "PatientName": PHICategory.NAMES,
    "PatientID": PHICategory.MRN,
    "PatientBirthDate": PHICategory.DATES,
    "PatientAddress": PHICategory.GEOGRAPHIC,
    "PatientTelephoneNumbers": PHICategory.PHONE_NUMBERS,
    "ReferringPhysicianName": PHICategory.NAMES,
    "InstitutionName": PHICategory.OTHER_UNIQUE_ID,
    "InstitutionAddress": PHICategory.GEOGRAPHIC,
    "StudyDate": PHICategory.DATES,
    "SeriesDate": PHICategory.DATES,
    "AcquisitionDate": PHICategory.DATES,
    "ContentDate": PHICategory.DATES,
    "AccessionNumber": PHICategory.ACCOUNT_NUMBERS,
    "OtherPatientIDs": PHICategory.OTHER_UNIQUE_ID,
    "PatientBirthName": PHICategory.NAMES,
    "MilitaryRank": PHICategory.OTHER_UNIQUE_ID,
    "EthnicGroup": PHICategory.OTHER_UNIQUE_ID,
    "Occupation": PHICategory.OTHER_UNIQUE_ID,
    "AdditionalPatientHistory": PHICategory.OTHER_UNIQUE_ID,
    "PatientComments": PHICategory.OTHER_UNIQUE_ID,
    "DeviceSerialNumber": PHICategory.DEVICE_ID,
    "ProtocolName": PHICategory.OTHER_UNIQUE_ID,
    "OperatorsName": PHICategory.NAMES,
    "PerformingPhysicianName": PHICategory.NAMES,
    "RequestingPhysician": PHICategory.NAMES,
}


# ---------------------------------------------------------------------------
# PHI Detector
# ---------------------------------------------------------------------------


class PHIDetector:
    """Comprehensive PHI/PII detection engine.

    Scans text, structured records, and optionally DICOM datasets for
    protected health information. All 18 HIPAA Safe Harbor identifiers
    per 45 CFR 164.514(b)(2) are covered.

    Features:
        - Regex-based pattern matching for SSN, MRN, phone, email, dates,
          geographic data, IP addresses, ZIP codes, URLs, account numbers.
        - Field-name heuristic detection for structured records.
        - DICOM tag scanning (requires optional ``pydicom`` dependency).
        - Optional Presidio NER integration for high-confidence name detection.
        - Batch scanning with aggregated ScanResult.
        - Non-destructive logging: PHI values are never logged in full.

    Args:
        custom_patterns: Additional regex patterns keyed by PHICategory.
        enable_presidio: Attempt to load and use Presidio for NER-based detection.
    """

    def __init__(
        self,
        custom_patterns: dict[PHICategory, list[re.Pattern[str]]] | None = None,
        enable_presidio: bool = False,
    ) -> None:
        self._patterns: dict[PHICategory, list[re.Pattern[str]]] = {}
        for cat, pats in _REGEX_PATTERNS.items():
            self._patterns[cat] = list(pats)
        if custom_patterns:
            for cat, pats in custom_patterns.items():
                existing = self._patterns.get(cat, [])
                existing.extend(pats)
                self._patterns[cat] = existing

        self._presidio_analyzer: Any = None
        if enable_presidio:
            self._presidio_analyzer = _try_load_presidio()

        self._pydicom_available = _check_pydicom()

    # ------------------------------------------------------------------
    # Text scanning
    # ------------------------------------------------------------------

    def scan_text(self, text: str, source: str = "text") -> list[PHIFinding]:
        """Scan free text for PHI patterns.

        Args:
            text: Input text to scan.
            source: Label for the source (used in finding location).

        Returns:
            List of PHIFinding objects, sorted by location.
        """
        if not text:
            return []

        findings: list[PHIFinding] = []

        # Regex-based scanning
        for category, patterns in self._patterns.items():
            risk = _CATEGORY_RISK.get(category, RiskLevel.MEDIUM)
            for pattern in patterns:
                for match in pattern.finditer(text):
                    findings.append(
                        PHIFinding(
                            phi_type=category,
                            location=f"{source}:offset:{match.start()}-{match.end()}",
                            value_preview=_redact_preview(match.group()),
                            confidence=0.85,
                            risk_level=risk,
                            detection_method=DetectionMethod.REGEX,
                            context=f"pattern={pattern.pattern[:40]}",
                        )
                    )

        # Presidio NER scanning (if available)
        if self._presidio_analyzer is not None:
            presidio_findings = self._scan_with_presidio(text, source)
            findings.extend(presidio_findings)

        # Sort by location string for consistent ordering
        findings.sort(key=lambda f: f.location)
        logger.debug(
            "Text scan of '%s' found %d PHI elements (value details redacted)",
            source,
            len(findings),
        )
        return findings

    # ------------------------------------------------------------------
    # Structured record scanning
    # ------------------------------------------------------------------

    def scan_record(self, record: dict[str, Any], record_id: str = "record") -> list[PHIFinding]:
        """Scan a structured data record for PHI.

        Checks field names against known PHI field names and scans
        string values with regex patterns.

        Args:
            record: Dictionary of field_name -> value.
            record_id: Identifier for the record (for logging/location).

        Returns:
            List of PHIFinding objects.
        """
        findings: list[PHIFinding] = []

        for key, value in record.items():
            key_normalized = key.lower().replace("-", "_").replace(" ", "_")

            # Field-name heuristic
            if key_normalized in _PHI_FIELD_NAMES:
                category = _PHI_FIELD_NAMES[key_normalized]
                risk = _CATEGORY_RISK.get(category, RiskLevel.MEDIUM)
                preview = _redact_preview(str(value)) if value is not None else "***"
                findings.append(
                    PHIFinding(
                        phi_type=category,
                        location=f"{record_id}:field:{key}",
                        value_preview=preview,
                        confidence=0.90,
                        risk_level=risk,
                        detection_method=DetectionMethod.FIELD_NAME,
                        context=f"field_name_match={key_normalized}",
                    )
                )

            # Regex scan on string values
            if isinstance(value, str) and value:
                text_findings = self.scan_text(value, source=f"{record_id}:field:{key}")
                findings.extend(text_findings)

        return findings

    # ------------------------------------------------------------------
    # DICOM scanning
    # ------------------------------------------------------------------

    def scan_dicom(self, dicom_path: str) -> list[PHIFinding]:
        """Scan a DICOM file for PHI in known tags.

        Requires ``pydicom`` to be installed. If unavailable, returns
        an empty list and logs a warning.

        Args:
            dicom_path: Path to a DICOM (.dcm) file.

        Returns:
            List of PHIFinding objects for PHI-containing DICOM tags.
        """
        if not self._pydicom_available:
            logger.warning("pydicom not available — skipping DICOM scan for %s", dicom_path)
            return []

        try:
            import pydicom  # type: ignore[import-untyped]

            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        except (ValueError, TypeError, AttributeError) as exc:
            logger.error("Failed to read DICOM file %s: %s", dicom_path, sanitize_log_message(str(exc)))
            return []

        findings: list[PHIFinding] = []
        for tag_name, category in _DICOM_PHI_TAGS.items():
            if hasattr(ds, tag_name):
                value = str(getattr(ds, tag_name, ""))
                if value and value.strip():
                    risk = _CATEGORY_RISK.get(category, RiskLevel.MEDIUM)
                    findings.append(
                        PHIFinding(
                            phi_type=category,
                            location=f"dicom:{dicom_path}:tag:{tag_name}",
                            value_preview=_redact_preview(value),
                            confidence=0.95,
                            risk_level=risk,
                            detection_method=DetectionMethod.DICOM_TAG,
                            context=f"DICOM_tag={tag_name}",
                        )
                    )

        logger.debug(
            "DICOM scan of '%s' found %d PHI tags (value details redacted)",
            dicom_path,
            len(findings),
        )
        return findings

    # ------------------------------------------------------------------
    # Batch scanning
    # ------------------------------------------------------------------

    def scan_batch(
        self,
        items: list[str | dict[str, Any]],
        source_prefix: str = "batch",
    ) -> ScanResult:
        """Scan a batch of text strings or records.

        Args:
            items: List of text strings or dictionaries to scan.
            source_prefix: Prefix for source labeling.

        Returns:
            Aggregated ScanResult with all findings.
        """
        result = ScanResult(scan_metadata={"source_prefix": source_prefix})

        for idx, item in enumerate(items):
            item_id = f"{source_prefix}_{idx}"
            if isinstance(item, dict):
                findings = self.scan_record(item, record_id=item_id)
            elif isinstance(item, str):
                findings = self.scan_text(item, source=item_id)
            else:
                logger.warning("Unsupported item type at index %d: %s", idx, type(item).__name__)
                continue

            result.total_items_scanned += 1
            for f in findings:
                result.add_finding(f)

        logger.info(
            "Batch scan complete: %d items scanned, %d PHI findings (details redacted)",
            result.total_items_scanned,
            result.total_findings,
        )
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_all_categories() -> list[PHICategory]:
        """Return all 18 HIPAA Safe Harbor identifier categories."""
        return list(PHICategory)

    @staticmethod
    def get_category_risk(category: PHICategory) -> RiskLevel:
        """Return the default risk level for a PHI category."""
        return _CATEGORY_RISK.get(category, RiskLevel.MEDIUM)

    # ------------------------------------------------------------------
    # Presidio integration (optional)
    # ------------------------------------------------------------------

    def _scan_with_presidio(self, text: str, source: str) -> list[PHIFinding]:
        """Use Presidio AnalyzerEngine for NER-based PHI detection."""
        if self._presidio_analyzer is None:
            return []

        try:
            results = self._presidio_analyzer.analyze(
                text=text,
                language="en",
                entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "LOCATION"],
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Presidio analysis failed: %s", sanitize_log_message(str(exc)))
            return []

        _presidio_to_phi: dict[str, PHICategory] = {
            "PERSON": PHICategory.NAMES,
            "PHONE_NUMBER": PHICategory.PHONE_NUMBERS,
            "EMAIL_ADDRESS": PHICategory.EMAIL_ADDRESSES,
            "US_SSN": PHICategory.SSN,
            "LOCATION": PHICategory.GEOGRAPHIC,
        }

        findings: list[PHIFinding] = []
        for r in results:
            category = _presidio_to_phi.get(r.entity_type, PHICategory.OTHER_UNIQUE_ID)
            risk = _CATEGORY_RISK.get(category, RiskLevel.MEDIUM)
            matched_text = text[r.start : r.end]
            findings.append(
                PHIFinding(
                    phi_type=category,
                    location=f"{source}:presidio:{r.start}-{r.end}",
                    value_preview=_redact_preview(matched_text),
                    confidence=r.score,
                    risk_level=risk,
                    detection_method=DetectionMethod.PRESIDIO,
                    context=f"entity_type={r.entity_type}",
                )
            )

        return findings


# ---------------------------------------------------------------------------
# Optional dependency helpers
# ---------------------------------------------------------------------------


def _check_pydicom() -> bool:
    """Check if pydicom is available without importing it eagerly."""
    try:
        import importlib

        return importlib.util.find_spec("pydicom") is not None
    except (ImportError, AttributeError):
        return False


def _try_load_presidio() -> Any:
    """Attempt to load the Presidio AnalyzerEngine."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]

        engine = AnalyzerEngine()
        logger.info("Presidio AnalyzerEngine loaded successfully")
        return engine
    except ImportError:
        logger.info("Presidio not available — using regex-only detection")
        return None
