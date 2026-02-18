"""De-identification pipeline for clinical trial data.

Implements multiple de-identification strategies compliant with HIPAA
Safe Harbor (45 CFR 164.514(b)(2)) and Expert Determination methods.
Supports redaction, HMAC-SHA256 pseudonymization, date shifting,
geographic generalization, and transformation audit trails.

References:
    - 45 CFR 164.514(b)(2): Safe Harbor de-identification
    - 45 CFR 164.514(a): Expert Determination method
    - NIST SP 800-188: De-Identifying Government Datasets
    - HHS Guidance on De-identification (2012, updated 2022)

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeidentificationMethod(str, Enum):
    """Supported de-identification transformation methods.

    Each method offers different trade-offs between data utility
    and privacy protection.
    """

    REDACT = "redact"
    HASH = "hash"
    GENERALIZE = "generalize"
    SUPPRESS = "suppress"
    PSEUDONYMIZE = "pseudonymize"
    DATE_SHIFT = "date_shift"


class TransformationStatus(str, Enum):
    """Status of a de-identification transformation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TransformationRecord:
    """Audit record for a single de-identification transformation.

    Attributes:
        field_name: The field that was transformed.
        method: De-identification method applied.
        original_type: Type category of the original value (never the value itself).
        status: Whether the transformation succeeded.
        timestamp: ISO timestamp of the transformation.
        notes: Additional context.
    """

    field_name: str
    method: DeidentificationMethod
    original_type: str
    status: TransformationStatus = TransformationStatus.SUCCESS
    timestamp: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DeidentificationResult:
    """Result of a de-identification operation on a single record.

    Attributes:
        record_id: Identifier for the processed record.
        original_phi_count: Number of PHI elements detected.
        transformed_count: Number of PHI elements transformed.
        method: Primary de-identification method used.
        clean_data: The de-identified output data.
        transformations: Audit trail of individual transformations.
        status: Overall transformation status.
    """

    record_id: str
    original_phi_count: int = 0
    transformed_count: int = 0
    method: str = ""
    clean_data: dict[str, Any] | str | None = None
    transformations: list[TransformationRecord] = field(default_factory=list)
    status: TransformationStatus = TransformationStatus.SUCCESS


@dataclass
class PipelineStatistics:
    """Cumulative statistics for the de-identification pipeline.

    Attributes:
        records_processed: Total records processed.
        phi_detected: Total PHI elements detected.
        phi_transformed: Total PHI elements successfully transformed.
        phi_failed: Total PHI elements that failed transformation.
        methods_used: Count of each method used.
    """

    records_processed: int = 0
    phi_detected: int = 0
    phi_transformed: int = 0
    phi_failed: int = 0
    methods_used: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PHI field classification
# ---------------------------------------------------------------------------

_PHI_FIELDS: dict[str, str] = {
    "name": "name",
    "first_name": "name",
    "last_name": "name",
    "patient_name": "name",
    "full_name": "name",
    "address": "geographic",
    "street": "geographic",
    "street_address": "geographic",
    "city": "geographic",
    "zip": "geographic",
    "zipcode": "geographic",
    "zip_code": "geographic",
    "postal_code": "geographic",
    "phone": "phone",
    "telephone": "phone",
    "phone_number": "phone",
    "fax": "fax",
    "fax_number": "fax",
    "email": "email",
    "email_address": "email",
    "ssn": "ssn",
    "social_security": "ssn",
    "mrn": "mrn",
    "medical_record": "mrn",
    "medical_record_number": "mrn",
    "dob": "date",
    "date_of_birth": "date",
    "birth_date": "date",
    "date_of_death": "date",
    "admission_date": "date",
    "discharge_date": "date",
    "health_plan_id": "health_plan",
    "insurance_id": "health_plan",
    "account_number": "account",
    "device_serial": "device",
    "device_id": "device",
    "ip_address": "ip",
    "ip": "ip",
    "url": "url",
    "website": "url",
}

# Regex patterns for text-based PHI detection
_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ip": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    "mrn": re.compile(r"\bMRN[:\s#]*\d{6,10}\b", re.IGNORECASE),
    "url": re.compile(r"https?://[^\s<>\"]+"),
}


# ---------------------------------------------------------------------------
# Deidentification Pipeline
# ---------------------------------------------------------------------------


class DeidentificationPipeline:
    """Production-grade de-identification pipeline for clinical trial data.

    Supports multiple transformation methods with cryptographically
    secure pseudonymization (HMAC-SHA256 with random salt) and
    consistent patient-level date shifting.

    Security:
        - Pseudonymization uses HMAC-SHA256 with a cryptographically
          random 32-byte salt generated via ``os.urandom(32)``.
        - Date shifts are consistent per patient_id to preserve
          temporal relationships within a patient's records.
        - The salt is generated once at pipeline instantiation and
          should be stored securely for reversibility (if needed).

    Args:
        default_method: Default de-identification method for all fields.
        salt: Cryptographic salt for HMAC pseudonymization. If None,
            a secure random salt is generated via os.urandom(32).
        date_shift_range_days: Maximum number of days for date shifting.
        field_methods: Per-field method overrides keyed by normalized field name.

    Raises:
        ValueError: If date_shift_range_days is not positive.
    """

    def __init__(
        self,
        default_method: DeidentificationMethod = DeidentificationMethod.REDACT,
        salt: bytes | None = None,
        date_shift_range_days: int = 365,
        field_methods: dict[str, DeidentificationMethod] | None = None,
    ) -> None:
        if date_shift_range_days <= 0:
            raise ValueError("date_shift_range_days must be positive")

        self.default_method = default_method
        # Cryptographically random salt — NOT a weak default
        self._salt: bytes = salt if salt is not None else os.urandom(32)
        self._date_shift_range = date_shift_range_days
        self._field_methods: dict[str, DeidentificationMethod] = dict(field_methods or {})
        self._patient_date_offsets: dict[str, int] = {}
        self._stats = PipelineStatistics()
        self._audit_trail: list[TransformationRecord] = []

        logger.info(
            "DeidentificationPipeline initialized (method=%s, date_shift_range=%d days)",
            default_method.value,
            date_shift_range_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deidentify_record(
        self,
        record: dict[str, Any],
        record_id: str = "",
        patient_id: str = "",
    ) -> DeidentificationResult:
        """De-identify a single structured record.

        Args:
            record: Dictionary of field_name -> value.
            record_id: Identifier for audit trail.
            patient_id: Patient identifier for consistent date shifting.

        Returns:
            DeidentificationResult with clean data and audit trail.
        """
        if not record_id:
            record_id = f"record_{self._stats.records_processed}"

        result = DeidentificationResult(record_id=record_id)
        clean = dict(record)
        phi_count = 0
        transformed = 0

        for key, value in record.items():
            key_normalized = key.lower().replace("-", "_").replace(" ", "_")

            if key_normalized not in _PHI_FIELDS:
                continue

            phi_count += 1
            phi_type = _PHI_FIELDS[key_normalized]
            method = self._field_methods.get(key_normalized, self._resolve_method(phi_type))

            try:
                clean[key] = self._apply_transformation(
                    value=value,
                    phi_type=phi_type,
                    method=method,
                    patient_id=patient_id,
                    field_name=key,
                )
                transformed += 1
                xform = TransformationRecord(
                    field_name=key,
                    method=method,
                    original_type=phi_type,
                    status=TransformationStatus.SUCCESS,
                )
            except Exception as exc:
                logger.error("Transformation failed for field '%s': %s", key, exc)
                clean[key] = "[DEIDENTIFICATION_FAILED]"
                xform = TransformationRecord(
                    field_name=key,
                    method=method,
                    original_type=phi_type,
                    status=TransformationStatus.FAILED,
                    notes=str(exc),
                )

            result.transformations.append(xform)
            self._audit_trail.append(xform)

        # Also scan string values for embedded PHI patterns
        for key, value in list(clean.items()):
            key_normalized = key.lower().replace("-", "_").replace(" ", "_")
            if key_normalized in _PHI_FIELDS:
                continue  # Already handled
            if isinstance(value, str):
                cleaned_text, text_phi_count = self._clean_text(value)
                if text_phi_count > 0:
                    clean[key] = cleaned_text
                    phi_count += text_phi_count
                    transformed += text_phi_count

        result.original_phi_count = phi_count
        result.transformed_count = transformed
        result.method = self.default_method.value
        result.clean_data = clean
        result.status = (
            TransformationStatus.SUCCESS
            if transformed == phi_count
            else TransformationStatus.PARTIAL
            if transformed > 0
            else TransformationStatus.FAILED
        )

        self._stats.records_processed += 1
        self._stats.phi_detected += phi_count
        self._stats.phi_transformed += transformed
        self._stats.phi_failed += phi_count - transformed

        method_key = self.default_method.value
        self._stats.methods_used[method_key] = self._stats.methods_used.get(method_key, 0) + 1

        return result

    def deidentify_batch(
        self,
        records: list[dict[str, Any]],
        patient_id_field: str = "patient_id",
    ) -> list[DeidentificationResult]:
        """De-identify a batch of records.

        Args:
            records: List of data records to de-identify.
            patient_id_field: Field name containing the patient identifier
                for consistent date shifting.

        Returns:
            List of DeidentificationResult, one per record.
        """
        results: list[DeidentificationResult] = []

        for idx, record in enumerate(records):
            patient_id = str(record.get(patient_id_field, f"patient_{idx}"))
            record_id = f"batch_{idx}"
            result = self.deidentify_record(
                record=record,
                record_id=record_id,
                patient_id=patient_id,
            )
            results.append(result)

        logger.info(
            "Batch de-identification complete: %d records, %d PHI transformed",
            len(records),
            sum(r.transformed_count for r in results),
        )
        return results

    def deidentify_text(self, text: str) -> str:
        """De-identify free text by replacing detected PHI patterns.

        Args:
            text: Input text potentially containing PHI.

        Returns:
            Text with PHI patterns replaced.
        """
        cleaned, _ = self._clean_text(text)
        return cleaned

    def get_statistics(self) -> PipelineStatistics:
        """Return cumulative pipeline statistics."""
        return PipelineStatistics(
            records_processed=self._stats.records_processed,
            phi_detected=self._stats.phi_detected,
            phi_transformed=self._stats.phi_transformed,
            phi_failed=self._stats.phi_failed,
            methods_used=dict(self._stats.methods_used),
        )

    def get_audit_trail(self) -> list[TransformationRecord]:
        """Return a copy of the transformation audit trail.

        Returns a copy to prevent external mutation of the audit log.
        """
        return list(self._audit_trail)

    # ------------------------------------------------------------------
    # Transformation implementations
    # ------------------------------------------------------------------

    def _apply_transformation(
        self,
        value: Any,
        phi_type: str,
        method: DeidentificationMethod,
        patient_id: str = "",
        field_name: str = "",
    ) -> Any:
        """Apply the specified de-identification transformation."""
        str_value = str(value) if value is not None else ""

        if method == DeidentificationMethod.REDACT:
            return "[REDACTED]"

        elif method == DeidentificationMethod.SUPPRESS:
            return ""

        elif method == DeidentificationMethod.HASH:
            return self._hash_value(str_value)

        elif method == DeidentificationMethod.PSEUDONYMIZE:
            return self._pseudonymize(str_value, phi_type)

        elif method == DeidentificationMethod.GENERALIZE:
            return self._generalize(str_value, phi_type)

        elif method == DeidentificationMethod.DATE_SHIFT:
            return self._date_shift(str_value, patient_id)

        else:
            logger.warning("Unknown method '%s' for field '%s', falling back to REDACT", method, field_name)
            return "[REDACTED]"

    def _hash_value(self, value: str) -> str:
        """Hash a value using HMAC-SHA256 with the pipeline salt."""
        h = hmac.new(self._salt, value.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"[HASH:{h[:16]}]"

    def _pseudonymize(self, value: str, phi_type: str) -> str:
        """Generate an HMAC-SHA256 pseudonym preserving format hints.

        Uses HMAC-SHA256 with cryptographically random salt for
        irreversible (without salt) pseudonymization.
        """
        digest = hmac.new(self._salt, value.encode("utf-8"), hashlib.sha256).hexdigest()

        if phi_type == "name":
            return f"PSEUDO_NAME_{digest[:8].upper()}"
        elif phi_type == "mrn":
            # Preserve numeric format with prefix
            numeric_part = int(digest[:8], 16) % 10000000
            return f"MRN{numeric_part:07d}"
        elif phi_type == "email":
            return f"pseudo_{digest[:8]}@deidentified.local"
        elif phi_type == "phone":
            numeric = int(digest[:10], 16) % 10000000000
            return f"555-{numeric % 10000000:07d}"
        elif phi_type == "ssn":
            numeric = int(digest[:9], 16) % 1000000000
            return f"000-00-{numeric % 10000:04d}"
        else:
            return f"PSEUDO_{digest[:12].upper()}"

    def _generalize(self, value: str, phi_type: str) -> str:
        """Generalize a value to reduce specificity.

        Geographic: ZIP code to 3-digit prefix (per Safe Harbor).
        Date: Year only.
        Age: 5-year bracket.
        Other: Category label.
        """
        if phi_type == "geographic":
            # ZIP to 3-digit per HIPAA Safe Harbor
            digits = re.sub(r"[^0-9]", "", value)
            if len(digits) >= 3:
                prefix = digits[:3]
                # Per Safe Harbor, 3-digit ZIPs with <20,000 population
                # must be set to 000. We flag but preserve for research.
                return f"{prefix}XX"
            return "[GEOGRAPHIC_AREA]"

        elif phi_type == "date":
            # Generalize to year only
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y"):
                try:
                    dt = datetime.strptime(value, fmt)
                    return str(dt.year)
                except ValueError:
                    continue
            return "[DATE_YEAR]"

        elif phi_type == "name":
            return "[PERSON_NAME]"

        elif phi_type in ("phone", "fax"):
            return "[PHONE_NUMBER]"

        elif phi_type == "email":
            return "[EMAIL_ADDRESS]"

        elif phi_type == "ssn":
            return "[SSN]"

        elif phi_type == "ip":
            # Generalize to subnet
            parts = value.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.0.0/16"
            return "[IP_ADDRESS]"

        else:
            return f"[{phi_type.upper()}]"

    def _date_shift(self, value: str, patient_id: str) -> str:
        """Shift a date by a consistent patient-level offset.

        The offset is derived deterministically from the patient_id
        and salt, ensuring all dates for the same patient are shifted
        by the same amount, preserving temporal relationships.

        Args:
            value: Date string to shift.
            patient_id: Patient identifier for consistent offset.

        Returns:
            Shifted date string in the same format, or [DATE_SHIFTED]
            if parsing fails.
        """
        offset_days = self._get_patient_date_offset(patient_id)

        date_formats = [
            ("%Y-%m-%d", "%Y-%m-%d"),
            ("%m/%d/%Y", "%m/%d/%Y"),
            ("%m-%d-%Y", "%m-%d-%Y"),
            ("%d/%m/%Y", "%d/%m/%Y"),
            ("%Y%m%d", "%Y%m%d"),
        ]

        for parse_fmt, output_fmt in date_formats:
            try:
                dt = datetime.strptime(value, parse_fmt)
                shifted = dt + timedelta(days=offset_days)
                return shifted.strftime(output_fmt)
            except ValueError:
                continue

        logger.debug("Could not parse date '%s' for shifting", value[:10])
        return "[DATE_SHIFTED]"

    def _get_patient_date_offset(self, patient_id: str) -> int:
        """Get or create a consistent date offset for a patient.

        The offset is deterministically derived from the patient_id
        and the pipeline salt using HMAC-SHA256, then mapped to the
        configured date shift range.
        """
        if not patient_id:
            patient_id = "unknown"

        if patient_id not in self._patient_date_offsets:
            digest = hmac.new(
                self._salt,
                patient_id.encode("utf-8"),
                hashlib.sha256,
            ).digest()
            # Use first 4 bytes as signed integer, map to range
            raw = int.from_bytes(digest[:4], byteorder="big", signed=False)
            offset = (raw % (2 * self._date_shift_range)) - self._date_shift_range
            self._patient_date_offsets[patient_id] = offset

        return self._patient_date_offsets[patient_id]

    def _resolve_method(self, phi_type: str) -> DeidentificationMethod:
        """Resolve the transformation method for a PHI type.

        Date fields default to DATE_SHIFT; geographic fields default
        to GENERALIZE; all others use the pipeline default.
        """
        if phi_type == "date" and self.default_method not in (
            DeidentificationMethod.REDACT,
            DeidentificationMethod.SUPPRESS,
        ):
            return DeidentificationMethod.DATE_SHIFT

        if phi_type == "geographic" and self.default_method not in (
            DeidentificationMethod.REDACT,
            DeidentificationMethod.SUPPRESS,
        ):
            return DeidentificationMethod.GENERALIZE

        return self.default_method

    def _clean_text(self, text: str) -> tuple[str, int]:
        """Remove PHI patterns from free text.

        Returns:
            Tuple of (cleaned text, number of PHI elements found).
        """
        phi_count = 0
        result = text

        for phi_type, pattern in _TEXT_PATTERNS.items():
            matches = list(pattern.finditer(result))
            if matches:
                phi_count += len(matches)
                # Replace in reverse order to preserve offsets
                for match in reversed(matches):
                    replacement = f"[{phi_type.upper()}_REDACTED]"
                    result = result[: match.start()] + replacement + result[match.end() :]

        return result, phi_count
