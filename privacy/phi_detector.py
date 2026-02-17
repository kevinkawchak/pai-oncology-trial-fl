"""Protected Health Information (PHI) detection.

Identifies the 18 HIPAA identifiers in text and structured data,
enabling automated screening before data enters the federated
learning pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The 18 HIPAA identifiers
HIPAA_IDENTIFIERS: list[str] = [
    "name",
    "address",
    "dates",
    "phone_number",
    "fax_number",
    "email",
    "ssn",
    "mrn",
    "health_plan_id",
    "account_number",
    "certificate_license",
    "vehicle_id",
    "device_id",
    "url",
    "ip_address",
    "biometric_id",
    "photo",
    "other_unique_id",
]


@dataclass
class PHIMatch:
    """A detected PHI occurrence in text.

    Attributes:
        phi_type: Category of the identifier (from HIPAA_IDENTIFIERS).
        value: The matched text.
        start: Start character offset.
        end: End character offset.
        confidence: Detection confidence score [0, 1].
    """

    phi_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


# Regex patterns for common PHI types
_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone_number": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "dates": re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
    "mrn": re.compile(r"\bMRN[:\s#]*\d{6,10}\b", re.IGNORECASE),
    "url": re.compile(r"https?://[^\s<>\"]+"),
}

# Field names that suggest PHI in structured data
_PHI_FIELD_NAMES: set[str] = {
    "name",
    "first_name",
    "last_name",
    "patient_name",
    "address",
    "street",
    "city",
    "zip",
    "zipcode",
    "zip_code",
    "phone",
    "telephone",
    "fax",
    "email",
    "ssn",
    "social_security",
    "mrn",
    "medical_record",
    "dob",
    "date_of_birth",
    "birth_date",
}


class PHIDetector:
    """Scans text and structured data for Protected Health Information.

    Uses regex-based pattern matching and field-name heuristics
    to identify the 18 HIPAA identifiers.

    Args:
        custom_patterns: Optional additional regex patterns to scan for.
    """

    def __init__(
        self,
        custom_patterns: dict[str, re.Pattern] | None = None,
    ):
        self.patterns = dict(_PATTERNS)
        if custom_patterns:
            self.patterns.update(custom_patterns)

    def scan_text(self, text: str) -> list[PHIMatch]:
        """Scan free text for PHI patterns.

        Args:
            text: Input text to scan.

        Returns:
            List of PHIMatch objects for each detected occurrence.
        """
        matches: list[PHIMatch] = []
        for phi_type, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                matches.append(
                    PHIMatch(
                        phi_type=phi_type,
                        value=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return sorted(matches, key=lambda x: x.start)

    def scan_record(self, record: dict) -> list[PHIMatch]:
        """Scan a structured data record for PHI.

        Checks both field names (against known PHI field names) and
        field values (against regex patterns).

        Args:
            record: Dictionary of field_name -> value.

        Returns:
            List of PHIMatch objects.
        """
        matches: list[PHIMatch] = []
        for key, value in record.items():
            key_lower = key.lower().replace("-", "_")
            # Check if the field name itself suggests PHI
            if key_lower in _PHI_FIELD_NAMES:
                matches.append(
                    PHIMatch(
                        phi_type=_field_name_to_phi_type(key_lower),
                        value=str(value),
                        start=0,
                        end=len(str(value)),
                        confidence=0.9,
                    )
                )
            # Also scan the value text
            if isinstance(value, str):
                text_matches = self.scan_text(value)
                matches.extend(text_matches)
        return matches

    def scan_dataset(self, records: list[dict]) -> dict[str, list[PHIMatch]]:
        """Scan a list of records and return matches keyed by record index.

        Args:
            records: List of data records.

        Returns:
            Mapping of ``"record_{i}"`` -> list of PHIMatch.
        """
        results: dict[str, list[PHIMatch]] = {}
        for i, record in enumerate(records):
            hits = self.scan_record(record)
            if hits:
                results[f"record_{i}"] = hits
        return results

    def has_phi(self, text: str) -> bool:
        """Quick check: does the text contain any PHI?"""
        return len(self.scan_text(text)) > 0

    @staticmethod
    def get_hipaa_identifiers() -> list[str]:
        """Return the list of 18 HIPAA identifier categories."""
        return list(HIPAA_IDENTIFIERS)


def _field_name_to_phi_type(field: str) -> str:
    """Map a field name to the closest HIPAA identifier type."""
    mapping = {
        "name": "name",
        "first_name": "name",
        "last_name": "name",
        "patient_name": "name",
        "address": "address",
        "street": "address",
        "city": "address",
        "zip": "address",
        "zipcode": "address",
        "zip_code": "address",
        "phone": "phone_number",
        "telephone": "phone_number",
        "fax": "fax_number",
        "email": "email",
        "ssn": "ssn",
        "social_security": "ssn",
        "mrn": "mrn",
        "medical_record": "mrn",
        "dob": "dates",
        "date_of_birth": "dates",
        "birth_date": "dates",
    }
    return mapping.get(field, "other_unique_id")
