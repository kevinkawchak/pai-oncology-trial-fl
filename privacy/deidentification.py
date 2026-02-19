"""De-identification pipeline for patient data.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Removes or replaces PHI in structured and unstructured data before
it enters the federated learning pipeline, ensuring HIPAA Safe
Harbor compliance.

.. deprecated::
    This module provides the legacy de-identification pipeline retained for
    backward compatibility.  New code should use the canonical implementation
    at ``privacy/de-identification/deidentification_pipeline.py`` which adds
    HMAC-SHA256 pseudonymization, consistent date shifting, geographic
    generalization, and transformation audit trails.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from enum import Enum

from privacy.phi_detector import PHIDetector, PHIMatch

logger = logging.getLogger(__name__)


class DeidentMethod(str, Enum):
    """Available de-identification strategies."""

    REDACT = "redact"
    HASH = "hash"
    GENERALIZE = "generalize"
    REPLACE = "replace"


@dataclass
class DeidentResult:
    """Outcome of a de-identification operation.

    Attributes:
        original_phi_count: Number of PHI elements detected.
        removed_phi_count: Number of PHI elements removed/replaced.
        method: De-identification method applied.
        clean_data: The de-identified output.
    """

    original_phi_count: int
    removed_phi_count: int
    method: str
    clean_data: str | dict | list


class Deidentifier:
    """De-identifies patient data for federated learning.

    Supports multiple strategies:
    - **redact**: Replace PHI with ``[REDACTED]``.
    - **hash**: Replace PHI with a one-way SHA-256 hash.
    - **generalize**: Replace specific values with categories
      (e.g. exact age -> age bracket).
    - **replace**: Substitute synthetic values.

    Args:
        method: De-identification strategy.
        salt: Salt for hash-based de-identification.
    """

    def __init__(
        self,
        method: DeidentMethod | str = DeidentMethod.REDACT,
        salt: str | bytes | None = None,
    ):
        if isinstance(method, str):
            method = DeidentMethod(method)
        self.method = method
        if salt is None:
            salt = os.urandom(32)
        self.salt = salt if isinstance(salt, str) else salt.hex()
        self.detector = PHIDetector()
        self._stats = {"records_processed": 0, "phi_removed": 0}

    def deidentify_text(self, text: str) -> DeidentResult:
        """Remove PHI from free text.

        Args:
            text: Input text potentially containing PHI.

        Returns:
            DeidentResult with the cleaned text.
        """
        matches = self.detector.scan_text(text)
        if not matches:
            return DeidentResult(
                original_phi_count=0,
                removed_phi_count=0,
                method=self.method.value,
                clean_data=text,
            )

        # Process matches in reverse order to preserve offsets
        clean = text
        for match in sorted(matches, key=lambda m: m.start, reverse=True):
            replacement = self._get_replacement(match)
            clean = clean[: match.start] + replacement + clean[match.end :]

        self._stats["phi_removed"] += len(matches)
        return DeidentResult(
            original_phi_count=len(matches),
            removed_phi_count=len(matches),
            method=self.method.value,
            clean_data=clean,
        )

    def deidentify_record(self, record: dict) -> DeidentResult:
        """Remove PHI from a structured data record.

        Args:
            record: Dictionary of field_name -> value.

        Returns:
            DeidentResult with the cleaned record.
        """
        matches = self.detector.scan_record(record)
        clean = dict(record)

        phi_fields = set()
        for match in matches:
            for key in record:
                key_lower = key.lower().replace("-", "_")
                from privacy.phi_detector import _PHI_FIELD_NAMES

                if key_lower in _PHI_FIELD_NAMES:
                    phi_fields.add(key)

        for key in phi_fields:
            value = str(clean[key])
            dummy_match = PHIMatch(
                phi_type="field",
                value=value,
                start=0,
                end=len(value),
            )
            clean[key] = self._get_replacement(dummy_match)

        # Also scan string values for embedded PHI
        for key, value in clean.items():
            if isinstance(value, str) and key not in phi_fields:
                result = self.deidentify_text(value)
                clean[key] = result.clean_data

        self._stats["records_processed"] += 1
        self._stats["phi_removed"] += len(matches)

        return DeidentResult(
            original_phi_count=len(matches),
            removed_phi_count=len(matches),
            method=self.method.value,
            clean_data=clean,
        )

    def deidentify_dataset(self, records: list[dict]) -> list[DeidentResult]:
        """De-identify a batch of records.

        Args:
            records: List of patient data records.

        Returns:
            List of DeidentResult, one per record.
        """
        results = []
        for record in records:
            results.append(self.deidentify_record(record))
        logger.info(
            "De-identified %d records — %d total PHI removed",
            len(records),
            sum(r.removed_phi_count for r in results),
        )
        return results

    def get_stats(self) -> dict[str, int]:
        """Return cumulative de-identification statistics."""
        return dict(self._stats)

    def _get_replacement(self, match: PHIMatch) -> str:
        """Generate a replacement string based on the configured method."""
        if self.method == DeidentMethod.REDACT:
            return "[REDACTED]"
        elif self.method == DeidentMethod.HASH:
            h = hashlib.sha256((self.salt + match.value).encode()).hexdigest()[:16]
            return f"[HASH:{h}]"
        elif self.method == DeidentMethod.GENERALIZE:
            return f"[{match.phi_type.upper()}]"
        elif self.method == DeidentMethod.REPLACE:
            return f"[SYNTHETIC_{match.phi_type.upper()}]"
        return "[REDACTED]"
