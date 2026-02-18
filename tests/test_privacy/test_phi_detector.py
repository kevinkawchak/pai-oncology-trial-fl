"""Tests for privacy/phi_detector.py — PHI detection module.

Covers PHIDetector creation, regex-based pattern matching for SSN,
MRN, phone, email, IP, dates, URL patterns; empty input; no-match
input; structured record scanning; batch dataset scanning; HIPAA
identifiers enumeration; and custom pattern support.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("privacy.phi_detector", "privacy/phi_detector.py")

PHIDetector = mod.PHIDetector
PHIMatch = mod.PHIMatch
HIPAA_IDENTIFIERS = mod.HIPAA_IDENTIFIERS


class TestPHIDetectorCreation:
    """Tests for PHIDetector instantiation and configuration."""

    def test_init_default(self):
        """Default PHIDetector is created with built-in patterns."""
        detector = PHIDetector()
        assert detector.patterns is not None
        assert "ssn" in detector.patterns
        assert "email" in detector.patterns

    def test_init_custom_patterns(self):
        """PHIDetector accepts custom regex patterns that extend defaults."""
        custom = {"custom_id": re.compile(r"CID-\d{6}")}
        detector = PHIDetector(custom_patterns=custom)
        assert "custom_id" in detector.patterns
        # Built-in patterns should still be present
        assert "ssn" in detector.patterns

    def test_init_custom_patterns_override_builtin(self):
        """Custom patterns can override built-in patterns."""
        custom_ssn = re.compile(r"\d{4}-\d{4}-\d{4}")
        detector = PHIDetector(custom_patterns={"ssn": custom_ssn})
        assert detector.patterns["ssn"] is custom_ssn


class TestScanTextPatterns:
    """Tests for scan_text PHI pattern detection."""

    def test_scan_text_ssn(self):
        """Detects SSN pattern (###-##-####)."""
        detector = PHIDetector()
        matches = detector.scan_text("SSN: 123-45-6789")
        ssn_matches = [m for m in matches if m.phi_type == "ssn"]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].value == "123-45-6789"

    def test_scan_text_mrn(self):
        """Detects MRN pattern (MRN followed by 6-10 digits)."""
        detector = PHIDetector()
        matches = detector.scan_text("MRN: 12345678")
        mrn_matches = [m for m in matches if m.phi_type == "mrn"]
        assert len(mrn_matches) == 1
        assert "12345678" in mrn_matches[0].value

    def test_scan_text_phone(self):
        """Detects phone number pattern."""
        detector = PHIDetector()
        matches = detector.scan_text("Call (555) 123-4567")
        phone_matches = [m for m in matches if m.phi_type == "phone_number"]
        assert len(phone_matches) == 1
        assert "555" in phone_matches[0].value

    def test_scan_text_email(self):
        """Detects email address pattern."""
        detector = PHIDetector()
        matches = detector.scan_text("Contact: john.doe@example.com")
        email_matches = [m for m in matches if m.phi_type == "email"]
        assert len(email_matches) == 1
        assert email_matches[0].value == "john.doe@example.com"

    def test_scan_text_ip_address(self):
        """Detects IP address pattern."""
        detector = PHIDetector()
        matches = detector.scan_text("IP: 192.168.1.100")
        ip_matches = [m for m in matches if m.phi_type == "ip_address"]
        assert len(ip_matches) == 1
        assert ip_matches[0].value == "192.168.1.100"

    def test_scan_text_date(self):
        """Detects date patterns (MM/DD/YYYY)."""
        detector = PHIDetector()
        matches = detector.scan_text("Admitted on 01/15/2026")
        date_matches = [m for m in matches if m.phi_type == "dates"]
        assert len(date_matches) == 1
        assert date_matches[0].value == "01/15/2026"

    def test_scan_text_url(self):
        """Detects URL pattern."""
        detector = PHIDetector()
        matches = detector.scan_text("Visit https://hospital.example.com/patient")
        url_matches = [m for m in matches if m.phi_type == "url"]
        assert len(url_matches) == 1
        assert url_matches[0].value.startswith("https://")

    def test_scan_text_multiple_phi(self):
        """Detects multiple PHI types in a single text."""
        detector = PHIDetector()
        text = "SSN: 123-45-6789, Email: a@b.com, MRN: 12345678"
        matches = detector.scan_text(text)
        types_found = {m.phi_type for m in matches}
        assert "ssn" in types_found
        assert "email" in types_found
        assert "mrn" in types_found

    def test_scan_text_results_sorted_by_start(self):
        """Matches are returned sorted by start offset."""
        detector = PHIDetector()
        text = "Email: a@b.com followed by SSN: 123-45-6789"
        matches = detector.scan_text(text)
        starts = [m.start for m in matches]
        assert starts == sorted(starts)


class TestScanTextEdgeCases:
    """Tests for edge cases in scan_text."""

    def test_scan_text_empty_string(self):
        """Empty input returns no matches."""
        detector = PHIDetector()
        matches = detector.scan_text("")
        assert matches == []

    def test_scan_text_no_phi(self):
        """Text without PHI returns no matches."""
        detector = PHIDetector()
        matches = detector.scan_text("The patient has stage IIIB NSCLC.")
        assert matches == []

    def test_scan_text_whitespace_only(self):
        """Whitespace-only input returns no matches."""
        detector = PHIDetector()
        matches = detector.scan_text("   \t\n  ")
        assert matches == []


class TestPHIMatchDataclass:
    """Tests for the PHIMatch dataclass."""

    def test_phi_match_default_confidence(self):
        """PHIMatch defaults to confidence=1.0."""
        m = PHIMatch(phi_type="ssn", value="123-45-6789", start=0, end=11)
        np.testing.assert_allclose(m.confidence, 1.0)

    def test_phi_match_custom_confidence(self):
        """PHIMatch accepts custom confidence."""
        m = PHIMatch(phi_type="ssn", value="123-45-6789", start=0, end=11, confidence=0.85)
        np.testing.assert_allclose(m.confidence, 0.85)


class TestScanRecord:
    """Tests for scan_record structured data scanning."""

    def test_scan_record_phi_field_name(self):
        """Record with a known PHI field name is detected."""
        detector = PHIDetector()
        record = {"name": "John Doe", "tumor_type": "NSCLC"}
        matches = detector.scan_record(record)
        phi_types = [m.phi_type for m in matches]
        assert "name" in phi_types

    def test_scan_record_embedded_text_phi(self):
        """Record string values are scanned for embedded PHI patterns."""
        detector = PHIDetector()
        record = {"notes": "Contact at john@example.com"}
        matches = detector.scan_record(record)
        email_matches = [m for m in matches if m.phi_type == "email"]
        assert len(email_matches) >= 1

    def test_scan_record_field_confidence(self):
        """Field-name based PHI detections have confidence=0.9."""
        detector = PHIDetector()
        record = {"ssn": "123-45-6789"}
        matches = detector.scan_record(record)
        field_matches = [m for m in matches if m.confidence == 0.9]
        assert len(field_matches) >= 1


class TestScanDataset:
    """Tests for scan_dataset batch scanning."""

    def test_scan_dataset_batch(self):
        """Scans a batch of records and keys results by record index."""
        detector = PHIDetector()
        records = [
            {"name": "Alice", "age": 45},
            {"notes": "no phi here"},
            {"email": "bob@test.com"},
        ]
        results = detector.scan_dataset(records)
        assert "record_0" in results
        assert "record_2" in results

    def test_scan_dataset_empty(self):
        """Empty dataset returns empty dict."""
        detector = PHIDetector()
        results = detector.scan_dataset([])
        assert results == {}


class TestHasPHI:
    """Tests for the has_phi quick check."""

    def test_has_phi_true(self):
        """Returns True when PHI is present."""
        detector = PHIDetector()
        assert detector.has_phi("SSN: 123-45-6789") is True

    def test_has_phi_false(self):
        """Returns False when no PHI is present."""
        detector = PHIDetector()
        assert detector.has_phi("No personal info here") is False


class TestHIPAAIdentifiers:
    """Tests for HIPAA identifier list."""

    def test_get_hipaa_identifiers_count(self):
        """There should be 18 HIPAA identifiers."""
        identifiers = PHIDetector.get_hipaa_identifiers()
        assert len(identifiers) == 18

    def test_get_hipaa_identifiers_is_copy(self):
        """get_hipaa_identifiers returns a new list (not a reference)."""
        ids1 = PHIDetector.get_hipaa_identifiers()
        ids2 = PHIDetector.get_hipaa_identifiers()
        assert ids1 is not ids2
        assert ids1 == ids2
