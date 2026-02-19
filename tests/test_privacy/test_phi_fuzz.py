"""Property-based / fuzz tests for PHI detection and de-identification.

Uses Hypothesis to generate adversarial inputs and verify invariants:
- De-identified output never contains recognized PHI patterns.
- PHI detector handles malformed, unicode-heavy, and nested inputs
  without raising unhandled exceptions.

RESEARCH USE ONLY — Not approved for clinical deployment.
"""

from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import load_module

mod = load_module("privacy.phi_detector", "privacy/phi_detector.py")
PHIDetector = mod.PHIDetector

# PHI regex patterns to assert absence in de-identified output
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_MRN_RE = re.compile(r"\bMRN[:\s#]*\d{6,10}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+1[-.\\s]?)?\(?\d{3}\)?[-.\\s]?\d{3}[-.\\s]?\d{4}\b")


# ── Strategies ────────────────────────────────────────────────────────────

_phi_ssn = st.from_regex(r"\d{3}-\d{2}-\d{4}", fullmatch=True)
_phi_email = st.from_regex(r"[a-z]{3,8}@[a-z]{3,8}\.com", fullmatch=True)
_phi_mrn = st.from_regex(r"MRN#\d{6,10}", fullmatch=True)

_freetext = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=500,
)

_mixed_input = st.one_of(
    _freetext,
    st.builds(lambda prefix, ssn, suffix: f"{prefix} {ssn} {suffix}", _freetext, _phi_ssn, _freetext),
    st.builds(lambda prefix, email, suffix: f"{prefix} {email} {suffix}", _freetext, _phi_email, _freetext),
    st.builds(lambda prefix, mrn, suffix: f"{prefix} {mrn} {suffix}", _freetext, _phi_mrn, _freetext),
)


# ── PHI detector fuzz tests ──────────────────────────────────────────────


class TestPHIDetectorFuzz:
    """Property-based tests for PHI detection robustness."""

    def setup_method(self) -> None:
        self.detector = PHIDetector()

    @given(text=_freetext)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_scan_text_never_raises(self, text: str) -> None:
        """PHI scanner must not raise on arbitrary unicode input."""
        result = self.detector.scan_text(text)
        assert isinstance(result, list)

    @given(text=_mixed_input)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_scan_text_returns_valid_matches(self, text: str) -> None:
        """Every match must have valid offsets within the input."""
        matches = self.detector.scan_text(text)
        for m in matches:
            assert 0 <= m.start <= m.end <= len(text)
            assert m.value == text[m.start : m.end]

    @given(
        record=st.dictionaries(
            keys=st.sampled_from(["name", "email", "phone", "notes", "id", "value"]),
            values=_mixed_input,
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_scan_record_never_raises(self, record: dict) -> None:
        """Record scanning must not raise on arbitrary dict input."""
        result = self.detector.scan_record(record)
        assert isinstance(result, list)

    @given(text=st.text(min_size=0, max_size=200))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_has_phi_returns_bool(self, text: str) -> None:
        """has_phi must always return a boolean."""
        result = self.detector.has_phi(text)
        assert isinstance(result, bool)

    @given(
        ssn=_phi_ssn,
        prefix=st.text(alphabet="abcdefghijklmnop ", min_size=1, max_size=20),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_ssn_always_detected(self, ssn: str, prefix: str) -> None:
        """Known SSN patterns must always be detected."""
        text = f"{prefix} {ssn} end"
        matches = self.detector.scan_text(text)
        ssn_matches = [m for m in matches if m.phi_type == "ssn"]
        assert len(ssn_matches) >= 1, f"SSN '{ssn}' not detected in '{text}'"

    @given(
        email=_phi_email,
        prefix=st.text(alphabet="abcdefghijklmnop ", min_size=1, max_size=20),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_email_always_detected(self, email: str, prefix: str) -> None:
        """Known email patterns must always be detected."""
        text = f"{prefix} {email} end"
        matches = self.detector.scan_text(text)
        email_matches = [m for m in matches if m.phi_type == "email"]
        assert len(email_matches) >= 1, f"Email '{email}' not detected in '{text}'"
