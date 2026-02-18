"""Tests for privacy/deidentification.py — De-identification pipeline.

Covers Deidentifier creation, REDACT method, HASH method, GENERALIZE
method, REPLACE method, os.urandom salt default, custom salt, empty
input, text with no PHI, record de-identification, dataset batch
processing, and statistics tracking.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import load_module

# Load phi_detector first since deidentification.py imports from it
_phi_mod = load_module("privacy.phi_detector", "privacy/phi_detector.py")
mod = load_module("privacy.deidentification", "privacy/deidentification.py")

Deidentifier = mod.Deidentifier
DeidentMethod = mod.DeidentMethod
DeidentResult = mod.DeidentResult


class TestDeidentMethodEnum:
    """Tests for DeidentMethod enum values and transitions."""

    def test_enum_values(self):
        """All four de-identification methods exist."""
        assert DeidentMethod.REDACT.value == "redact"
        assert DeidentMethod.HASH.value == "hash"
        assert DeidentMethod.GENERALIZE.value == "generalize"
        assert DeidentMethod.REPLACE.value == "replace"

    def test_enum_from_string(self):
        """DeidentMethod can be constructed from string values."""
        assert DeidentMethod("redact") == DeidentMethod.REDACT
        assert DeidentMethod("hash") == DeidentMethod.HASH

    def test_enum_invalid_string_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            DeidentMethod("nonexistent")


class TestDeidentifierCreation:
    """Tests for Deidentifier initialization."""

    def test_default_creation(self):
        """Default Deidentifier uses REDACT method."""
        d = Deidentifier()
        assert d.method == DeidentMethod.REDACT

    def test_default_salt_is_generated(self):
        """Default salt is generated via os.urandom (non-empty hex string)."""
        d = Deidentifier()
        assert isinstance(d.salt, str)
        assert len(d.salt) > 0
        # os.urandom(32).hex() produces a 64-char hex string
        assert len(d.salt) == 64

    def test_custom_salt_string(self):
        """Custom string salt is stored as-is."""
        d = Deidentifier(salt="my_custom_salt")
        assert d.salt == "my_custom_salt"

    def test_custom_salt_bytes(self):
        """Custom bytes salt is converted to hex."""
        salt_bytes = b"\xaa\xbb\xcc"
        d = Deidentifier(salt=salt_bytes)
        assert d.salt == salt_bytes.hex()

    def test_string_method_accepted(self):
        """Method can be provided as a string."""
        d = Deidentifier(method="hash")
        assert d.method == DeidentMethod.HASH

    def test_initial_stats(self):
        """Initial stats are zeroed."""
        d = Deidentifier()
        stats = d.get_stats()
        assert stats["records_processed"] == 0
        assert stats["phi_removed"] == 0


class TestDeidentifyTextRedact:
    """Tests for REDACT de-identification method on text."""

    def test_redact_ssn(self):
        """SSN is replaced with [REDACTED]."""
        d = Deidentifier(method=DeidentMethod.REDACT)
        result = d.deidentify_text("SSN: 123-45-6789")
        assert "[REDACTED]" in result.clean_data
        assert "123-45-6789" not in result.clean_data

    def test_redact_multiple_phi(self):
        """Multiple PHI elements are all replaced."""
        d = Deidentifier(method=DeidentMethod.REDACT)
        text = "SSN: 123-45-6789, Email: test@example.com"
        result = d.deidentify_text(text)
        assert "123-45-6789" not in result.clean_data
        assert "test@example.com" not in result.clean_data
        assert result.original_phi_count >= 2
        assert result.removed_phi_count == result.original_phi_count

    def test_redact_preserves_non_phi_text(self):
        """Non-PHI portions of text are preserved."""
        d = Deidentifier(method=DeidentMethod.REDACT)
        result = d.deidentify_text("Name: SSN: 123-45-6789 end")
        assert "Name:" in result.clean_data
        assert "end" in result.clean_data

    def test_redact_method_in_result(self):
        """Result records the method used."""
        d = Deidentifier(method=DeidentMethod.REDACT)
        result = d.deidentify_text("SSN: 123-45-6789")
        assert result.method == "redact"


class TestDeidentifyTextHash:
    """Tests for HASH de-identification method on text."""

    def test_hash_replaces_phi(self):
        """PHI is replaced with [HASH:...] pattern."""
        d = Deidentifier(method=DeidentMethod.HASH, salt="test_salt")
        result = d.deidentify_text("SSN: 123-45-6789")
        assert "123-45-6789" not in result.clean_data
        assert "[HASH:" in result.clean_data

    def test_hash_deterministic_with_same_salt(self):
        """Same salt produces same hash for same input."""
        d1 = Deidentifier(method=DeidentMethod.HASH, salt="salt_a")
        d2 = Deidentifier(method=DeidentMethod.HASH, salt="salt_a")
        r1 = d1.deidentify_text("SSN: 123-45-6789")
        r2 = d2.deidentify_text("SSN: 123-45-6789")
        assert r1.clean_data == r2.clean_data

    def test_hash_different_salt_produces_different_hash(self):
        """Different salts produce different hashes."""
        d1 = Deidentifier(method=DeidentMethod.HASH, salt="salt_a")
        d2 = Deidentifier(method=DeidentMethod.HASH, salt="salt_b")
        r1 = d1.deidentify_text("SSN: 123-45-6789")
        r2 = d2.deidentify_text("SSN: 123-45-6789")
        assert r1.clean_data != r2.clean_data

    def test_hash_mathematical_correctness(self):
        """Verify the hash matches SHA-256 of salt+value truncated to 16 chars."""
        salt = "test_salt"
        d = Deidentifier(method=DeidentMethod.HASH, salt=salt)
        result = d.deidentify_text("SSN: 123-45-6789")
        expected_hash = hashlib.sha256((salt + "123-45-6789").encode()).hexdigest()[:16]
        assert f"[HASH:{expected_hash}]" in result.clean_data


class TestDeidentifyTextOtherMethods:
    """Tests for GENERALIZE and REPLACE de-identification methods."""

    def test_generalize_method(self):
        """GENERALIZE replaces PHI with [TYPE] pattern."""
        d = Deidentifier(method=DeidentMethod.GENERALIZE)
        result = d.deidentify_text("SSN: 123-45-6789")
        assert "123-45-6789" not in result.clean_data
        assert "[SSN]" in result.clean_data

    def test_replace_method(self):
        """REPLACE replaces PHI with [SYNTHETIC_TYPE] pattern."""
        d = Deidentifier(method=DeidentMethod.REPLACE)
        result = d.deidentify_text("SSN: 123-45-6789")
        assert "123-45-6789" not in result.clean_data
        assert "[SYNTHETIC_SSN]" in result.clean_data


class TestDeidentifyTextEdgeCases:
    """Tests for edge cases in text de-identification."""

    def test_empty_string(self):
        """Empty string returns clean_data as empty string with zero counts."""
        d = Deidentifier()
        result = d.deidentify_text("")
        assert result.clean_data == ""
        assert result.original_phi_count == 0
        assert result.removed_phi_count == 0

    def test_no_phi_text(self):
        """Text with no PHI returns original text unchanged."""
        d = Deidentifier()
        text = "The patient has NSCLC stage IIIB."
        result = d.deidentify_text(text)
        assert result.clean_data == text
        assert result.original_phi_count == 0

    def test_stats_accumulate(self):
        """Stats accumulate across multiple calls."""
        d = Deidentifier()
        d.deidentify_text("SSN: 123-45-6789")
        d.deidentify_text("Email: a@b.com")
        stats = d.get_stats()
        assert stats["phi_removed"] >= 2


class TestDeidentifyRecord:
    """Tests for record-level de-identification."""

    def test_deidentify_record_phi_field(self):
        """PHI fields in a record are de-identified."""
        d = Deidentifier(method=DeidentMethod.REDACT)
        record = {"name": "John Doe", "age": 45}
        result = d.deidentify_record(record)
        assert result.clean_data["name"] == "[REDACTED]"
        # Non-PHI field is preserved
        assert result.clean_data["age"] == 45

    def test_deidentify_record_increments_stats(self):
        """records_processed stat is incremented per call."""
        d = Deidentifier()
        d.deidentify_record({"name": "Alice"})
        d.deidentify_record({"email": "bob@test.com"})
        stats = d.get_stats()
        assert stats["records_processed"] == 2


class TestDeidentifyDataset:
    """Tests for batch dataset de-identification."""

    def test_deidentify_dataset_returns_list(self):
        """Returns a list of DeidentResult, one per record."""
        d = Deidentifier()
        records = [{"name": "A"}, {"email": "b@c.com"}]
        results = d.deidentify_dataset(records)
        assert len(results) == 2
        assert all(isinstance(r, DeidentResult) for r in results)
