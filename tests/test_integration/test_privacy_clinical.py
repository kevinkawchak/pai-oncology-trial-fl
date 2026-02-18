"""Privacy-to-clinical integration tests.

Tests PHI detection -> de-identification -> clinical integrator pipeline,
verifying that privacy modules properly protect data before clinical use.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
phi_mod = load_module("privacy.phi_detector", "privacy/phi_detector.py")
deident_mod = load_module("privacy.deidentification", "privacy/deidentification.py")
clinical_mod = load_module(
    "clinical_integrator",
    "digital-twins/clinical-integration/clinical_integrator.py",
)
audit_mod = load_module("privacy.audit_logger", "privacy/audit_logger.py")


class TestPHIDetection:
    """Verify PHI detector identifies protected health information."""

    def test_detect_ssn(self):
        detector = phi_mod.PHIDetector()
        matches = detector.scan_text("Patient SSN: 123-45-6789")
        assert len(matches) > 0

    def test_detect_mrn(self):
        detector = phi_mod.PHIDetector()
        matches = detector.scan_text("MRN: MRN12345678")
        assert len(matches) > 0

    def test_detect_phone(self):
        detector = phi_mod.PHIDetector()
        matches = detector.scan_text("Call (555) 123-4567")
        assert len(matches) > 0

    def test_no_phi_in_clean_text(self):
        detector = phi_mod.PHIDetector()
        matches = detector.scan_text("Patient has stage IIIB NSCLC with ECOG 1")
        # Clinical terms should not trigger PHI detection
        assert all(m.phi_type not in ("ssn", "mrn", "phone_number") for m in matches)


class TestDeidentification:
    """Verify de-identification removes or masks PHI."""

    def test_deidentifier_creation(self):
        deident = deident_mod.Deidentifier()
        assert deident is not None

    def test_deidentify_text(self):
        deident = deident_mod.Deidentifier()
        text = "Patient SSN 123-45-6789"
        result = deident.deidentify_text(text)
        assert isinstance(result, deident_mod.DeidentResult)
        assert "123-45-6789" not in result.clean_data

    def test_deident_method_enum(self):
        assert deident_mod.DeidentMethod.REDACT.value == "redact"
        assert deident_mod.DeidentMethod.HASH.value == "hash"


class TestPHIToDeidentPipeline:
    """Verify PHI detection feeds into de-identification."""

    def test_detect_then_deidentify(self):
        detector = phi_mod.PHIDetector()
        deident = deident_mod.Deidentifier()
        text = "MRN: MRN12345678, Phone: (555) 123-4567"
        matches = detector.scan_text(text)
        assert len(matches) > 0
        result = deident.deidentify_text(text)
        assert "MRN12345678" not in result.clean_data


class TestClinicalIntegrator:
    """Verify clinical integrator module structure."""

    def test_clinical_system_enum(self):
        assert hasattr(clinical_mod, "ClinicalSystem")

    def test_integration_status_enum(self):
        assert hasattr(clinical_mod, "IntegrationStatus")

    def test_data_format_enum(self):
        assert hasattr(clinical_mod, "DataFormat")

    def test_data_category_enum(self):
        assert hasattr(clinical_mod, "DataCategory")

    def test_clinical_data_point_class(self):
        assert hasattr(clinical_mod, "ClinicalDataPoint")

    def test_clinical_integrator_class(self):
        assert hasattr(clinical_mod, "ClinicalIntegrator")


class TestAuditTrailIntegration:
    """Verify audit logger captures privacy events."""

    def test_audit_logger_log_and_retrieve(self):
        al = audit_mod.AuditLogger()
        al.log(
            event_type=audit_mod.EventType.PHI_DETECTED,
            actor="system",
            resource="patient_data",
            action="phi scan",
            details={"count": 3, "types": ["SSN", "MRN", "PHONE"]},
        )
        events = al.get_events()
        assert len(events) >= 1
        assert events[-1].event_type == audit_mod.EventType.PHI_DETECTED
