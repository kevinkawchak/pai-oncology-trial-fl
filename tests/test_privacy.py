"""Tests for privacy modules."""

from privacy.access_control import AccessControlManager, Permission, Role
from privacy.audit_logger import AuditLogger, EventType, Severity
from privacy.breach_response import (
    BreachIndicator,
    BreachResponseProtocol,
    BreachSeverity,
    IncidentStatus,
)
from privacy.consent_manager import ConsentManager, ConsentStatus
from privacy.deidentification import Deidentifier
from privacy.phi_detector import PHIDetector


class TestPHIDetector:
    def test_detect_ssn(self):
        detector = PHIDetector()
        matches = detector.scan_text("SSN is 123-45-6789")
        assert any(m.phi_type == "ssn" for m in matches)

    def test_detect_email(self):
        detector = PHIDetector()
        matches = detector.scan_text("Contact: john@hospital.org")
        assert any(m.phi_type == "email" for m in matches)

    def test_detect_phone(self):
        detector = PHIDetector()
        matches = detector.scan_text("Phone: (555) 123-4567")
        assert any(m.phi_type == "phone_number" for m in matches)

    def test_detect_date(self):
        detector = PHIDetector()
        matches = detector.scan_text("DOB: 01/15/1980")
        assert any(m.phi_type == "dates" for m in matches)

    def test_detect_ip(self):
        detector = PHIDetector()
        matches = detector.scan_text("IP: 192.168.1.100")
        assert any(m.phi_type == "ip_address" for m in matches)

    def test_detect_mrn(self):
        detector = PHIDetector()
        matches = detector.scan_text("MRN: 12345678")
        assert any(m.phi_type == "mrn" for m in matches)

    def test_no_phi(self):
        detector = PHIDetector()
        matches = detector.scan_text("Tumor volume is 3.5 cm3")
        assert len(matches) == 0

    def test_has_phi(self):
        detector = PHIDetector()
        assert detector.has_phi("SSN 123-45-6789") is True
        assert detector.has_phi("No PHI here") is False

    def test_scan_record(self):
        detector = PHIDetector()
        record = {
            "patient_name": "John Doe",
            "tumor_volume": "3.5",
            "email": "john@example.com",
        }
        matches = detector.scan_record(record)
        assert len(matches) >= 2

    def test_scan_dataset(self):
        detector = PHIDetector()
        records = [
            {"name": "Alice", "age": "55"},
            {"diagnosis": "NSCLC", "stage": "IIIa"},
        ]
        results = detector.scan_dataset(records)
        assert "record_0" in results

    def test_hipaa_identifiers(self):
        identifiers = PHIDetector.get_hipaa_identifiers()
        assert len(identifiers) == 18


class TestDeidentifier:
    def test_redact_ssn(self):
        deid = Deidentifier(method="redact")
        result = deid.deidentify_text("SSN is 123-45-6789")
        assert "[REDACTED]" in result.clean_data
        assert "123-45-6789" not in result.clean_data

    def test_hash_method(self):
        deid = Deidentifier(method="hash")
        result = deid.deidentify_text("Email: test@example.com")
        assert "[HASH:" in result.clean_data

    def test_generalize_method(self):
        deid = Deidentifier(method="generalize")
        result = deid.deidentify_text("SSN is 123-45-6789")
        assert "[SSN]" in result.clean_data

    def test_no_phi_unchanged(self):
        deid = Deidentifier()
        result = deid.deidentify_text("Tumor volume is 3.5 cm3")
        assert result.clean_data == "Tumor volume is 3.5 cm3"
        assert result.original_phi_count == 0

    def test_deidentify_record(self):
        deid = Deidentifier()
        record = {
            "patient_name": "John Doe",
            "diagnosis": "NSCLC",
            "phone": "555-123-4567",
        }
        result = deid.deidentify_record(record)
        clean = result.clean_data
        assert clean["patient_name"] == "[REDACTED]"
        assert clean["diagnosis"] == "NSCLC"

    def test_deidentify_dataset(self):
        deid = Deidentifier()
        records = [
            {"name": "Alice", "age": "55"},
            {"name": "Bob", "diagnosis": "AML"},
        ]
        results = deid.deidentify_dataset(records)
        assert len(results) == 2

    def test_stats(self):
        deid = Deidentifier()
        deid.deidentify_text("SSN 123-45-6789")
        stats = deid.get_stats()
        assert stats["phi_removed"] >= 1


class TestConsentManager:
    def test_register_consent(self):
        mgr = ConsentManager()
        record = mgr.register_consent("P001", "STUDY_01")
        assert record.status == ConsentStatus.GRANTED

    def test_verify_consent(self):
        mgr = ConsentManager()
        mgr.register_consent("P001", "STUDY_01")
        assert mgr.verify_consent("P001", "STUDY_01") is True
        assert mgr.verify_consent("P001", "STUDY_02") is False

    def test_verify_required_use(self):
        mgr = ConsentManager()
        mgr.register_consent("P001", "STUDY_01", data_uses=["model_training"])
        assert mgr.verify_consent("P001", "STUDY_01", required_use="model_training")
        assert not mgr.verify_consent("P001", "STUDY_01", required_use="commercial_use")

    def test_revoke_consent(self):
        mgr = ConsentManager()
        mgr.register_consent("P001", "STUDY_01")
        assert mgr.revoke_consent("P001", "STUDY_01") is True
        assert mgr.verify_consent("P001", "STUDY_01") is False

    def test_revoke_nonexistent(self):
        mgr = ConsentManager()
        assert mgr.revoke_consent("P999", "STUDY_99") is False

    def test_get_consented_patients(self):
        mgr = ConsentManager()
        mgr.register_consent("P001", "STUDY_01")
        mgr.register_consent("P002", "STUDY_01")
        mgr.register_consent("P003", "STUDY_02")
        patients = mgr.get_consented_patients("STUDY_01")
        assert set(patients) == {"P001", "P002"}

    def test_audit_trail(self):
        mgr = ConsentManager()
        mgr.register_consent("P001", "STUDY_01")
        mgr.revoke_consent("P001", "STUDY_01")
        trail = mgr.get_audit_trail()
        assert len(trail) == 2
        assert trail[0]["event"] == "consent_granted"
        assert trail[1]["event"] == "consent_revoked"


class TestAuditLogger:
    def test_log_event(self):
        al = AuditLogger()
        event = al.log(
            EventType.DATA_ACCESS,
            actor="researcher_1",
            resource="patient_data",
            action="read",
        )
        assert event.event_type == EventType.DATA_ACCESS
        assert event.integrity_hash

    def test_log_data_access(self):
        al = AuditLogger()
        event = al.log_data_access("user_1", "dataset_A", "model_training")
        assert event.event_type == EventType.DATA_ACCESS

    def test_log_breach(self):
        al = AuditLogger()
        event = al.log_breach("system", {"type": "unauthorized_access"})
        assert event.severity == Severity.CRITICAL

    def test_get_events_filtered(self):
        al = AuditLogger()
        al.log(EventType.DATA_ACCESS, "user_1", "data", "read")
        al.log(EventType.MODEL_TRAINING, "user_2", "model", "train")
        al.log(EventType.DATA_ACCESS, "user_1", "data2", "read")

        access_events = al.get_events(event_type=EventType.DATA_ACCESS)
        assert len(access_events) == 2

        user1_events = al.get_events(actor="user_1")
        assert len(user1_events) == 2

    def test_generate_report(self):
        al = AuditLogger()
        al.log(EventType.DATA_ACCESS, "user_1", "data", "read")
        al.log(EventType.MODEL_TRAINING, "user_2", "model", "train")
        report = al.generate_report()
        assert report["total_events"] == 2

    def test_verify_integrity(self):
        al = AuditLogger()
        al.log(EventType.DATA_ACCESS, "user_1", "data", "read")
        failed = al.verify_integrity()
        assert len(failed) == 0


class TestAccessControlManager:
    def test_register_user(self):
        acm = AccessControlManager()
        user = acm.register_user("u1", "Admin", Role.COORDINATOR)
        assert user.role == Role.COORDINATOR
        assert user.active is True

    def test_check_permission_granted(self):
        acm = AccessControlManager()
        acm.register_user("u1", "Admin", Role.COORDINATOR)
        assert acm.check_permission("u1", Permission.START_FEDERATION) is True

    def test_check_permission_denied(self):
        acm = AccessControlManager()
        acm.register_user("u1", "Researcher", Role.RESEARCHER)
        assert acm.check_permission("u1", Permission.START_FEDERATION) is False

    def test_deactivate_user(self):
        acm = AccessControlManager()
        acm.register_user("u1", "Admin", Role.COORDINATOR)
        acm.deactivate_user("u1")
        assert acm.check_permission("u1", Permission.START_FEDERATION) is False

    def test_grant_extra_permission(self):
        acm = AccessControlManager()
        acm.register_user("u1", "Researcher", Role.RESEARCHER)
        assert acm.check_permission("u1", Permission.START_FEDERATION) is False
        acm.grant_permission("u1", Permission.START_FEDERATION)
        assert acm.check_permission("u1", Permission.START_FEDERATION) is True

    def test_revoke_permission(self):
        acm = AccessControlManager()
        acm.register_user("u1", "Admin", Role.COORDINATOR)
        acm.revoke_permission("u1", Permission.START_FEDERATION)
        assert acm.check_permission("u1", Permission.START_FEDERATION) is False

    def test_access_log(self):
        acm = AccessControlManager()
        acm.register_user("u1", "Admin", Role.COORDINATOR)
        acm.check_permission("u1", Permission.START_FEDERATION)
        log = acm.get_access_log(user_id="u1")
        assert len(log) >= 1


class TestBreachResponseProtocol:
    def test_create_incident(self):
        brp = BreachResponseProtocol()
        iid = brp.create_incident(severity=BreachSeverity.HIGH)
        inc = brp.get_incident(iid)
        assert inc is not None
        assert inc.severity == BreachSeverity.HIGH

    def test_incident_lifecycle(self):
        brp = BreachResponseProtocol()
        iid = brp.create_incident(severity=BreachSeverity.MEDIUM)
        assert brp.investigate(iid) is True
        assert brp.contain(iid, "isolated affected system") is True
        assert brp.resolve(iid, root_cause="misconfigured ACL") is True
        assert brp.report_to_authorities(iid) is True
        inc = brp.get_incident(iid)
        assert inc.status == IncidentStatus.REPORTED

    def test_auto_escalation(self):
        brp = BreachResponseProtocol(auto_escalate_threshold=2)
        # First indicator — no incident yet
        result1 = brp.report_indicator(BreachIndicator("failed_auth", "Failed login attempt"))
        assert result1 is None
        # Second indicator — should auto-escalate
        result2 = brp.report_indicator(BreachIndicator("anomalous_access", "Unusual data access pattern"))
        assert result2 is not None
        assert result2.startswith("INC-")

    def test_get_open_incidents(self):
        brp = BreachResponseProtocol()
        iid = brp.create_incident()
        open_incs = brp.get_open_incidents()
        assert len(open_incs) == 1
        brp.resolve(iid)
        open_incs = brp.get_open_incidents()
        assert len(open_incs) == 0
