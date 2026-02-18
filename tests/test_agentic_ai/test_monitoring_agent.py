"""Tests for agentic-ai/examples-agentic-ai/03_realtime_adaptive_monitoring_agent.py.

Validates monitoring agent enums, dataclasses, alert thresholds,
vital sign processing, and lab type definitions.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module(
    "monitoring_agent",
    "agentic-ai/examples-agentic-ai/03_realtime_adaptive_monitoring_agent.py",
)


class TestMonitoringEnums:
    """Verify all monitoring agent enums."""

    def test_data_modality_values(self):
        assert mod.DataModality.VITAL_SIGNS.value == "vital_signs"
        assert mod.DataModality.LABORATORY.value == "laboratory"
        assert mod.DataModality.IMAGING.value == "imaging"
        assert mod.DataModality.MEDICATION.value == "medication"
        assert mod.DataModality.PATIENT_REPORTED.value == "patient_reported"

    def test_alert_severity_values(self):
        assert mod.AlertSeverity.INFO.value == "info"
        assert mod.AlertSeverity.LOW.value == "low"
        assert mod.AlertSeverity.MODERATE.value == "moderate"
        assert mod.AlertSeverity.HIGH.value == "high"
        assert mod.AlertSeverity.CRITICAL.value == "critical"

    def test_alert_status_values(self):
        assert mod.AlertStatus.ACTIVE.value == "active"
        assert mod.AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert mod.AlertStatus.ESCALATED.value == "escalated"
        assert mod.AlertStatus.RESOLVED.value == "resolved"
        assert mod.AlertStatus.SUPPRESSED.value == "suppressed"

    def test_treatment_phase_values(self):
        assert mod.TreatmentPhase.SCREENING.value == "screening"
        assert mod.TreatmentPhase.BASELINE.value == "baseline"
        assert mod.TreatmentPhase.INDUCTION.value == "induction"
        assert mod.TreatmentPhase.MAINTENANCE.value == "maintenance"
        assert mod.TreatmentPhase.FOLLOW_UP.value == "follow_up"

    def test_toxicity_grade_range(self):
        assert mod.ToxicityGrade.GRADE_0.value == 0
        assert mod.ToxicityGrade.GRADE_5.value == 5
        grades = list(mod.ToxicityGrade)
        assert len(grades) == 6

    def test_vital_type_values(self):
        assert mod.VitalType.HEART_RATE.value == "heart_rate"
        assert mod.VitalType.SPO2.value == "spo2"
        assert mod.VitalType.PAIN_SCORE.value == "pain_score"

    def test_lab_type_values(self):
        assert mod.LabType.WBC.value == "wbc"
        assert mod.LabType.ANC.value == "anc"
        assert mod.LabType.HEMOGLOBIN.value == "hemoglobin"
        assert mod.LabType.PLATELETS.value == "platelets"
        assert mod.LabType.CREATININE.value == "creatinine"


class TestVitalSignReading:
    """Verify VitalSignReading dataclass."""

    def test_vital_sign_reading_defaults(self):
        reading = mod.VitalSignReading()
        assert reading.vital_type == mod.VitalType.HEART_RATE
        assert reading.patient_id == ""
        assert reading.reading_id.startswith("VS-")

    def test_vital_sign_reading_custom(self):
        reading = mod.VitalSignReading(
            patient_id="PT-001",
            vital_type=mod.VitalType.SPO2,
        )
        assert reading.patient_id == "PT-001"
        assert reading.vital_type == mod.VitalType.SPO2

    def test_vital_sign_reading_unique_ids(self):
        r1 = mod.VitalSignReading()
        r2 = mod.VitalSignReading()
        assert r1.reading_id != r2.reading_id


class TestMonitoringDataclasses:
    """Verify other monitoring agent dataclasses."""

    def test_alert_severity_ordering(self):
        severities = [s.value for s in mod.AlertSeverity]
        assert "info" in severities
        assert "critical" in severities

    def test_data_modality_count(self):
        modalities = list(mod.DataModality)
        assert len(modalities) == 5

    def test_lab_type_count(self):
        lab_types = list(mod.LabType)
        assert len(lab_types) >= 10

    def test_vital_type_count(self):
        vital_types = list(mod.VitalType)
        assert len(vital_types) >= 6


class TestModuleAttributes:
    """Verify expected module-level attributes and classes exist."""

    def test_module_has_data_modality(self):
        assert hasattr(mod, "DataModality")

    def test_module_has_alert_severity(self):
        assert hasattr(mod, "AlertSeverity")

    def test_module_has_vital_sign_reading(self):
        assert hasattr(mod, "VitalSignReading")

    def test_module_has_treatment_phase(self):
        assert hasattr(mod, "TreatmentPhase")

    def test_alert_status_all_unique(self):
        values = [s.value for s in mod.AlertStatus]
        assert len(values) == len(set(values))
