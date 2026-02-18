"""Tests for clinical-analytics/trial_data_manager.py.

Covers enums (DataDomain with 10 members, DataQualityLevel, RetentionPolicy,
ConsentScope), dataclasses (ClinicalDataset, DataQualityReport,
SiteDataInventory), and the TrialDataManager class: register dataset,
quality assessment (scores bounded [0,1]), schema validation, site
inventory, retention policy, audit trail (defensive copy), data manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import load_module

mod = load_module(
    "trial_data_manager",
    "clinical-analytics/trial_data_manager.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestDataDomain:
    """Tests for the DataDomain enum."""

    def test_data_domain_has_10_members(self):
        """DataDomain has exactly 10 clinical data domains."""
        assert len(mod.DataDomain) == 10

    def test_data_domain_demographics(self):
        """DEMOGRAPHICS domain exists with correct value."""
        assert mod.DataDomain.DEMOGRAPHICS.value == "demographics"

    def test_data_domain_laboratory(self):
        """LABORATORY domain exists."""
        assert mod.DataDomain.LABORATORY.value == "laboratory"

    def test_data_domain_adverse_events(self):
        """ADVERSE_EVENTS domain exists."""
        assert mod.DataDomain.ADVERSE_EVENTS.value == "adverse_events"

    def test_data_domain_is_str_enum(self):
        """DataDomain members are instances of str."""
        assert isinstance(mod.DataDomain.DEMOGRAPHICS, str)


class TestDataQualityLevel:
    """Tests for the DataQualityLevel enum."""

    def test_quality_levels(self):
        """Five quality tiers are defined."""
        expected = {"EXCELLENT", "GOOD", "ACCEPTABLE", "POOR", "UNACCEPTABLE"}
        assert set(mod.DataQualityLevel.__members__.keys()) == expected


class TestRetentionPolicy:
    """Tests for the RetentionPolicy enum."""

    def test_retention_states(self):
        """Four retention lifecycle states are defined."""
        expected = {"ACTIVE", "ARCHIVED", "PENDING_DESTRUCTION", "DESTROYED"}
        assert set(mod.RetentionPolicy.__members__.keys()) == expected


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestClinicalDataset:
    """Tests for the ClinicalDataset dataclass."""

    def test_creation(self):
        """ClinicalDataset is created with required fields."""
        ds = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
        )
        assert ds.dataset_id == "DS-001"
        assert ds.site_id == "SITE-A"
        assert ds.created_at != ""

    def test_default_quality_level(self):
        """Default quality level is ACCEPTABLE."""
        ds = mod.ClinicalDataset(
            dataset_id="DS-002",
            domain=mod.DataDomain.LABORATORY,
            site_id="SITE-B",
        )
        assert ds.quality_level == mod.DataQualityLevel.ACCEPTABLE


class TestDataQualityReport:
    """Tests for the DataQualityReport dataclass."""

    def test_scores_clamped_to_01(self):
        """Quality scores are clamped to [0, 1]."""
        report = mod.DataQualityReport(
            completeness=1.5,
            accuracy=-0.3,
            consistency=0.5,
            timeliness=2.0,
            composite_score=3.0,
        )
        assert 0.0 <= report.completeness <= 1.0
        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.consistency <= 1.0
        assert 0.0 <= report.timeliness <= 1.0
        assert 0.0 <= report.composite_score <= 1.0


# ---------------------------------------------------------------------------
# TrialDataManager tests
# ---------------------------------------------------------------------------
class TestTrialDataManager:
    """Tests for the TrialDataManager lifecycle and methods."""

    def test_init(self):
        """Manager initializes for a given trial ID."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        assert mgr.trial_id == "NCT-001"

    def test_register_dataset(self):
        """register_dataset stores and returns a ClinicalDataset."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        ds = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
            record_count=100,
        )
        result = mgr.register_dataset(ds)
        assert isinstance(result, mod.ClinicalDataset)
        assert result.dataset_id == "DS-001"
        assert result.record_count == 100

    def test_register_and_retrieve(self):
        """Registered dataset is retrievable by ID."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        ds = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.VITALS,
            site_id="SITE-A",
        )
        mgr.register_dataset(ds)
        retrieved = mgr.get_dataset("DS-001")
        assert retrieved is not None
        assert retrieved.domain == mod.DataDomain.VITALS

    def test_register_nonexistent_retrieve_none(self):
        """Retrieving unregistered dataset returns None."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        assert mgr.get_dataset("FAKE-ID") is None

    def test_quality_assessment_scores_bounded(self):
        """Quality assessment scores are all in [0, 1]."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        records = [
            {"subject_id": "P1", "age": 55, "sex": "M"},
            {"subject_id": "P2", "age": None, "sex": "F"},
        ]
        ds = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
            record_count=2,
        )
        mgr.register_dataset(ds, records=records)
        report = mgr.assess_data_quality("DS-001")
        assert isinstance(report, mod.DataQualityReport)
        assert 0.0 <= report.completeness <= 1.0
        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.consistency <= 1.0
        assert 0.0 <= report.timeliness <= 1.0
        assert 0.0 <= report.composite_score <= 1.0

    def test_schema_validation_valid(self):
        """Schema validation passes when all required fields are present."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        schema = mod.SchemaDefinition(
            fields=["subject_id", "age", "sex"],
            required=["subject_id"],
        )
        records = [{"subject_id": "P1", "age": 60, "sex": "M"}]
        violations = mgr.validate_schema(records, schema)
        assert len(violations) == 0

    def test_schema_validation_missing_required(self):
        """Schema validation fails when required fields are missing."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        schema = mod.SchemaDefinition(
            fields=["subject_id", "lab_test"],
            required=["subject_id", "lab_test"],
        )
        records = [{"subject_id": "P1"}]  # missing lab_test
        violations = mgr.validate_schema(records, schema)
        assert len(violations) > 0

    def test_site_inventory(self):
        """compute_site_inventory returns site-level summary."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        ds1 = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
            record_count=50,
        )
        ds2 = mod.ClinicalDataset(
            dataset_id="DS-002",
            domain=mod.DataDomain.LABORATORY,
            site_id="SITE-A",
            record_count=100,
        )
        mgr.register_dataset(ds1)
        mgr.register_dataset(ds2)
        inv = mgr.compute_site_inventory("SITE-A")
        assert isinstance(inv, mod.SiteDataInventory)
        assert inv.total_records == 150

    def test_retention_policy_application(self):
        """Retention policy can be applied to a dataset."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        ds = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
        )
        mgr.register_dataset(ds)
        result = mgr.apply_retention_policy("DS-001", mod.RetentionPolicy.ARCHIVED)
        assert isinstance(result, mod.DataRetentionRecord)
        assert result.retention_policy == mod.RetentionPolicy.ARCHIVED
        rec = mgr.get_retention_record("DS-001")
        assert rec is not None
        assert rec.retention_policy == mod.RetentionPolicy.ARCHIVED

    def test_audit_trail_populated(self):
        """Audit trail is populated after operations."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        ds = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
        )
        mgr.register_dataset(ds)
        trail = mgr.get_audit_trail()
        assert len(trail) >= 1

    def test_audit_trail_defensive_copy(self):
        """get_audit_trail returns a defensive copy."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        ds = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
        )
        mgr.register_dataset(ds)
        trail1 = mgr.get_audit_trail()
        trail1.append({"fake": "entry"})
        trail2 = mgr.get_audit_trail()
        assert len(trail2) < len(trail1)

    def test_generate_data_manifest(self):
        """Data manifest includes registered datasets and integrity hash."""
        mgr = mod.TrialDataManager(trial_id="NCT-001")
        ds1 = mod.ClinicalDataset(
            dataset_id="DS-001",
            domain=mod.DataDomain.DEMOGRAPHICS,
            site_id="SITE-A",
            record_count=10,
        )
        ds2 = mod.ClinicalDataset(
            dataset_id="DS-002",
            domain=mod.DataDomain.LABORATORY,
            site_id="SITE-B",
            record_count=20,
        )
        mgr.register_dataset(ds1)
        mgr.register_dataset(ds2)
        manifest = mgr.generate_data_manifest()
        assert manifest["trial_id"] == "NCT-001"
        assert "manifest_integrity_hash" in manifest
        assert len(manifest["datasets"]) >= 2
