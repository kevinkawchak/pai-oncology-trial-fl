"""Tests for regulatory-submissions/submission_orchestrator.py.

Covers enums (SubmissionPhase, SubmissionType, WorkflowStatus, TaskPriority,
AuditAction), dataclasses (SubmissionConfig, SubmissionMilestone, WorkflowTask,
AuditTrailEntry, SubmissionManifest), and RegulatorySubmissionOrchestrator:
  - Initialization and configuration
  - Task creation and status management
  - Phase advancement with dependency checks
  - Milestone tracking and deadline alerting
  - Reviewer assignment
  - Document registration and integrity verification
  - Audit trail (21 CFR Part 11 compliance)
  - Manifest compilation
  - Status report generation
  - Workflow integrity validation
  - Standard 510(k) factory
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
    "submission_orchestrator",
    "regulatory-submissions/submission_orchestrator.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_config(**overrides):
    defaults = {
        "submission_id": "SUB-TEST-001",
        "submission_type": mod.SubmissionType.ECTD_510K,
        "sponsor_name": "Test Sponsor",
        "device_name": "Test Device v1.0",
    }
    defaults.update(overrides)
    return mod.SubmissionConfig(**defaults)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestSubmissionPhase:
    def test_members(self):
        names = set(mod.SubmissionPhase.__members__.keys())
        assert "PLANNING" in names
        assert "SUBMISSION" in names
        assert "TRACKING" in names

    def test_values_are_strings(self):
        assert isinstance(mod.SubmissionPhase.PLANNING, str)
        assert mod.SubmissionPhase.PLANNING.value == "planning"

    def test_phase_count(self):
        assert len(mod.SubmissionPhase) == 7


class TestSubmissionType:
    def test_members(self):
        names = set(mod.SubmissionType.__members__.keys())
        assert "ECTD_510K" in names
        assert "ECTD_PMA" in names
        assert "IND_INITIAL" in names
        assert "BRIEFING_DOCUMENT" in names

    def test_type_count(self):
        assert len(mod.SubmissionType) == 8


class TestWorkflowStatus:
    def test_members(self):
        names = set(mod.WorkflowStatus.__members__.keys())
        assert "DRAFT" in names
        assert "SUBMITTED" in names
        assert "ACKNOWLEDGED" in names

    def test_status_count(self):
        assert len(mod.WorkflowStatus) == 6


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestSubmissionConfig:
    def test_creation(self):
        cfg = _make_config()
        assert cfg.submission_id == "SUB-TEST-001"
        assert cfg.submission_type == mod.SubmissionType.ECTD_510K

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            mod.SubmissionConfig(
                submission_id="",
                submission_type=mod.SubmissionType.ECTD_510K,
                sponsor_name="Test",
                device_name="Test",
            )

    def test_empty_sponsor_raises(self):
        with pytest.raises(ValueError):
            mod.SubmissionConfig(
                submission_id="SUB-1",
                submission_type=mod.SubmissionType.ECTD_510K,
                sponsor_name="",
                device_name="Test",
            )

    def test_invalid_review_days_defaults(self):
        cfg = _make_config(review_days_target=-5)
        assert cfg.review_days_target == mod.DEFAULT_REVIEW_DAYS


class TestSubmissionMilestone:
    def test_creation_defaults(self):
        ms = mod.SubmissionMilestone(name="Test Milestone")
        assert ms.name == "Test Milestone"
        assert not ms.is_completed
        assert ms.milestone_id

    def test_days_until_target(self):
        ms = mod.SubmissionMilestone(target_date="2099-12-31")
        days = ms.days_until_target("2099-12-01")
        assert days == 30

    def test_mark_completed(self):
        ms = mod.SubmissionMilestone()
        ms.mark_completed("2026-01-15")
        assert ms.is_completed
        assert ms.actual_date == "2026-01-15"


class TestAuditTrailEntry:
    def test_compute_hash(self):
        entry = mod.AuditTrailEntry(action=mod.AuditAction.CREATE, user="tester", entity_id="E1", new_value="v1")
        h = entry.compute_hash()
        assert len(h) == 64
        assert entry.content_hash == h


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------
class TestRegulatorySubmissionOrchestrator:
    def test_init(self):
        cfg = _make_config()
        orch = mod.RegulatorySubmissionOrchestrator(cfg)
        assert orch.current_phase == mod.SubmissionPhase.PLANNING
        assert orch.status == mod.WorkflowStatus.DRAFT
        assert len(orch.audit_trail) >= 1

    def test_add_task(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        task = orch.add_task("Task 1", mod.SubmissionPhase.PLANNING)
        assert task.name == "Task 1"
        assert orch.task_count == 1

    def test_update_task_status(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        task = orch.add_task("Task 1")
        ok = orch.update_task_status(task.task_id, mod.WorkflowStatus.APPROVED)
        assert ok is True

    def test_update_nonexistent_task_fails(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        ok = orch.update_task_status("nonexistent", mod.WorkflowStatus.APPROVED)
        assert ok is False

    def test_advance_phase(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        task = orch.add_task("T1", mod.SubmissionPhase.PLANNING)
        orch.update_task_status(task.task_id, mod.WorkflowStatus.APPROVED)
        new_phase = orch.advance_phase()
        assert new_phase == mod.SubmissionPhase.AUTHORING

    def test_advance_phase_blocked(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        orch.add_task("T1", mod.SubmissionPhase.PLANNING)
        phase = orch.advance_phase()
        assert phase == mod.SubmissionPhase.PLANNING

    def test_milestone_management(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        ms = orch.add_milestone("MS1", mod.SubmissionPhase.PLANNING, "2026-12-31")
        assert ms.name == "MS1"
        ok = orch.complete_milestone(ms.milestone_id, "2026-06-01")
        assert ok is True

    def test_register_document(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        orch.register_document("DOC-1", "Test Document")
        assert orch.verify_document_integrity("DOC-1", "") is False

    def test_assign_reviewer(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        task = orch.add_task("Task 1")
        assignment = orch.assign_reviewer("Dr. Smith", "Clinical", task.task_id)
        assert assignment is not None
        assert assignment.reviewer_name == "Dr. Smith"

    def test_compile_manifest(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        orch.add_task("T1")
        orch.register_document("D1", "Doc 1")
        manifest = orch.compile_manifest()
        assert manifest.submission_id == "SUB-TEST-001"
        assert manifest.total_documents == 1
        assert manifest.integrity_hash

    def test_status_report(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        orch.add_task("T1")
        report = orch.generate_status_report()
        assert "SUB-TEST-001" in report
        assert "Milestone" in report

    def test_workflow_integrity(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        orch.add_task("T1")
        is_valid, issues = orch.validate_workflow_integrity()
        assert is_valid is True
        assert len(issues) == 0

    def test_completion_percentage(self):
        orch = mod.RegulatorySubmissionOrchestrator(_make_config())
        t1 = orch.add_task("T1")
        t2 = orch.add_task("T2")
        orch.update_task_status(t1.task_id, mod.WorkflowStatus.APPROVED)
        pct = orch.compute_completion_percentage()
        assert pct == 50.0


class TestStandard510kFactory:
    def test_factory_creates_tasks(self):
        cfg = _make_config()
        orch = mod.create_standard_510k_workflow(cfg)
        assert orch.task_count > 10

    def test_factory_creates_milestones(self):
        cfg = _make_config()
        orch = mod.create_standard_510k_workflow(cfg)
        manifest = orch.compile_manifest()
        assert manifest.total_sections > 0
