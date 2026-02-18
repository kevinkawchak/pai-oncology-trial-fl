"""Tests for regulatory/fda-compliance/fda_submission_tracker.py — FDA tracker.

Covers FDASubmissionTracker creation, submission types, AI/ML component
model_type defaults to "unspecified", submission lifecycle states,
document management, review cycles, readiness reporting, and audit trail.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "fda_submission_tracker",
    "regulatory/fda-compliance/fda_submission_tracker.py",
)

FDASubmissionTracker = mod.FDASubmissionTracker
SubmissionType = mod.SubmissionType
SubmissionStatus = mod.SubmissionStatus
DeviceClass = mod.DeviceClass
DocumentCategory = mod.DocumentCategory
DocumentStatus = mod.DocumentStatus
ReviewCycleType = mod.ReviewCycleType
AIMLComponent = mod.AIMLComponent
SubmissionPackage = mod.SubmissionPackage


class TestSubmissionTypeEnum:
    """Tests for SubmissionType enum values."""

    def test_all_submission_types(self):
        """All five FDA pathways are defined."""
        assert SubmissionType.K510.value == "510k"
        assert SubmissionType.DE_NOVO.value == "de_novo"
        assert SubmissionType.PMA.value == "pma"
        assert SubmissionType.BREAKTHROUGH.value == "breakthrough_device"
        assert SubmissionType.PRESUBMISSION.value == "pre_submission"


class TestSubmissionStatusEnum:
    """Tests for SubmissionStatus lifecycle enum."""

    def test_terminal_statuses(self):
        """Terminal status values are correct."""
        assert SubmissionStatus.CLEARED.value == "cleared"
        assert SubmissionStatus.GRANTED.value == "granted"
        assert SubmissionStatus.APPROVED.value == "approved"
        assert SubmissionStatus.DENIED.value == "denied"
        assert SubmissionStatus.WITHDRAWN.value == "withdrawn"

    def test_active_statuses(self):
        """Active workflow statuses exist."""
        assert SubmissionStatus.PLANNING.value == "planning"
        assert SubmissionStatus.SUBMITTED.value == "submitted"
        assert SubmissionStatus.FDA_REVIEW.value == "fda_review"


class TestFDASubmissionTrackerCreation:
    """Tests for FDASubmissionTracker initialization."""

    def test_default_creation(self):
        """Newly created tracker has no submissions."""
        tracker = FDASubmissionTracker()
        assert tracker.list_submissions() == []

    def test_empty_audit_trail(self):
        """Newly created tracker has empty audit trail."""
        tracker = FDASubmissionTracker()
        assert tracker.get_audit_trail() == []


class TestCreateSubmission:
    """Tests for creating submission packages."""

    def test_create_510k_submission(self):
        """Creates a 510(k) submission with standard documents."""
        tracker = FDASubmissionTracker()
        pkg = tracker.create_submission(
            submission_id="SUB-001",
            submission_type=SubmissionType.K510,
            device_name="OncoAI Detector",
        )
        assert pkg.submission_id == "SUB-001"
        assert pkg.submission_type == SubmissionType.K510
        assert pkg.status == SubmissionStatus.PLANNING
        assert len(pkg.documents) > 0

    def test_create_pma_submission(self):
        """Creates a PMA submission with more documents than 510(k)."""
        tracker = FDASubmissionTracker()
        k510 = tracker.create_submission("K510", SubmissionType.K510, "Device A")
        pma = tracker.create_submission("PMA", SubmissionType.PMA, "Device B")
        assert len(pma.documents) > len(k510.documents)

    def test_create_with_string_type(self):
        """Submission type can be provided as string."""
        tracker = FDASubmissionTracker()
        pkg = tracker.create_submission("SUB-002", "510k", "DeviceX")
        assert pkg.submission_type == SubmissionType.K510

    def test_create_with_string_device_class(self):
        """Device class can be provided as string."""
        tracker = FDASubmissionTracker()
        pkg = tracker.create_submission("SUB-003", SubmissionType.PMA, "DeviceY", device_class="class_iii")
        assert pkg.device_class == DeviceClass.CLASS_III

    def test_create_with_predicate(self):
        """Predicate device is stored on 510(k) submissions."""
        tracker = FDASubmissionTracker()
        pkg = tracker.create_submission(
            "SUB-004",
            SubmissionType.K510,
            "DeviceZ",
            predicate_device="K123456",
        )
        assert pkg.predicate_device == "K123456"

    def test_create_logs_audit(self):
        """Submission creation creates an audit trail entry."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-005", SubmissionType.K510, "Dev")
        trail = tracker.get_audit_trail(submission_id="SUB-005")
        assert len(trail) >= 1
        assert trail[0]["action"] == "submission_created"

    def test_presubmission_has_fewer_docs(self):
        """Pre-submission has fewer documents than full submissions."""
        tracker = FDASubmissionTracker()
        pre = tracker.create_submission("PRE", SubmissionType.PRESUBMISSION, "Dev")
        k510 = tracker.create_submission("K510", SubmissionType.K510, "Dev")
        assert len(pre.documents) < len(k510.documents)


class TestAIMLComponent:
    """Tests for AI/ML component management."""

    def test_default_model_type_unspecified(self):
        """AI/ML component model_type defaults to 'unspecified'."""
        component = AIMLComponent(component_id="C1", name="Classifier")
        assert component.model_type == "unspecified"

    def test_add_ai_ml_component(self):
        """AI/ML component is added to the submission package."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        result = tracker.add_ai_ml_component(
            "SUB-001",
            component_id="C1",
            name="Tumor Classifier",
            model_type="CNN",
            input_types=["CT_scan"],
            output_types=["classification"],
        )
        assert result is True
        pkg = tracker.get_submission("SUB-001")
        assert len(pkg.ai_ml_components) == 1
        assert pkg.ai_ml_components[0].model_type == "CNN"

    def test_add_component_default_model_type(self):
        """model_type defaults to 'unspecified' when not provided."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        tracker.add_ai_ml_component("SUB-001", "C1", "Model")
        pkg = tracker.get_submission("SUB-001")
        assert pkg.ai_ml_components[0].model_type == "unspecified"

    def test_adaptive_component_pccp_applicable(self):
        """Adaptive AI/ML component has pccp_applicable=True."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        tracker.add_ai_ml_component("SUB-001", "C1", "Adaptive Model", locked_or_adaptive="adaptive")
        pkg = tracker.get_submission("SUB-001")
        assert pkg.ai_ml_components[0].pccp_applicable is True

    def test_add_component_to_nonexistent_submission(self):
        """Adding component to nonexistent submission returns False."""
        tracker = FDASubmissionTracker()
        assert tracker.add_ai_ml_component("FAKE", "C1", "Model") is False


class TestSubmissionLifecycle:
    """Tests for submission status transitions."""

    def test_update_status_to_submitted(self):
        """Moving to SUBMITTED sets submitted_at timestamp."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        tracker.update_submission_status("SUB-001", SubmissionStatus.SUBMITTED)
        pkg = tracker.get_submission("SUB-001")
        assert pkg.status == SubmissionStatus.SUBMITTED
        assert pkg.submitted_at != ""

    def test_update_status_to_cleared(self):
        """Moving to CLEARED sets decision_date."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        tracker.update_submission_status("SUB-001", SubmissionStatus.CLEARED)
        pkg = tracker.get_submission("SUB-001")
        assert pkg.status == SubmissionStatus.CLEARED
        assert pkg.decision_date != ""

    def test_update_status_nonexistent(self):
        """Updating status for nonexistent submission returns False."""
        tracker = FDASubmissionTracker()
        assert tracker.update_submission_status("FAKE", SubmissionStatus.SUBMITTED) is False


class TestDocumentManagement:
    """Tests for document status management."""

    def test_update_document_status(self):
        """Document status can be updated within a submission."""
        tracker = FDASubmissionTracker()
        pkg = tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        doc_id = pkg.documents[0].doc_id
        result = tracker.update_document_status("SUB-001", doc_id, DocumentStatus.IN_PROGRESS)
        assert result is True
        updated_pkg = tracker.get_submission("SUB-001")
        updated_doc = next(d for d in updated_pkg.documents if d.doc_id == doc_id)
        assert updated_doc.status == DocumentStatus.IN_PROGRESS

    def test_update_document_nonexistent_doc(self):
        """Updating nonexistent document returns False."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        assert tracker.update_document_status("SUB-001", "FAKE_DOC", DocumentStatus.FINALIZED) is False


class TestReadinessReport:
    """Tests for readiness report generation."""

    def test_readiness_report_basic(self):
        """Readiness report contains expected fields."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        report = tracker.get_readiness_report("SUB-001")
        assert report["submission_id"] == "SUB-001"
        assert report["submission_type"] == "510k"
        assert report["total_documents"] > 0
        assert report["ready_for_submission"] is False  # All NOT_STARTED
        assert "expected_review_days" in report

    def test_readiness_report_nonexistent(self):
        """Readiness report for nonexistent submission returns error."""
        tracker = FDASubmissionTracker()
        report = tracker.get_readiness_report("FAKE")
        assert "error" in report

    def test_readiness_when_all_finalized(self):
        """ready_for_submission is True when all docs are finalized."""
        tracker = FDASubmissionTracker()
        pkg = tracker.create_submission("SUB-001", SubmissionType.PRESUBMISSION, "Dev")
        for doc in pkg.documents:
            tracker.update_document_status("SUB-001", doc.doc_id, DocumentStatus.FINALIZED)
        report = tracker.get_readiness_report("SUB-001")
        assert report["ready_for_submission"] is True


class TestReviewCycles:
    """Tests for FDA review cycle management."""

    def test_add_review_cycle(self):
        """Review cycle is added to the submission."""
        tracker = FDASubmissionTracker()
        tracker.create_submission("SUB-001", SubmissionType.K510, "Dev")
        cycle_id = tracker.add_review_cycle(
            "SUB-001",
            ReviewCycleType.INITIAL_REVIEW,
            findings=["Clarify training data source"],
        )
        assert cycle_id is not None
        assert cycle_id.startswith("SUB-001_RC_")

    def test_add_review_cycle_nonexistent(self):
        """Adding review cycle to nonexistent submission returns None."""
        tracker = FDASubmissionTracker()
        assert tracker.add_review_cycle("FAKE", ReviewCycleType.INITIAL_REVIEW) is None
