#!/usr/bin/env python3
"""Regulatory submission orchestrator -- lifecycle management for FDA/EMA/PMDA submissions.

RESEARCH USE ONLY -- Not intended for clinical deployment without validation.

This module implements the core orchestration layer for regulatory submission
workflows targeting AI/ML-enabled medical devices and combination products in
oncology clinical trials.  It manages the full submission lifecycle from initial
planning through post-submission tracking, including task dependency resolution,
milestone management, reviewer assignments, deadline enforcement, and 21 CFR
Part 11-compliant audit logging.

Supported submission types:
    * eCTD 510(k) -- traditional, special, abbreviated, and De Novo pathways
    * eCTD PMA -- premarket approval for Class III devices
    * eCTD BLA -- biologics license applications
    * IND -- initial, amendments, and annual reports
    * Briefing documents -- pre-submission and mid-cycle meetings

All data structures are in-memory dataclasses.  No network calls or external
API dependencies.  Audit trail entries are append-only and include SHA-256
content hashes for tamper detection.

DISCLAIMER: RESEARCH USE ONLY.  Not validated for regulatory use.
LICENSE: MIT
VERSION: 0.9.1
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_TASKS_PER_PHASE: int = 200
MAX_MILESTONES: int = 100
MAX_REVIEWERS: int = 50
MAX_AUDIT_ENTRIES: int = 10000
DEFAULT_REVIEW_DAYS: int = 30
HASH_ALGORITHM: str = "sha256"


# ===================================================================
# Enums
# ===================================================================
class SubmissionPhase(str, Enum):
    """Lifecycle phases of a regulatory submission."""

    PLANNING = "planning"
    AUTHORING = "authoring"
    REVIEW = "review"
    COMPILATION = "compilation"
    VALIDATION = "validation"
    SUBMISSION = "submission"
    TRACKING = "tracking"


class SubmissionType(str, Enum):
    """FDA/EMA submission pathway types."""

    ECTD_510K = "ectd_510k"
    ECTD_PMA = "ectd_pma"
    ECTD_DE_NOVO = "ectd_de_novo"
    ECTD_BLA = "ectd_bla"
    IND_INITIAL = "ind_initial"
    IND_AMENDMENT = "ind_amendment"
    IND_ANNUAL_REPORT = "ind_annual_report"
    BRIEFING_DOCUMENT = "briefing_document"


class WorkflowStatus(str, Enum):
    """Status of an individual workflow task or the overall submission."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"


class TaskPriority(str, Enum):
    """Priority levels for workflow tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AuditAction(str, Enum):
    """Actions recorded in the 21 CFR Part 11 audit trail."""

    CREATE = "create"
    UPDATE = "update"
    APPROVE = "approve"
    REJECT = "reject"
    SUBMIT = "submit"
    SIGN = "sign"
    REVIEW = "review"
    COMPILE = "compile"
    VALIDATE = "validate"


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class SubmissionConfig:
    """Top-level configuration for a regulatory submission."""

    submission_id: str
    submission_type: SubmissionType
    sponsor_name: str
    device_name: str
    target_date: Optional[str] = None
    regulatory_authority: str = "FDA"
    contact_email: str = ""
    protocol_number: str = ""
    indication: str = "oncology"
    predicate_device: str = ""
    classification_panel: str = ""
    product_code: str = ""
    review_days_target: int = DEFAULT_REVIEW_DAYS
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.submission_id:
            raise ValueError("submission_id must not be empty")
        if not self.sponsor_name:
            raise ValueError("sponsor_name must not be empty")
        if self.review_days_target < 1:
            self.review_days_target = DEFAULT_REVIEW_DAYS


@dataclass
class SubmissionMilestone:
    """A tracked milestone within the submission lifecycle."""

    milestone_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    phase: SubmissionPhase = SubmissionPhase.PLANNING
    target_date: str = ""
    actual_date: str = ""
    is_completed: bool = False
    owner: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)

    def days_until_target(self, reference_date: Optional[str] = None) -> Optional[int]:
        """Return days until target date from reference (or today)."""
        if not self.target_date:
            return None
        try:
            target = datetime.fromisoformat(self.target_date)
            ref = datetime.fromisoformat(reference_date) if reference_date else datetime.now()
            return (target - ref).days
        except (ValueError, TypeError):
            return None

    def mark_completed(self, completion_date: Optional[str] = None) -> None:
        """Mark milestone as completed."""
        self.is_completed = True
        self.actual_date = completion_date or datetime.now().isoformat()


@dataclass
class WorkflowTask:
    """A single task within the submission workflow."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    phase: SubmissionPhase = SubmissionPhase.PLANNING
    status: WorkflowStatus = WorkflowStatus.DRAFT
    priority: TaskPriority = TaskPriority.MEDIUM
    assignee: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""
    due_date: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    review_comments: List[str] = field(default_factory=list)

    def is_blocked(self, completed_tasks: Sequence[str]) -> bool:
        """Check whether this task is blocked by incomplete dependencies."""
        for dep in self.dependencies:
            if dep not in completed_tasks:
                return True
        return False


@dataclass
class AuditTrailEntry:
    """21 CFR Part 11 compliant audit trail entry."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    action: AuditAction = AuditAction.CREATE
    user: str = ""
    entity_type: str = ""
    entity_id: str = ""
    previous_value: str = ""
    new_value: str = ""
    content_hash: str = ""
    reason: str = ""
    electronic_signature: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of entry content for tamper detection."""
        payload = f"{self.timestamp}|{self.action.value}|{self.user}|{self.entity_id}|{self.new_value}"
        self.content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.content_hash


@dataclass
class SubmissionManifest:
    """Complete manifest describing a submission package."""

    manifest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = ""
    submission_type: SubmissionType = SubmissionType.ECTD_510K
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: WorkflowStatus = WorkflowStatus.DRAFT
    total_documents: int = 0
    total_sections: int = 0
    phases_completed: List[str] = field(default_factory=list)
    pending_tasks: int = 0
    completed_tasks: int = 0
    compliance_score: float = 0.0
    integrity_hash: str = ""
    document_registry: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def compute_integrity_hash(self) -> str:
        """Compute overall integrity hash of the manifest."""
        payload = (
            f"{self.manifest_id}|{self.submission_id}|{self.total_documents}|"
            f"{self.total_sections}|{self.completed_tasks}"
        )
        self.integrity_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.integrity_hash


@dataclass
class ReviewerAssignment:
    """Assignment of a reviewer to a submission task."""

    assignment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_name: str = ""
    reviewer_role: str = ""
    task_id: str = ""
    assigned_date: str = field(default_factory=lambda: datetime.now().isoformat())
    due_date: str = ""
    status: WorkflowStatus = WorkflowStatus.DRAFT
    comments: str = ""


@dataclass
class DeadlineTracker:
    """Tracks regulatory deadlines and sends alerts for approaching dates."""

    submission_id: str = ""
    milestones: List[SubmissionMilestone] = field(default_factory=list)
    alert_threshold_days: int = 14

    def get_upcoming_deadlines(self, reference_date: Optional[str] = None) -> List[SubmissionMilestone]:
        """Return milestones approaching their deadline within alert threshold."""
        upcoming: List[SubmissionMilestone] = []
        for ms in self.milestones:
            if ms.is_completed:
                continue
            days = ms.days_until_target(reference_date)
            if days is not None and 0 <= days <= self.alert_threshold_days:
                upcoming.append(ms)
        return upcoming

    def get_overdue_milestones(self, reference_date: Optional[str] = None) -> List[SubmissionMilestone]:
        """Return milestones past their deadline."""
        overdue: List[SubmissionMilestone] = []
        for ms in self.milestones:
            if ms.is_completed:
                continue
            days = ms.days_until_target(reference_date)
            if days is not None and days < 0:
                overdue.append(ms)
        return overdue


# ===================================================================
# Main class
# ===================================================================
class RegulatorySubmissionOrchestrator:
    """Manages the full lifecycle of regulatory submissions.

    This orchestrator coordinates planning, authoring, review, compilation,
    validation, submission, and tracking phases.  It maintains an append-only
    audit trail compatible with 21 CFR Part 11 requirements and enforces
    task dependency ordering.

    Parameters
    ----------
    config : SubmissionConfig
        Top-level submission configuration.
    """

    def __init__(self, config: SubmissionConfig) -> None:
        self._config = config
        self._current_phase: SubmissionPhase = SubmissionPhase.PLANNING
        self._status: WorkflowStatus = WorkflowStatus.DRAFT
        self._tasks: Dict[str, WorkflowTask] = {}
        self._milestones: Dict[str, SubmissionMilestone] = {}
        self._audit_trail: List[AuditTrailEntry] = []
        self._reviewer_assignments: Dict[str, ReviewerAssignment] = {}
        self._document_registry: Dict[str, str] = {}
        self._deadline_tracker = DeadlineTracker(submission_id=config.submission_id)
        self._phase_order = list(SubmissionPhase)
        logger.info(
            "Initialized orchestrator for submission %s (%s)", config.submission_id, config.submission_type.value
        )
        self._record_audit(AuditAction.CREATE, "submission", config.submission_id, "", "initialized")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> SubmissionConfig:
        """Return the submission configuration."""
        return self._config

    @property
    def current_phase(self) -> SubmissionPhase:
        """Return the current submission phase."""
        return self._current_phase

    @property
    def status(self) -> WorkflowStatus:
        """Return the current workflow status."""
        return self._status

    @property
    def audit_trail(self) -> List[AuditTrailEntry]:
        """Return the audit trail (read-only copy)."""
        return list(self._audit_trail)

    @property
    def task_count(self) -> int:
        """Return total number of tasks."""
        return len(self._tasks)

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------
    def _record_audit(
        self,
        action: AuditAction,
        entity_type: str,
        entity_id: str,
        previous_value: str,
        new_value: str,
        user: str = "system",
        reason: str = "",
    ) -> AuditTrailEntry:
        """Append an entry to the 21 CFR Part 11 audit trail."""
        if len(self._audit_trail) >= MAX_AUDIT_ENTRIES:
            logger.warning("Audit trail capacity reached (%d entries)", MAX_AUDIT_ENTRIES)
            return self._audit_trail[-1]
        entry = AuditTrailEntry(
            action=action,
            user=user,
            entity_type=entity_type,
            entity_id=entity_id,
            previous_value=previous_value,
            new_value=new_value,
            reason=reason,
        )
        entry.compute_hash()
        self._audit_trail.append(entry)
        logger.debug("Audit: %s %s %s", action.value, entity_type, entity_id)
        return entry

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------
    def advance_phase(self) -> SubmissionPhase:
        """Advance to the next submission phase if all tasks in current phase are complete."""
        current_idx = self._phase_order.index(self._current_phase)
        if current_idx >= len(self._phase_order) - 1:
            logger.info("Already in final phase: %s", self._current_phase.value)
            return self._current_phase

        phase_tasks = [t for t in self._tasks.values() if t.phase == self._current_phase]
        incomplete = [t for t in phase_tasks if t.status not in (WorkflowStatus.APPROVED, WorkflowStatus.SUBMITTED)]
        if incomplete:
            logger.warning(
                "Cannot advance phase: %d incomplete tasks in %s",
                len(incomplete),
                self._current_phase.value,
            )
            return self._current_phase

        previous = self._current_phase
        self._current_phase = self._phase_order[current_idx + 1]
        self._record_audit(
            AuditAction.UPDATE,
            "phase",
            self._config.submission_id,
            previous.value,
            self._current_phase.value,
            reason="Phase advancement",
        )
        logger.info("Phase advanced: %s -> %s", previous.value, self._current_phase.value)
        return self._current_phase

    def set_phase(self, phase: SubmissionPhase) -> None:
        """Explicitly set the current phase (for orchestration overrides)."""
        previous = self._current_phase
        self._current_phase = phase
        self._record_audit(
            AuditAction.UPDATE,
            "phase",
            self._config.submission_id,
            previous.value,
            phase.value,
            reason="Manual phase override",
        )

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------
    def add_task(
        self,
        name: str,
        phase: Optional[SubmissionPhase] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        assignee: str = "",
        due_date: str = "",
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> WorkflowTask:
        """Create and register a new workflow task."""
        if len(self._tasks) >= MAX_TASKS_PER_PHASE * len(SubmissionPhase):
            raise ValueError("Maximum task limit reached")

        task = WorkflowTask(
            name=name,
            phase=phase or self._current_phase,
            priority=priority,
            assignee=assignee,
            due_date=due_date,
            description=description,
            dependencies=dependencies or [],
        )
        self._tasks[task.task_id] = task
        self._record_audit(AuditAction.CREATE, "task", task.task_id, "", name)
        logger.info("Task created: %s (%s)", name, task.task_id[:8])
        return task

    def update_task_status(self, task_id: str, new_status: WorkflowStatus, user: str = "system") -> bool:
        """Update the status of a workflow task."""
        if task_id not in self._tasks:
            logger.error("Task not found: %s", task_id)
            return False

        task = self._tasks[task_id]
        completed_ids = {tid for tid, t in self._tasks.items() if t.status == WorkflowStatus.APPROVED}
        if task.is_blocked(list(completed_ids)) and new_status != WorkflowStatus.REJECTED:
            logger.warning("Task %s is blocked by dependencies", task_id[:8])
            return False

        previous_status = task.status
        task.status = new_status
        task.updated_at = datetime.now().isoformat()
        self._record_audit(
            AuditAction.UPDATE,
            "task",
            task_id,
            previous_status.value,
            new_status.value,
            user=user,
        )
        return True

    def get_tasks_by_phase(self, phase: SubmissionPhase) -> List[WorkflowTask]:
        """Return all tasks assigned to a specific phase."""
        return [t for t in self._tasks.values() if t.phase == phase]

    def get_tasks_by_status(self, status: WorkflowStatus) -> List[WorkflowTask]:
        """Return all tasks with a specific status."""
        return [t for t in self._tasks.values() if t.status == status]

    def get_blocked_tasks(self) -> List[WorkflowTask]:
        """Return tasks blocked by incomplete dependencies."""
        completed_ids = {tid for tid, t in self._tasks.items() if t.status == WorkflowStatus.APPROVED}
        return [t for t in self._tasks.values() if t.is_blocked(list(completed_ids))]

    # ------------------------------------------------------------------
    # Milestone management
    # ------------------------------------------------------------------
    def add_milestone(
        self,
        name: str,
        phase: SubmissionPhase,
        target_date: str,
        owner: str = "",
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> SubmissionMilestone:
        """Create and register a submission milestone."""
        if len(self._milestones) >= MAX_MILESTONES:
            raise ValueError("Maximum milestone limit reached")

        milestone = SubmissionMilestone(
            name=name,
            phase=phase,
            target_date=target_date,
            owner=owner,
            description=description,
            dependencies=dependencies or [],
        )
        self._milestones[milestone.milestone_id] = milestone
        self._deadline_tracker.milestones.append(milestone)
        self._record_audit(AuditAction.CREATE, "milestone", milestone.milestone_id, "", name)
        logger.info("Milestone created: %s targeting %s", name, target_date)
        return milestone

    def complete_milestone(self, milestone_id: str, completion_date: Optional[str] = None) -> bool:
        """Mark a milestone as completed."""
        if milestone_id not in self._milestones:
            logger.error("Milestone not found: %s", milestone_id)
            return False
        milestone = self._milestones[milestone_id]
        milestone.mark_completed(completion_date)
        self._record_audit(AuditAction.APPROVE, "milestone", milestone_id, "pending", "completed")
        return True

    def get_upcoming_deadlines(self, reference_date: Optional[str] = None) -> List[SubmissionMilestone]:
        """Return milestones approaching within alert threshold."""
        return self._deadline_tracker.get_upcoming_deadlines(reference_date)

    def get_overdue_milestones(self, reference_date: Optional[str] = None) -> List[SubmissionMilestone]:
        """Return overdue milestones."""
        return self._deadline_tracker.get_overdue_milestones(reference_date)

    # ------------------------------------------------------------------
    # Reviewer management
    # ------------------------------------------------------------------
    def assign_reviewer(
        self,
        reviewer_name: str,
        reviewer_role: str,
        task_id: str,
        due_date: str = "",
    ) -> Optional[ReviewerAssignment]:
        """Assign a reviewer to a task."""
        if task_id not in self._tasks:
            logger.error("Cannot assign reviewer: task %s not found", task_id)
            return None
        if len(self._reviewer_assignments) >= MAX_REVIEWERS:
            logger.warning("Maximum reviewer assignments reached")
            return None

        assignment = ReviewerAssignment(
            reviewer_name=reviewer_name,
            reviewer_role=reviewer_role,
            task_id=task_id,
            due_date=due_date,
        )
        self._reviewer_assignments[assignment.assignment_id] = assignment
        self._record_audit(
            AuditAction.REVIEW,
            "reviewer_assignment",
            assignment.assignment_id,
            "",
            f"{reviewer_name} ({reviewer_role})",
        )
        return assignment

    # ------------------------------------------------------------------
    # Document registry
    # ------------------------------------------------------------------
    def register_document(self, document_id: str, document_name: str, content_hash: str = "") -> None:
        """Register a document in the submission package."""
        if not content_hash:
            content_hash = hashlib.sha256(f"{document_id}:{document_name}".encode()).hexdigest()
        self._document_registry[document_id] = content_hash
        self._record_audit(AuditAction.CREATE, "document", document_id, "", document_name)
        logger.info("Document registered: %s (hash: %s...)", document_name, content_hash[:12])

    def verify_document_integrity(self, document_id: str, expected_hash: str) -> bool:
        """Verify a document's SHA-256 hash matches the registered value."""
        stored = self._document_registry.get(document_id)
        if stored is None:
            logger.error("Document not found in registry: %s", document_id)
            return False
        return stored == expected_hash

    # ------------------------------------------------------------------
    # Submission actions
    # ------------------------------------------------------------------
    def submit(self, user: str = "system", reason: str = "") -> bool:
        """Mark the submission as submitted (simulated)."""
        draft_tasks = [t for t in self._tasks.values() if t.status == WorkflowStatus.DRAFT]
        if draft_tasks:
            logger.warning("Cannot submit: %d tasks still in DRAFT status", len(draft_tasks))
            return False
        self._status = WorkflowStatus.SUBMITTED
        self._current_phase = SubmissionPhase.TRACKING
        self._record_audit(
            AuditAction.SUBMIT,
            "submission",
            self._config.submission_id,
            "draft",
            "submitted",
            user=user,
            reason=reason,
        )
        logger.info("Submission %s marked as SUBMITTED", self._config.submission_id)
        return True

    def acknowledge(self, user: str = "system") -> None:
        """Record regulatory authority acknowledgement."""
        self._status = WorkflowStatus.ACKNOWLEDGED
        self._record_audit(
            AuditAction.UPDATE,
            "submission",
            self._config.submission_id,
            "submitted",
            "acknowledged",
            user=user,
        )

    # ------------------------------------------------------------------
    # Manifest compilation
    # ------------------------------------------------------------------
    def compile_manifest(self) -> SubmissionManifest:
        """Compile a complete submission manifest summarizing current state."""
        completed = [t for t in self._tasks.values() if t.status in (WorkflowStatus.APPROVED, WorkflowStatus.SUBMITTED)]
        pending = [t for t in self._tasks.values() if t.status == WorkflowStatus.DRAFT]
        phases_completed = []
        for phase in self._phase_order:
            phase_tasks = self.get_tasks_by_phase(phase)
            if phase_tasks and all(
                t.status in (WorkflowStatus.APPROVED, WorkflowStatus.SUBMITTED) for t in phase_tasks
            ):
                phases_completed.append(phase.value)

        warnings: List[str] = []
        errors: List[str] = []
        overdue = self.get_overdue_milestones()
        if overdue:
            warnings.append(f"{len(overdue)} overdue milestone(s)")
        blocked = self.get_blocked_tasks()
        if blocked:
            warnings.append(f"{len(blocked)} blocked task(s)")
        rejected = self.get_tasks_by_status(WorkflowStatus.REJECTED)
        if rejected:
            errors.append(f"{len(rejected)} rejected task(s) require remediation")

        total_tasks = len(self._tasks)
        completed_count = len(completed)
        score = (completed_count / total_tasks * 100.0) if total_tasks > 0 else 0.0

        manifest = SubmissionManifest(
            submission_id=self._config.submission_id,
            submission_type=self._config.submission_type,
            status=self._status,
            total_documents=len(self._document_registry),
            total_sections=len(self._milestones),
            phases_completed=phases_completed,
            pending_tasks=len(pending),
            completed_tasks=completed_count,
            compliance_score=round(score, 2),
            document_registry=dict(self._document_registry),
            warnings=warnings,
            errors=errors,
        )
        manifest.compute_integrity_hash()
        self._record_audit(AuditAction.COMPILE, "manifest", manifest.manifest_id, "", "compiled")
        logger.info("Manifest compiled: %d docs, score=%.1f%%", manifest.total_documents, score)
        return manifest

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_status_report(self) -> str:
        """Generate a Markdown-formatted status report."""
        lines: List[str] = [
            f"# Submission Status Report: {self._config.submission_id}",
            "",
            f"**Type:** {self._config.submission_type.value}",
            f"**Sponsor:** {self._config.sponsor_name}",
            f"**Device:** {self._config.device_name}",
            f"**Current Phase:** {self._current_phase.value}",
            f"**Status:** {self._status.value}",
            "",
            "## Task Summary",
            "",
            "| Status | Count |",
            "|--------|-------|",
        ]
        for ws in WorkflowStatus:
            count = len(self.get_tasks_by_status(ws))
            lines.append(f"| {ws.value} | {count} |")

        lines.extend(["", "## Milestones", ""])
        for ms in self._milestones.values():
            status = "DONE" if ms.is_completed else "PENDING"
            lines.append(f"- [{status}] {ms.name} (target: {ms.target_date})")

        lines.extend(
            [
                "",
                "## Audit Trail Summary",
                "",
                f"Total entries: {len(self._audit_trail)}",
                "",
                "| Timestamp | Action | Entity | User |",
                "|-----------|--------|--------|------|",
            ]
        )
        for entry in self._audit_trail[-10:]:
            lines.append(
                f"| {entry.timestamp[:19]} | {entry.action.value} | "
                f"{entry.entity_type}/{entry.entity_id[:8]} | {entry.user} |"
            )

        lines.extend(["", f"*Report generated: {datetime.now().isoformat()[:19]}*", ""])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize orchestrator state to a dictionary."""
        return {
            "submission_id": self._config.submission_id,
            "submission_type": self._config.submission_type.value,
            "current_phase": self._current_phase.value,
            "status": self._status.value,
            "task_count": len(self._tasks),
            "milestone_count": len(self._milestones),
            "document_count": len(self._document_registry),
            "audit_entries": len(self._audit_trail),
            "reviewer_assignments": len(self._reviewer_assignments),
        }

    def compute_completion_percentage(self) -> float:
        """Compute overall completion percentage across all tasks."""
        if not self._tasks:
            return 0.0
        completed = sum(
            1 for t in self._tasks.values() if t.status in (WorkflowStatus.APPROVED, WorkflowStatus.SUBMITTED)
        )
        return round(completed / len(self._tasks) * 100.0, 2)

    def get_critical_path(self) -> List[WorkflowTask]:
        """Identify tasks on the critical path (critical priority with dependencies)."""
        critical = []
        for task in self._tasks.values():
            if task.priority == TaskPriority.CRITICAL:
                critical.append(task)
            elif task.dependencies and task.status == WorkflowStatus.DRAFT:
                critical.append(task)
        return sorted(critical, key=lambda t: self._phase_order.index(t.phase))

    def validate_workflow_integrity(self) -> Tuple[bool, List[str]]:
        """Validate that the workflow is internally consistent."""
        issues: List[str] = []
        all_task_ids = set(self._tasks.keys())
        for task in self._tasks.values():
            for dep in task.dependencies:
                if dep not in all_task_ids:
                    issues.append(f"Task {task.task_id[:8]} depends on missing task {dep[:8]}")

        for assignment in self._reviewer_assignments.values():
            if assignment.task_id not in all_task_ids:
                issues.append(f"Reviewer assignment {assignment.assignment_id[:8]} references missing task")

        if not self._audit_trail:
            issues.append("Audit trail is empty")

        for i in range(1, len(self._audit_trail)):
            if self._audit_trail[i].timestamp < self._audit_trail[i - 1].timestamp:
                issues.append("Audit trail timestamps are not monotonically increasing")
                break

        return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def create_standard_510k_workflow(config: SubmissionConfig) -> RegulatorySubmissionOrchestrator:
    """Factory: create an orchestrator pre-populated with standard 510(k) tasks."""
    orch = RegulatorySubmissionOrchestrator(config)

    planning_tasks = [
        ("Identify predicate device", TaskPriority.CRITICAL),
        ("Define intended use statement", TaskPriority.CRITICAL),
        ("Determine classification panel", TaskPriority.HIGH),
        ("Prepare regulatory strategy", TaskPriority.HIGH),
    ]
    for name, priority in planning_tasks:
        orch.add_task(name, SubmissionPhase.PLANNING, priority)

    authoring_tasks = [
        ("Draft device description", TaskPriority.HIGH),
        ("Prepare substantial equivalence comparison", TaskPriority.CRITICAL),
        ("Draft software description (IEC 62304)", TaskPriority.HIGH),
        ("Prepare performance testing summary", TaskPriority.HIGH),
        ("Draft biocompatibility assessment", TaskPriority.MEDIUM),
        ("Prepare labeling", TaskPriority.MEDIUM),
    ]
    for name, priority in authoring_tasks:
        orch.add_task(name, SubmissionPhase.AUTHORING, priority)

    review_tasks = [
        ("Internal regulatory review", TaskPriority.HIGH),
        ("Quality assurance review", TaskPriority.HIGH),
        ("Legal review", TaskPriority.MEDIUM),
    ]
    for name, priority in review_tasks:
        orch.add_task(name, SubmissionPhase.REVIEW, priority)

    orch.add_task("Compile eCTD package", SubmissionPhase.COMPILATION, TaskPriority.CRITICAL)
    orch.add_task("Run validation suite", SubmissionPhase.VALIDATION, TaskPriority.CRITICAL)
    orch.add_task("Submit to FDA ESG", SubmissionPhase.SUBMISSION, TaskPriority.CRITICAL)

    orch.add_milestone("Predicate identified", SubmissionPhase.PLANNING, "2026-03-01")
    orch.add_milestone("Authoring complete", SubmissionPhase.AUTHORING, "2026-04-15")
    orch.add_milestone("Review sign-off", SubmissionPhase.REVIEW, "2026-05-01")
    orch.add_milestone("eCTD compiled", SubmissionPhase.COMPILATION, "2026-05-15")
    orch.add_milestone("Submission filed", SubmissionPhase.SUBMISSION, "2026-06-01")

    return orch


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = SubmissionConfig(
        submission_id="SUB-2026-DEMO",
        submission_type=SubmissionType.ECTD_510K,
        sponsor_name="PAI Oncology Research",
        device_name="AI Tumor Detector v2.0",
    )
    orch = create_standard_510k_workflow(cfg)
    report = orch.generate_status_report()
    print(report)
