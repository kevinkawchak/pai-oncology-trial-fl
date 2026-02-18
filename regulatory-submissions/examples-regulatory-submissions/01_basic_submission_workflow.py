#!/usr/bin/env python3
"""Basic Submission Workflow -- Minimal Working Example.

CLINICAL CONTEXT
================
This example demonstrates the foundational workflow for configuring and
managing a regulatory submission using the PAI Regulatory Submission
Orchestrator.  It covers submission configuration, task creation, milestone
tracking, document registration, and status reporting.

The workflow mirrors the typical lifecycle of a 510(k) submission for an
AI/ML-enabled oncology diagnostic device, from initial planning through
compilation and submission tracking.

USE CASES COVERED
=================
1. Configuring the orchestrator with submission parameters including
   submission type, sponsor information, and target dates.
2. Creating and managing workflow tasks across submission phases.
3. Registering documents with SHA-256 integrity checksums.
4. Tracking milestones with deadline alerting.
5. Generating structured status reports in Markdown format.
6. Reviewing the 21 CFR Part 11 audit trail.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

REFERENCES
==========
- FDA eCTD Technical Conformance Guide
- 21 CFR Part 11 -- Electronic Records; Electronic Signatures
- FDA Guidance: Acceptance and Filing Reviews for Premarket Submissions

DISCLAIMER
==========
RESEARCH USE ONLY. This software is provided for research and educational
purposes only. It has NOT been validated for clinical use, is NOT approved
by the FDA or any other regulatory body, and MUST NOT be used to make
regulatory decisions. All outputs must be reviewed by qualified regulatory
professionals before any action is taken.

LICENSE: MIT
VERSION: 0.9.1
LAST UPDATED: 2026-02-18
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def _load_module(name: str, rel_path: str):
    """Load a module from a hyphenated directory via importlib."""
    filepath = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.regulatory_submissions.example_01")

# ---------------------------------------------------------------------------
# Imports from regulatory-submissions module
# ---------------------------------------------------------------------------
_mod = _load_module("submission_orchestrator", "regulatory-submissions/submission_orchestrator.py")

RegulatorySubmissionOrchestrator = _mod.RegulatorySubmissionOrchestrator
SubmissionConfig = _mod.SubmissionConfig
SubmissionMilestone = _mod.SubmissionMilestone
SubmissionPhase = _mod.SubmissionPhase
SubmissionType = _mod.SubmissionType
TaskPriority = _mod.TaskPriority
WorkflowStatus = _mod.WorkflowStatus
create_standard_510k_workflow = _mod.create_standard_510k_workflow


def run_basic_workflow() -> None:
    """Execute the basic submission workflow demonstration."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 01: Basic Submission Workflow")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Configure the submission
    # ------------------------------------------------------------------
    logger.info("Step 1: Configuring submission...")
    config = SubmissionConfig(
        submission_id="SUB-2026-EX01",
        submission_type=SubmissionType.ECTD_510K,
        sponsor_name="PAI Oncology Research Institute",
        device_name="OncoDetect AI v2.0",
        target_date="2026-06-30",
        regulatory_authority="FDA",
        contact_email="regulatory@pai-oncology.example",
        protocol_number="PROTO-2026-ONC-001",
        indication="AI-assisted oncology tumor detection",
        predicate_device="TumorCAD v1.5 (K201234)",
        product_code="QAS",
    )
    logger.info("Submission configured: %s (%s)", config.submission_id, config.submission_type.value)

    # ------------------------------------------------------------------
    # Step 2: Initialize the orchestrator
    # ------------------------------------------------------------------
    logger.info("Step 2: Initializing orchestrator...")
    orchestrator = RegulatorySubmissionOrchestrator(config)
    logger.info("Current phase: %s", orchestrator.current_phase.value)
    logger.info("Current status: %s", orchestrator.status.value)

    # ------------------------------------------------------------------
    # Step 3: Add tasks to the workflow
    # ------------------------------------------------------------------
    logger.info("Step 3: Adding workflow tasks...")
    task1 = orchestrator.add_task(
        "Identify predicate device",
        SubmissionPhase.PLANNING,
        TaskPriority.CRITICAL,
        assignee="Regulatory Lead",
        description="Search FDA 510(k) database for suitable predicate",
    )
    task2 = orchestrator.add_task(
        "Draft device description",
        SubmissionPhase.AUTHORING,
        TaskPriority.HIGH,
        assignee="Technical Writer",
    )
    task3 = orchestrator.add_task(
        "Prepare clinical evidence summary",
        SubmissionPhase.AUTHORING,
        TaskPriority.HIGH,
        assignee="Clinical Affairs",
        dependencies=[task1.task_id],
    )
    logger.info("Tasks created: %d total", orchestrator.task_count)

    # ------------------------------------------------------------------
    # Step 4: Register documents
    # ------------------------------------------------------------------
    logger.info("Step 4: Registering documents...")
    orchestrator.register_document("DOC-001", "Cover Letter")
    orchestrator.register_document("DOC-002", "FDA Form 356(h)")
    orchestrator.register_document("DOC-003", "Device Description")
    orchestrator.register_document("DOC-004", "Substantial Equivalence Comparison")
    logger.info("Documents registered: 4")

    # ------------------------------------------------------------------
    # Step 5: Add and track milestones
    # ------------------------------------------------------------------
    logger.info("Step 5: Setting up milestones...")
    ms1 = orchestrator.add_milestone(
        "Predicate device identified",
        SubmissionPhase.PLANNING,
        "2026-03-15",
        owner="Regulatory Lead",
    )
    ms2 = orchestrator.add_milestone(
        "Authoring complete",
        SubmissionPhase.AUTHORING,
        "2026-04-30",
        owner="Technical Writer",
    )
    ms3 = orchestrator.add_milestone(
        "Submission filed",
        SubmissionPhase.SUBMISSION,
        "2026-06-30",
        owner="Regulatory Lead",
    )
    logger.info("Milestones created: 3")

    # ------------------------------------------------------------------
    # Step 6: Progress tasks through the workflow
    # ------------------------------------------------------------------
    logger.info("Step 6: Progressing workflow...")
    orchestrator.update_task_status(task1.task_id, WorkflowStatus.IN_REVIEW, user="regulatory_lead")
    orchestrator.update_task_status(task1.task_id, WorkflowStatus.APPROVED, user="qa_manager")
    orchestrator.complete_milestone(ms1.milestone_id, "2026-03-10")
    logger.info("Task 1 approved, milestone 1 completed")

    # ------------------------------------------------------------------
    # Step 7: Assign reviewers
    # ------------------------------------------------------------------
    logger.info("Step 7: Assigning reviewers...")
    orchestrator.assign_reviewer("Dr. Smith", "Clinical Reviewer", task2.task_id, due_date="2026-04-15")
    orchestrator.assign_reviewer("Jane Doe", "QA Reviewer", task3.task_id, due_date="2026-04-20")

    # ------------------------------------------------------------------
    # Step 8: Generate status report
    # ------------------------------------------------------------------
    logger.info("Step 8: Generating status report...")
    report = orchestrator.generate_status_report()
    print("\n" + report)

    # ------------------------------------------------------------------
    # Step 9: Compile manifest
    # ------------------------------------------------------------------
    logger.info("Step 9: Compiling manifest...")
    manifest = orchestrator.compile_manifest()
    logger.info("Manifest ID: %s", manifest.manifest_id[:12])
    logger.info("Total documents: %d", manifest.total_documents)
    logger.info("Compliance score: %.1f%%", manifest.compliance_score)
    logger.info("Integrity hash: %s...", manifest.integrity_hash[:16])

    # ------------------------------------------------------------------
    # Step 10: Review audit trail
    # ------------------------------------------------------------------
    logger.info("Step 10: Reviewing audit trail...")
    audit = orchestrator.audit_trail
    logger.info("Audit trail entries: %d", len(audit))
    for entry in audit[:5]:
        logger.info(
            "  [%s] %s %s/%s by %s",
            entry.timestamp[:19],
            entry.action.value,
            entry.entity_type,
            entry.entity_id[:8],
            entry.user,
        )

    # ------------------------------------------------------------------
    # Step 11: Validate workflow integrity
    # ------------------------------------------------------------------
    logger.info("Step 11: Validating workflow integrity...")
    is_valid, issues = orchestrator.validate_workflow_integrity()
    logger.info("Workflow valid: %s", is_valid)
    if issues:
        for issue in issues:
            logger.warning("  Issue: %s", issue)

    logger.info("=" * 80)
    logger.info("Example 01 complete. Submission workflow demonstrated successfully.")
    logger.info("=" * 80)


def run_standard_510k_factory() -> None:
    """Demonstrate the standard 510(k) factory function."""
    logger.info("")
    logger.info("-" * 60)
    logger.info("BONUS: Standard 510(k) Workflow Factory")
    logger.info("-" * 60)

    config = SubmissionConfig(
        submission_id="SUB-2026-FACTORY",
        submission_type=SubmissionType.ECTD_510K,
        sponsor_name="PAI Oncology Research",
        device_name="AI Tumor Detector v2.0",
    )
    orchestrator = create_standard_510k_workflow(config)
    logger.info("Factory created workflow with %d tasks", orchestrator.task_count)
    logger.info("Completion: %.1f%%", orchestrator.compute_completion_percentage())

    critical_path = orchestrator.get_critical_path()
    logger.info("Critical path: %d tasks", len(critical_path))
    for task in critical_path[:5]:
        logger.info("  [%s] %s (%s)", task.priority.value, task.name, task.phase.value)


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_basic_workflow()
    run_standard_510k_factory()
