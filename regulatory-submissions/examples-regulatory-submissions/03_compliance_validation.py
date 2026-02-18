#!/usr/bin/env python3
"""Compliance Validation -- Multi-Regulation Checking Example.

CLINICAL CONTEXT
================
This example demonstrates multi-regulation compliance validation for a
regulatory submission.  It runs checklist-driven assessments against FDA
21 CFR 820 (Quality System Regulation), ISO 14971 (Risk Management), and
IEC 62304 (Medical Device Software Lifecycle), performing gap analysis,
computing compliance scores, tracking remediation actions, and generating
a comprehensive compliance report.

Compliance validation is a critical pre-submission activity ensuring that
all regulatory requirements are addressed before filing.

USE CASES COVERED
=================
1. Configuring a multi-regulation compliance validator.
2. Submitting evidence against regulatory requirements.
3. Running automated compliance assessments.
4. Performing gap analysis by regulation.
5. Creating and tracking remediation actions.
6. Computing weighted and unweighted compliance scores.
7. Generating compliance reports in Markdown format.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

REFERENCES
==========
- FDA 21 CFR 820 -- Quality System Regulation
- ISO 14971:2019 -- Risk Management for Medical Devices
- IEC 62304:2015 -- Medical Device Software Lifecycle
- FDA 21 CFR Part 11 -- Electronic Records

DISCLAIMER
==========
RESEARCH USE ONLY. This software is provided for research and educational
purposes only. It has NOT been validated for clinical use, is NOT approved
by the FDA or any other regulatory body, and MUST NOT be used to make
regulatory decisions.

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
# Project root on sys.path
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
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.regulatory_submissions.example_03")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
_mod = _load_module("compliance_validator", "regulatory-submissions/compliance_validator.py")

ComplianceAssessment = _mod.ComplianceAssessment
ComplianceFinding = _mod.ComplianceFinding
ComplianceStatus = _mod.ComplianceStatus
ComplianceValidator = _mod.ComplianceValidator
FindingSeverity = _mod.FindingSeverity
Regulation = _mod.Regulation
RemediationStatus = _mod.RemediationStatus


def run_compliance_validation() -> None:
    """Execute the compliance validation demonstration."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 03: Multi-Regulation Compliance Validation")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Configure the validator
    # ------------------------------------------------------------------
    logger.info("Step 1: Configuring compliance validator...")
    validator = ComplianceValidator(
        submission_id="SUB-2026-EX03",
        regulations=[
            Regulation.FDA_21CFR820,
            Regulation.FDA_21CFR11,
            Regulation.ISO_14971,
            Regulation.IEC_62304,
        ],
    )
    logger.info("Validator configured with %d requirements", validator.requirement_count)

    # ------------------------------------------------------------------
    # Step 2: Submit evidence for key requirements
    # ------------------------------------------------------------------
    logger.info("Step 2: Submitting evidence for requirements...")
    reqs_820 = validator.get_requirements_by_regulation(Regulation.FDA_21CFR820)
    reqs_14971 = validator.get_requirements_by_regulation(Regulation.ISO_14971)
    reqs_62304 = validator.get_requirements_by_regulation(Regulation.IEC_62304)
    reqs_11 = validator.get_requirements_by_regulation(Regulation.FDA_21CFR11)

    evidence_submitted = 0
    # Submit evidence for about half of 820 requirements
    for req in reqs_820[:8]:
        validator.submit_evidence(
            req.requirement_id,
            f"Design History File reference: DHF-{req.section}",
            user="qa_engineer",
        )
        evidence_submitted += 1

    # Submit evidence for most ISO 14971 requirements
    for req in reqs_14971[:12]:
        validator.submit_evidence(
            req.requirement_id,
            f"Risk Management File reference: RMF-{req.section}",
            user="risk_engineer",
        )
        evidence_submitted += 1

    # Submit evidence for IEC 62304 requirements
    for req in reqs_62304[:9]:
        validator.submit_evidence(
            req.requirement_id,
            f"Software Development Plan reference: SDP-{req.section}",
            user="sw_engineer",
        )
        evidence_submitted += 1

    # Submit evidence for 21 CFR 11 requirements
    for req in reqs_11[:8]:
        validator.submit_evidence(
            req.requirement_id,
            f"System validation report reference: SVR-{req.section}",
            user="validation_engineer",
        )
        evidence_submitted += 1

    logger.info("Evidence submitted for %d requirements", evidence_submitted)

    # ------------------------------------------------------------------
    # Step 3: Run compliance assessment
    # ------------------------------------------------------------------
    logger.info("Step 3: Running compliance assessment...")
    assessment = validator.run_assessment(assessed_by="regulatory_lead")
    logger.info("Assessment complete:")
    logger.info("  Total requirements: %d", assessment.total_requirements)
    logger.info("  Compliant: %d", assessment.compliant_count)
    logger.info("  Non-compliant: %d", assessment.non_compliant_count)
    logger.info("  Partial: %d", assessment.partial_count)
    logger.info("  Pending: %d", assessment.pending_count)
    logger.info("  Not applicable: %d", assessment.not_applicable_count)
    logger.info("  Weighted score: %.1f%%", assessment.weighted_score)
    logger.info("  Unweighted score: %.1f%%", assessment.unweighted_score)
    logger.info("  Passing: %s", assessment.is_passing)

    # ------------------------------------------------------------------
    # Step 4: Review gap analysis
    # ------------------------------------------------------------------
    logger.info("Step 4: Performing gap analysis...")
    gap_analysis = validator.generate_gap_analysis()
    for reg, data in gap_analysis.items():
        logger.info(
            "  %s: score=%.1f%%, compliant=%d/%d, critical_gaps=%d",
            reg.value,
            data["score"],
            data["compliant"],
            data["total_assessed"],
            len(data["critical_gaps"]),
        )
        if data["critical_gaps"]:
            for gap in data["critical_gaps"][:3]:
                logger.info("    Critical: %s", gap)

    # ------------------------------------------------------------------
    # Step 5: Create remediations for findings
    # ------------------------------------------------------------------
    logger.info("Step 5: Creating remediation actions...")
    critical_findings = validator.get_findings_by_severity(FindingSeverity.CRITICAL)
    major_findings = validator.get_findings_by_severity(FindingSeverity.MAJOR)
    logger.info("Critical findings: %d, Major findings: %d", len(critical_findings), len(major_findings))

    remediation_count = 0
    for finding in critical_findings[:5]:
        rem = validator.create_remediation(
            finding.finding_id,
            f"Address critical gap: {finding.title}",
            owner="regulatory_lead",
            target_date="2026-04-01",
        )
        if rem:
            remediation_count += 1

    for finding in major_findings[:3]:
        rem = validator.create_remediation(
            finding.finding_id,
            f"Address major gap: {finding.title}",
            owner="quality_engineer",
            target_date="2026-04-15",
        )
        if rem:
            remediation_count += 1

    logger.info("Remediation actions created: %d", remediation_count)

    # ------------------------------------------------------------------
    # Step 6: Progress some remediations
    # ------------------------------------------------------------------
    logger.info("Step 6: Progressing remediation actions...")
    open_rems = validator.get_open_remediations()
    for rem in open_rems[:2]:
        validator.update_remediation_status(
            rem.action_id,
            RemediationStatus.IN_PROGRESS,
            notes="Work in progress",
        )
    if open_rems:
        validator.update_remediation_status(
            open_rems[0].action_id,
            RemediationStatus.RESOLVED,
            notes="Evidence provided and verified",
        )
    logger.info("Open remediations remaining: %d", len(validator.get_open_remediations()))

    # ------------------------------------------------------------------
    # Step 7: Review audit trail
    # ------------------------------------------------------------------
    logger.info("Step 7: Reviewing audit trail...")
    audit = validator.audit_trail
    logger.info("Audit trail entries: %d", len(audit))
    for entry in audit[:5]:
        logger.info("  [%s] %s: %s", entry.timestamp[:19], entry.action, entry.details[:60])

    # ------------------------------------------------------------------
    # Step 8: Generate compliance report
    # ------------------------------------------------------------------
    logger.info("Step 8: Generating compliance report...")
    report = validator.generate_compliance_report(assessment)
    print("\n" + report)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Example 03 complete. Multi-regulation compliance validated successfully.")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_compliance_validation()
