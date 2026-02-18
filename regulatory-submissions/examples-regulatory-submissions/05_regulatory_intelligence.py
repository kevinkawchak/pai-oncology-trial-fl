#!/usr/bin/env python3
"""Regulatory Intelligence -- Guidance Tracking and Impact Assessment.

CLINICAL CONTEXT
================
This example demonstrates the regulatory intelligence engine for tracking
FDA, EMA, and PMDA guidance documents relevant to AI/ML medical devices,
assessing their impact on ongoing submissions, generating compliance
timelines, monitoring enforcement actions, and performing device
classification lookups.

Regulatory intelligence is essential for proactive submission management,
ensuring that evolving regulatory requirements are identified and addressed
before they cause submission delays or deficiencies.

USE CASES COVERED
=================
1. Searching and filtering guidance documents by jurisdiction and category.
2. Assessing the impact of guidance documents on specific submissions.
3. Generating compliance timelines with milestones.
4. Tracking enforcement actions and extracting lessons learned.
5. Looking up device classifications by product code.
6. Generating comprehensive regulatory intelligence reports.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

REFERENCES
==========
- FDA AI/ML Framework for SaMD
- FDA PCCP Guidance (2024)
- EU MDR 2017/745
- IMDRF SaMD Clinical Evaluation Framework

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
logger = logging.getLogger("pai.regulatory_submissions.example_05")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
_mod = _load_module("regulatory_intelligence", "regulatory-submissions/regulatory_intelligence.py")

DeviceClassification = _mod.DeviceClassification
EnforcementAction = _mod.EnforcementAction
EnforcementType = _mod.EnforcementType
GuidanceCategory = _mod.GuidanceCategory
GuidanceDocument = _mod.GuidanceDocument
GuidanceStatus = _mod.GuidanceStatus
ImpactLevel = _mod.ImpactLevel
Jurisdiction = _mod.Jurisdiction
RegulatoryIntelligenceEngine = _mod.RegulatoryIntelligenceEngine


def run_regulatory_intelligence() -> None:
    """Execute the regulatory intelligence demonstration."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 05: Regulatory Intelligence Engine")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Initialize the intelligence engine
    # ------------------------------------------------------------------
    logger.info("Step 1: Initializing regulatory intelligence engine...")
    engine = RegulatoryIntelligenceEngine(
        submission_id="SUB-2026-EX05",
        jurisdictions=[Jurisdiction.FDA, Jurisdiction.EMA, Jurisdiction.PMDA],
    )
    logger.info("Engine initialized: %d guidance documents tracked", engine.guidance_count)
    logger.info("Jurisdictions: %s", ", ".join(j.value.upper() for j in engine.jurisdictions))

    # ------------------------------------------------------------------
    # Step 2: Browse guidance by jurisdiction
    # ------------------------------------------------------------------
    logger.info("Step 2: Browsing guidance by jurisdiction...")
    for jurisdiction in [Jurisdiction.FDA, Jurisdiction.EMA, Jurisdiction.PMDA]:
        docs = engine.get_guidance_by_jurisdiction(jurisdiction)
        logger.info("  %s: %d documents", jurisdiction.value.upper(), len(docs))
        for doc in docs[:3]:
            logger.info("    - [%s] %s", doc.status.value, doc.title)

    # ------------------------------------------------------------------
    # Step 3: Search by category and keyword
    # ------------------------------------------------------------------
    logger.info("Step 3: Searching guidance documents...")
    ai_ml_docs = engine.get_guidance_by_category(GuidanceCategory.AI_ML)
    logger.info("AI/ML guidance documents: %d", len(ai_ml_docs))

    pccp_results = engine.search_guidance("PCCP")
    logger.info("Search 'PCCP': %d results", len(pccp_results))
    for doc in pccp_results:
        logger.info("  - %s (%s)", doc.title, doc.jurisdiction.value.upper())

    cyber_results = engine.search_guidance("cybersecurity")
    logger.info("Search 'cybersecurity': %d results", len(cyber_results))

    # ------------------------------------------------------------------
    # Step 4: Assess guidance impact
    # ------------------------------------------------------------------
    logger.info("Step 4: Assessing guidance impact on submission...")
    active_guidance = engine.get_active_guidance()
    for doc in active_guidance[:4]:
        assessment = engine.assess_impact(
            doc.guidance_id,
            affected_sections=["Module 2", "Module 5", "Software Documentation"],
            assessed_by="regulatory_intelligence_analyst",
        )
        if assessment:
            logger.info(
                "  Impact: %s -> %s (score=%.1f, effort=%d days)",
                doc.title[:40],
                assessment.impact_level.value,
                assessment.impact_score(),
                assessment.estimated_effort_days,
            )

    high_impact = engine.get_high_impact_items()
    logger.info("High-impact items: %d", len(high_impact))

    # ------------------------------------------------------------------
    # Step 5: Generate compliance timelines
    # ------------------------------------------------------------------
    logger.info("Step 5: Generating compliance timelines...")
    for doc in active_guidance[:2]:
        timeline = engine.generate_compliance_timeline(
            doc.guidance_id,
            start_date="2026-03-01",
        )
        if timeline:
            logger.info(
                "  Timeline: %s -- %d days (%s to %s)",
                timeline.title[:50],
                timeline.total_duration_days,
                timeline.start_date,
                timeline.end_date,
            )
            for ms in timeline.milestones:
                logger.info("    Day %s: %s (%s days)", ms["offset_days"], ms["name"], ms["duration_days"])

    # ------------------------------------------------------------------
    # Step 6: Track enforcement actions
    # ------------------------------------------------------------------
    logger.info("Step 6: Tracking enforcement actions...")
    engine.add_enforcement_action(
        EnforcementAction(
            title="Warning Letter: AI Diagnostic Device Marketing",
            jurisdiction=Jurisdiction.FDA,
            action_type=EnforcementType.WARNING_LETTER,
            date_issued="2025-11-15",
            company_name="Example Medical AI Corp",
            device_type="AI diagnostic",
            description="Marketing AI/ML device without valid 510(k) clearance",
            relevance_score=0.8,
            lessons_learned=[
                "Ensure marketing claims match cleared intended use",
                "Maintain 510(k) clearance before commercial distribution",
            ],
        )
    )
    engine.add_enforcement_action(
        EnforcementAction(
            title="Recall: Software-Based Diagnostic Tool",
            jurisdiction=Jurisdiction.FDA,
            action_type=EnforcementType.RECALL,
            date_issued="2025-08-20",
            company_name="DiagTech Solutions",
            device_type="AI diagnostic",
            description="Class II recall due to algorithmic error in tumor detection",
            relevance_score=0.9,
            lessons_learned=[
                "Implement comprehensive V&V per IEC 62304",
                "Monitor real-world performance post-deployment",
                "Establish clear recall procedures",
            ],
        )
    )
    relevant = engine.get_relevant_enforcement_actions("AI diagnostic")
    logger.info("Relevant enforcement actions: %d", len(relevant))
    for action in relevant:
        logger.info("  [%s] %s (%s)", action.action_type.value, action.title, action.date_issued)
        for lesson in action.lessons_learned:
            logger.info("    Lesson: %s", lesson)

    # ------------------------------------------------------------------
    # Step 7: Device classification lookup
    # ------------------------------------------------------------------
    logger.info("Step 7: Looking up device classifications...")
    qas_results = engine.lookup_classification(product_code="QAS")
    logger.info("Product code 'QAS': %d result(s)", len(qas_results))
    for entry in qas_results:
        logger.info(
            "  %s: %s (%s, %s)",
            entry.product_code,
            entry.device_name,
            entry.classification.value,
            entry.submission_type,
        )

    ai_results = engine.lookup_classification(device_name="AI")
    logger.info("Device name search 'AI': %d result(s)", len(ai_results))
    for entry in ai_results:
        logger.info("  %s: %s (%s)", entry.product_code, entry.device_name, entry.classification.value)

    # ------------------------------------------------------------------
    # Step 8: Generate intelligence report
    # ------------------------------------------------------------------
    logger.info("Step 8: Generating regulatory intelligence report...")
    report = engine.generate_intelligence_report()
    print("\n" + report)

    logger.info("=" * 80)
    logger.info("Example 05 complete. Regulatory intelligence demonstrated successfully.")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_regulatory_intelligence()
