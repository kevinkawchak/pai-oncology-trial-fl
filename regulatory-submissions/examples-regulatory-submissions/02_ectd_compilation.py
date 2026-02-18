#!/usr/bin/env python3
"""eCTD Compilation -- Single Feature Example.

CLINICAL CONTEXT
================
This example demonstrates the eCTD (Electronic Common Technical Document)
compilation workflow for a 510(k) submission.  It covers module structure
initialization, document placement across all five ICH CTD modules,
validation against FDA Technical Conformance Guide requirements, XML
backbone generation, and compilation reporting.

The eCTD is the standard format for electronic submissions to FDA, EMA,
and other regulatory authorities worldwide.  This example shows how to
programmatically build and validate the eCTD structure.

USE CASES COVERED
=================
1. Initializing the eCTD module structure (Modules 1-5).
2. Adding documents to specific sections within each module.
3. Running validation to detect missing required sections.
4. Generating the XML backbone stub for the eCTD package.
5. Computing completeness scores per module.
6. Generating a structured compilation report.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

REFERENCES
==========
- ICH M4 (Common Technical Document)
- FDA eCTD Technical Conformance Guide v2.5
- FDA Guidance: Providing Regulatory Submissions in Electronic Format

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
logger = logging.getLogger("pai.regulatory_submissions.example_02")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
_mod = _load_module("ectd_compiler", "regulatory-submissions/ectd_compiler.py")

CompilationResult = _mod.CompilationResult
DocumentType = _mod.DocumentType
ECTDCompiler = _mod.ECTDCompiler
ECTDModule = _mod.ECTDModule
ValidationLevel = _mod.ValidationLevel


def run_ectd_compilation() -> None:
    """Execute the eCTD compilation demonstration."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 02: eCTD Compilation")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Initialize the compiler
    # ------------------------------------------------------------------
    logger.info("Step 1: Initializing eCTD compiler...")
    compiler = ECTDCompiler(
        submission_id="SUB-2026-EX02",
        submission_type="510k",
        sequence_number="0000",
    )
    logger.info("Compiler initialized with %d modules", len(compiler.modules))

    # ------------------------------------------------------------------
    # Step 2: Add Module 1 documents (Regional)
    # ------------------------------------------------------------------
    logger.info("Step 2: Adding Module 1 (Regional) documents...")
    compiler.add_document(
        title="Cover Letter to FDA CDRH",
        document_type=DocumentType.COVER_LETTER,
        module=ECTDModule.M1_REGIONAL,
        section_number="1.1",
        content="Dear CDRH Review Division, We are pleased to submit this 510(k)...",
        author="Regulatory Affairs Director",
    )
    compiler.add_document(
        title="FDA Form 356(h) -- Premarket Notification",
        document_type=DocumentType.FORM_FDA_356H,
        module=ECTDModule.M1_REGIONAL,
        section_number="1.2",
        content="Application form content for premarket notification...",
        author="Regulatory Affairs",
    )
    compiler.add_document(
        title="510(k) Summary per 21 CFR 807.92",
        document_type=DocumentType.SUMMARY_OVERVIEW,
        module=ECTDModule.M1_REGIONAL,
        section_number="1.5",
        content="This 510(k) summary provides an overview of the AI-assisted diagnostic...",
        author="Regulatory Affairs",
    )
    compiler.add_document(
        title="Indications for Use Statement",
        document_type=DocumentType.LABELING,
        module=ECTDModule.M1_REGIONAL,
        section_number="1.6",
        content="The OncoDetect AI v2.0 is intended to assist qualified clinicians...",
        author="Regulatory Affairs",
    )
    logger.info("Module 1: %d documents added", len(compiler.get_documents_by_module(ECTDModule.M1_REGIONAL)))

    # ------------------------------------------------------------------
    # Step 3: Add Module 2 documents (Summaries)
    # ------------------------------------------------------------------
    logger.info("Step 3: Adding Module 2 (Summaries) documents...")
    compiler.add_document(
        title="Clinical Overview",
        document_type=DocumentType.CLINICAL_OVERVIEW,
        module=ECTDModule.M2_SUMMARIES,
        section_number="2.5",
        content="Clinical overview summarizing safety and effectiveness evidence...",
        author="Medical Writer",
    )
    compiler.add_document(
        title="Clinical Summary",
        document_type=DocumentType.SUMMARY_OVERVIEW,
        module=ECTDModule.M2_SUMMARIES,
        section_number="2.7",
        content="Summary of clinical study results and safety data...",
        author="Medical Writer",
    )
    logger.info("Module 2: %d documents added", len(compiler.get_documents_by_module(ECTDModule.M2_SUMMARIES)))

    # ------------------------------------------------------------------
    # Step 4: Add Module 4 documents (Nonclinical)
    # ------------------------------------------------------------------
    logger.info("Step 4: Adding Module 4 (Nonclinical) documents...")
    compiler.add_document(
        title="Software Verification and Validation Report",
        document_type=DocumentType.SOFTWARE_DESCRIPTION,
        module=ECTDModule.M4_NONCLINICAL,
        section_number="4.2",
        content="V&V report for AI/ML software component...",
        author="Software QA",
    )
    logger.info("Module 4: %d documents added", len(compiler.get_documents_by_module(ECTDModule.M4_NONCLINICAL)))

    # ------------------------------------------------------------------
    # Step 5: Add Module 5 documents (Clinical)
    # ------------------------------------------------------------------
    logger.info("Step 5: Adding Module 5 (Clinical) documents...")
    compiler.add_document(
        title="Pivotal Clinical Study Report -- OncoDetect v2.0",
        document_type=DocumentType.CLINICAL_STUDY_REPORT,
        module=ECTDModule.M5_CLINICAL,
        section_number="5.3",
        content="Clinical study report per ICH E3 guidelines...",
        author="Biostatistics Lead",
    )
    compiler.add_document(
        title="Statistical Analysis Plan",
        document_type=DocumentType.STATISTICAL_ANALYSIS_PLAN,
        module=ECTDModule.M5_CLINICAL,
        section_number="5.3",
        content="SAP defining pre-specified analyses for the pivotal study...",
        author="Biostatistician",
    )
    logger.info("Module 5: %d documents added", len(compiler.get_documents_by_module(ECTDModule.M5_CLINICAL)))

    # ------------------------------------------------------------------
    # Step 6: Run compilation
    # ------------------------------------------------------------------
    logger.info("Step 6: Running eCTD compilation...")
    result = compiler.compile()
    logger.info("Compilation complete: valid=%s", result.is_valid)
    logger.info("Total documents: %d", result.total_documents)
    logger.info("Total sections: %d", result.total_sections)
    logger.info("Overall completeness: %.1f%%", result.overall_completeness)

    # ------------------------------------------------------------------
    # Step 7: Review validation findings
    # ------------------------------------------------------------------
    logger.info("Step 7: Reviewing validation findings...")
    logger.info("Errors: %d, Warnings: %d, Info: %d", result.error_count, result.warning_count, result.info_count)
    for finding in result.findings:
        logger.info(
            "  [%s] %s/%s: %s",
            finding.level.value.upper(),
            finding.module.value,
            finding.section,
            finding.message,
        )

    # ------------------------------------------------------------------
    # Step 8: Review XML backbone
    # ------------------------------------------------------------------
    logger.info("Step 8: XML backbone generated (%d characters)", len(result.xml_backbone))
    print("\n--- XML Backbone (first 1500 chars) ---")
    print(result.xml_backbone[:1500])
    print("--- End XML Backbone ---\n")

    # ------------------------------------------------------------------
    # Step 9: Generate compilation report
    # ------------------------------------------------------------------
    logger.info("Step 9: Generating compilation report...")
    report = compiler.generate_compilation_report(result)
    print(report)

    # ------------------------------------------------------------------
    # Step 10: Serialization
    # ------------------------------------------------------------------
    logger.info("Step 10: Serializing compiler state...")
    state = compiler.to_dict()
    for key, value in state.items():
        if isinstance(value, dict):
            logger.info("  %s:", key)
            for k, v in value.items():
                logger.info("    %s: %s", k, v)
        else:
            logger.info("  %s: %s", key, value)

    logger.info("=" * 80)
    logger.info("Example 02 complete. eCTD compilation demonstrated successfully.")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ectd_compilation()
