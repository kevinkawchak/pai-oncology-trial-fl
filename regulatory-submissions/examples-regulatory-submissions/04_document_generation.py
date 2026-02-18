#!/usr/bin/env python3
"""Document Generation -- Template-Driven Regulatory Documents.

CLINICAL CONTEXT
================
This example demonstrates automated generation of regulatory submission
documents using template-driven workflows.  It generates a 510(k) summary,
clinical study report synopsis, software description (IEC 62304), risk
analysis report (ISO 14971), and AI/ML Predetermined Change Control Plan
(PCCP) -- all structured as Markdown documents.

These documents form the core of a regulatory submission package for an
AI/ML-enabled oncology diagnostic device.

USE CASES COVERED
=================
1. Generating a 510(k) summary document.
2. Generating a Clinical Study Report (CSR) synopsis with performance data.
3. Generating a software description per IEC 62304.
4. Generating a risk analysis report per ISO 14971.
5. Generating an AI/ML PCCP document.
6. Batch-generating a complete submission document package.
7. Reviewing document completeness and metadata.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

REFERENCES
==========
- ICH E3 -- Structure and Content of Clinical Study Reports
- IEC 62304:2015 -- Medical Device Software Lifecycle
- ISO 14971:2019 -- Risk Management for Medical Devices
- FDA Guidance: Marketing Submission Recommendations for PCCP

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
logger = logging.getLogger("pai.regulatory_submissions.example_04")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
_mod = _load_module("document_generator", "regulatory-submissions/document_generator.py")

DocumentCategory = _mod.DocumentCategory
GeneratedDocument = _mod.GeneratedDocument
RegulatoryDocumentGenerator = _mod.RegulatoryDocumentGenerator
TemplateConfig = _mod.TemplateConfig
TemplateType = _mod.TemplateType


def run_document_generation() -> None:
    """Execute the document generation demonstration."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 04: Template-Driven Document Generation")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Initialize the document generator
    # ------------------------------------------------------------------
    logger.info("Step 1: Initializing document generator...")
    generator = RegulatoryDocumentGenerator(
        submission_id="SUB-2026-EX04",
        sponsor_name="PAI Oncology Research Institute",
    )
    logger.info("Generator initialized. Available templates: %d", len(generator.available_templates))
    for tmpl in generator.available_templates:
        logger.info("  - %s", tmpl.value)

    # ------------------------------------------------------------------
    # Step 2: Configure template parameters
    # ------------------------------------------------------------------
    logger.info("Step 2: Configuring template parameters...")
    config = TemplateConfig(
        template_type=TemplateType.CSR_SYNOPSIS,
        sponsor_name="PAI Oncology Research Institute",
        device_name="OncoDetect AI v2.0",
        submission_id="SUB-2026-EX04",
        indication="AI-assisted oncology tumor detection and classification",
        predicate_device="TumorCAD v1.5 (K201234)",
        software_version="2.0.0",
        study_id="ONCO-DETECT-PIVOTAL-001",
        study_title="Prospective Multi-Center Evaluation of OncoDetect AI v2.0",
        sample_size=450,
        primary_endpoint="Sensitivity and specificity for tumor detection",
        secondary_endpoints=[
            "Time to diagnosis (hours)",
            "Reader agreement (Cohen's kappa)",
            "Adverse event rate",
            "Subgroup performance by tumor type",
        ],
        risk_class="Class II",
        software_safety_class="Class B",
        ai_ml_model_type="Deep learning convolutional neural network (ResNet-50)",
        training_data_description=(
            "Multi-site oncology imaging dataset comprising 25,000 annotated cases "
            "from 12 clinical sites with histopathological ground truth."
        ),
        performance_metrics={
            "Sensitivity": 0.9234,
            "Specificity": 0.8891,
            "AUC": 0.9456,
            "PPV": 0.8723,
            "NPV": 0.9345,
            "F1 Score": 0.8974,
        },
        custom_fields={"pathway": "510(k)", "product_code": "QAS"},
    )

    # ------------------------------------------------------------------
    # Step 3: Generate individual documents
    # ------------------------------------------------------------------
    logger.info("Step 3: Generating individual documents...")

    # 510(k) Summary
    doc_510k = generator.generate_document(TemplateType.K510_SUMMARY, config, author="Regulatory Affairs Lead")
    logger.info(
        "Generated: %s (%d words, complete=%s)",
        doc_510k.metadata.title,
        doc_510k.metadata.word_count,
        doc_510k.is_complete,
    )

    # CSR Synopsis
    doc_csr = generator.generate_document(TemplateType.CSR_SYNOPSIS, config, author="Medical Writer")
    logger.info(
        "Generated: %s (%d words, complete=%s)",
        doc_csr.metadata.title,
        doc_csr.metadata.word_count,
        doc_csr.is_complete,
    )

    # Software Description
    doc_sw = generator.generate_document(TemplateType.SOFTWARE_DESCRIPTION, config, author="Software QA Lead")
    logger.info(
        "Generated: %s (%d words, complete=%s)", doc_sw.metadata.title, doc_sw.metadata.word_count, doc_sw.is_complete
    )

    # Risk Analysis
    doc_risk = generator.generate_document(TemplateType.RISK_ANALYSIS, config, author="Risk Engineer")
    logger.info(
        "Generated: %s (%d words, complete=%s)",
        doc_risk.metadata.title,
        doc_risk.metadata.word_count,
        doc_risk.is_complete,
    )

    # AI/ML PCCP
    doc_pccp = generator.generate_document(TemplateType.AI_ML_SUPPLEMENT, config, author="AI/ML Engineer")
    logger.info(
        "Generated: %s (%d words, complete=%s)",
        doc_pccp.metadata.title,
        doc_pccp.metadata.word_count,
        doc_pccp.is_complete,
    )

    # Predicate Comparison
    doc_pred = generator.generate_document(TemplateType.PREDICATE_COMPARISON, config, author="Regulatory Affairs")
    logger.info(
        "Generated: %s (%d words, complete=%s)",
        doc_pred.metadata.title,
        doc_pred.metadata.word_count,
        doc_pred.is_complete,
    )

    logger.info("Total documents generated: %d", generator.document_count)

    # ------------------------------------------------------------------
    # Step 4: Display sample document content
    # ------------------------------------------------------------------
    logger.info("Step 4: Displaying CSR Synopsis content...")
    print("\n" + "=" * 60)
    print("CSR SYNOPSIS (first 2000 characters)")
    print("=" * 60)
    print(doc_csr.content[:2000])
    print("...\n")

    # ------------------------------------------------------------------
    # Step 5: Review document metadata
    # ------------------------------------------------------------------
    logger.info("Step 5: Reviewing document metadata...")
    all_docs = generator.get_all_documents()
    for doc in all_docs:
        completeness = doc.compute_completeness()
        logger.info(
            "  [%s] %s -- %d words, %d sections, %.0f%% complete, hash=%s...",
            doc.metadata.category.value[:8],
            doc.metadata.title,
            doc.metadata.word_count,
            doc.metadata.section_count,
            completeness,
            doc.metadata.content_hash[:12],
        )
        if doc.warnings:
            for w in doc.warnings:
                logger.warning("    Warning: %s", w)

    # ------------------------------------------------------------------
    # Step 6: Serialization
    # ------------------------------------------------------------------
    logger.info("Step 6: Serializing generator state...")
    state = generator.to_dict()
    logger.info("  Submission ID: %s", state["submission_id"])
    logger.info("  Document count: %d", state["document_count"])
    logger.info("  Templates available: %d", len(state["available_templates"]))

    logger.info("=" * 80)
    logger.info("Example 04 complete. Document generation demonstrated successfully.")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_document_generation()
