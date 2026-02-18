#!/usr/bin/env python3
"""Full Submission Pipeline -- End-to-End Integration with Clinical Analytics.

CLINICAL CONTEXT
================
This example demonstrates the complete regulatory submission pipeline,
integrating ALL regulatory-submissions modules and showing how outputs
from the clinical-analytics module feed into regulatory documents.

The pipeline covers:
1. Orchestrating the submission lifecycle (planning through tracking)
2. Generating regulatory documents from clinical analytics results
3. Compiling the eCTD package with all documents
4. Validating multi-regulation compliance
5. Monitoring the regulatory landscape
6. Analyzing submission performance metrics

This represents the full downstream regulatory workflow that consumes
upstream analytics (survival analysis, risk stratification, PK/PD) from
the clinical-analytics module.

USE CASES COVERED
=================
1. Creating a submission orchestrator with standard 510(k) workflow.
2. Simulating clinical-analytics outputs (survival, risk, PK/PD).
3. Generating CSR synopsis from survival analysis results.
4. Generating PCCP from risk stratification metrics.
5. Compiling full eCTD package across all 5 modules.
6. Running multi-regulation compliance validation.
7. Assessing regulatory intelligence impact.
8. Computing submission analytics and benchmarks.
9. Generating an executive summary report.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)
    scipy >= 1.11.0  (https://scipy.org)

REFERENCES
==========
- ICH M4 (Common Technical Document)
- ICH E3 (Clinical Study Reports)
- FDA eCTD Technical Conformance Guide
- FDA AI/ML SaMD Framework
- ISO 14971:2019, IEC 62304:2015

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
logger = logging.getLogger("pai.regulatory_submissions.example_06")

# ---------------------------------------------------------------------------
# Imports from regulatory-submissions modules
# ---------------------------------------------------------------------------
_orch = _load_module("submission_orchestrator", "regulatory-submissions/submission_orchestrator.py")
RegulatorySubmissionOrchestrator = _orch.RegulatorySubmissionOrchestrator
SubmissionConfig = _orch.SubmissionConfig
SubmissionPhase = _orch.SubmissionPhase
SubmissionType = _orch.SubmissionType
TaskPriority = _orch.TaskPriority
WorkflowStatus = _orch.WorkflowStatus
create_standard_510k_workflow = _orch.create_standard_510k_workflow

_ectd = _load_module("ectd_compiler", "regulatory-submissions/ectd_compiler.py")
DocumentType = _ectd.DocumentType
ECTDCompiler = _ectd.ECTDCompiler
ECTDModule = _ectd.ECTDModule

_comp = _load_module("compliance_validator", "regulatory-submissions/compliance_validator.py")
ComplianceValidator = _comp.ComplianceValidator
Regulation = _comp.Regulation

_docgen = _load_module("document_generator", "regulatory-submissions/document_generator.py")
RegulatoryDocumentGenerator = _docgen.RegulatoryDocumentGenerator
TemplateConfig = _docgen.TemplateConfig
TemplateType = _docgen.TemplateType

_intel = _load_module("regulatory_intelligence", "regulatory-submissions/regulatory_intelligence.py")
GuidanceCategory = _intel.GuidanceCategory
Jurisdiction = _intel.Jurisdiction
RegulatoryIntelligenceEngine = _intel.RegulatoryIntelligenceEngine

_analytics = _load_module("submission_analytics", "regulatory-submissions/submission_analytics.py")
MetricType = _analytics.MetricType
SubmissionAnalyticsEngine = _analytics.SubmissionAnalyticsEngine
SubmissionOutcome = _analytics.SubmissionOutcome
SubmissionRecord = _analytics.SubmissionRecord


# ---------------------------------------------------------------------------
# Simulated clinical-analytics outputs
# ---------------------------------------------------------------------------
def _simulate_survival_analysis_results() -> dict:
    """Simulate survival analysis results from clinical-analytics module."""
    rng = np.random.default_rng(42)
    return {
        "study_id": "ONCO-DETECT-PIVOTAL-001",
        "n_patients": 450,
        "median_os_months": 18.5,
        "median_pfs_months": 12.3,
        "hazard_ratio": 0.72,
        "hr_ci_lower": 0.58,
        "hr_ci_upper": 0.89,
        "log_rank_p_value": 0.003,
        "km_survival_12m": 0.78,
        "km_survival_24m": 0.54,
        "kaplan_meier_times": np.sort(rng.exponential(15.0, 100)).tolist(),
    }


def _simulate_risk_stratification_results() -> dict:
    """Simulate risk stratification results from clinical-analytics module."""
    return {
        "model_name": "OncoRisk Neural Network v2.0",
        "auc_roc": 0.9456,
        "sensitivity": 0.9234,
        "specificity": 0.8891,
        "ppv": 0.8723,
        "npv": 0.9345,
        "f1_score": 0.8974,
        "n_risk_tiers": 3,
        "risk_tiers": {
            "low": {"threshold": 0.3, "n_patients": 180, "event_rate": 0.08},
            "medium": {"threshold": 0.7, "n_patients": 170, "event_rate": 0.25},
            "high": {"threshold": 1.0, "n_patients": 100, "event_rate": 0.62},
        },
        "calibration_slope": 1.02,
        "brier_score": 0.12,
    }


def _simulate_pkpd_results() -> dict:
    """Simulate PK/PD results from clinical-analytics module."""
    return {
        "model_type": "Two-compartment population PK",
        "n_subjects": 320,
        "clearance_l_h": 15.2,
        "volume_central_l": 45.6,
        "half_life_h": 8.4,
        "auc_0_inf": 1250.0,
        "cmax_ng_ml": 485.0,
        "tmax_h": 2.1,
    }


def run_full_pipeline() -> None:
    """Execute the full end-to-end submission pipeline."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 06: Full Regulatory Submission Pipeline")
    logger.info("=" * 80)

    # ==================================================================
    # PHASE 1: Gather clinical-analytics outputs
    # ==================================================================
    logger.info("")
    logger.info("PHASE 1: Gathering clinical-analytics outputs")
    logger.info("-" * 60)

    survival_results = _simulate_survival_analysis_results()
    risk_results = _simulate_risk_stratification_results()
    pkpd_results = _simulate_pkpd_results()

    logger.info(
        "Survival: median OS = %.1f months, HR = %.2f (p=%.4f)",
        survival_results["median_os_months"],
        survival_results["hazard_ratio"],
        survival_results["log_rank_p_value"],
    )
    logger.info(
        "Risk stratification: AUC = %.4f, sensitivity = %.4f", risk_results["auc_roc"], risk_results["sensitivity"]
    )
    logger.info("PK/PD: CL = %.1f L/h, t1/2 = %.1f h", pkpd_results["clearance_l_h"], pkpd_results["half_life_h"])

    # ==================================================================
    # PHASE 2: Initialize submission orchestrator
    # ==================================================================
    logger.info("")
    logger.info("PHASE 2: Initializing submission orchestrator")
    logger.info("-" * 60)

    config = SubmissionConfig(
        submission_id="SUB-2026-FULL-PIPELINE",
        submission_type=SubmissionType.ECTD_510K,
        sponsor_name="PAI Oncology Research Institute",
        device_name="OncoDetect AI v2.0",
        target_date="2026-06-30",
        indication="AI-assisted oncology tumor detection",
        predicate_device="TumorCAD v1.5 (K201234)",
        product_code="QAS",
    )
    orchestrator = create_standard_510k_workflow(config)
    logger.info("Orchestrator: %d tasks, phase=%s", orchestrator.task_count, orchestrator.current_phase.value)

    # ==================================================================
    # PHASE 3: Generate regulatory documents from clinical results
    # ==================================================================
    logger.info("")
    logger.info("PHASE 3: Generating regulatory documents")
    logger.info("-" * 60)

    doc_generator = RegulatoryDocumentGenerator(
        submission_id="SUB-2026-FULL-PIPELINE",
        sponsor_name="PAI Oncology Research Institute",
    )

    # CSR Synopsis incorporating survival analysis results
    csr_config = TemplateConfig(
        device_name="OncoDetect AI v2.0",
        indication="AI-assisted oncology tumor detection",
        study_id=survival_results["study_id"],
        study_title="Prospective Multi-Center Evaluation of OncoDetect AI v2.0",
        sample_size=survival_results["n_patients"],
        primary_endpoint="Sensitivity and specificity for tumor detection",
        secondary_endpoints=[
            f"Median OS: {survival_results['median_os_months']:.1f} months",
            f"HR: {survival_results['hazard_ratio']:.2f} (95% CI: {survival_results['hr_ci_lower']:.2f}-{survival_results['hr_ci_upper']:.2f})",
            f"12-month survival: {survival_results['km_survival_12m']:.0%}",
        ],
        performance_metrics={
            "Sensitivity": risk_results["sensitivity"],
            "Specificity": risk_results["specificity"],
            "AUC-ROC": risk_results["auc_roc"],
            "PPV": risk_results["ppv"],
            "NPV": risk_results["npv"],
            "F1 Score": risk_results["f1_score"],
        },
    )
    doc_csr = doc_generator.generate_document(TemplateType.CSR_SYNOPSIS, csr_config, author="Medical Writer")
    logger.info("CSR Synopsis: %d words, complete=%s", doc_csr.metadata.word_count, doc_csr.is_complete)

    # AI/ML PCCP incorporating risk stratification results
    pccp_config = TemplateConfig(
        device_name="OncoDetect AI v2.0",
        indication="AI-assisted oncology tumor detection",
        ai_ml_model_type=risk_results["model_name"],
        training_data_description="25,000 annotated oncology imaging cases from 12 sites",
        performance_metrics={
            "Sensitivity": risk_results["sensitivity"],
            "Specificity": risk_results["specificity"],
            "AUC-ROC": risk_results["auc_roc"],
            "Calibration Slope": risk_results["calibration_slope"],
            "Brier Score": risk_results["brier_score"],
        },
    )
    doc_pccp = doc_generator.generate_document(TemplateType.AI_ML_SUPPLEMENT, pccp_config, author="AI/ML Engineer")
    logger.info("PCCP: %d words, complete=%s", doc_pccp.metadata.word_count, doc_pccp.is_complete)

    # Additional documents
    doc_510k = doc_generator.generate_document(TemplateType.K510_SUMMARY, csr_config, author="Regulatory Lead")
    doc_sw = doc_generator.generate_document(TemplateType.SOFTWARE_DESCRIPTION, csr_config, author="SW QA Lead")
    doc_risk = doc_generator.generate_document(TemplateType.RISK_ANALYSIS, csr_config, author="Risk Engineer")
    doc_pred = doc_generator.generate_document(TemplateType.PREDICATE_COMPARISON, csr_config, author="Regulatory")

    logger.info("Total documents generated: %d", doc_generator.document_count)

    # ==================================================================
    # PHASE 4: Compile eCTD package
    # ==================================================================
    logger.info("")
    logger.info("PHASE 4: Compiling eCTD package")
    logger.info("-" * 60)

    compiler = ECTDCompiler("SUB-2026-FULL-PIPELINE", "510k", "0000")

    # Module 1: Regional
    compiler.add_document(
        "Cover Letter",
        DocumentType.COVER_LETTER,
        ECTDModule.M1_REGIONAL,
        "1.1",
        content="Cover letter for OncoDetect AI v2.0 510(k)",
    )
    compiler.add_document(
        "FDA Form 356(h)", DocumentType.FORM_FDA_356H, ECTDModule.M1_REGIONAL, "1.2", content=doc_510k.content[:200]
    )
    compiler.add_document(
        "510(k) Summary", DocumentType.SUMMARY_OVERVIEW, ECTDModule.M1_REGIONAL, "1.5", content=doc_510k.content[:500]
    )
    compiler.add_document(
        "Indications for Use",
        DocumentType.LABELING,
        ECTDModule.M1_REGIONAL,
        "1.6",
        content="Intended use statement for OncoDetect AI v2.0",
    )

    # Module 2: Summaries
    compiler.add_document(
        "Clinical Overview",
        DocumentType.CLINICAL_OVERVIEW,
        ECTDModule.M2_SUMMARIES,
        "2.5",
        content=doc_csr.content[:500],
    )
    compiler.add_document(
        "Clinical Summary", DocumentType.SUMMARY_OVERVIEW, ECTDModule.M2_SUMMARIES, "2.7", content=doc_csr.content[:500]
    )

    # Module 3: Quality
    compiler.add_document(
        "Software Description (IEC 62304)",
        DocumentType.SOFTWARE_DESCRIPTION,
        ECTDModule.M3_QUALITY,
        "3.2",
        content=doc_sw.content[:500],
    )

    # Module 4: Nonclinical
    compiler.add_document(
        "Risk Analysis (ISO 14971)",
        DocumentType.RISK_ANALYSIS,
        ECTDModule.M4_NONCLINICAL,
        "4.2",
        content=doc_risk.content[:500],
    )

    # Module 5: Clinical
    compiler.add_document(
        "Clinical Study Report",
        DocumentType.CLINICAL_STUDY_REPORT,
        ECTDModule.M5_CLINICAL,
        "5.3",
        content=doc_csr.content[:500],
    )
    compiler.add_document(
        "Statistical Analysis Plan",
        DocumentType.STATISTICAL_ANALYSIS_PLAN,
        ECTDModule.M5_CLINICAL,
        "5.3",
        content="SAP with survival endpoints",
    )
    compiler.add_document(
        "AI/ML PCCP", DocumentType.SOFTWARE_DESCRIPTION, ECTDModule.M5_CLINICAL, "5.3", content=doc_pccp.content[:500]
    )

    compilation_result = compiler.compile()
    logger.info(
        "eCTD compilation: valid=%s, docs=%d, completeness=%.1f%%",
        compilation_result.is_valid,
        compilation_result.total_documents,
        compilation_result.overall_completeness,
    )
    logger.info("Findings: errors=%d, warnings=%d", compilation_result.error_count, compilation_result.warning_count)

    # ==================================================================
    # PHASE 5: Run compliance validation
    # ==================================================================
    logger.info("")
    logger.info("PHASE 5: Running compliance validation")
    logger.info("-" * 60)

    validator = ComplianceValidator(
        "SUB-2026-FULL-PIPELINE",
        [Regulation.FDA_21CFR820, Regulation.FDA_21CFR11, Regulation.ISO_14971, Regulation.IEC_62304],
    )

    # Submit evidence for key requirements
    for req in validator.get_requirements_by_regulation(Regulation.FDA_21CFR820)[:10]:
        validator.submit_evidence(req.requirement_id, f"DHF reference for {req.section}")
    for req in validator.get_requirements_by_regulation(Regulation.ISO_14971)[:12]:
        validator.submit_evidence(req.requirement_id, f"RMF reference for {req.section}")
    for req in validator.get_requirements_by_regulation(Regulation.IEC_62304)[:10]:
        validator.submit_evidence(req.requirement_id, f"SDP reference for {req.section}")
    for req in validator.get_requirements_by_regulation(Regulation.FDA_21CFR11)[:8]:
        validator.submit_evidence(req.requirement_id, f"SVR reference for {req.section}")

    assessment = validator.run_assessment(assessed_by="regulatory_lead")
    logger.info("Compliance: score=%.1f%%, passing=%s", assessment.weighted_score, assessment.is_passing)

    # ==================================================================
    # PHASE 6: Regulatory intelligence check
    # ==================================================================
    logger.info("")
    logger.info("PHASE 6: Regulatory intelligence assessment")
    logger.info("-" * 60)

    intel_engine = RegulatoryIntelligenceEngine(
        "SUB-2026-FULL-PIPELINE",
        [Jurisdiction.FDA, Jurisdiction.EMA],
    )

    active_guidance = intel_engine.get_active_guidance()
    logger.info("Active guidance documents: %d", len(active_guidance))
    for doc in active_guidance[:3]:
        assessment_result = intel_engine.assess_impact(doc.guidance_id, assessed_by="regulatory_analyst")
        if assessment_result:
            logger.info(
                "  %s: impact=%s, effort=%d days",
                doc.title[:50],
                assessment_result.impact_level.value,
                assessment_result.estimated_effort_days,
            )

    high_impact = intel_engine.get_high_impact_items()
    logger.info("High-impact regulatory items: %d", len(high_impact))

    # ==================================================================
    # PHASE 7: Submission analytics
    # ==================================================================
    logger.info("")
    logger.info("PHASE 7: Submission analytics and benchmarking")
    logger.info("-" * 60)

    analytics = SubmissionAnalyticsEngine("PAI Oncology Portfolio")
    analytics.generate_synthetic_portfolio(40, seed=42)

    # Add current submission
    analytics.add_submission(
        SubmissionRecord(
            submission_id="SUB-2026-FULL-PIPELINE",
            submission_type="510k",
            device_name="OncoDetect AI v2.0",
            submitted_date="2026-03-01",
            decision_date="2026-06-15",
            outcome=SubmissionOutcome.CLEARED,
            deficiency_count=0,
            review_cycles=1,
            compliance_score=assessment.weighted_score,
        )
    )

    timeline = analytics.analyze_cycle_times("510k")
    logger.info(
        "510(k) cycle times: mean=%.1f days, median=%.1f days, on-time=%.1f%%",
        timeline.mean_cycle_time,
        timeline.median_cycle_time,
        timeline.on_time_rate,
    )

    benchmarks = analytics.benchmark_against_mdufa("510k")
    for bm in benchmarks:
        logger.info(
            "  %s: internal=%.1f vs MDUFA=%.1f (%s)",
            bm.metric_type.value,
            bm.internal_value,
            bm.benchmark_value,
            bm.assessment,
        )

    trend_dir, slope = analytics.analyze_trend(MetricType.CYCLE_TIME)
    logger.info("Cycle time trend: %s (slope=%.2f)", trend_dir.value, slope)

    recommendations = analytics.generate_recommendations()
    logger.info("Recommendations:")
    for rec in recommendations[:5]:
        logger.info("  - %s", rec)

    # ==================================================================
    # PHASE 8: Executive summary
    # ==================================================================
    logger.info("")
    logger.info("PHASE 8: Executive summary")
    logger.info("-" * 60)

    summary_lines = [
        "# Executive Summary: SUB-2026-FULL-PIPELINE",
        "",
        "## Clinical Evidence (from clinical-analytics)",
        f"- Survival: Median OS = {survival_results['median_os_months']:.1f} months, "
        f"HR = {survival_results['hazard_ratio']:.2f}",
        f"- Risk Model: AUC = {risk_results['auc_roc']:.4f}, Sensitivity = {risk_results['sensitivity']:.4f}",
        f"- PK/PD: CL = {pkpd_results['clearance_l_h']:.1f} L/h, t1/2 = {pkpd_results['half_life_h']:.1f} h",
        "",
        "## Regulatory Package",
        f"- Documents generated: {doc_generator.document_count}",
        f"- eCTD valid: {compilation_result.is_valid}",
        f"- eCTD completeness: {compilation_result.overall_completeness:.1f}%",
        f"- Compliance score: {assessment.weighted_score:.1f}%",
        f"- Passing: {assessment.is_passing}",
        "",
        "## Regulatory Landscape",
        f"- Active guidance monitored: {len(active_guidance)}",
        f"- High-impact items: {len(high_impact)}",
        "",
        "## Portfolio Performance",
        f"- Mean 510(k) cycle time: {timeline.mean_cycle_time:.1f} days",
        f"- Cycle time trend: {trend_dir.value}",
    ]
    executive_summary = "\n".join(summary_lines)
    print("\n" + executive_summary)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Example 06 complete. Full submission pipeline demonstrated successfully.")
    logger.info("This pipeline integrated clinical-analytics results into regulatory documents,")
    logger.info("compiled eCTD, validated compliance, assessed regulatory landscape, and")
    logger.info("computed submission analytics -- all in a single end-to-end workflow.")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_pipeline()
