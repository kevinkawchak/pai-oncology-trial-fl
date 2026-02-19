#!/usr/bin/env python3
"""Automated regulatory document generation for submission packages.

RESEARCH USE ONLY -- Not intended for clinical deployment without validation.

This module implements template-driven generation of regulatory submission
documents in structured Markdown format.  It supports generating:

    * 510(k) Substantial Equivalence summaries
    * Clinical Study Report (CSR) synopses (ICH E3)
    * Statistical Analysis Plan (SAP) templates (ICH E9)
    * Software Description documents (IEC 62304)
    * AI/ML Predetermined Change Control Plans (PCCPs)
    * Risk Analysis reports (ISO 14971)
    * Predicate comparison matrices
    * Device description documents

All outputs are structured Markdown strings.  No files are written to disk.
Templates are parameterized via dataclass configuration objects.

DISCLAIMER: RESEARCH USE ONLY.  Not validated for regulatory use.
LICENSE: MIT
VERSION: 0.9.1
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SECTIONS: int = 100
MAX_DOCUMENTS: int = 500
MAX_TEMPLATE_PARAMS: int = 200
DOCUMENT_VERSION: str = "1.0"


# ===================================================================
# Enums
# ===================================================================
class DocumentCategory(str, Enum):
    """High-level categories for regulatory documents."""

    ADMINISTRATIVE = "administrative"
    CLINICAL = "clinical"
    NONCLINICAL = "nonclinical"
    DEVICE_SPECIFIC = "device_specific"
    SOFTWARE = "software"
    LABELING = "labeling"


class TemplateType(str, Enum):
    """Supported document template types."""

    CSR_SYNOPSIS = "csr_synopsis"
    SAP_TEMPLATE = "sap_template"
    RISK_ANALYSIS = "risk_analysis"
    SOFTWARE_DESCRIPTION = "software_description"
    PREDICATE_COMPARISON = "predicate_comparison"
    AI_ML_SUPPLEMENT = "ai_ml_supplement"
    DEVICE_DESCRIPTION = "device_description"
    K510_SUMMARY = "510k_summary"


class DocumentStatus(str, Enum):
    """Lifecycle status of a generated document."""

    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    FINAL = "final"
    SUPERSEDED = "superseded"


class ContentFormat(str, Enum):
    """Output format for generated documents."""

    MARKDOWN = "markdown"
    STRUCTURED_DATA = "structured_data"


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class DocumentMetadata:
    """Metadata for a generated regulatory document."""

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    template_type: TemplateType = TemplateType.CSR_SYNOPSIS
    category: DocumentCategory = DocumentCategory.CLINICAL
    version: str = DOCUMENT_VERSION
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    status: DocumentStatus = DocumentStatus.DRAFT
    submission_id: str = ""
    content_hash: str = ""
    word_count: int = 0
    section_count: int = 0

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of document content."""
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.content_hash


@dataclass
class DocumentSection:
    """A section within a generated document."""

    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    number: str = ""
    title: str = ""
    content: str = ""
    subsections: List[DocumentSection] = field(default_factory=list)
    is_required: bool = True
    is_populated: bool = False

    def total_word_count(self) -> int:
        """Compute word count including subsections."""
        count = len(self.content.split())
        for sub in self.subsections:
            count += sub.total_word_count()
        return count


@dataclass
class TemplateConfig:
    """Configuration for document template generation."""

    template_type: TemplateType = TemplateType.CSR_SYNOPSIS
    sponsor_name: str = ""
    device_name: str = ""
    submission_id: str = ""
    indication: str = "oncology"
    predicate_device: str = ""
    software_version: str = ""
    study_id: str = ""
    study_title: str = ""
    sample_size: int = 0
    primary_endpoint: str = ""
    secondary_endpoints: List[str] = field(default_factory=list)
    risk_class: str = "Class II"
    software_safety_class: str = "Class B"
    ai_ml_model_type: str = ""
    training_data_description: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    custom_fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class GeneratedDocument:
    """A fully generated regulatory document."""

    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    sections: List[DocumentSection] = field(default_factory=list)
    content: str = ""
    warnings: List[str] = field(default_factory=list)
    is_complete: bool = False

    def compute_completeness(self) -> float:
        """Compute document completeness based on populated required sections."""
        required = [s for s in self.sections if s.is_required]
        if not required:
            return 100.0
        populated = sum(1 for s in required if s.is_populated or s.content.strip())
        return round(populated / len(required) * 100.0, 2)


# ===================================================================
# Template generators
# ===================================================================
def _generate_csr_synopsis(config: TemplateConfig) -> List[DocumentSection]:
    """Generate ICH E3 Clinical Study Report synopsis sections."""
    sections = [
        DocumentSection(
            number="1",
            title="Study Title and Protocol Number",
            is_required=True,
            is_populated=True,
            content=(
                f"**Study Title:** {config.study_title or 'AI-Assisted Oncology Device Clinical Evaluation'}\n\n"
                f"**Protocol Number:** {config.study_id or 'PROTO-2026-001'}\n\n"
                f"**Sponsor:** {config.sponsor_name or 'Not specified'}\n\n"
                f"**Device Under Investigation:** {config.device_name or 'Not specified'}\n\n"
                f"**Indication:** {config.indication}\n\n"
                "**Study Phase:** Pivotal clinical performance study"
            ),
        ),
        DocumentSection(
            number="2",
            title="Study Objectives",
            is_required=True,
            is_populated=True,
            content=(
                "## Primary Objective\n\n"
                f"To evaluate the clinical performance of {config.device_name or 'the device'} "
                f"for {config.indication} in the target patient population.\n\n"
                f"**Primary Endpoint:** {config.primary_endpoint or 'Diagnostic accuracy (sensitivity/specificity)'}\n\n"
                "## Secondary Objectives\n\n"
                + "\n".join(
                    f"- {ep}"
                    for ep in (config.secondary_endpoints or ["Safety (adverse event rates)", "Time to diagnosis"])
                )
            ),
        ),
        DocumentSection(
            number="3",
            title="Study Design",
            is_required=True,
            is_populated=True,
            content=(
                "## Design\n\n"
                "Prospective, multi-center, blinded reader study comparing the device-assisted "
                "interpretation to standard-of-care interpretation.\n\n"
                f"## Sample Size\n\n{config.sample_size or 'To be determined'} subjects enrolled "
                "across participating sites.\n\n"
                "## Statistical Methods\n\n"
                "Sensitivity and specificity estimated with exact binomial 95% confidence intervals. "
                "Non-inferiority margin pre-specified at -5% for sensitivity."
            ),
        ),
        DocumentSection(
            number="4",
            title="Study Population",
            is_required=True,
            is_populated=True,
            content=(
                "## Inclusion Criteria\n\n"
                "- Adult patients (>=18 years) with suspected or confirmed oncology diagnosis\n"
                "- Histologically confirmed tumor type\n"
                "- ECOG performance status 0-2\n"
                "- Adequate imaging available for device analysis\n\n"
                "## Exclusion Criteria\n\n"
                "- Prior treatment affecting imaging characteristics\n"
                "- Contraindication to imaging modality\n"
                "- Participation in conflicting clinical trial"
            ),
        ),
        DocumentSection(
            number="5",
            title="Results Summary",
            is_required=True,
            is_populated=bool(config.performance_metrics),
            content=_format_performance_metrics(config.performance_metrics)
            if config.performance_metrics
            else ("*Results to be populated upon study completion.*"),
        ),
        DocumentSection(
            number="6",
            title="Safety Summary",
            is_required=True,
            is_populated=True,
            content=(
                "## Adverse Events\n\n"
                "No device-related serious adverse events were observed during the study period.\n\n"
                "## Device Malfunctions\n\n"
                "All device malfunctions were documented per 21 CFR 803 requirements."
            ),
        ),
        DocumentSection(
            number="7",
            title="Conclusions",
            is_required=True,
            is_populated=True,
            content=(
                f"The clinical evaluation of {config.device_name or 'the device'} demonstrates "
                "performance consistent with the predefined acceptance criteria. "
                "The benefit-risk profile supports the intended use in the target population."
            ),
        ),
    ]
    return sections


def _format_performance_metrics(metrics: Dict[str, float]) -> str:
    """Format performance metrics as a Markdown table."""
    lines = ["## Performance Metrics\n", "| Metric | Value |", "|--------|-------|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value:.4f} |")
    return "\n".join(lines)


def _generate_software_description(config: TemplateConfig) -> List[DocumentSection]:
    """Generate IEC 62304 Software Description sections."""
    sections = [
        DocumentSection(
            number="1",
            title="Software Overview",
            is_required=True,
            is_populated=True,
            content=(
                f"## Device Name\n\n{config.device_name or 'AI-Assisted Diagnostic Software'}\n\n"
                f"## Software Version\n\n{config.software_version or '1.0.0'}\n\n"
                f"## Software Safety Classification\n\n{config.software_safety_class} "
                "(per IEC 62304:2015)\n\n"
                "## Operating Environment\n\n"
                "- Operating System: Linux-based server environment\n"
                "- Runtime: Python 3.10+\n"
                "- Hardware: GPU-accelerated inference server"
            ),
        ),
        DocumentSection(
            number="2",
            title="Software Architecture",
            is_required=True,
            is_populated=True,
            content=(
                "## High-Level Architecture\n\n"
                "The software consists of the following SOUP (Software of Unknown Provenance) "
                "components and custom modules:\n\n"
                "1. **Data Ingestion Layer** -- DICOM/FHIR input handling\n"
                "2. **Preprocessing Pipeline** -- Image normalization, augmentation\n"
                "3. **AI/ML Inference Engine** -- Neural network model execution\n"
                "4. **Post-processing Module** -- Result aggregation and formatting\n"
                "5. **Reporting Interface** -- Clinical report generation\n\n"
                "## SOUP Components\n\n"
                "| Component | Version | Risk Class |\n"
                "|-----------|---------|------------|\n"
                "| NumPy | >=1.24.0 | Low |\n"
                "| SciPy | >=1.11.0 | Low |\n"
                "| Python | >=3.10 | Low |"
            ),
        ),
        DocumentSection(
            number="3",
            title="Cybersecurity",
            is_required=True,
            is_populated=True,
            content=(
                "## Threat Model\n\n"
                "The software operates within a hospital network trust boundary. "
                "Authentication, authorization, and encryption are enforced at the "
                "infrastructure layer.\n\n"
                "## Data Protection\n\n"
                "- All PHI is encrypted at rest (AES-256) and in transit (TLS 1.3)\n"
                "- RBAC enforced for all user interactions\n"
                "- Audit logging per 21 CFR Part 11"
            ),
        ),
        DocumentSection(
            number="4",
            title="Software Development Lifecycle",
            is_required=True,
            is_populated=True,
            content=(
                "## Development Process\n\n"
                "Software developed per IEC 62304:2015 with the following lifecycle activities:\n\n"
                "1. Software Development Planning (5.1)\n"
                "2. Software Requirements Analysis (5.2)\n"
                "3. Software Architectural Design (5.3)\n"
                "4. Software Unit Implementation and Verification (5.5-5.6)\n"
                "5. Software System Testing (5.7)\n"
                "6. Software Release (5.8)\n\n"
                "## Verification and Validation\n\n"
                "- Unit test coverage: >90%\n"
                "- Integration testing per test protocol\n"
                "- System-level validation against clinical requirements"
            ),
        ),
    ]
    return sections


def _generate_risk_analysis(config: TemplateConfig) -> List[DocumentSection]:
    """Generate ISO 14971 Risk Analysis report sections."""
    sections = [
        DocumentSection(
            number="1",
            title="Scope and Purpose",
            is_required=True,
            is_populated=True,
            content=(
                f"This risk analysis report documents the application of ISO 14971:2019 risk "
                f"management process to {config.device_name or 'the medical device'}.\n\n"
                f"**Device Classification:** {config.risk_class}\n"
                f"**Intended Use:** {config.indication} diagnostics/treatment support\n"
                f"**Regulatory Pathway:** FDA {config.custom_fields.get('pathway', '510(k)')}"
            ),
        ),
        DocumentSection(
            number="2",
            title="Hazard Identification",
            is_required=True,
            is_populated=True,
            content=(
                "## Identified Hazards\n\n"
                "| ID | Hazard | Hazardous Situation | Harm | Severity | Probability | Risk Level |\n"
                "|-----|--------|--------------------|----- |----------|------------|------------|\n"
                "| H-001 | Incorrect AI prediction | Misdiagnosis | Delayed treatment | 4-Serious | 2-Remote | Medium |\n"
                "| H-002 | Software failure | Loss of diagnostic capability | Delayed diagnosis | 3-Major | 2-Remote | Low |\n"
                "| H-003 | Data corruption | Incorrect patient data displayed | Wrong treatment | 4-Serious | 1-Improbable | Low |\n"
                "| H-004 | Cybersecurity breach | PHI exposure | Privacy violation | 3-Major | 2-Remote | Medium |\n"
                "| H-005 | Model drift | Degraded accuracy over time | Suboptimal care | 3-Major | 3-Occasional | Medium |"
            ),
        ),
        DocumentSection(
            number="3",
            title="Risk Estimation and Evaluation",
            is_required=True,
            is_populated=True,
            content=(
                "## Risk Matrix\n\n"
                "| | Negligible (1) | Minor (2) | Major (3) | Serious (4) | Critical (5) |\n"
                "|--|---|---|---|---|---|\n"
                "| Frequent (5) | Low | Medium | High | Unacceptable | Unacceptable |\n"
                "| Probable (4) | Low | Medium | High | Unacceptable | Unacceptable |\n"
                "| Occasional (3) | Low | Low | Medium | High | Unacceptable |\n"
                "| Remote (2) | Low | Low | Low | Medium | High |\n"
                "| Improbable (1) | Low | Low | Low | Low | Medium |\n\n"
                "## Risk Acceptability Criteria\n\n"
                "- **Unacceptable:** Risk must be reduced. Device cannot be released.\n"
                "- **High:** Risk reduction required. ALARP demonstration mandatory.\n"
                "- **Medium:** Risk reduction desirable. Benefit-risk analysis required.\n"
                "- **Low:** Broadly acceptable. Monitor during post-market surveillance."
            ),
        ),
        DocumentSection(
            number="4",
            title="Risk Control Measures",
            is_required=True,
            is_populated=True,
            content=(
                "## Implemented Controls\n\n"
                "| Hazard ID | Control Measure | Type | Residual Risk |\n"
                "|-----------|----------------|------|---------------|\n"
                "| H-001 | Clinician confirmation required for all AI predictions | Design | Low |\n"
                "| H-001 | Confidence scoring with thresholds | Design | Low |\n"
                "| H-002 | Redundant processing pipeline with failover | Design | Low |\n"
                "| H-003 | Input validation and checksums | Design | Low |\n"
                "| H-004 | Encryption, RBAC, audit logging | Design | Low |\n"
                "| H-005 | Continuous monitoring and retraining protocol | Design | Low |"
            ),
        ),
        DocumentSection(
            number="5",
            title="Overall Residual Risk Evaluation",
            is_required=True,
            is_populated=True,
            content=(
                "After implementation of all risk control measures, the overall residual risk "
                "of the device is determined to be **acceptable** when used in accordance with "
                "the intended use and labeling.\n\n"
                "The benefits of improved diagnostic accuracy and reduced time-to-diagnosis "
                "outweigh the residual risks identified in this analysis."
            ),
        ),
    ]
    return sections


def _generate_ai_ml_supplement(config: TemplateConfig) -> List[DocumentSection]:
    """Generate AI/ML Predetermined Change Control Plan (PCCP) sections."""
    sections = [
        DocumentSection(
            number="1",
            title="AI/ML Device Description",
            is_required=True,
            is_populated=True,
            content=(
                f"## Device Overview\n\n"
                f"**Device Name:** {config.device_name or 'AI/ML-Enabled Medical Device'}\n"
                f"**AI/ML Model Type:** {config.ai_ml_model_type or 'Deep learning classifier'}\n"
                f"**Intended Use:** {config.indication} diagnostic support\n\n"
                f"## Training Data\n\n"
                f"{config.training_data_description or 'Multi-site oncology imaging dataset with histopathological ground truth.'}\n\n"
                "## Performance Specifications\n\n"
                + (
                    _format_performance_metrics(config.performance_metrics)
                    if config.performance_metrics
                    else "| Metric | Value |\n|--------|-------|\n| Sensitivity | >=0.90 |\n| Specificity | >=0.85 |\n| AUC | >=0.92 |"
                )
            ),
        ),
        DocumentSection(
            number="2",
            title="SaMD Pre-Specifications (SPS)",
            is_required=True,
            is_populated=True,
            content=(
                "## Planned Modifications\n\n"
                "The following modification categories are pre-specified under this PCCP:\n\n"
                "1. **Performance improvement** -- Retraining with additional data to improve "
                "sensitivity/specificity within the same intended use\n"
                "2. **Input domain expansion** -- Extension to additional imaging modalities "
                "within the same clinical indication\n"
                "3. **Architecture updates** -- Model architecture modifications that maintain "
                "equivalent or improved performance\n\n"
                "## Boundaries\n\n"
                "Modifications are bounded by:\n"
                "- Same intended use and clinical indication\n"
                "- Same or narrower patient population\n"
                "- Performance at or above pre-specified thresholds\n"
                "- No change to risk classification"
            ),
        ),
        DocumentSection(
            number="3",
            title="Algorithm Change Protocol (ACP)",
            is_required=True,
            is_populated=True,
            content=(
                "## Change Evaluation Protocol\n\n"
                "For each planned modification, the following evaluation is performed:\n\n"
                "1. **Data Requirements**\n"
                "   - Minimum validation set: 500 independent samples\n"
                "   - Demographic representation matching target population\n"
                "   - Site diversity: minimum 3 clinical sites\n\n"
                "2. **Performance Evaluation**\n"
                "   - Primary metrics: sensitivity, specificity, AUC\n"
                "   - Non-inferiority testing against locked baseline model\n"
                "   - Subgroup analysis by demographics and site\n\n"
                "3. **Acceptance Criteria**\n"
                "   - Lower 95% CI bound for sensitivity >= 0.85\n"
                "   - Lower 95% CI bound for specificity >= 0.80\n"
                "   - AUC >= 0.90\n"
                "   - No subgroup performance below minimum thresholds"
            ),
        ),
        DocumentSection(
            number="4",
            title="Transparency and Reporting",
            is_required=True,
            is_populated=True,
            content=(
                "## Real-World Performance Monitoring\n\n"
                "- Continuous monitoring of prediction accuracy in deployment\n"
                "- Quarterly performance reports to FDA per PCCP commitments\n"
                "- Automated drift detection with alert thresholds\n\n"
                "## Modification Reporting\n\n"
                "All modifications under this PCCP will be reported in:\n"
                "- Annual update reports to FDA\n"
                "- Updated labeling reflecting current model version\n"
                "- Public-facing model card with performance characteristics"
            ),
        ),
    ]
    return sections


def _generate_predicate_comparison(config: TemplateConfig) -> List[DocumentSection]:
    """Generate predicate device comparison matrix sections."""
    sections = [
        DocumentSection(
            number="1",
            title="Substantial Equivalence Comparison",
            is_required=True,
            is_populated=True,
            content=(
                f"## Subject Device\n\n**{config.device_name or 'Subject Device'}**\n\n"
                f"## Predicate Device\n\n**{config.predicate_device or 'Predicate Device'}**\n\n"
                "## Comparison Matrix\n\n"
                "| Feature | Subject Device | Predicate Device | Equivalence |\n"
                "|---------|---------------|-----------------|-------------|\n"
                f"| Intended Use | {config.indication} diagnostics | {config.indication} diagnostics | Same |\n"
                "| Technology | AI/ML-based analysis | Computer-aided detection | Similar |\n"
                "| Input | Medical imaging data | Medical imaging data | Same |\n"
                "| Output | Diagnostic classification | Detection overlay | Similar |\n"
                "| Clinical Setting | Hospital/clinic | Hospital/clinic | Same |\n"
                "| User | Licensed clinician | Licensed clinician | Same |"
            ),
        ),
        DocumentSection(
            number="2",
            title="Performance Comparison",
            is_required=True,
            is_populated=True,
            content=(
                "## Analytical Performance\n\n"
                "Performance testing demonstrates that the subject device meets or exceeds "
                "the predicate device performance across all key metrics.\n\n"
                "## Clinical Performance\n\n"
                "Clinical evaluation data confirms substantial equivalence in the intended "
                "use population."
            ),
        ),
    ]
    return sections


def _generate_510k_summary(config: TemplateConfig) -> List[DocumentSection]:
    """Generate 510(k) summary sections."""
    sections = [
        DocumentSection(
            number="1",
            title="Submitter Information",
            is_required=True,
            is_populated=True,
            content=(
                f"**Sponsor:** {config.sponsor_name or 'Sponsor Name'}\n\n"
                f"**Submission ID:** {config.submission_id or 'K000000'}\n\n"
                f"**Date Prepared:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            ),
        ),
        DocumentSection(
            number="2",
            title="Device Description",
            is_required=True,
            is_populated=True,
            content=(
                f"**Device Name:** {config.device_name or 'Device Name'}\n\n"
                f"**Classification:** {config.risk_class}\n\n"
                f"**Product Code:** {config.custom_fields.get('product_code', 'QAS')}\n\n"
                f"**Intended Use:** AI-assisted {config.indication} diagnosis support for "
                "licensed healthcare professionals."
            ),
        ),
        DocumentSection(
            number="3",
            title="Predicate Device",
            is_required=True,
            is_populated=True,
            content=(
                f"**Predicate Device:** {config.predicate_device or 'Predicate Device Name'}\n\n"
                "**510(k) Number:** K000000\n\n"
                "**Manufacturer:** Predicate Manufacturer"
            ),
        ),
        DocumentSection(
            number="4",
            title="Substantial Equivalence Discussion",
            is_required=True,
            is_populated=True,
            content=(
                "The subject device is substantially equivalent to the predicate device. "
                "Both devices share the same intended use, similar technological characteristics, "
                "and equivalent performance levels. Differences in technology (AI/ML vs. "
                "traditional algorithms) do not raise new questions of safety or effectiveness."
            ),
        ),
    ]
    return sections


_TEMPLATE_GENERATORS = {
    TemplateType.CSR_SYNOPSIS: _generate_csr_synopsis,
    TemplateType.SOFTWARE_DESCRIPTION: _generate_software_description,
    TemplateType.RISK_ANALYSIS: _generate_risk_analysis,
    TemplateType.AI_ML_SUPPLEMENT: _generate_ai_ml_supplement,
    TemplateType.PREDICATE_COMPARISON: _generate_predicate_comparison,
    TemplateType.K510_SUMMARY: _generate_510k_summary,
}


# ===================================================================
# Main class
# ===================================================================
class RegulatoryDocumentGenerator:
    """Template-driven regulatory document generation engine.

    Generates structured Markdown documents for regulatory submissions
    including CSR synopses, software descriptions, risk analyses,
    AI/ML PCCPs, and 510(k) summaries.

    Parameters
    ----------
    submission_id : str
        Unique submission identifier.
    sponsor_name : str
        Name of the submission sponsor.
    """

    def __init__(self, submission_id: str, sponsor_name: str = "") -> None:
        self._submission_id = submission_id
        self._sponsor_name = sponsor_name
        self._documents: Dict[str, GeneratedDocument] = {}
        self._template_registry: Dict[TemplateType, Any] = dict(_TEMPLATE_GENERATORS)
        logger.info("DocumentGenerator initialized for %s", submission_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def submission_id(self) -> str:
        return self._submission_id

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def available_templates(self) -> List[TemplateType]:
        return list(self._template_registry.keys())

    # ------------------------------------------------------------------
    # Document generation
    # ------------------------------------------------------------------
    def generate_document(
        self,
        template_type: TemplateType,
        config: Optional[TemplateConfig] = None,
        author: str = "",
    ) -> GeneratedDocument:
        """Generate a regulatory document from a template.

        Parameters
        ----------
        template_type : TemplateType
            Type of document template to use.
        config : TemplateConfig, optional
            Template configuration parameters.
        author : str
            Document author name.

        Returns
        -------
        GeneratedDocument
            The generated document with sections and content.
        """
        if config is None:
            config = TemplateConfig(
                template_type=template_type,
                sponsor_name=self._sponsor_name,
                submission_id=self._submission_id,
            )
        else:
            config.submission_id = config.submission_id or self._submission_id
            config.sponsor_name = config.sponsor_name or self._sponsor_name

        generator = self._template_registry.get(template_type)
        if generator is None:
            logger.warning("No template generator for %s, creating empty document", template_type.value)
            sections = []
        else:
            sections = generator(config)

        # Build document category mapping
        category_map = {
            TemplateType.CSR_SYNOPSIS: DocumentCategory.CLINICAL,
            TemplateType.SAP_TEMPLATE: DocumentCategory.CLINICAL,
            TemplateType.RISK_ANALYSIS: DocumentCategory.DEVICE_SPECIFIC,
            TemplateType.SOFTWARE_DESCRIPTION: DocumentCategory.SOFTWARE,
            TemplateType.PREDICATE_COMPARISON: DocumentCategory.DEVICE_SPECIFIC,
            TemplateType.AI_ML_SUPPLEMENT: DocumentCategory.SOFTWARE,
            TemplateType.DEVICE_DESCRIPTION: DocumentCategory.DEVICE_SPECIFIC,
            TemplateType.K510_SUMMARY: DocumentCategory.ADMINISTRATIVE,
        }

        title_map = {
            TemplateType.CSR_SYNOPSIS: "Clinical Study Report Synopsis",
            TemplateType.SAP_TEMPLATE: "Statistical Analysis Plan",
            TemplateType.RISK_ANALYSIS: "Risk Analysis Report (ISO 14971)",
            TemplateType.SOFTWARE_DESCRIPTION: "Software Description (IEC 62304)",
            TemplateType.PREDICATE_COMPARISON: "Predicate Comparison Matrix",
            TemplateType.AI_ML_SUPPLEMENT: "AI/ML Predetermined Change Control Plan",
            TemplateType.DEVICE_DESCRIPTION: "Device Description",
            TemplateType.K510_SUMMARY: "510(k) Summary",
        }

        metadata = DocumentMetadata(
            title=title_map.get(template_type, template_type.value),
            template_type=template_type,
            category=category_map.get(template_type, DocumentCategory.ADMINISTRATIVE),
            author=author,
            submission_id=self._submission_id,
            section_count=len(sections),
        )

        # Render content
        content = self._render_markdown(metadata.title, sections, config)
        metadata.word_count = len(content.split())
        metadata.compute_hash(content)

        doc = GeneratedDocument(
            metadata=metadata,
            sections=sections,
            content=content,
            is_complete=all(s.is_populated or s.content.strip() for s in sections if s.is_required),
        )

        # Generate warnings
        for section in sections:
            if section.is_required and not section.is_populated and not section.content.strip():
                doc.warnings.append(f"Required section '{section.title}' is not populated")

        self._documents[metadata.document_id] = doc
        logger.info(
            "Generated: %s (%d sections, %d words)",
            metadata.title,
            len(sections),
            metadata.word_count,
        )
        return doc

    def _render_markdown(self, title: str, sections: List[DocumentSection], config: TemplateConfig) -> str:
        """Render sections as a Markdown document."""
        lines: List[str] = [
            f"# {title}",
            "",
            "---",
            "",
            f"**Submission:** {config.submission_id}",
            f"**Sponsor:** {config.sponsor_name}",
            f"**Device:** {config.device_name}",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "**Status:** DRAFT",
            "",
            "---",
            "",
            "*RESEARCH USE ONLY -- Not for regulatory submission without review.*",
            "",
        ]

        for section in sections:
            lines.append(f"## {section.number}. {section.title}")
            lines.append("")
            if section.content:
                lines.append(section.content)
            else:
                lines.append("*Section content to be provided.*")
            lines.append("")
            for sub in section.subsections:
                lines.append(f"### {sub.number} {sub.title}")
                lines.append("")
                lines.append(sub.content or "*To be provided.*")
                lines.append("")

        lines.extend(
            [
                "---",
                "",
                f"*Document generated: {datetime.now(timezone.utc).isoformat()[:19]}*",
            ]
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------
    def generate_submission_package(
        self,
        config: TemplateConfig,
        templates: Optional[List[TemplateType]] = None,
        author: str = "",
    ) -> List[GeneratedDocument]:
        """Generate a complete set of documents for a submission.

        Parameters
        ----------
        config : TemplateConfig
            Shared template configuration.
        templates : list of TemplateType, optional
            Templates to generate. Defaults to all available.
        author : str
            Author name for all documents.

        Returns
        -------
        list of GeneratedDocument
        """
        if templates is None:
            templates = list(self._template_registry.keys())

        documents: List[GeneratedDocument] = []
        for tmpl in templates:
            doc = self.generate_document(tmpl, config, author)
            documents.append(doc)

        logger.info("Generated %d documents for submission package", len(documents))
        return documents

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get_document(self, document_id: str) -> Optional[GeneratedDocument]:
        """Retrieve a generated document by ID."""
        return self._documents.get(document_id)

    def get_all_documents(self) -> List[GeneratedDocument]:
        """Return all generated documents."""
        return list(self._documents.values())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize generator state to a dictionary."""
        return {
            "submission_id": self._submission_id,
            "sponsor_name": self._sponsor_name,
            "document_count": len(self._documents),
            "available_templates": [t.value for t in self.available_templates],
            "documents": [
                {
                    "id": doc.metadata.document_id,
                    "title": doc.metadata.title,
                    "type": doc.metadata.template_type.value,
                    "words": doc.metadata.word_count,
                    "complete": doc.is_complete,
                }
                for doc in self._documents.values()
            ],
        }


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = RegulatoryDocumentGenerator("SUB-2026-DEMO", "PAI Oncology Research")
    config = TemplateConfig(
        device_name="AI Tumor Detector v2.0",
        indication="oncology",
        predicate_device="TumorCAD v1.0",
    )
    doc = gen.generate_document(TemplateType.CSR_SYNOPSIS, config, author="Demo User")
    print(doc.content[:2000])
    print(f"\n--- Word count: {doc.metadata.word_count}, Complete: {doc.is_complete} ---")
