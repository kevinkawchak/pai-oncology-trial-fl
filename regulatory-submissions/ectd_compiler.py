#!/usr/bin/env python3
"""eCTD compilation engine -- Electronic Common Technical Document assembly.

RESEARCH USE ONLY -- Not intended for clinical deployment without validation.

This module implements the compilation engine for generating eCTD (Electronic
Common Technical Document) module structures conforming to ICH M4 organization
and FDA Technical Conformance Guide requirements.  It produces XML backbone
stubs, validates document completeness per module, computes SHA-256 checksums
for all registered documents, and generates compilation reports.

Supported eCTD module structure (ICH CTD):
    * Module 1 -- Regional Administrative Information
    * Module 2 -- Common Technical Document Summaries
    * Module 3 -- Quality (CMC)
    * Module 4 -- Nonclinical Study Reports
    * Module 5 -- Clinical Study Reports

The compiler does NOT write files to disk.  All outputs are in-memory data
structures (dataclasses) and structured Markdown/XML strings suitable for
downstream processing or display.

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

MAX_DOCUMENTS_PER_MODULE: int = 500
MAX_SECTIONS: int = 200
ECTD_VERSION: str = "4.0"
XML_ENCODING: str = "UTF-8"
HASH_ALGORITHM: str = "sha256"


# ===================================================================
# Enums
# ===================================================================
class ECTDModule(str, Enum):
    """ICH CTD module identifiers."""

    M1_REGIONAL = "m1_regional"
    M2_SUMMARIES = "m2_summaries"
    M3_QUALITY = "m3_quality"
    M4_NONCLINICAL = "m4_nonclinical"
    M5_CLINICAL = "m5_clinical"


class DocumentType(str, Enum):
    """Regulatory document types within eCTD submissions."""

    COVER_LETTER = "cover_letter"
    FORM_FDA_356H = "form_fda_356h"
    DEVICE_DESCRIPTION = "device_description"
    SOFTWARE_DESCRIPTION = "software_description"
    CLINICAL_STUDY_REPORT = "clinical_study_report"
    STATISTICAL_ANALYSIS_PLAN = "statistical_analysis_plan"
    RISK_ANALYSIS = "risk_analysis"
    PREDICATE_COMPARISON = "predicate_comparison"
    LABELING = "labeling"
    SUMMARY_OVERVIEW = "summary_overview"
    NONCLINICAL_OVERVIEW = "nonclinical_overview"
    CLINICAL_OVERVIEW = "clinical_overview"
    QUALITY_OVERALL_SUMMARY = "quality_overall_summary"
    BIOCOMPATIBILITY = "biocompatibility"
    STERILITY = "sterility"
    EMC_REPORT = "emc_report"


class ValidationLevel(str, Enum):
    """Severity levels for eCTD validation findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class FileFormat(str, Enum):
    """Accepted file formats for eCTD documents."""

    PDF = "pdf"
    XML = "xml"
    STF = "stf"
    SAS_XPORT = "sas_xport"
    DATASET_XML = "dataset_xml"


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class ECTDDocument:
    """A single document within the eCTD submission."""

    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    document_type: DocumentType = DocumentType.COVER_LETTER
    module: ECTDModule = ECTDModule.M1_REGIONAL
    section_number: str = ""
    file_format: FileFormat = FileFormat.PDF
    content_hash: str = ""
    file_size_bytes: int = 0
    version: str = "1.0"
    effective_date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    author: str = ""
    status: str = "draft"
    leaf_id: str = ""
    operation: str = "new"
    checksum_verified: bool = False

    def compute_hash(self, content: str = "") -> str:
        """Compute SHA-256 hash for document content."""
        if not content:
            content = f"{self.document_id}:{self.title}:{self.version}"
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.content_hash

    def to_xml_leaf(self) -> str:
        """Generate XML leaf element for eCTD backbone."""
        leaf_id = self.leaf_id or self.document_id[:8]
        return (
            f'  <leaf ID="{leaf_id}" '
            f'operation="{self.operation}" '
            f'checksum="{self.content_hash}" '
            f'checksum-type="{HASH_ALGORITHM}">\n'
            f"    <title>{self.title}</title>\n"
            f"  </leaf>"
        )


@dataclass
class ECTDSection:
    """A section within an eCTD module."""

    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    section_number: str = ""
    title: str = ""
    module: ECTDModule = ECTDModule.M1_REGIONAL
    documents: List[ECTDDocument] = field(default_factory=list)
    subsections: List[ECTDSection] = field(default_factory=list)
    is_required: bool = False
    is_populated: bool = False

    def document_count(self) -> int:
        """Total documents in this section and subsections."""
        count = len(self.documents)
        for sub in self.subsections:
            count += sub.document_count()
        return count


@dataclass
class ModuleStructure:
    """Complete structure for one eCTD module."""

    module: ECTDModule = ECTDModule.M1_REGIONAL
    module_title: str = ""
    sections: List[ECTDSection] = field(default_factory=list)
    total_documents: int = 0
    is_complete: bool = False
    completeness_score: float = 0.0

    def compute_completeness(self) -> float:
        """Compute completeness score based on required sections."""
        required = [s for s in self.sections if s.is_required]
        if not required:
            self.completeness_score = 100.0
            return self.completeness_score
        populated = sum(1 for s in required if s.is_populated or s.document_count() > 0)
        self.completeness_score = round(populated / len(required) * 100.0, 2)
        self.is_complete = self.completeness_score >= 100.0
        return self.completeness_score


@dataclass
class ValidationFinding:
    """A single validation finding from eCTD compilation."""

    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: ValidationLevel = ValidationLevel.INFO
    module: ECTDModule = ECTDModule.M1_REGIONAL
    section: str = ""
    message: str = ""
    rule_id: str = ""
    remediation: str = ""


@dataclass
class CompilationResult:
    """Result of an eCTD compilation run."""

    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = ""
    compiled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modules: List[ModuleStructure] = field(default_factory=list)
    findings: List[ValidationFinding] = field(default_factory=list)
    total_documents: int = 0
    total_sections: int = 0
    overall_completeness: float = 0.0
    xml_backbone: str = ""
    is_valid: bool = False
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def compute_summary(self) -> None:
        """Compute summary statistics for the compilation result."""
        self.error_count = sum(1 for f in self.findings if f.level == ValidationLevel.ERROR)
        self.warning_count = sum(1 for f in self.findings if f.level == ValidationLevel.WARNING)
        self.info_count = sum(1 for f in self.findings if f.level == ValidationLevel.INFO)
        self.is_valid = self.error_count == 0
        if self.modules:
            scores = [m.completeness_score for m in self.modules]
            self.overall_completeness = round(float(np.mean(scores)), 2) if scores else 0.0
        self.total_documents = sum(m.total_documents for m in self.modules)
        self.total_sections = sum(len(m.sections) for m in self.modules)


# ===================================================================
# Standard module definitions
# ===================================================================
_MODULE_1_SECTIONS: List[Tuple[str, str, bool]] = [
    ("1.1", "Cover Letter", False),
    ("1.2", "Form FDA 356(h)", True),
    ("1.3", "Administrative Information", True),
    ("1.4", "Labeling", True),
    ("1.5", "510(k) Summary or Statement", True),
    ("1.6", "Indications for Use Statement", True),
    ("1.7", "Truthful and Accurate Statement", True),
    ("1.8", "Financial Certification or Disclosure", False),
]

_MODULE_2_SECTIONS: List[Tuple[str, str, bool]] = [
    ("2.1", "Table of Contents", True),
    ("2.2", "Introduction", True),
    ("2.3", "Quality Overall Summary", False),
    ("2.4", "Nonclinical Overview", False),
    ("2.5", "Clinical Overview", True),
    ("2.6", "Nonclinical Written and Tabulated Summaries", False),
    ("2.7", "Clinical Summary", True),
]

_MODULE_3_SECTIONS: List[Tuple[str, str, bool]] = [
    ("3.1", "Table of Contents", False),
    ("3.2", "Body of Data", False),
    ("3.3", "Literature References", False),
]

_MODULE_4_SECTIONS: List[Tuple[str, str, bool]] = [
    ("4.1", "Table of Contents", False),
    ("4.2", "Study Reports", False),
    ("4.3", "Literature References", False),
]

_MODULE_5_SECTIONS: List[Tuple[str, str, bool]] = [
    ("5.1", "Table of Contents", False),
    ("5.2", "Tabular Listing of All Clinical Studies", True),
    ("5.3", "Clinical Study Reports", True),
    ("5.4", "Literature References", False),
]


# ===================================================================
# Main class
# ===================================================================
class ECTDCompiler:
    """Electronic Common Technical Document compilation engine.

    Generates eCTD module structures, validates document completeness per
    FDA Technical Conformance Guide, produces XML backbone stubs, and
    checksums all documents with SHA-256.

    Parameters
    ----------
    submission_id : str
        Unique submission identifier.
    submission_type : str
        Submission pathway (e.g., '510k', 'pma', 'de_novo').
    sequence_number : str
        eCTD sequence number (e.g., '0000').
    """

    _MODULE_DEFS: Dict[ECTDModule, Tuple[str, List[Tuple[str, str, bool]]]] = {
        ECTDModule.M1_REGIONAL: ("Module 1: Regional Administrative Information", _MODULE_1_SECTIONS),
        ECTDModule.M2_SUMMARIES: ("Module 2: Common Technical Document Summaries", _MODULE_2_SECTIONS),
        ECTDModule.M3_QUALITY: ("Module 3: Quality", _MODULE_3_SECTIONS),
        ECTDModule.M4_NONCLINICAL: ("Module 4: Nonclinical Study Reports", _MODULE_4_SECTIONS),
        ECTDModule.M5_CLINICAL: ("Module 5: Clinical Study Reports", _MODULE_5_SECTIONS),
    }

    def __init__(
        self,
        submission_id: str,
        submission_type: str = "510k",
        sequence_number: str = "0000",
    ) -> None:
        self._submission_id = submission_id
        self._submission_type = submission_type
        self._sequence_number = sequence_number
        self._modules: Dict[ECTDModule, ModuleStructure] = {}
        self._documents: Dict[str, ECTDDocument] = {}
        self._findings: List[ValidationFinding] = []
        self._initialize_modules()
        logger.info(
            "ECTDCompiler initialized: %s (type=%s, seq=%s)",
            submission_id,
            submission_type,
            sequence_number,
        )

    def _initialize_modules(self) -> None:
        """Initialize the standard eCTD module structure."""
        for mod_enum, (title, section_defs) in self._MODULE_DEFS.items():
            sections = []
            for sec_num, sec_title, is_required in section_defs:
                section = ECTDSection(
                    section_number=sec_num,
                    title=sec_title,
                    module=mod_enum,
                    is_required=is_required,
                )
                sections.append(section)
            module = ModuleStructure(
                module=mod_enum,
                module_title=title,
                sections=sections,
            )
            self._modules[mod_enum] = module

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def submission_id(self) -> str:
        return self._submission_id

    @property
    def modules(self) -> Dict[ECTDModule, ModuleStructure]:
        return dict(self._modules)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def add_document(
        self,
        title: str,
        document_type: DocumentType,
        module: ECTDModule,
        section_number: str,
        content: str = "",
        author: str = "",
        version: str = "1.0",
    ) -> ECTDDocument:
        """Add a document to the eCTD structure."""
        if len(self._documents) >= MAX_DOCUMENTS_PER_MODULE * len(ECTDModule):
            raise ValueError("Maximum document capacity reached")

        doc = ECTDDocument(
            title=title,
            document_type=document_type,
            module=module,
            section_number=section_number,
            author=author,
            version=version,
        )
        doc.compute_hash(content)
        doc.file_size_bytes = len(content.encode("utf-8")) if content else 0
        doc.leaf_id = f"leaf-{doc.document_id[:8]}"
        self._documents[doc.document_id] = doc

        # Place document in the appropriate section
        if module in self._modules:
            for section in self._modules[module].sections:
                if section.section_number == section_number:
                    section.documents.append(doc)
                    section.is_populated = True
                    break

        self._modules[module].total_documents += 1
        logger.info("Document added: %s -> %s/%s", title, module.value, section_number)
        return doc

    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the eCTD structure."""
        if document_id not in self._documents:
            return False
        doc = self._documents.pop(document_id)
        if doc.module in self._modules:
            for section in self._modules[doc.module].sections:
                section.documents = [d for d in section.documents if d.document_id != document_id]
            self._modules[doc.module].total_documents -= 1
        return True

    def get_document(self, document_id: str) -> Optional[ECTDDocument]:
        """Retrieve a document by ID."""
        return self._documents.get(document_id)

    def get_documents_by_module(self, module: ECTDModule) -> List[ECTDDocument]:
        """Retrieve all documents in a specific module."""
        return [d for d in self._documents.values() if d.module == module]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> List[ValidationFinding]:
        """Validate the eCTD structure for completeness and conformance."""
        self._findings.clear()

        # Check required sections
        for mod_enum, module in self._modules.items():
            for section in module.sections:
                if section.is_required and not section.is_populated and section.document_count() == 0:
                    self._findings.append(
                        ValidationFinding(
                            level=ValidationLevel.ERROR,
                            module=mod_enum,
                            section=section.section_number,
                            message=f"Required section {section.section_number} '{section.title}' is empty",
                            rule_id="ECTD-REQ-001",
                            remediation=f"Add document(s) to section {section.section_number}",
                        )
                    )

        # Check document checksums
        for doc in self._documents.values():
            if not doc.content_hash:
                self._findings.append(
                    ValidationFinding(
                        level=ValidationLevel.WARNING,
                        module=doc.module,
                        section=doc.section_number,
                        message=f"Document '{doc.title}' has no checksum",
                        rule_id="ECTD-CHK-001",
                        remediation="Compute SHA-256 checksum for document",
                    )
                )

        # Check for Module 1 mandatory elements (510(k))
        if self._submission_type == "510k":
            m1_docs = self.get_documents_by_module(ECTDModule.M1_REGIONAL)
            doc_types = {d.document_type for d in m1_docs}
            if DocumentType.FORM_FDA_356H not in doc_types:
                self._findings.append(
                    ValidationFinding(
                        level=ValidationLevel.ERROR,
                        module=ECTDModule.M1_REGIONAL,
                        section="1.2",
                        message="Form FDA 356(h) is required for 510(k) submissions",
                        rule_id="ECTD-510K-001",
                        remediation="Add FDA Form 356(h) to Module 1 section 1.2",
                    )
                )

        # Check Module 5 for clinical data
        m5_docs = self.get_documents_by_module(ECTDModule.M5_CLINICAL)
        if not m5_docs:
            self._findings.append(
                ValidationFinding(
                    level=ValidationLevel.WARNING,
                    module=ECTDModule.M5_CLINICAL,
                    section="5.3",
                    message="No clinical study reports found in Module 5",
                    rule_id="ECTD-CLIN-001",
                    remediation="Add clinical study report(s) to Module 5",
                )
            )

        logger.info("Validation complete: %d finding(s)", len(self._findings))
        return list(self._findings)

    # ------------------------------------------------------------------
    # XML backbone
    # ------------------------------------------------------------------
    def generate_xml_backbone(self) -> str:
        """Generate the eCTD XML backbone structure."""
        lines: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f"<!-- eCTD Backbone for submission {self._submission_id} -->",
            f"<!-- Sequence: {self._sequence_number} -->",
            f"<!-- Generated: {datetime.now(timezone.utc).isoformat()} -->",
            f'<ectd version="{ECTD_VERSION}">',
            f'  <submission id="{self._submission_id}" type="{self._submission_type}" '
            f'sequence="{self._sequence_number}">',
        ]

        for mod_enum in ECTDModule:
            module = self._modules.get(mod_enum)
            if module is None:
                continue
            lines.append(f'    <module id="{mod_enum.value}" title="{module.module_title}">')
            for section in module.sections:
                lines.append(
                    f'      <section number="{section.section_number}" '
                    f'title="{section.title}" '
                    f'required="{str(section.is_required).lower()}">'
                )
                for doc in section.documents:
                    lines.append(f"      {doc.to_xml_leaf()}")
                lines.append("      </section>")
            lines.append("    </module>")

        lines.extend(["  </submission>", "</ectd>"])
        xml_output = "\n".join(lines)
        logger.info("XML backbone generated: %d lines", len(lines))
        return xml_output

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    def compile(self) -> CompilationResult:
        """Run full eCTD compilation: validate, compute checksums, generate backbone."""
        # Compute checksums for all documents
        for doc in self._documents.values():
            if not doc.content_hash:
                doc.compute_hash()
            doc.checksum_verified = True

        # Compute completeness for all modules
        for module in self._modules.values():
            module.compute_completeness()

        # Run validation
        findings = self.validate()

        # Generate XML backbone
        xml_backbone = self.generate_xml_backbone()

        # Build result
        result = CompilationResult(
            submission_id=self._submission_id,
            modules=list(self._modules.values()),
            findings=findings,
            xml_backbone=xml_backbone,
        )
        result.compute_summary()
        logger.info(
            "Compilation complete: valid=%s, docs=%d, errors=%d, warnings=%d",
            result.is_valid,
            result.total_documents,
            result.error_count,
            result.warning_count,
        )
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_compilation_report(self, result: Optional[CompilationResult] = None) -> str:
        """Generate a Markdown compilation report."""
        if result is None:
            result = self.compile()

        lines: List[str] = [
            f"# eCTD Compilation Report: {self._submission_id}",
            "",
            f"**Submission Type:** {self._submission_type}",
            f"**Sequence:** {self._sequence_number}",
            f"**eCTD Version:** {ECTD_VERSION}",
            f"**Compiled At:** {result.compiled_at[:19]}",
            f"**Overall Status:** {'VALID' if result.is_valid else 'INVALID'}",
            f"**Overall Completeness:** {result.overall_completeness:.1f}%",
            "",
            "## Module Summary",
            "",
            "| Module | Title | Documents | Completeness | Status |",
            "|--------|-------|-----------|-------------|--------|",
        ]
        for module in result.modules:
            status = "Complete" if module.is_complete else "Incomplete"
            lines.append(
                f"| {module.module.value} | {module.module_title} | "
                f"{module.total_documents} | {module.completeness_score:.1f}% | {status} |"
            )

        lines.extend(["", "## Validation Findings", ""])
        if result.findings:
            lines.append("| Level | Module | Section | Message | Rule |")
            lines.append("|-------|--------|---------|---------|------|")
            for finding in result.findings:
                lines.append(
                    f"| {finding.level.value.upper()} | {finding.module.value} | "
                    f"{finding.section} | {finding.message} | {finding.rule_id} |"
                )
        else:
            lines.append("No findings.")

        lines.extend(
            [
                "",
                "## Statistics",
                "",
                f"- Total Documents: {result.total_documents}",
                f"- Total Sections: {result.total_sections}",
                f"- Errors: {result.error_count}",
                f"- Warnings: {result.warning_count}",
                f"- Info: {result.info_count}",
                "",
                f"*Report generated: {datetime.now(timezone.utc).isoformat()[:19]}*",
            ]
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize compiler state to a dictionary."""
        return {
            "submission_id": self._submission_id,
            "submission_type": self._submission_type,
            "sequence_number": self._sequence_number,
            "document_count": len(self._documents),
            "modules": {
                m.value: {
                    "title": self._modules[m].module_title,
                    "documents": self._modules[m].total_documents,
                    "completeness": self._modules[m].completeness_score,
                }
                for m in ECTDModule
                if m in self._modules
            },
        }


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    compiler = ECTDCompiler("SUB-2026-DEMO", "510k", "0000")
    compiler.add_document(
        "Cover Letter",
        DocumentType.COVER_LETTER,
        ECTDModule.M1_REGIONAL,
        "1.1",
        content="To Whom It May Concern...",
    )
    compiler.add_document(
        "FDA Form 356(h)",
        DocumentType.FORM_FDA_356H,
        ECTDModule.M1_REGIONAL,
        "1.2",
        content="Form content...",
    )
    result = compiler.compile()
    print(compiler.generate_compilation_report(result))
