"""Tests for regulatory-submissions/ectd_compiler.py.

Covers enums (ECTDModule, DocumentType, ValidationLevel, FileFormat),
dataclasses (ECTDDocument, ECTDSection, ModuleStructure, CompilationResult),
and ECTDCompiler:
  - Module initialization (5 ICH CTD modules)
  - Document addition and retrieval
  - Document removal
  - SHA-256 checksum computation
  - Validation (required sections, checksums, 510(k)-specific)
  - XML backbone generation
  - Full compilation pipeline
  - Compilation reporting
  - Completeness scoring
  - Serialization
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import load_module

mod = load_module(
    "ectd_compiler",
    "regulatory-submissions/ectd_compiler.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_compiler(sub_id="SUB-TEST", sub_type="510k", seq="0000"):
    return mod.ECTDCompiler(sub_id, sub_type, seq)


def _add_standard_docs(compiler):
    """Add a basic set of documents across modules."""
    compiler.add_document("Cover Letter", mod.DocumentType.COVER_LETTER, mod.ECTDModule.M1_REGIONAL, "1.1", "content")
    compiler.add_document("Form 356(h)", mod.DocumentType.FORM_FDA_356H, mod.ECTDModule.M1_REGIONAL, "1.2", "form")
    compiler.add_document(
        "Clinical Overview", mod.DocumentType.CLINICAL_OVERVIEW, mod.ECTDModule.M2_SUMMARIES, "2.5", "overview"
    )
    compiler.add_document("CSR", mod.DocumentType.CLINICAL_STUDY_REPORT, mod.ECTDModule.M5_CLINICAL, "5.3", "report")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestECTDModule:
    def test_members(self):
        names = set(mod.ECTDModule.__members__.keys())
        assert "M1_REGIONAL" in names
        assert "M5_CLINICAL" in names

    def test_module_count(self):
        assert len(mod.ECTDModule) == 5

    def test_is_str_enum(self):
        assert isinstance(mod.ECTDModule.M1_REGIONAL, str)


class TestDocumentType:
    def test_members(self):
        names = set(mod.DocumentType.__members__.keys())
        assert "COVER_LETTER" in names
        assert "CLINICAL_STUDY_REPORT" in names

    def test_has_many_types(self):
        assert len(mod.DocumentType) >= 10


class TestValidationLevel:
    def test_members(self):
        assert mod.ValidationLevel.ERROR.value == "error"
        assert mod.ValidationLevel.WARNING.value == "warning"
        assert mod.ValidationLevel.INFO.value == "info"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestECTDDocument:
    def test_creation(self):
        doc = mod.ECTDDocument(title="Test Doc")
        assert doc.title == "Test Doc"
        assert doc.document_id

    def test_compute_hash(self):
        doc = mod.ECTDDocument(title="Test")
        h = doc.compute_hash("test content")
        assert len(h) == 64
        assert doc.content_hash == h

    def test_xml_leaf(self):
        doc = mod.ECTDDocument(title="Test", leaf_id="leaf-001")
        doc.compute_hash("content")
        xml = doc.to_xml_leaf()
        assert "leaf-001" in xml
        assert "Test" in xml
        assert doc.content_hash in xml


class TestECTDSection:
    def test_document_count_empty(self):
        section = mod.ECTDSection()
        assert section.document_count() == 0

    def test_document_count_with_docs(self):
        doc = mod.ECTDDocument(title="D1")
        section = mod.ECTDSection(documents=[doc])
        assert section.document_count() == 1


class TestModuleStructure:
    def test_completeness_no_required(self):
        ms = mod.ModuleStructure(sections=[mod.ECTDSection(is_required=False)])
        score = ms.compute_completeness()
        assert score == 100.0

    def test_completeness_with_required(self):
        sections = [
            mod.ECTDSection(is_required=True, is_populated=True),
            mod.ECTDSection(is_required=True, is_populated=False),
        ]
        ms = mod.ModuleStructure(sections=sections)
        score = ms.compute_completeness()
        assert score == 50.0


# ---------------------------------------------------------------------------
# Compiler tests
# ---------------------------------------------------------------------------
class TestECTDCompiler:
    def test_init(self):
        c = _make_compiler()
        assert c.submission_id == "SUB-TEST"
        assert len(c.modules) == 5
        assert c.document_count == 0

    def test_add_document(self):
        c = _make_compiler()
        doc = c.add_document(
            "Cover Letter", mod.DocumentType.COVER_LETTER, mod.ECTDModule.M1_REGIONAL, "1.1", "content"
        )
        assert doc.title == "Cover Letter"
        assert doc.content_hash
        assert c.document_count == 1

    def test_get_document(self):
        c = _make_compiler()
        doc = c.add_document("CL", mod.DocumentType.COVER_LETTER, mod.ECTDModule.M1_REGIONAL, "1.1")
        retrieved = c.get_document(doc.document_id)
        assert retrieved is not None
        assert retrieved.title == "CL"

    def test_get_document_missing(self):
        c = _make_compiler()
        assert c.get_document("nonexistent") is None

    def test_get_documents_by_module(self):
        c = _make_compiler()
        c.add_document("D1", mod.DocumentType.COVER_LETTER, mod.ECTDModule.M1_REGIONAL, "1.1")
        c.add_document("D2", mod.DocumentType.FORM_FDA_356H, mod.ECTDModule.M1_REGIONAL, "1.2")
        docs = c.get_documents_by_module(mod.ECTDModule.M1_REGIONAL)
        assert len(docs) == 2

    def test_remove_document(self):
        c = _make_compiler()
        doc = c.add_document("CL", mod.DocumentType.COVER_LETTER, mod.ECTDModule.M1_REGIONAL, "1.1")
        assert c.remove_document(doc.document_id) is True
        assert c.document_count == 0

    def test_remove_nonexistent(self):
        c = _make_compiler()
        assert c.remove_document("nonexistent") is False

    def test_validate_empty(self):
        c = _make_compiler()
        findings = c.validate()
        errors = [f for f in findings if f.level == mod.ValidationLevel.ERROR]
        assert len(errors) > 0

    def test_validate_with_docs(self):
        c = _make_compiler()
        _add_standard_docs(c)
        findings = c.validate()
        assert isinstance(findings, list)

    def test_xml_backbone(self):
        c = _make_compiler()
        _add_standard_docs(c)
        xml = c.generate_xml_backbone()
        assert '<?xml version="1.0"' in xml
        assert "SUB-TEST" in xml
        assert "ectd" in xml

    def test_compile(self):
        c = _make_compiler()
        _add_standard_docs(c)
        result = c.compile()
        assert result.total_documents == 4
        assert result.xml_backbone
        assert result.compiled_at

    def test_compile_checksums_verified(self):
        c = _make_compiler()
        doc = c.add_document("D1", mod.DocumentType.COVER_LETTER, mod.ECTDModule.M1_REGIONAL, "1.1", "content")
        c.compile()
        updated_doc = c.get_document(doc.document_id)
        assert updated_doc.checksum_verified is True

    def test_compilation_report(self):
        c = _make_compiler()
        _add_standard_docs(c)
        result = c.compile()
        report = c.generate_compilation_report(result)
        assert "SUB-TEST" in report
        assert "Module Summary" in report
        assert "Validation Findings" in report

    def test_to_dict(self):
        c = _make_compiler()
        _add_standard_docs(c)
        d = c.to_dict()
        assert d["submission_id"] == "SUB-TEST"
        assert d["document_count"] == 4
        assert "modules" in d
