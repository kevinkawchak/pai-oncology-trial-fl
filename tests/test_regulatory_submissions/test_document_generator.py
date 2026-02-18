"""Tests for regulatory-submissions/document_generator.py.

Covers enums (DocumentCategory, TemplateType, DocumentStatus, ContentFormat),
dataclasses (DocumentMetadata, DocumentSection, TemplateConfig, GeneratedDocument),
and RegulatoryDocumentGenerator:
  - Initialization
  - Template availability
  - CSR synopsis generation
  - Software description generation (IEC 62304)
  - Risk analysis generation (ISO 14971)
  - AI/ML PCCP generation
  - Predicate comparison generation
  - 510(k) summary generation
  - Batch document generation
  - Document retrieval
  - Completeness scoring
  - Content hash computation
  - Markdown rendering quality
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
    "document_generator",
    "regulatory-submissions/document_generator.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_generator():
    return mod.RegulatoryDocumentGenerator("SUB-TEST", "Test Sponsor")


def _make_config(**overrides):
    defaults = {
        "device_name": "AI Tumor Detector v1.0",
        "indication": "oncology",
        "sponsor_name": "Test Sponsor",
        "predicate_device": "Predicate v1.0",
        "software_version": "1.0.0",
    }
    defaults.update(overrides)
    return mod.TemplateConfig(**defaults)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestDocumentCategory:
    def test_members(self):
        names = set(mod.DocumentCategory.__members__.keys())
        assert "ADMINISTRATIVE" in names
        assert "CLINICAL" in names
        assert "SOFTWARE" in names
        assert "LABELING" in names

    def test_count(self):
        assert len(mod.DocumentCategory) == 6


class TestTemplateType:
    def test_members(self):
        names = set(mod.TemplateType.__members__.keys())
        assert "CSR_SYNOPSIS" in names
        assert "SOFTWARE_DESCRIPTION" in names
        assert "RISK_ANALYSIS" in names
        assert "AI_ML_SUPPLEMENT" in names
        assert "K510_SUMMARY" in names

    def test_count(self):
        assert len(mod.TemplateType) == 8


class TestDocumentStatus:
    def test_members(self):
        assert mod.DocumentStatus.DRAFT.value == "draft"
        assert mod.DocumentStatus.FINAL.value == "final"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestDocumentMetadata:
    def test_creation(self):
        meta = mod.DocumentMetadata(title="Test Document")
        assert meta.title == "Test Document"
        assert meta.document_id

    def test_compute_hash(self):
        meta = mod.DocumentMetadata()
        h = meta.compute_hash("test content")
        assert len(h) == 64
        assert meta.content_hash == h


class TestDocumentSection:
    def test_word_count_empty(self):
        section = mod.DocumentSection()
        assert section.total_word_count() == 0

    def test_word_count_with_content(self):
        section = mod.DocumentSection(content="one two three")
        assert section.total_word_count() == 3


class TestGeneratedDocument:
    def test_completeness_all_populated(self):
        sections = [
            mod.DocumentSection(is_required=True, is_populated=True, content="content"),
            mod.DocumentSection(is_required=True, is_populated=True, content="more"),
        ]
        doc = mod.GeneratedDocument(sections=sections)
        assert doc.compute_completeness() == 100.0

    def test_completeness_partial(self):
        sections = [
            mod.DocumentSection(is_required=True, is_populated=True, content="c"),
            mod.DocumentSection(is_required=True, is_populated=False, content=""),
        ]
        doc = mod.GeneratedDocument(sections=sections)
        assert doc.compute_completeness() == 50.0

    def test_completeness_no_required(self):
        sections = [mod.DocumentSection(is_required=False)]
        doc = mod.GeneratedDocument(sections=sections)
        assert doc.compute_completeness() == 100.0


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------
class TestRegulatoryDocumentGenerator:
    def test_init(self):
        gen = _make_generator()
        assert gen.submission_id == "SUB-TEST"
        assert gen.document_count == 0

    def test_available_templates(self):
        gen = _make_generator()
        templates = gen.available_templates
        assert len(templates) >= 5

    def test_generate_csr_synopsis(self):
        gen = _make_generator()
        config = _make_config(
            study_title="Pivotal Study",
            sample_size=200,
            primary_endpoint="Sensitivity",
        )
        doc = gen.generate_document(mod.TemplateType.CSR_SYNOPSIS, config, author="Writer")
        assert doc.metadata.title == "Clinical Study Report Synopsis"
        assert doc.metadata.word_count > 100
        assert doc.is_complete is True
        assert "RESEARCH USE ONLY" in doc.content

    def test_generate_software_description(self):
        gen = _make_generator()
        doc = gen.generate_document(mod.TemplateType.SOFTWARE_DESCRIPTION, _make_config(), author="SW Lead")
        assert "IEC 62304" in doc.metadata.title or "Software" in doc.metadata.title
        assert doc.metadata.word_count > 50

    def test_generate_risk_analysis(self):
        gen = _make_generator()
        doc = gen.generate_document(mod.TemplateType.RISK_ANALYSIS, _make_config())
        assert "Risk" in doc.metadata.title or "ISO 14971" in doc.metadata.title
        assert doc.is_complete is True

    def test_generate_ai_ml_supplement(self):
        gen = _make_generator()
        config = _make_config(
            ai_ml_model_type="CNN ResNet-50",
            training_data_description="25k annotated images",
            performance_metrics={"AUC": 0.95, "Sensitivity": 0.92},
        )
        doc = gen.generate_document(mod.TemplateType.AI_ML_SUPPLEMENT, config)
        assert "PCCP" in doc.metadata.title or "AI/ML" in doc.metadata.title
        assert doc.is_complete is True

    def test_generate_predicate_comparison(self):
        gen = _make_generator()
        doc = gen.generate_document(mod.TemplateType.PREDICATE_COMPARISON, _make_config())
        assert "Predicate" in doc.metadata.title or "Equivalence" in doc.content
        assert doc.metadata.word_count > 50

    def test_generate_510k_summary(self):
        gen = _make_generator()
        doc = gen.generate_document(mod.TemplateType.K510_SUMMARY, _make_config())
        assert "510" in doc.metadata.title
        assert doc.is_complete is True

    def test_generate_submission_package(self):
        gen = _make_generator()
        config = _make_config()
        docs = gen.generate_submission_package(config, author="Team Lead")
        assert len(docs) >= 5
        assert gen.document_count >= 5

    def test_get_document(self):
        gen = _make_generator()
        doc = gen.generate_document(mod.TemplateType.CSR_SYNOPSIS, _make_config())
        retrieved = gen.get_document(doc.metadata.document_id)
        assert retrieved is not None
        assert retrieved.metadata.title == doc.metadata.title

    def test_get_all_documents(self):
        gen = _make_generator()
        gen.generate_document(mod.TemplateType.CSR_SYNOPSIS, _make_config())
        gen.generate_document(mod.TemplateType.RISK_ANALYSIS, _make_config())
        all_docs = gen.get_all_documents()
        assert len(all_docs) == 2

    def test_content_hash_unique(self):
        gen = _make_generator()
        d1 = gen.generate_document(mod.TemplateType.CSR_SYNOPSIS, _make_config())
        d2 = gen.generate_document(mod.TemplateType.RISK_ANALYSIS, _make_config())
        assert d1.metadata.content_hash != d2.metadata.content_hash

    def test_to_dict(self):
        gen = _make_generator()
        gen.generate_document(mod.TemplateType.CSR_SYNOPSIS, _make_config())
        d = gen.to_dict()
        assert d["submission_id"] == "SUB-TEST"
        assert d["document_count"] == 1
        assert "documents" in d

    def test_performance_metrics_in_csr(self):
        gen = _make_generator()
        config = _make_config(performance_metrics={"AUC": 0.95, "Sensitivity": 0.90})
        doc = gen.generate_document(mod.TemplateType.CSR_SYNOPSIS, config)
        assert "0.95" in doc.content or "0.9500" in doc.content
