"""Tests for regulatory-submissions/regulatory_intelligence.py.

Covers enums (Jurisdiction, GuidanceStatus, ImpactLevel, GuidanceCategory,
EnforcementType, DeviceClassification), dataclasses (GuidanceDocument,
RegulatoryUpdate, ImpactAssessment, ComplianceTimeline, EnforcementAction,
ClassificationEntry), and RegulatoryIntelligenceEngine:
  - Initialization and guidance database loading
  - Guidance retrieval by jurisdiction and category
  - Guidance search by keyword
  - Active guidance filtering
  - Impact assessment with scoring
  - Compliance timeline generation
  - Enforcement action tracking
  - Device classification lookups
  - Intelligence report generation
  - Custom guidance addition
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
    "regulatory_intelligence",
    "regulatory-submissions/regulatory_intelligence.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_engine(jurisdictions=None):
    j = jurisdictions or [mod.Jurisdiction.FDA, mod.Jurisdiction.EMA]
    return mod.RegulatoryIntelligenceEngine("SUB-TEST", j)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestJurisdiction:
    def test_members(self):
        names = set(mod.Jurisdiction.__members__.keys())
        assert "FDA" in names
        assert "EMA" in names
        assert "PMDA" in names
        assert "MHRA" in names

    def test_count(self):
        assert len(mod.Jurisdiction) == 7

    def test_is_str(self):
        assert isinstance(mod.Jurisdiction.FDA, str)


class TestGuidanceStatus:
    def test_members(self):
        assert mod.GuidanceStatus.DRAFT.value == "draft"
        assert mod.GuidanceStatus.FINAL.value == "final"
        assert mod.GuidanceStatus.SUPERSEDED.value == "superseded"
        assert mod.GuidanceStatus.WITHDRAWN.value == "withdrawn"


class TestImpactLevel:
    def test_members(self):
        assert mod.ImpactLevel.HIGH.value == "high"
        assert mod.ImpactLevel.INFORMATIONAL.value == "informational"

    def test_count(self):
        assert len(mod.ImpactLevel) == 4


class TestDeviceClassification:
    def test_members(self):
        names = set(mod.DeviceClassification.__members__.keys())
        assert "CLASS_I" in names
        assert "CLASS_II" in names
        assert "CLASS_III" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestGuidanceDocument:
    def test_creation(self):
        doc = mod.GuidanceDocument(title="Test Guidance", jurisdiction=mod.Jurisdiction.FDA)
        assert doc.title == "Test Guidance"
        assert doc.guidance_id

    def test_is_active(self):
        doc = mod.GuidanceDocument(status=mod.GuidanceStatus.FINAL)
        assert doc.is_active() is True
        doc2 = mod.GuidanceDocument(status=mod.GuidanceStatus.WITHDRAWN)
        assert doc2.is_active() is False

    def test_days_since_publication(self):
        doc = mod.GuidanceDocument(publication_date="2026-01-01")
        days = doc.days_since_publication("2026-01-31")
        assert days == 30


class TestImpactAssessment:
    def test_impact_score_high(self):
        ia = mod.ImpactAssessment(impact_level=mod.ImpactLevel.HIGH, affected_sections=["M2", "M5"])
        score = ia.impact_score()
        assert 80.0 <= score <= 100.0

    def test_impact_score_low(self):
        ia = mod.ImpactAssessment(impact_level=mod.ImpactLevel.LOW, affected_sections=[])
        score = ia.impact_score()
        assert score == 20.0


class TestClassificationEntry:
    def test_creation(self):
        entry = mod.ClassificationEntry(product_code="QAS", device_name="Test CAD")
        assert entry.product_code == "QAS"


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------
class TestRegulatoryIntelligenceEngine:
    def test_init(self):
        engine = _make_engine()
        assert engine.submission_id == "SUB-TEST"
        assert engine.guidance_count > 0

    def test_init_all_jurisdictions(self):
        engine = mod.RegulatoryIntelligenceEngine("SUB-ALL")
        assert engine.guidance_count > 0

    def test_guidance_by_jurisdiction(self):
        engine = _make_engine()
        fda_docs = engine.get_guidance_by_jurisdiction(mod.Jurisdiction.FDA)
        assert len(fda_docs) > 0
        for doc in fda_docs:
            assert doc.jurisdiction == mod.Jurisdiction.FDA

    def test_guidance_by_category(self):
        engine = _make_engine()
        ai_docs = engine.get_guidance_by_category(mod.GuidanceCategory.AI_ML)
        assert len(ai_docs) > 0

    def test_active_guidance(self):
        engine = _make_engine()
        active = engine.get_active_guidance()
        assert len(active) > 0
        for doc in active:
            assert doc.is_active()

    def test_search_guidance_ai(self):
        engine = _make_engine()
        results = engine.search_guidance("AI")
        assert len(results) > 0

    def test_search_guidance_no_results(self):
        engine = _make_engine()
        results = engine.search_guidance("xyznonexistent")
        assert len(results) == 0

    def test_add_guidance(self):
        engine = _make_engine()
        initial = engine.guidance_count
        engine.add_guidance(mod.GuidanceDocument(title="Custom Guidance", jurisdiction=mod.Jurisdiction.FDA))
        assert engine.guidance_count == initial + 1

    def test_assess_impact(self):
        engine = _make_engine()
        active = engine.get_active_guidance()
        assessment = engine.assess_impact(active[0].guidance_id, assessed_by="analyst")
        assert assessment is not None
        assert assessment.impact_level in mod.ImpactLevel

    def test_assess_impact_nonexistent(self):
        engine = _make_engine()
        result = engine.assess_impact("nonexistent")
        assert result is None

    def test_high_impact_items(self):
        engine = _make_engine()
        active = engine.get_active_guidance()
        for doc in active:
            engine.assess_impact(doc.guidance_id)
        high = engine.get_high_impact_items()
        assert isinstance(high, list)

    def test_compliance_timeline(self):
        engine = _make_engine()
        active = engine.get_active_guidance()
        timeline = engine.generate_compliance_timeline(active[0].guidance_id, "2026-03-01")
        assert timeline is not None
        assert timeline.total_duration_days > 0
        assert len(timeline.milestones) > 0

    def test_compliance_timeline_nonexistent(self):
        engine = _make_engine()
        result = engine.generate_compliance_timeline("nonexistent")
        assert result is None

    def test_enforcement_action_tracking(self):
        engine = _make_engine()
        action = mod.EnforcementAction(
            title="Test Warning",
            jurisdiction=mod.Jurisdiction.FDA,
            action_type=mod.EnforcementType.WARNING_LETTER,
            device_type="AI diagnostic",
            relevance_score=0.8,
        )
        engine.add_enforcement_action(action)
        relevant = engine.get_relevant_enforcement_actions("AI diagnostic")
        assert len(relevant) >= 1

    def test_classification_lookup_by_code(self):
        engine = _make_engine()
        results = engine.lookup_classification(product_code="QAS")
        assert len(results) >= 1
        assert results[0].product_code == "QAS"

    def test_classification_lookup_by_name(self):
        engine = _make_engine()
        results = engine.lookup_classification(device_name="CAD")
        assert len(results) >= 1

    def test_intelligence_report(self):
        engine = _make_engine()
        report = engine.generate_intelligence_report()
        assert "Regulatory Intelligence Report" in report
        assert "FDA" in report

    def test_to_dict(self):
        engine = _make_engine()
        d = engine.to_dict()
        assert d["submission_id"] == "SUB-TEST"
        assert "guidance_count" in d
        assert "jurisdictions" in d
