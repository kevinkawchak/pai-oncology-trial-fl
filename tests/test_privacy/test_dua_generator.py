"""Tests for privacy/dua-templates-generator/dua_generator.py — DUA generation.

Covers DUAGenerator creation, template generation for each DUAType,
security requirements per type, retention policies per type,
fully-identified data security escalation, status updates, amendments,
and section generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("dua_generator", "privacy/dua-templates-generator/dua_generator.py")

DUAGenerator = mod.DUAGenerator
DUAType = mod.DUAType
DUAStatus = mod.DUAStatus
DataCategory = mod.DataCategory
SecurityTier = mod.SecurityTier
DUAParty = mod.DUAParty
DUADocument = mod.DUADocument
SecurityRequirements = mod.SecurityRequirements
RetentionPolicy = mod.RetentionPolicy


class TestDUATypeEnum:
    """Tests for DUAType enum values."""

    def test_all_types_exist(self):
        """All five DUA types are defined."""
        assert DUAType.INTRA_INSTITUTIONAL.value == "intra_institutional"
        assert DUAType.MULTI_SITE.value == "multi_site"
        assert DUAType.COMMERCIAL.value == "commercial"
        assert DUAType.ACADEMIC_COLLABORATION.value == "academic_collaboration"
        assert DUAType.GOVERNMENT.value == "government"

    def test_dua_type_from_string(self):
        """DUAType can be constructed from string."""
        assert DUAType("commercial") == DUAType.COMMERCIAL


class TestDUAGeneratorCreation:
    """Tests for DUAGenerator initialization."""

    def test_default_creation(self):
        """Generator starts with no documents."""
        gen = DUAGenerator()
        assert gen.list_documents() == []


class TestDUAGeneration:
    """Tests for generating DUA documents by type."""

    def test_generate_intra_institutional(self):
        """Generates an intra-institutional DUA."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.INTRA_INSTITUTIONAL)
        assert doc.dua_type == DUAType.INTRA_INSTITUTIONAL
        assert doc.dua_id.startswith("DUA-")
        assert doc.status == DUAStatus.DRAFT

    def test_generate_multi_site(self):
        """Generates a multi-site DUA."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.MULTI_SITE, title="Multi-Site Trial DUA")
        assert doc.dua_type == DUAType.MULTI_SITE
        assert doc.title == "Multi-Site Trial DUA"

    def test_generate_commercial(self):
        """Generates a commercial DUA."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.COMMERCIAL)
        assert doc.dua_type == DUAType.COMMERCIAL

    def test_generate_academic(self):
        """Generates an academic collaboration DUA."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.ACADEMIC_COLLABORATION)
        assert doc.dua_type == DUAType.ACADEMIC_COLLABORATION

    def test_generate_government(self):
        """Generates a government DUA."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.GOVERNMENT)
        assert doc.dua_type == DUAType.GOVERNMENT

    def test_generate_from_string_type(self):
        """DUA type can be provided as a string."""
        gen = DUAGenerator()
        doc = gen.generate("commercial")
        assert doc.dua_type == DUAType.COMMERCIAL

    def test_generate_with_parties(self):
        """Parties are included in the document."""
        gen = DUAGenerator()
        party = DUAParty(organization_name="Test Hospital", role="provider")
        doc = gen.generate(DUAType.MULTI_SITE, parties=[party])
        assert len(doc.parties) == 1
        assert doc.parties[0].organization_name == "Test Hospital"

    def test_generate_auto_title(self):
        """Auto-generated title matches the DUA type."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.INTRA_INSTITUTIONAL)
        assert "Intra Institutional" in doc.title

    def test_generate_increments_id(self):
        """Each generated DUA gets a unique ID."""
        gen = DUAGenerator()
        doc1 = gen.generate(DUAType.MULTI_SITE)
        doc2 = gen.generate(DUAType.COMMERCIAL)
        assert doc1.dua_id != doc2.dua_id


class TestSecurityRequirements:
    """Tests for per-type security requirements."""

    def test_intra_institutional_no_dlp(self):
        """Intra-institutional DUA does not require DLP."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.INTRA_INSTITUTIONAL)
        assert doc.security_requirements.data_loss_prevention is False

    def test_commercial_requires_background_checks(self):
        """Commercial DUA requires background checks."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.COMMERCIAL)
        assert doc.security_requirements.background_checks is True

    def test_multi_site_requires_network_segmentation(self):
        """Multi-site DUA requires network segmentation."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.MULTI_SITE)
        assert doc.security_requirements.network_segmentation is True

    def test_fully_identified_data_escalates_security(self):
        """Fully identified data escalates all security flags to True."""
        gen = DUAGenerator()
        doc = gen.generate(
            DUAType.INTRA_INSTITUTIONAL,
            data_category=DataCategory.FULLY_IDENTIFIED,
        )
        sec = doc.security_requirements
        assert sec.mfa_required is True
        assert sec.data_loss_prevention is True
        assert sec.network_segmentation is True
        assert sec.annual_security_audit is True
        assert sec.background_checks is True


class TestRetentionPolicies:
    """Tests for per-type retention policies."""

    def test_intra_institutional_retention_3_years(self):
        """Intra-institutional retention is 3 years."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.INTRA_INSTITUTIONAL)
        assert doc.retention_policy.retention_period_years == 3

    def test_commercial_retention_7_years(self):
        """Commercial retention is 7 years with extension allowed."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.COMMERCIAL)
        assert doc.retention_policy.retention_period_years == 7
        assert doc.retention_policy.extension_allowed is True

    def test_government_retention_10_years(self):
        """Government retention is 10 years."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.GOVERNMENT)
        assert doc.retention_policy.retention_period_years == 10

    def test_government_nist_destruction(self):
        """Government uses NIST 800-88 purge destruction method."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.GOVERNMENT)
        assert doc.retention_policy.destruction_method == "nist_800_88_purge"


class TestDUALifecycle:
    """Tests for DUA status updates and amendments."""

    def test_update_status(self):
        """Status can be updated through the lifecycle."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.MULTI_SITE)
        assert gen.update_status(doc.dua_id, DUAStatus.APPROVED) is True
        updated = gen.get_document(doc.dua_id)
        assert updated.status == DUAStatus.APPROVED

    def test_update_status_unknown_id(self):
        """Updating status for unknown ID returns False."""
        gen = DUAGenerator()
        assert gen.update_status("DUA-FAKE", DUAStatus.APPROVED) is False

    def test_add_amendment(self):
        """Amendments are tracked on the document."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.COMMERCIAL)
        assert gen.add_amendment(doc.dua_id, "Extended data categories", approved_by="PI") is True
        updated = gen.get_document(doc.dua_id)
        assert len(updated.amendments) == 1
        assert updated.amendments[0]["description"] == "Extended data categories"

    def test_sections_generated(self):
        """Generated DUA document has standard sections."""
        gen = DUAGenerator()
        doc = gen.generate(DUAType.MULTI_SITE)
        section_titles = [s["title"] for s in doc.sections]
        assert "Preamble" in section_titles
        assert "Security Requirements" in section_titles
        assert "Breach Notification" in section_titles
        assert "Signatures" in section_titles
