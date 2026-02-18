"""Tests for clinical-analytics/clinical_interoperability.py.

Covers enums (ClinicalStandard with 10 members, MappingConfidence),
dataclasses (VocabularyMapping, SchemaAlignment),
and the ClinicalInteroperabilityEngine class: vocabulary mapping,
schema alignment, ICD-10 to SNOMED, LOINC mapping, FHIR validation,
CDISC export, and cross-site alignment report.
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
    "clinical_interoperability",
    "clinical-analytics/clinical_interoperability.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestClinicalStandard:
    """Tests for the ClinicalStandard enum."""

    def test_clinical_standard_has_10_members(self):
        """ClinicalStandard has 10 supported standards."""
        assert len(mod.ClinicalStandard) == 10

    def test_fhir_r4_exists(self):
        """FHIR_R4 standard is defined."""
        assert hasattr(mod.ClinicalStandard, "FHIR_R4")

    def test_icd_10_cm_exists(self):
        """ICD_10_CM standard is defined."""
        assert hasattr(mod.ClinicalStandard, "ICD_10_CM")

    def test_snomed_ct_exists(self):
        """SNOMED_CT standard is defined."""
        assert hasattr(mod.ClinicalStandard, "SNOMED_CT")

    def test_loinc_exists(self):
        """LOINC standard is defined."""
        assert hasattr(mod.ClinicalStandard, "LOINC")

    def test_cdisc_sdtm_exists(self):
        """CDISC SDTM standard is defined."""
        assert hasattr(mod.ClinicalStandard, "CDISC_SDTM")

    def test_rxnorm_exists(self):
        """RxNorm standard is defined."""
        assert hasattr(mod.ClinicalStandard, "RxNorm")

    def test_is_str_enum(self):
        """ClinicalStandard members are instances of str."""
        assert isinstance(mod.ClinicalStandard.LOINC, str)


class TestMappingConfidence:
    """Tests for the MappingConfidence enum."""

    def test_mapping_confidence_members(self):
        """MappingConfidence has expected members."""
        names = set(mod.MappingConfidence.__members__.keys())
        assert "EXACT" in names
        assert "BROADER" in names or "BROAD" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestVocabularyMapping:
    """Tests for the VocabularyMapping dataclass."""

    def test_creation(self):
        """VocabularyMapping is created with expected fields."""
        mapping = mod.VocabularyMapping(
            source_code="C34.1",
            source_system=mod.ClinicalStandard.ICD_10_CM,
            target_code="254637007",
            target_system=mod.ClinicalStandard.SNOMED_CT,
        )
        assert mapping.source_code == "C34.1"
        assert mapping.target_code == "254637007"

    def test_default_confidence(self):
        """VocabularyMapping has a default confidence level of EXACT."""
        mapping = mod.VocabularyMapping(
            source_code="TEST",
            source_system=mod.ClinicalStandard.LOINC,
            target_code="TEST",
            target_system=mod.ClinicalStandard.LOINC,
        )
        assert mapping.confidence is not None
        assert mapping.confidence == mod.MappingConfidence.EXACT


# ---------------------------------------------------------------------------
# ClinicalInteroperabilityEngine tests
# ---------------------------------------------------------------------------
class TestClinicalInteroperabilityEngine:
    """Tests for the ClinicalInteroperabilityEngine."""

    def test_init_default(self):
        """Engine initializes with default configuration."""
        engine = mod.ClinicalInteroperabilityEngine()
        assert engine.site_id is not None

    def test_map_icd10_to_snomed_known(self):
        """Known ICD-10 code maps to SNOMED-CT."""
        engine = mod.ClinicalInteroperabilityEngine()
        mapping = engine.map_icd10_to_snomed("C34.1")
        assert mapping.target_code != ""
        assert mapping.confidence == mod.MappingConfidence.EXACT

    def test_map_icd10_to_snomed_unknown(self):
        """Unknown ICD-10 code returns non-exact mapping."""
        engine = mod.ClinicalInteroperabilityEngine()
        mapping = engine.map_icd10_to_snomed("Z99.99")
        # Should be unmapped or approximate
        assert mapping.confidence != mod.MappingConfidence.EXACT or mapping.target_code == ""

    def test_vocabulary_mapping_audit(self):
        """Vocabulary mapping operations are recorded in the audit log."""
        engine = mod.ClinicalInteroperabilityEngine()
        engine.map_icd10_to_snomed("C34.1")
        assert len(engine.audit_log) >= 1

    def test_loinc_mapping(self):
        """LOINC code mapping returns a VocabularyMapping."""
        engine = mod.ClinicalInteroperabilityEngine()
        mapping = engine.map_vocabulary(
            "718-7",
            mod.ClinicalStandard.LOINC,
            mod.ClinicalStandard.LOINC,
        )
        assert isinstance(mapping, mod.VocabularyMapping)

    def test_fhir_validation_valid(self):
        """Valid FHIR resource passes validation."""
        engine = mod.ClinicalInteroperabilityEngine()
        resource = {
            "resourceType": "Patient",
            "id": "pt-001",
            "name": [{"family": "Doe", "given": ["John"]}],
        }
        result = engine.validate_fhir_resource(resource)
        assert result["valid"] is True
        assert len(result.get("errors", [])) == 0

    def test_fhir_validation_missing_resource_type(self):
        """FHIR resource without resourceType fails validation."""
        engine = mod.ClinicalInteroperabilityEngine()
        resource = {"id": "pt-002"}
        result = engine.validate_fhir_resource(resource)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_fhir_validation_missing_id(self):
        """FHIR resource without id fails validation."""
        engine = mod.ClinicalInteroperabilityEngine()
        resource = {"resourceType": "Observation"}
        result = engine.validate_fhir_resource(resource)
        assert result["valid"] is False

    def test_cdisc_export(self):
        """CDISC export produces records with expected structure."""
        engine = mod.ClinicalInteroperabilityEngine()
        data = [
            {"study_id": "NCT-001", "patient_id": "P1", "age": 55},
            {"study_id": "NCT-001", "patient_id": "P2", "age": 62},
        ]
        result = engine.export_to_cdisc(data)
        assert "export_id" in result
        # Should have at least processed some records
        total_rows = result.get("total_rows", 0)
        assert total_rows >= 2

    def test_schema_alignment(self):
        """Schema alignment computes similarity score between two schemas."""
        engine = mod.ClinicalInteroperabilityEngine()
        schema_a = {"subject_id": "str", "age": "int", "sex": "str"}
        schema_b = {"subject_id": "str", "age": "float", "weight": "float"}
        result = engine.align_schemas(schema_a, schema_b)
        assert isinstance(result, mod.SchemaAlignment)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_cross_site_report(self):
        """Cross-site alignment report covers pairwise comparisons."""
        engine = mod.ClinicalInteroperabilityEngine()
        sites = {
            "SITE-A": {"subject_id": "str", "age": "int", "sex": "str"},
            "SITE-B": {"subject_id": "str", "age": "int", "weight": "float"},
            "SITE-C": {"subject_id": "str", "tumor_stage": "str"},
        }
        report = engine.generate_cross_site_alignment_report(sites)
        assert isinstance(report, mod.CrossSiteSchemaReport)
        # The alignment matrix should be a square matrix with shape (3, 3)
        assert report.alignment_matrix.shape == (3, 3)
        # Diagonal should be 1.0 (self-alignment)
        for i in range(3):
            assert report.alignment_matrix[i, i] == 1.0
