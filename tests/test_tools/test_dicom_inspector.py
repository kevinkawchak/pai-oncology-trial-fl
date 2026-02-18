"""Tests for tools/dicom-inspector/dicom_inspector.py.

Covers enums, dataclasses, pydicom availability check, required tags,
constants, and helper functions. Actual DICOM file I/O tests are skipped
when pydicom is not available.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module("dicom_inspector", "tools/dicom-inspector/dicom_inspector.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestDicomModality:
    """Tests for the DicomModality enum."""

    def test_common_modalities(self):
        """CT, MR, PT, RTDOSE, RTSTRUCT are defined."""
        assert mod.DicomModality.CT.value == "CT"
        assert mod.DicomModality.MR.value == "MR"
        assert mod.DicomModality.PT.value == "PT"
        assert mod.DicomModality.RTDOSE.value == "RTDOSE"
        assert mod.DicomModality.RTSTRUCT.value == "RTSTRUCT"

    def test_other_modality(self):
        """OTHER modality exists."""
        assert mod.DicomModality.OTHER.value == "OTHER"

    def test_from_string_valid(self):
        """from_string resolves known modality strings."""
        assert mod.DicomModality.from_string("CT") == mod.DicomModality.CT
        assert mod.DicomModality.from_string("mr") == mod.DicomModality.MR

    def test_from_string_unknown(self):
        """from_string maps unknown strings to OTHER."""
        assert mod.DicomModality.from_string("UNKNOWN_MOD") == mod.DicomModality.OTHER

    def test_modality_count(self):
        """DicomModality has 14 members."""
        assert len(mod.DicomModality) == 14


class TestValidationSeverity:
    """Tests for the ValidationSeverity enum."""

    def test_members(self):
        """ValidationSeverity has ERROR, WARNING, INFO."""
        expected = {"ERROR", "WARNING", "INFO"}
        assert set(mod.ValidationSeverity.__members__.keys()) == expected

    def test_values(self):
        """Values are lowercase."""
        assert mod.ValidationSeverity.ERROR.value == "error"
        assert mod.ValidationSeverity.WARNING.value == "warning"
        assert mod.ValidationSeverity.INFO.value == "info"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestValidationFinding:
    """Tests for the ValidationFinding dataclass."""

    def test_creation(self):
        """ValidationFinding can be created with required fields."""
        vf = mod.ValidationFinding(
            tag_name="PatientID",
            severity=mod.ValidationSeverity.ERROR,
            message="Missing tag",
        )
        assert vf.tag_name == "PatientID"
        assert vf.severity == mod.ValidationSeverity.ERROR
        assert vf.expected == ""


class TestDicomInspectionResult:
    """Tests for the DicomInspectionResult dataclass."""

    def test_defaults(self):
        """Default result has empty fields and is_valid=False."""
        result = mod.DicomInspectionResult()
        assert result.file_path == ""
        assert result.modality == ""
        assert result.is_valid is False
        assert result.tags_present == []
        assert result.tags_missing == []

    def test_custom_values(self):
        """DicomInspectionResult accepts custom values."""
        result = mod.DicomInspectionResult(
            file_path="/path/to/file.dcm",
            modality="CT",
            is_valid=True,
        )
        assert result.file_path == "/path/to/file.dcm"
        assert result.modality == "CT"
        assert result.is_valid is True


class TestDicomSummary:
    """Tests for the DicomSummary dataclass."""

    def test_defaults(self):
        """Default summary has zero counts."""
        s = mod.DicomSummary()
        assert s.total_files == 0
        assert s.valid_count == 0
        assert s.patients == []


# ---------------------------------------------------------------------------
# Constants and reference data
# ---------------------------------------------------------------------------
class TestRequiredOncologyTags:
    """Tests for the REQUIRED_ONCOLOGY_TAGS constant."""

    def test_has_patient_id(self):
        """PatientID is a required tag."""
        assert "PatientID" in mod.REQUIRED_ONCOLOGY_TAGS

    def test_has_modality(self):
        """Modality is a required tag."""
        assert "Modality" in mod.REQUIRED_ONCOLOGY_TAGS

    def test_tag_count(self):
        """There are 14 required oncology tags."""
        assert len(mod.REQUIRED_ONCOLOGY_TAGS) == 14

    def test_tags_have_group_element(self):
        """Each tag value is a DICOM group,element string."""
        for tag_name, tag_id in mod.REQUIRED_ONCOLOGY_TAGS.items():
            parts = tag_id.split(",")
            assert len(parts) == 2


class TestSliceThicknessBounds:
    """Tests for SLICE_THICKNESS_BOUNDS_MM constant."""

    def test_ct_bounds(self):
        """CT slice thickness bounds are [0.5, 10.0]."""
        assert mod.SLICE_THICKNESS_BOUNDS_MM["CT"] == (0.5, 10.0)

    def test_pt_bounds(self):
        """PT slice thickness bounds are [1.0, 5.0]."""
        assert mod.SLICE_THICKNESS_BOUNDS_MM["PT"] == (1.0, 5.0)


# ---------------------------------------------------------------------------
# Pydicom availability
# ---------------------------------------------------------------------------
class TestPydicomAvailability:
    """Tests for HAS_PYDICOM flag and _require_pydicom."""

    def test_has_pydicom_is_bool(self):
        """HAS_PYDICOM is a boolean."""
        assert isinstance(mod.HAS_PYDICOM, bool)

    def test_require_pydicom_when_missing(self):
        """_require_pydicom raises RuntimeError when pydicom is absent."""
        if not mod.HAS_PYDICOM:
            with pytest.raises(RuntimeError, match="pydicom is required"):
                mod._require_pydicom()

    def test_inspect_without_pydicom(self):
        """inspect_dicom_file raises RuntimeError when pydicom absent."""
        if not mod.HAS_PYDICOM:
            with pytest.raises(RuntimeError, match="pydicom"):
                mod.inspect_dicom_file("/nonexistent/file.dcm")
