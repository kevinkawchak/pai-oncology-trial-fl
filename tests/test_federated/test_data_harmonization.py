"""Tests for federated/data_harmonization.py — DataHarmonizer.

Covers creation, harmonize_batch, field mapping (DICOM/FHIR codes),
unit conversion, encoding normalization (zscore/minmax), edge cases,
and HarmonizationReport contents.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import PROJECT_ROOT, load_module

mod = load_module("federated.data_harmonization", "federated/data_harmonization.py")

DataHarmonizer = mod.DataHarmonizer
HarmonizationReport = mod.HarmonizationReport
ONCOLOGY_VOCABULARY = mod.ONCOLOGY_VOCABULARY
UNIT_CONVERSIONS = mod.UNIT_CONVERSIONS


class TestDataHarmonizerCreation:
    """Tests for DataHarmonizer instantiation."""

    def test_creation_defaults(self):
        dh = DataHarmonizer()
        assert isinstance(dh.vocabulary, dict)
        assert len(dh.vocabulary) > 0
        assert dh.normalisation is None

    def test_creation_custom_vocabulary(self):
        custom = {"custom_field": {"dicom": "999", "fhir": "obs-custom", "display": "Custom", "unit": "mg"}}
        dh = DataHarmonizer(vocabulary=custom)
        assert "custom_field" in dh.vocabulary

    def test_creation_with_target_units(self):
        dh = DataHarmonizer(target_units={"tumor_volume": "mm3"})
        assert dh.target_units["tumor_volume"] == "mm3"

    def test_creation_with_normalisation(self):
        dh = DataHarmonizer(normalisation="zscore")
        assert dh.normalisation == "zscore"


class TestFieldMapping:
    """Tests for vocabulary-based field name mapping."""

    def test_map_canonical_name_unchanged(self):
        dh = DataHarmonizer()
        assert dh.map_field_name("tumor_volume") == "tumor_volume"

    def test_map_dicom_code_to_canonical(self):
        dh = DataHarmonizer()
        # DICOM code for tumor_volume is "118565006"
        result = dh.map_field_name("118565006")
        assert result == "tumor_volume"

    def test_map_fhir_code_to_canonical(self):
        dh = DataHarmonizer()
        result = dh.map_field_name("observation-tumor-volume")
        assert result == "tumor_volume"

    def test_map_unknown_field_returns_original(self):
        dh = DataHarmonizer()
        assert dh.map_field_name("completely_unknown") == "completely_unknown"

    def test_map_case_insensitive_match(self):
        dh = DataHarmonizer()
        # "Tumor_Volume" lowercase with underscore -> "tumor_volume"
        result = dh.map_field_name("Tumor_Volume")
        assert result == "tumor_volume"

    def test_map_record_all_fields(self):
        dh = DataHarmonizer()
        record = {"118565006": 25.0, "observation-pdl1": 80.0, "unknown_field": "abc"}
        mapped = dh.map_record(record)
        assert "tumor_volume" in mapped
        assert "pdl1_expression" in mapped
        assert "unknown_field" in mapped


class TestUnitConversion:
    """Tests for unit conversion functions."""

    def test_convert_mm_to_cm(self):
        dh = DataHarmonizer()
        result = dh.convert_units(100.0, "mm", "cm")
        npt.assert_allclose(result, 10.0)

    def test_convert_cm_to_mm(self):
        dh = DataHarmonizer()
        result = dh.convert_units(5.0, "cm", "mm")
        npt.assert_allclose(result, 50.0)

    def test_convert_cgy_to_gy(self):
        dh = DataHarmonizer()
        result = dh.convert_units(200.0, "cGy", "Gy")
        npt.assert_allclose(result, 2.0)

    def test_convert_same_unit_no_change(self):
        dh = DataHarmonizer()
        result = dh.convert_units(42.0, "cm", "cm")
        npt.assert_allclose(result, 42.0)

    def test_convert_fahrenheit_to_celsius(self):
        dh = DataHarmonizer()
        result = dh.convert_units(212.0, "F", "C")
        npt.assert_allclose(result, 100.0, atol=1e-6)

    def test_convert_celsius_to_fahrenheit(self):
        dh = DataHarmonizer()
        result = dh.convert_units(0.0, "C", "F")
        npt.assert_allclose(result, 32.0, atol=1e-6)

    def test_convert_unknown_units_returns_original(self):
        dh = DataHarmonizer()
        result = dh.convert_units(99.0, "bananas", "apples")
        npt.assert_allclose(result, 99.0)


class TestFeatureNormalisation:
    """Tests for zscore and minmax normalization."""

    def test_zscore_normalisation(self):
        dh = DataHarmonizer(normalisation="zscore")
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed, stats = dh.normalise_features(x)
        npt.assert_allclose(np.mean(normed, axis=0), [0.0, 0.0], atol=1e-10)
        npt.assert_allclose(np.std(normed, axis=0), [1.0, 1.0], atol=0.1)
        assert "mean" in stats
        assert "std" in stats

    def test_minmax_normalisation(self):
        dh = DataHarmonizer(normalisation="minmax")
        x = np.array([[1.0, 10.0], [5.0, 20.0], [9.0, 30.0]])
        normed, stats = dh.normalise_features(x, method="minmax")
        npt.assert_allclose(normed.min(axis=0), [0.0, 0.0], atol=1e-10)
        npt.assert_allclose(normed.max(axis=0), [1.0, 1.0], atol=1e-10)
        assert "min" in stats
        assert "max" in stats

    def test_zscore_constant_column(self):
        dh = DataHarmonizer()
        x = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        normed, stats = dh.normalise_features(x, method="zscore")
        # Constant column std is replaced with 1.0 to avoid div-by-zero
        assert np.isfinite(normed).all()

    def test_unknown_normalisation_raises(self):
        dh = DataHarmonizer()
        x = np.ones((3, 2))
        with pytest.raises(ValueError, match="Unknown normalisation"):
            dh.normalise_features(x, method="weird_method")


class TestHarmonizeBatch:
    """Tests for batch harmonization."""

    def test_harmonize_batch_basic(self):
        dh = DataHarmonizer()
        records = [
            {"tumor_volume": 25.0, "pdl1_expression": 60.0},
            {"tumor_volume": 30.0, "pdl1_expression": 80.0},
        ]
        harmonized, report = dh.harmonize_batch(records)
        assert len(harmonized) == 2
        assert report.records_processed == 2

    def test_harmonize_batch_maps_fields(self):
        dh = DataHarmonizer()
        records = [{"118565006": 15.0}]
        harmonized, report = dh.harmonize_batch(records)
        assert "tumor_volume" in harmonized[0]
        assert report.fields_mapped >= 1

    def test_harmonize_batch_empty_list(self):
        dh = DataHarmonizer()
        harmonized, report = dh.harmonize_batch([])
        assert len(harmonized) == 0
        assert report.records_processed == 0

    def test_harmonize_report_fields(self):
        report = HarmonizationReport()
        assert report.records_processed == 0
        assert report.fields_mapped == 0
        assert report.units_converted == 0
        assert report.normalisation_applied is False
        assert report.warnings == []
