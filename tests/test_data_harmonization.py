"""Tests for the data harmonization module."""

import numpy as np

from federated.data_harmonization import ONCOLOGY_VOCABULARY, DataHarmonizer


class TestDataHarmonizer:
    def test_map_field_name_canonical(self):
        harmonizer = DataHarmonizer()
        assert harmonizer.map_field_name("tumor_volume") == "tumor_volume"

    def test_map_field_name_fhir_code(self):
        harmonizer = DataHarmonizer()
        assert harmonizer.map_field_name("observation-tumor-volume") == "tumor_volume"

    def test_map_field_name_dicom_code(self):
        harmonizer = DataHarmonizer()
        assert harmonizer.map_field_name("118565006") == "tumor_volume"

    def test_map_field_name_unknown(self):
        harmonizer = DataHarmonizer()
        assert harmonizer.map_field_name("unknown_field") == "unknown_field"

    def test_map_record(self):
        harmonizer = DataHarmonizer()
        record = {"observation-tumor-volume": 3.5, "unknown": "value"}
        mapped = harmonizer.map_record(record)
        assert "tumor_volume" in mapped
        assert "unknown" in mapped

    def test_convert_units_mm_to_cm(self):
        harmonizer = DataHarmonizer()
        assert abs(harmonizer.convert_units(10.0, "mm", "cm") - 1.0) < 0.01

    def test_convert_units_same(self):
        harmonizer = DataHarmonizer()
        assert harmonizer.convert_units(5.0, "cm", "cm") == 5.0

    def test_convert_units_fahrenheit_celsius(self):
        harmonizer = DataHarmonizer()
        assert abs(harmonizer.convert_units(212.0, "F", "C") - 100.0) < 0.01

    def test_normalise_features_zscore(self):
        harmonizer = DataHarmonizer()
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalised, stats = harmonizer.normalise_features(x, method="zscore")
        assert normalised.shape == x.shape
        assert "mean" in stats
        assert "std" in stats
        # Mean of normalised columns should be ~0
        assert np.allclose(np.mean(normalised, axis=0), 0, atol=1e-10)

    def test_normalise_features_minmax(self):
        harmonizer = DataHarmonizer()
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalised, stats = harmonizer.normalise_features(x, method="minmax")
        assert normalised.min() >= 0.0
        assert normalised.max() <= 1.0

    def test_harmonize_batch(self):
        harmonizer = DataHarmonizer()
        records = [
            {"observation-tumor-volume": 3.5, "age": 60},
            {"observation-ki67": 0.25, "age": 55},
        ]
        harmonized, report = harmonizer.harmonize_batch(records)
        assert report.records_processed == 2
        assert report.fields_mapped >= 1
        assert len(harmonized) == 2

    def test_vocabulary_coverage(self):
        assert len(ONCOLOGY_VOCABULARY) >= 10
