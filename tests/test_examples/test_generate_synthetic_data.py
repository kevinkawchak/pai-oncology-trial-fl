"""Tests for examples/generate_synthetic_data.py.

Validates data generation constants, feature names, class names,
and argument parsing definitions.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

# Module requires pandas -- load lazily via fixture to avoid module-level skip
pytestmark = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("pandas"),
    reason="pandas not installed",
)


@pytest.fixture(scope="module")
def mod():
    return load_module(
        "gen_synth_data",
        "examples/generate_synthetic_data.py",
    )


class TestFeatureNames:
    """Verify FEATURE_NAMES list."""

    def test_feature_names_length(self, mod):
        assert len(mod.FEATURE_NAMES) == 30

    def test_feature_names_contains_tumor_volume(self, mod):
        assert "tumor_volume_cm3" in mod.FEATURE_NAMES

    def test_feature_names_contains_growth_rate(self, mod):
        assert "growth_rate_days" in mod.FEATURE_NAMES

    def test_feature_names_contains_sensitivity(self, mod):
        assert "chemo_sensitivity" in mod.FEATURE_NAMES
        assert "radio_sensitivity" in mod.FEATURE_NAMES

    def test_feature_names_contains_biomarkers(self, mod):
        assert "pdl1_expression" in mod.FEATURE_NAMES
        assert "ki67_index" in mod.FEATURE_NAMES
        assert "her2_status" in mod.FEATURE_NAMES

    def test_feature_names_contains_clinical(self, mod):
        assert "ecog_score" in mod.FEATURE_NAMES
        assert "stage_numeric" in mod.FEATURE_NAMES
        assert "bmi" in mod.FEATURE_NAMES

    def test_feature_names_all_unique(self, mod):
        assert len(mod.FEATURE_NAMES) == len(set(mod.FEATURE_NAMES))


class TestClassNames:
    """Verify CLASS_NAMES list."""

    def test_class_names_length(self, mod):
        assert len(mod.CLASS_NAMES) == 2

    def test_class_names_content(self, mod):
        assert "favorable_response" in mod.CLASS_NAMES
        assert "poor_response" in mod.CLASS_NAMES


class TestModuleAttributes:
    """Verify the module exposes expected top-level attributes."""

    def test_has_main_function(self, mod):
        assert hasattr(mod, "main")
        assert callable(mod.main)

    def test_has_parse_args_function(self, mod):
        assert hasattr(mod, "parse_args")
        assert callable(mod.parse_args)
