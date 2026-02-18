"""Tests for q1-2026-standards/objective-1-model-conversion/conversion_pipeline.py.

Covers enums (ModelFormat, ConversionStatus), dataclasses (ConversionConfig,
ConversionResult), exceptions, helper functions, and ConversionPipeline.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "conversion_pipeline",
    "q1-2026-standards/objective-1-model-conversion/conversion_pipeline.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestModelFormat:
    """Tests for the ModelFormat enum."""

    def test_members(self):
        """ModelFormat has five members."""
        expected = {"ONNX", "PYTORCH", "TENSORFLOW", "SAFETENSORS", "MONAI"}
        assert set(mod.ModelFormat.__members__.keys()) == expected

    def test_values(self):
        """ModelFormat values are lowercase."""
        assert mod.ModelFormat.ONNX.value == "onnx"
        assert mod.ModelFormat.PYTORCH.value == "pytorch"
        assert mod.ModelFormat.SAFETENSORS.value == "safetensors"


class TestConversionStatus:
    """Tests for the ConversionStatus enum."""

    def test_members(self):
        """ConversionStatus has five members."""
        expected = {"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "VERIFIED"}
        assert set(mod.ConversionStatus.__members__.keys()) == expected

    def test_transition_order(self):
        """Typical conversion goes PENDING -> IN_PROGRESS -> COMPLETED -> VERIFIED."""
        assert mod.ConversionStatus.PENDING.value == "pending"
        assert mod.ConversionStatus.VERIFIED.value == "verified"


# ---------------------------------------------------------------------------
# ConversionConfig dataclass
# ---------------------------------------------------------------------------
class TestConversionConfig:
    """Tests for the ConversionConfig dataclass."""

    def test_defaults(self):
        """Default config is PYTORCH -> ONNX."""
        cfg = mod.ConversionConfig()
        assert cfg.source_format == mod.ModelFormat.PYTORCH
        assert cfg.target_format == mod.ModelFormat.ONNX
        assert cfg.opset_version == 17
        assert cfg.device == "cpu"

    def test_same_format_raises(self):
        """Same source and target raises ValueError."""
        with pytest.raises(ValueError, match="must differ"):
            mod.ConversionConfig(
                source_format=mod.ModelFormat.PYTORCH,
                target_format=mod.ModelFormat.PYTORCH,
            )

    def test_negative_atol_raises(self):
        """Negative atol raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            mod.ConversionConfig(
                source_format=mod.ModelFormat.PYTORCH,
                target_format=mod.ModelFormat.ONNX,
                atol=-1e-6,
            )

    def test_negative_rtol_raises(self):
        """Negative rtol raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            mod.ConversionConfig(
                source_format=mod.ModelFormat.PYTORCH,
                target_format=mod.ModelFormat.ONNX,
                rtol=-1e-5,
            )

    def test_custom_values(self):
        """ConversionConfig accepts custom values."""
        cfg = mod.ConversionConfig(
            source_format=mod.ModelFormat.ONNX,
            target_format=mod.ModelFormat.PYTORCH,
            model_name="tumor_model",
            input_shape=[1, 30],
            opset_version=13,
        )
        assert cfg.model_name == "tumor_model"
        assert cfg.input_shape == [1, 30]
        assert cfg.opset_version == 13


# ---------------------------------------------------------------------------
# ConversionResult dataclass
# ---------------------------------------------------------------------------
class TestConversionResult:
    """Tests for the ConversionResult dataclass."""

    def test_defaults(self):
        """Default result is PENDING."""
        result = mod.ConversionResult()
        assert result.status == mod.ConversionStatus.PENDING
        assert result.numerical_equivalence is False
        assert result.conversion_id != ""

    def test_to_dict(self):
        """to_dict returns JSON-serializable dictionary."""
        result = mod.ConversionResult(
            status=mod.ConversionStatus.VERIFIED,
            numerical_equivalence=True,
            max_absolute_diff=1e-7,
        )
        d = result.to_dict()
        assert d["status"] == "verified"
        assert d["numerical_equivalence"] is True
        assert isinstance(d["max_absolute_diff"], float)

    def test_unique_id(self):
        """Each ConversionResult gets a unique ID."""
        r1 = mod.ConversionResult()
        r2 = mod.ConversionResult()
        assert r1.conversion_id != r2.conversion_id


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TestExceptions:
    """Tests for custom exceptions."""

    def test_conversion_error(self):
        """ConversionError is an Exception."""
        assert issubclass(mod.ConversionError, Exception)

    def test_verification_error(self):
        """VerificationError is an Exception."""
        assert issubclass(mod.VerificationError, Exception)

    def test_unsupported_conversion_error(self):
        """UnsupportedConversionError is a ConversionError."""
        assert issubclass(mod.UnsupportedConversionError, mod.ConversionError)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestComputeSha256:
    """Tests for the compute_sha256 helper."""

    def test_known_hash(self):
        """SHA-256 of known content is reproducible."""
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
            f.write(b"hello world")
            path = f.name
        try:
            h = mod.compute_sha256(path)
            assert len(h) == 64
            assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        finally:
            os.unlink(path)


class TestGenerateDummyInput:
    """Tests for the _generate_dummy_input helper."""

    def test_shape(self):
        """Output shape matches input_shape."""
        arr = mod._generate_dummy_input([2, 5])
        assert arr.shape == (2, 5)

    def test_deterministic(self):
        """Same seed produces same output."""
        a = mod._generate_dummy_input([1, 10])
        b = mod._generate_dummy_input([1, 10])
        np.testing.assert_array_equal(a, b)

    def test_dtype(self):
        """Output is float32."""
        arr = mod._generate_dummy_input([1, 3])
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# ConversionPipeline
# ---------------------------------------------------------------------------
class TestConversionPipeline:
    """Tests for the ConversionPipeline class."""

    def test_creation(self):
        """ConversionPipeline can be instantiated."""
        pipeline = mod.ConversionPipeline()
        assert pipeline is not None

    def test_get_supported_conversions(self):
        """Supported conversions returns 11 paths."""
        pipeline = mod.ConversionPipeline()
        conversions = pipeline.get_supported_conversions()
        assert isinstance(conversions, list)
        assert len(conversions) == 11

    def test_supported_conversions_pytorch_onnx(self):
        """PyTorch -> ONNX is a supported path."""
        pipeline = mod.ConversionPipeline()
        conversions = pipeline.get_supported_conversions()
        pairs = [(c["source"], c["target"]) for c in conversions]
        assert ("pytorch", "onnx") in pairs

    def test_conversion_history_empty(self):
        """Conversion history starts empty."""
        pipeline = mod.ConversionPipeline()
        assert pipeline.get_conversion_history() == []

    def test_validate_unsupported_path(self):
        """Validate raises UnsupportedConversionError for invalid path."""
        pipeline = mod.ConversionPipeline()
        cfg = mod.ConversionConfig(
            source_format=mod.ModelFormat.SAFETENSORS,
            target_format=mod.ModelFormat.MONAI,
        )
        with pytest.raises(mod.UnsupportedConversionError):
            pipeline.validate(cfg)

    def test_validate_missing_source_file(self):
        """Validate raises ConversionError for missing source or framework."""
        pipeline = mod.ConversionPipeline()
        cfg = mod.ConversionConfig(
            source_format=mod.ModelFormat.PYTORCH,
            target_format=mod.ModelFormat.ONNX,
            source_path="/nonexistent/model.pt",
        )
        with pytest.raises(mod.ConversionError):
            pipeline.validate(cfg)

    def test_run_missing_source_returns_failed(self):
        """Run with missing source returns FAILED status."""
        pipeline = mod.ConversionPipeline()
        cfg = mod.ConversionConfig(
            source_format=mod.ModelFormat.PYTORCH,
            target_format=mod.ModelFormat.ONNX,
            source_path="/nonexistent/model.pt",
        )
        result = pipeline.run(cfg)
        assert result.status == mod.ConversionStatus.FAILED
        assert result.error_message != ""
        assert len(pipeline.get_conversion_history()) == 1

    def test_export_history(self):
        """export_history writes JSON file."""
        pipeline = mod.ConversionPipeline()
        # Run a failing conversion to populate history
        cfg = mod.ConversionConfig(
            source_format=mod.ModelFormat.PYTORCH,
            target_format=mod.ModelFormat.ONNX,
            source_path="/nonexistent/model.pt",
        )
        pipeline.run(cfg)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pipeline.export_history(path)
            with open(path) as fh:
                data = json.load(fh)
            assert isinstance(data, list)
            assert len(data) == 1
        finally:
            os.unlink(path)
