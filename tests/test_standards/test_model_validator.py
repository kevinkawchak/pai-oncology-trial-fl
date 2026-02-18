"""Tests for q1-2026-standards/objective-2-model-registry/model_validator.py.

Covers enums (ValidationStatus, ValidationSeverity, ModelStage),
dataclasses (ModelMetadata, ValidationCheckResult, ValidationReport,
PerformanceThresholds), helper functions, and ModelValidator.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "model_validator",
    "q1-2026-standards/objective-2-model-registry/model_validator.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestValidationStatus:
    """Tests for the ValidationStatus enum."""

    def test_members(self):
        """ValidationStatus has four members."""
        expected = {"PENDING", "PASSED", "FAILED", "REQUIRES_REVIEW"}
        assert set(mod.ValidationStatus.__members__.keys()) == expected

    def test_values(self):
        """Values are uppercase."""
        assert mod.ValidationStatus.PASSED.value == "PASSED"
        assert mod.ValidationStatus.FAILED.value == "FAILED"


class TestValidationSeverity:
    """Tests for the ValidationSeverity enum."""

    def test_members(self):
        """ValidationSeverity has four members."""
        expected = {"INFO", "WARNING", "ERROR", "CRITICAL"}
        assert set(mod.ValidationSeverity.__members__.keys()) == expected

    def test_values(self):
        """Values are lowercase."""
        assert mod.ValidationSeverity.INFO.value == "info"
        assert mod.ValidationSeverity.CRITICAL.value == "critical"


class TestModelStage:
    """Tests for the ModelStage enum."""

    def test_members(self):
        """ModelStage has four lifecycle stages."""
        expected = {"DEVELOPMENT", "STAGING", "PRODUCTION", "ARCHIVED"}
        assert set(mod.ModelStage.__members__.keys()) == expected

    def test_values(self):
        """Values are lowercase."""
        assert mod.ModelStage.DEVELOPMENT.value == "development"
        assert mod.ModelStage.PRODUCTION.value == "production"


# ---------------------------------------------------------------------------
# ModelMetadata dataclass
# ---------------------------------------------------------------------------
class TestModelMetadata:
    """Tests for the ModelMetadata dataclass."""

    def test_defaults(self):
        """Default metadata has empty strings and DEVELOPMENT stage."""
        mm = mod.ModelMetadata()
        assert mm.model_id == ""
        assert mm.stage == mod.ModelStage.DEVELOPMENT
        assert mm.metrics == {}

    def test_to_dict(self):
        """to_dict includes all fields."""
        mm = mod.ModelMetadata(
            model_id="MCC-tumor-001",
            model_name="Tumor Classifier",
            version="1.0.0",
        )
        d = mm.to_dict()
        assert d["model_id"] == "MCC-tumor-001"
        assert d["model_name"] == "Tumor Classifier"
        assert d["stage"] == "development"

    def test_custom_metrics(self):
        """Metrics are preserved."""
        mm = mod.ModelMetadata(metrics={"accuracy": 0.92, "auc_roc": 0.95})
        assert mm.metrics["accuracy"] == 0.92


# ---------------------------------------------------------------------------
# ValidationCheckResult dataclass
# ---------------------------------------------------------------------------
class TestValidationCheckResult:
    """Tests for the ValidationCheckResult dataclass."""

    def test_defaults(self):
        """Default check result is not passed."""
        cr = mod.ValidationCheckResult()
        assert cr.passed is False
        assert cr.severity == mod.ValidationSeverity.INFO

    def test_custom(self):
        """Custom check result with all fields."""
        cr = mod.ValidationCheckResult(
            check_name="integrity.sha256",
            passed=True,
            severity=mod.ValidationSeverity.INFO,
            message="Hash matches",
        )
        assert cr.check_name == "integrity.sha256"
        assert cr.passed is True


# ---------------------------------------------------------------------------
# ValidationReport dataclass
# ---------------------------------------------------------------------------
class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_defaults(self):
        """Default report is PENDING."""
        report = mod.ValidationReport()
        assert report.status == mod.ValidationStatus.PENDING
        assert report.report_id != ""

    def test_to_dict(self):
        """to_dict serializes all fields."""
        report = mod.ValidationReport(
            model_id="MCC-test-001",
            status=mod.ValidationStatus.PASSED,
            checks_passed=5,
        )
        d = report.to_dict()
        assert d["model_id"] == "MCC-test-001"
        assert d["status"] == "PASSED"
        assert d["checks_passed"] == 5

    def test_unique_ids(self):
        """Each report gets a unique ID."""
        r1 = mod.ValidationReport()
        r2 = mod.ValidationReport()
        assert r1.report_id != r2.report_id


# ---------------------------------------------------------------------------
# PerformanceThresholds dataclass
# ---------------------------------------------------------------------------
class TestPerformanceThresholds:
    """Tests for the PerformanceThresholds dataclass."""

    def test_defaults(self):
        """Default thresholds are reasonable for oncology models."""
        pt = mod.PerformanceThresholds()
        assert pt.min_accuracy == 0.80
        assert pt.min_auc_roc == 0.75
        assert pt.max_loss == 1.0
        assert pt.max_inference_latency_ms == 100.0

    def test_custom_thresholds(self):
        """Custom thresholds are preserved."""
        pt = mod.PerformanceThresholds(min_accuracy=0.95, min_f1_score=0.90)
        assert pt.min_accuracy == 0.95
        assert pt.min_f1_score == 0.90


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestIsValidSemver:
    """Tests for the _is_valid_semver helper."""

    def test_valid(self):
        """Standard semver passes."""
        assert mod._is_valid_semver("1.0.0") is True
        assert mod._is_valid_semver("0.1.2") is True
        assert mod._is_valid_semver("10.20.30") is True

    def test_invalid(self):
        """Non-semver strings fail."""
        assert mod._is_valid_semver("1.0") is False
        assert mod._is_valid_semver("v1.0.0") is False
        assert mod._is_valid_semver("1.0.0-beta") is False
        assert mod._is_valid_semver("abc") is False


class TestIsValidModelId:
    """Tests for the _is_valid_model_id helper."""

    def test_valid(self):
        """Valid model IDs match the pattern."""
        assert mod._is_valid_model_id("MCC-tumor-response-001") is True
        assert mod._is_valid_model_id("FL-site-a-12345") is True

    def test_invalid(self):
        """Invalid model IDs fail."""
        assert mod._is_valid_model_id("mcc-tumor-001") is False  # lowercase prefix
        assert mod._is_valid_model_id("MCC_tumor_001") is False  # underscores
        assert mod._is_valid_model_id("") is False


class TestComputeSha256:
    """Tests for the _compute_sha256 helper."""

    def test_known_hash(self):
        """SHA-256 of known content is correct."""
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
            f.write(b"test content")
            path = f.name
        try:
            h = mod._compute_sha256(path)
            assert len(h) == 64
            assert h == "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ModelValidator
# ---------------------------------------------------------------------------
class TestModelValidator:
    """Tests for the ModelValidator class."""

    def test_creation(self):
        """ModelValidator can be created with defaults."""
        v = mod.ModelValidator()
        assert v.thresholds.min_accuracy == 0.80

    def test_creation_custom_thresholds(self):
        """ModelValidator accepts custom thresholds."""
        pt = mod.PerformanceThresholds(min_accuracy=0.95)
        v = mod.ModelValidator(thresholds=pt)
        assert v.thresholds.min_accuracy == 0.95

    def test_validate_minimal_dev_model_passes(self):
        """Minimal development model with required fields passes."""
        metadata = mod.ModelMetadata(
            model_id="MCC-test-model-001",
            model_name="Test Model",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
            stage=mod.ModelStage.DEVELOPMENT,
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        # model_id format check should pass
        format_checks = [c for c in report.checks if c.check_name == "format.model_id"]
        assert len(format_checks) == 1
        assert format_checks[0].passed is True

    def test_validate_invalid_model_id_format(self):
        """Invalid model ID format produces an error."""
        metadata = mod.ModelMetadata(
            model_id="bad_id",
            model_name="Test",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        format_checks = [c for c in report.checks if c.check_name == "format.model_id"]
        assert len(format_checks) == 1
        assert format_checks[0].passed is False

    def test_validate_invalid_version_format(self):
        """Invalid version format produces an error."""
        metadata = mod.ModelMetadata(
            model_id="MCC-test-001",
            model_name="Test",
            framework="pytorch",
            version="v1.0",
            artifact_path="/nonexistent/path",
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        version_checks = [c for c in report.checks if c.check_name == "format.version"]
        assert len(version_checks) == 1
        assert version_checks[0].passed is False

    def test_validate_performance_thresholds_pass(self):
        """Good metrics pass performance thresholds."""
        metadata = mod.ModelMetadata(
            model_id="MCC-perf-001",
            model_name="Perf Model",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
            metrics={"accuracy": 0.92, "auc_roc": 0.95, "f1_score": 0.88},
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True)
        perf_checks = [c for c in report.checks if c.check_name.startswith("performance.")]
        assert all(c.passed for c in perf_checks)

    def test_validate_performance_thresholds_fail(self):
        """Poor metrics fail performance thresholds."""
        metadata = mod.ModelMetadata(
            model_id="MCC-fail-001",
            model_name="Bad Model",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
            metrics={"accuracy": 0.50, "auc_roc": 0.55},
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True)
        perf_checks = [c for c in report.checks if c.check_name.startswith("performance.")]
        assert any(not c.passed for c in perf_checks)

    def test_validate_compliance_missing_intended_use(self):
        """Missing intended use produces compliance warning."""
        metadata = mod.ModelMetadata(
            model_id="MCC-comp-001",
            model_name="Compliance Test",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        compliance_checks = [c for c in report.checks if c.check_name == "compliance.intended_use"]
        assert len(compliance_checks) == 1
        assert compliance_checks[0].passed is False

    def test_validate_for_promotion_production_strict(self):
        """Production promotion with warnings fails."""
        metadata = mod.ModelMetadata(
            model_id="MCC-prod-001",
            model_name="Prod Model",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
            stage=mod.ModelStage.PRODUCTION,
        )
        v = mod.ModelValidator()
        report = v.validate_for_promotion(metadata, target_stage=mod.ModelStage.PRODUCTION)
        # Missing many production fields, so should fail
        assert report.status == mod.ValidationStatus.FAILED

    def test_validation_history(self):
        """Validation history grows with each call."""
        v = mod.ModelValidator()
        metadata = mod.ModelMetadata(
            model_id="MCC-hist-001",
            model_name="History Test",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
        )
        v.validate(metadata, skip_integrity=True, skip_performance=True)
        v.validate(metadata, skip_integrity=True, skip_performance=True)
        history = v.get_validation_history()
        assert len(history) == 2

    def test_export_report(self):
        """export_report writes JSON to disk."""
        v = mod.ModelValidator()
        metadata = mod.ModelMetadata(
            model_id="MCC-export-001",
            model_name="Export Test",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
        )
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            v.export_report(report, path)
            with open(path) as fh:
                data = json.load(fh)
            assert data["model_id"] == "MCC-export-001"
            assert "checks" in data
        finally:
            os.unlink(path)

    def test_known_framework_passes(self):
        """Known framework 'pytorch' passes format check."""
        metadata = mod.ModelMetadata(
            model_id="MCC-fw-001",
            model_name="FW Test",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        fw_checks = [c for c in report.checks if c.check_name == "format.framework"]
        assert len(fw_checks) == 1
        assert fw_checks[0].passed is True

    def test_unknown_framework_warning(self):
        """Unknown framework triggers a warning."""
        metadata = mod.ModelMetadata(
            model_id="MCC-uf-001",
            model_name="UF Test",
            framework="exotic_framework",
            version="1.0.0",
            artifact_path="/nonexistent/path",
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        fw_checks = [c for c in report.checks if c.check_name == "format.framework"]
        assert len(fw_checks) == 1
        assert fw_checks[0].passed is False
        assert fw_checks[0].severity == mod.ValidationSeverity.WARNING

    def test_sha256_format_valid(self):
        """Valid 64-char hex SHA-256 passes format check."""
        metadata = mod.ModelMetadata(
            model_id="MCC-sha-001",
            model_name="SHA Test",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
            artifact_sha256="a" * 64,
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        sha_checks = [c for c in report.checks if c.check_name == "format.artifact_sha256"]
        assert len(sha_checks) == 1
        assert sha_checks[0].passed is True

    def test_sha256_format_invalid(self):
        """Invalid SHA-256 fails format check."""
        metadata = mod.ModelMetadata(
            model_id="MCC-sha2-001",
            model_name="SHA Test 2",
            framework="pytorch",
            version="1.0.0",
            artifact_path="/nonexistent/path",
            artifact_sha256="tooshort",
        )
        v = mod.ModelValidator()
        report = v.validate(metadata, skip_integrity=True, skip_performance=True)
        sha_checks = [c for c in report.checks if c.check_name == "format.artifact_sha256"]
        assert len(sha_checks) == 1
        assert sha_checks[0].passed is False
