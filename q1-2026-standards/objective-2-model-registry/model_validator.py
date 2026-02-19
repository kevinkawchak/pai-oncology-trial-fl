"""Model validation engine for the federated model registry.

Validates model artifacts before they are registered or promoted in the
federated model registry. Checks include metadata completeness, SHA-256
integrity verification, performance threshold gates, numerical stability,
and regulatory compliance flags.

The validator is designed to run at each federated site as part of the
model ingestion workflow. Models that fail validation are rejected with
detailed diagnostic information.

DISCLAIMER: RESEARCH USE ONLY. This software is not approved for clinical
decision-making. Validation results do not constitute regulatory clearance
or approval for any clinical use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports -- frameworks may or may not be installed at each site
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from safetensors import safe_open

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import monai

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ValidationStatus(Enum):
    """Status of a model validation run."""

    PENDING = "PENDING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class ValidationSeverity(Enum):
    """Severity level for individual validation check results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModelStage(Enum):
    """Model promotion stages in the registry lifecycle."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelMetadata:
    """Metadata for a model artifact in the federated registry.

    Attributes:
        model_id: Globally unique model identifier.
        model_name: Human-readable model name.
        framework: Training framework (pytorch, tensorflow, onnx, etc.).
        model_format: Serialization format (pt, onnx, savedmodel, etc.).
        version: Semantic version string (MAJOR.MINOR.PATCH).
        architecture: Model architecture name.
        input_shape: Expected input tensor shape (excluding batch dim).
        output_shape: Expected output tensor shape (excluding batch dim).
        num_parameters: Total number of trainable parameters.
        clinical_domain: Primary oncology domain.
        cancer_types: List of cancer types the model is trained on.
        intended_use: Statement of intended use.
        artifact_path: Path to the model artifact file.
        artifact_sha256: SHA-256 hash of the artifact.
        artifact_size_bytes: File size in bytes.
        source_site: Identifier of the training site.
        created_by: User or system that created the artifact.
        created_at: ISO 8601 timestamp of creation.
        metrics: Performance metrics from evaluation.
        training_config: Training hyperparameters and data summary.
        stage: Current lifecycle stage.
        tags: Arbitrary key-value tags.
    """

    model_id: str = ""
    model_name: str = ""
    framework: str = ""
    model_format: str = ""
    version: str = ""
    architecture: str = ""
    input_shape: list[int] = field(default_factory=list)
    output_shape: list[int] = field(default_factory=list)
    num_parameters: int = 0
    clinical_domain: str = ""
    cancer_types: list[str] = field(default_factory=list)
    intended_use: str = ""
    artifact_path: str = ""
    artifact_sha256: str = ""
    artifact_size_bytes: int = 0
    source_site: str = ""
    created_by: str = ""
    created_at: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    stage: ModelStage = ModelStage.DEVELOPMENT
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "framework": self.framework,
            "model_format": self.model_format,
            "version": self.version,
            "architecture": self.architecture,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "num_parameters": self.num_parameters,
            "clinical_domain": self.clinical_domain,
            "cancer_types": self.cancer_types,
            "intended_use": self.intended_use,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "source_site": self.source_site,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "metrics": dict(self.metrics),
            "training_config": dict(self.training_config),
            "stage": self.stage.value,
            "tags": dict(self.tags),
        }


@dataclass
class ValidationCheckResult:
    """Result of a single validation check.

    Attributes:
        check_name: Identifier for the check.
        passed: Whether the check passed.
        severity: Severity level of a failure.
        message: Human-readable result message.
        details: Additional structured details.
    """

    check_name: str = ""
    passed: bool = False
    severity: ValidationSeverity = ValidationSeverity.INFO
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated report from a full validation run.

    Attributes:
        report_id: Unique identifier for this validation run.
        model_id: The model that was validated.
        model_version: Version of the model that was validated.
        status: Overall validation status.
        checks: List of individual check results.
        checks_passed: Number of checks that passed.
        checks_failed: Number of checks that failed.
        checks_warning: Number of checks with warnings.
        validation_time_seconds: Time taken for the full validation.
        timestamp: ISO 8601 timestamp of validation completion.
        validator_version: Version of the validator that produced this report.
        notes: Free-text notes.
    """

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    model_version: str = ""
    status: ValidationStatus = ValidationStatus.PENDING
    checks: list[ValidationCheckResult] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warning: int = 0
    validation_time_seconds: float = 0.0
    timestamp: str = ""
    validator_version: str = "1.0.0"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a dictionary for JSON export."""
        return {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "status": self.status.value,
            "checks": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity.value,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warning": self.checks_warning,
            "validation_time_seconds": self.validation_time_seconds,
            "timestamp": self.timestamp,
            "validator_version": self.validator_version,
            "notes": self.notes,
        }


@dataclass
class PerformanceThresholds:
    """Minimum performance thresholds for model promotion.

    Models must meet these thresholds to pass the performance gate.
    Thresholds can be configured per clinical domain.

    Attributes:
        min_accuracy: Minimum classification accuracy.
        min_auc_roc: Minimum AUC-ROC score.
        max_loss: Maximum acceptable loss value.
        min_sensitivity: Minimum sensitivity (true positive rate).
        min_specificity: Minimum specificity (true negative rate).
        min_f1_score: Minimum F1 score.
        max_inference_latency_ms: Maximum inference latency in milliseconds.
    """

    min_accuracy: float = 0.80
    min_auc_roc: float = 0.75
    max_loss: float = 1.0
    min_sensitivity: float = 0.70
    min_specificity: float = 0.70
    min_f1_score: float = 0.70
    max_inference_latency_ms: float = 100.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ValidationError(Exception):
    """Raised when a validation check encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _iso_timestamp() -> str:
    """Return current UTC time as ISO 8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_valid_semver(version: str) -> bool:
    """Check if a string is a valid semantic version (MAJOR.MINOR.PATCH)."""
    parts = version.split(".")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)


def _is_valid_model_id(model_id: str) -> bool:
    """Check if a model ID matches the expected pattern."""
    import re

    pattern = r"^[A-Z]{2,6}-[a-z0-9-]+-[0-9]{3,6}$"
    return bool(re.match(pattern, model_id))


# ---------------------------------------------------------------------------
# Model Validator
# ---------------------------------------------------------------------------
class ModelValidator:
    """Validates model artifacts for the federated model registry.

    Runs a configurable set of validation checks against a model artifact
    and its metadata. Checks are organized into categories:

        - **Metadata checks**: Completeness and format of registry fields.
        - **Integrity checks**: SHA-256 hash verification, file existence.
        - **Performance checks**: Metrics against configurable thresholds.
        - **Stability checks**: NaN/Inf detection in model weights.
        - **Compliance checks**: Regulatory and privacy flag verification.

    Example::

        metadata = ModelMetadata(
            model_id="MCC-tumor-response-001",
            model_name="Tumor Response Predictor",
            framework="pytorch",
            model_format="pt",
            version="1.0.0",
            artifact_path="models/model.pt",
            metrics={"accuracy": 0.89, "auc_roc": 0.91},
        )
        validator = ModelValidator()
        report = validator.validate(metadata)
        print(report.status.value)  # "PASSED" or "FAILED"
    """

    # Required metadata fields for each promotion stage
    _REQUIRED_FIELDS_BY_STAGE: dict[ModelStage, list[str]] = {
        ModelStage.DEVELOPMENT: [
            "model_id",
            "model_name",
            "framework",
            "version",
            "artifact_path",
        ],
        ModelStage.STAGING: [
            "model_id",
            "model_name",
            "framework",
            "model_format",
            "version",
            "architecture",
            "input_shape",
            "output_shape",
            "num_parameters",
            "artifact_path",
            "artifact_sha256",
            "source_site",
            "created_by",
            "clinical_domain",
            "intended_use",
        ],
        ModelStage.PRODUCTION: [
            "model_id",
            "model_name",
            "framework",
            "model_format",
            "version",
            "architecture",
            "input_shape",
            "output_shape",
            "num_parameters",
            "clinical_domain",
            "cancer_types",
            "intended_use",
            "artifact_path",
            "artifact_sha256",
            "artifact_size_bytes",
            "source_site",
            "created_by",
            "created_at",
        ],
        ModelStage.ARCHIVED: [
            "model_id",
            "version",
        ],
    }

    def __init__(
        self,
        thresholds: PerformanceThresholds | None = None,
        schema_path: str | None = None,
    ) -> None:
        """Initialize the model validator.

        Args:
            thresholds: Performance thresholds for metric gates.
                Uses defaults if not provided.
            schema_path: Path to the YAML registry schema for schema
                validation. If not provided, schema validation is skipped.
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self.schema_path = schema_path
        self._validation_history: list[ValidationReport] = []

        logger.info(
            "ModelValidator initialized (min_accuracy=%.2f, min_auc=%.2f)",
            self.thresholds.min_accuracy,
            self.thresholds.min_auc_roc,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        metadata: ModelMetadata,
        target_stage: ModelStage | None = None,
        skip_integrity: bool = False,
        skip_performance: bool = False,
    ) -> ValidationReport:
        """Run the full validation suite on a model artifact.

        Args:
            metadata: Model metadata to validate.
            target_stage: Target promotion stage. Determines which fields
                are required. Defaults to the model's current stage.
            skip_integrity: If True, skip SHA-256 integrity verification.
            skip_performance: If True, skip performance threshold checks.

        Returns:
            ValidationReport with aggregated results.
        """
        start_time = time.monotonic()
        stage = target_stage or metadata.stage

        report = ValidationReport(
            model_id=metadata.model_id,
            model_version=metadata.version,
            timestamp=_iso_timestamp(),
        )

        # Run all check categories
        report.checks.extend(self._check_metadata_completeness(metadata, stage))
        report.checks.extend(self._check_metadata_format(metadata))

        if not skip_integrity:
            report.checks.extend(self._check_artifact_integrity(metadata))

        if not skip_performance and metadata.metrics:
            report.checks.extend(self._check_performance_thresholds(metadata))

        report.checks.extend(self._check_numerical_stability(metadata))
        report.checks.extend(self._check_compliance_flags(metadata))

        # Aggregate results
        report.checks_passed = sum(1 for c in report.checks if c.passed)
        report.checks_failed = sum(
            1
            for c in report.checks
            if not c.passed and c.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )
        report.checks_warning = sum(
            1 for c in report.checks if not c.passed and c.severity == ValidationSeverity.WARNING
        )
        report.validation_time_seconds = round(time.monotonic() - start_time, 3)

        # Determine overall status
        has_critical = any(not c.passed and c.severity == ValidationSeverity.CRITICAL for c in report.checks)
        has_errors = report.checks_failed > 0
        has_warnings = report.checks_warning > 0

        if has_critical or has_errors:
            report.status = ValidationStatus.FAILED
        elif has_warnings:
            report.status = ValidationStatus.REQUIRES_REVIEW
        else:
            report.status = ValidationStatus.PASSED

        self._validation_history.append(report)

        logger.info(
            "Validation %s for model %s v%s: %s (%d passed, %d failed, %d warnings)",
            report.report_id,
            metadata.model_id,
            metadata.version,
            report.status.value,
            report.checks_passed,
            report.checks_failed,
            report.checks_warning,
        )

        return report

    def validate_for_promotion(
        self,
        metadata: ModelMetadata,
        target_stage: ModelStage,
    ) -> ValidationReport:
        """Validate a model for promotion to a specific lifecycle stage.

        This is a stricter validation that enforces stage-specific
        requirements. Production promotions require all checks to pass.

        Args:
            metadata: Model metadata to validate.
            target_stage: Target promotion stage.

        Returns:
            ValidationReport with promotion-specific results.
        """
        report = self.validate(metadata, target_stage=target_stage)

        if target_stage == ModelStage.PRODUCTION:
            if report.status != ValidationStatus.PASSED:
                report.notes = (
                    f"Production promotion requires all checks to pass. "
                    f"Current status: {report.status.value}. "
                    f"Resolve {report.checks_failed} error(s) and "
                    f"{report.checks_warning} warning(s) before retrying."
                )
                report.status = ValidationStatus.FAILED

        return report

    def get_validation_history(self) -> list[dict[str, Any]]:
        """Return the history of all validation runs."""
        return [report.to_dict() for report in self._validation_history]

    def export_report(self, report: ValidationReport, output_path: str) -> None:
        """Export a validation report to a JSON file.

        Args:
            report: The validation report to export.
            output_path: File path for the JSON output.
        """
        output_dir = str(Path(output_path).parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)

        logger.info("Validation report exported to %s", output_path)

    # ------------------------------------------------------------------
    # Metadata completeness checks
    # ------------------------------------------------------------------

    def _check_metadata_completeness(
        self,
        metadata: ModelMetadata,
        stage: ModelStage,
    ) -> list[ValidationCheckResult]:
        """Check that all required metadata fields are populated.

        Args:
            metadata: Model metadata to check.
            stage: Target stage determining required fields.

        Returns:
            List of check results, one per required field.
        """
        results: list[ValidationCheckResult] = []
        required_fields = self._REQUIRED_FIELDS_BY_STAGE.get(stage, [])

        for field_name in required_fields:
            value = getattr(metadata, field_name, None)

            is_populated = False
            if isinstance(value, str):
                is_populated = len(value.strip()) > 0
            elif isinstance(value, list):
                is_populated = len(value) > 0
            elif isinstance(value, (int, float)):
                is_populated = value != 0
            elif isinstance(value, dict):
                is_populated = len(value) > 0
            elif value is not None:
                is_populated = True

            results.append(
                ValidationCheckResult(
                    check_name=f"metadata.{field_name}",
                    passed=is_populated,
                    severity=ValidationSeverity.ERROR if not is_populated else ValidationSeverity.INFO,
                    message=(
                        f"Required field '{field_name}' is populated"
                        if is_populated
                        else f"Required field '{field_name}' is missing or empty for {stage.value} stage"
                    ),
                    details={"field": field_name, "stage": stage.value, "has_value": is_populated},
                )
            )

        return results

    # ------------------------------------------------------------------
    # Metadata format checks
    # ------------------------------------------------------------------

    def _check_metadata_format(self, metadata: ModelMetadata) -> list[ValidationCheckResult]:
        """Check that metadata field values conform to expected formats.

        Validates:
            - model_id matches the expected pattern
            - version is valid semver
            - framework is a recognized value
            - SHA-256 hash is a 64-character hex string

        Args:
            metadata: Model metadata to check.

        Returns:
            List of check results.
        """
        results: list[ValidationCheckResult] = []

        # Check model_id format
        if metadata.model_id:
            valid_id = _is_valid_model_id(metadata.model_id)
            results.append(
                ValidationCheckResult(
                    check_name="format.model_id",
                    passed=valid_id,
                    severity=ValidationSeverity.ERROR if not valid_id else ValidationSeverity.INFO,
                    message=(
                        f"Model ID '{metadata.model_id}' matches expected pattern"
                        if valid_id
                        else f"Model ID '{metadata.model_id}' does not match pattern ^[A-Z]{{2,6}}-[a-z0-9-]+-[0-9]{{3,6}}$"
                    ),
                )
            )

        # Check version format
        if metadata.version:
            valid_version = _is_valid_semver(metadata.version)
            results.append(
                ValidationCheckResult(
                    check_name="format.version",
                    passed=valid_version,
                    severity=ValidationSeverity.ERROR if not valid_version else ValidationSeverity.INFO,
                    message=(
                        f"Version '{metadata.version}' is valid semver"
                        if valid_version
                        else f"Version '{metadata.version}' is not valid semver (expected MAJOR.MINOR.PATCH)"
                    ),
                )
            )

        # Check framework
        known_frameworks = {"pytorch", "tensorflow", "onnx", "safetensors", "monai", "numpy", "jax"}
        if metadata.framework:
            valid_fw = metadata.framework.lower() in known_frameworks
            results.append(
                ValidationCheckResult(
                    check_name="format.framework",
                    passed=valid_fw,
                    severity=ValidationSeverity.WARNING if not valid_fw else ValidationSeverity.INFO,
                    message=(
                        f"Framework '{metadata.framework}' is recognized"
                        if valid_fw
                        else f"Framework '{metadata.framework}' is not in known list: {sorted(known_frameworks)}"
                    ),
                )
            )

        # Check SHA-256 format
        if metadata.artifact_sha256:
            valid_hash = len(metadata.artifact_sha256) == 64 and all(
                c in "0123456789abcdef" for c in metadata.artifact_sha256.lower()
            )
            results.append(
                ValidationCheckResult(
                    check_name="format.artifact_sha256",
                    passed=valid_hash,
                    severity=ValidationSeverity.ERROR if not valid_hash else ValidationSeverity.INFO,
                    message=(
                        "SHA-256 hash format is valid"
                        if valid_hash
                        else f"SHA-256 hash is not a valid 64-character hex string (length={len(metadata.artifact_sha256)})"
                    ),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Artifact integrity checks
    # ------------------------------------------------------------------

    def _check_artifact_integrity(self, metadata: ModelMetadata) -> list[ValidationCheckResult]:
        """Verify the integrity of the model artifact file.

        Checks:
            - File exists and is readable
            - File size matches metadata
            - SHA-256 hash matches metadata

        Args:
            metadata: Model metadata with artifact path and hash.

        Returns:
            List of check results.
        """
        results: list[ValidationCheckResult] = []

        # Check file existence
        artifact_path = metadata.artifact_path
        file_exists = artifact_path and Path(artifact_path).exists()

        results.append(
            ValidationCheckResult(
                check_name="integrity.file_exists",
                passed=bool(file_exists),
                severity=ValidationSeverity.CRITICAL if not file_exists else ValidationSeverity.INFO,
                message=(
                    f"Artifact file exists: {artifact_path}"
                    if file_exists
                    else f"Artifact file not found: {artifact_path}"
                ),
            )
        )

        if not file_exists:
            return results

        # Check file size
        actual_size = Path(artifact_path).stat().st_size
        if metadata.artifact_size_bytes > 0:
            size_match = actual_size == metadata.artifact_size_bytes
            results.append(
                ValidationCheckResult(
                    check_name="integrity.file_size",
                    passed=size_match,
                    severity=ValidationSeverity.ERROR if not size_match else ValidationSeverity.INFO,
                    message=(
                        f"File size matches metadata ({actual_size} bytes)"
                        if size_match
                        else f"File size mismatch: actual={actual_size}, expected={metadata.artifact_size_bytes}"
                    ),
                    details={"actual_size": actual_size, "expected_size": metadata.artifact_size_bytes},
                )
            )

        # Check SHA-256 hash
        if metadata.artifact_sha256:
            actual_hash = _compute_sha256(artifact_path)
            hash_match = actual_hash == metadata.artifact_sha256.lower()
            results.append(
                ValidationCheckResult(
                    check_name="integrity.sha256",
                    passed=hash_match,
                    severity=ValidationSeverity.CRITICAL if not hash_match else ValidationSeverity.INFO,
                    message=(
                        "SHA-256 hash matches metadata"
                        if hash_match
                        else f"SHA-256 mismatch: actual={actual_hash[:16]}..., expected={metadata.artifact_sha256[:16]}..."
                    ),
                    details={"actual_hash": actual_hash, "expected_hash": metadata.artifact_sha256},
                )
            )

        return results

    # ------------------------------------------------------------------
    # Performance threshold checks
    # ------------------------------------------------------------------

    def _check_performance_thresholds(self, metadata: ModelMetadata) -> list[ValidationCheckResult]:
        """Check model performance metrics against configured thresholds.

        Args:
            metadata: Model metadata with performance metrics.

        Returns:
            List of check results, one per metric threshold.
        """
        results: list[ValidationCheckResult] = []
        metrics = metadata.metrics

        threshold_checks = [
            ("accuracy", self.thresholds.min_accuracy, ">="),
            ("auc_roc", self.thresholds.min_auc_roc, ">="),
            ("loss", self.thresholds.max_loss, "<="),
            ("sensitivity", self.thresholds.min_sensitivity, ">="),
            ("specificity", self.thresholds.min_specificity, ">="),
            ("f1_score", self.thresholds.min_f1_score, ">="),
            ("inference_latency_ms", self.thresholds.max_inference_latency_ms, "<="),
        ]

        for metric_name, threshold, direction in threshold_checks:
            if metric_name not in metrics:
                continue

            value = metrics[metric_name]
            if direction == ">=":
                passed = value >= threshold
                msg_template = "{name}={value:.4f} >= {threshold:.4f}"
                fail_template = "{name}={value:.4f} < {threshold:.4f} (minimum required)"
            else:
                passed = value <= threshold
                msg_template = "{name}={value:.4f} <= {threshold:.4f}"
                fail_template = "{name}={value:.4f} > {threshold:.4f} (maximum allowed)"

            results.append(
                ValidationCheckResult(
                    check_name=f"performance.{metric_name}",
                    passed=passed,
                    severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
                    message=(
                        msg_template.format(name=metric_name, value=value, threshold=threshold)
                        if passed
                        else fail_template.format(name=metric_name, value=value, threshold=threshold)
                    ),
                    details={
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "direction": direction,
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Numerical stability checks
    # ------------------------------------------------------------------

    def _check_numerical_stability(self, metadata: ModelMetadata) -> list[ValidationCheckResult]:
        """Check model weights for NaN, Inf, and extreme values.

        Loads the model artifact and inspects all tensors for numerical
        issues that could indicate training instability.

        Args:
            metadata: Model metadata with artifact path and framework.

        Returns:
            List of check results.
        """
        results: list[ValidationCheckResult] = []

        artifact_path = metadata.artifact_path
        if not artifact_path or not Path(artifact_path).exists():
            results.append(
                ValidationCheckResult(
                    check_name="stability.weight_check",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="Cannot check numerical stability: artifact file not found",
                )
            )
            return results

        try:
            tensors = self._load_model_tensors(metadata)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            results.append(
                ValidationCheckResult(
                    check_name="stability.weight_load",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Cannot load model tensors for stability check: {exc}",
                )
            )
            return results

        total_params = 0
        nan_count = 0
        inf_count = 0
        max_abs_value = 0.0

        for name, tensor in tensors.items():
            arr = np.asarray(tensor, dtype=np.float32) if not isinstance(tensor, np.ndarray) else tensor
            total_params += arr.size
            nan_count += int(np.sum(np.isnan(arr)))
            inf_count += int(np.sum(np.isinf(arr)))
            tensor_max = float(np.max(np.abs(arr[~np.isnan(arr) & ~np.isinf(arr)]))) if arr.size > 0 else 0.0
            max_abs_value = max(max_abs_value, tensor_max)

        # NaN check
        has_nan = nan_count > 0
        results.append(
            ValidationCheckResult(
                check_name="stability.nan_check",
                passed=not has_nan,
                severity=ValidationSeverity.CRITICAL if has_nan else ValidationSeverity.INFO,
                message=(
                    f"No NaN values found in {total_params} parameters"
                    if not has_nan
                    else f"Found {nan_count} NaN values in model weights ({nan_count / max(total_params, 1) * 100:.2f}%)"
                ),
                details={"nan_count": nan_count, "total_params": total_params},
            )
        )

        # Inf check
        has_inf = inf_count > 0
        results.append(
            ValidationCheckResult(
                check_name="stability.inf_check",
                passed=not has_inf,
                severity=ValidationSeverity.CRITICAL if has_inf else ValidationSeverity.INFO,
                message=(
                    f"No Inf values found in {total_params} parameters"
                    if not has_inf
                    else f"Found {inf_count} Inf values in model weights"
                ),
                details={"inf_count": inf_count, "total_params": total_params},
            )
        )

        # Extreme value check (weights > 1000 are suspicious)
        extreme_threshold = 1000.0
        has_extreme = max_abs_value > extreme_threshold
        results.append(
            ValidationCheckResult(
                check_name="stability.extreme_values",
                passed=not has_extreme,
                severity=ValidationSeverity.WARNING if has_extreme else ValidationSeverity.INFO,
                message=(
                    f"Max absolute weight value is {max_abs_value:.2f} (within normal range)"
                    if not has_extreme
                    else f"Max absolute weight value is {max_abs_value:.2f} (exceeds threshold of {extreme_threshold})"
                ),
                details={"max_abs_value": max_abs_value, "threshold": extreme_threshold},
            )
        )

        return results

    # ------------------------------------------------------------------
    # Compliance checks
    # ------------------------------------------------------------------

    def _check_compliance_flags(self, metadata: ModelMetadata) -> list[ValidationCheckResult]:
        """Check regulatory and privacy compliance flags.

        Verifies that models intended for clinical domains have the
        required compliance metadata.

        Args:
            metadata: Model metadata to check.

        Returns:
            List of check results.
        """
        results: list[ValidationCheckResult] = []

        # Check intended use statement
        has_intended_use = bool(metadata.intended_use and len(metadata.intended_use.strip()) > 10)
        results.append(
            ValidationCheckResult(
                check_name="compliance.intended_use",
                passed=has_intended_use,
                severity=ValidationSeverity.WARNING if not has_intended_use else ValidationSeverity.INFO,
                message=(
                    "Intended use statement is provided"
                    if has_intended_use
                    else "Intended use statement is missing or too short (required for regulatory submissions)"
                ),
            )
        )

        # Check clinical domain
        has_domain = bool(metadata.clinical_domain and len(metadata.clinical_domain.strip()) > 0)
        results.append(
            ValidationCheckResult(
                check_name="compliance.clinical_domain",
                passed=has_domain,
                severity=ValidationSeverity.WARNING if not has_domain else ValidationSeverity.INFO,
                message=(
                    f"Clinical domain specified: {metadata.clinical_domain}"
                    if has_domain
                    else "Clinical domain is not specified (required for FDA submissions)"
                ),
            )
        )

        # Check source site for provenance
        has_source = bool(metadata.source_site and len(metadata.source_site.strip()) > 0)
        results.append(
            ValidationCheckResult(
                check_name="compliance.source_site",
                passed=has_source,
                severity=ValidationSeverity.WARNING if not has_source else ValidationSeverity.INFO,
                message=(
                    f"Source site identified: {metadata.source_site}"
                    if has_source
                    else "Source site is not specified (required for audit trail)"
                ),
            )
        )

        # Check that training config includes DP info if model is for production
        if metadata.stage == ModelStage.PRODUCTION:
            dp_config = metadata.training_config.get("differential_privacy", {})
            has_dp_info = bool(dp_config and "epsilon" in dp_config)
            results.append(
                ValidationCheckResult(
                    check_name="compliance.differential_privacy",
                    passed=has_dp_info,
                    severity=ValidationSeverity.WARNING if not has_dp_info else ValidationSeverity.INFO,
                    message=(
                        f"Differential privacy documented (epsilon={dp_config.get('epsilon', 'N/A')})"
                        if has_dp_info
                        else "Differential privacy configuration not documented for production model"
                    ),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Model tensor loading
    # ------------------------------------------------------------------

    def _load_model_tensors(self, metadata: ModelMetadata) -> dict[str, Any]:
        """Load model tensors from the artifact file.

        Dispatches to the appropriate loader based on model format.

        Args:
            metadata: Model metadata with format and path.

        Returns:
            Dictionary of tensor name -> tensor data.
        """
        fmt = metadata.model_format.lower()

        if fmt == "pt" and HAS_TORCH:
            data = torch.load(metadata.artifact_path, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                return {k: v.numpy() if hasattr(v, "numpy") else v for k, v in data.items()}
            return {"model": data}

        elif fmt == "safetensors" and HAS_SAFETENSORS:
            tensors = {}
            with safe_open(metadata.artifact_path, framework="numpy") as fh:
                for key in fh.keys():
                    tensors[key] = fh.get_tensor(key)
            return tensors

        elif fmt == "onnx" and HAS_ONNX:
            onnx_model = onnx.load(metadata.artifact_path)
            tensors = {}
            for init in onnx_model.graph.initializer:
                tensor = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                tensors[init.name] = tensor.copy()
            return tensors

        elif fmt in ("numpy_npz", "npz"):
            data = np.load(metadata.artifact_path, allow_pickle=False)
            return dict(data)

        else:
            raise ValidationError(
                f"Cannot load tensors for format '{fmt}'. Ensure the required framework is installed."
            )
