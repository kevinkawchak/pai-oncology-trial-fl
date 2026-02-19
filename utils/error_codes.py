"""Structured error codes for operational observability.

Provides stable, machine-oriented error codes for monitoring and
alert routing.  Each code maps to a human-readable description
and a suggested severity level.

RESEARCH USE ONLY — Not approved for clinical deployment.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Machine-oriented error codes for key failure modes."""

    # ── Model operations ──────────────────────────────────────────────
    MODEL_LOAD_FAILED = "E-MOD-001"
    MODEL_VALIDATION_FAILED = "E-MOD-002"
    MODEL_EXPORT_FAILED = "E-MOD-003"
    MODEL_CONVERSION_FAILED = "E-MOD-004"

    # ── Simulation ────────────────────────────────────────────────────
    SIM_BACKEND_INIT_FAILED = "E-SIM-001"
    SIM_SYNC_FAILED = "E-SIM-002"
    SIM_STATE_INVALID = "E-SIM-003"

    # ── Privacy / PHI ─────────────────────────────────────────────────
    PHI_DETECTION_FAILED = "E-PHI-001"
    DEIDENTIFICATION_FAILED = "E-PHI-002"
    AUDIT_INTEGRITY_VIOLATION = "E-PHI-003"

    # ── Data management ───────────────────────────────────────────────
    DATA_VALIDATION_FAILED = "E-DAT-001"
    DATA_QUALITY_BELOW_THRESHOLD = "E-DAT-002"
    SCHEMA_MISMATCH = "E-DAT-003"

    # ── Configuration / Security ──────────────────────────────────────
    CONFIG_MISSING_REQUIRED = "E-CFG-001"
    SECRET_NOT_CONFIGURED = "E-CFG-002"
    AUTH_FAILED = "E-SEC-001"

    # ── Analytics ─────────────────────────────────────────────────────
    ANALYSIS_CONVERGENCE_FAILED = "E-ANA-001"
    INSUFFICIENT_SAMPLE_SIZE = "E-ANA-002"


# Human-readable descriptions for each error code
ERROR_DESCRIPTIONS: dict[str, str] = {
    ErrorCode.MODEL_LOAD_FAILED: "Failed to load model from specified path or registry.",
    ErrorCode.MODEL_VALIDATION_FAILED: "Model did not pass validation checks.",
    ErrorCode.MODEL_EXPORT_FAILED: "Failed to export model to target format.",
    ErrorCode.MODEL_CONVERSION_FAILED: "Model format conversion failed.",
    ErrorCode.SIM_BACKEND_INIT_FAILED: "Simulation backend initialization failed.",
    ErrorCode.SIM_SYNC_FAILED: "Cross-simulator state synchronization failed.",
    ErrorCode.SIM_STATE_INVALID: "Simulation state contains invalid values.",
    ErrorCode.PHI_DETECTION_FAILED: "PHI detection pipeline encountered an error.",
    ErrorCode.DEIDENTIFICATION_FAILED: "De-identification transformation failed.",
    ErrorCode.AUDIT_INTEGRITY_VIOLATION: "Audit trail integrity check failed.",
    ErrorCode.DATA_VALIDATION_FAILED: "Input data failed validation constraints.",
    ErrorCode.DATA_QUALITY_BELOW_THRESHOLD: "Data quality score below acceptable threshold.",
    ErrorCode.SCHEMA_MISMATCH: "Data schema does not match expected format.",
    ErrorCode.CONFIG_MISSING_REQUIRED: "Required configuration value is missing.",
    ErrorCode.SECRET_NOT_CONFIGURED: "Required secret/key is not configured in environment.",
    ErrorCode.AUTH_FAILED: "Authentication or authorization failed.",
    ErrorCode.ANALYSIS_CONVERGENCE_FAILED: "Statistical analysis did not converge.",
    ErrorCode.INSUFFICIENT_SAMPLE_SIZE: "Sample size is too small for the requested analysis.",
}


def get_error_description(code: ErrorCode) -> str:
    """Return a human-readable description for the given error code."""
    return ERROR_DESCRIPTIONS.get(code, f"Unknown error code: {code}")
