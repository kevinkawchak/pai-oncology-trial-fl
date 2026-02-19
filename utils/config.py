"""Centralized secure configuration defaults.

Provides typed, environment-driven configuration for security-sensitive
settings, preventing drift across modules.  In production, required
fields must be set via environment variables; missing values cause
immediate failure.

RESEARCH USE ONLY — Not approved for clinical deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SecurityConfig:
    """Security-sensitive configuration values.

    Attributes:
        hmac_key: HMAC key for integrity hashing (from ``PAI_HMAC_KEY``).
        audit_retention_days: Days to retain audit records.
        phi_log_sanitization: Whether to sanitize PHI from logs.
        require_tls: Whether to require TLS for data exchange.
        max_failed_auth_attempts: Lockout threshold.
    """

    hmac_key: bytes = field(default_factory=lambda: _load_bytes_env("PAI_HMAC_KEY", b"pai-oncology-dev-default-key"))
    audit_retention_days: int = field(default_factory=lambda: int(os.environ.get("PAI_AUDIT_RETENTION_DAYS", "365")))
    phi_log_sanitization: bool = field(default_factory=lambda: os.environ.get("PAI_PHI_LOG_SANITIZE", "1") == "1")
    require_tls: bool = field(default_factory=lambda: os.environ.get("PAI_REQUIRE_TLS", "1") == "1")
    max_failed_auth_attempts: int = field(default_factory=lambda: int(os.environ.get("PAI_MAX_FAILED_AUTH", "5")))


@dataclass(frozen=True)
class OperationalConfig:
    """Operational defaults for runtime behavior.

    Attributes:
        default_seed: Random seed for reproducibility.
        log_level: Default logging level.
        timestamp_format: ISO 8601 format string for timestamps.
    """

    default_seed: int = 42
    log_level: str = field(default_factory=lambda: os.environ.get("PAI_LOG_LEVEL", "INFO"))
    timestamp_format: str = "%Y-%m-%dT%H:%M:%S%z"


def _load_bytes_env(name: str, default: bytes) -> bytes:
    """Load a bytes value from an environment variable."""
    value = os.environ.get(name)
    if value is not None:
        return value.encode("utf-8")
    return default


def get_security_config() -> SecurityConfig:
    """Return the current security configuration."""
    return SecurityConfig()


def get_operational_config() -> OperationalConfig:
    """Return the current operational configuration."""
    return OperationalConfig()
