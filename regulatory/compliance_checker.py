"""Compliance validation for federated oncology trials.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Checks that the federated learning configuration and data handling
practices meet HIPAA, GDPR, and FDA regulatory requirements before
training begins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Regulation(str, Enum):
    """Supported regulatory frameworks."""

    HIPAA = "hipaa"
    GDPR = "gdpr"
    FDA_21CFR11 = "fda_21cfr11"
    ICH_GCP = "ich_gcp"


class CheckStatus(str, Enum):
    """Result of a single compliance check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "n/a"


@dataclass
class CheckResult:
    """Result of one compliance check.

    Attributes:
        check_id: Unique identifier for the check.
        regulation: Which regulation this check belongs to.
        description: Human-readable description.
        status: Pass/fail/warning result.
        details: Additional context.
    """

    check_id: str
    regulation: Regulation
    description: str
    status: CheckStatus
    details: str = ""


@dataclass
class ComplianceReport:
    """Full compliance report across all applicable regulations.

    Attributes:
        checks: Individual check results.
        overall_status: Overall pass/fail.
        summary: Counts by status.
    """

    checks: list[CheckResult] = field(default_factory=list)
    overall_status: CheckStatus = CheckStatus.PASS
    summary: dict[str, int] = field(default_factory=dict)


class ComplianceChecker:
    """Validates federated learning configurations against regulations.

    Runs a suite of checks covering data privacy, access control,
    audit logging, and differential privacy configuration.

    Args:
        regulations: List of regulatory frameworks to check against.
    """

    def __init__(
        self,
        regulations: list[Regulation | str] | None = None,
    ):
        if regulations is None:
            regulations = [Regulation.HIPAA, Regulation.GDPR]
        self.regulations = [Regulation(r) if isinstance(r, str) else r for r in regulations]

    def check_federation_config(
        self,
        config: dict,
    ) -> ComplianceReport:
        """Run all applicable compliance checks on a federation config.

        Args:
            config: Dictionary describing the federation setup.
                Expected keys:
                - ``use_differential_privacy``: bool
                - ``dp_epsilon``: float
                - ``use_secure_aggregation``: bool
                - ``use_deidentification``: bool
                - ``audit_logging_enabled``: bool
                - ``consent_management_enabled``: bool
                - ``encryption_in_transit``: bool
                - ``min_clients``: int

        Returns:
            ComplianceReport with all check results.
        """
        checks: list[CheckResult] = []

        for reg in self.regulations:
            if reg == Regulation.HIPAA:
                checks.extend(self._check_hipaa(config))
            elif reg == Regulation.GDPR:
                checks.extend(self._check_gdpr(config))
            elif reg == Regulation.FDA_21CFR11:
                checks.extend(self._check_fda(config))

        status_counts: dict[str, int] = {}
        for c in checks:
            s = c.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        overall = CheckStatus.PASS
        if status_counts.get("fail", 0) > 0:
            overall = CheckStatus.FAIL
        elif status_counts.get("warning", 0) > 0:
            overall = CheckStatus.WARNING

        report = ComplianceReport(
            checks=checks,
            overall_status=overall,
            summary=status_counts,
        )

        logger.info(
            "Compliance check complete: %s (%s)",
            overall.value,
            status_counts,
        )
        return report

    @staticmethod
    def _check_hipaa(config: dict) -> list[CheckResult]:
        """HIPAA-specific compliance checks."""
        results: list[CheckResult] = []

        # PHI de-identification
        results.append(
            CheckResult(
                check_id="hipaa_deident",
                regulation=Regulation.HIPAA,
                description="PHI de-identification is enabled",
                status=(CheckStatus.PASS if config.get("use_deidentification", False) else CheckStatus.FAIL),
                details="HIPAA Safe Harbor requires de-identification of 18 identifiers.",
            )
        )

        # Audit logging
        results.append(
            CheckResult(
                check_id="hipaa_audit",
                regulation=Regulation.HIPAA,
                description="Audit logging is enabled",
                status=(CheckStatus.PASS if config.get("audit_logging_enabled", False) else CheckStatus.FAIL),
                details="HIPAA requires audit trails for all PHI access.",
            )
        )

        # Encryption
        results.append(
            CheckResult(
                check_id="hipaa_encryption",
                regulation=Regulation.HIPAA,
                description="Encryption in transit is enabled",
                status=(CheckStatus.PASS if config.get("encryption_in_transit", False) else CheckStatus.WARNING),
                details="HIPAA recommends encryption for data in transit.",
            )
        )

        # Minimum necessary
        dp_enabled = config.get("use_differential_privacy", False)
        sa_enabled = config.get("use_secure_aggregation", False)
        results.append(
            CheckResult(
                check_id="hipaa_minimum_necessary",
                regulation=Regulation.HIPAA,
                description="Privacy-preserving techniques are enabled (DP or SA)",
                status=CheckStatus.PASS if (dp_enabled or sa_enabled) else CheckStatus.WARNING,
                details="Differential privacy or secure aggregation limits data exposure.",
            )
        )

        return results

    @staticmethod
    def _check_gdpr(config: dict) -> list[CheckResult]:
        """GDPR-specific compliance checks."""
        results: list[CheckResult] = []

        # Consent management
        results.append(
            CheckResult(
                check_id="gdpr_consent",
                regulation=Regulation.GDPR,
                description="Consent management is enabled",
                status=(CheckStatus.PASS if config.get("consent_management_enabled", False) else CheckStatus.FAIL),
                details="GDPR requires explicit consent for data processing.",
            )
        )

        # Data minimization (DP)
        results.append(
            CheckResult(
                check_id="gdpr_minimization",
                regulation=Regulation.GDPR,
                description="Data minimization via differential privacy",
                status=(CheckStatus.PASS if config.get("use_differential_privacy", False) else CheckStatus.WARNING),
                details="GDPR data minimization principle recommends DP.",
            )
        )

        # Right to erasure
        results.append(
            CheckResult(
                check_id="gdpr_erasure",
                regulation=Regulation.GDPR,
                description="Consent revocation is supported",
                status=(CheckStatus.PASS if config.get("consent_management_enabled", False) else CheckStatus.FAIL),
                details="GDPR right to erasure requires consent revocation.",
            )
        )

        return results

    @staticmethod
    def _check_fda(config: dict) -> list[CheckResult]:
        """FDA 21 CFR Part 11 compliance checks."""
        results: list[CheckResult] = []

        results.append(
            CheckResult(
                check_id="fda_audit_trail",
                regulation=Regulation.FDA_21CFR11,
                description="Audit trail for electronic records",
                status=(CheckStatus.PASS if config.get("audit_logging_enabled", False) else CheckStatus.FAIL),
                details="21 CFR Part 11 requires audit trails for electronic records.",
            )
        )

        return results
