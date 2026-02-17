"""Deployment readiness checker for federated oncology AI systems.

Performs comprehensive checklist-based validation across five categories
before a trained model or simulation policy can be promoted to a
production or clinical-adjacent environment:

1. Model validation — performance metrics, bias audits, robustness.
2. Safety compliance — dose bounds, fail-safe behaviour, anomaly detection.
3. Regulatory — FDA 21 CFR Part 11, HIPAA, GDPR, ICH E6(R2).
4. Infrastructure — compute, networking, monitoring, disaster recovery.
5. Documentation — SOPs, training records, version control.

DISCLAIMER: RESEARCH USE ONLY — This tool is intended for research and
educational purposes. It is NOT approved as a substitute for formal
regulatory review. All deployment decisions must be made by qualified
personnel in accordance with applicable regulations.

VERSION: 0.4.0
LICENSE: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference checklists — each item cites its authoritative source.
# ---------------------------------------------------------------------------

# Model validation checks.
# Source: FDA Artificial Intelligence/Machine Learning Action Plan, 2021.
# Source: Health Canada — Guidance on Good ML Practice (GMLP), 2021.
MODEL_VALIDATION_CHECKS: list[dict[str, str]] = [
    {
        "id": "MV-001",
        "name": "AUC-ROC threshold",
        "description": "Model AUC-ROC >= 0.80 on held-out test set",
        "source": "FDA AI/ML Action Plan 2021",
    },
    {
        "id": "MV-002",
        "name": "Sensitivity threshold",
        "description": "Sensitivity >= 0.85 for primary endpoint",
        "source": "FDA AI/ML Action Plan 2021",
    },
    {
        "id": "MV-003",
        "name": "Specificity threshold",
        "description": "Specificity >= 0.80 for primary endpoint",
        "source": "FDA AI/ML Action Plan 2021",
    },
    {
        "id": "MV-004",
        "name": "Subgroup fairness audit",
        "description": "Performance parity across demographic subgroups (age, sex, ethnicity)",
        "source": "Health Canada GMLP 2021",
    },
    {
        "id": "MV-005",
        "name": "Calibration check",
        "description": "Brier score <= 0.25 on validation set",
        "source": "FDA AI/ML Action Plan 2021",
    },
    {
        "id": "MV-006",
        "name": "Robustness to distribution shift",
        "description": "Performance degradation <= 10% on external validation cohort",
        "source": "Health Canada GMLP 2021",
    },
    {
        "id": "MV-007",
        "name": "Reproducibility verified",
        "description": "Results reproducible across 3 independent training runs (seed variation)",
        "source": "FDA AI/ML Action Plan 2021",
    },
]

# Safety compliance checks.
# Source: IEC 62304 — Medical device software lifecycle, 2015.
# Source: ISO 14971 — Application of risk management to medical devices, 2019.
SAFETY_COMPLIANCE_CHECKS: list[dict[str, str]] = [
    {
        "id": "SC-001",
        "name": "Dose bound enforcement",
        "description": "All predicted doses bounded within NCCN/RTOG limits",
        "source": "NCCN v2024; IEC 62304",
    },
    {
        "id": "SC-002",
        "name": "Fail-safe mechanism",
        "description": "System defaults to safe state on prediction failure",
        "source": "ISO 14971:2019",
    },
    {
        "id": "SC-003",
        "name": "Anomaly detection",
        "description": "Out-of-distribution input detection enabled and tested",
        "source": "IEC 62304:2015",
    },
    {
        "id": "SC-004",
        "name": "Human-in-the-loop",
        "description": "Clinical override mechanism available for all predictions",
        "source": "FDA Guidance on CDS, 2022",
    },
    {
        "id": "SC-005",
        "name": "Adverse event logging",
        "description": "All model predictions logged with timestamps for audit trail",
        "source": "ISO 14971:2019",
    },
    {
        "id": "SC-006",
        "name": "Risk classification",
        "description": "Software risk class determined per IEC 62304 (Class A/B/C)",
        "source": "IEC 62304:2015",
    },
]

# Regulatory compliance checks.
# Source: 21 CFR Part 11 — Electronic Records; Electronic Signatures.
# Source: HIPAA Security Rule, 45 CFR 164.
# Source: GDPR Article 35 — Data Protection Impact Assessment.
REGULATORY_CHECKS: list[dict[str, str]] = [
    {
        "id": "RG-001",
        "name": "21 CFR Part 11 compliance",
        "description": "Electronic records have audit trails, access controls, and electronic signatures",
        "source": "21 CFR Part 11",
    },
    {
        "id": "RG-002",
        "name": "HIPAA de-identification",
        "description": "All patient data de-identified per Safe Harbor or Expert Determination",
        "source": "45 CFR 164.514",
    },
    {
        "id": "RG-003",
        "name": "GDPR DPIA completed",
        "description": "Data Protection Impact Assessment completed for EU site data",
        "source": "GDPR Article 35",
    },
    {
        "id": "RG-004",
        "name": "ICH E6(R2) GCP compliance",
        "description": "Trial conduct follows Good Clinical Practice guidelines",
        "source": "ICH E6(R2) 2016",
    },
    {
        "id": "RG-005",
        "name": "IRB/Ethics approval",
        "description": "Institutional Review Board approval documented for each site",
        "source": "21 CFR Part 56",
    },
    {
        "id": "RG-006",
        "name": "Informed consent verification",
        "description": "Electronic consent records complete and version-matched",
        "source": "ICH E6(R2) 2016",
    },
]

# Infrastructure checks.
# Source: NIST Cybersecurity Framework v2.0, 2024.
# Source: AWS Well-Architected Framework — Healthcare Lens, 2023.
INFRASTRUCTURE_CHECKS: list[dict[str, str]] = [
    {
        "id": "IN-001",
        "name": "Compute capacity",
        "description": "Sufficient GPU/CPU resources allocated for inference workload",
        "source": "AWS Well-Architected Healthcare Lens 2023",
    },
    {
        "id": "IN-002",
        "name": "Network latency",
        "description": "Inter-site FL communication latency < 500 ms p99",
        "source": "AWS Well-Architected Healthcare Lens 2023",
    },
    {
        "id": "IN-003",
        "name": "TLS encryption",
        "description": "All data in transit encrypted with TLS 1.3",
        "source": "NIST Cybersecurity Framework v2.0",
    },
    {
        "id": "IN-004",
        "name": "Encryption at rest",
        "description": "All data at rest encrypted with AES-256",
        "source": "NIST Cybersecurity Framework v2.0",
    },
    {
        "id": "IN-005",
        "name": "Monitoring and alerting",
        "description": "Health checks, log aggregation, and alerting pipelines operational",
        "source": "NIST Cybersecurity Framework v2.0",
    },
    {
        "id": "IN-006",
        "name": "Disaster recovery plan",
        "description": "Documented DR plan with RTO < 4 hours and RPO < 1 hour",
        "source": "AWS Well-Architected Healthcare Lens 2023",
    },
    {
        "id": "IN-007",
        "name": "Backup verification",
        "description": "Model artifacts and configuration backups tested within last 30 days",
        "source": "NIST Cybersecurity Framework v2.0",
    },
]

# Documentation checks.
# Source: ISO 13485 — Quality management systems for medical devices, 2016.
# Source: FDA Design Control Guidance, 1997.
DOCUMENTATION_CHECKS: list[dict[str, str]] = [
    {
        "id": "DC-001",
        "name": "Standard operating procedures",
        "description": "SOPs for model training, validation, and deployment documented",
        "source": "ISO 13485:2016",
    },
    {
        "id": "DC-002",
        "name": "Training records",
        "description": "Personnel training records current for all operators",
        "source": "ISO 13485:2016",
    },
    {
        "id": "DC-003",
        "name": "Version control",
        "description": "Model, data, and code versions tracked in version control system",
        "source": "FDA Design Control Guidance 1997",
    },
    {
        "id": "DC-004",
        "name": "Change control log",
        "description": "All changes to production system documented with rationale",
        "source": "ISO 13485:2016",
    },
    {
        "id": "DC-005",
        "name": "User documentation",
        "description": "End-user instructions for use (IFU) reviewed and approved",
        "source": "FDA Design Control Guidance 1997",
    },
    {
        "id": "DC-006",
        "name": "Risk management file",
        "description": "Complete risk management file per ISO 14971 maintained",
        "source": "ISO 14971:2019",
    },
]

# All check categories mapped by name.
ALL_CHECK_CATEGORIES: dict[str, list[dict[str, str]]] = {
    "model_validation": MODEL_VALIDATION_CHECKS,
    "safety_compliance": SAFETY_COMPLIANCE_CHECKS,
    "regulatory": REGULATORY_CHECKS,
    "infrastructure": INFRASTRUCTURE_CHECKS,
    "documentation": DOCUMENTATION_CHECKS,
}


# ── Enums ──────────────────────────────────────────────────────────────────


class CheckStatus(str, Enum):
    """Status of an individual readiness check.

    PASS — Check has been verified and meets requirements.
    FAIL — Check does not meet requirements; blocks deployment.
    WARNING — Check partially met; requires attention but not blocking.
    REQUIRES_VERIFICATION — Check needs manual human verification.
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    REQUIRES_VERIFICATION = "requires_verification"


class DeploymentDecision(str, Enum):
    """Overall deployment recommendation.

    Computed from the aggregate of all individual check statuses.
    """

    APPROVED = "approved"
    CONDITIONAL = "conditional"
    BLOCKED = "blocked"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class ReadinessCheck:
    """Result of a single deployment readiness check.

    Attributes:
        check_id: Unique check identifier (e.g. MV-001).
        category: Category the check belongs to.
        name: Short human-readable check name.
        description: Detailed description of what is checked.
        status: Result status of the check.
        notes: Reviewer notes or automated findings.
        source: Regulatory or standards citation.
        verified_by: Name or ID of the person who verified.
        verified_at: ISO 8601 timestamp of verification.
    """

    check_id: str = ""
    category: str = ""
    name: str = ""
    description: str = ""
    status: str = CheckStatus.REQUIRES_VERIFICATION.value
    notes: str = ""
    source: str = ""
    verified_by: str = ""
    verified_at: str = ""


@dataclass
class ReadinessReport:
    """Aggregated deployment readiness report.

    Attributes:
        report_id: Unique report identifier.
        model_name: Name or identifier of the model being assessed.
        model_version: Semantic version of the model.
        generated_at: ISO 8601 timestamp of report generation.
        decision: Overall deployment decision.
        total_checks: Total number of checks evaluated.
        passed: Number of checks that passed.
        failed: Number of checks that failed.
        warnings: Number of checks with warnings.
        requires_verification: Number requiring manual review.
        checks: Detailed list of individual check results.
        summary_by_category: Pass/fail counts grouped by category.
    """

    report_id: str = ""
    model_name: str = ""
    model_version: str = ""
    generated_at: str = ""
    decision: str = DeploymentDecision.BLOCKED.value
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    requires_verification: int = 0
    checks: list[dict[str, str]] = field(default_factory=list)
    summary_by_category: dict[str, dict[str, int]] = field(default_factory=dict)


# ── Core helpers ───────────────────────────────────────────────────────────


def build_checklist(
    categories: list[str] | None = None,
) -> list[ReadinessCheck]:
    """Build the full checklist of readiness checks.

    Parameters
    ----------
    categories:
        Optional list of category names to include.  If ``None``, all
        categories are included.

    Returns
    -------
    list[ReadinessCheck]:
        Ordered list of checks to evaluate.
    """
    selected = categories or list(ALL_CHECK_CATEGORIES.keys())
    checks: list[ReadinessCheck] = []
    for cat_name in selected:
        cat_checks = ALL_CHECK_CATEGORIES.get(cat_name, [])
        for item in cat_checks:
            checks.append(
                ReadinessCheck(
                    check_id=item["id"],
                    category=cat_name,
                    name=item["name"],
                    description=item["description"],
                    source=item.get("source", ""),
                )
            )
    logger.info("Built checklist with %d checks across %d categories.", len(checks), len(selected))
    return checks


def evaluate_checks(
    checks: list[ReadinessCheck],
    overrides: dict[str, str] | None = None,
) -> list[ReadinessCheck]:
    """Evaluate a list of readiness checks.

    Without real system integration, all checks default to
    ``REQUIRES_VERIFICATION``.  The *overrides* dict allows the caller
    to set specific check IDs to a known status (e.g. from a CI
    pipeline or manual review).

    Parameters
    ----------
    checks:
        List of checks to evaluate.
    overrides:
        Mapping of ``check_id`` to ``CheckStatus`` value.

    Returns
    -------
    list[ReadinessCheck]:
        Checks with updated statuses.
    """
    overrides = overrides or {}
    now = datetime.now(timezone.utc).isoformat()
    for check in checks:
        if check.check_id in overrides:
            status_value = overrides[check.check_id]
            try:
                check.status = CheckStatus(status_value).value
            except ValueError:
                check.status = CheckStatus.REQUIRES_VERIFICATION.value
                check.notes = f"Invalid override status '{status_value}'; reset to requires_verification."
            check.verified_at = now
            check.notes = check.notes or f"Status set via override to '{check.status}'."
        else:
            check.status = CheckStatus.REQUIRES_VERIFICATION.value
    return checks


def generate_readiness_report(
    model_name: str,
    model_version: str,
    checks: list[ReadinessCheck],
) -> ReadinessReport:
    """Generate a deployment readiness report from evaluated checks.

    Parameters
    ----------
    model_name:
        Name of the model or system being assessed.
    model_version:
        Semantic version string.
    checks:
        Evaluated readiness checks.

    Returns
    -------
    ReadinessReport:
        Complete report with deployment decision.
    """
    report = ReadinessReport(
        report_id=f"DR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        model_name=model_name,
        model_version=model_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_checks=len(checks),
    )

    cat_summary: dict[str, dict[str, int]] = {}
    for check in checks:
        report.checks.append(asdict(check))
        cat = check.category
        if cat not in cat_summary:
            cat_summary[cat] = {"pass": 0, "fail": 0, "warning": 0, "requires_verification": 0}

        if check.status == CheckStatus.PASS.value:
            report.passed += 1
            cat_summary[cat]["pass"] += 1
        elif check.status == CheckStatus.FAIL.value:
            report.failed += 1
            cat_summary[cat]["fail"] += 1
        elif check.status == CheckStatus.WARNING.value:
            report.warnings += 1
            cat_summary[cat]["warning"] += 1
        else:
            report.requires_verification += 1
            cat_summary[cat]["requires_verification"] += 1

    report.summary_by_category = cat_summary

    # Determine deployment decision.
    if report.failed > 0:
        report.decision = DeploymentDecision.BLOCKED.value
    elif report.requires_verification > 0 or report.warnings > 0:
        report.decision = DeploymentDecision.CONDITIONAL.value
    else:
        report.decision = DeploymentDecision.APPROVED.value

    logger.info(
        "Report %s: decision=%s (pass=%d fail=%d warn=%d verify=%d)",
        report.report_id,
        report.decision,
        report.passed,
        report.failed,
        report.warnings,
        report.requires_verification,
    )
    return report


def export_report(report: ReadinessReport, output_path: str, fmt: str = "json") -> str:
    """Export a readiness report to a file.

    Parameters
    ----------
    report:
        The report to export.
    output_path:
        Destination file path.
    fmt:
        Export format — currently only ``"json"`` is supported.

    Returns
    -------
    str:
        Path to the written file.
    """
    if fmt != "json":
        raise ValueError(f"Unsupported export format '{fmt}'. Only 'json' is supported.")

    data = asdict(report)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Report exported to %s", output_path)
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="deployment_readiness",
        description=("Deployment readiness checker for federated oncology AI systems. RESEARCH USE ONLY."),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.4.0")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging verbosity (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands.")

    # check ---
    sp_check = subparsers.add_parser("check", help="Run readiness checks for a specific category.")
    sp_check.add_argument(
        "--category",
        choices=list(ALL_CHECK_CATEGORIES.keys()),
        default=None,
        help="Category to check (default: all).",
    )

    # report ---
    sp_report = subparsers.add_parser("report", help="Generate a full deployment readiness report.")
    sp_report.add_argument("--model-name", default="oncology-fl-model", help="Model name.")
    sp_report.add_argument("--model-version", default="0.4.0", help="Model version.")

    # export ---
    sp_export = subparsers.add_parser("export", help="Generate and export a readiness report to file.")
    sp_export.add_argument("--model-name", default="oncology-fl-model", help="Model name.")
    sp_export.add_argument("--model-version", default="0.4.0", help="Model version.")
    sp_export.add_argument("--output", default="readiness_report.json", help="Output file path.")
    sp_export.add_argument("--format", choices=["json"], default="json", help="Export format.")

    return parser


def _output(data: Any, as_json: bool) -> None:
    """Print *data* as JSON or human-readable text."""
    if as_json:
        print(json.dumps(asdict(data) if hasattr(data, "__dataclass_fields__") else data, indent=2, default=str))
    else:
        if hasattr(data, "__dataclass_fields__"):
            for k, v in asdict(data).items():
                print(f"  {k}: {v}")
        else:
            print(data)


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the deployment readiness CLI.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int:
        Exit code — 0 on success, 1 on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "check":
            categories = [args.category] if args.category else None
            checks = build_checklist(categories=categories)
            checks = evaluate_checks(checks)
            if args.json:
                print(json.dumps([asdict(c) for c in checks], indent=2, default=str))
            else:
                for check in checks:
                    status_display = check.status.upper()
                    print(f"  [{status_display}] {check.check_id}: {check.name}")
                    print(f"           {check.description}")
                    print(f"           Source: {check.source}")
            return 0

        if args.command == "report":
            checks = build_checklist()
            checks = evaluate_checks(checks)
            report = generate_readiness_report(
                model_name=args.model_name,
                model_version=args.model_version,
                checks=checks,
            )
            _output(report, args.json)
            return 0

        if args.command == "export":
            checks = build_checklist()
            checks = evaluate_checks(checks)
            report = generate_readiness_report(
                model_name=args.model_name,
                model_version=args.model_version,
                checks=checks,
            )
            path = export_report(report, args.output, fmt=args.format)
            print(f"  Report exported to: {path}")
            return 0
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
