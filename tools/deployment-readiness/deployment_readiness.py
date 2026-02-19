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

# Reference checklists — each entry is (id, name, description, source).
# Sources: FDA AI/ML Action Plan 2021, Health Canada GMLP 2021,
# IEC 62304:2015, ISO 14971:2019, 21 CFR Part 11, HIPAA 45 CFR 164,
# GDPR Art. 35, ICH E6(R2) 2016, NIST CSF v2.0, AWS Healthcare Lens 2023,
# ISO 13485:2016, FDA Design Control 1997.

_CHECK_FIELDS = ("id", "name", "description", "source")


def _build_checks(rows: list[tuple[str, str, str, str]]) -> list[dict[str, str]]:
    """Convert compact tuples into list-of-dict check definitions."""
    return [dict(zip(_CHECK_FIELDS, r)) for r in rows]


MODEL_VALIDATION_CHECKS = _build_checks(
    [
        ("MV-001", "AUC-ROC threshold", "AUC-ROC >= 0.80 on held-out test set", "FDA AI/ML Action Plan 2021"),
        ("MV-002", "Sensitivity threshold", "Sensitivity >= 0.85 for primary endpoint", "FDA AI/ML Action Plan 2021"),
        ("MV-003", "Specificity threshold", "Specificity >= 0.80 for primary endpoint", "FDA AI/ML Action Plan 2021"),
        (
            "MV-004",
            "Subgroup fairness audit",
            "Performance parity across demographic subgroups",
            "Health Canada GMLP 2021",
        ),
        ("MV-005", "Calibration check", "Brier score <= 0.25 on validation set", "FDA AI/ML Action Plan 2021"),
        ("MV-006", "Robustness check", "Degradation <= 10% on external cohort", "Health Canada GMLP 2021"),
        ("MV-007", "Reproducibility", "Reproducible across 3 training runs", "FDA AI/ML Action Plan 2021"),
    ]
)

SAFETY_COMPLIANCE_CHECKS = _build_checks(
    [
        ("SC-001", "Dose bound enforcement", "Doses bounded within NCCN/RTOG limits", "NCCN v2024; IEC 62304"),
        ("SC-002", "Fail-safe mechanism", "System defaults to safe state on failure", "ISO 14971:2019"),
        ("SC-003", "Anomaly detection", "OOD input detection enabled and tested", "IEC 62304:2015"),
        ("SC-004", "Human-in-the-loop", "Clinical override for all predictions", "FDA CDS Guidance 2022"),
        ("SC-005", "Adverse event logging", "Predictions logged with timestamps", "ISO 14971:2019"),
        ("SC-006", "Risk classification", "IEC 62304 risk class determined (A/B/C)", "IEC 62304:2015"),
    ]
)

REGULATORY_CHECKS = _build_checks(
    [
        ("RG-001", "21 CFR Part 11", "Audit trails, access controls, e-signatures", "21 CFR Part 11"),
        ("RG-002", "HIPAA de-identification", "Safe Harbor or Expert Determination", "45 CFR 164.514"),
        ("RG-003", "GDPR DPIA", "Data Protection Impact Assessment for EU sites", "GDPR Article 35"),
        ("RG-004", "ICH E6(R2) GCP", "Good Clinical Practice guidelines followed", "ICH E6(R2) 2016"),
        ("RG-005", "IRB/Ethics approval", "IRB approval documented for each site", "21 CFR Part 56"),
        ("RG-006", "Informed consent", "E-consent records complete and version-matched", "ICH E6(R2) 2016"),
    ]
)

INFRASTRUCTURE_CHECKS = _build_checks(
    [
        ("IN-001", "Compute capacity", "GPU/CPU resources allocated for inference", "AWS Healthcare Lens 2023"),
        ("IN-002", "Network latency", "FL communication latency < 500 ms p99", "AWS Healthcare Lens 2023"),
        ("IN-003", "TLS encryption", "Data in transit encrypted with TLS 1.3", "NIST CSF v2.0"),
        ("IN-004", "Encryption at rest", "Data at rest encrypted with AES-256", "NIST CSF v2.0"),
        ("IN-005", "Monitoring", "Health checks and alerting operational", "NIST CSF v2.0"),
        ("IN-006", "Disaster recovery", "DR plan: RTO < 4h, RPO < 1h", "AWS Healthcare Lens 2023"),
        ("IN-007", "Backup verification", "Backups tested within last 30 days", "NIST CSF v2.0"),
    ]
)

DOCUMENTATION_CHECKS = _build_checks(
    [
        ("DC-001", "SOPs", "Training, validation, deployment SOPs", "ISO 13485:2016"),
        ("DC-002", "Training records", "Personnel training records current", "ISO 13485:2016"),
        ("DC-003", "Version control", "Model/data/code versions tracked", "FDA Design Control 1997"),
        ("DC-004", "Change control log", "Production changes documented", "ISO 13485:2016"),
        ("DC-005", "User documentation", "Instructions for use (IFU) approved", "FDA Design Control 1997"),
        ("DC-006", "Risk management file", "ISO 14971 risk management file", "ISO 14971:2019"),
    ]
)

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
    """Result of a single deployment readiness check (id, category, status, source)."""

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
    """Aggregated deployment readiness report with APPROVED/CONDITIONAL/BLOCKED decision."""

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


def build_checklist(categories: list[str] | None = None) -> list[ReadinessCheck]:
    """Build checklist of readiness checks. If *categories* is None, include all."""
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
    """Evaluate checks. Defaults to REQUIRES_VERIFICATION; use *overrides* to set specific statuses."""
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
    """Generate a deployment readiness report with APPROVED/CONDITIONAL/BLOCKED decision."""
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
    """Export a readiness report to a JSON file. Returns the output path."""
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
    """Entry-point for the deployment readiness CLI. Returns 0 on success, 1 on failure."""
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
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        logger.exception("Unexpected error (%s): %s", type(exc).__name__, exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
