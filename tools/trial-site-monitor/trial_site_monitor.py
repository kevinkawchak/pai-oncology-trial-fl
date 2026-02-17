"""Multi-site clinical trial monitoring for federated oncology studies.

Monitors enrollment rates, data quality, protocol compliance, and FL
infrastructure health across distributed clinical trial sites.

DISCLAIMER: RESEARCH USE ONLY — Not approved for regulatory submission.
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

# Reference thresholds. Sources: ICH E6(R2) 2016; FDA Oversight Guidance 2013.

# Enrollment rate thresholds (patients per month).
ENROLLMENT_RATE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "green": (2.0, 1000.0),
    "yellow": (0.5, 2.0),
    "red": (0.0, 0.5),
}

# Data quality thresholds (percentage of clean records).
DATA_QUALITY_THRESHOLDS: dict[str, tuple[float, float]] = {
    "green": (95.0, 100.0),
    "yellow": (80.0, 95.0),
    "red": (0.0, 80.0),
}

# Protocol deviation thresholds (deviations per 100 subjects).
PROTOCOL_DEVIATION_THRESHOLDS: dict[str, tuple[float, float]] = {
    "green": (0.0, 5.0),
    "yellow": (5.0, 15.0),
    "red": (15.0, 100.0),
}

# Query resolution time thresholds (days).
# Source: TransCelerate BioPharma — Risk-Based Monitoring, 2013.
QUERY_RESOLUTION_DAYS: dict[str, tuple[float, float]] = {
    "green": (0.0, 7.0),
    "yellow": (7.0, 21.0),
    "red": (21.0, 365.0),
}

# Maximum allowed sites per trial (practical bound).
MAX_SITES_PER_TRIAL: int = 500

# Bounded metric ranges to prevent invalid inputs.
ENROLLMENT_RATE_BOUNDS: tuple[float, float] = (0.0, 100.0)
DATA_QUALITY_PCT_BOUNDS: tuple[float, float] = (0.0, 100.0)
SCREEN_FAIL_RATE_BOUNDS: tuple[float, float] = (0.0, 100.0)


# ── Enums ──────────────────────────────────────────────────────────────────


class SiteStatus(str, Enum):
    """Overall operational status of a clinical trial site.

    Colour-coded to align with ICH E6(R2) risk-based monitoring
    recommendations.
    """

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    INACTIVE = "inactive"
    PENDING_ACTIVATION = "pending_activation"


class EnrollmentStatus(str, Enum):
    """Enrollment performance classification for a trial site.

    Based on monthly accrual rate relative to the site's target.
    Source: NCI CTSU enrollment benchmarking guidance, 2022.
    """

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    NOT_STARTED = "not_started"
    COMPLETED = "completed"


class DataQualityStatus(str, Enum):
    """Data quality classification for site-submitted data.

    Reflects the proportion of case report forms (CRFs) free of
    queries after initial entry.
    Source: SCDM Good Clinical Data Management Practices, 2021.
    """

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    INSUFFICIENT_DATA = "insufficient_data"


class AlertLevel(str, Enum):
    """Alert severity for monitoring notifications.

    Follows a standard three-tier alerting model recommended by
    TransCelerate risk-based monitoring guidance.
    """

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class SiteMetrics:
    """Quantitative metrics for a single trial site.

    All numeric fields are bounded to prevent erroneous calculations.
    Enrollment 0-100/mo, quality 0-100%, deviations 0-100/100 subjects,
    query days 0-365, convergence 0-1.
    """

    site_id: str = ""
    site_name: str = ""
    principal_investigator: str = ""
    enrollment_rate_per_month: float = 0.0
    enrolled_count: int = 0
    target_enrollment: int = 0
    screen_fail_rate_pct: float = 0.0
    data_quality_pct: float = 100.0
    protocol_deviations_per_100: float = 0.0
    query_resolution_days: float = 0.0
    last_updated: str = ""
    fl_rounds_completed: int = 0
    model_convergence_score: float = 0.0

    def __post_init__(self) -> None:
        """Clamp all numeric values to safe bounds."""
        self.enrollment_rate_per_month = max(
            ENROLLMENT_RATE_BOUNDS[0], min(self.enrollment_rate_per_month, ENROLLMENT_RATE_BOUNDS[1])
        )
        self.enrolled_count = max(0, self.enrolled_count)
        self.target_enrollment = max(0, self.target_enrollment)
        self.screen_fail_rate_pct = max(
            SCREEN_FAIL_RATE_BOUNDS[0], min(self.screen_fail_rate_pct, SCREEN_FAIL_RATE_BOUNDS[1])
        )
        self.data_quality_pct = max(DATA_QUALITY_PCT_BOUNDS[0], min(self.data_quality_pct, DATA_QUALITY_PCT_BOUNDS[1]))
        self.protocol_deviations_per_100 = max(0.0, min(self.protocol_deviations_per_100, 100.0))
        self.query_resolution_days = max(0.0, min(self.query_resolution_days, 365.0))
        self.model_convergence_score = max(0.0, min(self.model_convergence_score, 1.0))
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()


@dataclass
class SiteAlert:
    """An alert generated during site monitoring (site_id, level, category, message)."""

    site_id: str = ""
    level: str = AlertLevel.INFO.value
    category: str = ""
    message: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class MonitoringReport:
    """Aggregated monitoring report with per-site GREEN/YELLOW/RED status and alerts."""

    trial_id: str = ""
    report_date: str = ""
    total_sites: int = 0
    sites_green: int = 0
    sites_yellow: int = 0
    sites_red: int = 0
    total_enrolled: int = 0
    overall_enrollment_pct: float = 0.0
    mean_data_quality_pct: float = 0.0
    alerts: list[dict[str, str]] = field(default_factory=list)
    site_details: list[dict[str, Any]] = field(default_factory=list)


# ── Core monitoring helpers ────────────────────────────────────────────────


def _classify(value: float, thresholds: dict[str, tuple[float, float]]) -> str:
    """Classify a numeric value as green/yellow/red given threshold bands."""
    for level in ("green", "yellow", "red"):
        lo, hi = thresholds[level]
        if lo <= value <= hi:
            return level
    return "red"


def classify_enrollment(rate_per_month: float) -> EnrollmentStatus:
    """Classify enrollment performance (rate bounded 0-100 patients/month)."""
    rate_per_month = max(ENROLLMENT_RATE_BOUNDS[0], min(rate_per_month, ENROLLMENT_RATE_BOUNDS[1]))
    level = _classify(rate_per_month, ENROLLMENT_RATE_THRESHOLDS)
    return EnrollmentStatus(level)


def classify_data_quality(quality_pct: float) -> DataQualityStatus:
    """Classify data quality (percentage of clean CRFs, bounded 0-100)."""
    quality_pct = max(DATA_QUALITY_PCT_BOUNDS[0], min(quality_pct, DATA_QUALITY_PCT_BOUNDS[1]))
    level = _classify(quality_pct, DATA_QUALITY_THRESHOLDS)
    return DataQualityStatus(level)


def evaluate_site(metrics: SiteMetrics) -> tuple[SiteStatus, list[SiteAlert]]:
    """Evaluate a single site and return (overall SiteStatus, list of alerts)."""
    alerts: list[SiteAlert] = []
    sid = metrics.site_id
    crit, warn = AlertLevel.CRITICAL.value, AlertLevel.WARNING.value
    enr = classify_enrollment(metrics.enrollment_rate_per_month)
    dq = classify_data_quality(metrics.data_quality_pct)
    dev = _classify(metrics.protocol_deviations_per_100, PROTOCOL_DEVIATION_THRESHOLDS)
    qry = _classify(metrics.query_resolution_days, QUERY_RESOLUTION_DAYS)

    rate = metrics.enrollment_rate_per_month
    if enr == EnrollmentStatus.RED:
        alerts.append(SiteAlert(sid, crit, "enrollment", f"Rate {rate:.1f}/mo critically low."))
    elif enr == EnrollmentStatus.YELLOW:
        alerts.append(SiteAlert(sid, warn, "enrollment", f"Rate {rate:.1f}/mo below target."))
    qual = metrics.data_quality_pct
    if dq == DataQualityStatus.RED:
        alerts.append(SiteAlert(sid, crit, "data_quality", f"Quality {qual:.1f}% critically low."))
    elif dq == DataQualityStatus.YELLOW:
        alerts.append(SiteAlert(sid, warn, "data_quality", f"Quality {qual:.1f}% needs improvement."))
    if dev == "red":
        devn = metrics.protocol_deviations_per_100
        alerts.append(SiteAlert(sid, crit, "protocol_compliance", f"Deviations ({devn:.1f}/100) exceed threshold."))
    if qry == "red":
        qd = metrics.query_resolution_days
        alerts.append(SiteAlert(sid, warn, "query_resolution", f"Resolution time ({qd:.0f}d) exceeds threshold."))

    statuses = [enr.value, dq.value, dev, qry]
    if "red" in statuses:
        overall = SiteStatus.RED
    elif "yellow" in statuses:
        overall = SiteStatus.YELLOW
    else:
        overall = SiteStatus.GREEN
    logger.info("Site %s evaluated: %s (%d alerts)", sid, overall.value, len(alerts))
    return overall, alerts


def generate_monitoring_report(
    trial_id: str,
    sites: list[SiteMetrics],
    total_target: int = 0,
) -> MonitoringReport:
    """Generate a cross-site monitoring report. Target bounded 0-10000."""
    total_target = max(0, min(total_target, 10000))
    report = MonitoringReport(
        trial_id=trial_id,
        report_date=datetime.now(timezone.utc).isoformat(),
        total_sites=len(sites),
    )

    quality_sum = 0.0
    for site in sites:
        status, alerts = evaluate_site(site)
        report.total_enrolled += site.enrolled_count
        quality_sum += site.data_quality_pct

        if status == SiteStatus.GREEN:
            report.sites_green += 1
        elif status == SiteStatus.YELLOW:
            report.sites_yellow += 1
        elif status == SiteStatus.RED:
            report.sites_red += 1

        for alert in alerts:
            report.alerts.append(asdict(alert))

        report.site_details.append(
            {
                "site_id": site.site_id,
                "site_name": site.site_name,
                "status": status.value,
                "enrolled": site.enrolled_count,
                "target": site.target_enrollment,
                "data_quality_pct": site.data_quality_pct,
                "fl_rounds": site.fl_rounds_completed,
            }
        )

    if total_target > 0:
        report.overall_enrollment_pct = round(min((report.total_enrolled / total_target) * 100.0, 100.0), 1)
    if report.total_sites > 0:
        report.mean_data_quality_pct = round(quality_sum / report.total_sites, 1)

    return report


def _build_demo_sites() -> list[SiteMetrics]:
    """Build a small set of demo sites for CLI demonstration."""
    return [
        SiteMetrics(
            site_id="SITE-001",
            site_name="Memorial Research Hospital",
            principal_investigator="Dr. A. Smith",
            enrollment_rate_per_month=3.5,
            enrolled_count=42,
            target_enrollment=60,
            data_quality_pct=97.2,
            protocol_deviations_per_100=2.1,
            query_resolution_days=4.0,
            fl_rounds_completed=15,
            model_convergence_score=0.89,
        ),
        SiteMetrics(
            site_id="SITE-002",
            site_name="University Cancer Center",
            principal_investigator="Dr. B. Jones",
            enrollment_rate_per_month=1.2,
            enrolled_count=18,
            target_enrollment=50,
            data_quality_pct=82.5,
            protocol_deviations_per_100=8.3,
            query_resolution_days=14.0,
            fl_rounds_completed=10,
            model_convergence_score=0.74,
        ),
        SiteMetrics(
            site_id="SITE-003",
            site_name="Regional Oncology Institute",
            principal_investigator="Dr. C. Lee",
            enrollment_rate_per_month=0.3,
            enrolled_count=5,
            target_enrollment=40,
            data_quality_pct=68.0,
            protocol_deviations_per_100=18.5,
            query_resolution_days=28.0,
            fl_rounds_completed=3,
            model_convergence_score=0.45,
        ),
    ]


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="trial_site_monitor",
        description=("Multi-site clinical trial monitoring tool for federated oncology studies. RESEARCH USE ONLY."),
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

    # status ---
    sp_status = subparsers.add_parser("status", help="Show status for a specific site (demo data).")
    sp_status.add_argument("--site-id", default="SITE-001", help="Site identifier (default: SITE-001).")

    # report ---
    sp_report = subparsers.add_parser("report", help="Generate a cross-site monitoring report (demo data).")
    sp_report.add_argument("--trial-id", default="NCT00000000", help="Clinical trial NCT number.")
    sp_report.add_argument("--target", type=int, default=150, help="Overall enrollment target.")

    # alerts ---
    sp_alerts = subparsers.add_parser("alerts", help="List active alerts across all sites (demo data).")
    sp_alerts.add_argument(
        "--min-level",
        choices=["info", "warning", "critical"],
        default="info",
        help="Minimum alert level to display (default: info).",
    )

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
    """Entry-point for the trial site monitor CLI. Returns 0 on success, 1 on failure."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.command is None:
        parser.print_help()
        return 0

    demo_sites = _build_demo_sites()

    try:
        if args.command == "status":
            site = next((s for s in demo_sites if s.site_id == args.site_id), None)
            if site is None:
                logger.error("Site '%s' not found. Available: %s", args.site_id, [s.site_id for s in demo_sites])
                return 1
            status, alerts = evaluate_site(site)
            output_data = {
                "site": asdict(site),
                "overall_status": status.value,
                "alerts": [asdict(a) for a in alerts],
            }
            if args.json:
                print(json.dumps(output_data, indent=2, default=str))
            else:
                print(f"  Site: {site.site_id} — {site.site_name}")
                print(f"  Overall status: {status.value.upper()}")
                print(f"  Enrollment: {site.enrolled_count}/{site.target_enrollment}")
                print(f"  Data quality: {site.data_quality_pct:.1f}%")
                print(f"  FL rounds: {site.fl_rounds_completed}")
                for alert in alerts:
                    print(f"  [{alert.level.upper()}] {alert.message}")
            return 0

        if args.command == "report":
            report = generate_monitoring_report(
                trial_id=args.trial_id,
                sites=demo_sites,
                total_target=args.target,
            )
            _output(report, args.json)
            return 0

        if args.command == "alerts":
            level_priority = {"info": 0, "warning": 1, "critical": 2}
            min_priority = level_priority.get(args.min_level, 0)

            all_alerts: list[SiteAlert] = []
            for site in demo_sites:
                _, alerts = evaluate_site(site)
                all_alerts.extend(alerts)

            filtered = [a for a in all_alerts if level_priority.get(a.level, 0) >= min_priority]
            filtered.sort(key=lambda a: level_priority.get(a.level, 0), reverse=True)

            if args.json:
                print(json.dumps([asdict(a) for a in filtered], indent=2, default=str))
            else:
                if not filtered:
                    print("  No alerts matching the specified criteria.")
                for alert in filtered:
                    print(f"  [{alert.level.upper()}] {alert.site_id}: {alert.message}")
            return 0
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
