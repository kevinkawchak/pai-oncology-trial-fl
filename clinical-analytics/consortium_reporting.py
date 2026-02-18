"""Consortium reporting — DSMB packages and multi-site audit reporting.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Generates Data Safety Monitoring Board (DSMB) interim and final analysis
packages, regulatory annual reports (IND annual report structure), cross-site
data quality comparisons, and sponsor-facing enrollment updates.  All report
artefacts carry a SHA-256 hash chain linking every audit-trail entry to its
predecessor, providing tamper-evident integrity for 21 CFR Part 11 compliance.

Key capabilities:
    * Blinded / unblinded package generation — the DSMB receives unblinded
      efficacy summaries while the sponsor sees only blinded safety data.
    * Cross-site enrollment and AE aggregation with guarded division and
      bounded percentages.
    * Markdown rendering for human review of any ConsortiumReport.
    * Hash-chain verification for the full audit trail.

DISCLAIMER: RESEARCH USE ONLY — This module has NOT been validated for
    clinical deployment.  Independent verification of all statistical
    summaries is required before use in any regulatory or patient-safety
    decision.
LICENSE: MIT
VERSION: 0.9.0
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_HMAC_KEY = b"pai-consortium-audit-integrity-key"  # 21 CFR Part 11


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportType(str, Enum):
    """Type of consortium report to generate."""
    DSMB_INTERIM = "dsmb_interim"
    DSMB_FINAL = "dsmb_final"
    REGULATORY_ANNUAL = "regulatory_annual"
    SAFETY_SIGNAL = "safety_signal"
    ENROLLMENT_STATUS = "enrollment_status"
    DATA_QUALITY = "data_quality"
    SPONSOR_UPDATE = "sponsor_update"


class DSMBRecommendation(str, Enum):
    """Possible DSMB recommendations following an interim analysis."""
    CONTINUE_UNCHANGED = "continue_unchanged"
    MODIFY_PROTOCOL = "modify_protocol"
    SUSPEND_ENROLLMENT = "suspend_enrollment"
    TERMINATE_EARLY = "terminate_early"
    REQUEST_ADDITIONAL_DATA = "request_additional_data"


class AuditSeverity(str, Enum):
    """Severity classification for audit-trail entries."""
    INFORMATIONAL = "informational"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class BlindingStatus(str, Enum):
    """Blinding context in which a report is generated."""
    BLINDED = "blinded"
    UNBLINDED_DSMB = "unblinded_dsmb"
    UNBLINDED_FULL = "unblinded_full"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SafetySummary:
    """Aggregated safety data for adverse-event reporting."""
    total_aes: int = 0
    serious_aes: int = 0
    deaths: int = 0
    ae_by_grade: dict[int, int] = field(default_factory=dict)
    ae_by_system_organ_class: dict[str, int] = field(default_factory=dict)


@dataclass
class EnrollmentSummary:
    """Cross-site enrollment snapshot."""
    target: int = 0
    enrolled: int = 0
    active: int = 0
    completed: int = 0
    withdrawn: int = 0
    by_site: dict[str, int] = field(default_factory=dict)


@dataclass
class AuditTrailEntry:
    """Single entry in the tamper-evident SHA-256 audit trail."""
    timestamp: str = ""
    action: str = ""
    user_id: str = ""
    details: str = ""
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dictionary."""
        return {
            "timestamp": self.timestamp, "action": self.action,
            "user_id": self.user_id, "details": self.details,
            "hash": self.hash,
        }


@dataclass
class DSMBPackage:
    """Full DSMB interim or final analysis package.

    Attributes:
        report_id: UUID identifying this package.
        trial_id: Parent clinical trial identifier.
        interim_number: Sequential interim analysis number (1-based).
        blinding_status: Blinded or unblinded designation.
        enrollment_summary: Aggregate enrollment data.
        safety_summary: Aggregate adverse-event data.
        efficacy_summary: Efficacy endpoints (arm -> metric -> value).
        recommendation: DSMB recommendation, if assigned.
        hash_chain: Final hash of the audit chain at generation time.
        generated_at: ISO-8601 UTC timestamp.
    """
    report_id: str = ""
    trial_id: str = ""
    interim_number: int = 0
    blinding_status: BlindingStatus = BlindingStatus.BLINDED
    enrollment_summary: EnrollmentSummary = field(default_factory=EnrollmentSummary)
    safety_summary: SafetySummary = field(default_factory=SafetySummary)
    efficacy_summary: dict[str, Any] = field(default_factory=dict)
    recommendation: DSMBRecommendation | None = None
    hash_chain: str = ""
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.report_id:
            self.report_id = str(uuid.uuid4())
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ConsortiumReport:
    """Generic consortium report with signature hash for integrity."""
    report_type: ReportType = ReportType.ENROLLMENT_STATUS
    sections: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    signature_hash: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division guarded against zero-denominator."""
    return default if denominator == 0 else numerator / denominator


def _bound_pct(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a percentage to *[low, high]*."""
    return max(low, min(high, value))


def _compute_hmac(payload: str) -> str:
    """Return hex-encoded HMAC-SHA256 of *payload*."""
    return hmac.new(_HMAC_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _chain_payload(prev_hash: str, entry: AuditTrailEntry) -> str:
    """Build the canonical payload string for hash-chain computation."""
    return f"{prev_hash}:{entry.timestamp}:{entry.action}:{entry.user_id}:{entry.details}"


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ConsortiumReportingEngine:
    """Generate consortium reports, DSMB packages, and audit artefacts.

    Maintains an internal audit trail with SHA-256 hash-chain integrity.
    Each mutating operation appends an audit entry whose hash links to
    its predecessor, providing 21 CFR Part 11 compliant tamper evidence.

    Args:
        trial_id: Identifier for the clinical trial.
        user_id: Default user identity recorded in audit entries.
    """

    def __init__(self, trial_id: str, user_id: str = "system") -> None:
        self.trial_id = trial_id
        self._user_id = user_id
        self._audit_trail: list[AuditTrailEntry] = []

        genesis = AuditTrailEntry(
            action="genesis", user_id=user_id,
            details=f"Audit chain initialised for trial {trial_id}",
        )
        genesis.hash = _compute_hmac(
            f"genesis:{genesis.timestamp}:{genesis.action}:"
            f"{genesis.user_id}:{genesis.details}"
        )
        self._audit_trail.append(genesis)
        logger.info("ConsortiumReportingEngine initialised for trial %s", trial_id)

    # ------------------------------------------------------------------
    # Audit-trail management
    # ------------------------------------------------------------------

    def _append_audit(
        self, action: str, details: str = "",
        user_id: str | None = None,
        severity: AuditSeverity = AuditSeverity.INFORMATIONAL,
    ) -> AuditTrailEntry:
        """Append an entry whose hash incorporates the previous entry's hash."""
        prev_hash = self._audit_trail[-1].hash if self._audit_trail else ""
        entry = AuditTrailEntry(
            action=action, user_id=user_id or self._user_id, details=details,
        )
        entry.hash = _compute_hmac(_chain_payload(prev_hash, entry))
        self._audit_trail.append(entry)

        if severity in (AuditSeverity.MAJOR, AuditSeverity.CRITICAL):
            logger.warning(
                "AUDIT [%s] %s — %s (user=%s)",
                severity.value.upper(), action, details, entry.user_id,
            )
        else:
            logger.debug("AUDIT [%s] %s — %s", severity.value, action, details)
        return entry

    def compute_hash_chain(self, entries: Sequence[AuditTrailEntry]) -> list[str]:
        """Re-derive the SHA-256 hash chain for *entries*.

        The first entry is treated as genesis; subsequent entries
        incorporate the previous hash.
        """
        if not entries:
            return []
        first = entries[0]
        hashes: list[str] = [_compute_hmac(
            f"genesis:{first.timestamp}:{first.action}:"
            f"{first.user_id}:{first.details}"
        )]
        for i in range(1, len(entries)):
            hashes.append(_compute_hmac(_chain_payload(hashes[i - 1], entries[i])))
        return hashes

    def verify_hash_chain(
        self, entries: Sequence[AuditTrailEntry] | None = None,
    ) -> tuple[bool, list[int]]:
        """Validate hash-chain integrity.

        Returns:
            ``(all_valid, list_of_failed_indices)``.
        """
        target = entries if entries is not None else self._audit_trail
        if not target:
            return (True, [])

        expected = self.compute_hash_chain(target)
        failed = [i for i, e in enumerate(target) if e.hash != expected[i]]

        if failed:
            logger.warning("Hash-chain verification FAILED at indices %s", failed)
            self._append_audit(
                action="hash_chain_verification_failed",
                details=f"Failed indices: {failed}",
                severity=AuditSeverity.CRITICAL,
            )
        else:
            logger.info("Hash-chain verification passed (%d entries)", len(target))
        return (len(failed) == 0, failed)

    def get_audit_trail(self) -> list[AuditTrailEntry]:
        """Return a defensive copy of the full audit trail."""
        return copy.deepcopy(self._audit_trail)

    # ------------------------------------------------------------------
    # Safety summary
    # ------------------------------------------------------------------

    def build_safety_summary(
        self, adverse_events: Sequence[dict[str, Any]],
    ) -> SafetySummary:
        """Aggregate adverse-event records into a :class:`SafetySummary`.

        Each AE dict should contain ``grade`` (int 1-5), ``serious``
        (bool), and ``system_organ_class`` (str).  Missing keys handled
        gracefully.
        """
        total = serious = deaths = 0
        by_grade: dict[int, int] = {}
        by_soc: dict[str, int] = {}

        for ae in adverse_events:
            total += 1
            grade = ae.get("grade", 0)
            grade = int(grade) if isinstance(grade, (int, float)) else 0
            grade = max(0, min(5, grade))
            by_grade[grade] = by_grade.get(grade, 0) + 1
            if ae.get("serious", False):
                serious += 1
            if grade == 5:
                deaths += 1
            soc = ae.get("system_organ_class", "Unknown")
            if not isinstance(soc, str) or not soc.strip():
                soc = "Unknown"
            by_soc[soc] = by_soc.get(soc, 0) + 1

        summary = SafetySummary(
            total_aes=total, serious_aes=serious, deaths=deaths,
            ae_by_grade=by_grade, ae_by_system_organ_class=by_soc,
        )
        self._append_audit(
            action="build_safety_summary",
            details=f"total_aes={total}, serious={serious}, deaths={deaths}",
        )
        logger.info("Safety summary: %d AEs, %d serious, %d deaths", total, serious, deaths)
        return summary

    # ------------------------------------------------------------------
    # Enrollment summary
    # ------------------------------------------------------------------

    def build_enrollment_summary(
        self, site_data: Sequence[dict[str, Any]],
    ) -> EnrollmentSummary:
        """Aggregate per-site enrollment into an :class:`EnrollmentSummary`.

        Each site dict should have ``site_id``, ``enrolled``, ``active``,
        ``completed``, ``withdrawn``, and optionally ``target``.
        """
        target = enrolled = active = completed = withdrawn = 0
        by_site: dict[str, int] = {}
        for site in site_data:
            sid = str(site.get("site_id", "unknown"))
            s_enrolled = int(site.get("enrolled", 0))
            enrolled += s_enrolled
            active += int(site.get("active", 0))
            completed += int(site.get("completed", 0))
            withdrawn += int(site.get("withdrawn", 0))
            by_site[sid] = s_enrolled
            st = int(site.get("target", 0))
            if st > 0:
                target += st

        summary = EnrollmentSummary(
            target=target, enrolled=enrolled, active=active,
            completed=completed, withdrawn=withdrawn, by_site=by_site,
        )
        self._append_audit(
            action="build_enrollment_summary",
            details=f"enrolled={enrolled}/{target}, active={active}, sites={len(by_site)}",
        )
        logger.info("Enrollment: %d/%d across %d sites", enrolled, target, len(by_site))
        return summary

    # ------------------------------------------------------------------
    # DSMB package generation
    # ------------------------------------------------------------------

    def generate_dsmb_package(
        self, trial_data: dict[str, Any], interim_number: int,
        blinding: BlindingStatus = BlindingStatus.UNBLINDED_DSMB,
    ) -> DSMBPackage:
        """Generate a full DSMB interim-analysis package.

        Args:
            trial_data: Dict with ``adverse_events``, ``site_data``, and
                ``efficacy`` (arm name -> endpoint metrics dict).
            interim_number: Sequential interim analysis number.
            blinding: Blinding level for this package.
        """
        self._append_audit(
            action="dsmb_package_start",
            details=f"interim={interim_number}, blinding={blinding.value}",
            severity=AuditSeverity.MAJOR,
        )

        safety = self.build_safety_summary(trial_data.get("adverse_events", []))
        enrollment = self.build_enrollment_summary(trial_data.get("site_data", []))

        raw_eff: dict[str, Any] = trial_data.get("efficacy", {})
        if blinding == BlindingStatus.BLINDED:
            efficacy_out: dict[str, Any] = {"status": "blinded"}
        elif blinding == BlindingStatus.UNBLINDED_DSMB:
            efficacy_out = self._prepare_efficacy_summary(raw_eff)
        else:
            efficacy_out = copy.deepcopy(raw_eff)

        chain_hash = self._audit_trail[-1].hash if self._audit_trail else ""
        package = DSMBPackage(
            trial_id=self.trial_id, interim_number=interim_number,
            blinding_status=blinding, enrollment_summary=enrollment,
            safety_summary=safety, efficacy_summary=efficacy_out,
            hash_chain=chain_hash,
        )

        self._append_audit(
            action="dsmb_package_generated",
            details=f"report_id={package.report_id}, interim={interim_number}",
            severity=AuditSeverity.MAJOR,
        )
        logger.info(
            "DSMB package %s generated (trial %s, interim %d)",
            package.report_id, self.trial_id, interim_number,
        )
        return package

    def _prepare_efficacy_summary(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Derive per-arm summary statistics (mean, median, response rate)."""
        summary: dict[str, Any] = {}
        for arm, endpoints in raw.items():
            if not isinstance(endpoints, dict):
                summary[arm] = endpoints
                continue
            arm_stats: dict[str, Any] = {}
            for ep, values in endpoints.items():
                if not isinstance(values, (list, np.ndarray)):
                    arm_stats[ep] = values
                    continue
                arr = np.asarray(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                n = len(arr)
                if n == 0:
                    arm_stats[ep] = {"n": 0, "mean": None, "median": None, "response_rate": None}
                else:
                    resp = int(np.sum(arr > 0))
                    arm_stats[ep] = {
                        "n": n, "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "response_rate": _bound_pct(_safe_div(resp, n) * 100),
                    }
            summary[arm] = arm_stats
        return summary

    # ------------------------------------------------------------------
    # Regulatory annual report
    # ------------------------------------------------------------------

    def generate_regulatory_annual_report(
        self, trial_data: dict[str, Any],
    ) -> ConsortiumReport:
        """Generate an IND annual report structure (21 CFR 312.33).

        Produces sections: trial status, enrollment, safety/AE profile,
        protocol amendments, and audit-trail integrity.
        """
        self._append_audit(
            action="regulatory_annual_start", details=f"trial={self.trial_id}",
            severity=AuditSeverity.MAJOR,
        )

        safety = self.build_safety_summary(trial_data.get("adverse_events", []))
        enrollment = self.build_enrollment_summary(trial_data.get("site_data", []))

        amendment_section = [
            {"amendment_number": am.get("number", "N/A"),
             "date": am.get("date", "N/A"), "summary": am.get("summary", "")}
            for am in trial_data.get("protocol_amendments", [])
        ]

        pct_enrolled = _bound_pct(_safe_div(enrollment.enrolled, enrollment.target) * 100)

        sections: dict[str, Any] = {
            "trial_status": {
                "trial_id": self.trial_id,
                "reporting_period": trial_data.get("reporting_period", "N/A"),
                "overall_status": trial_data.get("overall_status", "ongoing"),
                "enrollment_pct": round(pct_enrolled, 1),
            },
            "enrollment": {
                "target": enrollment.target, "enrolled": enrollment.enrolled,
                "active": enrollment.active, "completed": enrollment.completed,
                "withdrawn": enrollment.withdrawn, "by_site": enrollment.by_site,
            },
            "safety": {
                "total_aes": safety.total_aes, "serious_aes": safety.serious_aes,
                "deaths": safety.deaths, "ae_by_grade": safety.ae_by_grade,
                "ae_by_system_organ_class": safety.ae_by_system_organ_class,
            },
            "protocol_amendments": amendment_section,
            "audit_integrity": self._audit_integrity_section(),
        }

        report = self._build_consortium_report(ReportType.REGULATORY_ANNUAL, sections)
        self._append_audit(
            action="regulatory_annual_generated",
            details=f"signature={report.signature_hash[:16]}",
            severity=AuditSeverity.MAJOR,
        )
        logger.info("Regulatory annual report generated for trial %s", self.trial_id)
        return report

    # ------------------------------------------------------------------
    # Data quality report
    # ------------------------------------------------------------------

    def generate_data_quality_report(
        self, site_metrics: Sequence[dict[str, Any]],
    ) -> ConsortiumReport:
        """Cross-site data-quality comparison report.

        Ranks sites by composite quality score and flags outliers (> 2 SD
        from the mean) on query_rate, missing_data_pct, or
        protocol_deviations.
        """
        self._append_audit(action="data_quality_start", details=f"sites={len(site_metrics)}")

        if not site_metrics:
            return self._build_consortium_report(
                ReportType.DATA_QUALITY, {"sites": [], "overall": {}, "outliers": []},
            )

        site_ids: list[str] = []
        query_rates: list[float] = []
        missing_pcts: list[float] = []
        deviations: list[int] = []
        for sm in site_metrics:
            site_ids.append(str(sm.get("site_id", "unknown")))
            query_rates.append(float(sm.get("query_rate", 0.0)))
            missing_pcts.append(_bound_pct(float(sm.get("missing_data_pct", 0.0))))
            deviations.append(int(sm.get("protocol_deviations", 0)))

        qr_arr = np.array(query_rates, dtype=float)
        mp_arr = np.array(missing_pcts, dtype=float)
        dv_arr = np.array(deviations, dtype=float)
        qr_mean, qr_std = float(np.mean(qr_arr)), float(np.std(qr_arr))
        mp_mean, mp_std = float(np.mean(mp_arr)), float(np.std(mp_arr))
        dv_mean, dv_std = float(np.mean(dv_arr)), float(np.std(dv_arr))

        site_details: list[dict[str, Any]] = []
        outliers: list[dict[str, Any]] = []

        for i, sid in enumerate(site_ids):
            qr_z = _safe_div(query_rates[i] - qr_mean, qr_std) if qr_std > 0 else 0.0
            mp_z = _safe_div(missing_pcts[i] - mp_mean, mp_std) if mp_std > 0 else 0.0
            dv_z = _safe_div(deviations[i] - dv_mean, dv_std) if dv_std > 0 else 0.0
            avg_abs_z = (abs(qr_z) + abs(mp_z) + abs(dv_z)) / 3.0
            quality_score = _bound_pct(100.0 - avg_abs_z * 20.0)

            site_details.append({
                "site_id": sid,
                "query_rate": round(query_rates[i], 2),
                "missing_data_pct": round(missing_pcts[i], 2),
                "protocol_deviations": deviations[i],
                "monitoring_visits": int(site_metrics[i].get("monitoring_visits", 0)),
                "overdue_queries": int(site_metrics[i].get("overdue_queries", 0)),
                "quality_score": round(quality_score, 1),
            })

            flagged: list[str] = []
            if abs(qr_z) > 2.0:
                flagged.append("query_rate")
            if abs(mp_z) > 2.0:
                flagged.append("missing_data_pct")
            if abs(dv_z) > 2.0:
                flagged.append("protocol_deviations")
            if flagged:
                outliers.append({
                    "site_id": sid, "flagged_metrics": flagged,
                    "quality_score": round(quality_score, 1),
                })

        site_details.sort(key=lambda d: d["quality_score"], reverse=True)

        sections: dict[str, Any] = {
            "sites": site_details,
            "overall": {
                "num_sites": len(site_ids),
                "mean_query_rate": round(qr_mean, 2),
                "mean_missing_data_pct": round(mp_mean, 2),
                "mean_deviations": round(dv_mean, 2),
                "num_outlier_sites": len(outliers),
            },
            "outliers": outliers,
        }
        report = self._build_consortium_report(ReportType.DATA_QUALITY, sections)
        self._append_audit(
            action="data_quality_generated",
            details=f"sites={len(site_ids)}, outliers={len(outliers)}, sig={report.signature_hash[:16]}",
        )
        logger.info("Data-quality report: %d sites, %d outliers", len(site_ids), len(outliers))
        return report

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def format_report_markdown(self, report: ConsortiumReport) -> str:
        """Render a :class:`ConsortiumReport` as Markdown."""
        lines: list[str] = []
        title = report.report_type.value.replace("_", " ").title()
        lines.append(f"# {title} Report")
        lines.append("")
        lines.append(f"**Trial:** {self.trial_id}  ")
        lines.append(f"**Generated:** {report.generated_at}  ")
        lines.append(f"**Signature:** `{report.signature_hash[:16]}...`")
        lines.append("")
        lines.append("---")
        lines.append("")
        for sec_name, sec_content in report.sections.items():
            lines.append(f"## {sec_name.replace('_', ' ').title()}")
            lines.append("")
            self._render_section_md(lines, sec_content, depth=0)
            lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated automatically. RESEARCH USE ONLY.*")
        return "\n".join(lines)

    def _render_section_md(self, lines: list[str], content: Any, depth: int) -> None:
        """Recursively render section content as Markdown bullet lists."""
        indent = "  " * depth
        if isinstance(content, dict):
            for key, val in content.items():
                label = str(key).replace("_", " ").title()
                if isinstance(val, (dict, list)):
                    lines.append(f"{indent}- **{label}:**")
                    self._render_section_md(lines, val, depth + 1)
                else:
                    lines.append(f"{indent}- **{label}:** {val}")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    first_val = next(iter(item.values()), "")
                    lines.append(f"{indent}- *{first_val}*")
                    self._render_section_md(lines, item, depth + 1)
                else:
                    lines.append(f"{indent}- {item}")
        else:
            lines.append(f"{indent}{content}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_consortium_report(
        self, report_type: ReportType, sections: dict[str, Any],
    ) -> ConsortiumReport:
        """Create a :class:`ConsortiumReport` with integrity signature."""
        report = ConsortiumReport(report_type=report_type, sections=sections)
        blob = json.dumps(sections, sort_keys=True, default=str)
        report.signature_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return report

    def _audit_integrity_section(self) -> dict[str, Any]:
        """Produce an audit-integrity summary for inclusion in reports."""
        is_valid, failed = self.verify_hash_chain()
        return {
            "chain_length": len(self._audit_trail),
            "integrity_valid": is_valid,
            "failed_indices": failed,
            "latest_hash": self._audit_trail[-1].hash if self._audit_trail else "",
        }
