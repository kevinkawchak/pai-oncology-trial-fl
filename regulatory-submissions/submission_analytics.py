#!/usr/bin/env python3
"""Submission analytics engine -- KPIs, benchmarks, and trend analysis.

RESEARCH USE ONLY -- Not intended for clinical deployment without validation.

This module implements analytics on regulatory submission quality, timelines,
and outcomes.  It computes submission KPIs (cycle time, deficiency rates,
first-pass yield), identifies deficiency patterns, benchmarks against FDA
MDUFA performance goals, generates improvement recommendations, and performs
trend analysis across submission portfolios.

All analytics are computed in-memory using numpy and scipy.  No external
data sources or network calls.

Features:
    * Submission cycle time analysis
    * Deficiency rate computation and pattern detection
    * First-pass yield tracking
    * FDA MDUFA benchmark comparison
    * Trend analysis with direction detection
    * Portfolio-level performance dashboards
    * Improvement recommendation engine

DISCLAIMER: RESEARCH USE ONLY.  Not validated for regulatory use.
LICENSE: MIT
VERSION: 0.9.1
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SUBMISSIONS: int = 10000
MAX_DEFICIENCIES: int = 50000
MAX_METRICS: int = 1000
MIN_SAMPLE_SIZE: int = 2
DEFAULT_CONFIDENCE: float = 0.95


# ===================================================================
# Enums
# ===================================================================
class MetricType(str, Enum):
    """Types of submission performance metrics."""

    CYCLE_TIME = "cycle_time"
    DEFICIENCY_RATE = "deficiency_rate"
    FIRST_PASS_YIELD = "first_pass_yield"
    REVIEW_DURATION = "review_duration"
    COMPLIANCE_SCORE = "compliance_score"


class TrendDirection(str, Enum):
    """Direction of a performance trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class BenchmarkSource(str, Enum):
    """Source of benchmark data."""

    INTERNAL = "internal"
    FDA_MDUFA = "fda_mdufa"
    INDUSTRY_PUBLISHED = "industry_published"


class DeficiencyCategory(str, Enum):
    """Categories of regulatory deficiencies."""

    CLINICAL_DATA = "clinical_data"
    LABELING = "labeling"
    SOFTWARE = "software"
    BIOCOMPATIBILITY = "biocompatibility"
    RISK_ANALYSIS = "risk_analysis"
    PERFORMANCE_TESTING = "performance_testing"
    CYBERSECURITY = "cybersecurity"
    MANUFACTURING = "manufacturing"
    STERILIZATION = "sterilization"
    ADMINISTRATIVE = "administrative"


class SubmissionOutcome(str, Enum):
    """Outcomes of a regulatory submission."""

    CLEARED = "cleared"
    APPROVED = "approved"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    ADDITIONAL_INFO_REQUESTED = "additional_info_requested"


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class SubmissionRecord:
    """Historical record of a regulatory submission."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = ""
    submission_type: str = "510k"
    device_name: str = ""
    submitted_date: str = ""
    decision_date: str = ""
    outcome: SubmissionOutcome = SubmissionOutcome.PENDING
    cycle_time_days: int = 0
    deficiency_count: int = 0
    review_cycles: int = 1
    compliance_score: float = 0.0
    product_code: str = ""
    reviewer_division: str = ""
    is_first_pass: bool = False

    def compute_cycle_time(self) -> int:
        """Compute cycle time in days from submitted to decision date."""
        if not self.submitted_date or not self.decision_date:
            return 0
        try:
            start = datetime.fromisoformat(self.submitted_date)
            end = datetime.fromisoformat(self.decision_date)
            self.cycle_time_days = max(0, (end - start).days)
        except (ValueError, TypeError):
            self.cycle_time_days = 0
        return self.cycle_time_days


@dataclass
class SubmissionMetric:
    """A computed performance metric."""

    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_type: MetricType = MetricType.CYCLE_TIME
    value: float = 0.0
    unit: str = ""
    period_start: str = ""
    period_end: str = ""
    sample_size: int = 0
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    trend: TrendDirection = TrendDirection.STABLE


@dataclass
class TimelineAnalysis:
    """Analysis of submission timelines."""

    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    total_submissions: int = 0
    mean_cycle_time: float = 0.0
    median_cycle_time: float = 0.0
    std_cycle_time: float = 0.0
    min_cycle_time: float = 0.0
    max_cycle_time: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_90: float = 0.0
    on_time_rate: float = 0.0
    target_days: int = 90


@dataclass
class DeficiencyPattern:
    """A detected pattern in submission deficiencies."""

    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: DeficiencyCategory = DeficiencyCategory.CLINICAL_DATA
    frequency: int = 0
    percentage: float = 0.0
    trend: TrendDirection = TrendDirection.STABLE
    common_findings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    recurrence_rate: float = 0.0


@dataclass
class DeficiencyRecord:
    """A single deficiency record."""

    deficiency_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = ""
    category: DeficiencyCategory = DeficiencyCategory.CLINICAL_DATA
    description: str = ""
    date_identified: str = ""
    date_resolved: str = ""
    is_resolved: bool = False
    severity: str = "major"


@dataclass
class PerformanceBenchmark:
    """Benchmark comparison result."""

    benchmark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_type: MetricType = MetricType.CYCLE_TIME
    internal_value: float = 0.0
    benchmark_value: float = 0.0
    benchmark_source: BenchmarkSource = BenchmarkSource.FDA_MDUFA
    delta: float = 0.0
    delta_percentage: float = 0.0
    assessment: str = ""

    def compute_delta(self) -> None:
        """Compute difference between internal and benchmark values."""
        self.delta = round(self.internal_value - self.benchmark_value, 4)
        if self.benchmark_value != 0:
            self.delta_percentage = round(self.delta / self.benchmark_value * 100.0, 2)


# ===================================================================
# MDUFA Benchmarks
# ===================================================================
_MDUFA_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "510k": {
        "cycle_time": 108.0,
        "deficiency_rate": 35.0,
        "first_pass_yield": 65.0,
        "review_duration": 90.0,
    },
    "de_novo": {
        "cycle_time": 150.0,
        "deficiency_rate": 45.0,
        "first_pass_yield": 55.0,
        "review_duration": 120.0,
    },
    "pma": {
        "cycle_time": 180.0,
        "deficiency_rate": 50.0,
        "first_pass_yield": 50.0,
        "review_duration": 150.0,
    },
}


# ===================================================================
# Main class
# ===================================================================
class SubmissionAnalyticsEngine:
    """Analytics engine for regulatory submission performance.

    Computes submission KPIs, identifies deficiency patterns, benchmarks
    against FDA MDUFA goals, and generates improvement recommendations.

    Parameters
    ----------
    portfolio_name : str
        Name of the submission portfolio being analyzed.
    """

    def __init__(self, portfolio_name: str = "default") -> None:
        self._portfolio_name = portfolio_name
        self._submissions: Dict[str, SubmissionRecord] = {}
        self._deficiencies: Dict[str, DeficiencyRecord] = {}
        self._metrics_history: List[SubmissionMetric] = []
        self._rng = np.random.default_rng(42)
        logger.info("SubmissionAnalyticsEngine initialized for portfolio: %s", portfolio_name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def portfolio_name(self) -> str:
        return self._portfolio_name

    @property
    def submission_count(self) -> int:
        return len(self._submissions)

    @property
    def deficiency_count(self) -> int:
        return len(self._deficiencies)

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    def add_submission(self, record: SubmissionRecord) -> None:
        """Add a submission record to the portfolio."""
        if len(self._submissions) >= MAX_SUBMISSIONS:
            logger.warning("Maximum submission capacity reached")
            return
        record.compute_cycle_time()
        record.is_first_pass = record.review_cycles <= 1 and record.deficiency_count == 0
        self._submissions[record.record_id] = record
        logger.debug("Submission added: %s (%s)", record.submission_id, record.outcome.value)

    def add_deficiency(self, deficiency: DeficiencyRecord) -> None:
        """Add a deficiency record."""
        if len(self._deficiencies) >= MAX_DEFICIENCIES:
            return
        self._deficiencies[deficiency.deficiency_id] = deficiency

    def generate_synthetic_portfolio(self, n_submissions: int = 50, seed: int = 42) -> None:
        """Generate a synthetic portfolio of submission records for analysis."""
        rng = np.random.default_rng(seed)
        n_submissions = max(1, min(n_submissions, MAX_SUBMISSIONS))

        sub_types = ["510k", "de_novo", "pma"]
        outcomes = [
            SubmissionOutcome.CLEARED,
            SubmissionOutcome.APPROVED,
            SubmissionOutcome.ADDITIONAL_INFO_REQUESTED,
            SubmissionOutcome.DENIED,
        ]
        outcome_weights = [0.50, 0.20, 0.25, 0.05]

        for i in range(n_submissions):
            sub_type = sub_types[int(rng.integers(0, len(sub_types)))]
            base_cycle = _MDUFA_BENCHMARKS.get(sub_type, {}).get("cycle_time", 108.0)
            cycle_time = max(30, int(rng.normal(base_cycle, base_cycle * 0.3)))
            deficiency_count = max(0, int(rng.poisson(3)))
            review_cycles = 1 + (1 if deficiency_count > 2 else 0)
            outcome_idx = int(rng.choice(len(outcomes), p=outcome_weights))
            outcome = outcomes[outcome_idx]
            compliance_score = float(max(50.0, min(100.0, rng.normal(82.0, 10.0))))

            start_date = datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 730)))
            decision_date = start_date + timedelta(days=cycle_time)

            record = SubmissionRecord(
                submission_id=f"SUB-{start_date.year}-{i + 1:04d}",
                submission_type=sub_type,
                device_name=f"Device-{i + 1:03d}",
                submitted_date=start_date.strftime("%Y-%m-%d"),
                decision_date=decision_date.strftime("%Y-%m-%d"),
                outcome=outcome,
                cycle_time_days=cycle_time,
                deficiency_count=deficiency_count,
                review_cycles=review_cycles,
                compliance_score=round(compliance_score, 2),
            )
            self.add_submission(record)

            categories = list(DeficiencyCategory)
            for _ in range(deficiency_count):
                cat = categories[int(rng.integers(0, len(categories)))]
                deficiency = DeficiencyRecord(
                    submission_id=record.submission_id,
                    category=cat,
                    description=f"Deficiency in {cat.value}",
                    date_identified=start_date.strftime("%Y-%m-%d"),
                    is_resolved=outcome != SubmissionOutcome.DENIED,
                )
                self.add_deficiency(deficiency)

        logger.info(
            "Generated synthetic portfolio: %d submissions, %d deficiencies",
            len(self._submissions),
            len(self._deficiencies),
        )

    # ------------------------------------------------------------------
    # Cycle time analysis
    # ------------------------------------------------------------------
    def analyze_cycle_times(self, submission_type: str = "") -> TimelineAnalysis:
        """Analyze submission cycle times."""
        records = list(self._submissions.values())
        if submission_type:
            records = [r for r in records if r.submission_type == submission_type]

        if not records:
            return TimelineAnalysis()

        cycle_times = np.array([r.cycle_time_days for r in records if r.cycle_time_days > 0], dtype=np.float64)
        if len(cycle_times) == 0:
            return TimelineAnalysis(total_submissions=len(records))

        target = _MDUFA_BENCHMARKS.get(submission_type, {}).get("cycle_time", 108.0)
        on_time = float(np.sum(cycle_times <= target) / len(cycle_times) * 100.0)

        analysis = TimelineAnalysis(
            total_submissions=len(records),
            mean_cycle_time=round(float(np.mean(cycle_times)), 2),
            median_cycle_time=round(float(np.median(cycle_times)), 2),
            std_cycle_time=round(float(np.std(cycle_times, ddof=1)) if len(cycle_times) > 1 else 0.0, 2),
            min_cycle_time=round(float(np.min(cycle_times)), 2),
            max_cycle_time=round(float(np.max(cycle_times)), 2),
            percentile_25=round(float(np.percentile(cycle_times, 25)), 2),
            percentile_75=round(float(np.percentile(cycle_times, 75)), 2),
            percentile_90=round(float(np.percentile(cycle_times, 90)), 2),
            on_time_rate=round(on_time, 2),
            target_days=int(target),
        )
        return analysis

    # ------------------------------------------------------------------
    # Deficiency analysis
    # ------------------------------------------------------------------
    def analyze_deficiency_patterns(self) -> List[DeficiencyPattern]:
        """Identify patterns in submission deficiencies."""
        if not self._deficiencies:
            return []

        category_counts: Dict[DeficiencyCategory, int] = {}
        for d in self._deficiencies.values():
            category_counts[d.category] = category_counts.get(d.category, 0) + 1

        total = len(self._deficiencies)
        patterns = []
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            pct = round(count / total * 100.0, 2)
            resolved = sum(1 for d in self._deficiencies.values() if d.category == cat and d.is_resolved)
            recurrence = round(1.0 - (resolved / count) if count > 0 else 0.0, 4)
            pattern = DeficiencyPattern(
                category=cat,
                frequency=count,
                percentage=pct,
                recurrence_rate=recurrence,
                common_findings=[f"Common issue in {cat.value}"],
                recommended_actions=[f"Improve {cat.value} review process"],
            )
            if pct > 15:
                pattern.trend = TrendDirection.DECLINING
            elif pct < 5:
                pattern.trend = TrendDirection.IMPROVING
            patterns.append(pattern)

        return patterns

    def compute_deficiency_rate(self, submission_type: str = "") -> float:
        """Compute overall deficiency rate (percentage of submissions with deficiencies)."""
        records = list(self._submissions.values())
        if submission_type:
            records = [r for r in records if r.submission_type == submission_type]
        if not records:
            return 0.0
        with_deficiencies = sum(1 for r in records if r.deficiency_count > 0)
        return round(with_deficiencies / len(records) * 100.0, 2)

    # ------------------------------------------------------------------
    # First-pass yield
    # ------------------------------------------------------------------
    def compute_first_pass_yield(self, submission_type: str = "") -> float:
        """Compute first-pass yield (submissions cleared without deficiencies)."""
        records = list(self._submissions.values())
        if submission_type:
            records = [r for r in records if r.submission_type == submission_type]
        if not records:
            return 0.0
        first_pass = sum(1 for r in records if r.is_first_pass)
        return round(first_pass / len(records) * 100.0, 2)

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    def benchmark_against_mdufa(self, submission_type: str = "510k") -> List[PerformanceBenchmark]:
        """Benchmark submission performance against MDUFA goals."""
        benchmarks_data = _MDUFA_BENCHMARKS.get(submission_type, _MDUFA_BENCHMARKS["510k"])
        results: List[PerformanceBenchmark] = []

        # Cycle time
        timeline = self.analyze_cycle_times(submission_type)
        if timeline.mean_cycle_time > 0:
            b = PerformanceBenchmark(
                metric_type=MetricType.CYCLE_TIME,
                internal_value=timeline.mean_cycle_time,
                benchmark_value=benchmarks_data["cycle_time"],
                benchmark_source=BenchmarkSource.FDA_MDUFA,
            )
            b.compute_delta()
            b.assessment = "Better than MDUFA" if b.delta < 0 else "Exceeds MDUFA target"
            results.append(b)

        # Deficiency rate
        def_rate = self.compute_deficiency_rate(submission_type)
        b2 = PerformanceBenchmark(
            metric_type=MetricType.DEFICIENCY_RATE,
            internal_value=def_rate,
            benchmark_value=benchmarks_data["deficiency_rate"],
            benchmark_source=BenchmarkSource.FDA_MDUFA,
        )
        b2.compute_delta()
        b2.assessment = "Better than MDUFA" if b2.delta < 0 else "Higher than MDUFA benchmark"
        results.append(b2)

        # First-pass yield
        fpy = self.compute_first_pass_yield(submission_type)
        b3 = PerformanceBenchmark(
            metric_type=MetricType.FIRST_PASS_YIELD,
            internal_value=fpy,
            benchmark_value=benchmarks_data["first_pass_yield"],
            benchmark_source=BenchmarkSource.FDA_MDUFA,
        )
        b3.compute_delta()
        b3.assessment = "Exceeds MDUFA benchmark" if b3.delta > 0 else "Below MDUFA benchmark"
        results.append(b3)

        return results

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------
    def analyze_trend(self, metric_type: MetricType, window_size: int = 10) -> Tuple[TrendDirection, float]:
        """Analyze the trend direction of a metric over time.

        Uses linear regression slope to determine trend.

        Parameters
        ----------
        metric_type : MetricType
            Metric to analyze.
        window_size : int
            Number of recent data points to consider.

        Returns
        -------
        tuple of (TrendDirection, float)
            Trend direction and slope magnitude.
        """
        records = sorted(self._submissions.values(), key=lambda r: r.submitted_date)
        if len(records) < MIN_SAMPLE_SIZE:
            return TrendDirection.STABLE, 0.0

        records = records[-min(window_size, len(records)) :]

        if metric_type == MetricType.CYCLE_TIME:
            values = np.array([r.cycle_time_days for r in records], dtype=np.float64)
        elif metric_type == MetricType.COMPLIANCE_SCORE:
            values = np.array([r.compliance_score for r in records], dtype=np.float64)
        elif metric_type == MetricType.DEFICIENCY_RATE:
            values = np.array([r.deficiency_count for r in records], dtype=np.float64)
        else:
            return TrendDirection.STABLE, 0.0

        if len(values) < MIN_SAMPLE_SIZE:
            return TrendDirection.STABLE, 0.0

        x = np.arange(len(values), dtype=np.float64)
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, values)

        threshold = 0.5
        if abs(slope) < threshold or p_value > 0.1:
            direction = TrendDirection.STABLE
        elif metric_type in (MetricType.CYCLE_TIME, MetricType.DEFICIENCY_RATE):
            direction = TrendDirection.IMPROVING if slope < 0 else TrendDirection.DECLINING
        else:
            direction = TrendDirection.IMPROVING if slope > 0 else TrendDirection.DECLINING

        return direction, round(float(slope), 4)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on analytics."""
        recommendations: List[str] = []

        patterns = self.analyze_deficiency_patterns()
        top_categories = [p for p in patterns if p.percentage > 10]
        for pat in top_categories[:3]:
            recommendations.append(
                f"Address high-frequency deficiency category: {pat.category.value} "
                f"({pat.percentage:.1f}% of all deficiencies)"
            )

        for sub_type in ["510k", "de_novo", "pma"]:
            timeline = self.analyze_cycle_times(sub_type)
            benchmark = _MDUFA_BENCHMARKS.get(sub_type, {}).get("cycle_time", 108.0)
            if timeline.mean_cycle_time > benchmark * 1.1:
                recommendations.append(
                    f"Reduce {sub_type} cycle time: current mean {timeline.mean_cycle_time:.0f} days "
                    f"exceeds MDUFA target of {benchmark:.0f} days"
                )

        fpy = self.compute_first_pass_yield()
        if fpy < 60.0:
            recommendations.append(f"Improve first-pass yield: current {fpy:.1f}% is below industry benchmark of 65%")

        cycle_trend, slope = self.analyze_trend(MetricType.CYCLE_TIME)
        if cycle_trend == TrendDirection.DECLINING:
            recommendations.append(
                f"Cycle time trend is worsening (slope={slope:.2f} days/submission). Investigate root causes."
            )

        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges. Continue monitoring.")

        return recommendations

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_analytics_report(self) -> str:
        """Generate a comprehensive Markdown analytics report."""
        lines: List[str] = [
            f"# Submission Analytics Report: {self._portfolio_name}",
            "",
            f"**Total Submissions:** {len(self._submissions)}",
            f"**Total Deficiencies:** {len(self._deficiencies)}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Cycle Time Analysis",
            "",
        ]

        for sub_type in ["510k", "de_novo", "pma"]:
            timeline = self.analyze_cycle_times(sub_type)
            if timeline.total_submissions > 0:
                lines.append(f"### {sub_type}")
                lines.append(f"- Mean: {timeline.mean_cycle_time:.1f} days")
                lines.append(f"- Median: {timeline.median_cycle_time:.1f} days")
                lines.append(f"- Std Dev: {timeline.std_cycle_time:.1f} days")
                lines.append(f"- On-time rate: {timeline.on_time_rate:.1f}%")
                lines.append(
                    f"- P25/P75/P90: {timeline.percentile_25:.0f}/{timeline.percentile_75:.0f}/{timeline.percentile_90:.0f}"
                )
                lines.append("")

        # Deficiency patterns
        lines.extend(["## Deficiency Patterns", ""])
        patterns = self.analyze_deficiency_patterns()
        if patterns:
            lines.append("| Category | Count | Percentage | Trend |")
            lines.append("|----------|-------|------------|-------|")
            for pat in patterns[:10]:
                lines.append(f"| {pat.category.value} | {pat.frequency} | {pat.percentage:.1f}% | {pat.trend.value} |")
            lines.append("")

        # Benchmarks
        lines.extend(["## MDUFA Benchmarks (510(k))", ""])
        benchmarks = self.benchmark_against_mdufa("510k")
        if benchmarks:
            lines.append("| Metric | Internal | MDUFA | Delta | Assessment |")
            lines.append("|--------|----------|-------|-------|------------|")
            for b in benchmarks:
                lines.append(
                    f"| {b.metric_type.value} | {b.internal_value:.1f} | "
                    f"{b.benchmark_value:.1f} | {b.delta:+.1f} | {b.assessment} |"
                )
            lines.append("")

        # Recommendations
        lines.extend(["## Recommendations", ""])
        for rec in self.generate_recommendations():
            lines.append(f"- {rec}")

        lines.extend(["", f"*Report generated: {datetime.now().isoformat()[:19]}*"])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state to a dictionary."""
        return {
            "portfolio_name": self._portfolio_name,
            "submission_count": len(self._submissions),
            "deficiency_count": len(self._deficiencies),
            "metrics_history_count": len(self._metrics_history),
        }


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = SubmissionAnalyticsEngine("PAI Oncology Portfolio")
    engine.generate_synthetic_portfolio(50, seed=42)
    print(engine.generate_analytics_report())
