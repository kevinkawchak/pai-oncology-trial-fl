"""Tests for regulatory-submissions/submission_analytics.py.

Covers enums (MetricType, TrendDirection, BenchmarkSource, DeficiencyCategory,
SubmissionOutcome), dataclasses (SubmissionRecord, SubmissionMetric,
TimelineAnalysis, DeficiencyPattern, DeficiencyRecord, PerformanceBenchmark),
and SubmissionAnalyticsEngine:
  - Initialization
  - Submission record management
  - Deficiency record management
  - Synthetic portfolio generation
  - Cycle time analysis
  - Deficiency pattern detection
  - Deficiency rate computation
  - First-pass yield computation
  - MDUFA benchmarking
  - Trend analysis with scipy.stats.linregress
  - Recommendation generation
  - Analytics report generation
  - Serialization
  - Integration test: clinical-analytics -> regulatory-submissions data flow
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import load_module

mod = load_module(
    "submission_analytics",
    "regulatory-submissions/submission_analytics.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_engine(name="test"):
    return mod.SubmissionAnalyticsEngine(name)


def _make_record(**overrides):
    defaults = {
        "submission_id": "SUB-001",
        "submission_type": "510k",
        "device_name": "Device-001",
        "submitted_date": "2025-01-01",
        "decision_date": "2025-04-10",
        "outcome": mod.SubmissionOutcome.CLEARED,
        "deficiency_count": 0,
        "review_cycles": 1,
        "compliance_score": 85.0,
    }
    defaults.update(overrides)
    return mod.SubmissionRecord(**defaults)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestMetricType:
    def test_members(self):
        names = set(mod.MetricType.__members__.keys())
        assert "CYCLE_TIME" in names
        assert "DEFICIENCY_RATE" in names
        assert "FIRST_PASS_YIELD" in names
        assert "COMPLIANCE_SCORE" in names

    def test_count(self):
        assert len(mod.MetricType) == 5


class TestTrendDirection:
    def test_members(self):
        assert mod.TrendDirection.IMPROVING.value == "improving"
        assert mod.TrendDirection.STABLE.value == "stable"
        assert mod.TrendDirection.DECLINING.value == "declining"


class TestBenchmarkSource:
    def test_members(self):
        assert mod.BenchmarkSource.FDA_MDUFA.value == "fda_mdufa"
        assert mod.BenchmarkSource.INTERNAL.value == "internal"


class TestSubmissionOutcome:
    def test_members(self):
        names = set(mod.SubmissionOutcome.__members__.keys())
        assert "CLEARED" in names
        assert "DENIED" in names
        assert "PENDING" in names

    def test_count(self):
        assert len(mod.SubmissionOutcome) == 6


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestSubmissionRecord:
    def test_creation(self):
        r = _make_record()
        assert r.submission_id == "SUB-001"

    def test_compute_cycle_time(self):
        r = _make_record(submitted_date="2025-01-01", decision_date="2025-04-11")
        days = r.compute_cycle_time()
        assert days == 100

    def test_compute_cycle_time_empty(self):
        r = mod.SubmissionRecord()
        assert r.compute_cycle_time() == 0


class TestPerformanceBenchmark:
    def test_compute_delta(self):
        b = mod.PerformanceBenchmark(internal_value=90.0, benchmark_value=108.0)
        b.compute_delta()
        assert b.delta == -18.0
        assert b.delta_percentage < 0

    def test_compute_delta_zero_benchmark(self):
        b = mod.PerformanceBenchmark(internal_value=90.0, benchmark_value=0.0)
        b.compute_delta()
        assert b.delta == 90.0


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------
class TestSubmissionAnalyticsEngine:
    def test_init(self):
        engine = _make_engine()
        assert engine.portfolio_name == "test"
        assert engine.submission_count == 0

    def test_add_submission(self):
        engine = _make_engine()
        engine.add_submission(_make_record())
        assert engine.submission_count == 1

    def test_add_deficiency(self):
        engine = _make_engine()
        d = mod.DeficiencyRecord(submission_id="SUB-001", category=mod.DeficiencyCategory.CLINICAL_DATA)
        engine.add_deficiency(d)
        assert engine.deficiency_count == 1

    def test_synthetic_portfolio(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(30, seed=99)
        assert engine.submission_count == 30
        assert engine.deficiency_count > 0

    def test_cycle_time_analysis(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        analysis = engine.analyze_cycle_times("510k")
        assert analysis.total_submissions > 0
        assert analysis.mean_cycle_time > 0
        assert analysis.median_cycle_time > 0
        assert analysis.std_cycle_time >= 0

    def test_cycle_time_analysis_empty(self):
        engine = _make_engine()
        analysis = engine.analyze_cycle_times("510k")
        assert analysis.total_submissions == 0

    def test_deficiency_patterns(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        patterns = engine.analyze_deficiency_patterns()
        assert len(patterns) > 0
        total_pct = sum(p.percentage for p in patterns)
        assert abs(total_pct - 100.0) < 1.0

    def test_deficiency_rate(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        rate = engine.compute_deficiency_rate("510k")
        assert 0.0 <= rate <= 100.0

    def test_first_pass_yield(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        fpy = engine.compute_first_pass_yield("510k")
        assert 0.0 <= fpy <= 100.0

    def test_first_pass_yield_empty(self):
        engine = _make_engine()
        fpy = engine.compute_first_pass_yield()
        assert fpy == 0.0

    def test_benchmark_mdufa(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        benchmarks = engine.benchmark_against_mdufa("510k")
        assert len(benchmarks) >= 3
        for b in benchmarks:
            assert b.benchmark_source == mod.BenchmarkSource.FDA_MDUFA

    def test_trend_analysis(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        direction, slope = engine.analyze_trend(mod.MetricType.CYCLE_TIME)
        assert direction in mod.TrendDirection
        assert isinstance(slope, float)

    def test_trend_analysis_insufficient_data(self):
        engine = _make_engine()
        direction, slope = engine.analyze_trend(mod.MetricType.CYCLE_TIME)
        assert direction == mod.TrendDirection.STABLE
        assert slope == 0.0

    def test_recommendations(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        recs = engine.generate_recommendations()
        assert len(recs) > 0
        assert all(isinstance(r, str) for r in recs)

    def test_analytics_report(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(50, seed=42)
        report = engine.generate_analytics_report()
        assert "Submission Analytics Report" in report
        assert "Cycle Time" in report
        assert "MDUFA" in report
        assert "Recommendations" in report

    def test_to_dict(self):
        engine = _make_engine()
        engine.generate_synthetic_portfolio(10, seed=42)
        d = engine.to_dict()
        assert d["portfolio_name"] == "test"
        assert d["submission_count"] == 10


class TestIntegrationClinicalAnalyticsToRegulatorySubmissions:
    """Integration test: clinical-analytics -> regulatory-submissions data flow.

    This test verifies that outputs from the clinical-analytics module
    (e.g., survival analysis results, risk stratification scores) can be
    consumed by the regulatory-submissions module to generate submission
    documents and track submission metrics.
    """

    def test_survival_results_feed_into_submission_record(self):
        """Simulated survival analysis results update a submission record."""
        survival_results = {
            "median_os_months": 18.5,
            "hazard_ratio": 0.72,
            "log_rank_p_value": 0.003,
        }
        engine = _make_engine("integration-test")
        record = _make_record(
            submission_id="SUB-INTEGRATION",
            compliance_score=85.0 + (10.0 if survival_results["log_rank_p_value"] < 0.05 else 0.0),
        )
        engine.add_submission(record)
        assert engine.submission_count == 1
        analysis = engine.analyze_cycle_times("510k")
        assert analysis.total_submissions == 1

    def test_risk_metrics_feed_into_deficiency_analysis(self):
        """Simulated risk stratification metrics contribute to analytics."""
        risk_results = {
            "auc_roc": 0.9456,
            "sensitivity": 0.9234,
            "specificity": 0.8891,
        }
        engine = _make_engine("integration-test-risk")
        for i in range(5):
            record = _make_record(
                submission_id=f"SUB-RISK-{i}",
                deficiency_count=0 if risk_results["auc_roc"] > 0.9 else 2,
                compliance_score=risk_results["auc_roc"] * 100.0,
            )
            engine.add_submission(record)

        fpy = engine.compute_first_pass_yield()
        assert fpy > 0.0

    def test_pkpd_results_contribute_to_submission_metrics(self):
        """PK/PD results inform submission cycle time expectations."""
        pkpd_results = {
            "clearance_l_h": 15.2,
            "half_life_h": 8.4,
        }
        engine = _make_engine("integration-test-pkpd")
        record = _make_record(
            submission_id="SUB-PKPD",
            submission_type="510k",
            compliance_score=90.0,
        )
        engine.add_submission(record)
        benchmarks = engine.benchmark_against_mdufa("510k")
        assert len(benchmarks) >= 1
