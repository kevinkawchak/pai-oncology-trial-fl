"""Tests for tools/trial-site-monitor/trial_site_monitor.py.

Covers enums, dataclasses (SiteMetrics, SiteAlert, MonitoringReport),
classification helpers, site evaluation, and report generation.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module("trial_site_monitor", "tools/trial-site-monitor/trial_site_monitor.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestSiteStatus:
    """Tests for the SiteStatus enum."""

    def test_members(self):
        """SiteStatus has five members."""
        expected = {"GREEN", "YELLOW", "RED", "INACTIVE", "PENDING_ACTIVATION"}
        assert set(mod.SiteStatus.__members__.keys()) == expected

    def test_values(self):
        """SiteStatus values are lowercase strings."""
        assert mod.SiteStatus.GREEN.value == "green"
        assert mod.SiteStatus.PENDING_ACTIVATION.value == "pending_activation"


class TestEnrollmentStatus:
    """Tests for the EnrollmentStatus enum."""

    def test_members(self):
        """EnrollmentStatus has five members."""
        assert len(mod.EnrollmentStatus) == 5
        assert mod.EnrollmentStatus.NOT_STARTED.value == "not_started"


class TestDataQualityStatus:
    """Tests for the DataQualityStatus enum."""

    def test_members(self):
        """DataQualityStatus has four members."""
        assert len(mod.DataQualityStatus) == 4
        assert mod.DataQualityStatus.INSUFFICIENT_DATA.value == "insufficient_data"


class TestAlertLevel:
    """Tests for the AlertLevel enum."""

    def test_members(self):
        """AlertLevel has three members."""
        expected = {"CRITICAL", "WARNING", "INFO"}
        assert set(mod.AlertLevel.__members__.keys()) == expected


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestSiteMetrics:
    """Tests for the SiteMetrics dataclass."""

    def test_defaults(self):
        """Default SiteMetrics has zero enrollment."""
        sm = mod.SiteMetrics()
        assert sm.enrollment_rate_per_month == 0.0
        assert sm.enrolled_count == 0
        assert sm.data_quality_pct == 100.0

    def test_clamping_enrollment(self):
        """Enrollment rate is clamped to [0, 100]."""
        sm = mod.SiteMetrics(enrollment_rate_per_month=200.0)
        assert sm.enrollment_rate_per_month == 100.0

    def test_clamping_quality(self):
        """Data quality is clamped to [0, 100]."""
        sm = mod.SiteMetrics(data_quality_pct=-10.0)
        assert sm.data_quality_pct == 0.0
        sm2 = mod.SiteMetrics(data_quality_pct=150.0)
        assert sm2.data_quality_pct == 100.0

    def test_clamping_convergence(self):
        """Model convergence score is clamped to [0, 1]."""
        sm = mod.SiteMetrics(model_convergence_score=2.0)
        assert sm.model_convergence_score == 1.0

    def test_last_updated_auto(self):
        """last_updated is set automatically if empty."""
        sm = mod.SiteMetrics()
        assert sm.last_updated != ""

    def test_query_resolution_clamped(self):
        """Query resolution days clamped to [0, 365]."""
        sm = mod.SiteMetrics(query_resolution_days=500.0)
        assert sm.query_resolution_days == 365.0


class TestSiteAlert:
    """Tests for the SiteAlert dataclass."""

    def test_defaults(self):
        """Default SiteAlert has INFO level."""
        sa = mod.SiteAlert()
        assert sa.level == mod.AlertLevel.INFO.value

    def test_timestamp_auto(self):
        """Timestamp is set automatically."""
        sa = mod.SiteAlert(site_id="SITE-001", category="enrollment")
        assert sa.timestamp != ""


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
class TestClassifyEnrollment:
    """Tests for classify_enrollment."""

    def test_green(self):
        """High enrollment rate is GREEN."""
        result = mod.classify_enrollment(5.0)
        assert result == mod.EnrollmentStatus.GREEN

    def test_yellow(self):
        """Moderate enrollment rate is YELLOW."""
        result = mod.classify_enrollment(1.0)
        assert result == mod.EnrollmentStatus.YELLOW

    def test_red(self):
        """Low enrollment rate is RED."""
        result = mod.classify_enrollment(0.2)
        assert result == mod.EnrollmentStatus.RED

    def test_boundary_green_yellow(self):
        """Rate of 2.0 is at the GREEN boundary."""
        result = mod.classify_enrollment(2.0)
        assert result == mod.EnrollmentStatus.GREEN


class TestClassifyDataQuality:
    """Tests for classify_data_quality."""

    def test_green(self):
        """High quality is GREEN."""
        result = mod.classify_data_quality(98.0)
        assert result == mod.DataQualityStatus.GREEN

    def test_yellow(self):
        """Moderate quality is YELLOW."""
        result = mod.classify_data_quality(90.0)
        assert result == mod.DataQualityStatus.YELLOW

    def test_red(self):
        """Low quality is RED."""
        result = mod.classify_data_quality(70.0)
        assert result == mod.DataQualityStatus.RED


# ---------------------------------------------------------------------------
# Site evaluation
# ---------------------------------------------------------------------------
class TestEvaluateSite:
    """Tests for evaluate_site."""

    def test_green_site(self):
        """Good metrics yield GREEN status with no alerts."""
        metrics = mod.SiteMetrics(
            site_id="SITE-A",
            enrollment_rate_per_month=5.0,
            data_quality_pct=98.0,
            protocol_deviations_per_100=2.0,
            query_resolution_days=3.0,
        )
        status, alerts = mod.evaluate_site(metrics)
        assert status == mod.SiteStatus.GREEN
        assert len(alerts) == 0

    def test_red_site(self):
        """Bad metrics yield RED status with alerts."""
        metrics = mod.SiteMetrics(
            site_id="SITE-B",
            enrollment_rate_per_month=0.1,
            data_quality_pct=60.0,
            protocol_deviations_per_100=20.0,
            query_resolution_days=30.0,
        )
        status, alerts = mod.evaluate_site(metrics)
        assert status == mod.SiteStatus.RED
        assert len(alerts) >= 3

    def test_yellow_site(self):
        """Mixed metrics yield YELLOW status."""
        metrics = mod.SiteMetrics(
            site_id="SITE-C",
            enrollment_rate_per_month=1.0,
            data_quality_pct=92.0,
            protocol_deviations_per_100=3.0,
            query_resolution_days=5.0,
        )
        status, alerts = mod.evaluate_site(metrics)
        assert status == mod.SiteStatus.YELLOW

    def test_alert_categories(self):
        """Alerts include the correct categories."""
        metrics = mod.SiteMetrics(
            site_id="SITE-D",
            enrollment_rate_per_month=0.1,
            data_quality_pct=60.0,
        )
        _, alerts = mod.evaluate_site(metrics)
        categories = {a.category for a in alerts}
        assert "enrollment" in categories
        assert "data_quality" in categories


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
class TestGenerateMonitoringReport:
    """Tests for generate_monitoring_report."""

    def test_basic_report(self):
        """Report aggregates site-level data."""
        sites = [
            mod.SiteMetrics(
                site_id="S1",
                enrollment_rate_per_month=5.0,
                enrolled_count=40,
                data_quality_pct=98.0,
                protocol_deviations_per_100=1.0,
                query_resolution_days=2.0,
            ),
            mod.SiteMetrics(
                site_id="S2",
                enrollment_rate_per_month=0.2,
                enrolled_count=5,
                data_quality_pct=65.0,
                protocol_deviations_per_100=20.0,
                query_resolution_days=30.0,
            ),
        ]
        report = mod.generate_monitoring_report("NCT-001", sites, total_target=100)
        assert report.total_sites == 2
        assert report.total_enrolled == 45
        assert report.sites_green >= 1
        assert report.sites_red >= 1
        assert report.overall_enrollment_pct == 45.0

    def test_empty_sites(self):
        """Report with no sites has zero totals."""
        report = mod.generate_monitoring_report("NCT-002", [], total_target=100)
        assert report.total_sites == 0
        assert report.total_enrolled == 0
        assert report.mean_data_quality_pct == 0.0

    def test_enrollment_pct_capped(self):
        """Enrollment percentage is capped at 100%."""
        sites = [
            mod.SiteMetrics(
                site_id="S1",
                enrollment_rate_per_month=5.0,
                enrolled_count=200,
                data_quality_pct=98.0,
                protocol_deviations_per_100=1.0,
                query_resolution_days=2.0,
            ),
        ]
        report = mod.generate_monitoring_report("NCT-003", sites, total_target=50)
        assert report.overall_enrollment_pct == 100.0

    def test_zero_target_no_division_error(self):
        """Zero target enrollment does not cause division by zero."""
        sites = [mod.SiteMetrics(site_id="S1", enrolled_count=10)]
        report = mod.generate_monitoring_report("NCT-004", sites, total_target=0)
        assert report.overall_enrollment_pct == 0.0
