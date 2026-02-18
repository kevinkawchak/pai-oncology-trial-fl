"""Tests for clinical-analytics/analytics_orchestrator.py.

Covers enums (AnalyticsPhase, PipelineStatus, AnalyticsStrategy),
dataclasses (AnalyticsConfig, SiteContribution, AggregatedResult,
PipelineRun, QualityMetrics), and the ClinicalAnalyticsOrchestrator
class: initialization, pipeline phases, audit trail (defensive copy),
convergence checking, contribution aggregation, and division-by-zero
guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root so ``tests.conftest`` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import load_module

mod = load_module(
    "analytics_orchestrator",
    "clinical-analytics/analytics_orchestrator.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestAnalyticsPhase:
    """Tests for the AnalyticsPhase enum."""

    def test_all_phases_exist(self):
        """All six pipeline phases are defined."""
        expected = {
            "DATA_COLLECTION", "PREPROCESSING", "LOCAL_COMPUTATION",
            "SECURE_AGGREGATION", "GLOBAL_INFERENCE", "REPORTING",
        }
        assert set(mod.AnalyticsPhase.__members__.keys()) == expected

    def test_phase_values_are_strings(self):
        """AnalyticsPhase values are lowercase strings."""
        assert mod.AnalyticsPhase.DATA_COLLECTION.value == "data_collection"
        assert mod.AnalyticsPhase.REPORTING.value == "reporting"

    def test_phase_is_str_enum(self):
        """AnalyticsPhase members are instances of str."""
        assert isinstance(mod.AnalyticsPhase.REPORTING, str)


class TestPipelineStatus:
    """Tests for the PipelineStatus enum."""

    def test_all_statuses_exist(self):
        """All eight statuses are defined."""
        expected = {
            "PENDING", "CONFIGURING", "VALIDATING", "RUNNING",
            "AGGREGATING", "COMPLETED", "FAILED", "CANCELLED",
        }
        assert set(mod.PipelineStatus.__members__.keys()) == expected

    def test_status_values(self):
        """PipelineStatus values are correct strings."""
        assert mod.PipelineStatus.PENDING.value == "pending"
        assert mod.PipelineStatus.COMPLETED.value == "completed"


class TestAnalyticsStrategy:
    """Tests for the AnalyticsStrategy enum."""

    def test_strategy_members(self):
        """AnalyticsStrategy has expected members."""
        names = set(mod.AnalyticsStrategy.__members__.keys())
        assert "DESCRIPTIVE" in names
        assert "INFERENTIAL" in names
        assert "PREDICTIVE" in names
        assert "SURVIVAL" in names
        assert "BAYESIAN" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestAnalyticsConfig:
    """Tests for the AnalyticsConfig dataclass."""

    def test_default_creation(self):
        """Default AnalyticsConfig has sensible values."""
        cfg = mod.AnalyticsConfig()
        assert cfg.convergence_tolerance > 0
        assert cfg.max_iterations >= 1
        assert cfg.privacy_budget_epsilon > 0

    def test_pipeline_id_auto_generated(self):
        """Default config auto-generates a pipeline_id."""
        cfg = mod.AnalyticsConfig()
        assert cfg.pipeline_id.startswith("PIPE-")

    def test_convergence_tolerance_clamped(self):
        """Convergence tolerance is clamped to [1e-10, 1.0]."""
        cfg = mod.AnalyticsConfig(convergence_tolerance=-1.0)
        assert cfg.convergence_tolerance >= 1e-10
        cfg2 = mod.AnalyticsConfig(convergence_tolerance=100.0)
        assert cfg2.convergence_tolerance <= 1.0

    def test_privacy_budget_epsilon_clamped(self):
        """Privacy budget epsilon is clamped to [0.01, 100.0]."""
        cfg = mod.AnalyticsConfig(privacy_budget_epsilon=-5.0)
        assert cfg.privacy_budget_epsilon >= 0.01
        cfg2 = mod.AnalyticsConfig(privacy_budget_epsilon=500.0)
        assert cfg2.privacy_budget_epsilon <= 100.0


class TestSiteContribution:
    """Tests for the SiteContribution dataclass."""

    def test_default_creation(self):
        """SiteContribution is created with default values."""
        sc = mod.SiteContribution(site_id="SITE-A", n_subjects=50)
        assert sc.site_id == "SITE-A"
        assert sc.n_subjects == 50
        assert sc.computed_at != ""
        assert sc.checksum != ""

    def test_checksum_verification(self):
        """SiteContribution checksum verifies successfully."""
        sc = mod.SiteContribution(
            site_id="SITE-B",
            n_subjects=30,
            local_mean={"os": 12.5},
            local_sum={"os": 375.0},
        )
        assert sc.verify_checksum() is True

    def test_checksum_tamper_detection(self):
        """Modified contribution fails checksum verification."""
        sc = mod.SiteContribution(
            site_id="SITE-C",
            n_subjects=25,
            local_mean={"os": 10.0},
        )
        sc.n_subjects = 999  # tamper
        assert sc.verify_checksum() is False


# ---------------------------------------------------------------------------
# ClinicalAnalyticsOrchestrator tests
# ---------------------------------------------------------------------------
class TestClinicalAnalyticsOrchestrator:
    """Tests for ClinicalAnalyticsOrchestrator lifecycle and methods."""

    def test_init_with_config(self):
        """Orchestrator initializes with provided config."""
        cfg = mod.AnalyticsConfig(
            pipeline_id="P-001",
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        assert orch.status == mod.PipelineStatus.PENDING

    def test_config_defensive_copy(self):
        """Orchestrator config property returns a defensive copy."""
        cfg = mod.AnalyticsConfig(
            pipeline_id="P-001",
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        copy = orch.config
        copy.trial_id = "MODIFIED"
        assert orch.config.trial_id == "NCT-001"

    def test_initialize_pipeline_success(self):
        """Pipeline initialization succeeds with valid config."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2", "S3"],
            min_sites_required=2,
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        assert orch.initialize_pipeline() is True
        assert orch.status == mod.PipelineStatus.VALIDATING

    def test_initialize_pipeline_insufficient_sites(self):
        """Pipeline initialization fails with too few sites."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1"],
            min_sites_required=3,
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        assert orch.initialize_pipeline() is False
        assert orch.status == mod.PipelineStatus.FAILED

    def test_audit_trail_populated_after_init(self):
        """Audit trail has entries after initialization."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        orch.initialize_pipeline()
        trail = orch.get_audit_trail()
        assert len(trail) >= 1

    def test_audit_trail_defensive_copy(self):
        """get_audit_trail returns a defensive copy."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        orch.initialize_pipeline()
        trail1 = orch.get_audit_trail()
        trail1.append({"fake": "entry"})
        trail2 = orch.get_audit_trail()
        assert len(trail2) < len(trail1)

    def test_collect_without_init_raises(self):
        """Collecting contributions without init raises RuntimeError."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        with pytest.raises(RuntimeError, match="not initialized"):
            orch.collect_site_contributions()

    def test_aggregate_no_contributions_raises(self):
        """Aggregation with no contributions raises RuntimeError."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        with pytest.raises(RuntimeError):
            orch.aggregate_contributions()

    def test_cancel_pipeline(self):
        """cancel_pipeline transitions to CANCELLED."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        orch.initialize_pipeline()
        orch.cancel_pipeline()
        assert orch.status == mod.PipelineStatus.CANCELLED


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------
class TestAggregation:
    """Tests for contribution aggregation in the orchestrator."""

    def test_aggregate_contributions_produces_result(self):
        """Aggregation with valid contributions produces an AggregatedResult."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
            strategy=mod.AnalyticsStrategy.DESCRIPTIVE,
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)

        c1 = mod.SiteContribution(
            site_id="S1", n_subjects=50,
            local_mean={"os": 12.0}, local_variance={"os": 4.0},
            local_sum={"os": 600.0},
            arm_counts={"control": 25, "treatment": 25},
            arm_means={"control": 10.0, "treatment": 14.0},
            arm_variances={"control": 3.0, "treatment": 5.0},
        )
        c2 = mod.SiteContribution(
            site_id="S2", n_subjects=50,
            local_mean={"os": 14.0}, local_variance={"os": 5.0},
            local_sum={"os": 700.0},
            arm_counts={"control": 25, "treatment": 25},
            arm_means={"control": 12.0, "treatment": 16.0},
            arm_variances={"control": 4.0, "treatment": 6.0},
        )
        result = orch.aggregate_contributions([c1, c2])
        assert isinstance(result, mod.AggregatedResult)
        assert result.n_sites == 2
        assert result.n_total_subjects == 100

    def test_aggregate_global_mean_weighted(self):
        """Global mean is weighted by sample size."""
        cfg = mod.AnalyticsConfig(trial_id="NCT-001", site_ids=["A", "B"])
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)

        c1 = mod.SiteContribution(
            site_id="A", n_subjects=100,
            local_mean={"ep": 10.0}, local_sum={"ep": 1000.0},
        )
        c2 = mod.SiteContribution(
            site_id="B", n_subjects=100,
            local_mean={"ep": 20.0}, local_sum={"ep": 2000.0},
        )
        result = orch.aggregate_contributions([c1, c2])
        np.testing.assert_allclose(result.global_mean["ep"], 15.0, atol=1e-6)

    def test_aggregate_zero_subjects_guard(self):
        """Aggregation guards against all-zero subject counts."""
        cfg = mod.AnalyticsConfig(trial_id="NCT-001", site_ids=["A", "B"])
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)

        c1 = mod.SiteContribution(
            site_id="A", n_subjects=0,
            local_mean={"ep": 5.0}, local_sum={"ep": 0.0},
        )
        c2 = mod.SiteContribution(
            site_id="B", n_subjects=0,
            local_mean={"ep": 10.0}, local_sum={"ep": 0.0},
        )
        result = orch.aggregate_contributions([c1, c2])
        # Should not crash; global_mean should be empty or 0
        assert result.n_total_subjects == 0

    def test_aggregate_logs_audit(self):
        """Aggregation creates an audit trail entry."""
        cfg = mod.AnalyticsConfig(trial_id="NCT-001", site_ids=["X"])
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)

        c = mod.SiteContribution(
            site_id="X", n_subjects=10,
            local_mean={"os": 5.0}, local_sum={"os": 50.0},
        )
        orch.aggregate_contributions([c])
        trail = orch.get_audit_trail()
        events = [e["event"] for e in trail]
        assert "aggregation_start" in events or "aggregation_complete" in events

    def test_aggregate_confidence_intervals(self):
        """Aggregation computes confidence intervals."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["A", "B"],
            confidence_level=0.95,
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)

        c1 = mod.SiteContribution(
            site_id="A", n_subjects=50,
            local_mean={"ep": 10.0}, local_variance={"ep": 4.0},
            local_sum={"ep": 500.0},
        )
        c2 = mod.SiteContribution(
            site_id="B", n_subjects=50,
            local_mean={"ep": 12.0}, local_variance={"ep": 5.0},
            local_sum={"ep": 600.0},
        )
        result = orch.aggregate_contributions([c1, c2])
        assert "ep" in result.confidence_intervals
        lo, hi = result.confidence_intervals["ep"]
        assert lo <= result.global_mean["ep"] <= hi


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------
class TestConvergence:
    """Tests for convergence checking."""

    def test_convergence_no_result_returns_false(self):
        """check_convergence returns False when no result exists."""
        cfg = mod.AnalyticsConfig(trial_id="NCT-001", site_ids=["A"])
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        assert orch.check_convergence() is False

    def test_convergence_first_iteration_returns_false(self):
        """First iteration always returns False (no previous to compare)."""
        cfg = mod.AnalyticsConfig(trial_id="NCT-001", site_ids=["A"])
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        c = mod.SiteContribution(
            site_id="A", n_subjects=10,
            local_mean={"ep": 5.0}, local_sum={"ep": 50.0},
        )
        orch.aggregate_contributions([c])
        assert orch.check_convergence() is False

    def test_verify_audit_chain(self):
        """verify_audit_chain returns True for untampered audit trail."""
        cfg = mod.AnalyticsConfig(
            trial_id="NCT-001",
            site_ids=["S1", "S2"],
        )
        orch = mod.ClinicalAnalyticsOrchestrator(config=cfg)
        orch.initialize_pipeline()
        assert orch.verify_audit_chain() is True


# ---------------------------------------------------------------------------
# QualityMetrics tests
# ---------------------------------------------------------------------------
class TestQualityMetrics:
    """Tests for the QualityMetrics dataclass."""

    def test_quality_metrics_clamped(self):
        """Quality metrics are clamped to valid ranges."""
        qm = mod.QualityMetrics(
            completeness_rate=1.5,
            consistency_score=-0.2,
            power_estimate=2.0,
        )
        assert 0.0 <= qm.completeness_rate <= 1.0
        assert 0.0 <= qm.consistency_score <= 1.0
        assert 0.0 <= qm.power_estimate <= 1.0

    def test_quality_metrics_assessed_at(self):
        """QualityMetrics has auto-generated assessed_at timestamp."""
        qm = mod.QualityMetrics()
        assert qm.assessed_at != ""
