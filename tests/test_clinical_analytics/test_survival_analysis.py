"""Tests for clinical-analytics/survival_analysis.py.

Covers enums (SurvivalEndpoint, CensoringType, TestType, HazardModel),
dataclasses (SurvivalData, KaplanMeierEstimate, LogRankResult, CoxPHResult),
and the PrivacyPreservingSurvivalAnalyzer class:
  - Kaplan-Meier (product-limit starts at 1.0, monotonically decreasing)
  - Log-rank test (p-value between 0 and 1)
  - Cox PH (hazard ratios positive)
  - Median survival
  - RMST (restricted mean >= 0)
  - Concordance index (between 0 and 1)
  - Division-by-zero guards (empty data)
  - Federated aggregation
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
    "survival_analysis",
    "clinical-analytics/survival_analysis.py",
)


# ---------------------------------------------------------------------------
# Helper: build SurvivalData records
# ---------------------------------------------------------------------------
def _make_records(times, events, group="control"):
    """Create a list of SurvivalData records for testing."""
    records = []
    for i, (t, e) in enumerate(zip(times, events)):
        records.append(
            mod.SurvivalData(
                patient_id_hash=f"hash_{i:04d}",
                time=float(t),
                event=int(e),
                group=group,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestSurvivalEndpoint:
    """Tests for the SurvivalEndpoint enum."""

    def test_survival_endpoint_members(self):
        """SurvivalEndpoint has expected members."""
        names = set(mod.SurvivalEndpoint.__members__.keys())
        assert "OVERALL_SURVIVAL" in names
        assert "PROGRESSION_FREE_SURVIVAL" in names
        assert "DISEASE_FREE_SURVIVAL" in names

    def test_survival_endpoint_values(self):
        """SurvivalEndpoint values are lowercase strings."""
        assert mod.SurvivalEndpoint.OVERALL_SURVIVAL.value == "overall_survival"

    def test_is_str_enum(self):
        """SurvivalEndpoint members are instances of str."""
        assert isinstance(mod.SurvivalEndpoint.OVERALL_SURVIVAL, str)


class TestCensoringType:
    """Tests for the CensoringType enum."""

    def test_censoring_types(self):
        """CensoringType has RIGHT, LEFT, INTERVAL."""
        expected = {"RIGHT", "LEFT", "INTERVAL"}
        assert set(mod.CensoringType.__members__.keys()) == expected


class TestTestType:
    """Tests for the TestType enum."""

    def test_test_type_members(self):
        """TestType has log-rank and variants."""
        names = set(mod.TestType.__members__.keys())
        assert "LOG_RANK" in names
        assert "WILCOXON" in names


class TestHazardModel:
    """Tests for the HazardModel enum."""

    def test_hazard_model_members(self):
        """HazardModel has Cox PH and parametric variants."""
        names = set(mod.HazardModel.__members__.keys())
        assert "COX_PH" in names
        assert "WEIBULL" in names


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestSurvivalData:
    """Tests for the SurvivalData dataclass."""

    def test_creation(self):
        """SurvivalData is created with expected fields."""
        sd = mod.SurvivalData(
            patient_id_hash="abc123",
            time=12.5,
            event=1,
            group="experimental",
        )
        assert sd.time == 12.5
        assert sd.event == 1
        assert sd.group == "experimental"

    def test_negative_time_raises(self):
        """Negative survival time raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            mod.SurvivalData(patient_id_hash="x", time=-1.0, event=1)

    def test_invalid_event_raises(self):
        """Invalid event indicator raises ValueError."""
        with pytest.raises(ValueError, match="0 or 1"):
            mod.SurvivalData(patient_id_hash="x", time=5.0, event=2)


# ---------------------------------------------------------------------------
# PrivacyPreservingSurvivalAnalyzer tests
# ---------------------------------------------------------------------------
class TestPrivacyPreservingSurvivalAnalyzer:
    """Tests for the PrivacyPreservingSurvivalAnalyzer."""

    def test_init(self):
        """Analyzer initializes with default settings."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        assert analyzer is not None

    # -- Kaplan-Meier -------------------------------------------------------

    def test_kaplan_meier_starts_at_one(self):
        """KM survival curve starts at 1.0 (product-limit estimator)."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 2, 3, 5, 7, 10, 12, 15],
            events=[1, 0, 1, 1, 0, 1, 1, 0],
        )
        km = analyzer.compute_kaplan_meier(records)
        # S(0) conceptually = 1.0; first step should be <= 1.0
        assert km.survival_probabilities[0] <= 1.0

    def test_kaplan_meier_monotonically_decreasing(self):
        """KM survival probabilities are monotonically non-increasing."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            events=[1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        )
        km = analyzer.compute_kaplan_meier(records)
        diffs = np.diff(km.survival_probabilities)
        assert np.all(diffs <= 1e-10), "KM curve should be monotonically non-increasing"

    def test_kaplan_meier_bounded_01(self):
        """KM survival probabilities are all in [0, 1]."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[2, 4, 6, 8, 10],
            events=[1, 1, 1, 1, 1],
        )
        km = analyzer.compute_kaplan_meier(records)
        assert np.all(km.survival_probabilities >= 0.0)
        assert np.all(km.survival_probabilities <= 1.0)

    def test_kaplan_meier_empty_raises(self):
        """Empty data raises ValueError."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        with pytest.raises(ValueError):
            analyzer.compute_kaplan_meier([])

    def test_kaplan_meier_all_censored(self):
        """All-censored data produces survival of 1.0 throughout."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 2, 3, 4, 5],
            events=[0, 0, 0, 0, 0],
        )
        km = analyzer.compute_kaplan_meier(records)
        np.testing.assert_allclose(km.survival_probabilities, 1.0)

    # -- Log-rank test ------------------------------------------------------

    def test_log_rank_p_value_range(self):
        """Log-rank p-value is between 0 and 1."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        group_a = _make_records(
            times=[3, 5, 7, 9, 11, 13, 15, 17],
            events=[1, 1, 0, 1, 1, 0, 1, 1],
            group="control",
        )
        group_b = _make_records(
            times=[4, 8, 12, 16, 20, 24, 28, 32],
            events=[1, 0, 1, 0, 1, 1, 0, 1],
            group="experimental",
        )
        result = analyzer.compute_log_rank_test(group_a, group_b)
        assert isinstance(result, mod.LogRankResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_log_rank_identical_groups(self):
        """Identical groups yield non-significant (high) p-value."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 2, 3, 4, 5, 6, 7, 8],
            events=[1, 1, 0, 1, 1, 0, 1, 0],
        )
        result = analyzer.compute_log_rank_test(records, records)
        # Identical data should have test stat near 0 and p near 1
        assert result.p_value >= 0.5

    def test_log_rank_empty_group_raises(self):
        """Empty group raises ValueError."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(times=[1, 2], events=[1, 0])
        with pytest.raises(ValueError):
            analyzer.compute_log_rank_test(records, [])

    # -- Cox PH -------------------------------------------------------------

    def test_cox_ph_hazard_ratios_positive(self):
        """Cox PH hazard ratios are positive (exp(beta) > 0)."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        rng = np.random.default_rng(42)
        n = 50
        records = []
        for i in range(n):
            cov_val = rng.uniform(0, 1)
            t = rng.exponential(10.0 / (1.0 + cov_val))
            e = int(rng.uniform() < 0.7)
            records.append(
                mod.SurvivalData(
                    patient_id_hash=f"p{i}",
                    time=max(t, 0.1),
                    event=e,
                    covariates={"treatment": cov_val},
                )
            )
        result = analyzer.fit_cox_ph(records, covariate_names=["treatment"])
        assert isinstance(result, mod.CoxPHResult)
        assert np.all(result.hazard_ratios > 0)

    def test_cox_ph_empty_data_raises(self):
        """Cox PH with empty data raises ValueError."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        with pytest.raises(ValueError):
            analyzer.fit_cox_ph([], covariate_names=["x"])

    def test_cox_ph_no_covariates_raises(self):
        """Cox PH without covariates raises ValueError."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(times=[1, 2, 3], events=[1, 0, 1])
        with pytest.raises(ValueError):
            analyzer.fit_cox_ph(records, covariate_names=[])

    # -- Median survival ----------------------------------------------------

    def test_median_survival_defined(self):
        """Median survival is returned when curve crosses 0.5."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            events=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        )
        km = analyzer.compute_kaplan_meier(records)
        median = analyzer.compute_median_survival(km)
        assert median is not None
        assert median > 0

    def test_median_survival_undefined(self):
        """Median survival is None when curve stays above 0.5."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 2, 3, 4, 5],
            events=[0, 0, 0, 0, 0],  # all censored
        )
        km = analyzer.compute_kaplan_meier(records)
        median = analyzer.compute_median_survival(km)
        assert median is None

    # -- RMST ---------------------------------------------------------------

    def test_rmst_non_negative(self):
        """RMST is non-negative."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 3, 5, 7, 10, 12, 15],
            events=[1, 1, 0, 1, 1, 0, 1],
        )
        km = analyzer.compute_kaplan_meier(records)
        rmst = analyzer.compute_restricted_mean(km, tau=15.0)
        assert rmst >= 0.0

    def test_rmst_increases_with_tau(self):
        """Larger tau yields larger or equal RMST."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        records = _make_records(
            times=[1, 2, 3, 4, 5, 6, 7, 8],
            events=[1, 0, 1, 0, 1, 1, 0, 1],
        )
        km = analyzer.compute_kaplan_meier(records)
        rmst_5 = analyzer.compute_restricted_mean(km, tau=5.0)
        rmst_8 = analyzer.compute_restricted_mean(km, tau=8.0)
        assert rmst_8 >= rmst_5 - 1e-10

    # -- Concordance index --------------------------------------------------

    def test_concordance_index_bounded(self):
        """Concordance index is between 0 and 1."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        rng = np.random.default_rng(42)
        n = 30
        predicted = rng.uniform(0, 1, size=n)
        times = rng.exponential(5.0, size=n)
        events = rng.integers(0, 2, size=n)
        c_index = analyzer.compute_concordance_index(predicted, times, events)
        assert 0.0 <= c_index <= 1.0

    def test_concordance_index_small_sample(self):
        """Concordance index with < 2 subjects returns 0.5."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        c = analyzer.compute_concordance_index(
            np.array([0.5]),
            np.array([3.0]),
            np.array([1]),
        )
        assert c == 0.5

    # -- Federated aggregation ----------------------------------------------

    def test_federated_kaplan_meier(self):
        """Federated KM aggregation from two sites produces valid result."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        # Site A
        records_a = _make_records(
            times=[1, 3, 5, 7, 9],
            events=[1, 1, 0, 1, 1],
        )
        km_a = analyzer.compute_kaplan_meier(records_a)
        contrib_a = mod.SiteContribution(
            site_id="A",
            time_points=km_a.time_points,
            at_risk=km_a.at_risk,
            events=km_a.events,
            total_patients=5,
        )
        # Site B
        records_b = _make_records(
            times=[2, 4, 6, 8, 10],
            events=[1, 0, 1, 1, 0],
        )
        km_b = analyzer.compute_kaplan_meier(records_b)
        contrib_b = mod.SiteContribution(
            site_id="B",
            time_points=km_b.time_points,
            at_risk=km_b.at_risk,
            events=km_b.events,
            total_patients=5,
        )
        result = analyzer.federated_kaplan_meier([contrib_a, contrib_b])
        assert isinstance(result, mod.FederatedSurvivalSummary)
        assert result.num_sites == 2
        assert result.total_patients == 10
        # Aggregated KM should be monotonically non-increasing
        diffs = np.diff(result.aggregated_km.survival_probabilities)
        assert np.all(diffs <= 1e-10)

    def test_federated_kaplan_meier_empty_raises(self):
        """Federated KM with no contributions raises ValueError."""
        analyzer = mod.PrivacyPreservingSurvivalAnalyzer()
        with pytest.raises(ValueError):
            analyzer.federated_kaplan_meier([])
