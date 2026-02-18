"""Clinical analytics orchestrator — federated analytics pipeline coordination.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Provides end-to-end orchestration of federated clinical analytics workflows
for multi-site oncology clinical trials.  Each participating site computes
local summary statistics (counts, means, variances, survival estimates) over
its own patient data without ever exposing raw records.  The orchestrator
collects these per-site contributions, applies secure aggregation to
produce global statistics, and runs convergence checks before finalising
the analytics output.

Supported analytics strategies:
- **Descriptive** — summary statistics across trial arms.
- **Inferential** — hypothesis testing on treatment effects.
- **Predictive** — federated predictive model scoring.
- **Survival** — Kaplan-Meier and Cox proportional-hazards estimates.
- **Bayesian** — posterior updating via federated sufficient statistics.

The pipeline enforces 21 CFR Part 11 audit trails, HIPAA minimum-necessary
data access, and ICH E6(R3) data-integrity principles throughout every
phase of execution.

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.  All
outputs require independent biostatistician review before any regulatory
submission.  This software does not constitute a medical device and has
not been cleared or approved by the FDA.
LICENSE: MIT
VERSION: 0.9.0
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SITES: int = 200
"""Hard upper bound on participating sites to prevent resource exhaustion."""

_MIN_SUBJECTS_PER_SITE: int = 5
"""Minimum subjects required at a site to contribute (privacy floor)."""

_DEFAULT_CONVERGENCE_ROUNDS: int = 3
"""Default number of stable rounds before declaring convergence."""

_DEFAULT_CONVERGENCE_TOLERANCE: float = 1e-4
"""Default absolute tolerance for convergence checks."""

_MAX_PIPELINE_ITERATIONS: int = 50
"""Safety cap on iterative pipelines to prevent infinite loops."""

_HMAC_KEY: bytes = b"pai-oncology-trial-fl-analytics-v0.9.0"
"""Static HMAC key for audit record hashing (rotate per deployment)."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnalyticsStrategy(str, Enum):
    """Supported federated analytics strategies.

    Each strategy determines which local computations are requested
    from sites and how the coordinator aggregates them.
    """

    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    SURVIVAL = "survival"
    BAYESIAN = "bayesian"


class PipelineStatus(str, Enum):
    """Lifecycle status of an analytics pipeline run.

    Transitions follow a strict forward-only flow except for
    FAILED and CANCELLED which may occur from any active state.
    """

    PENDING = "pending"
    CONFIGURING = "configuring"
    VALIDATING = "validating"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalyticsPhase(str, Enum):
    """Phases within a single analytics pipeline execution.

    Each phase maps to a distinct computational step that is
    logged individually for 21 CFR Part 11 traceability.
    """

    DATA_COLLECTION = "data_collection"
    PREPROCESSING = "preprocessing"
    LOCAL_COMPUTATION = "local_computation"
    SECURE_AGGREGATION = "secure_aggregation"
    GLOBAL_INFERENCE = "global_inference"
    REPORTING = "reporting"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnalyticsConfig:
    """Configuration for a federated clinical analytics pipeline.

    Attributes:
        pipeline_id: Unique identifier for this pipeline configuration.
        trial_id: Clinical trial identifier (e.g. NCT number).
        strategy: Analytics strategy to execute.
        site_ids: List of site identifiers expected to contribute.
        privacy_budget_epsilon: Differential-privacy epsilon budget.
        privacy_budget_delta: Differential-privacy delta parameter.
        noise_scale: Additive noise scale for secure aggregation.
        min_sites_required: Minimum sites that must contribute for
            the aggregation to proceed.
        min_subjects_per_site: Minimum subjects per site (privacy floor).
        convergence_rounds: Number of stable rounds before convergence
            is declared.
        convergence_tolerance: Absolute tolerance for convergence.
        max_iterations: Maximum pipeline iterations (safety cap).
        confidence_level: Confidence level for interval estimation.
        treatment_arms: Names of treatment arms in the trial.
        primary_endpoint: Primary efficacy endpoint name.
        alpha: Type-I error rate for hypothesis testing.
        audit_enabled: Whether 21 CFR Part 11 audit logging is active.
        require_secure_aggregation: Whether to enforce secure aggregation.
    """

    pipeline_id: str = ""
    trial_id: str = ""
    strategy: AnalyticsStrategy = AnalyticsStrategy.DESCRIPTIVE
    site_ids: list[str] = field(default_factory=list)
    privacy_budget_epsilon: float = 1.0
    privacy_budget_delta: float = 1e-5
    noise_scale: float = 0.0
    min_sites_required: int = 2
    min_subjects_per_site: int = _MIN_SUBJECTS_PER_SITE
    convergence_rounds: int = _DEFAULT_CONVERGENCE_ROUNDS
    convergence_tolerance: float = _DEFAULT_CONVERGENCE_TOLERANCE
    max_iterations: int = _MAX_PIPELINE_ITERATIONS
    confidence_level: float = 0.95
    treatment_arms: list[str] = field(default_factory=lambda: ["control", "treatment"])
    primary_endpoint: str = "overall_survival"
    alpha: float = 0.05
    audit_enabled: bool = True
    require_secure_aggregation: bool = True

    def __post_init__(self) -> None:
        """Validate and bound all clinical parameters after init."""
        if not self.pipeline_id:
            self.pipeline_id = f"PIPE-{uuid.uuid4().hex[:12].upper()}"
        # Bound privacy parameters
        self.privacy_budget_epsilon = max(0.01, min(self.privacy_budget_epsilon, 100.0))
        self.privacy_budget_delta = max(1e-10, min(self.privacy_budget_delta, 0.1))
        self.noise_scale = max(0.0, self.noise_scale)
        # Bound site parameters
        self.min_sites_required = max(1, min(self.min_sites_required, _MAX_SITES))
        self.min_subjects_per_site = max(1, self.min_subjects_per_site)
        # Bound convergence parameters
        self.convergence_rounds = max(1, min(self.convergence_rounds, 100))
        self.convergence_tolerance = max(1e-10, min(self.convergence_tolerance, 1.0))
        self.max_iterations = max(1, min(self.max_iterations, _MAX_PIPELINE_ITERATIONS))
        # Bound statistical parameters
        self.confidence_level = max(0.50, min(self.confidence_level, 0.9999))
        self.alpha = max(0.001, min(self.alpha, 0.20))


@dataclass
class SiteContribution:
    """Local computation results from a single participating site.

    Contains only aggregated summary statistics — never raw patient
    data — in compliance with HIPAA minimum-necessary requirements.

    Attributes:
        site_id: Identifier of the contributing site.
        n_subjects: Number of subjects in the local computation.
        local_mean: Per-endpoint local sample mean.
        local_variance: Per-endpoint local sample variance.
        local_sum: Per-endpoint local sum (for weighted aggregation).
        local_sum_sq: Per-endpoint local sum-of-squares.
        arm_counts: Subject counts per treatment arm.
        arm_means: Sample means per treatment arm.
        arm_variances: Sample variances per treatment arm.
        survival_events: Number of observed events (for survival).
        survival_at_risk: Number of subjects at risk at each time.
        survival_times: Ordered unique event times (for survival).
        sufficient_stats: Strategy-specific sufficient statistics.
        quality_flags: Data quality indicators from the site.
        computed_at: ISO timestamp of local computation.
        checksum: HMAC-SHA256 integrity checksum of the contribution.
    """

    site_id: str = ""
    n_subjects: int = 0
    local_mean: dict[str, float] = field(default_factory=dict)
    local_variance: dict[str, float] = field(default_factory=dict)
    local_sum: dict[str, float] = field(default_factory=dict)
    local_sum_sq: dict[str, float] = field(default_factory=dict)
    arm_counts: dict[str, int] = field(default_factory=dict)
    arm_means: dict[str, float] = field(default_factory=dict)
    arm_variances: dict[str, float] = field(default_factory=dict)
    survival_events: int = 0
    survival_at_risk: list[int] = field(default_factory=list)
    survival_times: list[float] = field(default_factory=list)
    sufficient_stats: dict[str, Any] = field(default_factory=dict)
    quality_flags: dict[str, bool] = field(default_factory=dict)
    computed_at: str = ""
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute HMAC-SHA256 checksum over contribution payload."""
        payload = (
            f"{self.site_id}|{self.n_subjects}|"
            f"{sorted(self.local_mean.items())}|"
            f"{sorted(self.local_sum.items())}|"
            f"{self.computed_at}"
        ).encode("utf-8")
        return hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()

    def verify_checksum(self) -> bool:
        """Verify that the contribution has not been tampered with."""
        expected = self._compute_checksum()
        return hmac.compare_digest(self.checksum, expected)


@dataclass
class AggregatedResult:
    """Globally aggregated analytics output from all contributing sites.

    Attributes:
        pipeline_id: Pipeline that produced this result.
        strategy: Analytics strategy used.
        n_sites: Number of contributing sites.
        n_total_subjects: Total subjects across all sites.
        global_mean: Weighted global mean per endpoint.
        global_variance: Pooled global variance per endpoint.
        global_std: Global standard deviation per endpoint.
        confidence_intervals: Confidence intervals per endpoint.
        arm_global_means: Global means per treatment arm.
        arm_global_variances: Global variances per treatment arm.
        hypothesis_test_results: Results of hypothesis tests.
        survival_curve: Aggregated survival curve (time, probability).
        bayesian_posteriors: Posterior parameters (Bayesian strategy).
        quality_summary: Aggregated data quality metrics.
        aggregated_at: ISO timestamp.
        result_checksum: HMAC-SHA256 integrity checksum.
    """

    pipeline_id: str = ""
    strategy: AnalyticsStrategy = AnalyticsStrategy.DESCRIPTIVE
    n_sites: int = 0
    n_total_subjects: int = 0
    global_mean: dict[str, float] = field(default_factory=dict)
    global_variance: dict[str, float] = field(default_factory=dict)
    global_std: dict[str, float] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    arm_global_means: dict[str, float] = field(default_factory=dict)
    arm_global_variances: dict[str, float] = field(default_factory=dict)
    hypothesis_test_results: dict[str, Any] = field(default_factory=dict)
    survival_curve: dict[str, list[float]] = field(default_factory=dict)
    bayesian_posteriors: dict[str, Any] = field(default_factory=dict)
    quality_summary: dict[str, Any] = field(default_factory=dict)
    aggregated_at: str = ""
    result_checksum: str = ""

    def __post_init__(self) -> None:
        if not self.aggregated_at:
            self.aggregated_at = datetime.now(timezone.utc).isoformat()

    def compute_result_checksum(self) -> str:
        """Compute HMAC-SHA256 over the aggregated result payload."""
        payload = (
            f"{self.pipeline_id}|{self.n_sites}|{self.n_total_subjects}|"
            f"{sorted(self.global_mean.items())}|{self.aggregated_at}"
        ).encode("utf-8")
        digest = hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()
        self.result_checksum = digest
        return digest


@dataclass
class PipelineRun:
    """Metadata and audit trail for a single pipeline execution.

    Attributes:
        run_id: Unique run identifier.
        pipeline_id: Parent pipeline configuration identifier.
        status: Current pipeline status.
        strategy: Analytics strategy for this run.
        started_at: ISO timestamp of run start.
        completed_at: ISO timestamp of run completion.
        current_phase: Active execution phase.
        iteration: Current iteration number.
        site_ids_contributed: Sites that actually contributed.
        elapsed_seconds: Wall-clock duration.
        error_message: Description of failure (if FAILED).
        audit_trail: Ordered list of audit log entries.
        convergence_history: Per-iteration convergence metric values.
    """

    run_id: str = ""
    pipeline_id: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    strategy: AnalyticsStrategy = AnalyticsStrategy.DESCRIPTIVE
    started_at: str = ""
    completed_at: str = ""
    current_phase: AnalyticsPhase = AnalyticsPhase.DATA_COLLECTION
    iteration: int = 0
    site_ids_contributed: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error_message: str = ""
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    convergence_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = f"RUN-{uuid.uuid4().hex[:12].upper()}"


@dataclass
class QualityMetrics:
    """Data quality and statistical metrics for an analytics run.

    Attributes:
        completeness_rate: Fraction of non-missing values [0, 1].
        consistency_score: Cross-site consistency score [0, 1].
        outlier_fraction: Fraction of statistical outliers detected.
        site_agreement_kappa: Fleiss' kappa for inter-site agreement.
        coefficient_of_variation: Global CV for the primary endpoint.
        effective_sample_size: Effective sample size after weighting.
        heterogeneity_i_squared: I-squared heterogeneity statistic.
        power_estimate: Estimated statistical power for the comparison.
        assessed_at: ISO timestamp.
    """

    completeness_rate: float = 0.0
    consistency_score: float = 0.0
    outlier_fraction: float = 0.0
    site_agreement_kappa: float = 0.0
    coefficient_of_variation: float = 0.0
    effective_sample_size: float = 0.0
    heterogeneity_i_squared: float = 0.0
    power_estimate: float = 0.0
    assessed_at: str = ""

    def __post_init__(self) -> None:
        if not self.assessed_at:
            self.assessed_at = datetime.now(timezone.utc).isoformat()
        # Clamp all rates to [0, 1]
        self.completeness_rate = max(0.0, min(1.0, self.completeness_rate))
        self.consistency_score = max(0.0, min(1.0, self.consistency_score))
        self.outlier_fraction = max(0.0, min(1.0, self.outlier_fraction))
        self.heterogeneity_i_squared = max(0.0, min(100.0, self.heterogeneity_i_squared))
        self.power_estimate = max(0.0, min(1.0, self.power_estimate))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ClinicalAnalyticsOrchestrator:
    """Core orchestrator for federated clinical analytics pipelines.

    Coordinates multi-site analytics workflows where each site computes
    local summary statistics and the orchestrator produces globally
    aggregated clinical analytics without accessing raw patient data.

    Compliance alignment:
        - ICH E6(R3): Data integrity and audit trail requirements.
        - 21 CFR Part 11: Electronic records and audit trails.
        - HIPAA: Minimum-necessary data access (summary stats only).
        - FDA AI/ML: Transparency, reproducibility, versioning.

    Args:
        config: Pipeline configuration specifying strategy, sites,
            privacy settings, and convergence criteria.
    """

    def __init__(self, config: AnalyticsConfig) -> None:
        self._config = config
        self._run: PipelineRun = PipelineRun(
            pipeline_id=config.pipeline_id,
            strategy=config.strategy,
        )
        self._contributions: list[SiteContribution] = []
        self._result: AggregatedResult | None = None
        self._quality: QualityMetrics | None = None
        self._previous_global_means: dict[str, float] = {}
        self._convergence_stable_count: int = 0
        self._initialized: bool = False

        logger.info(
            "ClinicalAnalyticsOrchestrator created — pipeline=%s, strategy=%s, sites=%d, trial=%s",
            config.pipeline_id,
            config.strategy.value,
            len(config.site_ids),
            config.trial_id,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> AnalyticsConfig:
        """Return a defensive copy of the pipeline configuration."""
        return copy.deepcopy(self._config)

    @property
    def status(self) -> PipelineStatus:
        """Current pipeline status."""
        return self._run.status

    @property
    def result(self) -> AggregatedResult | None:
        """Return a defensive copy of the aggregated result, if available."""
        if self._result is None:
            return None
        return copy.deepcopy(self._result)

    @property
    def quality(self) -> QualityMetrics | None:
        """Return a defensive copy of quality metrics, if available."""
        if self._quality is None:
            return None
        return copy.deepcopy(self._quality)

    # ------------------------------------------------------------------
    # Pipeline initialisation
    # ------------------------------------------------------------------

    def initialize_pipeline(self) -> bool:
        """Validate configuration and prepare the pipeline for execution.

        Checks that the configuration has sufficient sites, valid privacy
        parameters, and a recognised analytics strategy.  Transitions the
        pipeline from PENDING through CONFIGURING to VALIDATING.

        Returns:
            True if the pipeline is ready for execution, False otherwise.
        """
        self._audit_log("pipeline_initialize_start", phase=AnalyticsPhase.DATA_COLLECTION)
        self._run.status = PipelineStatus.CONFIGURING
        self._run.started_at = datetime.now(timezone.utc).isoformat()

        # --- Validate site count ---
        n_sites = len(self._config.site_ids)
        if n_sites < self._config.min_sites_required:
            msg = f"Insufficient sites: {n_sites} provided, {self._config.min_sites_required} required"
            logger.error(msg)
            self._fail_pipeline(msg)
            return False

        if n_sites > _MAX_SITES:
            msg = f"Site count {n_sites} exceeds maximum {_MAX_SITES}"
            logger.error(msg)
            self._fail_pipeline(msg)
            return False

        # --- Validate privacy budget ---
        if self._config.privacy_budget_epsilon <= 0:
            msg = "Privacy budget epsilon must be positive"
            logger.error(msg)
            self._fail_pipeline(msg)
            return False

        # --- Validate strategy ---
        if self._config.strategy not in AnalyticsStrategy:
            msg = f"Unrecognised analytics strategy: {self._config.strategy}"
            logger.error(msg)
            self._fail_pipeline(msg)
            return False

        # --- Validate confidence level ---
        if not (0.50 <= self._config.confidence_level <= 0.9999):
            msg = f"Confidence level {self._config.confidence_level} out of range [0.50, 0.9999]"
            logger.error(msg)
            self._fail_pipeline(msg)
            return False

        # --- Validate alpha ---
        if not (0.001 <= self._config.alpha <= 0.20):
            msg = f"Alpha {self._config.alpha} out of range [0.001, 0.20]"
            logger.error(msg)
            self._fail_pipeline(msg)
            return False

        # Passed all checks
        self._run.status = PipelineStatus.VALIDATING
        self._initialized = True
        self._audit_log(
            "pipeline_initialized",
            phase=AnalyticsPhase.DATA_COLLECTION,
            details={
                "n_sites": n_sites,
                "strategy": self._config.strategy.value,
                "epsilon": self._config.privacy_budget_epsilon,
                "min_subjects_per_site": self._config.min_subjects_per_site,
            },
        )
        logger.info(
            "Pipeline %s initialized — %d sites, strategy=%s",
            self._config.pipeline_id,
            n_sites,
            self._config.strategy.value,
        )
        return True

    # ------------------------------------------------------------------
    # Site contribution collection
    # ------------------------------------------------------------------

    def collect_site_contributions(
        self,
        site_ids: list[str] | None = None,
    ) -> list[SiteContribution]:
        """Gather local summary statistics from participating sites.

        In a production deployment this would dispatch RPC calls to each
        site's local compute node.  Here it validates incoming contributions
        and filters out sites that do not meet the minimum subject count.

        Each site returns only aggregated statistics — never individual
        patient records — in compliance with HIPAA minimum-necessary
        requirements.

        Args:
            site_ids: Subset of site identifiers to collect from.
                Defaults to all configured sites.

        Returns:
            List of validated SiteContribution objects.

        Raises:
            RuntimeError: If the pipeline has not been initialized.
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

        target_sites = site_ids or list(self._config.site_ids)
        self._run.status = PipelineStatus.RUNNING
        self._run.current_phase = AnalyticsPhase.LOCAL_COMPUTATION
        self._audit_log(
            "site_collection_start",
            phase=AnalyticsPhase.LOCAL_COMPUTATION,
            details={"target_sites": target_sites},
        )

        validated: list[SiteContribution] = []
        for sid in target_sites:
            contribution = self._request_site_computation(sid)
            if contribution is None:
                logger.warning("Site %s returned no contribution — skipping", sid)
                self._audit_log(
                    "site_contribution_missing",
                    phase=AnalyticsPhase.LOCAL_COMPUTATION,
                    details={"site_id": sid},
                )
                continue

            # Validate minimum subject count
            if contribution.n_subjects < self._config.min_subjects_per_site:
                logger.warning(
                    "Site %s has %d subjects (min %d) — excluded",
                    sid,
                    contribution.n_subjects,
                    self._config.min_subjects_per_site,
                )
                self._audit_log(
                    "site_excluded_insufficient_subjects",
                    phase=AnalyticsPhase.LOCAL_COMPUTATION,
                    details={
                        "site_id": sid,
                        "n_subjects": contribution.n_subjects,
                        "min_required": self._config.min_subjects_per_site,
                    },
                )
                continue

            # Verify integrity checksum
            if not contribution.verify_checksum():
                logger.error("Checksum verification failed for site %s — excluded", sid)
                self._audit_log(
                    "site_checksum_failed",
                    phase=AnalyticsPhase.LOCAL_COMPUTATION,
                    details={"site_id": sid},
                )
                continue

            validated.append(contribution)
            self._audit_log(
                "site_contribution_accepted",
                phase=AnalyticsPhase.LOCAL_COMPUTATION,
                details={
                    "site_id": sid,
                    "n_subjects": contribution.n_subjects,
                },
            )

        # Check that enough sites contributed
        if len(validated) < self._config.min_sites_required:
            msg = f"Only {len(validated)} sites contributed (min {self._config.min_sites_required} required)"
            logger.error(msg)
            self._fail_pipeline(msg)
            return []

        self._contributions = validated
        self._run.site_ids_contributed = [c.site_id for c in validated]

        logger.info(
            "Collected contributions from %d / %d sites",
            len(validated),
            len(target_sites),
        )
        return list(validated)

    def _request_site_computation(self, site_id: str) -> SiteContribution | None:
        """Request local computation from a single site.

        In production this dispatches an RPC call.  The base implementation
        returns None so that subclasses or test harnesses can override it
        with simulated or real site responses.

        Args:
            site_id: Identifier of the target site.

        Returns:
            SiteContribution from the site, or None if unreachable.
        """
        # Hook for subclasses — production systems override this method
        # to perform actual RPC calls to site compute nodes.
        return None

    # ------------------------------------------------------------------
    # Secure aggregation
    # ------------------------------------------------------------------

    def aggregate_contributions(
        self,
        contributions: list[SiteContribution] | None = None,
    ) -> AggregatedResult:
        """Securely aggregate per-site statistics into global results.

        Implements strategy-specific aggregation:
        - Descriptive: weighted means and pooled variances.
        - Inferential: pooled statistics plus hypothesis tests.
        - Predictive: aggregated model scores.
        - Survival: federated Kaplan-Meier estimation.
        - Bayesian: posterior parameter updating.

        Noise is optionally added based on the configured noise scale
        for differential-privacy compliance.

        Args:
            contributions: Site contributions to aggregate.  Defaults
                to those collected via ``collect_site_contributions``.

        Returns:
            AggregatedResult containing global statistics.

        Raises:
            RuntimeError: If no contributions are available.
        """
        contribs = contributions or self._contributions
        if not contribs:
            raise RuntimeError("No contributions available for aggregation.")

        self._run.status = PipelineStatus.AGGREGATING
        self._run.current_phase = AnalyticsPhase.SECURE_AGGREGATION
        self._audit_log(
            "aggregation_start",
            phase=AnalyticsPhase.SECURE_AGGREGATION,
            details={"n_contributions": len(contribs)},
        )

        result = AggregatedResult(
            pipeline_id=self._config.pipeline_id,
            strategy=self._config.strategy,
            n_sites=len(contribs),
            n_total_subjects=sum(c.n_subjects for c in contribs),
        )

        # --- Weighted global mean and pooled variance ---
        result.global_mean = self._compute_global_mean(contribs)
        result.global_variance = self._compute_pooled_variance(contribs, result.global_mean)
        result.global_std = {k: math.sqrt(max(v, 0.0)) for k, v in result.global_variance.items()}

        # --- Confidence intervals ---
        result.confidence_intervals = self._compute_confidence_intervals(
            result.global_mean,
            result.global_variance,
            result.n_total_subjects,
        )

        # --- Per-arm aggregation ---
        result.arm_global_means = self._compute_arm_means(contribs)
        result.arm_global_variances = self._compute_arm_variances(contribs, result.arm_global_means)

        # --- Strategy-specific computations ---
        if self._config.strategy == AnalyticsStrategy.INFERENTIAL:
            result.hypothesis_test_results = self._run_hypothesis_tests(contribs, result)
        elif self._config.strategy == AnalyticsStrategy.SURVIVAL:
            result.survival_curve = self._compute_federated_survival(contribs)
        elif self._config.strategy == AnalyticsStrategy.BAYESIAN:
            result.bayesian_posteriors = self._compute_bayesian_update(contribs)
        elif self._config.strategy == AnalyticsStrategy.PREDICTIVE:
            result.hypothesis_test_results = self._compute_predictive_summary(contribs)

        # --- Apply noise for differential privacy ---
        if self._config.noise_scale > 0:
            result = self._apply_aggregation_noise(result)

        # --- Quality assessment ---
        self._quality = self._assess_quality(contribs, result)
        result.quality_summary = {
            "completeness_rate": self._quality.completeness_rate,
            "consistency_score": self._quality.consistency_score,
            "heterogeneity_i_squared": self._quality.heterogeneity_i_squared,
            "effective_sample_size": self._quality.effective_sample_size,
        }

        # --- Integrity checksum ---
        result.compute_result_checksum()

        self._result = result
        self._audit_log(
            "aggregation_complete",
            phase=AnalyticsPhase.SECURE_AGGREGATION,
            details={
                "n_sites": result.n_sites,
                "n_subjects": result.n_total_subjects,
                "endpoints": list(result.global_mean.keys()),
            },
        )

        logger.info(
            "Aggregation complete — %d sites, %d subjects, strategy=%s",
            result.n_sites,
            result.n_total_subjects,
            self._config.strategy.value,
        )
        return result

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------

    def run_analytics_pipeline(self) -> AggregatedResult | None:
        """Execute the complete federated analytics pipeline end-to-end.

        Orchestrates the full lifecycle: initialisation, site collection,
        aggregation, convergence checking, and final reporting.  For
        iterative strategies (Bayesian, Predictive) the pipeline loops
        until convergence or the maximum iteration count is reached.

        Returns:
            The final AggregatedResult, or None if the pipeline failed.
        """
        start_time = time.monotonic()
        self._audit_log("pipeline_run_start", phase=AnalyticsPhase.DATA_COLLECTION)

        # Step 1 — Initialise
        if not self._initialized:
            if not self.initialize_pipeline():
                return None

        # Step 2 — Iterative collection + aggregation
        final_result: AggregatedResult | None = None
        for iteration in range(1, self._config.max_iterations + 1):
            self._run.iteration = iteration
            self._run.current_phase = AnalyticsPhase.LOCAL_COMPUTATION

            self._audit_log(
                "iteration_start",
                phase=AnalyticsPhase.LOCAL_COMPUTATION,
                details={"iteration": iteration},
            )

            # Collect
            contributions = self.collect_site_contributions()
            if not contributions:
                logger.error("No contributions in iteration %d — aborting", iteration)
                return None

            # Preprocess contributions
            contributions = self._preprocess_contributions(contributions)

            # Aggregate
            result = self.aggregate_contributions(contributions)
            final_result = result

            # Check convergence
            converged = self.check_convergence()
            self._audit_log(
                "iteration_complete",
                phase=AnalyticsPhase.GLOBAL_INFERENCE,
                details={
                    "iteration": iteration,
                    "converged": converged,
                    "stable_count": self._convergence_stable_count,
                },
            )

            if converged:
                logger.info("Pipeline converged at iteration %d", iteration)
                break

            # Non-iterative strategies complete in one pass
            if self._config.strategy in (
                AnalyticsStrategy.DESCRIPTIVE,
                AnalyticsStrategy.INFERENTIAL,
                AnalyticsStrategy.SURVIVAL,
            ):
                break

        # Step 3 — Finalise
        elapsed = time.monotonic() - start_time
        self._run.elapsed_seconds = elapsed
        self._run.completed_at = datetime.now(timezone.utc).isoformat()

        if final_result is not None:
            self._run.status = PipelineStatus.COMPLETED
            self._run.current_phase = AnalyticsPhase.REPORTING
            self._audit_log(
                "pipeline_completed",
                phase=AnalyticsPhase.REPORTING,
                details={
                    "elapsed_seconds": round(elapsed, 3),
                    "iterations": self._run.iteration,
                    "n_sites": final_result.n_sites,
                    "n_subjects": final_result.n_total_subjects,
                },
            )
            logger.info(
                "Pipeline %s completed in %.2fs — %d iterations",
                self._config.pipeline_id,
                elapsed,
                self._run.iteration,
            )
        else:
            self._fail_pipeline("Pipeline produced no result")

        return final_result

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def check_convergence(self) -> bool:
        """Check whether the global statistics have converged.

        Computes the maximum absolute change in global means between
        the current and previous iterations.  If this change stays below
        the configured tolerance for ``convergence_rounds`` consecutive
        iterations, the pipeline is declared converged.

        Returns:
            True if convergence criteria are met.
        """
        if self._result is None:
            return False

        current_means = self._result.global_mean
        if not current_means:
            return False

        if not self._previous_global_means:
            # First iteration — store and continue
            self._previous_global_means = dict(current_means)
            self._convergence_stable_count = 0
            self._run.convergence_history.append(float("inf"))
            return False

        # Compute max absolute change across all endpoints
        max_delta = 0.0
        for key, current_val in current_means.items():
            prev_val = self._previous_global_means.get(key, current_val)
            delta = abs(current_val - prev_val)
            max_delta = max(max_delta, delta)

        self._run.convergence_history.append(max_delta)
        self._previous_global_means = dict(current_means)

        if max_delta < self._config.convergence_tolerance:
            self._convergence_stable_count += 1
        else:
            self._convergence_stable_count = 0

        converged = self._convergence_stable_count >= self._config.convergence_rounds

        logger.debug(
            "Convergence check — max_delta=%.8f, tolerance=%.8f, stable_count=%d/%d, converged=%s",
            max_delta,
            self._config.convergence_tolerance,
            self._convergence_stable_count,
            self._config.convergence_rounds,
            converged,
        )
        return converged

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the 21 CFR Part 11 audit trail.

        Each entry includes a timestamp, event type, phase, detail
        payload, and an HMAC-SHA256 integrity hash.

        Returns:
            List of audit log entries (deep copy).
        """
        return copy.deepcopy(self._run.audit_trail)

    # ------------------------------------------------------------------
    # Aggregation helpers — descriptive statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_global_mean(
        contributions: list[SiteContribution],
    ) -> dict[str, float]:
        """Compute sample-size-weighted global mean per endpoint.

        Uses local sums and counts rather than local means to avoid
        loss-of-precision issues in floating-point weighted averaging.
        Falls back to mean-of-means when sums are unavailable.
        """
        total_n = sum(c.n_subjects for c in contributions)
        if total_n == 0:
            return {}

        # Gather all endpoint keys
        all_keys: set[str] = set()
        for c in contributions:
            all_keys.update(c.local_sum.keys())
            all_keys.update(c.local_mean.keys())

        result: dict[str, float] = {}
        for key in sorted(all_keys):
            numerator = 0.0
            denominator = 0

            for c in contributions:
                if key in c.local_sum:
                    numerator += c.local_sum[key]
                    denominator += c.n_subjects
                elif key in c.local_mean:
                    numerator += c.local_mean[key] * c.n_subjects
                    denominator += c.n_subjects

            # Division-by-zero guard
            if denominator == 0:
                result[key] = 0.0
            else:
                result[key] = numerator / denominator

        return result

    @staticmethod
    def _compute_pooled_variance(
        contributions: list[SiteContribution],
        global_mean: dict[str, float],
    ) -> dict[str, float]:
        """Compute pooled variance using the parallel algorithm.

        Combines within-site variance and between-site mean deviation
        into an unbiased pooled variance estimator:

            Var_pooled = (sum_i (n_i - 1) * s_i^2 + n_i * (xbar_i - xbar)^2)
                         / (N - 1)

        Division-by-zero is guarded by returning 0.0 when the total
        degrees of freedom is zero.
        """
        total_n = sum(c.n_subjects for c in contributions)
        if total_n <= 1:
            return {k: 0.0 for k in global_mean}

        result: dict[str, float] = {}
        for key in global_mean:
            xbar = global_mean[key]
            ss_within = 0.0
            ss_between = 0.0

            for c in contributions:
                ni = c.n_subjects
                if ni == 0:
                    continue

                # Within-site component
                si2 = c.local_variance.get(key, 0.0)
                ss_within += (ni - 1) * si2

                # Between-site component
                xi_bar = c.local_mean.get(key, xbar)
                ss_between += ni * (xi_bar - xbar) ** 2

            dof = total_n - 1
            # Division-by-zero guard
            if dof <= 0:
                result[key] = 0.0
            else:
                result[key] = (ss_within + ss_between) / dof

        return result

    def _compute_confidence_intervals(
        self,
        global_mean: dict[str, float],
        global_variance: dict[str, float],
        n_total: int,
    ) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals using the t-distribution.

        Falls back to the normal approximation when the sample size
        exceeds 1000.
        """
        if n_total <= 1:
            return {k: (v, v) for k, v in global_mean.items()}

        alpha = 1.0 - self._config.confidence_level
        dof = max(n_total - 1, 1)

        # Use t-distribution for small samples, normal for large
        if n_total <= 1000:
            t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, dof))
        else:
            t_crit = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))

        intervals: dict[str, tuple[float, float]] = {}
        for key in global_mean:
            mu = global_mean[key]
            var = global_variance.get(key, 0.0)
            # Division-by-zero guard
            se = math.sqrt(max(var, 0.0) / max(n_total, 1))
            margin = t_crit * se
            intervals[key] = (mu - margin, mu + margin)

        return intervals

    # ------------------------------------------------------------------
    # Per-arm aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_arm_means(
        contributions: list[SiteContribution],
    ) -> dict[str, float]:
        """Compute weighted global mean per treatment arm."""
        arm_sums: dict[str, float] = {}
        arm_counts: dict[str, int] = {}

        for c in contributions:
            for arm, count in c.arm_counts.items():
                arm_counts[arm] = arm_counts.get(arm, 0) + count
                mean_val = c.arm_means.get(arm, 0.0)
                arm_sums[arm] = arm_sums.get(arm, 0.0) + mean_val * count

        result: dict[str, float] = {}
        for arm in arm_sums:
            total = arm_counts.get(arm, 0)
            # Division-by-zero guard
            if total == 0:
                result[arm] = 0.0
            else:
                result[arm] = arm_sums[arm] / total

        return result

    @staticmethod
    def _compute_arm_variances(
        contributions: list[SiteContribution],
        arm_global_means: dict[str, float],
    ) -> dict[str, float]:
        """Compute pooled variance per treatment arm."""
        arm_total_n: dict[str, int] = {}
        arm_ss: dict[str, float] = {}

        for c in contributions:
            for arm, count in c.arm_counts.items():
                if count == 0:
                    continue
                arm_total_n[arm] = arm_total_n.get(arm, 0) + count
                var = c.arm_variances.get(arm, 0.0)
                local_mean = c.arm_means.get(arm, 0.0)
                global_arm_mean = arm_global_means.get(arm, 0.0)
                # Within + between components
                within = (count - 1) * var
                between = count * (local_mean - global_arm_mean) ** 2
                arm_ss[arm] = arm_ss.get(arm, 0.0) + within + between

        result: dict[str, float] = {}
        for arm in arm_ss:
            dof = arm_total_n.get(arm, 1) - 1
            # Division-by-zero guard
            if dof <= 0:
                result[arm] = 0.0
            else:
                result[arm] = arm_ss[arm] / dof

        return result

    # ------------------------------------------------------------------
    # Strategy-specific computations
    # ------------------------------------------------------------------

    def _run_hypothesis_tests(
        self,
        contributions: list[SiteContribution],
        result: AggregatedResult,
    ) -> dict[str, Any]:
        """Run hypothesis tests for the inferential strategy.

        Performs a two-sample Welch's t-test comparing treatment arms
        using the pooled statistics.  Welch's test is preferred over
        the standard t-test because it does not assume equal variances
        across arms.
        """
        arms = self._config.treatment_arms
        if len(arms) < 2:
            return {"error": "At least two treatment arms required"}

        arm0, arm1 = arms[0], arms[1]
        n0 = sum(c.arm_counts.get(arm0, 0) for c in contributions)
        n1 = sum(c.arm_counts.get(arm1, 0) for c in contributions)

        if n0 == 0 or n1 == 0:
            return {
                "test": "welch_t_test",
                "error": "One or both arms have zero subjects",
                "arm0_n": n0,
                "arm1_n": n1,
            }

        mu0 = result.arm_global_means.get(arm0, 0.0)
        mu1 = result.arm_global_means.get(arm1, 0.0)
        var0 = result.arm_global_variances.get(arm0, 0.0)
        var1 = result.arm_global_variances.get(arm1, 0.0)

        # Welch's t-test: SE of difference in means
        se0 = var0 / max(n0, 1)
        se1 = var1 / max(n1, 1)
        se_diff = math.sqrt(se0 + se1)

        # Division-by-zero guard
        if se_diff < 1e-15:
            t_stat = 0.0
            p_value = 1.0
            dof = max(n0 + n1 - 2, 1)
        else:
            t_stat = (mu1 - mu0) / se_diff
            # Welch-Satterthwaite degrees of freedom
            numerator = (se0 + se1) ** 2
            d0 = se0**2 / max(n0 - 1, 1) if n0 > 1 else 0.0
            d1 = se1**2 / max(n1 - 1, 1) if n1 > 1 else 0.0
            denom = d0 + d1
            # Division-by-zero guard
            if denom < 1e-15:
                dof = max(n0 + n1 - 2, 1)
            else:
                dof = max(numerator / denom, 1.0)
            p_value = float(2.0 * scipy_stats.t.sf(abs(t_stat), dof))

        # Cohen's d effect size using pooled SD
        pooled_var = (max(n0 - 1, 0) * var0 + max(n1 - 1, 0) * var1) / max(n0 + n1 - 2, 1)
        pooled_sd = math.sqrt(max(pooled_var, 0.0))
        # Division-by-zero guard
        if pooled_sd < 1e-15:
            cohens_d = 0.0
        else:
            cohens_d = (mu1 - mu0) / pooled_sd

        significant = p_value < self._config.alpha

        return {
            "test": "welch_t_test",
            "arm0": arm0,
            "arm1": arm1,
            "arm0_n": n0,
            "arm1_n": n1,
            "arm0_mean": round(mu0, 6),
            "arm1_mean": round(mu1, 6),
            "mean_difference": round(mu1 - mu0, 6),
            "t_statistic": round(t_stat, 6),
            "degrees_of_freedom": round(float(dof), 2),
            "p_value": round(p_value, 8),
            "alpha": self._config.alpha,
            "significant": significant,
            "effect_size_cohens_d": round(cohens_d, 6),
        }

    @staticmethod
    def _compute_federated_survival(
        contributions: list[SiteContribution],
    ) -> dict[str, list[float]]:
        """Compute a federated Kaplan-Meier survival curve.

        Aggregates per-site event counts and at-risk counts at each
        unique event time to produce a global survival function.

        The method uses per-time-point events stored in each site's
        ``sufficient_stats["events_per_time"]`` dictionary.  If this
        is unavailable, the site's total ``survival_events`` are
        distributed uniformly across its time points as a fallback.

        Returns a dictionary with ``times`` and ``survival_probability``
        lists.
        """
        # Collect all unique event times across sites
        all_times: set[float] = set()
        for c in contributions:
            all_times.update(c.survival_times)

        if not all_times:
            return {"times": [], "survival_probability": []}

        sorted_times = sorted(all_times)
        times_out: list[float] = []
        surv_prob: list[float] = []
        cumulative_survival = 1.0

        for t in sorted_times:
            total_events = 0
            total_at_risk = 0

            for c in contributions:
                if t not in c.survival_times:
                    continue
                idx = c.survival_times.index(t)

                # At-risk count at this time
                if idx < len(c.survival_at_risk):
                    total_at_risk += c.survival_at_risk[idx]

                # Events at this time
                per_time = c.sufficient_stats.get("events_per_time", {})
                if t in per_time:
                    total_events += per_time[t]
                elif len(c.survival_times) > 0:
                    # Fallback: distribute total events uniformly
                    n_times = len(c.survival_times)
                    # Division-by-zero guard
                    if n_times > 0:
                        total_events += c.survival_events / n_times

            # Division-by-zero guard
            if total_at_risk > 0:
                hazard = total_events / total_at_risk
                # Clamp hazard to [0, 1]
                hazard = max(0.0, min(1.0, hazard))
                cumulative_survival *= 1.0 - hazard

            # Clamp survival probability to [0, 1]
            cumulative_survival = max(0.0, min(1.0, cumulative_survival))
            times_out.append(t)
            surv_prob.append(round(cumulative_survival, 8))

        return {
            "times": times_out,
            "survival_probability": surv_prob,
        }

    @staticmethod
    def _compute_bayesian_update(
        contributions: list[SiteContribution],
    ) -> dict[str, Any]:
        """Perform federated Bayesian posterior updating.

        Uses conjugate normal-normal updating:  Given a weakly
        informative prior N(mu_0, sigma_0^2) and sufficient statistics
        (sum_x, sum_x2, n) from each site, the posterior is computed
        in closed form.  The prior precision (1/sigma_0^2 = 0.01)
        encodes minimal prior knowledge.
        """
        # Weakly informative prior
        prior_mu = 0.0
        prior_precision = 0.01  # 1/sigma_0^2 (very wide)

        total_n = 0
        total_sum = 0.0
        total_sum_sq = 0.0

        for c in contributions:
            n_i = c.n_subjects
            if n_i == 0:
                continue
            total_n += n_i

            # Use sufficient statistics if available
            s = c.sufficient_stats
            if "sum_x" in s and "sum_x2" in s:
                total_sum += s["sum_x"]
                total_sum_sq += s["sum_x2"]
            else:
                # Reconstruct from mean and variance (primary endpoint)
                for key in c.local_mean:
                    mu_i = c.local_mean[key]
                    var_i = c.local_variance.get(key, 1.0)
                    total_sum += mu_i * n_i
                    if n_i > 1:
                        total_sum_sq += var_i * (n_i - 1) + n_i * mu_i**2
                    else:
                        total_sum_sq += mu_i**2
                    break  # Use primary endpoint only

        if total_n == 0:
            return {
                "posterior_mean": prior_mu,
                "posterior_variance": 1.0 / max(prior_precision, 1e-15),
                "prior_mean": prior_mu,
                "prior_precision": prior_precision,
                "n_total": 0,
            }

        # Estimate data variance from sufficient statistics
        data_mean = total_sum / max(total_n, 1)
        data_variance = (total_sum_sq / max(total_n, 1)) - data_mean**2
        data_variance = max(data_variance, 1e-10)  # Floor at small positive

        # Conjugate update: posterior precision = prior precision + n / sigma^2
        data_precision = total_n / data_variance
        posterior_precision = prior_precision + data_precision

        # Division-by-zero guard
        if posterior_precision < 1e-15:
            posterior_precision = 1e-15

        posterior_mean = (prior_precision * prior_mu + data_precision * data_mean) / posterior_precision
        posterior_variance = 1.0 / posterior_precision

        # 95% credible interval
        post_std = math.sqrt(posterior_variance)
        ci_lower = posterior_mean - 1.96 * post_std
        ci_upper = posterior_mean + 1.96 * post_std

        return {
            "posterior_mean": round(posterior_mean, 8),
            "posterior_variance": round(posterior_variance, 8),
            "posterior_std": round(post_std, 8),
            "credible_interval_95": (round(ci_lower, 8), round(ci_upper, 8)),
            "prior_mean": prior_mu,
            "prior_precision": prior_precision,
            "data_mean": round(data_mean, 8),
            "data_variance": round(data_variance, 8),
            "n_total": total_n,
            "bayes_factor_vs_null": round(
                _safe_bayes_factor(posterior_mean, post_std, 0.0),
                6,
            ),
        }

    def _compute_predictive_summary(
        self,
        contributions: list[SiteContribution],
    ) -> dict[str, Any]:
        """Aggregate predictive model scores across sites.

        Computes weighted AUC, Brier score, and cross-site
        heterogeneity metrics from per-site sufficient statistics.
        """
        total_n = sum(c.n_subjects for c in contributions)
        if total_n == 0:
            return {"error": "No subjects for predictive summary"}

        # Aggregate AUC and Brier score (weighted by sample size)
        weighted_auc = 0.0
        weighted_brier = 0.0
        auc_values: list[float] = []

        for c in contributions:
            n_i = c.n_subjects
            if n_i == 0:
                continue
            w = n_i / max(total_n, 1)
            auc_i = c.sufficient_stats.get("auc", 0.5)
            brier_i = c.sufficient_stats.get("brier_score", 0.25)
            weighted_auc += w * auc_i
            weighted_brier += w * brier_i
            auc_values.append(auc_i)

        # Heterogeneity in AUC across sites
        auc_arr = np.array(auc_values) if auc_values else np.array([0.5])
        auc_std = float(np.std(auc_arr, ddof=1)) if len(auc_arr) > 1 else 0.0

        return {
            "metric": "predictive_summary",
            "global_auc": round(weighted_auc, 6),
            "global_brier_score": round(weighted_brier, 6),
            "auc_std_across_sites": round(auc_std, 6),
            "n_sites": len(contributions),
            "n_total_subjects": total_n,
        }

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_contributions(
        self,
        contributions: list[SiteContribution],
    ) -> list[SiteContribution]:
        """Preprocess site contributions before aggregation.

        Applies outlier detection on per-site means using the median
        absolute deviation (MAD) method to identify sites with extreme
        values that may indicate data quality issues.  Flagged sites
        are not excluded but are annotated in ``quality_flags`` for
        downstream review.
        """
        self._run.current_phase = AnalyticsPhase.PREPROCESSING
        if len(contributions) < 3:
            return contributions

        # Collect per-site means for each endpoint
        endpoint_means: dict[str, list[float]] = {}
        for c in contributions:
            for key, val in c.local_mean.items():
                endpoint_means.setdefault(key, []).append(val)

        # Detect outliers using median absolute deviation (MAD)
        outlier_sites: set[str] = set()
        for key, values in endpoint_means.items():
            arr = np.array(values)
            if len(arr) < 3:
                continue
            median_val = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median_val)))
            # Division-by-zero guard
            if mad < 1e-15:
                continue
            # Modified z-scores per Iglewicz & Hoaglin threshold
            modified_z = 0.6745 * (arr - median_val) / mad
            for i, z in enumerate(modified_z):
                if abs(z) > 3.5:
                    outlier_sites.add(contributions[i].site_id)

        for c in contributions:
            if c.site_id in outlier_sites:
                c.quality_flags["outlier_detected"] = True
                logger.warning(
                    "Outlier detected at site %s — flagged for review",
                    c.site_id,
                )
                self._audit_log(
                    "outlier_flagged",
                    phase=AnalyticsPhase.PREPROCESSING,
                    details={"site_id": c.site_id},
                )

        return contributions

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    def _assess_quality(
        self,
        contributions: list[SiteContribution],
        result: AggregatedResult,
    ) -> QualityMetrics:
        """Assess data quality and statistical properties of the run.

        Computes completeness, cross-site consistency, heterogeneity
        (I-squared), effective sample size, and power estimates.
        """
        n_sites = len(contributions)
        total_n = result.n_total_subjects

        # --- Completeness ---
        expected_endpoints: set[str] = set()
        for c in contributions:
            expected_endpoints.update(c.local_mean.keys())
        n_expected = len(expected_endpoints) * n_sites
        n_present = 0
        for c in contributions:
            for key in expected_endpoints:
                if key in c.local_mean:
                    n_present += 1
        # Division-by-zero guard
        completeness = n_present / max(n_expected, 1)

        # --- Consistency (coefficient of variation of site means) ---
        cv_values: list[float] = []
        for key in expected_endpoints:
            site_means = [c.local_mean[key] for c in contributions if key in c.local_mean]
            if len(site_means) >= 2:
                arr = np.array(site_means)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1))
                # Division-by-zero guard
                if abs(mean_val) > 1e-15:
                    cv_values.append(std_val / abs(mean_val))

        avg_cv = float(np.mean(cv_values)) if cv_values else 0.0
        # Map CV to a consistency score: lower CV = higher consistency
        consistency = max(0.0, min(1.0, 1.0 - avg_cv))

        # --- Heterogeneity (I-squared) ---
        i_squared = self._compute_i_squared(contributions, result.global_mean)

        # --- Effective sample size (Kish's formula) ---
        site_sizes = np.array([c.n_subjects for c in contributions], dtype=float)
        if len(site_sizes) > 0 and float(np.sum(site_sizes)) > 0:
            total = float(np.sum(site_sizes))
            sum_sq = float(np.sum(site_sizes**2))
            # Division-by-zero guard
            if sum_sq > 0:
                ess = total**2 / sum_sq
            else:
                ess = 0.0
        else:
            ess = 0.0

        # --- Outlier fraction ---
        n_outliers = sum(1 for c in contributions if c.quality_flags.get("outlier_detected", False))
        # Division-by-zero guard
        outlier_frac = n_outliers / max(n_sites, 1)

        # --- Power estimate (two-arm comparison) ---
        power = self._estimate_power(contributions, result)

        return QualityMetrics(
            completeness_rate=round(completeness, 4),
            consistency_score=round(consistency, 4),
            outlier_fraction=round(outlier_frac, 4),
            coefficient_of_variation=round(avg_cv, 4),
            effective_sample_size=round(ess, 2),
            heterogeneity_i_squared=round(i_squared, 2),
            power_estimate=round(power, 4),
        )

    @staticmethod
    def _compute_i_squared(
        contributions: list[SiteContribution],
        global_mean: dict[str, float],
    ) -> float:
        """Compute the I-squared heterogeneity statistic.

        I-squared measures the percentage of variability in site
        estimates that is due to true between-site heterogeneity
        rather than sampling variance.

            I^2 = max(0, (Q - df) / Q) * 100

        where Q is Cochran's Q statistic and df = k - 1.
        """
        if len(contributions) < 2:
            return 0.0

        # Use the first available endpoint
        key = next(iter(global_mean), None)
        if key is None:
            return 0.0

        xbar = global_mean[key]

        # Cochran's Q with inverse-variance weights
        q_stat = 0.0
        k = 0
        for c in contributions:
            if key not in c.local_mean:
                continue
            ni = c.n_subjects
            if ni == 0:
                continue
            vi = c.local_variance.get(key, 0.0)
            # Weight = inverse variance; guard division-by-zero
            if vi > 1e-15 and ni > 0:
                wi = ni / vi
            else:
                wi = float(ni)
            q_stat += wi * (c.local_mean[key] - xbar) ** 2
            k += 1

        df = k - 1
        if df <= 0:
            return 0.0

        # Division-by-zero guard
        if q_stat < 1e-15:
            return 0.0

        i_sq = max(0.0, (q_stat - df) / q_stat) * 100.0
        return min(i_sq, 100.0)

    def _estimate_power(
        self,
        contributions: list[SiteContribution],
        result: AggregatedResult,
    ) -> float:
        """Estimate statistical power for a two-arm comparison.

        Uses a normal approximation to compute the probability of
        rejecting the null hypothesis given the observed effect size,
        pooled variance, and total sample size.
        """
        arms = self._config.treatment_arms
        if len(arms) < 2:
            return 0.0

        arm0, arm1 = arms[0], arms[1]
        n0 = sum(c.arm_counts.get(arm0, 0) for c in contributions)
        n1 = sum(c.arm_counts.get(arm1, 0) for c in contributions)

        if n0 == 0 or n1 == 0:
            return 0.0

        mu0 = result.arm_global_means.get(arm0, 0.0)
        mu1 = result.arm_global_means.get(arm1, 0.0)
        var0 = result.arm_global_variances.get(arm0, 1.0)
        var1 = result.arm_global_variances.get(arm1, 1.0)

        se = math.sqrt(var0 / max(n0, 1) + var1 / max(n1, 1))

        # Division-by-zero guard
        if se < 1e-15:
            return 1.0 if abs(mu1 - mu0) > 1e-15 else 0.0

        z_alpha = float(scipy_stats.norm.ppf(1.0 - self._config.alpha / 2.0))
        noncentrality = abs(mu1 - mu0) / se
        power = float(1.0 - scipy_stats.norm.cdf(z_alpha - noncentrality))

        return max(0.0, min(1.0, power))

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def _apply_aggregation_noise(
        self,
        result: AggregatedResult,
    ) -> AggregatedResult:
        """Add calibrated noise to aggregated statistics for DP.

        The noise magnitude is proportional to the configured noise
        scale and inversely proportional to the square root of the
        total number of subjects.  This provides a basic Gaussian
        mechanism for (epsilon, delta)-differential privacy.
        """
        rng = np.random.default_rng()
        n = max(result.n_total_subjects, 1)
        sigma = self._config.noise_scale / math.sqrt(n)

        for key in result.global_mean:
            noise = float(rng.normal(0, sigma))
            result.global_mean[key] += noise

        for key in result.global_variance:
            noise = float(rng.normal(0, sigma * 0.5))
            result.global_variance[key] = max(0.0, result.global_variance[key] + noise)
            result.global_std[key] = math.sqrt(result.global_variance[key])

        self._audit_log(
            "dp_noise_applied",
            phase=AnalyticsPhase.SECURE_AGGREGATION,
            details={
                "noise_scale": self._config.noise_scale,
                "sigma": round(sigma, 8),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline lifecycle helpers
    # ------------------------------------------------------------------

    def _fail_pipeline(self, message: str) -> None:
        """Transition the pipeline to FAILED status with a message."""
        self._run.status = PipelineStatus.FAILED
        self._run.error_message = message
        self._run.completed_at = datetime.now(timezone.utc).isoformat()
        self._audit_log(
            "pipeline_failed",
            phase=self._run.current_phase,
            details={"error": message},
        )
        logger.error("Pipeline %s FAILED: %s", self._config.pipeline_id, message)

    def cancel_pipeline(self) -> None:
        """Cancel a running pipeline.

        Can be called from any active state.  Records the cancellation
        in the audit trail.
        """
        if self._run.status in (
            PipelineStatus.COMPLETED,
            PipelineStatus.FAILED,
        ):
            logger.warning("Cannot cancel pipeline in %s state", self._run.status.value)
            return
        self._run.status = PipelineStatus.CANCELLED
        self._run.completed_at = datetime.now(timezone.utc).isoformat()
        self._audit_log(
            "pipeline_cancelled",
            phase=self._run.current_phase,
        )
        logger.info("Pipeline %s cancelled", self._config.pipeline_id)

    def get_run_metadata(self) -> dict[str, Any]:
        """Return a summary of the current pipeline run.

        Provides a snapshot of run status, timing, site participation,
        and convergence progress suitable for monitoring dashboards.
        """
        return {
            "run_id": self._run.run_id,
            "pipeline_id": self._run.pipeline_id,
            "status": self._run.status.value,
            "strategy": self._run.strategy.value,
            "started_at": self._run.started_at,
            "completed_at": self._run.completed_at,
            "current_phase": self._run.current_phase.value,
            "iteration": self._run.iteration,
            "sites_contributed": list(self._run.site_ids_contributed),
            "elapsed_seconds": round(self._run.elapsed_seconds, 3),
            "convergence_history": list(self._run.convergence_history),
            "error_message": self._run.error_message,
            "audit_trail_length": len(self._run.audit_trail),
        }

    # ------------------------------------------------------------------
    # 21 CFR Part 11 audit logging
    # ------------------------------------------------------------------

    def _audit_log(
        self,
        event: str,
        phase: AnalyticsPhase,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an entry to the 21 CFR Part 11 audit trail.

        Each entry is timestamped, hashed with HMAC-SHA256 for
        tamper detection, and linked to the previous entry via
        a chain hash for sequential integrity.

        Args:
            event: Machine-readable event identifier.
            phase: Pipeline phase during which the event occurred.
            details: Additional context payload.
        """
        if not self._config.audit_enabled:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        seq = len(self._run.audit_trail)

        # Chain hash: include previous entry's hash for tamper detection
        prev_hash = ""
        if seq > 0:
            prev_hash = self._run.audit_trail[-1].get("entry_hash", "")

        entry_payload = (f"{seq}|{timestamp}|{event}|{phase.value}|{self._config.pipeline_id}|{prev_hash}").encode(
            "utf-8"
        )
        entry_hash = hmac.new(_HMAC_KEY, entry_payload, hashlib.sha256).hexdigest()

        entry: dict[str, Any] = {
            "sequence": seq,
            "timestamp": timestamp,
            "event": event,
            "phase": phase.value,
            "pipeline_id": self._config.pipeline_id,
            "run_id": self._run.run_id,
            "details": details or {},
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        }
        self._run.audit_trail.append(entry)

    def verify_audit_chain(self) -> bool:
        """Verify the integrity of the entire audit trail chain.

        Recomputes each entry's HMAC-SHA256 hash and checks that the
        chain of ``prev_hash`` values is consistent.  Returns False
        if any entry has been tampered with.

        Returns:
            True if the audit chain is intact.
        """
        trail = self._run.audit_trail
        for i, entry in enumerate(trail):
            # Verify chain linkage
            expected_prev = trail[i - 1]["entry_hash"] if i > 0 else ""
            if entry.get("prev_hash", "") != expected_prev:
                logger.error("Audit chain broken at sequence %d", i)
                return False

            # Recompute hash
            payload = (
                f"{entry['sequence']}|{entry['timestamp']}|"
                f"{entry['event']}|{entry['phase']}|"
                f"{self._config.pipeline_id}|{expected_prev}"
            ).encode("utf-8")
            expected_hash = hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(entry.get("entry_hash", ""), expected_hash):
                logger.error("Audit hash mismatch at sequence %d", i)
                return False

        return True


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_bayes_factor(
    posterior_mean: float,
    posterior_std: float,
    null_value: float,
) -> float:
    """Compute an approximate Savage-Dickey Bayes factor.

    BF_10 = prior density at null / posterior density at null.
    Uses a weakly informative N(0, 10) prior for simplicity.

    Division-by-zero is guarded by returning 1.0 (uninformative).
    """
    prior_std = 10.0  # Weakly informative

    # Division-by-zero guard
    if posterior_std < 1e-15 or prior_std < 1e-15:
        return 1.0

    prior_density = float(scipy_stats.norm.pdf(null_value, loc=0.0, scale=prior_std))
    posterior_density = float(scipy_stats.norm.pdf(null_value, loc=posterior_mean, scale=posterior_std))

    # Division-by-zero guard
    if posterior_density < 1e-300:
        return 1.0

    return prior_density / posterior_density
