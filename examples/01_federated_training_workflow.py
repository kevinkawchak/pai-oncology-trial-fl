#!/usr/bin/env python3
"""Multi-site federated model training workflow for oncology clinical trials.

CLINICAL CONTEXT
================
Oncology clinical trials increasingly span multiple hospital sites, each with
private patient cohorts that cannot be centralized due to HIPAA and GDPR
regulations.  Federated learning enables collaborative model training across
these sites without sharing raw patient data.  This example demonstrates the
complete workflow: site enrollment, data partitioning, privacy-preserving
aggregation, convergence monitoring, and regulatory-ready audit trails.

USE CASES COVERED
=================
1. Multi-site federated model training with FedAvg, FedProx, and SCAFFOLD
   aggregation strategies for heterogeneous oncology data.
2. Privacy-preserving training with differential privacy (Gaussian mechanism)
   and optional secure aggregation for HIPAA-compliant deployments.
3. Automated convergence detection with configurable rolling-window criteria
   to reduce wasted compute on already-converged models.
4. Regulatory audit trail generation with per-round metrics, privacy
   expenditure tracking, and compliance verification suitable for FDA
   pre-submission packages.
5. Non-IID data partitioning to simulate realistic cross-site distribution
   skew observed in multi-center oncology trials.

FRAMEWORK REQUIREMENTS
======================
Required:
    - numpy >= 1.24.0
    - scipy >= 1.11.0

Optional:
    - matplotlib >= 3.7.0  (training curve visualization)
      https://matplotlib.org/
    - flower >= 1.7.0  (production federated learning backend)
      https://flower.ai/
    - pandas >= 2.0.0  (tabular metric export)
      https://pandas.pydata.org/

HARDWARE REQUIREMENTS
=====================
    - CPU: Any modern x86-64 or ARM64 processor.
    - RAM: >= 4 GB for synthetic workloads; >= 16 GB for real clinical data.
    - GPU: Not required for this example (numpy-only model).

REFERENCES
==========
    - McMahan et al., "Communication-Efficient Learning of Deep Networks
      from Decentralized Data", AISTATS 2017.
      https://arxiv.org/abs/1602.05629
    - Li et al., "Federated Optimization in Heterogeneous Networks",
      MLSys 2020.  https://arxiv.org/abs/1812.06127
    - Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
      Federated Learning", ICML 2020.
      https://arxiv.org/abs/1910.06378
    - Dwork & Roth, "The Algorithmic Foundations of Differential Privacy",
      Foundations and Trends in TCS, 2014.
      https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf

DISCLAIMER
==========
RESEARCH USE ONLY.  This software is provided for research and educational
purposes only.  It has NOT been validated for clinical use, is NOT approved
by the FDA or any other regulatory body, and MUST NOT be used to make
clinical decisions or direct patient care.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

try:
    import flwr  # type: ignore[import-untyped]

    HAS_FLOWER = True
except ImportError:
    flwr = None  # type: ignore[assignment]
    HAS_FLOWER = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated.client import FederatedClient
from federated.coordinator import FederationCoordinator
from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data
from federated.model import ModelConfig
from federated.site_enrollment import SiteEnrollmentManager
from privacy.audit_logger import AuditLogger, EventType
from privacy.consent_manager import ConsentManager
from regulatory.compliance_checker import ComplianceChecker

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Section 1 — Configuration: Enums + Dataclasses
# ============================================================================


class AggregationMode(str, Enum):
    """Aggregation strategy selection for the federated workflow."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"


class PrivacyLevel(str, Enum):
    """Pre-configured privacy levels with epsilon budgets."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DataDistribution(str, Enum):
    """Strategy for partitioning data across federated sites."""

    IID = "iid"
    NON_IID = "non_iid"


class WorkflowStage(str, Enum):
    """Stages of the federated training workflow pipeline."""

    COMPLIANCE_CHECK = "compliance_check"
    SITE_ENROLLMENT = "site_enrollment"
    CONSENT_MANAGEMENT = "consent_management"
    DATA_PREPARATION = "data_preparation"
    MODEL_INITIALIZATION = "model_initialization"
    FEDERATED_TRAINING = "federated_training"
    EVALUATION = "evaluation"
    REPORTING = "reporting"


PRIVACY_EPSILON_MAP: dict[PrivacyLevel, float | None] = {
    PrivacyLevel.NONE: None,
    PrivacyLevel.LOW: 8.0,
    PrivacyLevel.MEDIUM: 2.0,
    PrivacyLevel.HIGH: 0.5,
}


@dataclass
class SiteConfig:
    """Configuration for a single federated trial site."""

    site_id: str
    name: str
    patient_count: int = 200
    capabilities: list[str] = field(default_factory=lambda: ["imaging", "biopsy"])


@dataclass
class WorkflowConfig:
    """Complete configuration for a federated training workflow.

    Attributes:
        trial_id: Unique clinical trial identifier.
        num_sites: Number of participating hospital sites.
        aggregation_mode: Federated aggregation strategy.
        privacy_level: Pre-configured privacy level.
        num_rounds: Total federated training rounds.
        local_epochs: Local training epochs per round per client.
        learning_rate: Optimizer learning rate.
        mu: FedProx proximal coefficient (ignored for other strategies).
        data_distribution: How data is split across sites.
        samples_per_site: Number of synthetic samples generated per site.
        input_dim: Number of input features.
        hidden_dims: Hidden layer dimensions for the MLP model.
        output_dim: Number of output classes.
        use_secure_aggregation: Enable secure aggregation protocol.
        convergence_window: Rolling window for convergence detection.
        convergence_threshold: Minimum accuracy improvement threshold.
        seed: Random seed for reproducibility.
        enable_audit: Whether to record an audit trail.
        enable_plotting: Whether to generate training curve plots.
        sites: Per-site configuration overrides.
    """

    trial_id: str = "FL_ONCO_001"
    num_sites: int = 3
    aggregation_mode: AggregationMode = AggregationMode.FEDAVG
    privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM
    num_rounds: int = 15
    local_epochs: int = 5
    learning_rate: float = 0.01
    mu: float = 0.01
    data_distribution: DataDistribution = DataDistribution.IID
    samples_per_site: int = 200
    input_dim: int = 30
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 2
    use_secure_aggregation: bool = False
    convergence_window: int = 5
    convergence_threshold: float = 0.001
    seed: int = 42
    enable_audit: bool = True
    enable_plotting: bool = True
    sites: list[SiteConfig] = field(default_factory=list)

    def get_dp_epsilon(self) -> float | None:
        """Resolve the differential privacy epsilon from the privacy level."""
        return PRIVACY_EPSILON_MAP.get(self.privacy_level)

    def get_site_configs(self) -> list[SiteConfig]:
        """Return site configurations, generating defaults if needed."""
        if self.sites:
            return self.sites
        default_names = [
            "Memorial Cancer Center",
            "University Oncology Institute",
            "Regional Medical Center",
            "National Cancer Research Lab",
            "Community Oncology Clinic",
        ]
        return [
            SiteConfig(
                site_id=f"site_{i}",
                name=default_names[i % len(default_names)],
                patient_count=self.samples_per_site,
            )
            for i in range(self.num_sites)
        ]


@dataclass
class StageResult:
    """Result from executing a single workflow stage."""

    stage: WorkflowStage
    success: bool = True
    duration_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class WorkflowReport:
    """Final report summarising the entire federated workflow."""

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: WorkflowConfig = field(default_factory=WorkflowConfig)
    stage_results: list[StageResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    final_accuracy: float = 0.0
    final_loss: float = 0.0
    privacy_spent: dict[str, float] = field(default_factory=dict)
    audit_event_count: int = 0
    round_metrics: list[dict[str, float]] = field(default_factory=list)


# ============================================================================
# Section 2 — Core Implementation: FederatedTrainingWorkflow
# ============================================================================


class FederatedTrainingWorkflow:
    """Orchestrates the end-to-end federated training lifecycle.

    This class encapsulates the full pipeline: compliance checking, site
    enrollment, consent management, data generation and partitioning,
    model initialization, federated training with the selected strategy,
    and final evaluation.

    Args:
        config: Workflow configuration dataclass.
    """

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config
        self._model_config = ModelConfig(
            input_dim=config.input_dim,
            hidden_dims=list(config.hidden_dims),
            output_dim=config.output_dim,
            learning_rate=config.learning_rate,
            seed=config.seed,
        )
        self._coordinator: FederationCoordinator | None = None
        self._clients: list[FederatedClient] = []
        self._audit: AuditLogger | None = None
        self._eval_data: tuple[np.ndarray, np.ndarray] | None = None
        self._round_metrics: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Stage: Compliance Check
    # ------------------------------------------------------------------

    def run_compliance_check(self) -> StageResult:
        """Verify that the federation configuration meets regulatory requirements."""
        start = time.time()
        logger.info("Stage: Compliance Check")

        dp_epsilon = self.config.get_dp_epsilon()
        federation_cfg = {
            "use_differential_privacy": dp_epsilon is not None,
            "dp_epsilon": dp_epsilon or 0,
            "use_secure_aggregation": self.config.use_secure_aggregation,
            "use_deidentification": True,
            "audit_logging_enabled": self.config.enable_audit,
            "consent_management_enabled": True,
            "encryption_in_transit": True,
            "min_clients": self.config.num_sites,
        }

        checker = ComplianceChecker()
        report = checker.check_federation_config(federation_cfg)
        passed = report.overall_status.value in ("pass", "warning")

        check_details = {c.description: c.status.value for c in report.checks}
        logger.info("Compliance result: %s — %d checks", report.overall_status.value, len(report.checks))

        return StageResult(
            stage=WorkflowStage.COMPLIANCE_CHECK,
            success=passed,
            duration_s=time.time() - start,
            metrics={"overall_status": report.overall_status.value, "num_checks": len(report.checks), **check_details},
        )

    # ------------------------------------------------------------------
    # Stage: Site Enrollment
    # ------------------------------------------------------------------

    def run_site_enrollment(self) -> StageResult:
        """Enroll and activate all hospital sites for the trial."""
        start = time.time()
        logger.info("Stage: Site Enrollment")

        site_mgr = SiteEnrollmentManager(self.config.trial_id, min_patients_per_site=10)
        site_configs = self.config.get_site_configs()

        for sc in site_configs:
            site_mgr.enroll_site(sc.site_id, sc.name, patient_count=sc.patient_count, capabilities=sc.capabilities)
            site_mgr.mark_data_ready(sc.site_id)
            site_mgr.mark_compliance_passed(sc.site_id)
            site_mgr.activate_site(sc.site_id)

        summary = site_mgr.get_enrollment_summary()
        logger.info("Enrolled %d sites with %d total patients", summary["total_sites"], summary["total_patients"])

        return StageResult(
            stage=WorkflowStage.SITE_ENROLLMENT,
            success=True,
            duration_s=time.time() - start,
            metrics=summary,
        )

    # ------------------------------------------------------------------
    # Stage: Consent Management
    # ------------------------------------------------------------------

    def run_consent_management(self) -> StageResult:
        """Register patient consent across all sites."""
        start = time.time()
        logger.info("Stage: Consent Management")

        consent_mgr = ConsentManager()
        site_configs = self.config.get_site_configs()
        patients_per_site = 50

        for sc in site_configs:
            for j in range(patients_per_site):
                consent_mgr.register_consent(f"patient_{sc.site_id}_{j}", self.config.trial_id)

        consented = consent_mgr.get_consented_patients(self.config.trial_id)
        logger.info("Consented %d patients for trial %s", len(consented), self.config.trial_id)

        return StageResult(
            stage=WorkflowStage.CONSENT_MANAGEMENT,
            success=True,
            duration_s=time.time() - start,
            metrics={"consented_patients": len(consented)},
        )

    # ------------------------------------------------------------------
    # Stage: Data Preparation
    # ------------------------------------------------------------------

    def run_data_preparation(self) -> StageResult:
        """Generate synthetic data and partition across sites."""
        start = time.time()
        logger.info("Stage: Data Preparation")

        total_samples = self.config.num_sites * self.config.samples_per_site
        x, y = generate_synthetic_oncology_data(
            n_samples=total_samples,
            n_features=self.config.input_dim,
            n_classes=self.config.output_dim,
            seed=self.config.seed,
        )

        partitioner = DataPartitioner(
            num_sites=self.config.num_sites,
            strategy=self.config.data_distribution.value,
            seed=self.config.seed,
        )
        sites = partitioner.partition(x, y)

        self._clients = []
        site_details = {}
        for site in sites:
            client = FederatedClient(site.site_id, self._model_config)
            client.set_data(site.x_train, site.y_train)
            self._clients.append(client)
            site_details[site.site_id] = {"train": site.num_train, "test": site.num_test}

        self._eval_data = (sites[0].x_test, sites[0].y_test)
        logger.info("Prepared %d samples across %d sites", total_samples, self.config.num_sites)

        return StageResult(
            stage=WorkflowStage.DATA_PREPARATION,
            success=True,
            duration_s=time.time() - start,
            metrics={"total_samples": total_samples, "sites": site_details},
        )

    # ------------------------------------------------------------------
    # Stage: Model Initialization
    # ------------------------------------------------------------------

    def run_model_initialization(self) -> StageResult:
        """Initialize the coordinator, global model, and audit logger."""
        start = time.time()
        logger.info("Stage: Model Initialization")

        dp_epsilon = self.config.get_dp_epsilon()
        self._coordinator = FederationCoordinator(
            model_config=self._model_config,
            num_rounds=self.config.num_rounds,
            min_clients=self.config.num_sites,
            strategy=self.config.aggregation_mode.value,
            mu=self.config.mu,
            use_secure_aggregation=self.config.use_secure_aggregation,
            use_differential_privacy=dp_epsilon is not None,
            dp_epsilon=dp_epsilon or 1.0,
            convergence_window=self.config.convergence_window,
            convergence_threshold=self.config.convergence_threshold,
        )
        self._coordinator.initialize()

        if self.config.enable_audit:
            self._audit = AuditLogger()
            self._audit.log(EventType.SYSTEM_EVENT, actor="coordinator", resource="federation", action="initialized")

        logger.info(
            "Coordinator initialized: strategy=%s, dp=%s",
            self.config.aggregation_mode.value,
            dp_epsilon is not None,
        )

        return StageResult(
            stage=WorkflowStage.MODEL_INITIALIZATION,
            success=True,
            duration_s=time.time() - start,
            metrics={
                "strategy": self.config.aggregation_mode.value,
                "dp_enabled": dp_epsilon is not None,
                "dp_epsilon": dp_epsilon,
            },
        )

    # ------------------------------------------------------------------
    # Stage: Federated Training
    # ------------------------------------------------------------------

    def run_federated_training(self) -> StageResult:
        """Execute the federated training loop across all rounds."""
        start = time.time()
        logger.info("Stage: Federated Training (%d rounds)", self.config.num_rounds)

        if self._coordinator is None:
            return StageResult(
                stage=WorkflowStage.FEDERATED_TRAINING,
                success=False,
                errors=["Coordinator not initialized"],
            )

        global_params = self._coordinator.get_global_parameters()
        self._round_metrics = []
        converged_at: int | None = None

        for round_num in range(self.config.num_rounds):
            client_updates = []
            sample_counts = []

            for client in self._clients:
                result = client.train_local(
                    global_params,
                    epochs=self.config.local_epochs,
                    lr=self.config.learning_rate,
                    mu=self.config.mu,
                )
                client_updates.append(client.get_parameters())
                sample_counts.append(client.get_sample_count())

                if self._audit is not None:
                    self._audit.log(
                        EventType.MODEL_TRAINING,
                        actor=client.client_id,
                        resource="local_model",
                        action=f"round_{round_num + 1}_epochs_{result.epochs_completed}",
                    )

            round_result = self._coordinator.run_round(
                client_updates,
                client_sample_counts=sample_counts,
                eval_data=self._eval_data,
            )
            global_params = self._coordinator.get_global_parameters()

            acc = round_result.global_metrics.get("accuracy", 0.0)
            loss = round_result.global_metrics.get("loss", 0.0)
            self._round_metrics.append({"round": round_num + 1, "accuracy": acc, "loss": loss})

            logger.info(
                "Round %d/%d — accuracy=%.4f, loss=%.4f, converged=%s",
                round_num + 1,
                self.config.num_rounds,
                acc,
                loss,
                round_result.converged,
            )

            if round_result.converged and round_num >= self.config.convergence_window:
                converged_at = round_num + 1
                logger.info("Convergence detected at round %d", converged_at)
                break

        return StageResult(
            stage=WorkflowStage.FEDERATED_TRAINING,
            success=True,
            duration_s=time.time() - start,
            metrics={
                "rounds_completed": len(self._round_metrics),
                "converged_at": converged_at,
                "final_accuracy": self._round_metrics[-1]["accuracy"] if self._round_metrics else 0.0,
                "final_loss": self._round_metrics[-1]["loss"] if self._round_metrics else 0.0,
            },
        )

    # ------------------------------------------------------------------
    # Stage: Evaluation
    # ------------------------------------------------------------------

    def run_evaluation(self) -> StageResult:
        """Evaluate the final global model and generate summary metrics."""
        start = time.time()
        logger.info("Stage: Evaluation")

        if self._coordinator is None or self._coordinator.global_model is None or self._eval_data is None:
            return StageResult(stage=WorkflowStage.EVALUATION, success=False, errors=["Missing model or data"])

        metrics = self._coordinator.global_model.evaluate(self._eval_data[0], self._eval_data[1])
        privacy_spent: dict[str, float] = {}
        if self._coordinator.dp is not None:
            privacy_spent = self._coordinator.dp.get_privacy_spent()

        logger.info("Final evaluation: accuracy=%.4f, loss=%.4f", metrics["accuracy"], metrics["loss"])

        return StageResult(
            stage=WorkflowStage.EVALUATION,
            success=True,
            duration_s=time.time() - start,
            metrics={**metrics, "privacy_spent": privacy_spent},
        )

    # ------------------------------------------------------------------
    # Stage: Reporting
    # ------------------------------------------------------------------

    def run_reporting(self, stage_results: list[StageResult]) -> WorkflowReport:
        """Compile all stage results into a final workflow report."""
        logger.info("Stage: Reporting")

        eval_result = next((s for s in stage_results if s.stage == WorkflowStage.EVALUATION), None)
        privacy_spent = eval_result.metrics.get("privacy_spent", {}) if eval_result else {}
        audit_count = 0
        if self._audit is not None:
            audit_report = self._audit.generate_report()
            audit_count = audit_report.get("total_events", 0)

        total_duration = sum(s.duration_s for s in stage_results)

        report = WorkflowReport(
            config=self.config,
            stage_results=stage_results,
            total_duration_s=total_duration,
            final_accuracy=eval_result.metrics.get("accuracy", 0.0) if eval_result else 0.0,
            final_loss=eval_result.metrics.get("loss", 0.0) if eval_result else 0.0,
            privacy_spent=privacy_spent,
            audit_event_count=audit_count,
            round_metrics=self._round_metrics,
        )

        if self.config.enable_plotting and HAS_MATPLOTLIB:
            self._plot_training_curves(report)

        return report

    def _plot_training_curves(self, report: WorkflowReport) -> None:
        """Generate and save training curve visualization."""
        if not HAS_MATPLOTLIB or not report.round_metrics:
            return

        rounds = [m["round"] for m in report.round_metrics]
        accuracies = [m["accuracy"] for m in report.round_metrics]
        losses = [m["loss"] for m in report.round_metrics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(rounds, accuracies, "b-o", markersize=4, label="Accuracy")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"Federated Training — {self.config.aggregation_mode.value}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(rounds, losses, "r-o", markersize=4, label="Loss")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info("Training curves plotted (not saved in demo mode)")
        plt.close(fig)


# ============================================================================
# Section 3 — Pipeline Orchestration: WorkflowManager
# ============================================================================


class WorkflowManager:
    """Manages the execution of one or more federated training workflows.

    Provides methods to run a single workflow through all stages or to
    compare multiple configurations side by side for strategy selection.
    """

    def __init__(self) -> None:
        self._completed_workflows: list[WorkflowReport] = []

    def run_workflow(self, config: WorkflowConfig) -> WorkflowReport:
        """Execute a complete federated training workflow.

        Runs all stages in sequence: compliance check, site enrollment,
        consent management, data preparation, model initialization,
        federated training, evaluation, and reporting.

        Args:
            config: Workflow configuration.

        Returns:
            WorkflowReport summarising the run.
        """
        logger.info(
            "Starting workflow: trial=%s, strategy=%s, privacy=%s",
            config.trial_id,
            config.aggregation_mode.value,
            config.privacy_level.value,
        )

        workflow = FederatedTrainingWorkflow(config)
        stage_results: list[StageResult] = []

        stages = [
            workflow.run_compliance_check,
            workflow.run_site_enrollment,
            workflow.run_consent_management,
            workflow.run_data_preparation,
            workflow.run_model_initialization,
            workflow.run_federated_training,
            workflow.run_evaluation,
        ]

        for stage_fn in stages:
            result = stage_fn()
            stage_results.append(result)
            if not result.success:
                logger.error("Stage %s failed: %s", result.stage.value, result.errors)
                break

        report = workflow.run_reporting(stage_results)
        self._completed_workflows.append(report)

        logger.info(
            "Workflow complete: accuracy=%.4f, loss=%.4f, duration=%.2fs",
            report.final_accuracy,
            report.final_loss,
            report.total_duration_s,
        )
        return report

    def compare_strategies(self, base_config: WorkflowConfig) -> list[WorkflowReport]:
        """Run the same workflow with each aggregation strategy and compare.

        Args:
            base_config: Base configuration (strategy will be overridden).

        Returns:
            List of WorkflowReport objects, one per strategy.
        """
        reports: list[WorkflowReport] = []
        for strategy in AggregationMode:
            cfg = WorkflowConfig(
                trial_id=f"{base_config.trial_id}_{strategy.value}",
                num_sites=base_config.num_sites,
                aggregation_mode=strategy,
                privacy_level=base_config.privacy_level,
                num_rounds=base_config.num_rounds,
                local_epochs=base_config.local_epochs,
                learning_rate=base_config.learning_rate,
                mu=base_config.mu,
                data_distribution=base_config.data_distribution,
                samples_per_site=base_config.samples_per_site,
                seed=base_config.seed,
                enable_audit=base_config.enable_audit,
                enable_plotting=False,
            )
            report = self.run_workflow(cfg)
            reports.append(report)

        self._print_comparison_table(reports)
        return reports

    def _print_comparison_table(self, reports: list[WorkflowReport]) -> None:
        """Print a comparison table of workflow results."""
        print("\n" + "=" * 80)
        print("  STRATEGY COMPARISON")
        print("=" * 80)
        header = f"  {'Strategy':<12} {'Rounds':>8} {'Accuracy':>10} {'Loss':>10} {'Duration':>10}"
        print(header)
        print("  " + "-" * 76)
        for report in reports:
            rounds = len(report.round_metrics)
            print(
                f"  {report.config.aggregation_mode.value:<12} "
                f"{rounds:>8} "
                f"{report.final_accuracy:>10.4f} "
                f"{report.final_loss:>10.4f} "
                f"{report.total_duration_s:>9.2f}s"
            )
        print("=" * 80)

    @property
    def completed_workflows(self) -> list[WorkflowReport]:
        """Return all completed workflow reports."""
        return list(self._completed_workflows)


# ============================================================================
# Section 4 — Demonstration
# ============================================================================


def _print_report(report: WorkflowReport) -> None:
    """Pretty-print a workflow report to stdout."""
    print("\n" + "=" * 70)
    print("  FEDERATED TRAINING WORKFLOW REPORT")
    print("=" * 70)
    print(f"  Workflow ID:    {report.workflow_id}")
    print(f"  Trial ID:       {report.config.trial_id}")
    print(f"  Strategy:       {report.config.aggregation_mode.value}")
    print(f"  Privacy Level:  {report.config.privacy_level.value}")
    print(f"  Sites:          {report.config.num_sites}")
    print(f"  Total Duration: {report.total_duration_s:.2f}s")
    print()

    print("  Stage Results:")
    for sr in report.stage_results:
        status = "PASS" if sr.success else "FAIL"
        print(f"    [{status}] {sr.stage.value:<25} ({sr.duration_s:.3f}s)")

    print()
    print(f"  Final Accuracy: {report.final_accuracy:.4f}")
    print(f"  Final Loss:     {report.final_loss:.4f}")
    print(f"  Audit Events:   {report.audit_event_count}")

    if report.privacy_spent:
        eps = report.privacy_spent.get("total_epsilon_spent", 0.0)
        print(f"  Privacy Spent:  epsilon={eps:.2f}")

    print()
    if report.round_metrics:
        print("  Training Progress (last 5 rounds):")
        print(f"    {'Round':>5}  {'Accuracy':>10}  {'Loss':>10}")
        for m in report.round_metrics[-5:]:
            print(f"    {m['round']:>5}  {m['accuracy']:>10.4f}  {m['loss']:>10.4f}")

    print("\n" + "=" * 70)
    print("  DISCLAIMER: RESEARCH USE ONLY")
    print("=" * 70)


if __name__ == "__main__":
    logger.info("Physical AI Federated Learning — Training Workflow Example v0.5.0")
    logger.info("Optional deps: matplotlib=%s, pandas=%s, flower=%s", HAS_MATPLOTLIB, HAS_PANDAS, HAS_FLOWER)

    config = WorkflowConfig(
        trial_id="FL_ONCO_DEMO_001",
        num_sites=3,
        aggregation_mode=AggregationMode.FEDAVG,
        privacy_level=PrivacyLevel.MEDIUM,
        num_rounds=10,
        local_epochs=3,
        learning_rate=0.01,
        data_distribution=DataDistribution.IID,
        samples_per_site=200,
        seed=42,
        enable_audit=True,
        enable_plotting=False,
    )

    manager = WorkflowManager()

    # Run a single workflow
    print("\n--- Running single federated workflow ---")
    report = manager.run_workflow(config)
    _print_report(report)

    # Compare strategies
    print("\n--- Comparing aggregation strategies ---")
    comparison_config = WorkflowConfig(
        trial_id="FL_ONCO_COMPARE",
        num_sites=3,
        num_rounds=8,
        local_epochs=2,
        samples_per_site=150,
        seed=42,
        enable_audit=False,
        enable_plotting=False,
    )
    comparison_reports = manager.compare_strategies(comparison_config)

    logger.info("All workflows complete. Total runs: %d", len(manager.completed_workflows))
    print("\nDISCLAIMER: RESEARCH USE ONLY. Not for clinical decision-making.")
