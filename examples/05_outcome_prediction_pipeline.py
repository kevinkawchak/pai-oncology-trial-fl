#!/usr/bin/env python3
"""End-to-end outcome prediction pipeline for federated oncology trials.

CLINICAL CONTEXT
================
Predicting treatment outcomes for oncology patients requires integrating
heterogeneous data sources — clinical features, digital twin simulations,
biomarker profiles, and prior treatment history — into a unified predictive
model.  In a federated setting, each hospital site trains the model locally
on its private patient cohort, and only model updates are shared for global
aggregation.  This example demonstrates the complete pipeline: loading
patient data, generating digital twin features, training a federated model,
evaluating predictions, and generating clinical outcome reports.

USE CASES COVERED
=================
1. Patient data loading with de-identified clinical features, biomarker
   profiles, and treatment history from synthetic oncology datasets.
2. Digital twin feature generation producing model-ready feature vectors
   from patient tumor models and biomarker data.
3. Federated model training with privacy-preserving aggregation and
   cross-site convergence monitoring for outcome prediction.
4. Prediction evaluation with per-site and global accuracy, confusion
   matrix analysis, and response category distribution.
5. Clinical outcome report generation with prediction summaries, cohort
   statistics, and model performance metrics for tumor board review.

FRAMEWORK REQUIREMENTS
======================
Required:
    - numpy >= 1.24.0
    - scipy >= 1.11.0

Optional:
    - matplotlib >= 3.7.0  (prediction visualization)
      https://matplotlib.org/
    - pandas >= 2.0.0  (tabular data and report export)
      https://pandas.pydata.org/
    - scikit-learn >= 1.3.0  (additional evaluation metrics)
      https://scikit-learn.org/

HARDWARE REQUIREMENTS
=====================
    - CPU: Any modern x86-64 or ARM64 processor.
    - RAM: >= 4 GB for synthetic workloads; >= 16 GB for real datasets.
    - GPU: Not required (numpy-only model).

REFERENCES
==========
    - Sheller et al., "Federated learning in medicine: facilitating
      multi-institutional collaborations without sharing patient data",
      Scientific Reports, 2020.
      https://doi.org/10.1038/s41598-020-69250-1
    - Esteva et al., "A guide to deep learning in healthcare", Nature
      Medicine, 2019.  https://doi.org/10.1038/s41591-018-0316-z
    - Rajkomar et al., "Scalable and accurate deep learning with
      electronic health records", npj Digital Medicine, 2018.
      https://doi.org/10.1038/s41746-018-0029-1

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
    import sklearn  # type: ignore[import-untyped]

    HAS_SKLEARN = True
except ImportError:
    sklearn = None  # type: ignore[assignment]
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated.client import FederatedClient
from federated.coordinator import FederationCoordinator
from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data
from federated.model import FederatedModel, ModelConfig
from physical_ai.digital_twin import PatientDigitalTwin, TumorModel
from privacy.audit_logger import AuditLogger, EventType

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


class PredictionTarget(str, Enum):
    """Outcome prediction targets."""

    TREATMENT_RESPONSE = "treatment_response"
    SURVIVAL_CATEGORY = "survival_category"
    TOXICITY_RISK = "toxicity_risk"


class PipelineStage(str, Enum):
    """Stages of the outcome prediction pipeline."""

    DATA_LOADING = "data_loading"
    TWIN_FEATURE_GENERATION = "twin_feature_generation"
    FEDERATED_TRAINING = "federated_training"
    PREDICTION_EVALUATION = "prediction_evaluation"
    REPORT_GENERATION = "report_generation"


class ResponseLabel(str, Enum):
    """Treatment response labels for the binary prediction task."""

    FAVORABLE = "favorable_response"
    POOR = "poor_response"


RESPONSE_LABELS: list[str] = [ResponseLabel.FAVORABLE.value, ResponseLabel.POOR.value]


@dataclass
class PatientRecord:
    """A single de-identified patient record.

    Attributes:
        patient_id: De-identified identifier.
        site_id: Source hospital site.
        features: Clinical feature vector.
        label: Binary outcome label (0=favorable, 1=poor).
        tumor_params: Tumor model parameters for digital twin.
        biomarkers: Biomarker name-value pairs.
        age: Patient age (bracket).
    """

    patient_id: str = ""
    site_id: str = ""
    features: np.ndarray = field(default_factory=lambda: np.zeros(30))
    label: int = 0
    tumor_params: dict[str, float] = field(default_factory=dict)
    biomarkers: dict[str, float] = field(default_factory=dict)
    age: int = 60


@dataclass
class PipelineConfig:
    """Configuration for the outcome prediction pipeline.

    Attributes:
        trial_id: Clinical trial identifier.
        num_sites: Number of federated hospital sites.
        samples_per_site: Synthetic samples per site.
        input_dim: Model input feature dimension.
        hidden_dims: Hidden layer sizes for the MLP.
        output_dim: Number of output classes.
        num_rounds: Federated training rounds.
        local_epochs: Local epochs per round.
        learning_rate: Optimizer learning rate.
        seed: Random seed.
        dp_epsilon: Differential privacy epsilon (None to disable).
        target: Prediction target.
        twin_augmentation: Whether to augment features with twin outputs.
        n_twin_simulations: Number of twin simulations per patient.
    """

    trial_id: str = "PREDICT_ONCO_001"
    num_sites: int = 3
    samples_per_site: int = 200
    input_dim: int = 30
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 2
    num_rounds: int = 12
    local_epochs: int = 4
    learning_rate: float = 0.01
    seed: int = 42
    dp_epsilon: float | None = 2.0
    target: PredictionTarget = PredictionTarget.TREATMENT_RESPONSE
    twin_augmentation: bool = True
    n_twin_simulations: int = 3


@dataclass
class SiteMetrics:
    """Per-site evaluation metrics.

    Attributes:
        site_id: Hospital site identifier.
        num_patients: Number of patients at this site.
        accuracy: Prediction accuracy.
        loss: Cross-entropy loss.
        confusion_matrix: 2x2 confusion matrix.
        response_distribution: Label distribution at this site.
    """

    site_id: str = ""
    num_patients: int = 0
    accuracy: float = 0.0
    loss: float = 0.0
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    response_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class PredictionReport:
    """Clinical outcome prediction report.

    Attributes:
        report_id: Unique report identifier.
        config: Pipeline configuration used.
        global_accuracy: Global model accuracy on held-out data.
        global_loss: Global model loss on held-out data.
        site_metrics: Per-site evaluation metrics.
        training_history: Per-round accuracy and loss.
        patient_predictions: Per-patient prediction summaries.
        cohort_statistics: Aggregate cohort statistics.
        privacy_spent: Differential privacy expenditure.
        total_duration_s: Total pipeline duration.
        disclaimer: Regulatory disclaimer.
    """

    report_id: str = field(default_factory=lambda: f"PRED-{uuid.uuid4().hex[:8].upper()}")
    config: PipelineConfig = field(default_factory=PipelineConfig)
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    site_metrics: list[SiteMetrics] = field(default_factory=list)
    training_history: list[dict[str, float]] = field(default_factory=list)
    patient_predictions: list[dict[str, Any]] = field(default_factory=list)
    cohort_statistics: dict[str, Any] = field(default_factory=dict)
    privacy_spent: dict[str, float] = field(default_factory=dict)
    total_duration_s: float = 0.0
    disclaimer: str = "RESEARCH USE ONLY. Not validated for clinical decisions."


@dataclass
class StageResult:
    """Result from executing a single pipeline stage."""

    stage: PipelineStage
    success: bool = True
    duration_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Section 2 — Core Implementation: OutcomePredictionPipeline
# ============================================================================


class PatientCohortGenerator:
    """Generates synthetic patient cohorts with digital twin parameters.

    Creates patient records with realistic tumor parameters and biomarkers
    suitable for digital twin creation and federated model training.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_cohort(
        self,
        site_id: str,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> list[PatientRecord]:
        """Generate patient records from raw features and labels.

        Args:
            site_id: Hospital site identifier.
            features: Feature matrix (n_patients x n_features).
            labels: Label vector (n_patients,).

        Returns:
            List of PatientRecord instances.
        """
        records = []
        for i in range(len(features)):
            tumor_params = {
                "volume_cm3": float(np.clip(features[i, 0] * 5 + 2, 0.5, 15.0)),
                "growth_rate": float(np.clip(features[i, 1] * 30 + 50, 20.0, 120.0)),
                "chemo_sensitivity": float(np.clip(features[i, 2], 0.1, 0.9)),
                "radio_sensitivity": float(np.clip(features[i, 3], 0.1, 0.9)),
            }

            biomarkers = {
                "pdl1_expression": float(np.clip(features[i, 5], 0.0, 1.0)),
                "ki67_index": float(np.clip(features[i, 6], 0.0, 1.0)),
                "her2_status": float(round(np.clip(features[i, 7], 0.0, 1.0))),
                "egfr_mutation": float(round(np.clip(features[i, 8], 0.0, 1.0))),
            }

            record = PatientRecord(
                patient_id=f"{site_id}_patient_{i:04d}",
                site_id=site_id,
                features=features[i],
                label=int(labels[i]),
                tumor_params=tumor_params,
                biomarkers=biomarkers,
                age=int(np.clip(features[i, 4] * 20 + 55, 30, 85)),
            )
            records.append(record)

        logger.info("Generated %d patient records for site %s", len(records), site_id)
        return records


class DigitalTwinFeatureAugmenter:
    """Augments patient features with digital twin simulation outputs.

    Creates digital twins for each patient and runs treatment simulations
    to generate additional predictive features.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def augment_features(
        self,
        records: list[PatientRecord],
        n_simulations: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate augmented feature vectors using digital twin simulations.

        Args:
            records: Patient records to augment.
            n_simulations: Number of treatment simulations per patient.

        Returns:
            Tuple of (augmented_features, labels) as numpy arrays.
        """
        features_list = []
        labels_list = []

        for record in records:
            tumor = TumorModel(
                volume_cm3=record.tumor_params.get("volume_cm3", 3.0),
                growth_rate=record.tumor_params.get("growth_rate", 60.0),
                chemo_sensitivity=record.tumor_params.get("chemo_sensitivity", 0.5),
                radio_sensitivity=record.tumor_params.get("radio_sensitivity", 0.5),
            )

            twin = PatientDigitalTwin(
                patient_id=record.patient_id,
                tumor=tumor,
                age=record.age,
                biomarkers=record.biomarkers,
            )

            # Generate base features from the twin
            base_features = twin.generate_feature_vector()

            # Run simulations and extract additional features
            sim_features = []
            treatment_types = ["chemotherapy", "radiation", "immunotherapy"]
            for i, treatment in enumerate(treatment_types[:n_simulations]):
                try:
                    if treatment == "chemotherapy":
                        result = twin.simulate_treatment(treatment, dose_mg=75.0, cycles=4)
                    elif treatment == "radiation":
                        result = twin.simulate_treatment(treatment, dose_gy=60.0, fractions=30)
                    else:
                        result = twin.simulate_treatment(treatment, seed=self._seed + i)

                    sim_features.extend(
                        [
                            result["predicted_volume_cm3"],
                            result["volume_reduction_pct"],
                        ]
                    )
                except Exception:
                    sim_features.extend([0.0, 0.0])

            # Combine base features with simulation features
            # Truncate or pad to maintain consistent dimensionality
            combined = np.concatenate([base_features[:24], np.array(sim_features[:6])])
            if len(combined) < 30:
                combined = np.concatenate([combined, np.zeros(30 - len(combined))])
            combined = combined[:30]

            features_list.append(combined)
            labels_list.append(record.label)

        features = np.array(features_list)
        labels = np.array(labels_list)

        logger.info(
            "Augmented features: %d patients, %d features, %d simulations/patient",
            len(records),
            features.shape[1],
            n_simulations,
        )
        return features, labels


class OutcomePredictionPipeline:
    """End-to-end pipeline for federated outcome prediction.

    Orchestrates data loading, digital twin feature generation, federated
    training, prediction evaluation, and report generation.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._model_config = ModelConfig(
            input_dim=config.input_dim,
            hidden_dims=list(config.hidden_dims),
            output_dim=config.output_dim,
            learning_rate=config.learning_rate,
            seed=config.seed,
        )
        self._cohort_gen = PatientCohortGenerator(seed=config.seed)
        self._augmenter = DigitalTwinFeatureAugmenter(seed=config.seed)
        self._coordinator: FederationCoordinator | None = None
        self._clients: list[FederatedClient] = []
        self._site_records: dict[str, list[PatientRecord]] = {}
        self._eval_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._training_history: list[dict[str, float]] = []
        self._audit = AuditLogger()

    # ------------------------------------------------------------------
    # Stage: Data Loading
    # ------------------------------------------------------------------

    def load_data(self) -> StageResult:
        """Generate synthetic data, partition across sites, and create patient records."""
        start = time.time()
        logger.info("Pipeline stage: Data Loading")

        total_samples = self.config.num_sites * self.config.samples_per_site
        x, y = generate_synthetic_oncology_data(
            n_samples=total_samples,
            n_features=self.config.input_dim,
            n_classes=self.config.output_dim,
            seed=self.config.seed,
        )

        partitioner = DataPartitioner(
            num_sites=self.config.num_sites,
            strategy="iid",
            seed=self.config.seed,
        )
        sites = partitioner.partition(x, y)

        for site in sites:
            records = self._cohort_gen.generate_cohort(site.site_id, site.x_train, site.y_train)
            self._site_records[site.site_id] = records
            self._eval_data[site.site_id] = (site.x_test, site.y_test)

        logger.info("Loaded %d patients across %d sites", total_samples, self.config.num_sites)

        return StageResult(
            stage=PipelineStage.DATA_LOADING,
            duration_s=time.time() - start,
            metrics={"total_samples": total_samples, "num_sites": self.config.num_sites},
        )

    # ------------------------------------------------------------------
    # Stage: Digital Twin Feature Generation
    # ------------------------------------------------------------------

    def generate_twin_features(self) -> StageResult:
        """Create digital twins and generate augmented feature vectors."""
        start = time.time()
        logger.info("Pipeline stage: Digital Twin Feature Generation")

        self._clients = []
        for site_id, records in self._site_records.items():
            if self.config.twin_augmentation:
                features, labels = self._augmenter.augment_features(
                    records,
                    n_simulations=self.config.n_twin_simulations,
                )
            else:
                features = np.array([r.features for r in records])
                labels = np.array([r.label for r in records])

            client = FederatedClient(site_id, self._model_config)
            client.set_data(features, labels)
            self._clients.append(client)

        logger.info(
            "Generated twin features for %d sites (augmentation=%s)",
            len(self._clients),
            self.config.twin_augmentation,
        )

        return StageResult(
            stage=PipelineStage.TWIN_FEATURE_GENERATION,
            duration_s=time.time() - start,
            metrics={
                "twin_augmentation": self.config.twin_augmentation,
                "n_simulations": self.config.n_twin_simulations,
            },
        )

    # ------------------------------------------------------------------
    # Stage: Federated Training
    # ------------------------------------------------------------------

    def train_federated_model(self) -> StageResult:
        """Train the outcome prediction model using federated learning."""
        start = time.time()
        logger.info("Pipeline stage: Federated Training")

        dp_enabled = self.config.dp_epsilon is not None
        self._coordinator = FederationCoordinator(
            model_config=self._model_config,
            num_rounds=self.config.num_rounds,
            min_clients=self.config.num_sites,
            strategy="fedavg",
            use_differential_privacy=dp_enabled,
            dp_epsilon=self.config.dp_epsilon or 1.0,
        )
        global_params = self._coordinator.initialize()

        self._audit.log(EventType.SYSTEM_EVENT, actor="pipeline", resource="coordinator", action="initialized")

        self._training_history = []
        first_eval_key = list(self._eval_data.keys())[0]
        eval_pair = self._eval_data[first_eval_key]

        for round_num in range(self.config.num_rounds):
            client_updates = []
            sample_counts = []

            for client in self._clients:
                client.train_local(
                    global_params,
                    epochs=self.config.local_epochs,
                    lr=self.config.learning_rate,
                )
                client_updates.append(client.get_parameters())
                sample_counts.append(client.get_sample_count())

            round_result = self._coordinator.run_round(
                client_updates,
                client_sample_counts=sample_counts,
                eval_data=eval_pair,
            )
            global_params = self._coordinator.get_global_parameters()

            acc = round_result.global_metrics.get("accuracy", 0.0)
            loss = round_result.global_metrics.get("loss", 0.0)
            self._training_history.append({"round": round_num + 1, "accuracy": acc, "loss": loss})

            logger.info(
                "Round %d/%d — accuracy=%.4f, loss=%.4f",
                round_num + 1,
                self.config.num_rounds,
                acc,
                loss,
            )

            if round_result.converged and round_num >= 5:
                logger.info("Converged at round %d", round_num + 1)
                break

        return StageResult(
            stage=PipelineStage.FEDERATED_TRAINING,
            duration_s=time.time() - start,
            metrics={
                "rounds_completed": len(self._training_history),
                "final_accuracy": self._training_history[-1]["accuracy"],
                "final_loss": self._training_history[-1]["loss"],
            },
        )

    # ------------------------------------------------------------------
    # Stage: Prediction Evaluation
    # ------------------------------------------------------------------

    def evaluate_predictions(self) -> StageResult:
        """Evaluate the trained model on held-out data from all sites."""
        start = time.time()
        logger.info("Pipeline stage: Prediction Evaluation")

        if self._coordinator is None or self._coordinator.global_model is None:
            return StageResult(stage=PipelineStage.PREDICTION_EVALUATION, success=False)

        model = self._coordinator.global_model
        site_metrics_list: list[SiteMetrics] = []
        all_predictions: list[dict[str, Any]] = []

        for site_id, (x_test, y_test) in self._eval_data.items():
            metrics = model.evaluate(x_test, y_test)
            predictions = model.predict(x_test)

            # Compute confusion matrix
            tp = int(np.sum((predictions == 1) & (y_test == 1)))
            tn = int(np.sum((predictions == 0) & (y_test == 0)))
            fp = int(np.sum((predictions == 1) & (y_test == 0)))
            fn = int(np.sum((predictions == 0) & (y_test == 1)))

            # Response distribution
            favorable = int(np.sum(y_test == 0))
            poor = int(np.sum(y_test == 1))

            site_metric = SiteMetrics(
                site_id=site_id,
                num_patients=len(y_test),
                accuracy=metrics["accuracy"],
                loss=metrics["loss"],
                confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                response_distribution={"favorable": favorable, "poor": poor},
            )
            site_metrics_list.append(site_metric)

            # Collect per-patient predictions (sample)
            probs = model.forward(x_test)
            for i in range(min(5, len(y_test))):
                all_predictions.append(
                    {
                        "patient_id": f"{site_id}_test_{i:04d}",
                        "site_id": site_id,
                        "true_label": RESPONSE_LABELS[int(y_test[i])],
                        "predicted_label": RESPONSE_LABELS[int(predictions[i])],
                        "confidence": float(np.max(probs[i])),
                        "correct": bool(predictions[i] == y_test[i]),
                    }
                )

        global_acc = np.mean([sm.accuracy for sm in site_metrics_list])
        logger.info("Evaluation complete: global accuracy=%.4f across %d sites", global_acc, len(site_metrics_list))

        return StageResult(
            stage=PipelineStage.PREDICTION_EVALUATION,
            duration_s=time.time() - start,
            metrics={
                "global_accuracy": float(global_acc),
                "site_metrics": site_metrics_list,
                "patient_predictions": all_predictions,
            },
        )

    # ------------------------------------------------------------------
    # Stage: Report Generation
    # ------------------------------------------------------------------

    def generate_report(self, stage_results: list[StageResult]) -> PredictionReport:
        """Compile a clinical outcome prediction report.

        Args:
            stage_results: Results from all pipeline stages.

        Returns:
            PredictionReport with comprehensive results.
        """
        logger.info("Pipeline stage: Report Generation")

        eval_result = next((s for s in stage_results if s.stage == PipelineStage.PREDICTION_EVALUATION), None)
        site_metrics = eval_result.metrics.get("site_metrics", []) if eval_result else []
        patient_preds = eval_result.metrics.get("patient_predictions", []) if eval_result else []

        # Privacy expenditure
        privacy_spent: dict[str, float] = {}
        if self._coordinator and self._coordinator.dp:
            privacy_spent = self._coordinator.dp.get_privacy_spent()

        # Cohort statistics
        total_patients = sum(sm.num_patients for sm in site_metrics)
        total_favorable = sum(sm.response_distribution.get("favorable", 0) for sm in site_metrics)
        total_poor = sum(sm.response_distribution.get("poor", 0) for sm in site_metrics)

        cohort_stats = {
            "total_patients": total_patients,
            "favorable_response": total_favorable,
            "poor_response": total_poor,
            "favorable_pct": total_favorable / max(total_patients, 1) * 100,
            "poor_pct": total_poor / max(total_patients, 1) * 100,
            "num_sites": len(site_metrics),
        }

        total_duration = sum(s.duration_s for s in stage_results)

        report = PredictionReport(
            config=self.config,
            global_accuracy=eval_result.metrics.get("global_accuracy", 0.0) if eval_result else 0.0,
            global_loss=self._training_history[-1]["loss"] if self._training_history else 0.0,
            site_metrics=site_metrics,
            training_history=self._training_history,
            patient_predictions=patient_preds,
            cohort_statistics=cohort_stats,
            privacy_spent=privacy_spent,
            total_duration_s=total_duration,
        )

        logger.info("Report generated: %s", report.report_id)
        return report

    # ------------------------------------------------------------------
    # Full Pipeline Execution
    # ------------------------------------------------------------------

    def run(self) -> PredictionReport:
        """Execute the complete outcome prediction pipeline.

        Returns:
            PredictionReport with all results.
        """
        logger.info("Starting outcome prediction pipeline: %s", self.config.trial_id)
        stage_results: list[StageResult] = []

        stages = [
            self.load_data,
            self.generate_twin_features,
            self.train_federated_model,
            self.evaluate_predictions,
        ]

        for stage_fn in stages:
            result = stage_fn()
            stage_results.append(result)
            if not result.success:
                logger.error("Stage %s failed", result.stage.value)
                break

        report = self.generate_report(stage_results)
        return report


# ============================================================================
# Section 3 — Pipeline Orchestration: PipelineManager
# ============================================================================


class PipelineManager:
    """Manages multiple prediction pipeline runs for comparison.

    Supports running the pipeline with different configurations (e.g.,
    with and without twin augmentation) and comparing results.
    """

    def __init__(self) -> None:
        self._reports: list[PredictionReport] = []

    def run_pipeline(self, config: PipelineConfig) -> PredictionReport:
        """Execute a single prediction pipeline.

        Args:
            config: Pipeline configuration.

        Returns:
            PredictionReport with results.
        """
        pipeline = OutcomePredictionPipeline(config)
        report = pipeline.run()
        self._reports.append(report)
        return report

    def run_augmentation_comparison(self, base_config: PipelineConfig) -> list[PredictionReport]:
        """Compare pipeline results with and without twin augmentation.

        Args:
            base_config: Base pipeline configuration.

        Returns:
            List of two PredictionReports.
        """
        reports = []

        # Without twin augmentation
        config_no_aug = PipelineConfig(
            trial_id=f"{base_config.trial_id}_no_aug",
            num_sites=base_config.num_sites,
            samples_per_site=base_config.samples_per_site,
            num_rounds=base_config.num_rounds,
            local_epochs=base_config.local_epochs,
            learning_rate=base_config.learning_rate,
            seed=base_config.seed,
            dp_epsilon=base_config.dp_epsilon,
            twin_augmentation=False,
        )
        reports.append(self.run_pipeline(config_no_aug))

        # With twin augmentation
        config_aug = PipelineConfig(
            trial_id=f"{base_config.trial_id}_with_aug",
            num_sites=base_config.num_sites,
            samples_per_site=base_config.samples_per_site,
            num_rounds=base_config.num_rounds,
            local_epochs=base_config.local_epochs,
            learning_rate=base_config.learning_rate,
            seed=base_config.seed,
            dp_epsilon=base_config.dp_epsilon,
            twin_augmentation=True,
            n_twin_simulations=3,
        )
        reports.append(self.run_pipeline(config_aug))

        self._print_comparison(reports)
        return reports

    def _print_comparison(self, reports: list[PredictionReport]) -> None:
        """Print a comparison table of pipeline results."""
        print("\n" + "=" * 70)
        print("  AUGMENTATION COMPARISON")
        print("=" * 70)
        print(f"  {'Configuration':<30} {'Accuracy':>10} {'Loss':>10} {'Duration':>10}")
        print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
        for report in reports:
            label = "Twin Augmented" if report.config.twin_augmentation else "Raw Features"
            print(
                f"  {label:<30} "
                f"{report.global_accuracy:>10.4f} "
                f"{report.global_loss:>10.4f} "
                f"{report.total_duration_s:>9.2f}s"
            )
        print("=" * 70)

    @property
    def reports(self) -> list[PredictionReport]:
        """Return all completed pipeline reports."""
        return list(self._reports)


# ============================================================================
# Section 4 — Demonstration
# ============================================================================


def _print_prediction_report(report: PredictionReport) -> None:
    """Pretty-print a prediction report."""
    print("\n" + "=" * 80)
    print("  OUTCOME PREDICTION PIPELINE REPORT")
    print("=" * 80)
    print(f"  Report ID:         {report.report_id}")
    print(f"  Trial:             {report.config.trial_id}")
    print(f"  Sites:             {report.config.num_sites}")
    print(f"  Twin Augmentation: {report.config.twin_augmentation}")
    print(f"  Global Accuracy:   {report.global_accuracy:.4f}")
    print(f"  Global Loss:       {report.global_loss:.4f}")
    print(f"  Duration:          {report.total_duration_s:.2f}s")

    if report.privacy_spent:
        eps = report.privacy_spent.get("total_epsilon_spent", 0.0)
        print(f"  Privacy Spent:     epsilon={eps:.2f}")

    # Cohort statistics
    stats = report.cohort_statistics
    print("\n  Cohort Statistics:")
    print(f"    Total Patients:   {stats.get('total_patients', 0)}")
    print(f"    Favorable:        {stats.get('favorable_response', 0)} ({stats.get('favorable_pct', 0):.1f}%)")
    print(f"    Poor:             {stats.get('poor_response', 0)} ({stats.get('poor_pct', 0):.1f}%)")

    # Per-site metrics
    if report.site_metrics:
        print("\n  Per-Site Metrics:")
        print(
            f"    {'Site':<15} {'Patients':>10} {'Accuracy':>10} {'Loss':>10} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"
        )
        print(f"    {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 5} {'-' * 5} {'-' * 5} {'-' * 5}")
        for sm in report.site_metrics:
            cm = sm.confusion_matrix
            print(
                f"    {sm.site_id:<15} "
                f"{sm.num_patients:>10} "
                f"{sm.accuracy:>10.4f} "
                f"{sm.loss:>10.4f} "
                f"{cm.get('tp', 0):>5} "
                f"{cm.get('tn', 0):>5} "
                f"{cm.get('fp', 0):>5} "
                f"{cm.get('fn', 0):>5}"
            )

    # Training history
    if report.training_history:
        print("\n  Training History (last 5 rounds):")
        print(f"    {'Round':>5} {'Accuracy':>10} {'Loss':>10}")
        for m in report.training_history[-5:]:
            print(f"    {m['round']:>5} {m['accuracy']:>10.4f} {m['loss']:>10.4f}")

    # Sample predictions
    if report.patient_predictions:
        print("\n  Sample Predictions:")
        print(f"    {'Patient':<25} {'True':>20} {'Predicted':>20} {'Conf':>6} {'Correct':>8}")
        for pred in report.patient_predictions[:8]:
            print(
                f"    {pred['patient_id']:<25} "
                f"{pred['true_label']:>20} "
                f"{pred['predicted_label']:>20} "
                f"{pred['confidence']:>6.2f} "
                f"{'yes' if pred['correct'] else 'no':>8}"
            )

    print("\n" + "=" * 80)
    print("  DISCLAIMER: RESEARCH USE ONLY")
    print("=" * 80)


if __name__ == "__main__":
    logger.info("Physical AI Federated Learning — Outcome Prediction Pipeline Example v0.5.0")
    logger.info("Optional deps: matplotlib=%s, pandas=%s, sklearn=%s", HAS_MATPLOTLIB, HAS_PANDAS, HAS_SKLEARN)

    # Configure the pipeline
    config = PipelineConfig(
        trial_id="PREDICT_DEMO_001",
        num_sites=3,
        samples_per_site=150,
        num_rounds=10,
        local_epochs=3,
        learning_rate=0.01,
        seed=42,
        dp_epsilon=2.0,
        twin_augmentation=True,
        n_twin_simulations=3,
    )

    # Run a single pipeline
    print("\n--- Running Outcome Prediction Pipeline ---")
    manager = PipelineManager()
    report = manager.run_pipeline(config)
    _print_prediction_report(report)

    # Run augmentation comparison
    print("\n--- Running Augmentation Comparison ---")
    comparison_config = PipelineConfig(
        trial_id="PREDICT_COMPARE",
        num_sites=3,
        samples_per_site=100,
        num_rounds=8,
        local_epochs=2,
        seed=42,
        dp_epsilon=2.0,
    )
    comparison_reports = manager.run_augmentation_comparison(comparison_config)

    logger.info("Pipeline complete. Total runs: %d", len(manager.reports))
    print("\nDISCLAIMER: RESEARCH USE ONLY. Not for clinical decision-making.")
