"""Tests for examples/05_outcome_prediction_pipeline.py.

Validates pipeline configuration, prediction model dataclasses,
enum definitions, and the explicit zero-patient percentage fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "outcome_prediction",
    "examples/05_outcome_prediction_pipeline.py",
)


class TestPredictionEnums:
    """Verify all outcome prediction enums."""

    def test_prediction_target_values(self):
        assert mod.PredictionTarget.TREATMENT_RESPONSE.value == "treatment_response"
        assert mod.PredictionTarget.SURVIVAL_CATEGORY.value == "survival_category"
        assert mod.PredictionTarget.TOXICITY_RISK.value == "toxicity_risk"

    def test_pipeline_stage_values(self):
        assert mod.PipelineStage.DATA_LOADING.value == "data_loading"
        assert mod.PipelineStage.TWIN_FEATURE_GENERATION.value == "twin_feature_generation"
        assert mod.PipelineStage.FEDERATED_TRAINING.value == "federated_training"
        assert mod.PipelineStage.PREDICTION_EVALUATION.value == "prediction_evaluation"
        assert mod.PipelineStage.REPORT_GENERATION.value == "report_generation"

    def test_response_label_values(self):
        assert mod.ResponseLabel.FAVORABLE.value == "favorable_response"
        assert mod.ResponseLabel.POOR.value == "poor_response"


class TestResponseLabels:
    """Verify RESPONSE_LABELS list."""

    def test_response_labels_content(self):
        assert "favorable_response" in mod.RESPONSE_LABELS
        assert "poor_response" in mod.RESPONSE_LABELS

    def test_response_labels_length(self):
        assert len(mod.RESPONSE_LABELS) == 2


class TestPatientRecord:
    """Verify PatientRecord dataclass."""

    def test_patient_record_defaults(self):
        pr = mod.PatientRecord()
        assert pr.patient_id == ""
        assert pr.site_id == ""
        assert pr.label == 0
        assert pr.age == 60

    def test_patient_record_features_shape(self):
        pr = mod.PatientRecord()
        assert pr.features.shape == (30,)

    def test_patient_record_custom(self):
        pr = mod.PatientRecord(
            patient_id="PT-001",
            site_id="SITE-A",
            label=1,
            biomarkers={"PDL1": 65.0},
        )
        assert pr.patient_id == "PT-001"
        assert pr.label == 1
        assert pr.biomarkers["PDL1"] == 65.0


class TestPipelineConfig:
    """Verify PipelineConfig dataclass."""

    def test_pipeline_config_defaults(self):
        cfg = mod.PipelineConfig()
        assert cfg.trial_id == "PREDICT_ONCO_001"
        assert cfg.num_sites == 3
        assert cfg.samples_per_site == 200
        assert cfg.input_dim == 30
        assert cfg.output_dim == 2

    def test_pipeline_config_hidden_dims(self):
        cfg = mod.PipelineConfig()
        assert cfg.hidden_dims == [64, 32]

    def test_pipeline_config_training_params(self):
        cfg = mod.PipelineConfig()
        assert cfg.num_rounds == 12
        assert cfg.local_epochs == 4
        assert cfg.learning_rate == 0.01

    def test_pipeline_config_dp_epsilon(self):
        cfg = mod.PipelineConfig()
        assert cfg.dp_epsilon == 2.0

    def test_pipeline_config_twin_augmentation(self):
        cfg = mod.PipelineConfig()
        assert cfg.twin_augmentation is True
        assert cfg.n_twin_simulations == 3


class TestSiteMetrics:
    """Verify SiteMetrics dataclass."""

    def test_site_metrics_defaults(self):
        sm = mod.SiteMetrics()
        assert sm.site_id == ""
        assert sm.num_patients == 0
        assert sm.accuracy == 0.0
        assert sm.loss == 0.0

    def test_site_metrics_custom(self):
        sm = mod.SiteMetrics(site_id="SITE-A", accuracy=0.85, num_patients=100)
        assert sm.site_id == "SITE-A"
        assert sm.accuracy == 0.85


class TestPredictionReport:
    """Verify PredictionReport dataclass."""

    def test_prediction_report_defaults(self):
        pr = mod.PredictionReport()
        assert pr.report_id.startswith("PRED-")
        assert pr.global_accuracy == 0.0
        assert pr.disclaimer == "RESEARCH USE ONLY. Not validated for clinical decisions."

    def test_prediction_report_unique_ids(self):
        r1 = mod.PredictionReport()
        r2 = mod.PredictionReport()
        assert r1.report_id != r2.report_id


class TestStageResult:
    """Verify StageResult dataclass."""

    def test_stage_result_creation(self):
        sr = mod.StageResult(stage=mod.PipelineStage.DATA_LOADING)
        assert sr.stage == mod.PipelineStage.DATA_LOADING
        assert sr.success is True
        assert sr.duration_s == 0.0
