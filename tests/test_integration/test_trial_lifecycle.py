"""Cross-module integration tests for end-to-end trial lifecycle.

Tests the complete flow: enrollment -> data ingestion -> harmonization ->
training -> evaluation, verifying that modules interoperate correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Load all required modules
# ---------------------------------------------------------------------------
coordinator_mod = load_module("federated.coordinator", "federated/coordinator.py")
client_mod = load_module("federated.client", "federated/client.py")
model_mod = load_module("federated.model", "federated/model.py")
data_mod = load_module("federated.data_ingestion", "federated/data_ingestion.py")
enrollment_mod = load_module("federated.site_enrollment", "federated/site_enrollment.py")
audit_mod = load_module("privacy.audit_logger", "privacy/audit_logger.py")
consent_mod = load_module("privacy.consent_manager", "privacy/consent_manager.py")
compliance_mod = load_module("regulatory.compliance_checker", "regulatory/compliance_checker.py")


class TestSiteEnrollmentToDataIngestion:
    """Verify enrollment feeds into data partitioning."""

    def test_enrollment_creates_sites(self):
        mgr = enrollment_mod.SiteEnrollmentManager(study_id="TEST-001")
        mgr.enroll_site("SITE-A", "Hospital A", patient_count=100)
        mgr.enroll_site("SITE-B", "Hospital B", patient_count=150)
        summary = mgr.get_enrollment_summary()
        assert summary["total_sites"] == 2
        assert summary["total_patients"] == 250

    def test_enrollment_to_partitioning(self):
        mgr = enrollment_mod.SiteEnrollmentManager(study_id="TEST-002")
        mgr.enroll_site("SITE-A", "Hospital A", patient_count=100)
        mgr.enroll_site("SITE-B", "Hospital B", patient_count=100)
        x, y = data_mod.generate_synthetic_oncology_data(n_samples=200, seed=42)
        partitioner = data_mod.DataPartitioner(num_sites=2, strategy="iid", seed=42)
        sites = partitioner.partition(x, y)
        assert len(sites) == 2
        total = sum(s.num_train + s.num_test for s in sites)
        assert total == 200

    def test_audit_trail_after_enrollment(self):
        mgr = enrollment_mod.SiteEnrollmentManager(study_id="TEST-003")
        mgr.enroll_site("SITE-A", "Hospital A", patient_count=50)
        trail = mgr.get_audit_trail()
        assert len(trail) >= 1
        assert trail[0]["site_id"] == "SITE-A"


class TestDataIngestionToModelTraining:
    """Verify data ingestion integrates with model creation and training."""

    def test_data_shapes_match_model_config(self):
        x, y = data_mod.generate_synthetic_oncology_data(
            n_samples=100,
            n_features=30,
            n_classes=2,
            seed=42,
        )
        config = model_mod.ModelConfig(input_dim=30, output_dim=2)
        model = model_mod.FederatedModel.from_config(config)
        probs = model.forward(x)
        assert probs.shape == (100, 2)

    def test_partitioned_data_trains_client(self):
        x, y = data_mod.generate_synthetic_oncology_data(
            n_samples=200,
            n_features=30,
            seed=42,
        )
        partitioner = data_mod.DataPartitioner(num_sites=2, strategy="iid", seed=42)
        sites = partitioner.partition(x, y)
        config = model_mod.ModelConfig(input_dim=30, output_dim=2)
        client = client_mod.FederatedClient(
            client_id="client_0",
            model_config=config,
            x_train=sites[0].x_train,
            y_train=sites[0].y_train,
        )
        global_params = client.model.get_parameters()
        result = client.train_local(global_params, epochs=1, lr=0.01)
        assert result.epochs_completed == 1
        assert result.final_loss >= 0

    def test_site_data_properties(self):
        x, y = data_mod.generate_synthetic_oncology_data(n_samples=100, seed=42)
        partitioner = data_mod.DataPartitioner(num_sites=2, test_fraction=0.2, seed=42)
        sites = partitioner.partition(x, y)
        for s in sites:
            assert s.num_train > 0
            assert s.num_test > 0


class TestCoordinatorTrainingLoop:
    """Verify coordinator can run a training round with clients."""

    def test_coordinator_single_round(self):
        config = model_mod.ModelConfig(input_dim=10, output_dim=2, seed=42)
        coord = coordinator_mod.FederationCoordinator(model_config=config)
        global_params = coord.initialize()
        x, y = data_mod.generate_synthetic_oncology_data(
            n_samples=60,
            n_features=10,
            n_classes=2,
            seed=42,
        )
        partitioner = data_mod.DataPartitioner(num_sites=2, strategy="iid", seed=42)
        sites = partitioner.partition(x, y)
        clients = []
        for i, site in enumerate(sites):
            c = client_mod.FederatedClient(
                client_id=f"client_{i}",
                model_config=config,
                x_train=site.x_train,
                y_train=site.y_train,
            )
            clients.append(c)
        client_updates = []
        sample_counts = []
        for c in clients:
            c.train_local(global_params, epochs=1)
            client_updates.append(c.model.get_parameters())
            sample_counts.append(c.x_train.shape[0])
        result = coord.run_round(client_updates, client_sample_counts=sample_counts)
        new_params = coord.get_global_parameters()
        assert len(new_params) == len(global_params)
        assert result.round_number == 1

    def test_coordinator_multiple_rounds(self):
        config = model_mod.ModelConfig(input_dim=10, output_dim=2, seed=42)
        coord = coordinator_mod.FederationCoordinator(model_config=config)
        coord.initialize()
        x, y = data_mod.generate_synthetic_oncology_data(
            n_samples=60,
            n_features=10,
            n_classes=2,
            seed=42,
        )
        partitioner = data_mod.DataPartitioner(num_sites=2, seed=42)
        sites = partitioner.partition(x, y)
        clients = [
            client_mod.FederatedClient(
                client_id=f"c{i}",
                model_config=config,
                x_train=s.x_train,
                y_train=s.y_train,
            )
            for i, s in enumerate(sites)
        ]
        for _ in range(3):
            gp = coord.get_global_parameters()
            updates = []
            counts = []
            for c in clients:
                c.train_local(gp, epochs=1)
                updates.append(c.model.get_parameters())
                counts.append(c.x_train.shape[0])
            coord.run_round(updates, client_sample_counts=counts)
        assert coord.current_round >= 3


class TestComplianceBeforeTraining:
    """Verify compliance checks integrate with training configuration."""

    def test_compliance_check_passes_with_privacy(self):
        checker = compliance_mod.ComplianceChecker(regulations=["hipaa"])
        config = {
            "use_differential_privacy": True,
            "dp_epsilon": 2.0,
            "use_secure_aggregation": True,
            "use_deidentification": True,
            "audit_logging_enabled": True,
            "consent_management_enabled": True,
            "encryption_in_transit": True,
            "min_clients": 3,
        }
        report = checker.check_federation_config(config)
        assert report.overall_status in (
            compliance_mod.CheckStatus.PASS,
            compliance_mod.CheckStatus.WARNING,
        )

    def test_compliance_check_fails_without_privacy(self):
        checker = compliance_mod.ComplianceChecker(regulations=["hipaa"])
        config = {
            "use_differential_privacy": False,
            "dp_epsilon": None,
            "use_secure_aggregation": False,
            "use_deidentification": False,
            "audit_logging_enabled": False,
            "consent_management_enabled": False,
            "encryption_in_transit": False,
            "min_clients": 1,
        }
        report = checker.check_federation_config(config)
        fail_count = sum(1 for c in report.checks if c.status == compliance_mod.CheckStatus.FAIL)
        assert fail_count > 0


class TestAuditLoggerIntegration:
    """Verify audit logger records events across lifecycle."""

    def test_audit_logger_records_events(self):
        al = audit_mod.AuditLogger()
        al.log(
            event_type=audit_mod.EventType.MODEL_TRAINING,
            actor="system",
            resource="global_model",
            action="training started",
            details={"trial_id": "TEST-001"},
        )
        al.log(
            event_type=audit_mod.EventType.MODEL_AGGREGATION,
            actor="coordinator",
            resource="global_model",
            action="round completed",
            details={"round": 1, "loss": 0.5},
        )
        events = al.get_events()
        assert len(events) == 2

    def test_consent_manager_records(self):
        cm = consent_mod.ConsentManager()
        cm.register_consent(
            patient_id="PT-001",
            study_id="STUDY-001",
            consent_type=consent_mod.ConsentType.STUDY_SPECIFIC,
        )
        trail = cm.get_audit_trail()
        assert len(trail) >= 1


class TestEvaluationPipeline:
    """Verify trained model can produce evaluation metrics."""

    def test_model_predict_produces_labels(self):
        model = model_mod.FederatedModel(input_dim=10, output_dim=2, seed=42)
        x = np.random.randn(20, 10)
        preds = model.predict(x)
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1})

    def test_model_forward_sums_to_one(self):
        model = model_mod.FederatedModel(input_dim=10, output_dim=2, seed=42)
        x = np.random.randn(10, 10)
        probs = model.forward(x)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
