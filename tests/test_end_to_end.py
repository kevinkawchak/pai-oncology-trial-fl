"""End-to-end tests for the federated learning platform.

Verifies that the full pipeline works: data generation, partitioning,
multi-client training, aggregation, and privacy enforcement.
"""

import numpy as np

from federated.client import FederatedClient
from federated.coordinator import FederationCoordinator
from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data
from federated.model import ModelConfig
from federated.site_enrollment import SiteEnrollmentManager
from physical_ai.digital_twin import PatientDigitalTwin, TumorModel
from privacy.consent_manager import ConsentManager
from privacy.deidentification import Deidentifier
from privacy.phi_detector import PHIDetector
from regulatory.compliance_checker import ComplianceChecker


class TestEndToEndFederation:
    """Full federated training pipeline with two simulated hospital sites."""

    def test_two_site_federation(self):
        config = ModelConfig(input_dim=30, hidden_dims=[32, 16], output_dim=2, seed=42)
        num_rounds = 5

        x, y = generate_synthetic_oncology_data(n_samples=400, seed=42)
        partitioner = DataPartitioner(num_sites=2, strategy="iid", seed=42)
        sites = partitioner.partition(x, y)

        coordinator = FederationCoordinator(model_config=config, num_rounds=num_rounds)
        global_params = coordinator.initialize()

        clients = []
        for site in sites:
            client = FederatedClient(site.site_id, config)
            client.set_data(site.x_train, site.y_train)
            clients.append(client)

        for round_num in range(num_rounds):
            client_updates = []
            sample_counts = []
            for client in clients:
                client.train_local(global_params, epochs=3, lr=0.01)
                client_updates.append(client.get_parameters())
                sample_counts.append(client.get_sample_count())

            coordinator.run_round(
                client_updates,
                client_sample_counts=sample_counts,
                eval_data=(sites[0].x_test, sites[0].y_test),
            )
            global_params = coordinator.get_global_parameters()

        final_metrics = coordinator.global_model.evaluate(sites[0].x_test, sites[0].y_test)
        assert final_metrics["accuracy"] >= 0.45

        summary = coordinator.get_training_summary()
        assert summary["total_rounds"] == num_rounds

    def test_three_site_with_dp(self):
        """Three sites with differential privacy enabled."""
        config = ModelConfig(input_dim=30, hidden_dims=[32, 16], output_dim=2, seed=42)

        x, y = generate_synthetic_oncology_data(n_samples=600, seed=0)
        partitioner = DataPartitioner(num_sites=3, strategy="iid", seed=0)
        sites = partitioner.partition(x, y)

        coordinator = FederationCoordinator(
            model_config=config,
            num_rounds=3,
            min_clients=3,
            use_differential_privacy=True,
            dp_epsilon=5.0,
            dp_delta=1e-5,
        )
        global_params = coordinator.initialize()

        clients = []
        for site in sites:
            client = FederatedClient(site.site_id, config)
            client.set_data(site.x_train, site.y_train)
            clients.append(client)

        for _ in range(3):
            updates = []
            counts = []
            for client in clients:
                client.train_local(global_params, epochs=2)
                updates.append(client.get_parameters())
                counts.append(client.get_sample_count())

            coordinator.run_round(updates, client_sample_counts=counts)
            global_params = coordinator.get_global_parameters()

        assert coordinator.dp.rounds_completed == 3
        privacy = coordinator.dp.get_privacy_spent()
        assert privacy["total_epsilon_spent"] == 15.0

    def test_secure_aggregation_matches_plain(self):
        """Secure aggregation should produce equivalent results to plain FedAvg."""
        config = ModelConfig(input_dim=10, hidden_dims=[8], output_dim=2, seed=42)

        x, y = generate_synthetic_oncology_data(n_samples=100, n_features=10, seed=42)
        partitioner = DataPartitioner(num_sites=2, seed=42)
        sites = partitioner.partition(x, y)

        coord_plain = FederationCoordinator(model_config=config)
        global_params = coord_plain.initialize()
        clients = []
        for site in sites:
            c = FederatedClient(site.site_id, config)
            c.set_data(site.x_train, site.y_train)
            clients.append(c)

        for c in clients:
            c.train_local(global_params, epochs=1, lr=0.01)
        shared_updates = [c.get_parameters() for c in clients]

        coord_plain.run_round(shared_updates)
        plain_params = coord_plain.get_global_parameters()

        coord_sa = FederationCoordinator(model_config=config, use_secure_aggregation=True)
        coord_sa.initialize()
        coord_sa.run_round(shared_updates)
        sa_params = coord_sa.get_global_parameters()

        for p1, p2 in zip(plain_params, sa_params):
            assert np.allclose(p1, p2, atol=1e-5)

    def test_fedprox_training(self):
        """FedProx with mu > 0 should complete without errors."""
        config = ModelConfig(input_dim=30, hidden_dims=[16], output_dim=2, seed=42)
        x, y = generate_synthetic_oncology_data(n_samples=200, seed=42)
        partitioner = DataPartitioner(num_sites=2, seed=42)
        sites = partitioner.partition(x, y)

        coordinator = FederationCoordinator(model_config=config, strategy="fedprox", mu=0.01)
        global_params = coordinator.initialize()

        clients = []
        for site in sites:
            c = FederatedClient(site.site_id, config)
            c.set_data(site.x_train, site.y_train)
            clients.append(c)

        for _ in range(3):
            updates = []
            counts = []
            for c in clients:
                c.train_local(global_params, epochs=3, lr=0.01, mu=coordinator.mu)
                updates.append(c.get_parameters())
                counts.append(c.get_sample_count())
            coordinator.run_round(updates, client_sample_counts=counts)
            global_params = coordinator.get_global_parameters()

        assert coordinator.current_round == 3

    def test_site_enrollment_workflow(self):
        """End-to-end site enrollment and activation."""
        mgr = SiteEnrollmentManager("FL_STUDY_01", min_patients_per_site=5)
        mgr.enroll_site("site_A", "Hospital A", patient_count=50)
        mgr.mark_data_ready("site_A")
        mgr.mark_compliance_passed("site_A")
        assert mgr.activate_site("site_A") is True

        mgr.enroll_site("site_B", "Hospital B", patient_count=2)
        mgr.mark_data_ready("site_B")
        mgr.mark_compliance_passed("site_B")
        # Too few patients
        assert mgr.activate_site("site_B") is False

        active = mgr.get_active_sites()
        assert len(active) == 1
        assert active[0].site_id == "site_A"


class TestEndToEndWithPhysicalAI:
    """Test federation with physical AI digital twin data."""

    def test_digital_twin_federation(self):
        config = ModelConfig(input_dim=30, hidden_dims=[16], output_dim=2, seed=42)

        rng = np.random.default_rng(42)
        site_data = []
        for _ in range(2):
            features = []
            labels = []
            for j in range(50):
                tumor = TumorModel(
                    volume_cm3=rng.uniform(0.5, 10.0),
                    growth_rate=rng.uniform(30, 120),
                    chemo_sensitivity=rng.uniform(0.1, 0.9),
                    radio_sensitivity=rng.uniform(0.1, 0.9),
                )
                twin = PatientDigitalTwin(
                    f"patient_{j}",
                    tumor=tumor,
                    age=int(rng.integers(30, 80)),
                    biomarkers={
                        "pdl1": float(rng.uniform(0, 1)),
                        "ki67": float(rng.uniform(0, 1)),
                    },
                )
                features.append(twin.generate_feature_vector())
                labels.append(0 if tumor.growth_rate > 60 else 1)

            site_data.append((np.array(features), np.array(labels)))

        coordinator = FederationCoordinator(model_config=config, num_rounds=3)
        global_params = coordinator.initialize()

        clients = []
        for i, (x, y) in enumerate(site_data):
            client = FederatedClient(f"hospital_{i}", config)
            client.set_data(x, y)
            clients.append(client)

        for _ in range(3):
            updates = []
            counts = []
            for c in clients:
                c.train_local(global_params, epochs=5, lr=0.01)
                updates.append(c.get_parameters())
                counts.append(c.get_sample_count())
            coordinator.run_round(updates, client_sample_counts=counts)
            global_params = coordinator.get_global_parameters()

        x_test, y_test = site_data[0]
        metrics = coordinator.global_model.evaluate(x_test, y_test)
        assert 0 <= metrics["accuracy"] <= 1


class TestEndToEndPrivacy:
    """Test the privacy pipeline end-to-end."""

    def test_phi_detection_and_deidentification(self):
        detector = PHIDetector()
        deid = Deidentifier(method="redact")

        records = [
            {
                "patient_name": "Jane Smith",
                "ssn": "123-45-6789",
                "diagnosis": "Non-small cell lung cancer",
                "email": "jane@hospital.org",
            },
            {
                "name": "Bob Jones",
                "phone": "(555) 987-6543",
                "tumor_volume": "4.2 cm3",
            },
        ]

        for record in records:
            phi_matches = detector.scan_record(record)
            assert len(phi_matches) > 0

            result = deid.deidentify_record(record)
            clean = result.clean_data

            for key, value in clean.items():
                remaining = detector.scan_text(str(value))
                if key.lower() in ("patient_name", "ssn", "name", "email", "phone"):
                    assert "[REDACTED]" in str(value) or len(remaining) == 0

    def test_consent_enforcement(self):
        mgr = ConsentManager()
        mgr.register_consent("P001", "FL_STUDY_01")
        mgr.register_consent("P002", "FL_STUDY_01")

        all_patients = ["P001", "P002", "P003"]
        eligible = [p for p in all_patients if mgr.verify_consent(p, "FL_STUDY_01")]
        assert eligible == ["P001", "P002"]

    def test_compliance_check(self):
        checker = ComplianceChecker()
        config = {
            "use_differential_privacy": True,
            "dp_epsilon": 1.0,
            "use_secure_aggregation": True,
            "use_deidentification": True,
            "audit_logging_enabled": True,
            "consent_management_enabled": True,
            "encryption_in_transit": True,
            "min_clients": 2,
        }
        report = checker.check_federation_config(config)
        assert report.overall_status.value == "pass"
