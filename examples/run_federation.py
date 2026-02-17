#!/usr/bin/env python3
"""Run a complete federated learning simulation.

Demonstrates a multi-site oncology trial where several hospitals
collaboratively train an AI model without sharing raw patient data.
Includes site enrollment validation, compliance checks, consent
management, and support for FedAvg/FedProx/SCAFFOLD strategies.

Usage:
    python examples/run_federation.py
    python examples/run_federation.py --num-sites 4 --rounds 20 --dp-epsilon 2.0
    python examples/run_federation.py --strategy fedprox --mu 0.01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated.client import FederatedClient
from federated.coordinator import FederationCoordinator
from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data
from federated.model import ModelConfig
from federated.site_enrollment import SiteEnrollmentManager
from privacy.audit_logger import AuditLogger, EventType
from privacy.consent_manager import ConsentManager
from regulatory.compliance_checker import ComplianceChecker


def main(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  Physical AI Federated Oncology Trial Simulation")
    print(f"  Strategy: {args.strategy.upper()}")
    print("=" * 70)

    # --- Model configuration ---
    config = ModelConfig(
        input_dim=30,
        hidden_dims=[64, 32],
        output_dim=2,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # --- Compliance check ---
    print("\n[1/7] Running regulatory compliance checks...")
    checker = ComplianceChecker()
    federation_config = {
        "use_differential_privacy": args.dp_epsilon is not None,
        "dp_epsilon": args.dp_epsilon or 0,
        "use_secure_aggregation": args.secure_agg,
        "use_deidentification": True,
        "audit_logging_enabled": True,
        "consent_management_enabled": True,
        "encryption_in_transit": True,
        "min_clients": args.num_sites,
    }
    report = checker.check_federation_config(federation_config)
    for check in report.checks:
        status_icon = {
            "pass": "PASS",
            "fail": "FAIL",
            "warning": "WARN",
        }.get(check.status.value, "    ")
        print(f"  [{status_icon}] {check.description}")
    print(f"  Overall: {report.overall_status.value.upper()}")

    # --- Site enrollment ---
    print("\n[2/7] Enrolling and activating hospital sites...")
    site_mgr = SiteEnrollmentManager("FL_TRIAL_001", min_patients_per_site=10)
    site_names = [
        "Memorial Cancer Center",
        "University Oncology Institute",
        "Regional Medical Center",
        "National Cancer Research Lab",
        "Community Oncology Clinic",
    ]
    for i in range(args.num_sites):
        name = site_names[i % len(site_names)]
        site_mgr.enroll_site(
            f"site_{i}",
            name,
            patient_count=200,
            capabilities=["imaging", "biopsy"],
        )
        site_mgr.mark_data_ready(f"site_{i}")
        site_mgr.mark_compliance_passed(f"site_{i}")
        site_mgr.activate_site(f"site_{i}")
    summary = site_mgr.get_enrollment_summary()
    print(f"  {summary['total_sites']} sites enrolled, {summary['total_patients']} patients")

    # --- Consent management ---
    print("\n[3/7] Registering patient consent...")
    consent_mgr = ConsentManager()
    for i in range(args.num_sites):
        for j in range(50):
            consent_mgr.register_consent(f"patient_{i}_{j}", "FL_TRIAL_001")
    consented = consent_mgr.get_consented_patients("FL_TRIAL_001")
    print(f"  {len(consented)} patients consented across {args.num_sites} sites")

    # --- Data generation and partitioning ---
    print("\n[4/7] Generating synthetic oncology data...")
    n_samples = args.num_sites * 200
    x, y = generate_synthetic_oncology_data(
        n_samples=n_samples, n_features=30, n_classes=2, seed=args.seed
    )
    partitioner = DataPartitioner(num_sites=args.num_sites, strategy="iid", seed=args.seed)
    sites = partitioner.partition(x, y)
    for site in sites:
        print(f"  {site.site_id}: {site.num_train} train, {site.num_test} test")

    # --- Initialize coordinator and clients ---
    print(f"\n[5/7] Initializing coordinator (strategy={args.strategy})...")
    coordinator = FederationCoordinator(
        model_config=config,
        num_rounds=args.rounds,
        min_clients=args.num_sites,
        strategy=args.strategy,
        mu=args.mu,
        use_secure_aggregation=args.secure_agg,
        use_differential_privacy=args.dp_epsilon is not None,
        dp_epsilon=args.dp_epsilon or 1.0,
    )
    global_params = coordinator.initialize()

    audit = AuditLogger()
    audit.log(
        EventType.SYSTEM_EVENT,
        actor="coordinator",
        resource="federation",
        action="initialized",
    )

    clients = []
    for site in sites:
        client = FederatedClient(site.site_id, config)
        client.set_data(site.x_train, site.y_train)
        clients.append(client)

    # --- Federated training ---
    print(f"\n[6/7] Running {args.rounds} federated training rounds...")
    print(f"  {'Round':>5}  {'Clients':>7}  {'Accuracy':>8}  {'Loss':>8}  {'Converged':>9}")
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 9}")

    for round_num in range(args.rounds):
        client_updates = []
        sample_counts = []

        for client in clients:
            result = client.train_local(
                global_params, epochs=args.local_epochs, lr=args.lr, mu=args.mu
            )
            client_updates.append(client.get_parameters())
            sample_counts.append(client.get_sample_count())
            audit.log(
                EventType.MODEL_TRAINING,
                actor=client.client_id,
                resource="local_model",
                action=f"trained epoch {result.epochs_completed}",
            )

        eval_data = (sites[0].x_test, sites[0].y_test)
        round_result = coordinator.run_round(
            client_updates,
            client_sample_counts=sample_counts,
            eval_data=eval_data,
        )
        global_params = coordinator.get_global_parameters()

        acc = round_result.global_metrics.get("accuracy", 0)
        loss = round_result.global_metrics.get("loss", 0)
        conv = "Yes" if round_result.converged else ""
        print(
            f"  {round_num + 1:>5}  {round_result.num_clients:>7}"
            f"  {acc:>8.4f}  {loss:>8.4f}  {conv:>9}"
        )

        if round_result.converged and round_num >= 5:
            print(f"\n  Converged at round {round_num + 1}.")
            break

    # --- Summary ---
    print("\n[7/7] Training complete!")
    training_summary = coordinator.get_training_summary()
    final = coordinator.global_model.evaluate(sites[0].x_test, sites[0].y_test)
    print(f"  Strategy:       {training_summary['strategy']}")
    print(f"  Total rounds:   {training_summary['total_rounds']}")
    print(f"  Final accuracy: {final['accuracy']:.4f}")
    print(f"  Final loss:     {final['loss']:.4f}")

    if coordinator.dp:
        privacy = coordinator.dp.get_privacy_spent()
        print(f"  Privacy spent:  epsilon={privacy['total_epsilon_spent']:.2f}")

    audit_report = audit.generate_report()
    print(f"  Audit events:   {audit_report['total_events']}")

    print("\n" + "=" * 70)
    print("  Simulation complete. No raw patient data was shared.")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated oncology trial simulation")
    parser.add_argument("--num-sites", type=int, default=3, help="Number of hospital sites")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--dp-epsilon", type=float, default=None, help="DP epsilon (None=off)")
    parser.add_argument("--secure-agg", action="store_true", help="Enable secure aggregation")
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox", "scaffold"],
        default="fedavg",
        help="Aggregation strategy",
    )
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx proximal term mu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
