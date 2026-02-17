#!/usr/bin/env python3
"""Run a complete federated learning simulation.

Demonstrates a multi-site oncology trial where several hospitals
collaboratively train an AI model without sharing raw patient data.

Usage:
    python examples/run_federation.py
    python examples/run_federation.py --num-sites 4 --rounds 20 --dp-epsilon 2.0
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
from privacy.audit_logger import AuditLogger, EventType
from privacy.consent_manager import ConsentManager
from regulatory.compliance_checker import ComplianceChecker


def main(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  Physical AI Federated Oncology Trial Simulation")
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
    print("\n[1/6] Running regulatory compliance checks...")
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

    # --- Consent management ---
    print("\n[2/6] Registering patient consent...")
    consent_mgr = ConsentManager()
    for i in range(args.num_sites):
        for j in range(50):
            consent_mgr.register_consent(f"patient_{i}_{j}", "FL_TRIAL_001")
    consented = consent_mgr.get_consented_patients("FL_TRIAL_001")
    print(f"  {len(consented)} patients consented across {args.num_sites} sites")

    # --- Data generation and partitioning ---
    print("\n[3/6] Generating synthetic oncology data...")
    n_samples = args.num_sites * 200
    x, y = generate_synthetic_oncology_data(
        n_samples=n_samples, n_features=30, n_classes=2, seed=args.seed
    )
    partitioner = DataPartitioner(num_sites=args.num_sites, strategy="iid", seed=args.seed)
    sites = partitioner.partition(x, y)
    for site in sites:
        print(f"  {site.site_id}: {site.num_train} train, {site.num_test} test")

    # --- Initialize coordinator and clients ---
    print("\n[4/6] Initializing federated coordinator and clients...")
    coordinator = FederationCoordinator(
        model_config=config,
        num_rounds=args.rounds,
        min_clients=args.num_sites,
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
    print(f"\n[5/6] Running {args.rounds} federated training rounds...")
    print(f"  {'Round':>5}  {'Clients':>7}  {'Accuracy':>8}  {'Loss':>8}")
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 8}  {'-' * 8}")

    for round_num in range(args.rounds):
        client_updates = []
        sample_counts = []

        for client in clients:
            result = client.train_local(global_params, epochs=args.local_epochs, lr=args.lr)
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
        print(f"  {round_num + 1:>5}  {round_result.num_clients:>7}  {acc:>8.4f}  {loss:>8.4f}")

    # --- Summary ---
    print("\n[6/6] Training complete!")
    summary = coordinator.get_training_summary()
    final = coordinator.global_model.evaluate(sites[0].x_test, sites[0].y_test)
    print(f"  Total rounds:  {summary['total_rounds']}")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
