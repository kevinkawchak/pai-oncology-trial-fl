# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-17

### Added

- **Federated Learning Framework**
  - `federated/coordinator.py` — Federation coordinator implementing FedAvg with configurable rounds, client weighting, and evaluation.
  - `federated/client.py` — Simulated hospital nodes with local training, parameter exchange, and sample counting.
  - `federated/model.py` — Numpy-based multilayer perceptron with forward/backward passes, parameter extraction/injection, and model evaluation.
  - `federated/secure_aggregation.py` — Mask-based secure aggregation protocol ensuring the coordinator only sees aggregate updates.
  - `federated/differential_privacy.py` — Gaussian mechanism with gradient clipping, calibrated noise injection, and privacy budget tracking.
  - `federated/data_ingestion.py` — Synthetic oncology data generation and IID/non-IID partitioning across sites.

- **Physical AI Integration**
  - `physical_ai/digital_twin.py` — Patient digital twin with tumor growth simulation, chemotherapy/radiation/immunotherapy response modeling, and feature vector generation for federated training.
  - `physical_ai/robotic_integration.py` — Surgical robot interface supporting procedure planning, execution simulation, and de-identified telemetry feature extraction.
  - `physical_ai/sensor_fusion.py` — Multi-modal clinical sensor fusion with real-time anomaly detection (Z-score based).
  - `physical_ai/simulation_bridge.py` — Cross-platform robot model conversion (URDF, MJCF, SDF, USD) with physics validation.

- **Privacy Infrastructure**
  - `privacy/phi_detector.py` — PHI detection for the 18 HIPAA identifiers using regex and field-name heuristics.
  - `privacy/deidentification.py` — De-identification pipeline with redact, hash, generalize, and replace strategies.
  - `privacy/consent_manager.py` — Patient consent and DUA management with audit trail.
  - `privacy/audit_logger.py` — HIPAA-compliant audit logging with integrity hashing and breach detection.
  - `privacy/dua_templates/` — Standard and multi-site Data Use Agreement templates.

- **Regulatory Compliance**
  - `regulatory/compliance_checker.py` — Automated HIPAA, GDPR, and FDA 21 CFR Part 11 compliance validation.
  - `regulatory/templates/` — IRB protocol and informed consent templates.

- **CI/CD**
  - GitHub Actions workflow with ruff linting and formatting checks across Python 3.10, 3.11, and 3.12.
  - Automated test suite execution.

- **Tests**
  - Comprehensive unit tests for all modules (`tests/`).
  - End-to-end tests verifying multi-site federation, DP integration, secure aggregation equivalence, digital twin federation, and privacy pipeline.

- **Examples & Documentation**
  - `examples/run_federation.py` — Complete federation simulation script with CLI arguments.
  - `examples/generate_synthetic_data.py` — Synthetic data generation and export to CSV.
  - `docs/federated_training_demo.ipynb` — Jupyter notebook demonstrating federated training, centralized comparison, DP impact, and digital twin integration.
  - `scripts/deploy.sh` — Docker-based deployment script.

- **Infrastructure**
  - `Dockerfile` and `docker-compose.yml` for containerized execution.
  - `environment.yml` for Conda-based reproducibility.
  - `pyproject.toml` with full project metadata and tool configuration.
  - `CITATION.cff` for academic citation.
  - MIT License.
