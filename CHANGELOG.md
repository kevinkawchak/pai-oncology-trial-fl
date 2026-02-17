# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-17

### Added

- **Aggregation Strategies**
  - `federated/coordinator.py` — Added FedProx (Li et al., 2020) with configurable proximal term `mu` for handling non-IID client data.
  - `federated/coordinator.py` — Added SCAFFOLD (Karimireddy et al., 2020) with server and client control variates for correcting client drift.
  - `federated/coordinator.py` — `AggregationStrategy` enum for selecting between `fedavg`, `fedprox`, and `scaffold`.
  - `federated/coordinator.py` — Automatic convergence detection with configurable window and threshold.

- **Data Harmonization**
  - `federated/data_harmonization.py` — New module for multi-site data alignment: DICOM/FHIR vocabulary mapping, unit conversion (mm/cm, cGy/Gy, F/C, mg-dL/mmol-L), z-score and min-max feature normalisation, and batch record harmonization with reporting.

- **Site Enrollment Management**
  - `federated/site_enrollment.py` — New module managing multi-site enrollment lifecycle: enrollment, data-readiness validation, compliance checks, activation, suspension, withdrawal, reactivation, quality-weighted site selection, round participation tracking, and audit trail.

- **Enhanced Digital Twins**
  - `physical_ai/digital_twin.py` — Added logistic and Gompertz tumor growth models alongside the existing exponential model, selectable via the `growth_model` parameter.
  - `physical_ai/digital_twin.py` — Added `simulate_immunotherapy_response()` with stochastic response modelling using `immuno_sensitivity`.
  - `physical_ai/digital_twin.py` — Added `simulate_combination_therapy()` for sequential multi-modality treatment simulation (chemo + radiation + immunotherapy).
  - `physical_ai/digital_twin.py` — Added `simulate_with_uncertainty()` for Monte-Carlo uncertainty quantification with configurable parameter noise.

- **Framework Detection**
  - `physical_ai/framework_detection.py` — New module that detects installed simulation frameworks (Isaac Lab, MuJoCo, Gazebo, PyBullet) and recommends a training/validation/deployment pipeline.

- **Surgical Task Definitions**
  - `physical_ai/surgical_tasks.py` — New module with standardised surgical task definitions for oncology procedures (needle biopsy, tissue resection, specimen handling, lymph node dissection, ablation) including clinical accuracy thresholds, reward structures, and domain randomization parameters.
  - `physical_ai/surgical_tasks.py` — `SurgicalTaskEvaluator` for evaluating robotic procedure performance against clinical thresholds.

- **Role-Based Access Control**
  - `privacy/access_control.py` — New module with six roles (coordinator, site admin, researcher, data engineer, auditor, patient) and thirteen granular permissions. All access decisions are logged for HIPAA/GDPR audit compliance.

- **Breach Response Protocol**
  - `privacy/breach_response.py` — New module for automated breach indicator detection, auto-escalation, and incident lifecycle management (detection, investigation, containment, resolution, regulatory reporting) compliant with HIPAA 72-hour and GDPR notification requirements.

- **FDA Submission Tracking**
  - `regulatory/fda_submission.py` — New module managing FDA submission packages for AI/ML-enabled medical devices. Supports Pre-Submission, De Novo, 510(k), PMA, and Breakthrough Device pathways with standard document checklists and readiness reporting.

- **New Tests**
  - `tests/test_data_harmonization.py` — Tests for vocabulary mapping, unit conversion, feature normalisation, and batch harmonization.
  - `tests/test_site_enrollment.py` — Tests for enrollment lifecycle, activation validation, quality-based auto-suspension, and round site selection.
  - Additional tests across existing files for FedProx, SCAFFOLD, convergence detection, growth models, combination therapy, uncertainty quantification, framework detection, surgical task evaluation, access control, and breach response.

### Changed

- `federated/coordinator.py` — Refactored to support pluggable aggregation strategies. Added `strategy`, `mu`, `convergence_window`, and `convergence_threshold` parameters. `RoundResult` now includes `elapsed_seconds` and `converged` fields.
- `federated/client.py` — Added `mu` parameter to `train_local()` for FedProx proximal term. Added `train_local_scaffold()` method. Added `get_training_history()` accessor.
- `physical_ai/digital_twin.py` — `TumorModel` gained `immuno_sensitivity`, `growth_model`, `carrying_capacity`, and `histology` fields. `PatientDigitalTwin` now supports combination therapy and uncertainty quantification.
- `examples/run_federation.py` — Added `--strategy` and `--mu` CLI arguments, site enrollment step, convergence display, and early stopping.
- `docker-compose.yml` — Added `fedprox` service profile for FedProx strategy simulation.
- `pyproject.toml` — Version bumped to 0.2.0. Development status upgraded to Beta. Added `digital-twins` and `surgical-robotics` keywords.
- `CITATION.cff` — Updated to v0.2.0 with expanded keywords and abstract.

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
