# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-02-17

### Added

- **Core Examples** (`examples/`)
  - `01_federated_training_workflow.py` — End-to-end federated training workflow with site enrollment, data harmonization, multi-round training, convergence monitoring, and cross-site evaluation.
  - `02_digital_twin_planning.py` — Digital twin treatment planning demonstrating patient twin creation, multi-modality treatment simulation, uncertainty quantification, and clinical report generation.
  - `03_cross_framework_validation.py` — Cross-framework policy validation across Isaac Sim, MuJoCo, and PyBullet with roundtrip conversion testing and state equivalence verification.
  - `04_agentic_clinical_workflow.py` — AI-driven clinical workflow using the unified agent interface for treatment planning, literature review, and compliance checking.
  - `05_outcome_prediction_pipeline.py` — End-to-end prediction pipeline from patient data loading through digital twin feature generation, federated model training, evaluation, and clinical reporting.
  - `examples/README.md` with overview table and quick start.

- **Physical AI Examples** (`examples-physical-ai/`)
  - `01_realtime_safety_monitoring.py` — Real-time IEC 80601-2-77 safety monitoring with force limits, workspace boundaries, emergency stop logic, and telemetry streaming.
  - `02_multi_sensor_fusion.py` — Multi-sensor fusion combining force/torque sensors, cameras, joint encoders, and patient monitoring with extended Kalman filter simulation.
  - `03_ros2_surgical_deployment.py` — ROS 2 surgical deployment pattern with node lifecycle, service discovery, and action server-based procedure execution.
  - `04_hand_eye_calibration.py` — AX=XB hand-eye calibration for surgical robots with transformation chain validation and statistical calibration error analysis.
  - `05_shared_autonomy_teleoperation.py` — Five levels of human-AI shared autonomy (manual, assisted, shared, supervised, autonomous) with safety gate transitions.
  - `06_robotic_sample_handling.py` — Automated specimen handling with 21 CFR Part 11 audit trails, chain of custody, and electronic signature simulation.
  - `examples-physical-ai/README.md` with regulatory standards cross-reference.

- **Agentic AI Examples** (`agentic-ai/examples-agentic-ai/`)
  - `01_mcp_oncology_server.py` — MCP server exposing oncology tools/resources for LLM agents with RobotMode/ProcedurePhase Enums, telemetry dataclass, and 21 CFR Part 11 audit logging.
  - `02_react_treatment_planner.py` — ReAct agent with thought/action/observation loops for iterative treatment planning using digital twin simulations.
  - `03_realtime_adaptive_monitoring_agent.py` — Real-time adaptive agent processing streaming multi-modal clinical data with cross-modal correlation and adaptive alerting.
  - `04_autonomous_simulation_orchestrator.py` — Multi-framework simulation orchestration agent managing jobs across Isaac Sim, MuJoCo, and PyBullet.
  - `05_safety_constrained_agent_executor.py` — Safety-gated agent with IEC 80601-2-77 / ISO 14971 alignment, pre/post-condition checks, human-in-the-loop, emergency stop, and rollback.
  - `06_oncology_rag_compliance_agent.py` — RAG agent grounded in FDA guidance, ICH E6(R3), and IEC standards with protocol citation tracking.
  - `agentic-ai/examples-agentic-ai/README.md` with example table and quick start.

### Changed

- `README.md` — Updated to v0.5.0, added all 17 example scripts to repository structure tree.
- `pyproject.toml` — Version bumped to 0.5.0.
- `CITATION.cff` — Updated to v0.5.0.
- `prompts.md` — Added Prompt 5 (v0.5.0) for domain examples and engineering demonstrations.
- `ruff.toml` — Per-file-ignores already covered examples-physical-ai/ and agentic-ai/ directories.

## [0.4.0] - 2026-02-17

### Added

- **Digital Twins Primary Domain** (`digital-twins/`)
  - `patient-modeling/patient_model.py` — Patient digital twin modeling engine with TumorType Enum (NSCLC, SCLC, BREAST_DUCTAL, BREAST_LOBULAR, COLORECTAL, PANCREATIC, etc.), GrowthModel Enum (EXPONENTIAL, LOGISTIC, GOMPERTZ, VON_BERTALANFFY), PatientStatus Enum, TumorProfile/PatientBiomarkers/OrganFunction/PatientDigitalTwinConfig dataclasses, PatientPhysiologyEngine with tumor growth kinetics and biomarker evolution, PatientDigitalTwinFactory.
  - `treatment-simulation/treatment_simulator.py` — Treatment simulation with TreatmentModality Enum (CHEMOTHERAPY, RADIATION, IMMUNOTHERAPY, TARGETED_THERAPY, COMBINATION), ResponseCategory Enum (RECIST criteria), ToxicityGrade Enum (CTCAE grades 0–5), ChemotherapyProtocol/RadiationPlan/ImmunotherapyRegimen/TreatmentOutcome/ToxicityProfile dataclasses, TreatmentSimulator with pharmacokinetic/pharmacodynamic modeling.
  - `clinical-integration/clinical_integrator.py` — Clinical system integration with ClinicalSystem Enum (EHR, PACS, LIS, RIS, TPS), IntegrationStatus/DataFormat Enums, ClinicalDataPoint/IntegrationConfig/SyncResult/ClinicalReport dataclasses, ClinicalIntegrator for data ingestion and report generation, FederatedClinicalBridge connecting digital twins to the federated learning pipeline.
  - Subdirectory README.md files with purpose, API overview, and usage examples.

- **CLI Tools** (`tools/`)
  - `dicom-inspector/dicom_inspector.py` — DICOM medical image inspection with argparse subcommands (inspect, validate, summarize), DicomModality Enum, DicomInspectionResult dataclass, JSON output support.
  - `dose-calculator/dose_calculator.py` — Radiation and chemotherapy dose calculator with reference constants citing NCCN guidelines and RTOG protocols, DoseUnit/ChemoAgent/RadiationTechnique Enums, DoseCalculation/PatientParameters dataclasses, all doses bounded.
  - `trial-site-monitor/trial_site_monitor.py` — Multi-site clinical trial monitoring with SiteStatus/EnrollmentStatus/DataQualityStatus Enums (GREEN/YELLOW/RED levels), SiteMetrics/MonitoringReport dataclasses.
  - `sim-job-runner/sim_job_runner.py` — Cross-framework simulation job launcher with SimFramework/JobStatus/TaskType Enums, SimJobConfig/SimJobResult dataclasses, framework auto-detection.
  - `deployment-readiness/deployment_readiness.py` — Deployment readiness checker with CheckStatus Enum (PASS/FAIL/WARNING/REQUIRES_VERIFICATION), ReadinessCheck/ReadinessReport dataclasses, validation across model, safety, regulatory, infrastructure, and documentation categories.
  - `tools/README.md` with directory tree, per-tool CLI usage summaries, and common workflows.

### Changed

- `README.md` — Updated to v0.4.0, added digital-twins/ and tools/ to repository structure tree.
- `pyproject.toml` — Version bumped to 0.4.0.
- `CITATION.cff` — Updated to v0.4.0.
- `prompts.md` — Added Prompt 4 (v0.4.0) for core domain modules and CLI tools.

## [0.3.0] - 2026-02-17

### Added

- **Unification Framework** (`unification/`)
  - `unification/simulation_physics/isaac_mujoco_bridge.py` — Bidirectional state conversion between NVIDIA Isaac Sim and MuJoCo with dataclass configs, Enum states, conditional imports, and structured logging.
  - `unification/simulation_physics/physics_parameter_mapping.yaml` — Cross-framework physics parameter equivalences (gravity, friction, damping, solver settings).
  - `unification/agentic_generative_ai/unified_agent_interface.py` — Unified agent interface with AgentBackend Enum (CrewAI/LangGraph/AutoGen/Custom), AgentRole Enum, Tool dataclass with `to_mcp_format()`, `to_openai_format()`, `to_anthropic_format()` methods.
  - `unification/cross_platform_tools/framework_detector.py` — @dataclass FrameworkInfo + SystemInfo, CLI with `--verbose`/`--json` flags.
  - `unification/cross_platform_tools/model_converter.py` — Cross-platform model format conversion (PyTorch, ONNX, SafeTensors, TorchScript).
  - `unification/cross_platform_tools/policy_exporter.py` — RL policy export across simulation frameworks.
  - `unification/cross_platform_tools/validation_suite.py` — Enum-based validation result states, cross-platform policy validation.
  - `unification/standards_protocols/` — Data formats, communication protocols, and safety standards documentation.
  - `unification/integration_workflows/workflow_templates.yaml` — Integration workflow templates.
  - `unification/surgical_robotics/` — Challenges and opportunities in unified surgical robotics.
  - `unification/README.md` — Core principles, directory structure, quarterly roadmap (Q1–Q4 2026).

- **Q1 2026 Standards** (`q1-2026-standards/`)
  - `objective-1-model-conversion/conversion_pipeline.py` — Model format conversion pipeline with Enum-based format types, ConversionConfig/ConversionResult dataclasses.
  - `objective-2-model-registry/model_registry.yaml` — Schema for federated model registry entries.
  - `objective-2-model-registry/model_validator.py` — Model validation with ValidationStatus Enum, ModelMetadata dataclass.
  - `objective-3-benchmarking/benchmark_runner.py` — Cross-platform benchmark runner with BenchmarkMetric Enum, latency/accuracy/memory tests.
  - `implementation-guide/timeline.md` — Q1 2026 implementation milestones.
  - `implementation-guide/compliance_checklist.md` — Standards adoption compliance checklist.
  - `q1-2026-standards/README.md` — Three objectives with status and deliverables.

- **Framework Integration Guides** (`frameworks/`)
  - `nvidia-isaac/INTEGRATION.md` — NVIDIA Isaac Sim/Lab system requirements, installation, code snippets.
  - `mujoco/INTEGRATION.md` — MuJoCo integration with physics parameter tuning for medical robotics.
  - `gazebo/INTEGRATION.md` — Gazebo + ROS 2 Humble/Iron integration for clinical robotics.
  - `pybullet/INTEGRATION.md` — PyBullet rapid prototyping guide.

- **Learning Domain Directories**
  - `supervised-learning/` — Strengths, limitations, and illustrative results for supervised learning in oncology.
  - `reinforcement-learning/` — Strengths, limitations, and illustrative results for RL in surgical robotics and treatment planning.
  - `self-supervised-learning/` — Strengths, limitations, and illustrative results for SSL in medical imaging.
  - `generative-ai/` — Strengths, limitations, and illustrative results for generative AI in oncology.

- **Configuration** (`configs/`)
  - `training_config.yaml` — Comprehensive training configuration with environment, domain randomization, PPO/SAC algorithms, federated learning parameters, network architecture, safety constraints, curriculum learning, evaluation metrics, deployment, and sim-to-real transfer settings.

- **Verification** (`scripts/`)
  - `verify_installation.py` — importlib-based dependency verification with colored output, version validation, and graceful handling of missing optional packages.

- **Community Health**
  - `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1 adapted for clinical software (patient safety, data privacy, scientific integrity).
  - `CONTRIBUTING.md` — Contribution types, recency/oncology/reproducibility requirements, development workflow, benchmark data policy, safety/compliance section.
  - `SECURITY.md` — Vulnerability reporting (7-day acknowledgment / 30-day fix SLA), supported versions, deployer responsibilities.
  - `SUPPORT.md` — Getting help, upstream framework support table, regulatory disclaimer.
  - `.github/PULL_REQUEST_TEMPLATE.md` — Summary, 7 change type categories, safety/compliance checklist.
  - `.github/ISSUE_TEMPLATE/bug_report.md` — Environment, reproduction steps, expected/actual behavior.
  - `.github/ISSUE_TEMPLATE/feature_request.md` — Domain category checkboxes, clinical relevance, implementation notes.

- **CI/CD Enhancements**
  - `.github/workflows/ci.yml` — Added yamllint for YAML validation, validate-scripts job, explicit `permissions: contents read`, ruff `--output-format=github`.
  - `ruff.toml` — Standalone ruff configuration with line-length 120, per-file-ignores for all directories.

- **Root Files**
  - `posts.md` — Placeholder for external references and publications.
  - `releases.md/v0.3.0.md` — Release notes for v0.3.0.

### Changed

- `README.md` — Added badges (MIT license, v0.3.0, Python 3.10+), responsible-use notice, repository structure tree with all new directories, Core Technologies table, Unification Framework section, Multi-Organization Cooperation table, expanded dependency listing, contributing guidelines summary.
- `requirements.txt` — Expanded with grouped dependencies (Core, Deep Learning, Medical Imaging, Agentic AI, Visualization, Deployment, Testing), pinned versions with release dates and GitHub URLs.
- `pyproject.toml` — Version bumped to 0.3.0. Ruff configuration moved to standalone `ruff.toml`. Added `unification*` to setuptools package discovery.
- `CITATION.cff` — Updated to v0.3.0, added `unification-framework`, `agentic-ai`, `simulation-physics`, `cross-platform` keywords.
- `.github/workflows/ci.yml` — Enhanced with yamllint, validate-scripts job, permissions, output format.

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
