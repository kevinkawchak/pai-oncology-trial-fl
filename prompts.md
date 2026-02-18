# Prompts Used in Development

This document records the prompts used to develop the PAI Oncology Trial FL platform.

---

## Prompt 6: v0.6.0 — Privacy Framework and Regulatory Compliance Infrastructure

Consolidates: v0.5.0 (privacy: PHI/PII, de-identification, access control, breach response, DUA + regulatory: FDA, IRB, ICH-GCP, regulatory intelligence)

Dependencies: Prompt 1

```
Build the complete privacy and regulatory compliance infrastructure for pai-oncology-trial-fl.
1. Privacy Framework (privacy/)
README.md: Version, overview, directory tree, HIPAA alignment, regulatory cross-references (HIPAA, 21 CFR Part 11, GDPR, EU AI Act, NIST Privacy Framework, ICH E6(R3)), all 18 HIPAA identifiers listed.
* phi-pii-management/: README.md + phi_detector.py (500-700 LOC). PHICategory Enum mapping all 18 Safe Harbor identifiers per 45 CFR 164.514(b)(2). @dataclass PHIFinding (phi_type, location, value_preview redacted, confidence, risk_level). PHIDetector class: regex patterns for SSN, MRN, phone, email, dates, geographic, IP, ZIP; DICOM tag scanning (conditional pydicom import); optional Presidio integration; batch scanning with ScanResult @dataclass. Non-destructive logging (never log full PHI values).
* de-identification/: README.md + deidentification_pipeline.py (500-700 LOC). DeidentificationMethod Enum (REDACT, HASH, GENERALIZE, SUPPRESS, PSEUDONYMIZE, DATE_SHIFT). HMAC-SHA256 pseudonymization with cryptographically random salt via os.urandom (NOT weak default). Date shifting with consistent patient-level offsets. Geographic generalization (ZIP to 3-digit). Transformation audit trail.
* access-control/: README.md + access_control_manager.py (400-600 LOC). Role-Based Access Control with AccessRole, Permission, AuditAction Enums. 21 CFR Part 11 audit trail. get_audit_log() must return a copy (not mutable reference). Invalid date formats must deny access by default.
* breach-response/: README.md + breach_response_protocol.py (400-600 LOC). BreachSeverity Enum. RiskAssessment with calculate_risk() that clamps out-of-range scores. 60-day notification timeline. Remediation tracking.
* dua-templates/: README.md + dua_generator.py (300-500 LOC). DUAType Enum. Templates for intra-institutional, multi-site, commercial. Data retention periods. Security requirements (encryption, MFA).
2. Regulatory Framework (regulatory/)
README.md: Version, regulatory landscape overview, directory tree, standards cross-reference, FDA device statistics.
* fda-compliance/: README.md + fda_submission_tracker.py (500-700 LOC). SubmissionType Enum (510k, De_Novo, PMA, Breakthrough). SubmissionStatus Enum. DeviceClass Enum. AI/ML components default to model_type="unspecified" (not "classification"). Reference FDA draft guidance for AI devices (Jan 2025), PCCP guidance (Aug 2025), QMSR (Feb 2026).
* irb-management/: README.md + irb_protocol_manager.py (400-600 LOC). Use from __future__ import annotations for forward references. Protocol lifecycle, amendment management, consent versioning.
* ich-gcp/: README.md + gcp_compliance_checker.py (400-600 LOC). Compliance score must exclude NOT_ASSESSED findings from denominator. Reference ICH E6(R3) published Sep 2025.
* regulatory-intelligence/: README.md + regulatory_tracker.py (400-600 LOC). Multi-jurisdiction monitoring (FDA, EMA, PMDA, TGA, Health Canada). get_recent_updates() must actually use computed cutoff date. Overdue status must be reachable (correct if/elif ordering: check past-due BEFORE imminent).
* human-oversight/: HUMAN_OVERSIGHT_QMS.md: CRF auto-fill risk tiers (low/medium/high with different automation levels), AE automation boundaries (detection: auto; grading: draft; causality: advisory; submission: prohibited without e-signature), safety gates (SAE immediate escalation, no autonomous regulatory submission, >95% sensitivity for AE detection).
3. Updates: Add per-file-ignores (regulatory/**/*.py = ["F401", "F821"], privacy/**/*.py = ["F401"]). Update README.md.
Quality gates: All files pass ruff check + ruff format --check. HIPAA cites specific 45 CFR sections. FDA cites guidance documents with dates. ICH cites E6(R3). IEC cites 62304/80601-2-77. ISO cites 14971. 21 CFR Part 11 in all audit trails. No actual PHI in any file. Pseudonymization uses HMAC-SHA256 with os.urandom.
```

---

## Prompt 5: v0.5.0 — Domain Examples and Engineering Demonstrations

Consolidates: v0.6.0 (5 core examples) + v0.7.0 (6 physical AI examples) + v0.9.0 (6 agentic AI examples)

Dependencies: Prompt 4

```
Build 17 production-quality example scripts across three domains for pai-oncology-trial-fl.
1. Core Examples (examples/) — README.md, 5 scripts (800-1000 LOC each):
   - 01_federated_training_workflow.py: End-to-end FL pipeline with coordinator setup, FedAvg/FedProx aggregation, DP integration, secure aggregation, convergence monitoring.
   - 02_digital_twin_planning.py: Patient digital twin construction from imaging data, tumor growth modeling (exponential/logistic/Gompertz), treatment response simulation, multi-organ tracking.
   - 03_cross_framework_validation.py: Policy validation across Isaac Sim/MuJoCo/PyBullet/Gazebo, kinematic drift analysis, force comparison, cross-engine metrics.
   - 04_agentic_clinical_workflow.py: Multi-agent clinical decision support with oncologist/radiologist/safety/data agents, unified interface, human-in-the-loop.
   - 05_outcome_prediction_pipeline.py: ML pipeline for treatment outcome prediction, survival analysis, biomarker integration, fairness auditing.
2. Physical AI Examples (examples-physical-ai/) — README.md, 6 scripts (1000-1300 LOC each):
   - 01_realtime_safety_monitoring.py: Force/torque monitoring, emergency stop, workspace bounds, ISO 13482.
   - 02_multi_sensor_fusion.py: Kalman/complementary filter fusion, optical+IMU+F/T+stereo, latency compensation.
   - 03_ros2_surgical_deployment.py: ROS2 lifecycle nodes, QoS, URDF, trajectory planning, real-time control.
   - 04_hand_eye_calibration.py: Tsai-Lenz calibration, fiducial detection, reprojection error, validation.
   - 05_shared_autonomy_teleoperation.py: Variable autonomy blending, intent prediction, haptic feedback, safety constraints.
   - 06_robotic_sample_handling.py: Grasp planning, force-controlled manipulation, contamination avoidance, chain-of-custody.
3. Agentic AI Examples (agentic-ai/examples-agentic-ai/) — README.md, 6 scripts (1000-1300 LOC each):
   - 01_mcp_oncology_server.py: MCP server with tool registration, resource management, PHI-safe prompts.
   - 02_react_treatment_planner.py: ReAct agent with thought-action-observation loops, NCCN retrieval, drug interactions.
   - 03_realtime_adaptive_monitoring_agent.py: Streaming monitor, adaptive thresholds, adverse event tracking, dashboards.
   - 04_autonomous_simulation_orchestrator.py: Simulation campaign management, parameter sweeps, job scheduling, convergence.
   - 05_safety_constrained_agent_executor.py: Multi-layer safety (PHI detection, action allowlists, HITL gates, audit trail).
   - 06_oncology_rag_compliance_agent.py: RAG for regulatory compliance, FDA guidance retrieval, protocol deviation detection.
4. Updates: per-file-ignores in ruff.toml, README.md structure tree, version bump to 0.5.0.
Quality gates: ruff check + ruff format pass. RESEARCH USE ONLY disclaimers. Structured logging. Conditional imports with HAS_X. All clinical parameters bounded.
```

---

## Prompt 4: v0.4.0 — Core Domain Modules and CLI Tools

Consolidates: v0.4.0 (digital twins: patient modeling, treatment sim, clinical integration) + v0.8.0 (5 CLI tools)

Dependencies: Prompt 3

```
Build the core domain modules and CLI tools for pai-oncology-trial-fl.
1. Primary Domain (digital-twins/) — README.md, 3 subdirectories:
   - patient-modeling/patient_model.py: 500-800 LOC with TumorType/GrowthModel/PatientStatus Enums, TumorProfile/PatientBiomarkers/OrganFunction/PatientDigitalTwinConfig dataclasses, PatientPhysiologyEngine, PatientDigitalTwinFactory.
   - treatment-simulation/treatment_simulator.py: 500-800 LOC with TreatmentModality/ResponseCategory/ToxicityGrade Enums, ChemotherapyProtocol/RadiationPlan/ImmunotherapyRegimen/TreatmentOutcome dataclasses, TreatmentSimulator.
   - clinical-integration/clinical_integrator.py: 500-800 LOC with ClinicalSystem/IntegrationStatus/DataFormat Enums, ClinicalIntegrator, FederatedClinicalBridge.
2. CLI Tools (tools/) — README.md, 5 tools:
   - dicom-inspector: DICOM inspection with argparse subcommands, JSON output.
   - dose-calculator: Radiation/chemo dose calculator with NCCN/RTOG reference constants.
   - trial-site-monitor: Multi-site monitoring with GREEN/YELLOW/RED status Enums.
   - sim-job-runner: Cross-framework simulation launcher with framework detection.
   - deployment-readiness: Checklist-based validation with PASS/FAIL/WARNING/REQUIRES_VERIFICATION.
3. Updates: per-file-ignores in ruff.toml, README.md structure tree.
Quality gates: ruff check + ruff format pass. RESEARCH USE ONLY disclaimers. Docstrings on all dataclasses/enums. CLI --help. Bounded clinical parameters.
```

---

## Prompt 3: v0.3.0 — Repository Foundation, Community Health, and CI/CD Infrastructure

Consolidates: v0.1.0 (initial structure) + v0.2.0 (cooperation model) + v0.3.0/v0.3.1 (standards, version corrections, citations) + v0.5.1 (community health, CI/CD, disclaimers)

Dependencies: None (first prompt)

```
Create a new GitHub repository called [REPO_NAME] for [DOMAIN_FOCUS] in oncology clinical trials. Build the complete foundation in a single submission.
1. Repository Root Files — README.md with badges, responsible-use notice, Quick Start, Repository Structure tree, Core Technologies table, Unification Framework section, Key Capabilities, Dependencies, Actively Maintained Repositories table, Multi-Organization Cooperation table, Citation, Contributing, License.
2. Unification Framework (unification/) — simulation_physics/ with Isaac-MuJoCo bridge and parameter mapping, agentic_generative_ai/ with unified agent interface, surgical_robotics/, cross_platform_tools/ with framework detector/model converter/policy exporter/validation suite, standards_protocols/, integration_workflows/.
3. Standards (q1-2026-standards/) — 3 objectives: model conversion pipeline, model registry with validator, cross-platform benchmark runner. Implementation guide with timeline and compliance checklist.
4. Framework Integration Guides (frameworks/) — nvidia-isaac/, mujoco/, gazebo/, pybullet/ with INTEGRATION.md each.
5. Learning Domain Directories — supervised-learning/, reinforcement-learning/, self-supervised-learning/, generative-ai/ with strengths.md, limitations.md, results.md.
6. Configuration (configs/) — training_config.yaml with environment, domain randomization, algorithm, network, safety constraints, training, evaluation, deployment, transfer settings.
7. Verification (scripts/) — verify_installation.py with importlib-based detection, colored output, graceful handling.
8. Community Health — CODE_OF_CONDUCT.md, CONTRIBUTING.md, SECURITY.md, SUPPORT.md, .github/PULL_REQUEST_TEMPLATE.md, .github/ISSUE_TEMPLATE/bug_report.md, .github/ISSUE_TEMPLATE/feature_request.md.
9. CI/CD (.github/workflows/ci.yml) — lint-and-format (ruff + yamllint), validate-scripts (py_compile), test (pytest). Matrix 3.10/3.11/3.12.
10. Linting (ruff.toml) — line-length 120, per-file-ignores for all directories.
11. Tests — tests/__init__.py placeholder.
Quality gates: ruff check + ruff format --check pass. yamllint -d relaxed pass. RESEARCH USE ONLY disclaimers. Structured logging. Conditional imports with HAS_X.
```

---

## Prompt 1: v0.1.0 — Integration and Implementation

```
Your goal is to integrate physical ai with the following federated learning platform instructions below into kevinkawchak/pai-oncology-trial-fl. Utilize kevinkawchak/physical-ai-oncology-trials for assistance regarding physical ai implementation and also for how to simultaneously unify the new field. Target the product as an aid for the customer, and not how the user can make money. The GitHub repository should result in a sellable product that is comprehensive and addresses key unmet needs for performing federated learning alongside robots in upcoming oncology clinical trials. Make sure the Readme and other files under main are complete, such as changelog.md (v0.1.0). Also include a prompts.md that includes the prompt(s) used in this conversation.

Be sure to fix and address errors that would cause failed checks for the pull request (such as Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Separately, at the end of this conversation provide new release notes (v0.1.0) in the format below (keep release notes separate from the pull request, and keep hashtag characters as is, not formatting the text bold).
```

### Federated Learning Platform Instructions

```
Concept: A Federated Learning platform for oncology that unifies data ingestion, privacy, and multi-institution collaboration. It lets multiple hospitals or research centers jointly train AI models (e.g. tumor progression, imaging analysis) without exchanging raw patient data. This product builds on the repository's federation/ and privacy/ modules. It includes secure aggregation, differential privacy, consent management (DUAs), and monitoring.

Claude Code Prompt:
From the kevinkawchak/pai-oncology-trial-fl repo implement a federated machine learning framework for physical ai oncology trials. The repo should include:
- A Python module `federated/` with submodules: `coordinator.py` (orchestrates training rounds), `client.py` (simulated hospital nodes), `secure_aggregation.py`, and `differential_privacy.py`.
- Data ingestion scripts to load synthetic oncology datasets (e.g. tumor images or patient records) and split them across sites.
- Privacy modules for PHI de-identification (reuse phi_detector/deidentification code).
- CI/CD setup (GitHub Actions) running unit tests and code style checks.
- Automated tests in `tests/` verifying model training works across two or more simulated sites.
- Jupyter notebooks in `docs/` demonstrating federated training of a simple model (e.g. tumor classification) with at least two clients.
- Example data or scripts to generate dummy data.
- A deployment script or docker-compose to launch federated learning simulation.
- Detailed `README.md` explaining usage, architecture, and installation.
Ensure reproducibility (Conda env or Dockerfile), include license (MIT), and a descriptive CITATION.cff.
Deliverables & Acceptance Criteria:
* Code Modules: A working physical ai federated learning codebase (Python) with modules for coordinator, clients, privacy, and secure aggregation. The code should build on concepts from federation/.
* CI/CD: GitHub Actions that automatically lint (e.g. flake8 or ruff), test federation logic, and verify example runs.
* Tests: Unit tests in tests/ confirming that model parameters update correctly across sites and privacy safeguards work. For example, a test that adding Gaussian noise in differential_privacy.py still yields a reasonable aggregated model.
* Documentation/Examples: Jupyter notebooks and examples (in docs/ or examples-federation/) that show how two or more synthetic "site" clients coordinate to train a model, with scripts for data splitting.
* Containers: Dockerfile or Conda environment supporting federated execution across containers or processes, with instructions for setup.
* Compliance Components: Example consent forms or DUA templates (leveraging privacy/dua-templates/) that accompany the federated process.
* Acceptance: The platform must run end-to-end: simulate at least two clients joining a study, train an AI model (e.g. CNN classifier), and produce aggregate model accuracy comparable to centralized training. Privacy checks (no raw data sharing) should be enforced. Documentation must guide a user through setup and interpretation of results.
Effort & Expertise: High. Requires expertise in machine learning, physical ai, federated algorithms, security, and healthcare data compliance. Development involves advanced topics like distributed systems and privacy engineering. Estimate ~3-6 person-months for an initial MVP.
Monetization & Customers: Target customers include research consortia, hospitals, and pharma companies collaborating on oncology AI models. Monetization could be via enterprise licensing or cloud-based SaaS (managed federated training). Premium features (advanced analytics, support) could be add-ons. The product leverages open MIT-licensed core but charges for value-added services (certifications, deployments).
Regulatory & Ethical: Must adhere to HIPAA/GDPR: all patient data must be de-identified and encrypted in transit. Differential privacy or secure aggregation should ensure no individual can be re-identified. Ethical concerns include data bias: consortium should audit model fairness across subgroups. Regulatory docs (e.g. IRB approvals) may be needed before federated trials. The product should include logging and audit trails (perhaps leveraging privacy/breach-response/).
```

---

## Prompt 2: v0.2.0 — Comprehensive Enhancement

```
Your goal is to make the existing kevinkawchak/pai-oncology-trial-fl more comprehensive. Utilize kevinkawchak/physical-ai-oncology-trials for assistance regarding physical ai implementation and also for how to simultaneously unify the new field. Make sure no duplicate and/or repetitive information is used from the other repo. Code should be longer, have greater context, and be more comprehensive. Add, modify, or remove code as needed. Target the product as an aid for the customer, and not how the user can make money. The GitHub repository should result in a sellable end to end product that is substantial and addresses key unmet needs for performing federated learning alongside robots in upcoming oncology clinical trials. Make sure the Readme and other files under main are complete, such as changelog.md (v0.2.0). Also update the prompts.md that includes the prompt(s) used in this conversation.

Be sure to fix and address errors that would cause failed checks for the pull request (such as Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Separately, at the end of this conversation provide new release notes (v0.2.0) in the format below (keep release notes separate from the pull request, and keep hashtag characters as is, not formatting the text bold).
```

### Federated Learning Platform Instructions (Same as v0.1.0)

```
Concept: A Federated Learning platform for oncology that unifies data ingestion, privacy, and multi-institution collaboration. It lets multiple hospitals or research centers jointly train AI models (e.g. tumor progression, imaging analysis) without exchanging raw patient data. This product builds on the repository's federation/ and privacy/ modules. It includes secure aggregation, differential privacy, consent management (DUAs), and monitoring.

Claude Code Prompt:
From the "pai-oncology-trial-fl" repo implement a federated machine learning framework for physical ai oncology trials. The repo should include:
- A Python module `federated/` with submodules: `coordinator.py` (orchestrates training rounds), `client.py` (simulated hospital nodes), `secure_aggregation.py`, and `differential_privacy.py`.
- Data ingestion scripts to load synthetic oncology datasets (e.g. tumor images or patient records) and split them across sites.
- Privacy modules for PHI de-identification (reuse phi_detector/deidentification code).
- CI/CD setup (GitHub Actions) running unit tests and code style checks.
- Automated tests in `tests/` verifying model training works across two or more simulated sites.
- Jupyter notebooks in `docs/` demonstrating federated training of a simple model (e.g. tumor classification) with at least two clients.
- Example data or scripts to generate dummy data.
- A deployment script or docker-compose to launch federated learning simulation.
- Detailed `README.md` explaining usage, architecture, and installation.
Ensure reproducibility (Conda env or Dockerfile), include license (MIT), and a descriptive CITATION.cff.
Deliverables & Acceptance Criteria:
* Code Modules: A working physical ai federated learning codebase (Python) with modules for coordinator, clients, privacy, and secure aggregation. The code should build on concepts from federation/.
* CI/CD: GitHub Actions that automatically lint (e.g. flake8 or ruff), test federation logic, and verify example runs.
* Tests: Unit tests in tests/ confirming that model parameters update correctly across sites and privacy safeguards work. For example, a test that adding Gaussian noise in differential_privacy.py still yields a reasonable aggregated model.
* Documentation/Examples: Jupyter notebooks and examples (in docs/ or examples-federation/) that show how two or more synthetic "site" clients coordinate to train a model, with scripts for data splitting.
* Containers: Dockerfile or Conda environment supporting federated execution across containers or processes, with instructions for setup.
* Compliance Components: Example consent forms or DUA templates (leveraging privacy/dua-templates/) that accompany the federated process.
* Acceptance: The platform must run end-to-end: simulate at least two clients joining a study, train an AI model (e.g. CNN classifier), and produce aggregate model accuracy comparable to centralized training. Privacy checks (no raw data sharing) should be enforced. Documentation must guide a user through setup and interpretation of results.
Effort & Expertise: High. Requires expertise in machine learning, physical ai, federated algorithms, security, and healthcare data compliance. Development involves advanced topics like distributed systems and privacy engineering. Estimate ~3–6 person-months for an initial MVP.
Monetization & Customers: Target customers include research consortia, hospitals, and pharma companies collaborating on oncology AI models. Monetization could be via enterprise licensing or cloud-based SaaS (managed federated training). Premium features (advanced analytics, support) could be add-ons. The product leverages open MIT-licensed core but charges for value-added services (certifications, deployments).
Regulatory & Ethical: Must adhere to HIPAA/GDPR: all patient data must be de-identified and encrypted in transit. Differential privacy or secure aggregation should ensure no individual can be re-identified. Ethical concerns include data bias: consortium should audit model fairness across subgroups. Regulatory docs (e.g. IRB approvals) may be needed before federated trials. The product should include logging and audit trails (perhaps leveraging privacy/breach-response/).
```
