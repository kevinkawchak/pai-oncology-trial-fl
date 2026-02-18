# PAI Oncology Trial FL

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.9.2-green.svg)](CHANGELOG.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Practical tools for Physical AI Federated Learning in oncology clinical trials, by Claude Code Opus 4.6.

> **Responsible-Use Notice:** This repository provides research-grade tooling for engineers building
> federated learning systems for oncology clinical trials. All modules are intended for research and
> development use only. Independent clinical validation, IRB approval, and regulatory clearance are
> required before any component is used in a clinical setting. Patient safety is the highest priority.

> **v0.9.2 Release:** 194 Python modules, ~80,600 LOC, 31 example scripts, 97 test files,
> 5 CLI tools, 80 Markdown docs. Full release details in [V1_RELEASE.md](V1_RELEASE.md).

## Quick Start

```bash
# Clone
git clone https://github.com/kevinkawchak/pai-oncology-trial-fl.git
cd pai-oncology-trial-fl

# Install
pip install -e ".[dev]"

# Verify
python scripts/verify_installation.py
```

## Repository Structure

```
pai-oncology-trial-fl/
├── federated/                        # Core federated learning framework
│   ├── coordinator.py                #   FedAvg / FedProx / SCAFFOLD coordinator
│   ├── client.py                     #   Simulated hospital nodes
│   ├── model.py                      #   Numpy-based MLP model
│   ├── secure_aggregation.py         #   Mask-based secure aggregation
│   ├── differential_privacy.py       #   Gaussian mechanism DP
│   ├── data_ingestion.py             #   Data generation & partitioning
│   ├── data_harmonization.py         #   DICOM/FHIR vocabulary mapping & normalisation
│   └── site_enrollment.py            #   Multi-site enrollment management
├── physical_ai/                      # Physical AI integration
│   ├── digital_twin.py               #   Patient digital twins (exponential/logistic/Gompertz)
│   ├── robotic_integration.py        #   Surgical robot interface
│   ├── sensor_fusion.py              #   Multi-modal sensor fusion
│   ├── simulation_bridge.py          #   Cross-platform simulation (URDF/MJCF/SDF/USD)
│   ├── framework_detection.py        #   Simulation framework detection & pipeline
│   └── surgical_tasks.py             #   Surgical task definitions & clinical thresholds
├── privacy/                          # Privacy framework (HIPAA Safe Harbor)
│   ├── phi_detector.py               #   PHI detection (18 HIPAA IDs) — legacy
│   ├── deidentification.py           #   De-identification pipeline — legacy
│   ├── consent_manager.py            #   Consent & DUA management
│   ├── audit_logger.py               #   Audit logging with integrity hashing
│   ├── access_control.py             #   Role-based access control — legacy
│   ├── breach_response.py            #   Breach detection & incident response — legacy
│   ├── dua_templates/                #   DUA templates (markdown)
│   ├── phi-pii-management/           #   Advanced PHI/PII detection (v0.6.0)
│   │   └── phi_detector.py           #     All 18 Safe Harbor IDs, DICOM scanning, Presidio
│   ├── de-identification/            #   Production de-identification pipeline (v0.6.0)
│   │   └── deidentification_pipeline.py  #  HMAC-SHA256, date shifting, generalization
│   ├── access-control/               #   RBAC with 21 CFR Part 11 audit trail (v0.6.0)
│   │   └── access_control_manager.py #     Time-limited access, fail-closed expiration
│   ├── breach-response/              #   Breach response protocol (v0.6.0)
│   │   └── breach_response_protocol.py #   Four-factor risk assessment, 60-day timeline
│   └── dua-templates-generator/      #   DUA template generator (v0.6.0)
│       └── dua_generator.py          #     5 DUA types, security tiers, retention policies
├── regulatory/                       # Regulatory compliance framework
│   ├── compliance_checker.py         #   HIPAA/GDPR/FDA checks — legacy
│   ├── fda_submission.py             #   FDA submission tracking — legacy
│   ├── templates/                    #   IRB & consent templates
│   ├── fda-compliance/               #   FDA submission tracker (v0.6.0)
│   │   └── fda_submission_tracker.py #     510(k)/De Novo/PMA/Breakthrough, PCCP, QMSR
│   ├── irb-management/               #   IRB protocol manager (v0.6.0)
│   │   └── irb_protocol_manager.py   #     Protocol lifecycle, amendments, consent versioning
│   ├── ich-gcp/                      #   ICH E6(R3) compliance checker (v0.6.0)
│   │   └── gcp_compliance_checker.py #     13 GCP principles, compliance scoring
│   ├── regulatory-intelligence/      #   Multi-jurisdiction tracker (v0.6.0)
│   │   └── regulatory_tracker.py     #     FDA/EMA/PMDA/TGA/Health Canada monitoring
│   └── human-oversight/              #   Human oversight QMS (v0.6.0)
│       └── HUMAN_OVERSIGHT_QMS.md    #     CRF risk tiers, AE boundaries, safety gates
├── unification/                      # Cross-framework unification layer
│   ├── simulation_physics/           #   Isaac ↔ MuJoCo bridge, parameter mapping
│   ├── agentic_generative_ai/        #   Unified agent interface (CrewAI/LangGraph/AutoGen)
│   ├── surgical_robotics/            #   Surgical robotics unification
│   ├── cross_platform_tools/         #   Framework detector, model converter, policy exporter
│   ├── standards_protocols/          #   Data formats, communication, safety standards
│   └── integration_workflows/        #   Workflow templates
├── q1-2026-standards/                # Q1 2026 standards objectives
│   ├── objective-1-model-conversion/ #   Cross-framework model conversion pipeline
│   ├── objective-2-model-registry/   #   Federated model registry & validation
│   ├── objective-3-benchmarking/     #   Cross-platform benchmark runner
│   └── implementation-guide/         #   Timeline & compliance checklist
├── frameworks/                       # Framework integration guides
│   ├── nvidia-isaac/                 #   NVIDIA Isaac Sim/Lab integration
│   ├── mujoco/                       #   MuJoCo integration
│   ├── gazebo/                       #   Gazebo + ROS 2 integration
│   └── pybullet/                     #   PyBullet integration
├── supervised-learning/              # Supervised learning for oncology
├── reinforcement-learning/           # RL for surgical robotics & treatment
├── self-supervised-learning/         # SSL for medical imaging
├── generative-ai/                    # Generative AI & agentic workflows
├── digital-twins/                    # Primary domain: patient digital twins
│   ├── patient-modeling/             #   Patient physiology engine & twin factory
│   ├── treatment-simulation/         #   Chemo/radiation/immunotherapy simulation
│   └── clinical-integration/         #   EHR/PACS/LIS integration & federated bridge
├── tools/                            # CLI tools for oncology workflows
│   ├── dicom-inspector/              #   DICOM image inspection & validation
│   ├── dose-calculator/              #   Radiation & chemotherapy dose calculator
│   ├── trial-site-monitor/           #   Multi-site trial monitoring (GREEN/YELLOW/RED)
│   ├── sim-job-runner/               #   Cross-framework simulation job launcher
│   └── deployment-readiness/         #   Deployment readiness checklist validation
├── clinical-analytics/               # Federated clinical trial analytics (v0.9.0)
│   ├── analytics_orchestrator.py     #   Pipeline orchestration & workflow management
│   ├── pkpd_engine.py                #   Population PK/PD compartmental modeling
│   ├── risk_stratification.py        #   Clinical risk scoring & adaptive enrichment
│   ├── trial_data_manager.py         #   Data lifecycle, quality checks, versioning
│   ├── clinical_interoperability.py  #   ICD-10/SNOMED/LOINC/CDISC mapping
│   ├── survival_analysis.py          #   Kaplan-Meier, Cox PH, C-index, RMST
│   ├── consortium_reporting.py       #   DSMB reports, SHA-256 integrity hashing
│   └── examples-clinical-analytics/  #   6 progressive examples
├── regulatory-submissions/           # Regulatory submission platform (v0.9.1)
│   ├── submission_orchestrator.py    #   Submission lifecycle & workflow management
│   ├── ectd_compiler.py              #   eCTD module compilation (FDA TCG)
│   ├── compliance_validator.py       #   Multi-regulation validation (9 frameworks)
│   ├── document_generator.py         #   CSR, SAP, PCCP, ISO 14971, IEC 62304
│   ├── regulatory_intelligence.py    #   7-jurisdiction guidance tracking
│   ├── submission_analytics.py       #   KPIs, trends, FDA MDUFA benchmarking
│   └── examples-regulatory-submissions/  # 6 progressive examples
├── configs/                          # Training & deployment configuration
│   └── training_config.yaml          #   PPO/SAC + federated + safety constraints
├── scripts/                          # Utility scripts
│   ├── verify_installation.py        #   Dependency verification
│   └── deploy.sh                     #   Docker deployment
├── tests/                            # Comprehensive test suite
├── examples/                         # Core example scripts (5 numbered)
│   ├── 01_federated_training_workflow.py
│   ├── 02_digital_twin_planning.py
│   ├── 03_cross_framework_validation.py
│   ├── 04_agentic_clinical_workflow.py
│   └── 05_outcome_prediction_pipeline.py
├── examples-physical-ai/             # Physical AI examples (6 numbered)
│   ├── 01_realtime_safety_monitoring.py
│   ├── 02_multi_sensor_fusion.py
│   ├── 03_ros2_surgical_deployment.py
│   ├── 04_hand_eye_calibration.py
│   ├── 05_shared_autonomy_teleoperation.py
│   └── 06_robotic_sample_handling.py
├── agentic-ai/examples-agentic-ai/   # Agentic AI examples (6 numbered)
│   ├── 01_mcp_oncology_server.py
│   ├── 02_react_treatment_planner.py
│   ├── 03_realtime_adaptive_monitoring_agent.py
│   ├── 04_autonomous_simulation_orchestrator.py
│   ├── 05_safety_constrained_agent_executor.py
│   └── 06_oncology_rag_compliance_agent.py
├── docs/                             # Documentation & notebooks
├── .github/                          # CI/CD, issue & PR templates
│   ├── workflows/ci.yml              #   Lint + format + YAML + test
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── SUPPORT.md
├── CHANGELOG.md
├── CITATION.cff
├── V1_RELEASE.md                     # v0.9.2 release documentation
├── DEVELOPMENT_PROPOSALS.md          # 3 future extension proposals
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── ruff.toml
├── posts.md
└── prompts.md
```

## Core Technologies

| Framework | Version | Last Update | Use Case | Unification Status |
|-----------|---------|-------------|----------|--------------------|
| NumPy | ≥1.24.0 | 2023-06-17 | Core computation, federated MLP | Integrated |
| SciPy | ≥1.11.0 | 2023-06-25 | Statistical analysis, optimization | Integrated |
| PyTorch | ≥2.5.0 | 2024-10-23 | Deep learning, model training | Bridge available |
| NVIDIA Isaac Sim | 4.2.0 | 2024-11-15 | High-fidelity surgical simulation | Bridge available |
| MuJoCo | ≥3.1.0 | 2024-01-10 | Physics simulation, contact modeling | Bridge available |
| Gazebo | Harmonic | 2024-09-20 | ROS 2 integration, sensor simulation | Guide available |
| PyBullet | ≥3.2.6 | 2023-06-15 | Rapid prototyping, lightweight sim | Guide available |
| MONAI | ≥1.3.0 | 2024-01-17 | Medical image analysis | Optional |
| LangChain | ≥0.3.0 | 2024-09-13 | Agentic AI workflows | Interface available |
| LangGraph | ≥0.2.0 | 2024-08-28 | Stateful agent graphs | Interface available |
| CrewAI | ≥0.80.0 | 2024-11-20 | Multi-agent orchestration | Interface available |
| ONNX | ≥1.15.0 | 2024-01-23 | Model export, cross-platform deploy | Converter available |

## Unification Framework

The `unification/` layer provides cross-framework interoperability for oncology clinical trial systems:

1. **Simulation Physics** — Bidirectional state conversion between Isaac Sim and MuJoCo, physics parameter mapping across frameworks, enabling seamless sim-to-sim transfer for surgical robotics validation.

2. **Agentic Generative AI** — Unified agent interface supporting CrewAI, LangGraph, AutoGen, and custom backends with tool format conversion (MCP, OpenAI, Anthropic) for clinical workflow automation.

3. **Surgical Robotics** — Standards-based integration layer for robotic surgical systems across simulation and physical platforms.

4. **Cross-Platform Tools** — Framework detection, model format conversion, policy export, and validation suite for ensuring reproducibility across heterogeneous environments.

5. **Standards & Protocols** — Data format specifications, communication protocols, and safety standards aligned with IEC 80601-2-77 and ISO 14971.

## Key Capabilities

### 1. Federated Learning

Multi-site model training with FedAvg, FedProx, and SCAFFOLD aggregation strategies. Includes convergence detection, differential privacy (Gaussian mechanism with budget tracking), secure aggregation (mask-based protocol), and site enrollment lifecycle management.

### 2. Physical AI Integration

Patient digital twins with exponential, logistic, and Gompertz tumor growth models. Chemotherapy, radiation, immunotherapy, and combination therapy simulation with Monte-Carlo uncertainty quantification. Surgical robot interface with telemetry and clinical threshold evaluation.

### 3. Privacy Framework

Comprehensive HIPAA Safe Harbor compliance covering all 18 identifiers per 45 CFR 164.514(b)(2). Advanced PHI/PII detection with regex, DICOM tag scanning, and optional Presidio NER integration. HMAC-SHA256 pseudonymization with cryptographically random salt (os.urandom). Date shifting with consistent patient-level offsets. Geographic generalization (ZIP to 3-digit). Role-based access control with 21 CFR Part 11 audit trails and fail-closed expiration. Four-factor breach risk assessment per 45 CFR 164.402 with 60-day notification tracking. DUA template generation for 5 agreement types with tiered security requirements.

### 4. Regulatory Infrastructure

Multi-pathway FDA submission tracking (510(k), De Novo, PMA, Breakthrough, Pre-Submission) referencing FDA AI/ML Device Guidance (Jan 2025), PCCP Guidance (Aug 2025), and QMSR (Feb 2026). IRB protocol lifecycle management with amendment tracking, consent versioning, and AI/ML disclosure requirements per ICH E6(R3) published September 2025. GCP compliance scoring excluding NOT_ASSESSED findings from denominator. Multi-jurisdiction regulatory intelligence across FDA, EMA, PMDA, TGA, and Health Canada with overdue-before-imminent status ordering. Human oversight QMS with CRF auto-fill risk tiers and AE automation boundaries.

### 5. Cross-Framework Unification

Framework-agnostic simulation bridge (URDF/MJCF/SDF/USD), cross-platform model conversion, unified agent interface for multiple LLM backends, and physics parameter mapping across simulation engines.

### 6. Standards & Benchmarking

Q1 2026 standards objectives for model format conversion, federated model registry with validation, and cross-platform benchmark runner for reproducible performance evaluation.

## Dependencies

### Core (required)
- `numpy>=1.24.0` — Array computation
- `scipy>=1.11.0` — Scientific computing
- `scikit-learn>=1.3.0` — ML algorithms
- `pandas>=2.0.0` — Data manipulation
- `cryptography>=41.0.0` — Encryption
- `pyyaml>=6.0` — Configuration parsing

### Framework-Specific (optional)
- `torch>=2.5.0` — Deep learning
- `monai>=1.3.0` — Medical imaging
- `pydicom>=2.4.0` — DICOM parsing
- `mujoco>=3.1.0` — Physics simulation

### Agentic AI (optional)
- `langchain>=0.3.0` — LLM chains
- `langgraph>=0.2.0` — Agent graphs
- `crewai>=0.80.0` — Multi-agent
- `anthropic>=0.39.0` — Claude API
- `openai>=1.50.0` — OpenAI API

### Visualization (optional)
- `matplotlib>=3.8.0` — Plotting
- `plotly>=5.18.0` — Interactive charts

### Deployment (optional)
- `onnx>=1.15.0` — Model export
- `onnxruntime>=1.17.0` — Inference

### Testing / Development
- `pytest>=7.4.0` — Testing
- `ruff>=0.4.0` — Linting

## Actively Maintained Repositories

| Repository | Last Commit | URL |
|------------|-------------|-----|
| PyTorch | 2024-10-23 | https://github.com/pytorch/pytorch |
| MONAI | 2024-01-17 | https://github.com/Project-MONAI/MONAI |
| MuJoCo | 2024-01-10 | https://github.com/google-deepmind/mujoco |
| LangChain | 2024-09-13 | https://github.com/langchain-ai/langchain |
| LangGraph | 2024-08-28 | https://github.com/langchain-ai/langgraph |
| CrewAI | 2024-11-20 | https://github.com/crewAIInc/crewAI |
| Anthropic SDK | 2024-11-06 | https://github.com/anthropics/anthropic-sdk-python |
| Ruff | 2024-04-01 | https://github.com/astral-sh/ruff |

## Multi-Organization Cooperation

| Organization Type | Integration Points |
|-------------------|--------------------|
| Academic Medical Centers | Federated training nodes, digital twin data, IRB coordination |
| Community Hospitals | Site enrollment, data harmonization, local model training |
| Pharmaceutical Companies | Treatment simulation, regulatory submission, compliance validation |
| Robotics Manufacturers | Simulation framework integration, surgical task definitions |
| Regulatory Bodies (FDA) | Submission tracking, 510(k)/De Novo/PMA workflow, compliance checks |
| Standards Organizations | IEC 80601-2-77, ISO 14971, ICH E6(R3) alignment |

## Usage Guide

### 1. Generate and Partition Data

```python
from federated.data_ingestion import generate_synthetic_oncology_data, DataPartitioner

X, y = generate_synthetic_oncology_data(n_samples=1000, n_features=30, n_classes=2)
partitioner = DataPartitioner(num_sites=3, strategy="iid")
sites = partitioner.partition(X, y)
```

### 2. Run Federated Training

```python
from federated.coordinator import FederationCoordinator
from federated.client import FederatedClient
from federated.model import ModelConfig

config = ModelConfig(input_dim=30, hidden_dims=[64, 32], output_dim=2)
coordinator = FederationCoordinator(
    model_config=config, num_rounds=10, strategy="fedprox", mu=0.01,
)
global_params = coordinator.initialize()

clients = [FederatedClient(s.site_id, config) for s in sites]
for c, s in zip(clients, sites):
    c.set_data(s.x_train, s.y_train)

for round_num in range(10):
    updates = [c.train_local(global_params, epochs=5, lr=0.01, mu=0.01) or c.get_parameters() for c in clients]
    counts = [c.get_sample_count() for c in clients]
    result = coordinator.run_round(updates, client_sample_counts=counts)
    global_params = coordinator.get_global_parameters()
    if result.converged:
        break
```

### 3. Patient Digital Twin Simulation

```python
from physical_ai.digital_twin import PatientDigitalTwin, TumorModel

twin = PatientDigitalTwin(
    "patient_001",
    tumor=TumorModel(volume_cm3=3.5, chemo_sensitivity=0.6, growth_model="gompertz"),
    biomarkers={"pdl1": 0.7, "ki67": 0.3},
)
result = twin.simulate_treatment("chemotherapy", dose_mg=75, cycles=4)
```

### 4. Verify Installation

```bash
python scripts/verify_installation.py
```

## Citation

```bibtex
@software{kawchak2026pai,
  title = {PAI Oncology Trial FL},
  author = {Kawchak, Kevin},
  year = {2026},
  url = {https://github.com/kevinkawchak/pai-oncology-trial-fl},
  version = {0.9.2}
}
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for requirements:

- **Recency** — Dependencies and references within 3 months of latest stable release.
- **Oncology Relevance** — All contributions must relate to oncology clinical trial workflows.
- **Reproducibility** — Include seed configuration, hardware specs, and version pinning.
- **Cross-Platform Compatibility** — Test across Python 3.10, 3.11, and 3.12.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Related

- [physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials) — Physical AI toolkit for oncology clinical trials.
