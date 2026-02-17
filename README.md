# PAI Oncology Trial FL

Federated machine learning framework for physical AI oncology clinical trials.

Enables multiple hospitals and research centers to collaboratively train AI models for tumor response prediction, surgical planning, and treatment optimization — **without exchanging raw patient data**.

## Key Capabilities

- **Federated Learning** — Train models across institutions using FedAvg, FedProx, or SCAFFOLD with configurable rounds, client weighting, convergence detection, and evaluation.
- **Physical AI Integration** — Patient digital twins with multiple tumor growth models, surgical robot telemetry with clinical threshold evaluation, multi-modal sensor fusion, cross-platform simulation interoperability, and framework detection.
- **Privacy Preservation** — Differential privacy (Gaussian mechanism with budget tracking), secure aggregation (mask-based protocol), automated PHI detection/de-identification for the 18 HIPAA identifiers, role-based access control, and breach response protocol.
- **Regulatory Compliance** — Automated HIPAA/GDPR/FDA compliance checks, consent management, audit logging with integrity verification, DUA/IRB templates, and FDA submission tracking for AI/ML devices.
- **Data Harmonization** — DICOM/FHIR vocabulary mapping, unit conversion, and feature normalisation for heterogeneous multi-site data.
- **Site Enrollment** — Multi-site enrollment lifecycle management with readiness validation, quality-weighted site selection, and audit trail.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                      Coordinator                           │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐ │
│  │  FedAvg / │  │ Secure   │  │Differential│  │Conver- │ │
│  │  FedProx /│  │ Aggreg.  │  │  Privacy   │  │gence   │ │
│  │  SCAFFOLD │  │          │  │            │  │Detect  │ │
│  └─────┬─────┘  └────┬─────┘  └─────┬──────┘  └────────┘ │
│        └──────┬───────┘              │                     │
│               │ model params only    │                     │
└───────────────┼──────────────────────┼─────────────────────┘
                │                      │
   ┌────────────┼──────────────────────┼──────────┐
   │            │                      │          │
┌──┴──────┐ ┌──┴──────┐ ┌──────────┐ ┌┴────────┐ │
│ Site A  │ │ Site B  │ │ Site C   │ │ Site D  │ │
│ Hospital│ │ Hospital│ │ Research │ │ Pharma  │ │
│         │ │         │ │ Center   │ │ Lab     │ │
│ ┌─────┐ │ │ ┌─────┐ │ │ ┌──────┐ │ │ ┌─────┐ │ │
│ │Local│ │ │ │Local│ │ │ │Local │ │ │ │Local│ │ │
│ │Model│ │ │ │Model│ │ │ │Model │ │ │ │Model│ │ │
│ └─────┘ │ │ └─────┘ │ │ └──────┘ │ │ └─────┘ │ │
│ ┌─────┐ │ │ ┌─────┐ │ │ ┌──────┐ │ │ ┌─────┐ │ │
│ │Data │ │ │ │Data │ │ │ │Data  │ │ │ │Data │ │ │
│ │(PHI │ │ │ │(PHI │ │ │ │(PHI  │ │ │ │(PHI │ │ │
│ │free)│ │ │ │free)│ │ │ │free) │ │ │ │free)│ │ │
│ └─────┘ │ │ └─────┘ │ │ └──────┘ │ │ └─────┘ │ │
└─────────┘ └─────────┘ └──────────┘ └─────────┘ │
   └──────────────────────────────────────────────┘
            Raw data never leaves sites
```

## Project Structure

```
pai-oncology-trial-fl/
├── federated/                     # Core federated learning framework
│   ├── coordinator.py             #   FedAvg / FedProx / SCAFFOLD coordinator
│   ├── client.py                  #   Simulated hospital nodes
│   ├── model.py                   #   Numpy-based MLP model
│   ├── secure_aggregation.py      #   Mask-based secure aggregation
│   ├── differential_privacy.py    #   Gaussian mechanism DP
│   ├── data_ingestion.py          #   Data generation & partitioning
│   ├── data_harmonization.py      #   DICOM/FHIR vocabulary mapping & normalisation
│   └── site_enrollment.py         #   Multi-site enrollment management
├── physical_ai/                   # Physical AI integration
│   ├── digital_twin.py            #   Patient digital twins (exponential/logistic/Gompertz)
│   ├── robotic_integration.py     #   Surgical robot interface
│   ├── sensor_fusion.py           #   Multi-modal sensor fusion
│   ├── simulation_bridge.py       #   Cross-platform simulation (URDF/MJCF/SDF/USD)
│   ├── framework_detection.py     #   Simulation framework detection & pipeline
│   └── surgical_tasks.py          #   Surgical task definitions & clinical thresholds
├── privacy/                       # Privacy infrastructure
│   ├── phi_detector.py            #   PHI detection (18 HIPAA IDs)
│   ├── deidentification.py        #   De-identification pipeline
│   ├── consent_manager.py         #   Consent & DUA management
│   ├── audit_logger.py            #   Audit logging with integrity hashing
│   ├── access_control.py          #   Role-based access control (RBAC)
│   ├── breach_response.py         #   Breach detection & incident response
│   └── dua_templates/             #   DUA templates
├── regulatory/                    # Regulatory compliance
│   ├── compliance_checker.py      #   HIPAA/GDPR/FDA checks
│   ├── fda_submission.py          #   FDA submission tracking (510k/De Novo/PMA)
│   └── templates/                 #   IRB & consent templates
├── tests/                         # Comprehensive test suite
├── examples/                      # Example scripts
│   ├── run_federation.py          #   Full simulation (FedAvg/FedProx/SCAFFOLD)
│   └── generate_synthetic_data.py #   Data generation
├── docs/                          # Documentation
│   └── federated_training_demo.ipynb
├── scripts/                       # Deployment scripts
│   └── deploy.sh                  #   Docker deployment
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── CHANGELOG.md
├── CITATION.cff
└── prompts.md
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kevinkawchak/pai-oncology-trial-fl.git
cd pai-oncology-trial-fl

# Option 1: pip
pip install -e ".[dev]"

# Option 2: Conda
conda env create -f environment.yml
conda activate pai-oncology-fl
pip install -e .

# Option 3: Docker
docker compose build
```

### Run a Federated Simulation

```bash
# Basic 3-site FedAvg simulation
python examples/run_federation.py

# FedProx with non-IID data handling
python examples/run_federation.py --strategy fedprox --mu 0.01 --num-sites 4

# With differential privacy and secure aggregation
python examples/run_federation.py --num-sites 4 --rounds 20 --dp-epsilon 2.0 --secure-agg

# Generate synthetic data to CSV
python examples/generate_synthetic_data.py --output-dir data/ --num-sites 3

# Docker deployment
./scripts/deploy.sh --sites 3
```

### Run Tests

```bash
pytest tests/ -v
```

## Usage Guide

### 1. Generate and Partition Data

```python
from federated.data_ingestion import generate_synthetic_oncology_data, DataPartitioner

# Generate 1000 synthetic patient records with 30 clinical features
X, y = generate_synthetic_oncology_data(n_samples=1000, n_features=30, n_classes=2)

# Split across 3 hospital sites (IID)
partitioner = DataPartitioner(num_sites=3, strategy="iid")
sites = partitioner.partition(X, y)
```

### 2. Harmonize Multi-Site Data

```python
from federated.data_harmonization import DataHarmonizer

harmonizer = DataHarmonizer(normalisation="zscore")

# Map FHIR/DICOM codes to canonical field names
record = {"observation-tumor-volume": 3.5, "observation-ki67": 0.25}
mapped = harmonizer.map_record(record)

# Convert units
value_cm = harmonizer.convert_units(15.0, "mm", "cm")  # 1.5

# Normalise features
normalised, stats = harmonizer.normalise_features(X, method="zscore")
```

### 3. Enroll Sites

```python
from federated.site_enrollment import SiteEnrollmentManager

mgr = SiteEnrollmentManager("FL_STUDY_01", min_patients_per_site=20)
mgr.enroll_site("hospital_A", "Memorial Cancer Center", patient_count=200)
mgr.mark_data_ready("hospital_A")
mgr.mark_compliance_passed("hospital_A")
mgr.activate_site("hospital_A")

# Select sites for a training round
active = mgr.select_sites_for_round(max_sites=5)
```

### 4. Run Federated Training

```python
from federated.coordinator import FederationCoordinator
from federated.client import FederatedClient
from federated.model import ModelConfig

config = ModelConfig(input_dim=30, hidden_dims=[64, 32], output_dim=2)

# Choose aggregation strategy: "fedavg", "fedprox", or "scaffold"
coordinator = FederationCoordinator(
    model_config=config,
    num_rounds=10,
    strategy="fedprox",
    mu=0.01,  # proximal term for FedProx
)
global_params = coordinator.initialize()

# Create clients for each site
clients = []
for site in sites:
    client = FederatedClient(site.site_id, config)
    client.set_data(site.x_train, site.y_train)
    clients.append(client)

# Federated training loop
for round_num in range(10):
    updates, counts = [], []
    for client in clients:
        client.train_local(global_params, epochs=5, lr=0.01, mu=0.01)
        updates.append(client.get_parameters())
        counts.append(client.get_sample_count())

    result = coordinator.run_round(
        updates, client_sample_counts=counts,
        eval_data=(sites[0].x_test, sites[0].y_test)
    )
    global_params = coordinator.get_global_parameters()
    print(f"Round {round_num+1}: accuracy={result.global_metrics['accuracy']:.4f}")

    if result.converged:
        print("Training converged.")
        break
```

### 5. Enable Privacy Protections

```python
# With differential privacy and secure aggregation
coordinator = FederationCoordinator(
    model_config=config,
    use_differential_privacy=True,
    dp_epsilon=2.0,
    dp_delta=1e-5,
    use_secure_aggregation=True,
)
```

### 6. Integrate Physical AI Digital Twins

```python
from physical_ai.digital_twin import PatientDigitalTwin, TumorModel

twin = PatientDigitalTwin(
    "patient_001",
    tumor=TumorModel(
        volume_cm3=3.5,
        chemo_sensitivity=0.6,
        growth_model="gompertz",  # exponential, logistic, or gompertz
    ),
    biomarkers={"pdl1": 0.7, "ki67": 0.3},
)

# Simulate treatment response
result = twin.simulate_treatment("chemotherapy", dose_mg=75, cycles=4)
print(f"Predicted response: {result['response_category']}")

# Simulate combination therapy
combo = twin.simulate_treatment(
    "combination",
    chemo_dose_mg=75, chemo_cycles=4,
    radiation_dose_gy=60, radiation_fractions=30,
)
print(f"Combined reduction: {combo['volume_reduction_pct']:.1f}%")

# Uncertainty quantification
uq = twin.simulate_with_uncertainty("chemotherapy", n_samples=100, dose_mg=75, cycles=4)
print(f"Mean volume: {uq['mean_volume_cm3']:.2f} +/- {uq['std_volume_cm3']:.2f}")

# Generate feature vector for federated model
features = twin.generate_feature_vector()
```

### 7. Evaluate Surgical Tasks

```python
from physical_ai.surgical_tasks import STANDARD_TASKS, ProcedureType, SurgicalTaskEvaluator

task_def = STANDARD_TASKS[ProcedureType.NEEDLE_BIOPSY]
evaluator = SurgicalTaskEvaluator(task_def)

result = evaluator.evaluate(
    position_errors_mm=[0.5, 0.8, 1.2],
    forces_n=[1.0, 1.5, 2.0],
    collisions=0,
    total_attempts=100,
    procedure_time_s=45.0,
)
print(f"Clinical ready: {result['clinical_ready']}")
```

### 8. Run Compliance Checks

```python
from regulatory.compliance_checker import ComplianceChecker

checker = ComplianceChecker()
report = checker.check_federation_config({
    "use_differential_privacy": True,
    "use_secure_aggregation": True,
    "use_deidentification": True,
    "audit_logging_enabled": True,
    "consent_management_enabled": True,
    "encryption_in_transit": True,
})
print(f"Compliance: {report.overall_status.value}")
```

### 9. Track FDA Submissions

```python
from regulatory.fda_submission import FDASubmissionTracker

tracker = FDASubmissionTracker()
pkg = tracker.create_submission(
    "SUB_001", pathway="de_novo",
    device_name="PAI Tumor Response Predictor",
    indication="AI-assisted tumor treatment response prediction",
)
print(f"Documents required: {len(pkg.documents)}")
readiness = tracker.get_readiness_report("SUB_001")
print(f"Ready: {readiness['ready_for_submission']}")
```

## Modules

### Federated Learning (`federated/`)

| Module | Description |
|--------|-------------|
| `coordinator.py` | Orchestrates training rounds with FedAvg, FedProx, or SCAFFOLD strategies. Includes convergence detection. |
| `client.py` | Simulated hospital node with local training, FedProx proximal term, and SCAFFOLD control variates. |
| `model.py` | Numpy-based MLP with backpropagation, supporting parameter extraction/injection. |
| `secure_aggregation.py` | Pairwise mask protocol — coordinator sees only the aggregate, not individual updates. |
| `differential_privacy.py` | Gaussian mechanism with gradient clipping, noise calibration, and epsilon budget tracking. |
| `data_ingestion.py` | Synthetic oncology data generation with IID and non-IID site partitioning. |
| `data_harmonization.py` | DICOM/FHIR vocabulary mapping, unit conversion, and feature normalisation for heterogeneous data. |
| `site_enrollment.py` | Multi-site enrollment lifecycle with readiness checks, quality scoring, and audit trail. |

### Physical AI (`physical_ai/`)

| Module | Description |
|--------|-------------|
| `digital_twin.py` | Patient digital twins with exponential, logistic, and Gompertz growth models. Supports chemotherapy, radiation, immunotherapy, and combination therapy simulation with Monte-Carlo uncertainty quantification. |
| `robotic_integration.py` | Surgical robot interface for procedure planning, execution, and telemetry collection. |
| `sensor_fusion.py` | Multi-modal clinical sensor fusion with real-time Z-score anomaly detection. |
| `simulation_bridge.py` | Cross-platform robot model conversion (URDF/MJCF/SDF/USD) with physics validation. |
| `framework_detection.py` | Detects installed simulation frameworks (Isaac Lab, MuJoCo, Gazebo, PyBullet) and recommends training/validation/deployment pipeline. |
| `surgical_tasks.py` | Standardised surgical task definitions for oncology procedures with clinical accuracy thresholds and RL reward structures. |

### Privacy (`privacy/`)

| Module | Description |
|--------|-------------|
| `phi_detector.py` | Detects all 18 HIPAA identifiers using regex and field-name analysis. |
| `deidentification.py` | Removes PHI via redact, hash, generalize, or replace strategies. |
| `consent_manager.py` | Tracks patient consent with support for registration, verification, and revocation. |
| `audit_logger.py` | Append-only event log with SHA-256 integrity hashing for tamper detection. |
| `access_control.py` | Role-based access control with six roles and thirteen permissions. All access decisions logged. |
| `breach_response.py` | Breach detection with auto-escalation, incident lifecycle management, and regulatory reporting. |

### Regulatory (`regulatory/`)

| Module | Description |
|--------|-------------|
| `compliance_checker.py` | Validates federation configs against HIPAA, GDPR, and FDA 21 CFR Part 11. |
| `fda_submission.py` | Tracks FDA submission packages for AI/ML devices (Pre-Submission, De Novo, 510(k), PMA, Breakthrough). |
| `templates/` | IRB protocol and informed consent templates for multi-site trials. |

## Regulatory Compliance

The platform includes infrastructure for:

- **HIPAA** — PHI detection and de-identification, audit logging, access control, breach response protocol.
- **GDPR** — Consent management, right to erasure support, data minimization via DP.
- **FDA 21 CFR Part 11** — Audit trail requirements for electronic records.
- **FDA AI/ML Guidance** — Submission tracking for 510(k), De Novo, PMA, and Breakthrough Device pathways.
- **ICH-GCP** — IRB protocol templates and informed consent forms.
- **DUA Templates** — Standard and multi-site Data Use Agreement templates.

## Requirements

- Python >= 3.10
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- cryptography >= 41.0.0
- pyyaml >= 6.0

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{kawchak2026pai,
  title = {PAI Oncology Trial FL},
  author = {Kawchak, Kevin},
  year = {2026},
  url = {https://github.com/kevinkawchak/pai-oncology-trial-fl},
  version = {0.2.0}
}
```

## Related

- [physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials) — Physical AI toolkit for oncology clinical trials (simulation, digital twins, agentic AI, regulatory compliance).
