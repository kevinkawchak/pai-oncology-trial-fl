# PAI Oncology Trial FL

Federated machine learning framework for physical AI oncology clinical trials.

Enables multiple hospitals and research centers to collaboratively train AI models for tumor response prediction, surgical planning, and treatment optimization — **without exchanging raw patient data**.

## Key Capabilities

- **Federated Learning** — Train models across institutions using Federated Averaging (FedAvg) with configurable rounds, client weighting, and evaluation.
- **Physical AI Integration** — Patient digital twins, surgical robot telemetry, multi-modal sensor fusion, and cross-platform simulation interoperability.
- **Privacy Preservation** — Differential privacy (Gaussian mechanism with budget tracking), secure aggregation (mask-based protocol), and automated PHI detection/de-identification for the 18 HIPAA identifiers.
- **Regulatory Compliance** — Automated HIPAA/GDPR/FDA compliance checks, consent management, audit logging with integrity verification, and DUA/IRB templates.

## Architecture

```
┌───────────────────────────────────────────────────┐
│                   Coordinator                      │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐ │
│  │   FedAvg    │  │ Secure   │  │ Differential │ │
│  │ Aggregation │  │ Aggreg.  │  │   Privacy    │ │
│  └──────┬──────┘  └────┬─────┘  └──────┬───────┘ │
│         └───────┬───────┘               │         │
│                 │ model parameters only  │         │
└─────────────────┼───────────────────────┼─────────┘
                  │                       │
     ┌────────────┼───────────────────────┼──────────┐
     │            │                       │          │
┌────┴────┐  ┌───┴─────┐  ┌─────────┐  ┌┴────────┐ │
│ Site A  │  │ Site B  │  │ Site C  │  │ Site D  │ │
│ Hospital│  │ Hospital│  │ Research│  │ Pharma  │ │
│         │  │         │  │ Center  │  │ Lab     │ │
│ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │ │
│ │Local│ │  │ │Local│ │  │ │Local│ │  │ │Local│ │ │
│ │Model│ │  │ │Model│ │  │ │Model│ │  │ │Model│ │ │
│ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │ │
│ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │ │
│ │Data │ │  │ │Data │ │  │ │Data │ │  │ │Data │ │ │
│ │(PHI │ │  │ │(PHI │ │  │ │(PHI │ │  │ │(PHI │ │ │
│ │free)│ │  │ │free)│ │  │ │free)│ │  │ │free)│ │ │
│ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │ │
└─────────┘  └─────────┘  └─────────┘  └─────────┘ │
     └───────────────────────────────────────────────┘
              Raw data never leaves sites
```

## Project Structure

```
pai-oncology-trial-fl/
├── federated/                     # Core federated learning framework
│   ├── coordinator.py             #   FedAvg coordinator
│   ├── client.py                  #   Simulated hospital nodes
│   ├── model.py                   #   Numpy-based MLP model
│   ├── secure_aggregation.py      #   Mask-based secure aggregation
│   ├── differential_privacy.py    #   Gaussian mechanism DP
│   └── data_ingestion.py          #   Data generation & partitioning
├── physical_ai/                   # Physical AI integration
│   ├── digital_twin.py            #   Patient digital twins
│   ├── robotic_integration.py     #   Surgical robot interface
│   ├── sensor_fusion.py           #   Multi-modal sensor fusion
│   └── simulation_bridge.py       #   Cross-platform simulation
├── privacy/                       # Privacy infrastructure
│   ├── phi_detector.py            #   PHI detection (18 HIPAA IDs)
│   ├── deidentification.py        #   De-identification pipeline
│   ├── consent_manager.py         #   Consent & DUA management
│   ├── audit_logger.py            #   Audit logging
│   └── dua_templates/             #   DUA templates
├── regulatory/                    # Regulatory compliance
│   ├── compliance_checker.py      #   HIPAA/GDPR/FDA checks
│   └── templates/                 #   IRB & consent templates
├── tests/                         # Comprehensive test suite
├── examples/                      # Example scripts
│   ├── run_federation.py          #   Full simulation
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
# Basic 3-site simulation
python examples/run_federation.py

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

### 2. Run Federated Training

```python
from federated.coordinator import FederationCoordinator
from federated.client import FederatedClient
from federated.model import ModelConfig

config = ModelConfig(input_dim=30, hidden_dims=[64, 32], output_dim=2)
coordinator = FederationCoordinator(model_config=config, num_rounds=10)
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
        client.train_local(global_params, epochs=5, lr=0.01)
        updates.append(client.get_parameters())
        counts.append(client.get_sample_count())

    result = coordinator.run_round(updates, client_sample_counts=counts,
                                    eval_data=(sites[0].x_test, sites[0].y_test))
    global_params = coordinator.get_global_parameters()
    print(f"Round {round_num+1}: accuracy={result.global_metrics['accuracy']:.4f}")
```

### 3. Enable Privacy Protections

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

### 4. Integrate Physical AI Digital Twins

```python
from physical_ai.digital_twin import PatientDigitalTwin, TumorModel

twin = PatientDigitalTwin(
    "patient_001",
    tumor=TumorModel(volume_cm3=3.5, chemo_sensitivity=0.6),
    biomarkers={"pdl1": 0.7, "ki67": 0.3},
)

# Simulate treatment response
result = twin.simulate_treatment("chemotherapy", dose_mg=75, cycles=4)
print(f"Predicted response: {result['response_category']}")

# Generate feature vector for federated model
features = twin.generate_feature_vector()
```

### 5. Run Compliance Checks

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

## Modules

### Federated Learning (`federated/`)

| Module | Description |
|--------|-------------|
| `coordinator.py` | Orchestrates FedAvg rounds, collects client updates, aggregates into global model |
| `client.py` | Simulated hospital node with local training — only parameters leave the site |
| `model.py` | Numpy-based MLP with backpropagation, supporting parameter extraction/injection |
| `secure_aggregation.py` | Pairwise mask protocol — coordinator sees only the aggregate, not individual updates |
| `differential_privacy.py` | Gaussian mechanism with gradient clipping, noise calibration, and epsilon budget tracking |
| `data_ingestion.py` | Synthetic oncology data generation with IID and non-IID site partitioning |

### Physical AI (`physical_ai/`)

| Module | Description |
|--------|-------------|
| `digital_twin.py` | Patient-specific tumor models with chemotherapy, radiation, and immunotherapy simulation |
| `robotic_integration.py` | Surgical robot interface for procedure planning, execution, and telemetry collection |
| `sensor_fusion.py` | Multi-modal clinical sensor fusion with real-time Z-score anomaly detection |
| `simulation_bridge.py` | Cross-platform robot model conversion (URDF/MJCF/SDF/USD) with physics validation |

### Privacy (`privacy/`)

| Module | Description |
|--------|-------------|
| `phi_detector.py` | Detects all 18 HIPAA identifiers using regex and field-name analysis |
| `deidentification.py` | Removes PHI via redact, hash, generalize, or replace strategies |
| `consent_manager.py` | Tracks patient consent with support for registration, verification, and revocation |
| `audit_logger.py` | Append-only event log with SHA-256 integrity hashing for tamper detection |

### Regulatory (`regulatory/`)

| Module | Description |
|--------|-------------|
| `compliance_checker.py` | Validates federation configs against HIPAA, GDPR, and FDA 21 CFR Part 11 |
| `templates/` | IRB protocol and informed consent templates for multi-site trials |

## Target Customers

- **Research Consortia** — Multi-institution oncology collaborations needing privacy-preserving model training.
- **Hospitals & Cancer Centers** — Institutions wanting to contribute to and benefit from collaborative AI without data-sharing barriers.
- **Pharmaceutical Companies** — Drug development programs requiring cross-site treatment response models.
- **Medical Device Companies** — Surgical robotics firms integrating AI-guided procedures across sites.
- **Clinical Research Organizations** — CROs managing multi-site oncology trials with AI endpoints.

## Regulatory Compliance

The platform includes infrastructure for:

- **HIPAA** — PHI detection and de-identification, audit logging, access control.
- **GDPR** — Consent management, right to erasure support, data minimization via DP.
- **FDA 21 CFR Part 11** — Audit trail requirements for electronic records.
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
  version = {0.1.0}
}
```

## Related

- [physical-ai-oncology-trials](https://github.com/kevinkawchak/physical-ai-oncology-trials) — Physical AI toolkit for oncology clinical trials (simulation, digital twins, agentic AI, regulatory compliance).
