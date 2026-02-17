# Examples — Physical AI Federated Learning for Oncology Clinical Trials

> **Version:** 0.5.0 | **Last Updated:** 2026-02-17

This directory contains end-to-end example scripts demonstrating the core capabilities of the Physical AI Federated Learning platform for oncology clinical trials.  Each example is self-contained, uses synthetic data, and can run without GPU or external hardware.

**DISCLAIMER: RESEARCH USE ONLY.**  These examples are provided for research and educational purposes only.  They have NOT been validated for clinical use and MUST NOT be used to make clinical decisions or direct patient care.

---

## Examples Overview

| # | Example | Description | Key Concepts | Frameworks |
|---|---------|-------------|--------------|------------|
| — | `run_federation.py` | Run a complete federated learning simulation with multi-site training, compliance checks, and audit logging | FedAvg / FedProx / SCAFFOLD, site enrollment, consent management, differential privacy, secure aggregation | numpy, federated, privacy, regulatory |
| — | `generate_synthetic_data.py` | Generate synthetic de-identified oncology datasets partitioned across federated sites | Data partitioning (IID / non-IID), feature generation, CSV export | numpy, pandas, federated |
| 01 | `01_federated_training_workflow.py` | Main federated training workflow with full pipeline orchestration, strategy comparison, and reporting | WorkflowManager, AggregationMode enum, PrivacyLevel presets, stage-based pipeline, convergence detection, training curve visualization | numpy, federated, privacy, regulatory, matplotlib (optional) |
| 02 | `02_digital_twin_planning.py` | Digital twin treatment planning with multi-protocol simulation and uncertainty quantification | PatientDigitalTwin, TumorModel, TreatmentProtocol, Monte-Carlo UQ, RECIST-like response classification, cohort planning | numpy, physical\_ai, matplotlib (optional) |
| 03 | `03_cross_framework_validation.py` | Cross-framework and cross-platform validation of surgical robot policies | SimulationBridge, ValidationSuite, roundtrip URDF/MJCF/SDF conversion, kinematics/safety/performance validation, threshold sensitivity | numpy, physical\_ai, unification, mujoco (optional), pybullet (optional) |
| 04 | `04_agentic_clinical_workflow.py` | AI-driven multi-agent clinical trial orchestration with audit trails | UnifiedAgent, AgentConfig, ToolRegistry, PHI detection, hash-chained audit trail, multi-agent synthesis, human-in-the-loop review | numpy, unification, crewai (optional), langgraph (optional), autogen (optional) |
| 05 | `05_outcome_prediction_pipeline.py` | End-to-end outcome prediction pipeline with digital twin feature augmentation | PatientCohortGenerator, DigitalTwinFeatureAugmenter, federated training, confusion matrix evaluation, augmentation comparison | numpy, federated, physical\_ai, privacy, sklearn (optional) |

---

## Quick Start

### Prerequisites

All examples require Python 3.10+ and the following core dependencies:

```bash
pip install numpy>=1.24.0 scipy>=1.11.0
```

Optional dependencies unlock additional features (visualization, tabular export, etc.):

```bash
# Visualization and data export
pip install matplotlib>=3.7.0 pandas>=2.0.0

# Machine learning metrics
pip install scikit-learn>=1.3.0

# Agentic frameworks (example 04)
pip install crewai>=0.30.0 langgraph>=0.1.0 autogen>=0.2.0

# Physics engines (example 03)
pip install mujoco>=3.0.0 pybullet>=3.2.5
```

### Running the Examples

All examples should be run from the **project root directory**:

```bash
cd pai-oncology-trial-fl
```

#### Existing Examples

```bash
# Run a federated learning simulation
python examples/run_federation.py
python examples/run_federation.py --num-sites 4 --rounds 20 --strategy fedprox

# Generate synthetic oncology datasets
python examples/generate_synthetic_data.py
python examples/generate_synthetic_data.py --output-dir data/ --num-sites 5
```

#### Example 01 — Federated Training Workflow

Demonstrates the complete federated training lifecycle with pipeline
orchestration, strategy comparison, and privacy tracking.

```bash
python examples/01_federated_training_workflow.py
```

Key features:
- Stage-based pipeline: compliance check, site enrollment, consent management, data preparation, model initialization, training, evaluation, reporting
- Aggregation strategy comparison (FedAvg vs FedProx vs SCAFFOLD)
- Pre-configured privacy levels (none, low, medium, high)
- Training curve visualization (when matplotlib is installed)

#### Example 02 — Digital Twin Treatment Planning

Demonstrates creating patient digital twins, simulating treatment protocols,
running Monte-Carlo uncertainty quantification, and generating clinical reports.

```bash
python examples/02_digital_twin_planning.py
```

Key features:
- Multi-patient cohort creation with realistic tumor parameters
- Five standard treatment protocols (chemo, radiation, immuno, combinations)
- Monte-Carlo uncertainty quantification with confidence intervals
- Structured clinical reports with treatment comparison tables

#### Example 03 — Cross-Framework Validation

Demonstrates validating a surgical robot policy across Isaac Sim, MuJoCo,
and PyBullet with roundtrip model format conversion testing.

```bash
python examples/03_cross_framework_validation.py
```

Key features:
- Cross-engine kinematic consistency validation (positions, velocities)
- Safety constraint validation (forces, velocities, workspace bounds)
- Roundtrip model conversion (URDF to MJCF to URDF, URDF to SDF to URDF)
- Threshold sensitivity analysis at multiple tolerance levels
- Regulatory compliance report generation (IEC 62304, IEC 80601-2-77)

#### Example 04 — Agentic Clinical Workflow

Demonstrates AI-driven multi-agent orchestration for clinical trial
management with treatment planning, literature review, and compliance agents.

```bash
python examples/04_agentic_clinical_workflow.py
```

Key features:
- Three specialized agents: treatment planner, literature reviewer, compliance checker
- Shared tool registry with MCP, OpenAI, Anthropic, CrewAI, and LangGraph format export
- Hash-chained audit trail with FDA 21 CFR Part 11 compliance
- PHI detection on agent inputs and outputs
- Simulated human-in-the-loop review

#### Example 05 — Outcome Prediction Pipeline

Demonstrates the end-to-end prediction pipeline from raw patient data to
clinical outcome reports, including digital twin feature augmentation.

```bash
python examples/05_outcome_prediction_pipeline.py
```

Key features:
- Patient cohort generation with de-identified clinical features
- Digital twin feature augmentation (tumor simulations as additional features)
- Federated model training with differential privacy
- Per-site and global evaluation with confusion matrices
- Augmentation comparison (raw features vs. twin-augmented features)

---

## Common Patterns

All example scripts follow these conventions:

- **`from __future__ import annotations`** for modern type hint syntax.
- **`from dataclasses import dataclass, field`** for configuration and result objects.
- **`from enum import Enum`** for type-safe enumerations.
- **`logging.basicConfig()` + `logger = logging.getLogger(__name__)`** for structured logging.
- **Conditional imports with `HAS_X` flags** for optional dependencies, ensuring examples are importable without GPU or specialized hardware.
- **`if __name__ == "__main__"` block** for standalone execution.
- **RESEARCH USE ONLY disclaimer** in all output and docstrings.

---

## License

MIT License.  See the project root [LICENSE](../LICENSE) file for details.
