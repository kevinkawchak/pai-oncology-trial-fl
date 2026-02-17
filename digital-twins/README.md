# Digital Twins Platform

**Version:** 0.4.0
**Last Updated:** 2026-02-17
**License:** MIT
**Status:** Research Use Only

## Overview

The Digital Twins platform provides patient-specific computational models for oncology clinical trials within the PAI Federated Learning framework. Each digital twin mirrors a real patient's tumor characteristics, physiological state, and treatment response dynamics, enabling safer treatment planning and federated model validation without sharing raw patient data across institutional boundaries.

Digital twins serve as the bridge between raw clinical observations and actionable federated learning insights. By creating high-fidelity computational representations of individual patients, clinical trial sites can simulate treatment outcomes, explore therapeutic strategies, and contribute to multi-site model training while preserving patient privacy.

## Architecture

The platform is organized into three interconnected subsystems:

```
digital-twins/
├── README.md                          # This file
├── patient-modeling/
│   ├── README.md                      # Patient modeling documentation
│   └── patient_model.py               # Patient digital twin modeling engine
├── treatment-simulation/
│   ├── README.md                      # Treatment simulation documentation
│   └── treatment_simulator.py         # Multi-modality treatment simulator
└── clinical-integration/
    ├── README.md                      # Clinical integration documentation
    └── clinical_integrator.py         # Clinical system integration bridge
```

### Data Flow

```
Clinical Systems (EHR/PACS/LIS)
        │
        ▼
┌──────────────────────┐
│ Clinical Integration │  ◄── HL7 FHIR, DICOM, CSV ingestion
│  (clinical_integrator)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Patient Modeling    │  ◄── Tumor kinetics, biomarkers, organ function
│  (patient_model)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Treatment Simulation │  ◄── Chemo, radiation, immuno, combination
│ (treatment_simulator)│
└──────────┬───────────┘
           │
           ▼
   Federated Learning Pipeline
   (federated/ module)
```

## Directory Structure

| Directory | Purpose |
|---|---|
| `patient-modeling/` | Core patient digital twin modeling including tumor growth kinetics, biomarker evolution, organ function tracking, and physiological state management |
| `treatment-simulation/` | Multi-modality treatment simulation including chemotherapy pharmacokinetics, radiation linear-quadratic modeling, immunotherapy response, and combination therapy |
| `clinical-integration/` | Integration layer for clinical systems (EHR, PACS, LIS, RIS, TPS) with data format translation and federated learning pipeline connectivity |

## Key Capabilities

### Patient Modeling

- **Tumor Growth Kinetics**: Exponential, logistic, Gompertz, and von Bertalanffy growth models calibrated to patient-specific parameters
- **Biomarker Evolution**: Time-series modeling of circulating tumor DNA (ctDNA), CEA, CA-125, PSA, and other relevant biomarkers
- **Organ Function Tracking**: Hepatic, renal, cardiac, and pulmonary function modeling to assess treatment tolerability
- **Multi-Tumor Support**: NSCLC, SCLC, breast (ductal/lobular), colorectal, pancreatic, and extensible to additional tumor types

### Treatment Simulation

- **Chemotherapy**: Pharmacokinetic/pharmacodynamic modeling with cycle-based response tracking and dose-limiting toxicity estimation
- **Radiation Therapy**: Linear-quadratic cell survival modeling with fractionation schedules and organ-at-risk dose constraints
- **Immunotherapy**: Stochastic response modeling incorporating immune checkpoint dynamics and tumor mutational burden
- **Combination Therapy**: Sequential and concurrent multi-modality treatment simulation with drug interaction effects
- **Toxicity Profiling**: CTCAE v5.0 aligned toxicity grading from Grade 0 through Grade 5

### Clinical Integration

- **Data Ingestion**: HL7 FHIR, DICOM, CSV, and JSON format support for clinical data import
- **System Connectivity**: EHR, PACS, LIS, RIS, and TPS integration interfaces
- **Report Generation**: Structured clinical reports for treatment planning review
- **Federated Bridge**: Seamless connectivity between digital twin outputs and the federated learning training pipeline

## Regulatory Alignment

This module is developed with awareness of the following regulatory frameworks:

- **FDA 21 CFR Part 11**: Electronic records and electronic signatures compliance considerations
- **FDA Guidance on Clinical Decision Support Software**: Classification and regulatory pathway awareness
- **EU MDR 2017/745**: Medical device regulation considerations for software as a medical device (SaMD)
- **ICH E6(R2) GCP**: Good Clinical Practice alignment for clinical trial data handling
- **HIPAA**: Protected Health Information safeguards integrated throughout the pipeline

All digital twin outputs are clearly marked as research-use-only and must not be used for clinical decision-making without independent validation and appropriate regulatory clearance.

## Integration with Unification Framework

The digital twins platform integrates with the broader PAI Oncology Trial FL unification framework:

- **Cross-Platform Tools** (`unification/cross_platform_tools/`): Digital twin models can be converted between PyTorch, TensorFlow, and ONNX formats using the model converter
- **Simulation Physics** (`unification/simulation_physics/`): The Isaac Sim and MuJoCo bridge supports digital twin visualization and surgical planning simulation
- **Federated Pipeline** (`federated/`): Digital twin feature vectors feed directly into the federated averaging workflow for multi-site model training
- **Privacy Module** (`privacy/`): All digital twin data passes through de-identification and PHI detection before any federated communication

## Quick Start

```python
from digital_twins.patient_modeling.patient_model import (
    PatientPhysiologyEngine,
    PatientDigitalTwinFactory,
    TumorType,
    GrowthModel,
)

# Create a patient digital twin
factory = PatientDigitalTwinFactory()
twin = factory.create_twin(
    patient_id="TRIAL-001-PT-042",
    tumor_type=TumorType.NSCLC,
    growth_model=GrowthModel.GOMPERTZ,
)

# Simulate tumor growth
engine = PatientPhysiologyEngine(twin)
trajectory = engine.simulate_tumor_growth(days=90, time_step_days=1)
```

## Requirements

- Python >= 3.10
- NumPy >= 1.24.0
- SciPy >= 1.11.0 (recommended)
- PyTorch >= 2.5.0 (optional, for neural surrogate models)
- MONAI >= 1.3.0 (optional, for medical imaging integration)
