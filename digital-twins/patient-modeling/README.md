# Patient Modeling

**Version:** 0.4.0
**Last Updated:** 2026-02-17

## Purpose

The patient modeling subsystem provides the core computational engine for creating and managing patient digital twins in oncology clinical trials. It models tumor growth kinetics, biomarker evolution, organ function, and overall physiological state to enable high-fidelity treatment response prediction within the federated learning framework.

Each patient digital twin encapsulates de-identified clinical parameters sufficient to simulate treatment outcomes without exposing protected health information (PHI). The modeling engine supports multiple tumor types (NSCLC, SCLC, breast ductal/lobular, colorectal, pancreatic) and growth models (exponential, logistic, Gompertz, von Bertalanffy).

## API Overview

### Enumerations

| Enum | Description |
|---|---|
| `TumorType` | Supported tumor histology types (NSCLC, SCLC, BREAST_DUCTAL, BREAST_LOBULAR, COLORECTAL, PANCREATIC, etc.) |
| `GrowthModel` | Tumor growth model selection (EXPONENTIAL, LOGISTIC, GOMPERTZ, VON_BERTALANFFY) |
| `PatientStatus` | Patient clinical status tracking (BASELINE, ON_TREATMENT, FOLLOW_UP, PROGRESSION, REMISSION) |

### Data Classes

| Class | Description |
|---|---|
| `TumorProfile` | Tumor characteristics including volume, growth rate, sensitivities, and histology |
| `PatientBiomarkers` | Collection of clinically relevant biomarker measurements with timestamps |
| `OrganFunction` | Hepatic, renal, cardiac, and pulmonary function parameters |
| `PatientDigitalTwinConfig` | Complete configuration for instantiating a patient digital twin |

### Engine Classes

| Class | Description |
|---|---|
| `PatientPhysiologyEngine` | Core simulation engine for tumor growth, biomarker evolution, and treatment response |
| `PatientDigitalTwinFactory` | Factory for creating configured patient digital twins with validated parameters |

## Usage Examples

### Creating a Patient Digital Twin

```python
from patient_model import (
    PatientDigitalTwinFactory,
    TumorType,
    GrowthModel,
    PatientStatus,
    TumorProfile,
    PatientBiomarkers,
)

# Create via factory with defaults
factory = PatientDigitalTwinFactory()
twin_config = factory.create_twin(
    patient_id="TRIAL-001-PT-042",
    tumor_type=TumorType.NSCLC,
    growth_model=GrowthModel.GOMPERTZ,
)

# Create with custom tumor profile
profile = TumorProfile(
    tumor_type=TumorType.BREAST_DUCTAL,
    volume_cm3=3.5,
    growth_rate_per_day=0.012,
    chemo_sensitivity=0.65,
    radio_sensitivity=0.55,
    immuno_sensitivity=0.40,
)
```

### Simulating Tumor Growth

```python
from patient_model import PatientPhysiologyEngine

engine = PatientPhysiologyEngine(twin_config)

# Simulate 90 days of tumor growth
trajectory = engine.simulate_tumor_growth(days=90, time_step_days=1)
print(f"Initial volume: {trajectory[0]:.2f} cm3")
print(f"Final volume: {trajectory[-1]:.2f} cm3")

# Simulate biomarker evolution
biomarker_series = engine.simulate_biomarker_evolution(
    biomarker_name="ctDNA",
    days=90,
    time_step_days=7,
)
```

### Generating Feature Vectors for Federated Learning

```python
# Generate de-identified feature vector for federated model input
features = engine.generate_feature_vector()
print(f"Feature vector shape: {features.shape}")
print(f"Feature vector dtype: {features.dtype}")
```

## DISCLAIMER

**RESEARCH USE ONLY.** This software is provided for research and educational purposes only. It has NOT been validated for clinical use, is NOT approved by the FDA or any other regulatory body, and MUST NOT be used for clinical decision-making. All simulation results require independent clinical validation.
