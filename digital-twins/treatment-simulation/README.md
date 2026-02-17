# Treatment Simulation

**Version:** 0.4.0
**Last Updated:** 2026-02-17

## Purpose

The treatment simulation subsystem provides multi-modality treatment response modeling for oncology digital twins. It simulates chemotherapy pharmacokinetics/pharmacodynamics, radiation therapy cell survival, immunotherapy immune checkpoint dynamics, targeted therapy pathway inhibition, and combination therapy interactions. All simulations produce bounded, clinically plausible results aligned with RECIST 1.1 response criteria and CTCAE v5.0 toxicity grading.

## API Overview

### Enumerations

| Enum | Description |
|---|---|
| `TreatmentModality` | Treatment type selection (CHEMOTHERAPY, RADIATION, IMMUNOTHERAPY, TARGETED_THERAPY, COMBINATION) |
| `ResponseCategory` | RECIST 1.1 response classification (COMPLETE_RESPONSE, PARTIAL_RESPONSE, STABLE_DISEASE, PROGRESSIVE_DISEASE) |
| `ToxicityGrade` | CTCAE v5.0 toxicity grading (GRADE_0 through GRADE_5) |

### Data Classes

| Class | Description |
|---|---|
| `ChemotherapyProtocol` | Chemotherapy regimen parameters including drug, dose, schedule, and cycle count |
| `RadiationPlan` | Radiation therapy plan with total dose, fractionation, and organ-at-risk constraints |
| `ImmunotherapyRegimen` | Immunotherapy protocol including agent, dosing schedule, and expected response dynamics |
| `TreatmentOutcome` | Simulation outcome with volume trajectory, response category, and confidence intervals |
| `ToxicityProfile` | Multi-organ toxicity assessment with CTCAE grading |

### Simulator Class

| Class | Description |
|---|---|
| `TreatmentSimulator` | Core treatment simulation engine with methods for each modality and combination therapy |

### Key Methods

| Method | Description |
|---|---|
| `simulate_chemotherapy()` | PK/PD-based chemotherapy response simulation with cycle tracking |
| `simulate_radiation()` | Linear-quadratic model radiation response with fractionation |
| `simulate_immunotherapy()` | Stochastic immune response modeling with checkpoint dynamics |
| `simulate_combination()` | Multi-modality combination therapy with interaction effects |
| `assess_toxicity()` | CTCAE v5.0 toxicity profile estimation |

## DISCLAIMER

**RESEARCH USE ONLY.** This software is provided for research and educational purposes only. It has NOT been validated for clinical use, is NOT approved by the FDA or any other regulatory body, and MUST NOT be used for clinical decision-making. All simulation results require independent clinical validation.
