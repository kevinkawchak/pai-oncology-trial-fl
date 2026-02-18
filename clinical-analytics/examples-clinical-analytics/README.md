# Clinical Analytics Examples for PAI Oncology Trial Federated Learning

**Version:** 0.9.0
**Last Updated:** 2026-02-18
**License:** MIT

> **DISCLAIMER: RESEARCH USE ONLY.**
> This software is provided for research and educational purposes only.
> It has NOT been validated for clinical use, is NOT approved by the FDA
> or any other regulatory body, and MUST NOT be used to make clinical
> decisions or direct patient care. All outputs must be reviewed by
> qualified clinical professionals before any action is taken.

## Overview

Six production-patterned clinical analytics examples demonstrating end-to-end
analytical workflows for oncology clinical trials within the PAI Federated
Learning platform. Each script is self-contained, runs with only standard
library plus NumPy and SciPy, and follows structured logging conventions with
bounded clinical parameters throughout.

## Examples

| # | Script | Description | LOC |
|---|--------|-------------|-----|
| 01 | `01_basic_analytics_pipeline.py` | Minimal working example: configure, initialize, and run a descriptive analytics pipeline with the clinical analytics orchestrator | ~300 |
| 02 | `02_pkpd_modeling.py` | PK/PD deep-dive: one- and two-compartment models, AUC/Cmax/Tmax computation, Emax dose-response, and covariate effects on clearance | ~350 |
| 03 | `03_risk_stratification.py` | Risk stratification: synthetic patient features, composite risk scores, cohort stratification, decision support output, and calibration assessment | ~350 |
| 04 | `04_data_harmonization.py` | Data interoperability: ICD-10 to SNOMED mapping, cross-site schema alignment, dataset harmonization, alignment reporting, and CDISC SDTM export | ~350 |
| 05 | `05_survival_analysis.py` | Survival analysis: synthetic time-to-event data, Kaplan-Meier curves, log-rank tests, Cox PH model fitting, concordance index, and RMST | ~350 |
| 06 | `06_full_clinical_analytics_workflow.py` | End-to-end workflow combining orchestration, data management, harmonization, PK/PD, risk stratification, survival analysis, and DSMB reporting | ~380 |

## Quick Start

### Prerequisites

```bash
# Core dependencies (required)
pip install numpy>=1.24.0
pip install scipy>=1.11.0
```

### Running Examples

Each script can be run directly from the repository root:

```bash
# Run any example directly
python clinical-analytics/examples-clinical-analytics/01_basic_analytics_pipeline.py
python clinical-analytics/examples-clinical-analytics/02_pkpd_modeling.py
python clinical-analytics/examples-clinical-analytics/03_risk_stratification.py
python clinical-analytics/examples-clinical-analytics/04_data_harmonization.py
python clinical-analytics/examples-clinical-analytics/05_survival_analysis.py
python clinical-analytics/examples-clinical-analytics/06_full_clinical_analytics_workflow.py
```

### Linting

All scripts are validated against the project ruff configuration:

```bash
ruff check clinical-analytics/examples-clinical-analytics/
ruff format --check clinical-analytics/examples-clinical-analytics/
```

## Conventions

- **`#!/usr/bin/env python3`** shebang on every script
- **`from __future__ import annotations`** in every file
- **`sys.path.insert`** to add project root for orchestrator imports
- **`@dataclass`** for structured configuration and results
- **Structured logging** via `logging.getLogger`
- **Bounded clinical parameters** with validated ranges
- **RESEARCH USE ONLY** disclaimer in every module docstring
- **`if __name__ == "__main__":`** guard in every script

## References

- [ICH E6(R3) Good Clinical Practice](https://www.ich.org/)
- [FDA 21 CFR Part 11](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11)
- [CDISC SDTM Implementation Guide](https://www.cdisc.org/)
- [RECIST 1.1](https://doi.org/10.1016/j.ejca.2008.10.026)
- [ICH E9(R1) Estimands](https://www.ich.org/)
