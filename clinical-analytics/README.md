# Clinical Analytics — Federated Clinical Trial Analytics Platform

[![Version](https://img.shields.io/badge/version-0.9.0-blue.svg)](../CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

> **DISCLAIMER: RESEARCH USE ONLY** — This module provides research-grade clinical analytics
> tooling for federated oncology trials. It is not a substitute for qualified biostatistical
> review, DSMB oversight, or regulatory authority engagement. Independent validation, IRB
> approval, and regulatory clearance are required before deployment in any clinical setting.

## Overview

The `clinical-analytics/` module provides privacy-preserving clinical analytics for multi-site
oncology clinical trials operating under federated learning architectures. It enables
participating trial sites to perform local computations on their patient-level data and share
only aggregate statistics, summary curves, and differentially private model parameters — never
raw patient records. This design ensures that analytics workflows maintain the same data
residency guarantees enforced by the `privacy/` and `federated/` modules elsewhere in the
repository.

Where the `federated/` module focuses on distributed model training (FedAvg, FedProx,
SCAFFOLD) and secure aggregation of model weights, this module addresses the complementary
set of clinical analytics workflows that trial sponsors, biostatisticians, and Data Safety
Monitoring Boards (DSMBs) require throughout the trial lifecycle: population pharmacokinetic
modeling, survival analysis, risk stratification, adaptive enrichment, data quality monitoring,
regulatory reporting, and cross-site interoperability.

All analytics components are designed to operate within the broader PAI Oncology Trial FL
pipeline. Local computations execute within each site's trust boundary, intermediate results
are protected via secure aggregation and differential privacy, and global analytics are
reconstructed at the coordinating node without exposing individual patient data.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                Federated Clinical Analytics Pipeline                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                    │
│  │ Site A  │  │ Site B  │  │ Site C  │  │ Site N  │                    │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘                    │
│      │           │           │           │                           │
│      ▼           ▼           ▼           ▼                           │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Local Computation Layer                   │           │
│  │  PK/PD estimation · KM curves · Risk scores          │           │
│  │  CDISC transforms · Quality checks · Vocab mapping   │           │
│  └─────────────────────┬────────────────────────────────┘           │
│                        ▼                                            │
│  ┌──────────────────────────────────────────────────────┐           │
│  │            Secure Aggregation Layer                    │           │
│  │  Mask-based aggregation · DP noise · Cell suppression │           │
│  └─────────────────────┬────────────────────────────────┘           │
│                        ▼                                            │
│  ┌──────────────────────────────────────────────────────┐           │
│  │             Global Analytics Layer                     │           │
│  │  Pop PK/PD fitting · Combined survival · Enrichment  │           │
│  │  Consortium risk profiles · Data harmonization       │           │
│  └─────────────────────┬────────────────────────────────┘           │
│                        ▼                                            │
│  ┌──────────────────────────────────────────────────────┐           │
│  │               Reporting Layer                         │           │
│  │  DSMB packages · Regulatory submissions · Audit logs │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
clinical-analytics/
├── README.md                              # This file — module overview
├── __init__.py                            # Public API surface
├── analytics_orchestrator.py              # Pipeline orchestration and workflow management
├── pkpd_engine.py                         # Population PK/PD compartmental modeling
├── risk_stratification.py                 # Clinical risk scoring and adaptive enrichment
├── trial_data_manager.py                  # Data lifecycle, quality checks, and versioning
├── clinical_interoperability.py           # Vocabulary mapping and standards conversion
├── consortium_reporting.py                # DSMB report generation and regulatory packages
├── survival_analysis.py                   # Privacy-preserving Kaplan-Meier and Cox PH
└── examples-clinical-analytics/           # Usage examples and integration notebooks
    ├── orchestrator_quickstart.py         #   End-to-end pipeline example
    ├── survival_analysis_demo.py          #   Federated KM and Cox PH walkthrough
    └── pkpd_simulation.py                 #   Multi-compartment PK/PD example
```

## Component Descriptions

### analytics_orchestrator.py

**Primary class:** `ClinicalAnalyticsOrchestrator`

Manages the end-to-end clinical analytics pipeline across federated trial sites. The
orchestrator coordinates task scheduling, dispatches local computation requests, collects
intermediate results through secure aggregation, and assembles global analytics outputs.
It maintains a DAG of analytics tasks with dependency tracking and integrates with
`federated/coordinator.py` for aggregation infrastructure.

### pkpd_engine.py

**Primary class:** `PopulationPKPDEngine`

Implements federated population pharmacokinetic/pharmacodynamic modeling using compartmental
analysis. Supports one-, two-, and three-compartment models with first-order absorption.
Each site estimates local PK parameters (clearance, volume of distribution, absorption rate
constants) via nonlinear mixed-effects methods; the engine aggregates these into
population-level parameter distributions without exchanging individual concentration-time
profiles.

### risk_stratification.py

**Primary classes:** `ClinicalRiskStratifier`, `AdaptiveEnrichmentAdvisor`

`ClinicalRiskStratifier` computes multi-factor clinical risk scores incorporating tumor
burden, biomarker profiles, performance status, comorbidity indices, and treatment history.
Risk models are trained locally at each site and combined via federated averaging.

`AdaptiveEnrichmentAdvisor` uses interim risk stratification results to recommend
protocol-defined enrichment actions — identifying patient subpopulations that show
differential treatment effects and advising on eligibility criteria modifications for
adaptive trial designs.

### trial_data_manager.py

**Primary class:** `TrialDataManager`

Manages the clinical data lifecycle from ingestion through analysis-ready dataset creation.
Responsibilities include schema validation against CDISC SDTM and ADaM standards, automated
data quality checks (completeness, consistency, range, referential integrity), dataset
versioning with immutable audit trails, and query management for data discrepancies. All
operations produce 21 CFR Part 11 compliant audit records.

### clinical_interoperability.py

**Primary class:** `ClinicalInteroperabilityEngine`

Provides vocabulary mapping and data standards conversion across heterogeneous clinical
systems. Maps between coding systems (MedDRA, SNOMED CT, ICD-10, LOINC, RxNorm, WHO-DD)
and converts between data exchange formats (FHIR R4, CDISC ODM, HL7 v2). Maintains a
version-controlled terminology registry with site-specific mapping extensions to preserve
semantic equivalence across the consortium.

### consortium_reporting.py

**Primary class:** `ConsortiumReportingEngine`

Generates regulatory-grade reporting packages for DSMBs, trial sponsors, and health
authorities. Report types include DSMB safety summaries (blinded and unblinded views),
interim analysis packages, site enrollment dashboards, and regulatory submission appendices.
All reports embed provenance metadata linking each statistic to the pipeline version,
aggregation method, and privacy parameters used in its computation.

### survival_analysis.py

**Primary class:** `PrivacyPreservingSurvivalAnalyzer`

Implements federated survival analysis with formal privacy guarantees. Supports Kaplan-Meier
estimation by aggregating at-risk tables and event counts across sites without sharing
individual event times. Cox proportional hazards models are fitted via distributed
Newton-Raphson on the partial likelihood, exchanging only gradient and Hessian summaries.
Both methods integrate with the differential privacy module to apply calibrated noise when
cell sizes fall below configurable thresholds.

## Quick Start

### 1. Run a federated survival analysis

```python
from clinical_analytics.survival_analysis import PrivacyPreservingSurvivalAnalyzer
from clinical_analytics.analytics_orchestrator import ClinicalAnalyticsOrchestrator

# Initialize the orchestrator with site connections
orchestrator = ClinicalAnalyticsOrchestrator(
    site_ids=["site_boston", "site_houston", "site_seattle"],
    aggregation_backend="secure_agg",
    dp_epsilon=1.0,
)

# Run a federated Kaplan-Meier analysis
analyzer = PrivacyPreservingSurvivalAnalyzer(orchestrator=orchestrator)
km_result = analyzer.federated_kaplan_meier(
    time_col="os_months",
    event_col="os_event",
    group_col="treatment_arm",
)

print(f"Median OS (treatment): {km_result.median_survival['treatment']:.1f} months")
print(f"Median OS (control):   {km_result.median_survival['control']:.1f} months")
print(f"Log-rank p-value:      {km_result.log_rank_p:.4f}")
```

### 2. Run population PK/PD modeling

```python
from clinical_analytics.pkpd_engine import PopulationPKPDEngine

# Configure a two-compartment PK model
engine = PopulationPKPDEngine(
    compartments=2,
    absorption="first_order",
    estimation_method="foce_interaction",
)

# Fit across federated sites (local estimation + global aggregation)
pop_result = engine.fit_federated(
    orchestrator=orchestrator,
    dose_col="dose_mg",
    concentration_col="plasma_conc",
    time_col="hours_post_dose",
)

print(f"Population CL:  {pop_result.params['clearance']:.2f} L/h")
print(f"Population Vd:  {pop_result.params['volume_dist']:.2f} L")
print(f"Inter-site CV:  {pop_result.between_site_variability:.1%}")
```

### 3. Generate a DSMB safety report

```python
from clinical_analytics.consortium_reporting import ConsortiumReportingEngine
from clinical_analytics.risk_stratification import ClinicalRiskStratifier

# Build consortium-wide risk profiles
stratifier = ClinicalRiskStratifier(orchestrator=orchestrator)
risk_profiles = stratifier.compute_federated_risk_scores(
    features=["ecog_ps", "ldh_ratio", "tumor_burden_mm", "prior_lines"],
)

# Generate the DSMB package
reporter = ConsortiumReportingEngine(orchestrator=orchestrator)
dsmb_package = reporter.generate_dsmb_package(
    risk_profiles=risk_profiles,
    survival_result=km_result,
    report_type="interim_safety",
    unblinded=False,
)

dsmb_package.export_pdf("dsmb_interim_report_2026Q1.pdf")
print(f"Report generated: {dsmb_package.report_id}")
print(f"Sections: {', '.join(dsmb_package.sections)}")
```

## Compliance Alignment

| Standard | Requirement | Scope | Implementation |
|----------|------------|-------|----------------|
| **ICH E6(R3)** | Risk-proportionate monitoring | GCP principles for technology-enabled trials | `ClinicalAnalyticsOrchestrator` implements risk-proportionate analytics scheduling; `ClinicalRiskStratifier` provides continuous quality-by-design monitoring |
| **21 CFR Part 11** | Audit trails, electronic signatures | Electronic records and signatures for FDA-regulated activities | `TrialDataManager` produces immutable audit records for all data transformations; `ConsortiumReportingEngine` embeds provenance metadata and signature fields |
| **HIPAA** | PHI protection, minimum necessary | Protected health information safeguards (45 CFR 164) | All analytics operate on de-identified data via `privacy/` module; `PrivacyPreservingSurvivalAnalyzer` enforces minimum cell-size suppression and differential privacy |
| **FDA AI/ML** | Model performance monitoring | FDA Action Plan for AI/ML-based SaMD | `ClinicalRiskStratifier` tracks model performance drift across federation rounds; `ClinicalAnalyticsOrchestrator` logs model versioning and performance metrics |
| **CDISC** | SDTM/ADaM data standards | Clinical Data Interchange Standards Consortium | `TrialDataManager` validates datasets against SDTM and ADaM specifications; `ClinicalInteroperabilityEngine` performs vocabulary mapping to CDISC-controlled terminology |
| **IEC 62304** | Software lifecycle traceability | Medical device software lifecycle processes | All components produce structured logs traceable to pipeline version, configuration hash, and execution context; `ConsortiumReportingEngine` links outputs to software release identifiers |

## Architecture Principles

1. **Data never leaves the site in identifiable form** — All analytics workflows consume
   de-identified data. Patient-level records remain within each site's trust boundary;
   only aggregate statistics and differentially private summaries cross site boundaries.

2. **Reproducible analytics** — Every pipeline execution is versioned. The orchestrator
   records the full configuration (component versions, privacy parameters, aggregation
   method, site participation) in an immutable execution manifest.

3. **Regulatory traceability** — Each reported statistic links back to the analytics
   pipeline version, the aggregation method, the privacy budget consumed, and the set
   of contributing sites. This chain of custody supports 21 CFR Part 11 and ICH E6(R3)
   audit requirements.

4. **Graceful degradation** — If a site drops out mid-computation, the orchestrator
   re-plans the aggregation using available sites and flags the reduced denominator in
   all downstream outputs.

5. **Privacy budget management** — The orchestrator tracks cumulative differential privacy
   expenditure (epsilon) across all analytics queries within a reporting period, refusing
   new queries that would exceed the pre-approved privacy budget.

## Roadmap Alignment

The following capabilities are planned for future releases, aligned with the broader
PAI Oncology Trial FL roadmap in `q1-2026-standards/`:

- **Real-time analytics streaming** — Event-driven pipeline supporting continuous safety
  signal detection as new data arrives at sites, replacing the current batch workflow.
- **Bayesian adaptive trial designs** — Bayesian response-adaptive randomization and
  predictive probability of success integrated into `AdaptiveEnrichmentAdvisor` for
  data-driven protocol modifications during ongoing trials.
- **Genomic data integration** — Extension of `ClinicalInteroperabilityEngine` to support
  federated analytics on genomic data (VCF, MAF) for biomarker-driven subgroup analyses
  and precision oncology workflows.

## Related Modules

- [`federated/`](../federated/) — Secure aggregation, differential privacy, federated model training
- [`privacy/`](../privacy/) — PHI detection, de-identification, consent management, RBAC
- [`regulatory/`](../regulatory/) — FDA submission tracking, ICH E6(R3) GCP compliance, IRB management
- [`digital-twins/`](../digital-twins/) — Patient digital twins, treatment simulation, clinical integration
- [`q1-2026-standards/`](../q1-2026-standards/) — Cross-framework compliance objectives

## License

MIT License. See [LICENSE](../LICENSE) for details.
