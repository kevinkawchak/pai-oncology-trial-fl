# Visualization Infrastructure — PAI Oncology Trial FL

## Overview

Publication-quality interactive visualizations generated from repository data using Plotly. All charts support light and dark rendering modes and are self-standing with sufficient context in titles, labels, and annotations.

## Prompt-to-Visualization Workflow

```
Prompt (images/prompts/)     Python Script (images/interactive/)     Output (HTML + PNG)
┌──────────────────────┐     ┌──────────────────────────────────┐     ┌─────────────────────┐
│ plan.md              │     │ create_[name](dark_mode=False)   │     │ [name]_light.html   │
│ 1st.md               │────>│ plotly.graph_objects              │────>│ [name]_dark.html    │
│ 2nd.md               │     │ plotly_white / plotly_dark        │     │ [name]_light.png    │
│ 3rd.md               │     │ 100-250 LOC per script           │     │ [name]_dark.png     │
└──────────────────────┘     └──────────────────────────────────┘     └─────────────────────┘
```

## Pipeline

```
Python Script (.py)
    │
    ├──> plotly.graph_objects.Figure
    │        │
    │        ├──> fig.write_html("name_light.html", include_plotlyjs=True)
    │        ├──> fig.write_html("name_dark.html", include_plotlyjs=True)
    │        ├──> fig.write_image("name_light.png", width=1920, height=1080, scale=2)
    │        └──> fig.write_image("name_dark.png", width=1920, height=1080, scale=2)
    │
    └──> Validated by: ruff check + ruff format --check
```

## Conversion Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Total scripts | 30/30 | Complete |
| HTML outputs (light + dark) | 60/60 | Ready to generate |
| PNG outputs (light + dark) | 60/60 | Ready to generate |
| Scripts passing ruff check | 30/30 | Validated |
| Scripts passing ruff format | 30/30 | Validated |

## Directory Structure

```
images/
├── README.md                          # This file
├── prompts/                           # Visualization planning documents
│   ├── plan.md                        # Overall strategy (30 charts, light+dark)
│   ├── 1st.md                         # Set 1 per-chart specifications
│   ├── 2nd.md                         # Set 2 per-chart specifications
│   └── 3rd.md                         # Set 3 per-chart specifications
└── interactive/                       # Python visualization scripts
    ├── 1st/                           # Set 1: Architecture & Simulation
    │   ├── README.md                  # Script table
    │   ├── 01_repository_architecture_treemap.py
    │   ├── 02_clinical_trial_workflow.py
    │   ├── 03_oncology_process_diagram.py
    │   ├── 04_framework_comparison_radar.py
    │   ├── 05_parameter_mapping_heatmap.py
    │   ├── 06_data_pipeline_throughput.py
    │   ├── 07_model_format_comparison.py
    │   ├── 08_state_vector_visualization.py
    │   ├── 09_domain_randomization_transfer.py
    │   └── 10_oncology_trajectories.py
    ├── 2nd/                           # Set 2: Training & Benchmarks
    │   ├── README.md
    │   ├── 01_training_curves.py
    │   ├── 02_benchmark_comparisons.py
    │   ├── 03_safety_metrics_dashboard.py
    │   ├── 04_performance_radars.py
    │   ├── 05_accuracy_confusion_matrix.py
    │   ├── 06_sim_to_real_transfer.py
    │   ├── 07_latency_distributions.py
    │   ├── 08_resource_utilization.py
    │   ├── 09_oncology_kpi_tracker.py
    │   └── 10_cross_framework_consistency.py
    └── 3rd/                           # Set 3: Regulatory & Readiness
        ├── README.md
        ├── 01_oncology_convergence.py
        ├── 02_multi_site_trial_dashboard.py
        ├── 03_algorithm_comparison_radar.py
        ├── 04_fda_device_classification_tree.py
        ├── 05_oncology_device_distribution.py
        ├── 06_regulatory_compliance_scorecard.py
        ├── 07_hipaa_phi_detection_matrix.py
        ├── 08_privacy_analytics_pipeline.py
        ├── 09_deployment_readiness_radar.py
        └── 10_production_readiness_scores.py
```

## Per-Set LOC Tables

### Set 1 — Repository Architecture and Simulation

| # | Script | Chart Type | LOC |
|---|--------|-----------|-----|
| 01 | repository_architecture_treemap | Treemap | ~150 |
| 02 | clinical_trial_workflow | Sankey | ~160 |
| 03 | oncology_process_diagram | Sunburst | ~170 |
| 04 | framework_comparison_radar | Scatterpolar | ~150 |
| 05 | parameter_mapping_heatmap | Heatmap | ~160 |
| 06 | data_pipeline_throughput | Bar | ~140 |
| 07 | model_format_comparison | Bar (grouped) | ~150 |
| 08 | state_vector_visualization | Scatter3d | ~170 |
| 09 | domain_randomization_transfer | Scatter (lines) | ~140 |
| 10 | oncology_trajectories | Scatter (multi-line) | ~180 |

### Set 2 — Training and Benchmarks

| # | Script | Chart Type | LOC |
|---|--------|-----------|-----|
| 01 | training_curves | Scatter (lines) | ~140 |
| 02 | benchmark_comparisons | Bar (grouped) | ~150 |
| 03 | safety_metrics_dashboard | Subplots (2x2) | ~200 |
| 04 | performance_radars | Scatterpolar | ~160 |
| 05 | accuracy_confusion_matrix | Heatmap | ~170 |
| 06 | sim_to_real_transfer | Scatter (dual-axis) | ~160 |
| 07 | latency_distributions | Box | ~150 |
| 08 | resource_utilization | Scatter (stacked area) | ~160 |
| 09 | oncology_kpi_tracker | Indicator | ~180 |
| 10 | cross_framework_consistency | Scatter | ~140 |

### Set 3 — Regulatory and Readiness

| # | Script | Chart Type | LOC |
|---|--------|-----------|-----|
| 01 | oncology_convergence | Scatter (lines) | ~150 |
| 02 | multi_site_trial_dashboard | Subplots (2x2) | ~210 |
| 03 | algorithm_comparison_radar | Scatterpolar | ~150 |
| 04 | fda_device_classification_tree | Treemap | ~160 |
| 05 | oncology_device_distribution | Sunburst | ~150 |
| 06 | regulatory_compliance_scorecard | Bar (horizontal) | ~170 |
| 07 | hipaa_phi_detection_matrix | Heatmap | ~170 |
| 08 | privacy_analytics_pipeline | Sankey | ~170 |
| 09 | deployment_readiness_radar | Scatterpolar | ~150 |
| 10 | production_readiness_scores | Bar (horizontal) | ~150 |

**Estimated Total:** ~4,800 LOC across 30 scripts

## Data Source Reference

| Data Source | Module Path | Charts Using |
|-------------|-------------|-------------|
| Module file counts | All directories | 1.01 |
| Trial workflow phases | regulatory/, clinical-analytics/ | 1.02 |
| Treatment modalities, drug classes | digital-twins/treatment-simulation/ | 1.03, 1.10 |
| Framework capabilities | physical_ai/, frameworks/ | 1.04, 2.02 |
| Physics parameters | unification/simulation_physics/ | 1.05, 2.07, 2.10 |
| Data pipeline stages | federated/, clinical-analytics/ | 1.06 |
| Model formats | q1-2026-standards/ | 1.07 |
| Simulation state vectors | unification/simulation_physics/ | 1.08 |
| Training config | configs/training_config.yaml | 1.09 |
| Tumor growth models | digital-twins/patient-modeling/ | 1.10, 3.01 |
| Aggregation strategies | federated/coordinator.py | 2.01, 3.03 |
| Benchmark metrics | q1-2026-standards/ | 2.02 |
| Safety constraints | physical_ai/, configs/ | 2.03 |
| Platform modules | All directories | 2.04, 3.10 |
| Tumor types | digital-twins/patient-modeling/ | 2.05 |
| Transfer learning | physical_ai/ | 2.06 |
| Resource utilization | federated/ | 2.08 |
| Clinical KPIs | clinical-analytics/ | 2.09 |
| Convergence metrics | federated/, clinical-analytics/ | 3.01 |
| Site enrollment | tools/trial-site-monitor/ | 3.02 |
| FDA device classes | regulatory/, regulatory-submissions/ | 3.04, 3.05 |
| Compliance scores | regulatory-submissions/ | 3.06 |
| HIPAA identifiers | privacy/phi-pii-management/ | 3.07 |
| De-identification methods | privacy/de-identification/ | 3.08 |
| Deployment checks | tools/deployment-readiness/ | 3.09 |

## Data Inputs Table (All 30 Charts)

| Chart | Primary Input | Secondary Input | Values |
|-------|--------------|-----------------|--------|
| 1.01 Repository Treemap | Directory scan | File counts | 194 Python files across 10+ modules |
| 1.02 Trial Workflow | Workflow phases | Patient counts | 7 phases, 1000 patient flow |
| 1.03 Oncology Process | TreatmentModality enum | DrugClass, ResponseCategory | 5 modalities, 7 drug classes, 4 responses |
| 1.04 Framework Radar | Framework INTEGRATION.md | Capability scores | 4 frameworks, 8 axes |
| 1.05 Parameter Heatmap | ParameterMapper class | Mapping scores | 8x8 parameter grid |
| 1.06 Pipeline Throughput | Pipeline stage names | Throughput metrics | 6 stages, 3 site sizes |
| 1.07 Model Comparison | ModelFormat enum | Performance metrics | 5 formats, 5 metrics |
| 1.08 State Vectors | JointState dataclass | Position/velocity/torque | 6 joints, 200 time steps |
| 1.09 Domain Randomization | Training config | Success rates | 4 levels, 500 epochs |
| 1.10 Tumor Trajectories | GrowthModel enum | Volume curves | 4 models, 365 days |
| 2.01 Training Curves | AggregationStrategy enum | Loss curves | 3 strategies, 100 rounds |
| 2.02 Benchmarks | BenchmarkMetric enum | Framework scores | 4 frameworks, 4 metrics |
| 2.03 Safety Dashboard | Safety constraints | Force/torque data | 4 panels, multiple metrics |
| 2.04 Performance Radars | Module capabilities | Radar scores | 6 modules, 5 axes |
| 2.05 Confusion Matrix | TumorType enum | Classification counts | 12x12 matrix |
| 2.06 Sim-to-Real | Transfer methods | Success/error | 3 methods, 100 iterations |
| 2.07 Latency Distributions | SyncReport timing | Distribution data | 3 directions, 1000 samples |
| 2.08 Resource Utilization | Training resources | Time series | 5 layers, 120 minutes |
| 2.09 KPI Tracker | Clinical KPIs | Target/actual | 6 KPIs with targets |
| 2.10 Framework Consistency | Cross-engine states | Joint positions | 6 joints, 500 steps |
| 3.01 Convergence | Clinical subtasks | Validation AUC | 5 tasks, 200 rounds |
| 3.02 Trial Dashboard | Site enrollment | Multi-panel | 10 sites, 4 panels |
| 3.03 Algorithm Radar | Aggregation strategies | Trade-off scores | 3 algorithms, 6 axes |
| 3.04 FDA Tree | DeviceClass enum | Submission pathways | 3 classes, 4 types |
| 3.05 Device Distribution | FDA device landscape | Category/status | 4 categories, 3 statuses |
| 3.06 Compliance Scorecard | Regulation enum | Compliance scores | 9 regulations, 3 levels |
| 3.07 PHI Matrix | PHICategory enum | Detection confidence | 18 identifiers, 5 sources |
| 3.08 Privacy Pipeline | DeidentificationMethod | Record flow | 6 methods, 7 stages |
| 3.09 Readiness Radar | CheckStatus enum | Readiness scores | 10 axes |
| 3.10 Production Scores | Platform modules | Readiness % | 9 modules with targets |
