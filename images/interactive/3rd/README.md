# Interactive Visualizations - Batch 3

Third batch of interactive Plotly visualizations for the PAI Oncology Trial Federated Learning platform. All scripts use `plotly.graph_objects` exclusively (no `plotly.express`) and support both light and dark mode via the `dark_mode` parameter.

## Scripts

| # | Script | Description | Chart Type | LOC |
|---|--------|-------------|------------|-----|
| 1 | `01_oncology_convergence.py` | Federated model convergence across oncology subtasks (Tumor Classification, Response Prediction, Survival Estimation, Toxicity Grading, Biomarker Detection) | Multi-line chart | 142 |
| 2 | `02_multi_site_trial_dashboard.py` | Multi-panel dashboard: enrollment waterfall by site, screening funnel, monthly trend, geographic bubble chart | Subplots (bar, funnel, scatter) | 189 |
| 3 | `03_algorithm_comparison_radar.py` | Comparison of FedAvg, FedProx, and SCAFFOLD across six performance dimensions | Radar chart | 136 |
| 4 | `04_fda_device_classification_tree.py` | FDA device classification hierarchy: Device Classes, Submission Types, AI/ML Categories (SaMD, SiMD, PCCP) | Treemap | 246 |
| 5 | `05_oncology_device_distribution.py` | Oncology AI/ML device distribution by category, regulation status, and device class | Sunburst chart | 172 |
| 6 | `06_regulatory_compliance_scorecard.py` | Compliance scores across 9 regulations (FDA, HIPAA, GDPR, ISO, IEC, ICH, EU MDR) with stacked status bars | Horizontal stacked bar | 155 |
| 7 | `07_hipaa_phi_detection_matrix.py` | PHI detection rates for 18 HIPAA Safe Harbor identifiers across 5 clinical data sources | Annotated heatmap | 175 |
| 8 | `08_privacy_analytics_pipeline.py` | De-identification data flow from raw sources through PHI detection, method routing, and validation | Sankey diagram | 200 |
| 9 | `09_deployment_readiness_radar.py` | Deployment readiness across 10 dimensions with current, target, and previous quarter scores | Radar chart | 163 |
| 10 | `10_production_readiness_scores.py` | Production readiness scores per platform module with target threshold and pass/fail indicators | Horizontal bullet bar | 186 |

**Total LOC:** 1,764

## Usage

Each script exposes a `create_<name>(dark_mode=False)` function that returns a Plotly `Figure` object:

```python
from images.interactive.third.oncology_convergence import create_oncology_convergence

fig = create_oncology_convergence(dark_mode=True)
fig.show()
```

Running a script directly generates both HTML and PNG outputs:

```bash
python images/interactive/3rd/01_oncology_convergence.py
# Output: 01_oncology_convergence.html, 01_oncology_convergence.png (1920x1080 @2x)
```

## Dependencies

- `plotly` (graph_objects, subplots)
- `numpy`
- `kaleido` (for PNG export, optional)
- Python standard library (`pathlib`)

## Code Quality

All scripts pass:

```bash
ruff check images/interactive/3rd/ --config ruff.toml
ruff format --check images/interactive/3rd/
```
