# Set 1 - Repository Architecture & Simulation Visualizations

Interactive Plotly visualizations covering repository structure, clinical trial workflows, simulation framework comparisons, and oncology-specific data pipelines.

## Scripts

| # | Script | Description | Chart Type | LOC |
|---|--------|-------------|-----------|-----|
| 01 | `01_repository_architecture_treemap.py` | Repository module structure with file counts per package | Treemap | 209 |
| 02 | `02_clinical_trial_workflow.py` | Patient flow through Phase III oncology clinical trial phases | Sankey | 170 |
| 03 | `03_oncology_process_diagram.py` | Treatment modalities, drug classes, and RECIST response categories | Sunburst | 161 |
| 04 | `04_framework_comparison_radar.py` | Isaac Sim / MuJoCo / Gazebo / PyBullet across 8 capability axes | Scatterpolar | 135 |
| 05 | `05_parameter_mapping_heatmap.py` | PhysX-to-MuJoCo parameter mapping strength in the bridge layer | Heatmap | 156 |
| 06 | `06_data_pipeline_throughput.py` | Federated pipeline throughput by stage and data type | Bar (grouped) | 178 |
| 07 | `07_model_format_comparison.py` | PyTorch / ONNX / SafeTensors / TF / MONAI format metrics | Bar (grouped) | 163 |
| 08 | `08_state_vector_visualization.py` | Simulation state vectors: synchronized vs drifted vs corrected | Scatter3d | 179 |
| 09 | `09_domain_randomization_transfer.py` | Sim-to-real transfer performance by domain randomization level | Scatter (lines) | 219 |
| 10 | `10_oncology_trajectories.py` | Tumor growth trajectories (exponential, logistic, Gompertz) under treatment | Multi-panel Line | 215 |

**Total LOC:** 1,785

## Usage

Each script can be run independently:

```bash
python images/interactive/1st/01_repository_architecture_treemap.py
```

Each script exposes a `create_<name>(dark_mode=False)` function that returns a Plotly `Figure` object:

```python
from images.interactive.first.01_repository_architecture_treemap import create_repository_architecture_treemap

fig = create_repository_architecture_treemap(dark_mode=True)
fig.show()
```

## Output

Each script generates:
- An interactive HTML file (CDN-hosted Plotly.js)
- A high-resolution PNG at 1920x1080 with 2x scale (3840x2160 effective)

## Requirements

- `plotly` (graph_objects only, no plotly.express)
- `numpy`
- `kaleido` (for PNG export)

## Conventions

- All scripts use `plotly.graph_objects` exclusively
- Light mode uses the `plotly_white` template; dark mode uses `plotly_dark`
- All scripts pass `ruff check` (line-length=120) and `ruff format --check`
- Double quotes used consistently throughout
