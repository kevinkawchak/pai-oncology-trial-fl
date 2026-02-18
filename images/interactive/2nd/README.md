# Interactive Visualizations - Set 2

Interactive Plotly visualizations for the PAI Oncology Trial FL platform. Each script generates both an HTML file (with interactive features) and a PNG file (1920x1080 @2x resolution).

## Requirements

- Python 3.10+
- `plotly` (with `kaleido` for PNG export)
- `numpy`

Install dependencies:

```bash
pip install plotly kaleido numpy
```

## Scripts

| # | Script | Description | Chart Type | Lines |
|---|--------|-------------|------------|-------|
| 01 | `01_training_curves.py` | Federated training convergence for FedAvg, FedProx, and SCAFFOLD algorithms | Multi-line chart | 178 |
| 02 | `02_benchmark_comparisons.py` | Cross-framework simulation benchmark across Isaac Sim, MuJoCo, Gazebo, PyBullet | Grouped bar chart | 123 |
| 03 | `03_safety_metrics_dashboard.py` | Surgical robotics safety monitoring with force/torque, violations, e-stops, ISO compliance | Multi-subplot dashboard (4 panels) | 154 |
| 04 | `04_performance_radars.py` | Platform module performance comparison across six core modules | Overlapping radar charts | 140 |
| 05 | `05_accuracy_confusion_matrix.py` | Tumor type classification confusion matrix across 12 oncology categories | Annotated heatmap | 157 |
| 06 | `06_sim_to_real_transfer.py` | Sim-to-real transfer metrics comparing direct, fine-tuned, and domain-adapted approaches | Dual-axis line chart | 179 |
| 07 | `07_latency_distributions.py` | Isaac-MuJoCo bridge synchronization latency distributions by direction | Violin/box plot | 162 |
| 08 | `08_resource_utilization.py` | Computational resource allocation during federated training sessions | Stacked area chart | 180 |
| 09 | `09_oncology_kpi_tracker.py` | Clinical trial KPI dashboard with enrollment, compliance, quality, and reporting metrics | Indicator gauges | 198 |
| 10 | `10_cross_framework_consistency.py` | Cross-engine joint position consistency between Isaac Sim and MuJoCo | Paired scatter plot | 200 |

## Usage

Each script can be run independently:

```bash
python 01_training_curves.py
```

Each script exposes a `create_<name>(dark_mode=False)` function that returns a Plotly `Figure` object:

```python
from images.interactive.second.01_training_curves import create_training_curves

fig = create_training_curves(dark_mode=True)
fig.show()
```

## Dark Mode Support

All visualizations support dark mode via the `dark_mode` parameter:

- `dark_mode=False` (default): Uses `plotly_white` template
- `dark_mode=True`: Uses `plotly_dark` template

## Output

Running each script generates two files in the same directory:

- `<script_name>.html` - Interactive HTML visualization (uses CDN for Plotly.js)
- `<script_name>.png` - Static PNG image at 1920x1080 resolution with 2x scaling

## Code Quality

All scripts pass `ruff check` and `ruff format --check` with `line-length=120`.
