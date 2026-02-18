# Visualization Strategy — PAI Oncology Trial FL

## Overview

Generate 30 publication-quality interactive Plotly charts organized across 3 sets of 10 visualizations each. Every chart supports light mode (`plotly_white`) and dark mode (`plotly_dark`) rendering. Data is sourced directly from repository modules, enums, dataclasses, and domain knowledge embedded in the platform.

## Architecture

```
images/
├── prompts/          # Visualization planning documents
│   ├── plan.md       # This file — overall strategy
│   ├── 1st.md        # Set 1 instructions (repository architecture + simulation)
│   ├── 2nd.md        # Set 2 instructions (training + benchmarks + metrics)
│   └── 3rd.md        # Set 3 instructions (regulatory + privacy + readiness)
├── interactive/
│   ├── 1st/          # Set 1: 10 scripts — architecture, workflows, frameworks
│   │   ├── README.md
│   │   └── 01_..10_ Python scripts
│   ├── 2nd/          # Set 2: 10 scripts — training, benchmarks, performance
│   │   ├── README.md
│   │   └── 01_..10_ Python scripts
│   └── 3rd/          # Set 3: 10 scripts — regulatory, privacy, readiness
│       ├── README.md
│       └── 01_..10_ Python scripts
└── README.md         # Top-level workflow + conversion metrics
```

## Chart Specifications

| Property | Value |
|----------|-------|
| Library | `plotly.graph_objects` (not express) |
| Light template | `plotly_white` |
| Dark template | `plotly_dark` |
| HTML output | Self-contained (include_plotlyjs=True) |
| PNG resolution | 1920x1080 @2x (3840x2160 pixels) |
| LOC per script | 100–250 lines |
| Total scripts | 30 |
| Total HTML | 60 (30 light + 30 dark) |
| Total PNG | 60 (30 light + 30 dark) |

## Data Sources

All data is derived from actual repository content:

| Source Module | Data Extracted | Used In |
|---------------|---------------|---------|
| `federated/` | Aggregation strategies, client counts, convergence metrics | Sets 1, 2 |
| `physical_ai/` | Framework enums, sensor types, robot types | Sets 1, 2 |
| `privacy/` | 18 HIPAA identifiers, de-id methods, access roles | Set 3 |
| `regulatory/` | FDA pathways, compliance frameworks, IRB statuses | Set 3 |
| `digital-twins/` | Tumor types, growth models, drug classes, response categories | Sets 1, 2 |
| `clinical-analytics/` | PK/PD models, risk categories, survival metrics | Sets 2, 3 |
| `regulatory-submissions/` | Submission types, eCTD modules, compliance scores | Set 3 |
| `unification/` | Bridge states, sync directions, coordinate conventions | Sets 1, 2 |
| `q1-2026-standards/` | Model formats, conversion paths, benchmark metrics | Sets 1, 2 |
| `tools/` | DICOM modalities, dose units, site statuses | Sets 2, 3 |
| `tests/` | Test counts per module, coverage metrics | Set 3 |

## Quality Gates

- All 30 scripts pass `ruff check` (line-length 120, py310 target)
- All 30 scripts pass `ruff format --check`
- Every script uses `plotly.graph_objects` (never `plotly.express`)
- Both light and dark modes produce valid output
- Self-standing visualizations with sufficient titles, labels, and annotations
- Total LOC: 4,000–6,000 across 30 scripts

## Color Schemes

### Light Mode
- Primary: `#1f77b4` (blue), `#ff7f0e` (orange), `#2ca02c` (green)
- Secondary: `#d62728` (red), `#9467bd` (purple), `#8c564b` (brown)
- Background: white, gridlines light gray

### Dark Mode
- Primary: `#636efa` (bright blue), `#EF553B` (coral), `#00cc96` (teal)
- Secondary: `#ab63fa` (purple), `#FFA15A` (orange), `#19d3f3` (cyan)
- Background: dark, gridlines dark gray
