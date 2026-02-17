# Tools — CLI Utilities for PAI Oncology Trial FL

> **VERSION:** 0.4.0 | **LICENSE:** MIT
>
> **DISCLAIMER: RESEARCH USE ONLY** — All tools in this directory are intended
> for research and educational purposes. They are NOT approved for clinical
> diagnostic use, treatment planning, or regulatory submission.

## Directory Structure

```
tools/
├── README.md                              # This file
├── dicom-inspector/
│   └── dicom_inspector.py                 # DICOM medical image inspection
├── dose-calculator/
│   └── dose_calculator.py                 # Radiation & chemotherapy dose calculator
├── trial-site-monitor/
│   └── trial_site_monitor.py              # Multi-site clinical trial monitoring
├── sim-job-runner/
│   └── sim_job_runner.py                  # Cross-framework simulation job launcher
└── deployment-readiness/
    └── deployment_readiness.py            # Deployment readiness checker
```

---

## Tool Summaries

### 1. DICOM Inspector (`dicom-inspector/dicom_inspector.py`)

Inspects, validates, and summarizes DICOM medical image files used in
federated learning pipelines for radiation oncology. Checks required tags,
validates pixel spacing and slice thickness against ACR-AAPM standards, and
generates aggregate summaries across file collections.

**Subcommands:**

| Command     | Description                                      |
|-------------|--------------------------------------------------|
| `inspect`   | Inspect a single DICOM file and extract metadata  |
| `validate`  | Validate a DICOM file against oncology QA bounds   |
| `summarize` | Summarize all DICOM files in a directory           |

**CLI Usage:**

```bash
# Inspect a single DICOM file
python tools/dicom-inspector/dicom_inspector.py inspect /path/to/scan.dcm

# Validate with JSON output
python tools/dicom-inspector/dicom_inspector.py --json validate /path/to/scan.dcm

# Summarize a directory of DICOM files
python tools/dicom-inspector/dicom_inspector.py summarize /path/to/dicom_dir/

# Show help
python tools/dicom-inspector/dicom_inspector.py --help
```

**Optional dependency:** `pydicom` (install with `pip install pydicom`).

---

### 2. Dose Calculator (`dose-calculator/dose_calculator.py`)

Computes and validates radiation therapy and chemotherapy dose parameters.
Implements the linear-quadratic model (BED/EQD2), organ-at-risk constraints
from QUANTEC, and BSA-based chemotherapy dosing with Calvert formula support
for carboplatin. All doses are bounded to prevent erroneous calculations.

**Subcommands:**

| Command        | Description                                     |
|----------------|--------------------------------------------------|
| `radiation`    | Calculate a radiation dose prescription           |
| `chemotherapy` | Calculate a chemotherapy dose (mg/m² or AUC)      |
| `validate`     | Validate an organ-at-risk dose against QUANTEC    |

**CLI Usage:**

```bash
# Calculate a radiation dose for lung IMRT
python tools/dose-calculator/dose_calculator.py radiation \
    --site lung --technique IMRT --dose-per-fraction 2.0 --fractions 33

# Calculate carboplatin dose using Calvert formula
python tools/dose-calculator/dose_calculator.py chemotherapy \
    --agent carboplatin --auc 6 --crcl 95 --bsa 1.8

# Calculate paclitaxel dose
python tools/dose-calculator/dose_calculator.py chemotherapy \
    --agent paclitaxel --dose 175 --bsa 1.9

# Validate spinal cord dose
python tools/dose-calculator/dose_calculator.py validate \
    --organ spinal_cord --dose 45.0

# JSON output
python tools/dose-calculator/dose_calculator.py --json radiation \
    --site prostate --dose-per-fraction 1.8 --fractions 44
```

---

### 3. Trial Site Monitor (`trial-site-monitor/trial_site_monitor.py`)

Monitors enrollment rates, data quality, protocol compliance, and federated
learning progress across geographically distributed clinical trial sites.
Uses colour-coded GREEN/YELLOW/RED status levels aligned with ICH E6(R2)
risk-based monitoring recommendations. Includes built-in demo data for
immediate CLI exploration.

**Subcommands:**

| Command   | Description                                        |
|-----------|----------------------------------------------------|
| `status`  | Show status for a specific site                     |
| `report`  | Generate a cross-site monitoring report              |
| `alerts`  | List active alerts across all sites                  |

**CLI Usage:**

```bash
# Check status of a specific site
python tools/trial-site-monitor/trial_site_monitor.py status --site-id SITE-001

# Generate a full monitoring report
python tools/trial-site-monitor/trial_site_monitor.py report \
    --trial-id NCT12345678 --target 150

# List critical alerts only
python tools/trial-site-monitor/trial_site_monitor.py alerts --min-level critical

# JSON output for programmatic consumption
python tools/trial-site-monitor/trial_site_monitor.py --json report
```

---

### 4. Simulation Job Runner (`sim-job-runner/sim_job_runner.py`)

Provides a unified interface for launching simulation jobs across multiple
robotics and physics frameworks: NVIDIA Isaac Lab, MuJoCo, PyBullet, and
Gazebo. Supports oncology-specific tasks such as needle insertion, tumour
resection, catheter navigation, and radiation patient positioning. Framework
availability is detected at import time without requiring GPU hardware.

**Subcommands:**

| Command   | Description                                        |
|-----------|----------------------------------------------------|
| `launch`  | Launch a new simulation job                         |
| `status`  | Query the status of a running job                   |
| `results` | Retrieve results from a completed job               |

**CLI Usage:**

```bash
# Launch a MuJoCo needle insertion simulation
python tools/sim-job-runner/sim_job_runner.py launch \
    --framework mujoco --task needle_insertion --episodes 100

# Launch an Isaac Lab tumour resection with parallel envs
python tools/sim-job-runner/sim_job_runner.py launch \
    --framework isaac --task tumour_resection --num-envs 64

# Check job status
python tools/sim-job-runner/sim_job_runner.py status sim-abc123def456

# Retrieve completed results as JSON
python tools/sim-job-runner/sim_job_runner.py --json results sim-abc123def456
```

**Optional dependencies:** `mujoco`, `pybullet`, `isaacgym`, `gazebo`
(framework-specific; none required for configuration and validation).

---

### 5. Deployment Readiness (`deployment-readiness/deployment_readiness.py`)

Performs comprehensive checklist-based validation before promoting a trained
model or simulation policy to production. Evaluates five categories:
model validation, safety compliance, regulatory, infrastructure, and
documentation. Each check cites its authoritative source (FDA, IEC 62304,
ISO 14971, NIST, etc.).

**Subcommands:**

| Command   | Description                                        |
|-----------|----------------------------------------------------|
| `check`   | Run readiness checks for a specific category        |
| `report`  | Generate a full deployment readiness report          |
| `export`  | Generate and export a report to a JSON file          |

**CLI Usage:**

```bash
# Run all checks
python tools/deployment-readiness/deployment_readiness.py check

# Run only safety compliance checks
python tools/deployment-readiness/deployment_readiness.py check --category safety_compliance

# Generate a readiness report
python tools/deployment-readiness/deployment_readiness.py report \
    --model-name oncology-fl-model --model-version 0.4.0

# Export report to JSON file
python tools/deployment-readiness/deployment_readiness.py export \
    --model-name oncology-fl-model --model-version 0.4.0 \
    --output readiness_report.json

# JSON output to stdout
python tools/deployment-readiness/deployment_readiness.py --json report
```

---

## Common Workflows

### Pre-Training Data Validation

```bash
# 1. Inspect DICOM files for completeness
python tools/dicom-inspector/dicom_inspector.py --json summarize /data/trial/

# 2. Check trial site enrollment and data quality
python tools/trial-site-monitor/trial_site_monitor.py --json report \
    --trial-id NCT12345678
```

### Simulation Training Pipeline

```bash
# 1. Launch a simulation training job
python tools/sim-job-runner/sim_job_runner.py launch \
    --framework mujoco --task needle_insertion --episodes 1000 --seed 42

# 2. Monitor job progress
python tools/sim-job-runner/sim_job_runner.py status <job-id>

# 3. Retrieve results
python tools/sim-job-runner/sim_job_runner.py --json results <job-id>
```

### Pre-Deployment Validation

```bash
# 1. Validate dose calculations
python tools/dose-calculator/dose_calculator.py --json radiation \
    --site lung --technique IMRT --dose-per-fraction 2.0 --fractions 33

# 2. Run deployment readiness checks
python tools/deployment-readiness/deployment_readiness.py report \
    --model-name oncology-fl-model --model-version 0.4.0

# 3. Export the readiness report for regulatory review
python tools/deployment-readiness/deployment_readiness.py export \
    --output pre_deployment_report.json
```

---

## Development

### Prerequisites

- Python >= 3.10
- No GPU or specialised hardware required for importing or running CLI help
- Optional: `pydicom` for DICOM inspection, `mujoco`/`pybullet` for simulation

### Linting and Formatting

All tools are checked with [ruff](https://docs.astral.sh/ruff/) using the
project's `ruff.toml` configuration:

```bash
# Check linting
ruff check tools/

# Check formatting
ruff format --check tools/

# Auto-fix formatting
ruff format tools/
```

### Running a Tool

Each tool is a standalone script with `if __name__ == "__main__"` entry
points:

```bash
# Any tool can be run directly
python tools/<tool-dir>/<tool>.py --help

# Or with the JSON flag for machine-readable output
python tools/<tool-dir>/<tool>.py --json <subcommand> [options]
```

### Adding a New Tool

1. Create a new directory under `tools/` with a descriptive name.
2. Add a Python script following the established patterns:
   - Module docstring with DISCLAIMER, VERSION, and LICENSE.
   - `from __future__ import annotations` and standard imports.
   - `logging.basicConfig` + `logger = logging.getLogger(__name__)`.
   - Enums and dataclasses with docstrings.
   - `argparse` with subcommands.
   - `if __name__ == "__main__": main()` with `sys.exit`.
3. Ensure the script passes `ruff check` and `ruff format --check`.
4. Update this README with the new tool's summary and usage examples.
