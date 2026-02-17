# Q1 2026 Standards -- PAI Oncology Trial FL

Standards initiative for the Physical AI Federated Learning platform for oncology clinical trials.
This directory defines cross-framework interoperability, model governance, and reproducible benchmarking
standards to be adopted across all participating clinical trial sites during Q1 2026
(January -- March 2026).

## Objectives

| # | Objective | Description | Status | Deliverables |
|---|-----------|-------------|--------|--------------|
| 1 | **Model Format Conversion** | Establish a standardized pipeline for cross-framework model interoperability. Federated oncology models trained in one framework (e.g., PyTorch, TensorFlow, MONAI) must be convertible to any other supported format without loss of clinical accuracy. The pipeline validates numerical equivalence across formats using configurable tolerance thresholds. | In Progress | `conversion_pipeline.py` -- Format conversion pipeline supporting ONNX, PyTorch, TensorFlow, SafeTensors, and MONAI bundle formats with round-trip validation and SHA-256 integrity checks. |
| 2 | **Federated Model Registry** | Define a standardized schema and validation workflow for model versioning across federated sites. Every model artifact entering or leaving a site must be registered with provenance metadata, clinical domain tags, and validation status. The registry enables reproducible audits required for FDA AI/ML device submissions. | In Progress | `model_registry.yaml` -- YAML schema for registry entries. `model_validator.py` -- Validation engine that checks model integrity, metadata completeness, performance thresholds, and regulatory compliance before a model version is promoted. |
| 3 | **Cross-Platform Benchmarking** | Create a reproducible benchmarking framework for evaluating federated oncology models across heterogeneous hardware and software environments. Benchmarks cover inference latency, prediction accuracy against clinical ground truth, memory footprint, and throughput under concurrent load. Results are published in a standardized report format for cross-site comparison. | In Progress | `benchmark_runner.py` -- Benchmark execution engine with pluggable metric collectors for latency, accuracy, memory, and throughput. Generates JSON and Markdown reports. |

## Directory Structure

```
q1-2026-standards/
├── README.md                                    # This file
├── objective-1-model-conversion/
│   └── conversion_pipeline.py                   # Cross-framework model conversion pipeline
├── objective-2-model-registry/
│   ├── model_registry.yaml                      # YAML schema for federated model registry
│   └── model_validator.py                       # Model validation engine
├── objective-3-benchmarking/
│   └── benchmark_runner.py                      # Cross-platform benchmark runner
└── implementation-guide/
    ├── timeline.md                              # Q1 2026 implementation timeline
    └── compliance_checklist.md                  # Standards adoption compliance checklist
```

## Scope

These standards apply to all model artifacts exchanged between federated sites in the
PAI Oncology Trial FL platform. They complement -- but do not replace -- the existing
privacy, regulatory, and federated learning modules in the repository root.

### In Scope

- Model serialization and deserialization across PyTorch, TensorFlow, ONNX, SafeTensors, and MONAI.
- Schema-driven model registration with provenance tracking.
- Automated validation gates for model promotion (development, staging, production).
- Hardware-agnostic benchmarking for latency, accuracy, memory, and throughput.
- Compliance mapping to FDA AI/ML guidance and ICH-GCP requirements.

### Out of Scope

- Raw data exchange formats (covered by `federated/data_harmonization.py`).
- Privacy mechanisms (covered by `privacy/` and `federated/differential_privacy.py`).
- Surgical robotics simulation formats (covered by `physical_ai/simulation_bridge.py`).

## Getting Started

All Python modules in this directory use conditional imports with `HAS_<LIBRARY>` boolean
flags, so they can be imported in any environment. Install optional dependencies as needed:

```bash
# Core dependencies (already in requirements.txt)
pip install numpy scikit-learn pyyaml

# Optional framework-specific dependencies
pip install torch torchvision          # PyTorch support
pip install onnx onnxruntime           # ONNX support
pip install tensorflow                 # TensorFlow support
pip install safetensors                # SafeTensors support
pip install monai                      # MONAI support
```

## Related Documentation

- [Implementation Timeline](implementation-guide/timeline.md)
- [Compliance Checklist](implementation-guide/compliance_checklist.md)
- [Project README](../README.md)
- [CHANGELOG](../CHANGELOG.md)
