# Set 2 — Training, Benchmarks, and Performance Visualizations

## Overview

10 visualizations covering federated training metrics, cross-framework benchmarks, safety monitoring, and clinical trial performance indicators.

## Scripts

### 01_training_curves.py
- **Chart type:** Multi-line chart
- **Data source:** FedAvg/FedProx/SCAFFOLD from federated/coordinator.py AggregationStrategy enum
- **X-axis:** Communication rounds (0-100)
- **Y-axis:** Training loss
- **Lines:** FedAvg, FedProx (mu=0.01), SCAFFOLD convergence curves

### 02_benchmark_comparisons.py
- **Chart type:** Grouped bar chart
- **Data source:** BenchmarkMetric enum from q1-2026-standards/objective-3-benchmarking/
- **X-axis:** Frameworks (Isaac Sim, MuJoCo, Gazebo, PyBullet)
- **Metrics:** Latency (ms), Step Rate (Hz), Memory (MB), Accuracy (%)

### 03_safety_metrics_dashboard.py
- **Chart type:** Multi-subplot (2x2 grid)
- **Data source:** Safety constraints from physical_ai/ and configs/training_config.yaml
- **Panels:** Force monitoring time series, Workspace violations by joint, E-stop frequency, ISO 13482 compliance scores

### 04_performance_radars.py
- **Chart type:** Overlapping radar charts
- **Data source:** Module capabilities across platform components
- **Modules:** Federated Learning, Digital Twins, Privacy, Regulatory, Clinical Analytics, Surgical Robotics
- **Axes:** Throughput, Accuracy, Coverage, Compliance, Scalability

### 05_accuracy_confusion_matrix.py
- **Chart type:** Annotated heatmap (confusion matrix)
- **Data source:** TumorType enum from digital-twins/patient-modeling/patient_model.py (12 types)
- **Axes:** Predicted vs Actual tumor type
- **Values:** Classification counts with accuracy annotations

### 06_sim_to_real_transfer.py
- **Chart type:** Dual-axis line chart
- **Data source:** Transfer learning metrics from physical_ai/ domain concepts
- **Left Y:** Task success rate (%)
- **Right Y:** Position error (mm)
- **Lines:** Direct transfer, Fine-tuned, Domain-adapted

### 07_latency_distributions.py
- **Chart type:** Box plot / violin
- **Data source:** SyncReport timing from unification/simulation_physics/isaac_mujoco_bridge.py
- **Categories:** Isaac→MuJoCo, MuJoCo→Isaac, Bidirectional
- **Metrics:** Sync duration (ms) distribution

### 08_resource_utilization.py
- **Chart type:** Stacked area chart
- **Data source:** Federated training resource allocation concepts from federated/ modules
- **X-axis:** Training time (minutes)
- **Layers:** Model Training, Secure Aggregation, Privacy Filtering, Communication, Idle

### 09_oncology_kpi_tracker.py
- **Chart type:** Indicator / bullet gauge chart
- **Data source:** Clinical trial KPIs from clinical-analytics/ and regulatory/ modules
- **Metrics:** Enrollment Rate, Protocol Compliance, Data Quality, Site Activation, SAE Reporting, Query Resolution

### 10_cross_framework_consistency.py
- **Chart type:** Scatter plot with reference line
- **Data source:** Cross-engine state comparison from IsaacMuJoCoBridge calibration
- **X-axis:** Isaac Sim joint positions (rad)
- **Y-axis:** MuJoCo joint positions (rad)
- **Reference:** Perfect agreement diagonal
