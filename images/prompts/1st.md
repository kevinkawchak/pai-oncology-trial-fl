# Set 1 — Repository Architecture and Simulation Visualizations

## Overview

10 visualizations covering repository structure, clinical trial workflows, simulation framework comparisons, and oncology-specific data pipelines.

## Scripts

### 01_repository_architecture_treemap.py
- **Chart type:** Treemap
- **Data source:** Repository module directory counts (federated: 9, physical_ai: 6, privacy: 10, regulatory: 6, digital-twins: 3, clinical-analytics: 7, regulatory-submissions: 6, unification: 5, tools: 5, examples: 17, tests: 97)
- **Color scheme:** Sequential blue palette by module size
- **Labels:** Module name, file count, percentage of total

### 02_clinical_trial_workflow.py
- **Chart type:** Sankey diagram
- **Data source:** Clinical trial workflow phases from regulatory/ and clinical-analytics/ modules
- **Flow:** Enrollment → Randomization → Treatment → Monitoring → Analysis → Reporting → Submission
- **Labels:** Phase name, record/patient count at each stage

### 03_oncology_process_diagram.py
- **Chart type:** Sunburst
- **Data source:** TreatmentModality, DrugClass, ResponseCategory Enums from digital-twins/treatment-simulation/
- **Hierarchy:** Inner ring = modalities, outer ring = drug classes and response categories
- **Labels:** Enum values with descriptive names

### 04_framework_comparison_radar.py
- **Chart type:** Radar / Scatterpolar
- **Data source:** Framework capabilities from frameworks/ INTEGRATION.md files and physical_ai/ SimulationFramework enum
- **Axes:** Physics Fidelity, Performance, GPU Support, ROS Integration, Rendering, Documentation, Ease of Use, Medical Robotics
- **Lines:** Isaac Sim, MuJoCo, Gazebo, PyBullet

### 05_parameter_mapping_heatmap.py
- **Chart type:** Annotated heatmap
- **Data source:** ParameterMapper class from unification/simulation_physics/isaac_mujoco_bridge.py
- **Rows:** Isaac Sim (PhysX) parameters
- **Columns:** MuJoCo parameters
- **Values:** Mapping confidence/compatibility scores

### 06_data_pipeline_throughput.py
- **Chart type:** Grouped bar chart
- **Data source:** Federated data pipeline stages from federated/ and clinical-analytics/ modules
- **X-axis:** Pipeline stages (Ingestion, Harmonization, Validation, Privacy, Aggregation, Update)
- **Y-axis:** Throughput (records/sec)
- **Groups:** Small site (100 patients), Medium (1000), Large (10000)

### 07_model_format_comparison.py
- **Chart type:** Grouped bar chart
- **Data source:** ModelFormat enum and ConversionPipeline from q1-2026-standards/
- **X-axis:** Metrics (File Size, Load Time, Inference Time, Memory, Precision)
- **Groups:** PyTorch, ONNX, SafeTensors, TensorFlow, MONAI

### 08_state_vector_visualization.py
- **Chart type:** 3D scatter plot
- **Data source:** SimulationState, JointState from unification/simulation_physics/isaac_mujoco_bridge.py
- **Axes:** Joint position (rad), Joint velocity (rad/s), Joint torque (N·m)
- **Color:** Synchronized (green) vs Drifted (red) states

### 09_domain_randomization_transfer.py
- **Chart type:** Multi-line chart
- **Data source:** Domain randomization parameters from configs/training_config.yaml concepts
- **X-axis:** Training epochs (0-500)
- **Y-axis:** Task success rate (%)
- **Lines:** No randomization, Low, Medium, High randomization levels

### 10_oncology_trajectories.py
- **Chart type:** Multi-panel line chart
- **Data source:** GrowthModel enum and PatientPhysiologyEngine from digital-twins/patient-modeling/
- **X-axis:** Time (days, 0-365)
- **Y-axis:** Tumor volume (cm³)
- **Lines:** Exponential, Logistic, Gompertz, Von Bertalanffy growth with treatment intervention markers
