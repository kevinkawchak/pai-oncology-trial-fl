# Set 3 — Regulatory, Privacy, and Readiness Visualizations

## Overview

10 visualizations covering regulatory compliance, privacy framework analysis, FDA classification, HIPAA PHI detection, and deployment readiness assessments.

## Scripts

### 01_oncology_convergence.py
- **Chart type:** Multi-line chart
- **Data source:** Federated convergence across clinical analytics subtasks
- **X-axis:** Communication rounds (0-200)
- **Y-axis:** Validation metric (AUC-ROC)
- **Lines:** Tumor Classification, Response Prediction, Survival Estimation, Toxicity Grading, Biomarker Detection

### 02_multi_site_trial_dashboard.py
- **Chart type:** Multi-subplot dashboard (2x2)
- **Data source:** Site enrollment data from clinical-analytics/ and tools/trial-site-monitor/
- **Panels:** Enrollment waterfall (10 sites), Screening funnel, Monthly trend, Site performance bubbles

### 03_algorithm_comparison_radar.py
- **Chart type:** Radar chart
- **Data source:** AggregationStrategy enum from federated/coordinator.py
- **Algorithms:** FedAvg, FedProx, SCAFFOLD
- **Axes:** Communication Efficiency, Convergence Speed, Non-IID Robustness, Privacy, Computational Cost, Model Quality

### 04_fda_device_classification_tree.py
- **Chart type:** Treemap
- **Data source:** DeviceClass, SubmissionType enums from regulatory/ and regulatory-submissions/
- **Hierarchy:** Device Class → Submission Type → AI/ML Category
- **Values:** Device counts per pathway

### 05_oncology_device_distribution.py
- **Chart type:** Sunburst / donut
- **Data source:** FDA AI/ML device landscape from regulatory/ domain knowledge
- **Rings:** Category (Radiology, Pathology, Treatment Planning, Monitoring) → Status → Class

### 06_regulatory_compliance_scorecard.py
- **Chart type:** Horizontal grouped bar chart
- **Data source:** Regulation enum from regulatory-submissions/compliance_validator.py
- **Rows:** FDA 21 CFR 820, Part 11, HIPAA, GDPR, ISO 14971, IEC 62304, ICH E6(R3), ICH E9(R1), EU MDR
- **Colors:** Compliant (green), Minor Finding (yellow), Major Finding (red)

### 07_hipaa_phi_detection_matrix.py
- **Chart type:** Annotated heatmap
- **Data source:** PHICategory enum from privacy/phi-pii-management/phi_detector.py (18 Safe Harbor IDs)
- **Rows:** 18 HIPAA Safe Harbor identifiers
- **Columns:** Data sources (EHR, DICOM, Lab Reports, Clinical Notes, Consent Forms)
- **Values:** Detection confidence (0.0–1.0)

### 08_privacy_analytics_pipeline.py
- **Chart type:** Sankey diagram
- **Data source:** DeidentificationMethod enum from privacy/de-identification/
- **Flow:** Raw Data → PHI Detection → [REDACT, HASH, GENERALIZE, SUPPRESS, PSEUDONYMIZE, DATE_SHIFT] → Validated → Audit
- **Labels:** Record counts at each node

### 09_deployment_readiness_radar.py
- **Chart type:** Radar chart
- **Data source:** CheckStatus enum from tools/deployment-readiness/ and platform metrics
- **Axes:** Code Quality, Test Coverage, Security Audit, Regulatory, Documentation, CI/CD, Dependencies, Performance, Privacy, Clinical Validation

### 10_production_readiness_scores.py
- **Chart type:** Horizontal bar chart with target lines
- **Data source:** Platform module readiness assessment across all directories
- **Modules:** Federated, Physical AI, Privacy, Regulatory, Digital Twins, Clinical Analytics, Regulatory Submissions, Unification, Tools
- **Values:** Readiness score (0-100) with 80% target line
