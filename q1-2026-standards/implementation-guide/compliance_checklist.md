# Standards Adoption Compliance Checklist

Compliance checklist for sites adopting the Q1 2026 PAI Oncology Trial FL standards.
Each enrolled site must complete all applicable items before participating in federated
model exchange under the new standards.

---

## How to Use This Checklist

1. Assign an owner for each section at your site.
2. Complete items in order within each section (some items depend on earlier ones).
3. Mark each item with its current status: `[ ]` Not Started, `[~]` In Progress, `[x]` Complete.
4. Submit the completed checklist to the standards committee for review before your site activation date.

---

## 1. Environment and Infrastructure

- [ ] **1.1** Python >= 3.10 installed on all model training and inference hosts.
- [ ] **1.2** Core dependencies installed: `numpy >= 1.24.0`, `scikit-learn >= 1.3.0`, `pandas >= 2.0.0`, `pyyaml >= 6.0`.
- [ ] **1.3** At least one deep learning framework installed: PyTorch >= 2.1.0, TensorFlow >= 2.14.0, or MONAI >= 1.3.0.
- [ ] **1.4** ONNX Runtime >= 1.16.0 installed for format conversion validation.
- [ ] **1.5** SafeTensors >= 0.4.0 installed for secure model serialization.
- [ ] **1.6** `ruff >= 0.4.0` installed for code quality checks on any local extensions.
- [ ] **1.7** Git configured with access to the `pai-oncology-trial-fl` repository.
- [ ] **1.8** CI/CD pipeline configured to run `ruff check` and `pytest` on standards modules.

## 2. Model Format Conversion (Objective 1)

- [ ] **2.1** `conversion_pipeline.py` imported successfully in site environment.
- [ ] **2.2** Round-trip conversion test passed for at least one format pair relevant to site workflows.
- [ ] **2.3** Numerical equivalence tolerance configured (default: `atol=1e-6`, `rtol=1e-5`).
- [ ] **2.4** SHA-256 checksums enabled for all model artifacts produced by the conversion pipeline.
- [ ] **2.5** Conversion audit log integrated with site's existing audit infrastructure.
- [ ] **2.6** Error handling verified: pipeline raises `ConversionError` with actionable messages for unsupported format pairs.
- [ ] **2.7** Performance validated: conversion of site's largest model completes within 60 seconds.

## 3. Federated Model Registry (Objective 2)

- [ ] **3.1** `model_registry.yaml` schema reviewed and accepted by site data governance team.
- [ ] **3.2** All existing models in site inventory cataloged with required registry fields:
  - [ ] `model_id` (unique identifier)
  - [ ] `framework` (training framework)
  - [ ] `version` (semantic version)
  - [ ] `metrics` (performance metrics from last evaluation)
  - [ ] `validation_status` (PENDING, PASSED, FAILED, or REQUIRES_REVIEW)
  - [ ] `clinical_domain` (oncology sub-domain)
  - [ ] `training_config` (hyperparameters and data configuration)
- [ ] **3.3** `model_validator.py` imported successfully in site environment.
- [ ] **3.4** Validation pipeline integrated into model promotion workflow (dev -> staging -> production).
- [ ] **3.5** Minimum performance thresholds configured for site's clinical domain:
  - [ ] Accuracy threshold (e.g., >= 0.85 for tumor response prediction)
  - [ ] AUC-ROC threshold (e.g., >= 0.80)
  - [ ] Maximum acceptable false negative rate
- [ ] **3.6** Model metadata completeness check passes for all registry entries.
- [ ] **3.7** Regulatory compliance flags reviewed: FDA submission readiness, HIPAA audit trail linkage.

## 4. Cross-Platform Benchmarking (Objective 3)

- [ ] **4.1** `benchmark_runner.py` imported successfully in site environment.
- [ ] **4.2** Hardware inventory documented for benchmark hosts:
  - [ ] CPU model, core count, clock speed
  - [ ] GPU model and VRAM (if applicable)
  - [ ] RAM capacity
  - [ ] Storage type (SSD/NVMe/HDD)
- [ ] **4.3** Baseline benchmarks executed on site hardware:
  - [ ] Inference latency (p50, p95, p99)
  - [ ] Prediction accuracy against clinical ground truth dataset
  - [ ] Peak memory usage during inference
  - [ ] Throughput under concurrent load (requests per second)
- [ ] **4.4** Benchmark results submitted in standardized JSON format.
- [ ] **4.5** Benchmark results reproducible: re-running produces results within 5% variance.
- [ ] **4.6** Benchmark report generated and reviewed by site QA lead.

## 5. Security and Privacy

- [ ] **5.1** Model artifacts encrypted at rest using site-approved encryption (AES-256 or equivalent).
- [ ] **5.2** Model artifacts encrypted in transit (TLS 1.2+ for all inter-site transfers).
- [ ] **5.3** Access control configured: only authorized roles can register, validate, or promote models.
- [ ] **5.4** Audit logging enabled for all registry operations (create, update, validate, promote).
- [ ] **5.5** PHI detection scan passed on all model metadata fields (no patient identifiers in registry).
- [ ] **5.6** Differential privacy budget tracked for models trained with DP mechanisms.

## 6. Regulatory Compliance

- [ ] **6.1** Site IRB approval covers federated model exchange under the new standards.
- [ ] **6.2** Data Use Agreement (DUA) updated to reference Q1 2026 standards for model artifacts.
- [ ] **6.3** FDA AI/ML device classification reviewed for models entering production.
- [ ] **6.4** 21 CFR Part 11 compliance verified for electronic records generated by registry and benchmarks.
- [ ] **6.5** HIPAA Security Rule compliance verified for model artifact storage and transmission.
- [ ] **6.6** GDPR compliance verified (if applicable) for model metadata containing EU site information.
- [ ] **6.7** Consent management system confirms patient consent covers model training and cross-site exchange.

## 7. Documentation and Training

- [ ] **7.1** Site engineering team has reviewed the [Implementation Timeline](timeline.md).
- [ ] **7.2** Site engineering team has completed onboarding for all three standards modules.
- [ ] **7.3** Runbook created for common operations:
  - [ ] Converting a model between formats
  - [ ] Registering a new model version
  - [ ] Running a benchmark suite
  - [ ] Troubleshooting conversion failures
- [ ] **7.4** Escalation path documented for standards-related issues.
- [ ] **7.5** Site contact designated for standards committee communications.

## 8. Go-Live Readiness

- [ ] **8.1** All items in sections 1--7 marked complete or granted a documented exception.
- [ ] **8.2** Standards committee review completed and approval received.
- [ ] **8.3** Go-live date confirmed and communicated to all site stakeholders.
- [ ] **8.4** Rollback plan documented in case of critical issues post-activation.
- [ ] **8.5** Post-go-live monitoring plan in place (first 2 weeks).

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Site Engineering Lead | | | |
| Clinical Informatics Lead | | | |
| Regulatory Affairs | | | |
| QA Lead | | | |
| Standards Committee Representative | | | |
