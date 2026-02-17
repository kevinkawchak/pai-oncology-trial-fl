# Q1 2026 Implementation Timeline

Implementation timeline for the PAI Oncology Trial FL standards initiative covering
model format conversion, federated model registry, and cross-platform benchmarking.

## Overview

- **Start Date:** January 6, 2026
- **End Date:** March 31, 2026
- **Duration:** 12 weeks (3 sprints of 4 weeks each)
- **Stakeholders:** Site engineering leads, clinical informatics teams, regulatory affairs, QA

---

## Sprint 1 -- Foundation (January 6 -- January 30, 2026)

### Week 1 (Jan 6 -- Jan 9)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Finalize format conversion requirements across all sites | Engineering Lead | Obj 1 | Pending |
| Distribute model registry YAML schema draft for review | Standards Committee | Obj 2 | Pending |
| Define benchmark metric definitions and acceptance criteria | QA Lead | Obj 3 | Pending |
| Set up CI pipeline for standards validation (ruff, pytest) | DevOps | All | Pending |

### Week 2 (Jan 12 -- Jan 16)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Implement ONNX and PyTorch conversion paths with round-trip validation | ML Engineer | Obj 1 | Pending |
| Publish model_registry.yaml v1.0 schema | Standards Committee | Obj 2 | Pending |
| Implement latency and memory benchmark collectors | Platform Engineer | Obj 3 | Pending |

### Week 3 (Jan 19 -- Jan 23)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Implement TensorFlow and SafeTensors conversion paths | ML Engineer | Obj 1 | Pending |
| Build ModelValidator core (metadata checks, integrity hashes) | Platform Engineer | Obj 2 | Pending |
| Implement accuracy benchmark with clinical ground truth comparison | ML Engineer | Obj 3 | Pending |

### Week 4 (Jan 26 -- Jan 30)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Implement MONAI bundle conversion path | ML Engineer | Obj 1 | Pending |
| Add performance threshold validation to ModelValidator | Platform Engineer | Obj 2 | Pending |
| Build benchmark report generator (JSON + Markdown) | Platform Engineer | Obj 3 | Pending |
| **Sprint 1 Review:** Demo all three objective prototypes | All | All | Pending |

---

## Sprint 2 -- Integration (February 2 -- February 27, 2026)

### Week 5 (Feb 2 -- Feb 6)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Add batch conversion support and parallel execution | ML Engineer | Obj 1 | Pending |
| Integrate model registry with federated coordinator | Platform Engineer | Obj 2 | Pending |
| Run baseline benchmarks across 3 reference hardware configs | QA Lead | Obj 3 | Pending |

### Week 6 (Feb 9 -- Feb 13)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Integrate conversion pipeline with differential privacy layer | ML Engineer | Obj 1 | Pending |
| Add regulatory compliance checks to validation pipeline | Regulatory Eng | Obj 2 | Pending |
| Implement throughput benchmarks with concurrent load simulation | Platform Engineer | Obj 3 | Pending |

### Week 7 (Feb 16 -- Feb 20)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Cross-site conversion testing (2 pilot sites) | Site Engineers | Obj 1 | Pending |
| Cross-site registry sync testing (2 pilot sites) | Site Engineers | Obj 2 | Pending |
| Cross-site benchmark comparison (2 pilot sites) | QA Lead | Obj 3 | Pending |

### Week 8 (Feb 23 -- Feb 27)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Address pilot site feedback for conversion pipeline | ML Engineer | Obj 1 | Pending |
| Address pilot site feedback for model registry | Platform Engineer | Obj 2 | Pending |
| Address pilot site feedback for benchmarking | QA Lead | Obj 3 | Pending |
| **Sprint 2 Review:** Pilot site integration report | All | All | Pending |

---

## Sprint 3 -- Hardening and Rollout (March 2 -- March 31, 2026)

### Week 9 (Mar 2 -- Mar 6)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Numerical equivalence test suite (tolerance < 1e-6) for all format pairs | QA Lead | Obj 1 | Pending |
| Model promotion workflow (dev -> staging -> production) | Platform Engineer | Obj 2 | Pending |
| Benchmark regression test suite with golden results | QA Lead | Obj 3 | Pending |

### Week 10 (Mar 9 -- Mar 13)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Performance optimization: conversion pipeline under 30s for 500M param models | ML Engineer | Obj 1 | Pending |
| Audit trail integration with existing audit_logger.py | Platform Engineer | Obj 2 | Pending |
| Benchmark report template approved by regulatory affairs | Regulatory Eng | Obj 3 | Pending |

### Week 11 (Mar 16 -- Mar 20)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Full rollout to all enrolled sites (conversion pipeline) | DevOps | Obj 1 | Pending |
| Full rollout to all enrolled sites (model registry) | DevOps | Obj 2 | Pending |
| Full rollout to all enrolled sites (benchmarking) | DevOps | Obj 3 | Pending |

### Week 12 (Mar 23 -- Mar 31)

| Milestone | Owner | Objective | Status |
|-----------|-------|-----------|--------|
| Documentation finalization and knowledge transfer sessions | All | All | Pending |
| Standards compliance audit across all sites | QA Lead | All | Pending |
| Q1 retrospective and Q2 planning inputs | Engineering Lead | All | Pending |
| **Sprint 3 Review and Q1 Standards Close-Out** | All | All | Pending |

---

## Key Dependencies

| Dependency | Required By | Risk | Mitigation |
|------------|-------------|------|------------|
| PyTorch >= 2.1.0 at all sites | Obj 1, Week 2 | Medium -- some sites on older versions | Provide upgrade runbook |
| ONNX Runtime >= 1.16.0 at all sites | Obj 1, Week 2 | Low | Include in requirements.txt |
| Network connectivity for registry sync | Obj 2, Week 7 | Medium -- air-gapped sites | Support offline registry export/import |
| Reference hardware for benchmarks (CPU, GPU, edge) | Obj 3, Week 5 | High -- hardware procurement lead time | Use cloud instances as fallback |
| Regulatory affairs review bandwidth | Obj 2 & 3, Week 10 | Medium | Schedule reviews early in Sprint 2 |

## Success Criteria

1. **Objective 1:** All 10 format conversion pairs (5 formats, bidirectional) pass round-trip numerical equivalence tests with tolerance < 1e-6 for model weights.
2. **Objective 2:** Model registry schema adopted by 100% of enrolled sites; all models in production have complete metadata and validation status.
3. **Objective 3:** Benchmark suite runs reproducibly on at least 3 hardware configurations; cross-site variance in reported latency is within 5% for identical models and hardware.
