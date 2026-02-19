# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.9] - 2026-02-19

### Added

- **Centralized Utilities Package** (`utils/`)
  - `crypto.py` — Environment-driven HMAC key management via `PAI_HMAC_KEY`, replacing 8 hardcoded key literals across the codebase.
  - `time.py` — Shared `utc_now_iso()` and `utc_now_date()` for consistent ISO 8601 UTC timestamp formatting.
  - `log_sanitizer.py` — PHI-safe log sanitizer stripping SSN, phone, email, MRN, IP, and date patterns from error messages.
  - `config.py` — Centralized `SecurityConfig` and `OperationalConfig` frozen dataclasses with environment variable loading.
  - `error_codes.py` — 18 structured, machine-oriented error codes across 6 operational categories (model, simulation, privacy, data, config, analytics).

- **Property-Based Fuzz Tests** (`tests/test_privacy/test_phi_fuzz.py`)
  - 6 Hypothesis-driven tests for PHI detection: arbitrary unicode input robustness, match offset validity, record scanning, SSN/email detection invariants.

- **SEC-006 Security Check** (`scripts/security_scan.py`)
  - Detects hardcoded secret-like literals (`*_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD` assigned 8+ character strings) with `# noqa: SEC006` inline suppression.

- **Peer Review Fixes Report** (`peer-review/v0.9.9-peer-review-fixes.md`)
  - Comprehensive fix report addressing all 10 v0.9.8 recommendations with per-fix traceability and correction metrics.

- **Release Documentation** (`releases.md/v0.9.9.md`)
  - Added v0.9.9 release notes.

### Changed

- **Hardcoded Key Removal (P1-1)** — All 8 modules now load HMAC keys from `utils.crypto.get_hmac_key()` (env-driven) instead of embedded byte literals.

- **Audit Hash Chaining (P1-2)** — `AuditEvent` now includes `previous_hash` field; integrity hashes chain to predecessor event, enabling tamper-evident ordering detection in `verify_integrity()`.

- **PHI-Safe Logging (P1-3)** — Exception messages in `deidentification_pipeline.py` and `phi_detector.py` are sanitized via `sanitize_log_message()` before logging.

- **Typed Exception Handlers (P2-1)** — Replaced 11 `except Exception` blocks with specific exception tuples across 7 files (model_validator, isaac_mujoco_bridge, model_converter, policy_exporter, deidentification_pipeline, phi_detector, framework_detector).

- **Timestamp Normalization (P2-2)** — `isaac_mujoco_bridge.py` record timestamps changed from epoch floats to ISO 8601 UTC strings via `utc_now_iso()`.

- **Legacy Deprecation Wrappers (P2-3)** — Added functional `warnings.warn(DeprecationWarning)` to `privacy/phi_detector.py` and `privacy/deidentification.py` at both module and class levels.

- **Version Bump** — `pyproject.toml`, `CITATION.cff`, `README.md` badge updated from 0.9.7 to 0.9.9.

- **Ruff Configuration** — Added `"utils/**/*.py" = ["F401"]` to `ruff.toml`.

### Contributors

- @kevinkawchak
- @claude
- @codex

## [0.9.8] - 2026-02-19

### Added

- **Senior Peer Review Recommendations** (`peer-review/v0.9.8-senior-peer-review-recommendations.md`)
  - Added an end-to-end senior engineering review from the current repository state with a prioritized remediation backlog for Claude Code.
  - Captured peer-review process metrics: repository inventory, full test validation snapshot, and static-risk findings.
  - Published 10 actionable recommended code fixes across P1-P3 priorities, focused on secrets management, audit integrity chaining, PHI-safe logging, typed exception handling, and CI security guardrails.

- **Release Documentation** (`releases.md/v0.9.8.md`)
  - Added v0.9.8 release notes summarizing the peer-review cycle, recommendation totals, and contributor list.

### Changed

- `CHANGELOG.md` — Added v0.9.8 entry for the latest senior peer-review cycle and recommendation metrics.

### Contributors

- @kevinkawchak
- @codex
- @claude

## [0.9.7] - 2026-02-19

### Added

- **Peer Review Fixes Document** (`peer-review/v0.9.7-peer-review-fixes.md`)
  - Comprehensive fix report addressing all 9 recommendations from the v0.9.6 senior peer review across P1-P3 priority bands, with per-fix traceability, correction process metrics, and final validation results.

- **Version Alignment CI Check** (`scripts/check_version_alignment.py`)
  - Automated version-string consistency check comparing `pyproject.toml`, `CITATION.cff`, and `README.md` badge. Added to CI `validate-scripts` job and local `scripts/lint.sh` workflow. 71 LOC.

- **Static Security Pattern Scanner** (`scripts/security_scan.py`)
  - Scans all Python files for 5 vulnerability patterns (SEC-001 through SEC-005): torch.load without weights_only, np.load with allow_pickle, hashlib.md5 usage, UTC-naive datetime.now(), and bare except clauses. Integrated into `scripts/lint.sh`. 96 LOC.

- **Release Metrics Generator** (`scripts/release_metrics.py`)
  - Automated snapshot of repository-wide metrics (Python files, LOC, test files, Markdown, YAML, directories). Supports `--json` output for CI integration. 89 LOC.

- **Release Documentation** (`releases.md/v0.9.7.md`)
  - Added v0.9.7 release notes summarizing peer-review fixes, new tooling, and contributors.

### Changed

- **UTC Timestamp Hardening** — Replaced all 24 `datetime.now()` calls with `datetime.now(timezone.utc)` across 6 regulatory-submissions modules (`submission_orchestrator.py`, `compliance_validator.py`, `document_generator.py`, `ectd_compiler.py`, `regulatory_intelligence.py`, `submission_analytics.py`).

- **Typed Exception Handlers** — Narrowed 17 broad `except Exception as exc` handlers to specific exception tuples across 12 files: `framework_detector.py`, `conversion_pipeline.py` (3), `benchmark_runner.py` (3), `verify_installation.py` (2), `deployment_readiness.py`, `dicom_inspector.py`, `dose_calculator.py`, `sim_job_runner.py`, `trial_site_monitor.py`.

- **MD5 to SHA-256** — Replaced `hashlib.md5()` with `hashlib.sha256()` in `06_oncology_rag_compliance_agent.py` embedding generator.

- **Privacy Module Deprecation Notices** — Added `.. deprecated::` docstrings to `privacy/phi_detector.py` and `privacy/deidentification.py` directing new code to canonical sub-package implementations.

- **CI Enhancement** — Added `Check version alignment` step to `.github/workflows/ci.yml` `validate-scripts` job.

- **Local Lint Enhancement** — Added security scan and version alignment checks to `scripts/lint.sh`.

- **Version Alignment** — Bumped version from `0.9.5` to `0.9.7` across `pyproject.toml`, `CITATION.cff`, and `README.md` badge.

- `CHANGELOG.md` — Added v0.9.7 entry.
- `prompts.md` — Added Prompt 14 (v0.9.7) for peer review fixes implementation.

### Contributors

- @kevinkawchak
- @claude
- @codex

## [0.9.6] - 2026-02-19

### Added

- **Senior Peer Review Recommendations** (`peer-review/v0.9.6-senior-peer-review-recommendations.md`)
  - Added a repository-wide senior engineering review from the current repo state with prioritized remediation guidance for Claude Code.
  - Included quantitative peer-review metrics: inventory coverage (225 Python files / 86,126 LOC), automated validation outcomes (1,819 tests passed), and risk-focused static findings (exception breadth, timestamp consistency, hash/compliance posture).
  - Documented 9 actionable code-fix recommendations across P1–P3 priority bands, including UTC timestamp hardening, typed exception handling, MD5 replacement guidance, version alignment automation, and privacy module consolidation strategy.

- **Release Documentation** (`releases.md/v0.9.6.md`)
  - Added v0.9.6 release notes summarizing this peer-review release, contributors, and implementation notes for downstream remediation work.

### Changed

- `CHANGELOG.md` — Added v0.9.6 entry for the latest senior peer-review cycle and remediation backlog metrics.

### Contributors

- @kevinkawchak
- @claude
- @codex

## [0.9.5] - 2026-02-19

### Added

- **Lint Constraints File** (`constraints/lint.txt`)
  - Pinned `ruff==0.15.1` and `yamllint==1.38.0` as single source of truth for CI and local development.

- **Local Lint Script** (`scripts/lint.sh`)
  - Mirrors CI `lint-and-format` job exactly — runs `ruff check`, `ruff format --check`, and `yamllint -d relaxed`. Supports `--fix` flag for auto-correction. 44 LOC.

- **Pre-commit Configuration** (`.pre-commit-config.yaml`)
  - Hooks for `ruff` (check + format) and `yamllint` pinned to the same versions as `constraints/lint.txt` and CI.

- **Peer Review Fixes Document** (`peer-review/v0.9.5-peer-review-fixes.md`)
  - Comprehensive fix report addressing all 12 recommendations from the v0.9.4 senior peer review across 4 finding categories (A–D), with per-fix traceability, correction process metrics, and final validation results.

- **Release Documentation** (`releases.md/v0.9.5.md`)
  - v0.9.5 release notes covering CI determinism fixes, local reproducibility tooling, version alignment, and process hardening.

### Changed

- `.github/workflows/ci.yml` — Replaced floating `pip install ruff yamllint` with constrained `pip install -c constraints/lint.txt ruff yamllint`; added "Show tool versions" diagnostic step printing `python --version`, `ruff --version`, `yamllint --version`, and `pip freeze` filter for auditability.
- `pyproject.toml` — Version bumped from `0.9.3` to `0.9.5`; pinned `ruff==0.15.1` (was `>=0.4.0`); added `yamllint==1.38.0` to `[project.optional-dependencies] dev`.
- `CITATION.cff` — Version bumped from `0.9.2` to `0.9.5`; date updated to `2026-02-19`.
- `README.md` — Release badge updated from `v0.9.2` to `v0.9.5`.
- `prompts.md` — Added Prompt 13 (v0.9.5) for peer review fixes implementation.

### Contributors

- @kevinkawchak
- @claude
- @codex

## [0.9.4] - 2026-02-19

### Added

- **Senior Peer Review Artifacts** (`peer-review/`)
  - `peer-review/v0.9.4-senior-peer-review.md` — Repository-wide engineering peer review with CI risk analysis, prioritized code-fix recommendations for Claude, acceptance criteria, and quantitative review metrics.

- **Release Documentation** (`releases.md/`)
  - `releases.md/v0.9.4.md` — v0.9.4 release notes covering summary, features, contributors, and notes for this peer-review release.

### Changed

- Documented targeted remediation guidance for PR check stability, with emphasis on preventing `lint-and-format` failures across Python 3.10/3.11/3.12 by recommending pinned lint toolchains and reproducible CI/local lint workflows.

### Removed

- **Interactive Visualization Outputs** (`images/interactive/`)
  - Removed 60 HTML files (20 per set × 3 sets) — light and dark mode Plotly HTML exports from `1st/`, `2nd/`, `3rd/` directories. These generated outputs can be reproduced by running the retained Python scripts.
  - Removed 60 PNG files (20 per set × 3 sets) — light and dark mode static image exports from `1st/`, `2nd/`, `3rd/` directories. These generated outputs can be reproduced by running the retained Python scripts.
  - Retained all 30 Python visualization scripts (10 per set), 3 README.md files (1 per set), and the `generate_all.py` master script.

### Contributors

- @kevinkawchak
- @claude
- @codex

## [0.9.3] - 2026-02-18

### Added

- **Visualization Infrastructure** (`images/`)
  - `images/prompts/plan.md` — Overall visualization strategy: 30 Plotly charts across 3 sets of 10, light + dark mode, data sourced from repository modules. Pipeline specification (Python → HTML → PNG), color scheme definitions, and quality gate requirements.
  - `images/prompts/1st.md` — Set 1 per-chart instructions: chart type, data source, color schemes, and labels for 10 visualizations covering repository architecture and simulation.
  - `images/prompts/2nd.md` — Set 2 per-chart instructions: 10 visualizations covering training metrics, benchmarks, and performance analysis.
  - `images/prompts/3rd.md` — Set 3 per-chart instructions: 10 visualizations covering regulatory compliance, privacy framework, and deployment readiness.

- **Set 1 — Repository Architecture & Simulation** (`images/interactive/1st/`, 10 scripts + README)
  - `01_repository_architecture_treemap.py` — Treemap of repository module structure (194 Python files across 10+ modules).
  - `02_clinical_trial_workflow.py` — Sankey diagram of clinical trial workflow phases (Enrollment → Submission).
  - `03_oncology_process_diagram.py` — Sunburst chart of treatment simulation pipeline (5 modalities, 7 drug classes, 4 responses).
  - `04_framework_comparison_radar.py` — Radar chart comparing Isaac Sim/MuJoCo/Gazebo/PyBullet across 8 capability axes.
  - `05_parameter_mapping_heatmap.py` — Heatmap of PhysX ↔ MuJoCo physics parameter compatibility.
  - `06_data_pipeline_throughput.py` — Grouped bar chart of federated pipeline throughput by site size.
  - `07_model_format_comparison.py` — Grouped bar chart comparing PyTorch/ONNX/SafeTensors/TensorFlow/MONAI.
  - `08_state_vector_visualization.py` — 3D scatter of simulation state vectors (synchronized vs drifted).
  - `09_domain_randomization_transfer.py` — Multi-line chart of sim-to-real transfer by randomization level.
  - `10_oncology_trajectories.py` — Tumor growth curves (Exponential/Logistic/Gompertz/Von Bertalanffy) with treatment intervention.

- **Set 2 — Training & Benchmarks** (`images/interactive/2nd/`, 10 scripts + README)
  - `01_training_curves.py` — Federated training convergence for FedAvg/FedProx/SCAFFOLD.
  - `02_benchmark_comparisons.py` — Cross-framework benchmark results (Latency, Step Rate, Memory, Accuracy).
  - `03_safety_metrics_dashboard.py` — 4-panel safety dashboard (force monitoring, workspace violations, e-stop frequency, ISO 13482).
  - `04_performance_radars.py` — Overlapping radar charts for 6 platform module performance profiles.
  - `05_accuracy_confusion_matrix.py` — 12×12 tumor type classification confusion matrix.
  - `06_sim_to_real_transfer.py` — Dual-axis chart (success rate + position error) for transfer methods.
  - `07_latency_distributions.py` — Box plots of Isaac-MuJoCo sync latency by direction.
  - `08_resource_utilization.py` — Stacked area chart of federated training resource allocation.
  - `09_oncology_kpi_tracker.py` — Indicator gauges for 6 clinical trial KPIs.
  - `10_cross_framework_consistency.py` — Scatter plot of cross-engine state agreement.

- **Set 3 — Regulatory & Readiness** (`images/interactive/3rd/`, 10 scripts + README)
  - `01_oncology_convergence.py` — Federated convergence across 5 oncology subtasks.
  - `02_multi_site_trial_dashboard.py` — 4-panel trial dashboard (enrollment waterfall, screening funnel, monthly trend, site performance).
  - `03_algorithm_comparison_radar.py` — FedAvg/FedProx/SCAFFOLD trade-off radar.
  - `04_fda_device_classification_tree.py` — FDA device classification treemap (Class I/II/III → submission pathways).
  - `05_oncology_device_distribution.py` — Sunburst of oncology AI/ML device landscape.
  - `06_regulatory_compliance_scorecard.py` — Horizontal bar chart of compliance scores across 9 regulations.
  - `07_hipaa_phi_detection_matrix.py` — 18×5 heatmap of PHI detection confidence (HIPAA Safe Harbor × data sources).
  - `08_privacy_analytics_pipeline.py` — Sankey diagram of de-identification data flow.
  - `09_deployment_readiness_radar.py` — 10-axis deployment readiness assessment.
  - `10_production_readiness_scores.py` — Per-module production readiness with target lines.

- **Top-Level Documentation** (`images/README.md`) — Prompt-to-visualization workflow, pipeline diagrams (Python → HTML → PNG), conversion metrics (30/30 scripts, 60/60 HTML, 60/60 PNG), per-set LOC tables, data source reference table, and data inputs table for all 30 charts.

- **Release Notes** (`releases.md/v0.9.3.md`) — Visualization infrastructure release documentation.

### Changed

- `ruff.toml` — Added per-file-ignores `"images/**/*.py" = ["F401"]` for visualization scripts.
- `pyproject.toml` — Version bumped to 0.9.3.
- `prompts.md` — Added Prompt 12 (v0.9.3) for visualization infrastructure.

## [0.9.2] - 2026-02-18

### Added

- **Release Documentation** (`V1_RELEASE.md`)
  - Section 1 — Needs Met: Platform gap analysis, development metrics table (194 Python files, ~80,600 LOC, 288 total files, 87 directories), AI model contributions (Claude Code Opus 4.6), 14 numbered deliverable summaries across all releases (v0.1.0 through v0.9.2).
  - Section 2 — Technical Achievements: Subsections for unification framework, simulation integration, domain examples, digital twin pipelines, agentic AI, CLI tools, regulatory/privacy infrastructure, and standards compliance.
  - Section 3 — Version History: Detailed timeline of all releases with security audit findings summary (61 findings: 12 security, 14 logic, 35 compliance).
  - Section 4 — v0.9.2 Standards Compliance: Semantic versioning, CI validation, changelog discipline, security posture, documentation completeness, licensing (MIT), dependency management, reproducibility, compliance/safety.

- **Development Proposals** (`DEVELOPMENT_PROPOSALS.md`)
  - Proposal 1: Real-Time Federated Safety Signal Detection — streaming AE surveillance with ICH E2B(R3) / MedDRA / FDA FAERS alignment.
  - Proposal 2: Genomic Biomarker Integration Pipeline — federated GWAS, VCF/MAF processing, precision oncology workflows.
  - Proposal 3: Adaptive Trial Design Engine — Bayesian adaptive designs, response-adaptive randomization, interim analysis automation.
  - Comparative feature table and strategic impact matrix across all proposals.

- **Release Notes** (`releases.md/v0.9.2.md`) — Consolidated release documentation.

### Changed

- `README.md` — Updated to v0.9.2 badge, added release summary callout with metrics, expanded repository structure tree with `clinical-analytics/` and `regulatory-submissions/` entries, added `V1_RELEASE.md` and `DEVELOPMENT_PROPOSALS.md` to tree, updated citation version.
- `pyproject.toml` — Version bumped to 0.9.2.
- `CITATION.cff` — Updated to v0.9.2 with `clinical-analytics`, `regulatory-submissions`, `survival-analysis`, `pkpd-modeling` keywords and expanded abstract.
- `prompts.md` — Added Prompt 11 (v0.9.2) for release documentation consolidation.

## [0.9.1] - 2026-02-18

### Added

- **Regulatory Submissions Platform Module** (`regulatory-submissions/`)
  - `submission_orchestrator.py` (870+ LOC) — Core workflow orchestrator with `SubmissionPhase`/`SubmissionType`/`WorkflowStatus`/`TaskPriority` Enums, `SubmissionConfig`/`SubmissionMilestone`/`WorkflowTask`/`SubmissionManifest` dataclasses, `RegulatorySubmissionOrchestrator` class managing submission lifecycle, task dependencies, milestone tracking, reviewer assignments, deadline management, and 21 CFR Part 11 audit logging. Includes `create_standard_510k_workflow()` factory.
  - `ectd_compiler.py` (670+ LOC) — Electronic Common Technical Document compilation engine with `ECTDModule`/`DocumentType`/`ValidationLevel` Enums, `ECTDDocument`/`ECTDSection`/`ModuleStructure`/`CompilationResult` dataclasses, `ECTDCompiler` class generating eCTD module structures per FDA Technical Conformance Guide, validating document completeness, producing XML backbone stubs, and computing SHA-256 checksums for all documents.
  - `compliance_validator.py` (750+ LOC) — Multi-regulation compliance validation engine with `Regulation`/`ComplianceStatus`/`FindingSeverity` Enums covering FDA 21 CFR 820/Part 11, HIPAA, GDPR, ISO 14971, IEC 62304, ICH E6(R3)/E9(R1), and EU MDR 2017/745. `ComplianceFinding`/`ComplianceAssessment`/`RegulatoryRequirement`/`AuditTrailEntry` dataclasses. `ComplianceValidator` with checklist-driven validation, gap analysis, remediation tracking, and compliance score computation.
  - `document_generator.py` (930+ LOC) — Automated regulatory document generation with `DocumentCategory`/`TemplateType` Enums supporting CSR synopses, statistical analysis plans, risk analysis reports (ISO 14971), software descriptions (IEC 62304), predicate comparisons, and AI/ML predetermined change control plans (PCCPs). `RegulatoryDocumentGenerator` produces structured Markdown output with section numbering, metadata, and revision tracking.
  - `regulatory_intelligence.py` (700+ LOC) — Multi-jurisdiction regulatory intelligence engine with `Jurisdiction`/`GuidanceStatus`/`ImpactLevel` Enums covering FDA, EMA, PMDA, TGA, Health Canada, MHRA, and NMPA. `GuidanceDocument`/`RegulatoryUpdate`/`ImpactAssessment`/`ComplianceTimeline` dataclasses. `RegulatoryIntelligenceEngine` tracking guidance documents relevant to AI/ML medical devices, assessing submission impact, generating compliance timelines.
  - `submission_analytics.py` (680+ LOC) — Submission quality and timeline analytics with `MetricType`/`TrendDirection`/`BenchmarkSource` Enums, `SubmissionMetric`/`TimelineAnalysis`/`DeficiencyPattern`/`PerformanceBenchmark` dataclasses. `SubmissionAnalyticsEngine` computing submission KPIs (cycle time, deficiency rates, first-pass yield), identifying patterns, benchmarking against FDA MDUFA performance goals.
  - `README.md` — Architecture diagram, directory tree, per-component descriptions with class names, compliance alignment table, **Relationship to clinical-analytics/** section with integration points table.

- **Regulatory Submissions Examples** (`regulatory-submissions/examples-regulatory-submissions/`, 6 scripts)
  - `01_basic_submission_workflow.py` — Minimal: submission config, document registration, milestone tracking.
  - `02_ectd_compilation.py` — Single feature: eCTD module structure compilation with validation.
  - `03_compliance_validation.py` — Multi-regulation compliance checking against FDA/ISO/IEC.
  - `04_document_generation.py` — Template-driven 510(k) summary and software description generation.
  - `05_regulatory_intelligence.py` — Guidance tracking, impact assessment, compliance timelines.
  - `06_full_submission_pipeline.py` — End-to-end combining all modules with clinical-analytics integration.
  - `README.md` — Example overview table, prerequisites, run commands.

- **Regulatory Submissions Tests** (`tests/test_regulatory_submissions/`, 6 files)
  - `test_submission_orchestrator.py` — Workflow lifecycle, task dependencies, milestone tracking.
  - `test_ectd_compiler.py` — Module compilation, document validation, checksum verification.
  - `test_compliance_validator.py` — Multi-regulation scoring, gap analysis, remediation tracking.
  - `test_document_generator.py` — Template rendering, section generation, metadata integrity.
  - `test_regulatory_intelligence.py` — Guidance tracking, impact assessment, timeline computation.
  - `test_submission_analytics.py` — KPI computation, trend analysis, benchmarking, clinical-analytics integration test.

### Changed

- `ruff.toml` — Added per-file-ignores for `regulatory-submissions/*.py` and `regulatory-submissions/**/*.py`.

## [0.9.0] - 2026-02-18

### Added

- **Clinical Analytics Platform Module** (`clinical-analytics/`)
  - `analytics_orchestrator.py` (700+ LOC) — Core pipeline orchestrator with `AnalyticsPhase`/`TaskStatus`/`AggregationMode` Enums, `AnalyticsConfig`/`AnalyticsTask`/`PipelineManifest` dataclasses, `ClinicalAnalyticsOrchestrator` class managing DAG-based task scheduling, federated dispatch, convergence tracking, and 21 CFR Part 11 audit logging.
  - `pkpd_engine.py` (700+ LOC) — Population PK/PD compartmental modeling with `CompartmentModel`/`EstimationMethod`/`AbsorptionRoute` Enums, `OneCompartmentModel`/`TwoCompartmentModel` classes with analytical and ODE-based solutions via `scipy.integrate.solve_ivp`, `EmaxModel` sigmoidal dose-response with Hill coefficient, NCA computation (trapezoidal AUC, Cmax, Tmax, terminal half-life), allometric clearance adjustment.
  - `risk_stratification.py` (700+ LOC) — Multi-factor clinical risk scoring with `RiskCategory`/`EnrichmentAction` Enums, `PatientRiskProfile`/`StratificationResult` dataclasses, `ClinicalRiskStratifier` computing composite weighted risk scores across tumor burden, biomarkers, ECOG PS, comorbidity indices, `AdaptiveEnrichmentAdvisor` with decision-support recommendations per stratum, Hosmer-Lemeshow calibration assessment.
  - `trial_data_manager.py` (600+ LOC) — Data lifecycle management with `DatasetStatus`/`QualityLevel`/`QueryStatus` Enums, `DatasetRegistration`/`QualityCheckResult`/`DataQuery` dataclasses, `TrialDataManager` for multi-site dataset registration, schema validation, completeness/range/consistency checks, immutable audit trails.
  - `clinical_interoperability.py` (800+ LOC) — Cross-system vocabulary mapping with ICD-10/SNOMED CT crosswalk (12 oncology codes), LOINC/RxNorm/MedDRA mappings, unit conversion (lb→kg, in→cm, F→C, mg/dL→mmol/L), `align_schemas()` for cross-site schema reconciliation detecting type mismatches and unit conflicts, `export_to_sdtm()` generating CDISC SDTM DM/LB/AE domain records.
  - `survival_analysis.py` (800+ LOC) — Privacy-preserving survival analysis with `kaplan_meier()` product-limit estimator using Greenwood standard errors and log-log confidence intervals, `log_rank_test()` Mantel-Haenszel test, `fit_cox_ph()` via BFGS partial likelihood maximization with Breslow ties, `concordance_index()` Harrell's C-index, `compute_rmst()` restricted mean survival time.
  - `consortium_reporting.py` (600+ LOC) — DSMB report generation with `ReportType`/`ReportStatus` Enums, `ReportSection`/`DSMBPackage` dataclasses, `ConsortiumReportingEngine` producing enrollment dashboards, safety summaries, efficacy snapshots, and regulatory submission appendices with SHA-256 integrity hashing and provenance metadata.
  - `README.md` — Architecture diagram, directory tree, per-component descriptions with class names and algorithms, quick start code examples, compliance alignment table (ICH E6(R3) / 21 CFR Part 11 / HIPAA / FDA AI/ML / CDISC / IEC 62304), roadmap alignment notes.

- **Clinical Analytics Examples** (`clinical-analytics/examples-clinical-analytics/`, 6 scripts)
  - `01_basic_analytics_pipeline.py` — Minimal working example: synthetic cohort generation, descriptive statistics, structured report.
  - `02_pkpd_modeling.py` — One-compartment and two-compartment PK models, NCA, Emax dose-response.
  - `03_risk_stratification.py` — Composite risk scoring, category assignment, calibration assessment.
  - `04_data_harmonization.py` — ICD-10→SNOMED mapping, schema alignment, unit conversion, SDTM export.
  - `05_survival_analysis.py` — Kaplan-Meier estimation, log-rank test, Cox PH fitting, C-index, RMST.
  - `06_full_clinical_analytics_workflow.py` — End-to-end pipeline combining all components with timing and DSMB report.
  - `README.md` — Example overview table, prerequisites, run commands, conventions.

- **Clinical Analytics Tests** (`tests/test_clinical_analytics/`, 7 files)
  - `test_analytics_orchestrator.py` — Orchestrator pipeline, task scheduling, convergence tracking.
  - `test_pkpd_engine.py` — Compartment models, NCA, Emax, clearance adjustment.
  - `test_risk_stratification.py` — Risk scoring, category assignment, enrichment advisory.
  - `test_trial_data_manager.py` — Dataset registration, quality checks, audit trails.
  - `test_clinical_interoperability.py` — Vocabulary mapping, schema alignment, SDTM export.
  - `test_survival_analysis.py` — KM estimator, log-rank, Cox PH, C-index, RMST.
  - `test_consortium_reporting.py` — Report generation, integrity hashing, provenance metadata.

### Changed

- `ruff.toml` — Added per-file-ignores for `clinical-analytics/*.py` and `clinical-analytics/**/*.py`.

## [0.8.0] - 2026-02-18

### Added

- **Comprehensive Test Suite** (`tests/`)
  - `conftest.py` — Shared fixtures and `load_module()` helper using `importlib.util.spec_from_file_location` for loading modules from hyphenated directories. NumPy RNG seeded (`np.random.seed(42)`) via autouse fixture. Section-organized fixtures for all module categories. Mock data factories for patient records, model weights, trial configurations, sensor data, and PHI text.
  - `tests/README.md` — Testing philosophy, how to run, directory tree, how to add tests, CI integration notes.

- **Federated Learning Tests** (`tests/test_federated/`, 8 files)
  - Tests for coordinator (aggregation, SCAFFOLD, zero-count guards), client, model, differential privacy (zero-norm clipping), secure aggregation (HMAC pair seeds), site enrollment (audit trail copies), data ingestion, data harmonization.

- **Physical AI Tests** (`tests/test_physical_ai/`, 6 files)
  - Tests for digital twin (growth models: exponential, logistic, Gompertz), sensor fusion, robotic integration, surgical tasks (thresholds), simulation bridge (URDF/MJCF/SDF), framework detection.

- **Digital Twin Tests** (`tests/test_digital_twins/`, 3 files)
  - Tests for clinical integrator (HMAC patient ID hashing), patient model (biomarker simulation, zero-value handling), treatment simulator (PK modeling, AUC, trapezoid/trapz).

- **Privacy Tests** (`tests/test_privacy/`, 5 files)
  - Tests for PHI detector (18 HIPAA identifier patterns), de-identification (REDACT/HASH/PSEUDONYMIZE, os.urandom salt), access control (RBAC, audit log defensive copies), breach response (risk assessment), DUA generator (template types, retention).

- **Regulatory Tests** (`tests/test_regulatory/`, 4 files)
  - Tests for FDA submission tracker (model_type="unspecified", lifecycle states), IRB protocol manager (amendments, consent versioning), GCP compliance (NOT_ASSESSED exclusion), regulatory tracker (overdue-before-imminent ordering).

- **Tool Tests** (`tests/test_tools/`, 5 files)
  - Tests for dose calculator (BED, EQD2, max_dose=0 truthiness fix), deployment readiness, trial site monitor, sim job runner, DICOM inspector.

- **Unification Tests** (`tests/test_unification/`, 5 files)
  - Tests for Isaac-MuJoCo bridge, model converter, unified agent interface, framework detector, validation suite.

- **Standards Tests** (`tests/test_standards/`, 3 files)
  - Tests for conversion pipeline (format enums, tolerance), benchmark runner (metrics), model validator (metadata).

- **Agentic AI Tests** (`tests/test_agentic_ai/`, 5 files)
  - Tests for MCP server, ReAct planner, monitoring agent, simulation orchestrator, safety executor.

- **Example Tests** (`tests/test_examples/`, 6 files)
  - Tests for federated workflow, digital twin planning, cross-framework validation, outcome prediction, physical AI examples, synthetic data generation.

- **Integration Tests** (`tests/test_integration/`, 6 files)
  - Cross-module workflow tests: trial lifecycle, cross-framework validation, privacy→clinical, domain→safety, agentic→regulatory, federated→privacy.

- **Regression Tests** (`tests/test_regression/`, 1 file)
  - Guards for all v0.7.0 security audit findings: torch.load weights_only, np.load allow_pickle, HMAC hashing, mutable audit log copies, os.urandom salt, division-by-zero guards, truthiness fixes, dead data fix, RESEARCH USE ONLY compliance.

## [0.7.0] - 2026-02-18

### Security

- **torch.load Arbitrary Code Execution (5 instances):**
  - `conversion_pipeline.py` — Four `torch.load()` calls changed from `weights_only=False` to `weights_only=True` (lines 575, 619, 1020, 1085)
  - `benchmark_runner.py` — One `torch.load()` call changed from `weights_only=False` to `weights_only=True` (line 933)

- **np.load Pickle Deserialization (1 instance):**
  - `model_validator.py` — Added `allow_pickle=False` to prevent arbitrary code execution (line 1096)

- **Weak Hashing Without HMAC (3 instances):**
  - `federated/secure_aggregation.py` — Replaced plain `hashlib.sha256` with `hmac.new()` HMAC-SHA256 (line 107)
  - `privacy/audit_logger.py` — `_compute_hash()` now uses HMAC-SHA256 with module-level key (line 80)
  - `digital-twins/clinical-integration/clinical_integrator.py` — Patient ID hashing migrated from SHA-256 to HMAC-SHA256 (line 298)

- **Mutable Audit Log References (3 instances):**
  - `privacy/access_control.py` — `get_access_log()` now returns `list()` copy (line 253)
  - `privacy/consent_manager.py` — `get_audit_trail()` now returns `list()` copy (line 191)
  - `federated/coordinator.py` — `get_server_control()` and `get_client_control()` return `[p.copy()]` (lines 158, 174)

- **Weak Pseudonymization Salt (1 instance):**
  - `privacy/deidentification.py` — Replaced hardcoded `"pai-oncology-salt"` with `os.urandom(32)` default (line 64)

### Fixed

- **Division by Zero — Federation Coordinator (CRITICAL):**
  - `federated/coordinator.py` line 217 — Weight computation when `client_sample_counts` sums to zero now falls back to uniform weights
  - `federated/coordinator.py` line 293 — SCAFFOLD control update with zero clients now early-returns

- **Division by Zero — Gradient Clipping:**
  - `federated/differential_privacy.py` line 59 — Explicit zero-norm check returns copies instead of dividing

- **Truthiness Bug — Dose Calculator:**
  - `tools/dose-calculator/dose_calculator.py` line 386 — `max_dose=0` no longer treated as falsy; changed `or` chain to explicit `is None` check

- **Truthiness Bug — Patient Model:**
  - `digital-twins/patient-modeling/patient_model.py` line 509 — Zero biomarker values no longer rejected; changed `<= 0.0` to `< 0.0`

- **Dead Data — Multi-Sensor Fusion:**
  - `examples-physical-ai/02_multi_sensor_fusion.py` line 651 — Anomaly record `"vital_name"` key was hardcoded to `0.0`; replaced with actual `vital_name` variable

- **Division by Zero — Outcome Prediction:**
  - `examples/05_outcome_prediction_pipeline.py` lines 721-722 — Explicit zero-patient check with `0.0` default percentages

- **Missing Null Check — Treatment Simulator:**
  - `digital-twins/treatment-simulation/treatment_simulator.py` line 301 — Added RuntimeError when NumPy trapezoid/trapz is unavailable

### Changed

- **RESEARCH USE ONLY Compliance:** Added disclaimer to 35 Python module docstrings that were missing it (all 82 modules now compliant)

### Added

- `SECURITY_AUDIT.md` — Comprehensive audit report documenting 61 findings (12 security, 14 logic, 35 compliance) with full remediation details, severity ratings, and ongoing security recommendations

## [0.6.0] - 2026-02-18

### Added

- **Privacy Framework — PHI/PII Management** (`privacy/phi-pii-management/`)
  - `phi_detector.py` — Advanced PHI/PII detection engine (600+ LOC) with `PHICategory` Enum mapping all 18 HIPAA Safe Harbor identifiers per 45 CFR 164.514(b)(2). `PHIFinding` dataclass with redacted value preview (never stores full PHI), confidence scoring, and risk-level classification. `PHIDetector` class with regex patterns for SSN, MRN, phone, email, dates, geographic data, IP, ZIP, account numbers, certificates, vehicle IDs. DICOM tag scanning via conditional `pydicom` import (DICOM PS3.15 Annex E tags). Optional Presidio NER integration. Batch scanning with `ScanResult` aggregation. Non-destructive logging throughout.
  - `README.md` — Module documentation with API overview and usage examples.

- **Privacy Framework — De-identification Pipeline** (`privacy/de-identification/`)
  - `deidentification_pipeline.py` — Production de-identification pipeline (550+ LOC) with `DeidentificationMethod` Enum (REDACT, HASH, GENERALIZE, SUPPRESS, PSEUDONYMIZE, DATE_SHIFT). HMAC-SHA256 pseudonymization with cryptographically random 32-byte salt via `os.urandom(32)` (not a weak default). Consistent patient-level date shifting derived from HMAC of patient_id and salt. Geographic generalization (ZIP to 3-digit prefix per Safe Harbor). Transformation audit trail with `TransformationRecord` dataclass. Batch processing with per-record results.
  - `README.md` — Module documentation.

- **Privacy Framework — Access Control** (`privacy/access-control/`)
  - `access_control_manager.py` — Role-Based Access Control with 21 CFR Part 11 audit trail (500+ LOC). `AccessRole` Enum (10 roles: PI, coordinator, site admin, clinical researcher, data engineer, biostatistician, safety officer, auditor, regulatory affairs, patient). `Permission` Enum (22 permissions). `AuditAction` Enum (13 action types). Time-limited access with expiration enforcement. Invalid date formats deny access by default (fail-closed). `get_audit_log()` returns a copy (not mutable reference). Comprehensive `AuditEntry` dataclass per 21 CFR Part 11.
  - `README.md` — Module documentation.

- **Privacy Framework — Breach Response** (`privacy/breach-response/`)
  - `breach_response_protocol.py` — Breach response protocol (500+ LOC) implementing HIPAA Breach Notification Rule (45 CFR 164.400-414). `BreachSeverity` Enum. `RiskAssessment` dataclass with four-factor risk assessment per 45 CFR 164.402 and `calculate_risk()` that clamps out-of-range scores to [0.0, 1.0]. 60-day HIPAA notification timeline tracking. `BreachNotification` for HHS OCR, state AG, individual, media (500+ individuals), GDPR DPA. `RemediationAction` tracking with status lifecycle. Incident audit trail.
  - `README.md` — Module documentation.

- **Privacy Framework — DUA Generator** (`privacy/dua-templates-generator/`)
  - `dua_generator.py` — Data Use Agreement template generator (400+ LOC). `DUAType` Enum (INTRA_INSTITUTIONAL, MULTI_SITE, COMMERCIAL, ACADEMIC_COLLABORATION, GOVERNMENT). Security requirements tiered by DUA type (encryption, MFA, DLP, network segmentation, annual audits, background checks). `RetentionPolicy` with per-type retention periods. 9-section document generation covering preamble, definitions, permitted uses, security, retention, breach notification, term/termination, regulatory compliance, signatures.
  - `README.md` — Module documentation.

- **Privacy Framework — Top-Level** (`privacy/`)
  - `README.md` — Framework overview with directory tree, all 18 HIPAA identifiers tabulated, regulatory cross-reference tables for HIPAA, 21 CFR Part 11, GDPR, EU AI Act, NIST Privacy Framework, and ICH E6(R3).

- **Regulatory Framework — FDA Compliance** (`regulatory/fda-compliance/`)
  - `fda_submission_tracker.py` — FDA submission lifecycle tracker (650+ LOC). `SubmissionType` Enum (510k, De_Novo, PMA, Breakthrough, Pre_Submission). `SubmissionStatus` Enum (12 states). `DeviceClass` Enum. `AIMLComponent` dataclass with `model_type` defaulting to `"unspecified"` (not `"classification"`). Standard document requirements per pathway referencing FDA draft guidance for AI devices (Jan 2025), PCCP guidance (Aug 2025), QMSR (Feb 2026). `ReviewCycle` tracking for FDA interactions. Readiness reporting.
  - `README.md` — Module documentation.

- **Regulatory Framework — IRB Management** (`regulatory/irb-management/`)
  - `irb_protocol_manager.py` — IRB protocol lifecycle manager (500+ LOC). Uses `from __future__ import annotations` for forward references. `ProtocolStatus` Enum (11 states), `ReviewType`, `AmendmentType`, `ConsentType`, `RiskLevel` Enums. Protocol lifecycle management with submission, approval (with expiration), suspension, termination, closure. Amendment management with version incrementing. Consent versioning with AI/ML disclosure tracking. Continuing review workflow. References ICH E6(R3) published September 2025.
  - `README.md` — Module documentation.

- **Regulatory Framework — ICH-GCP Compliance** (`regulatory/ich-gcp/`)
  - `gcp_compliance_checker.py` — ICH E6(R3) GCP compliance assessment engine (550+ LOC). 13 `GCPPrinciple` Enum values including risk-proportionate approach (E6(R3) innovation). `ComplianceLevel` Enum (COMPLIANT, MINOR_FINDING, MAJOR_FINDING, CRITICAL_FINDING, NOT_ASSESSED). Compliance score calculation excludes NOT_ASSESSED findings from the denominator. Standard checklist covering ethical conduct, scientific soundness, informed consent, IRB/IEC review, qualified personnel, data integrity, protocol compliance, participant safety, quality management, record keeping, and risk-proportionate approach. References ICH E6(R3) published September 2025.
  - `README.md` — Module documentation.

- **Regulatory Framework — Regulatory Intelligence** (`regulatory/regulatory-intelligence/`)
  - `regulatory_tracker.py` — Multi-jurisdiction regulatory intelligence tracker (550+ LOC). Monitors FDA, EMA, PMDA, TGA, and Health Canada. `Jurisdiction`, `UpdateType`, `Priority`, `ComplianceStatus`, `ActionStatus` Enums. Built-in jurisdiction profiles with AI/ML framework references. `get_recent_updates()` uses computed cutoff date from `timedelta(days=days)`. `get_overdue_requirements()` checks OVERDUE (past-due) BEFORE IMMINENT — correct if/elif ordering ensures overdue status is always reachable. Action item tracking linked to compliance requirements.
  - `README.md` — Module documentation.

- **Regulatory Framework — Human Oversight** (`regulatory/human-oversight/`)
  - `HUMAN_OVERSIGHT_QMS.md` — Human Oversight Quality Management System document defining CRF auto-fill risk tiers (low: auto-populate with audit; medium: draft + clinician review; high: advisory only + manual entry), AE automation boundaries (detection: auto; grading: draft; causality: advisory; submission: prohibited without e-signature), and safety gates (SAE immediate escalation, no autonomous regulatory submission, >95% sensitivity threshold for AE detection).

- **Regulatory Framework — Top-Level** (`regulatory/`)
  - `README.md` — Regulatory landscape overview with directory tree, standards cross-reference (FDA, EMA, PMDA, TGA, Health Canada), and FDA device statistics context.

### Changed

- `README.md` — Updated to v0.6.0, expanded privacy/ and regulatory/ sections in repository structure tree with new subdirectories, updated Key Capabilities sections 3 and 4 with detailed framework descriptions.
- `pyproject.toml` — Version bumped to 0.6.0.
- `CITATION.cff` — Updated to v0.6.0 with 2026-02-18 release date.
- `ruff.toml` — Added per-file-ignores for `privacy/**/*.py` and `regulatory/**/*.py` subdirectories.
- `prompts.md` — Added Prompt 6 (v0.6.0) for privacy framework and regulatory compliance infrastructure.

## [0.5.0] - 2026-02-17

### Added

- **Core Examples** (`examples/`)
  - `01_federated_training_workflow.py` — End-to-end federated training workflow with site enrollment, data harmonization, multi-round training, convergence monitoring, and cross-site evaluation.
  - `02_digital_twin_planning.py` — Digital twin treatment planning demonstrating patient twin creation, multi-modality treatment simulation, uncertainty quantification, and clinical report generation.
  - `03_cross_framework_validation.py` — Cross-framework policy validation across Isaac Sim, MuJoCo, and PyBullet with roundtrip conversion testing and state equivalence verification.
  - `04_agentic_clinical_workflow.py` — AI-driven clinical workflow using the unified agent interface for treatment planning, literature review, and compliance checking.
  - `05_outcome_prediction_pipeline.py` — End-to-end prediction pipeline from patient data loading through digital twin feature generation, federated model training, evaluation, and clinical reporting.
  - `examples/README.md` with overview table and quick start.

- **Physical AI Examples** (`examples-physical-ai/`)
  - `01_realtime_safety_monitoring.py` — Real-time IEC 80601-2-77 safety monitoring with force limits, workspace boundaries, emergency stop logic, and telemetry streaming.
  - `02_multi_sensor_fusion.py` — Multi-sensor fusion combining force/torque sensors, cameras, joint encoders, and patient monitoring with extended Kalman filter simulation.
  - `03_ros2_surgical_deployment.py` — ROS 2 surgical deployment pattern with node lifecycle, service discovery, and action server-based procedure execution.
  - `04_hand_eye_calibration.py` — AX=XB hand-eye calibration for surgical robots with transformation chain validation and statistical calibration error analysis.
  - `05_shared_autonomy_teleoperation.py` — Five levels of human-AI shared autonomy (manual, assisted, shared, supervised, autonomous) with safety gate transitions.
  - `06_robotic_sample_handling.py` — Automated specimen handling with 21 CFR Part 11 audit trails, chain of custody, and electronic signature simulation.
  - `examples-physical-ai/README.md` with regulatory standards cross-reference.

- **Agentic AI Examples** (`agentic-ai/examples-agentic-ai/`)
  - `01_mcp_oncology_server.py` — MCP server exposing oncology tools/resources for LLM agents with RobotMode/ProcedurePhase Enums, telemetry dataclass, and 21 CFR Part 11 audit logging.
  - `02_react_treatment_planner.py` — ReAct agent with thought/action/observation loops for iterative treatment planning using digital twin simulations.
  - `03_realtime_adaptive_monitoring_agent.py` — Real-time adaptive agent processing streaming multi-modal clinical data with cross-modal correlation and adaptive alerting.
  - `04_autonomous_simulation_orchestrator.py` — Multi-framework simulation orchestration agent managing jobs across Isaac Sim, MuJoCo, and PyBullet.
  - `05_safety_constrained_agent_executor.py` — Safety-gated agent with IEC 80601-2-77 / ISO 14971 alignment, pre/post-condition checks, human-in-the-loop, emergency stop, and rollback.
  - `06_oncology_rag_compliance_agent.py` — RAG agent grounded in FDA guidance, ICH E6(R3), and IEC standards with protocol citation tracking.
  - `agentic-ai/examples-agentic-ai/README.md` with example table and quick start.

### Changed

- `README.md` — Updated to v0.5.0, added all 17 example scripts to repository structure tree.
- `pyproject.toml` — Version bumped to 0.5.0.
- `CITATION.cff` — Updated to v0.5.0.
- `prompts.md` — Added Prompt 5 (v0.5.0) for domain examples and engineering demonstrations.
- `ruff.toml` — Per-file-ignores already covered examples-physical-ai/ and agentic-ai/ directories.

## [0.4.0] - 2026-02-17

### Added

- **Digital Twins Primary Domain** (`digital-twins/`)
  - `patient-modeling/patient_model.py` — Patient digital twin modeling engine with TumorType Enum (NSCLC, SCLC, BREAST_DUCTAL, BREAST_LOBULAR, COLORECTAL, PANCREATIC, etc.), GrowthModel Enum (EXPONENTIAL, LOGISTIC, GOMPERTZ, VON_BERTALANFFY), PatientStatus Enum, TumorProfile/PatientBiomarkers/OrganFunction/PatientDigitalTwinConfig dataclasses, PatientPhysiologyEngine with tumor growth kinetics and biomarker evolution, PatientDigitalTwinFactory.
  - `treatment-simulation/treatment_simulator.py` — Treatment simulation with TreatmentModality Enum (CHEMOTHERAPY, RADIATION, IMMUNOTHERAPY, TARGETED_THERAPY, COMBINATION), ResponseCategory Enum (RECIST criteria), ToxicityGrade Enum (CTCAE grades 0–5), ChemotherapyProtocol/RadiationPlan/ImmunotherapyRegimen/TreatmentOutcome/ToxicityProfile dataclasses, TreatmentSimulator with pharmacokinetic/pharmacodynamic modeling.
  - `clinical-integration/clinical_integrator.py` — Clinical system integration with ClinicalSystem Enum (EHR, PACS, LIS, RIS, TPS), IntegrationStatus/DataFormat Enums, ClinicalDataPoint/IntegrationConfig/SyncResult/ClinicalReport dataclasses, ClinicalIntegrator for data ingestion and report generation, FederatedClinicalBridge connecting digital twins to the federated learning pipeline.
  - Subdirectory README.md files with purpose, API overview, and usage examples.

- **CLI Tools** (`tools/`)
  - `dicom-inspector/dicom_inspector.py` — DICOM medical image inspection with argparse subcommands (inspect, validate, summarize), DicomModality Enum, DicomInspectionResult dataclass, JSON output support.
  - `dose-calculator/dose_calculator.py` — Radiation and chemotherapy dose calculator with reference constants citing NCCN guidelines and RTOG protocols, DoseUnit/ChemoAgent/RadiationTechnique Enums, DoseCalculation/PatientParameters dataclasses, all doses bounded.
  - `trial-site-monitor/trial_site_monitor.py` — Multi-site clinical trial monitoring with SiteStatus/EnrollmentStatus/DataQualityStatus Enums (GREEN/YELLOW/RED levels), SiteMetrics/MonitoringReport dataclasses.
  - `sim-job-runner/sim_job_runner.py` — Cross-framework simulation job launcher with SimFramework/JobStatus/TaskType Enums, SimJobConfig/SimJobResult dataclasses, framework auto-detection.
  - `deployment-readiness/deployment_readiness.py` — Deployment readiness checker with CheckStatus Enum (PASS/FAIL/WARNING/REQUIRES_VERIFICATION), ReadinessCheck/ReadinessReport dataclasses, validation across model, safety, regulatory, infrastructure, and documentation categories.
  - `tools/README.md` with directory tree, per-tool CLI usage summaries, and common workflows.

### Changed

- `README.md` — Updated to v0.4.0, added digital-twins/ and tools/ to repository structure tree.
- `pyproject.toml` — Version bumped to 0.4.0.
- `CITATION.cff` — Updated to v0.4.0.
- `prompts.md` — Added Prompt 4 (v0.4.0) for core domain modules and CLI tools.

## [0.3.0] - 2026-02-17

### Added

- **Unification Framework** (`unification/`)
  - `unification/simulation_physics/isaac_mujoco_bridge.py` — Bidirectional state conversion between NVIDIA Isaac Sim and MuJoCo with dataclass configs, Enum states, conditional imports, and structured logging.
  - `unification/simulation_physics/physics_parameter_mapping.yaml` — Cross-framework physics parameter equivalences (gravity, friction, damping, solver settings).
  - `unification/agentic_generative_ai/unified_agent_interface.py` — Unified agent interface with AgentBackend Enum (CrewAI/LangGraph/AutoGen/Custom), AgentRole Enum, Tool dataclass with `to_mcp_format()`, `to_openai_format()`, `to_anthropic_format()` methods.
  - `unification/cross_platform_tools/framework_detector.py` — @dataclass FrameworkInfo + SystemInfo, CLI with `--verbose`/`--json` flags.
  - `unification/cross_platform_tools/model_converter.py` — Cross-platform model format conversion (PyTorch, ONNX, SafeTensors, TorchScript).
  - `unification/cross_platform_tools/policy_exporter.py` — RL policy export across simulation frameworks.
  - `unification/cross_platform_tools/validation_suite.py` — Enum-based validation result states, cross-platform policy validation.
  - `unification/standards_protocols/` — Data formats, communication protocols, and safety standards documentation.
  - `unification/integration_workflows/workflow_templates.yaml` — Integration workflow templates.
  - `unification/surgical_robotics/` — Challenges and opportunities in unified surgical robotics.
  - `unification/README.md` — Core principles, directory structure, quarterly roadmap (Q1–Q4 2026).

- **Q1 2026 Standards** (`q1-2026-standards/`)
  - `objective-1-model-conversion/conversion_pipeline.py` — Model format conversion pipeline with Enum-based format types, ConversionConfig/ConversionResult dataclasses.
  - `objective-2-model-registry/model_registry.yaml` — Schema for federated model registry entries.
  - `objective-2-model-registry/model_validator.py` — Model validation with ValidationStatus Enum, ModelMetadata dataclass.
  - `objective-3-benchmarking/benchmark_runner.py` — Cross-platform benchmark runner with BenchmarkMetric Enum, latency/accuracy/memory tests.
  - `implementation-guide/timeline.md` — Q1 2026 implementation milestones.
  - `implementation-guide/compliance_checklist.md` — Standards adoption compliance checklist.
  - `q1-2026-standards/README.md` — Three objectives with status and deliverables.

- **Framework Integration Guides** (`frameworks/`)
  - `nvidia-isaac/INTEGRATION.md` — NVIDIA Isaac Sim/Lab system requirements, installation, code snippets.
  - `mujoco/INTEGRATION.md` — MuJoCo integration with physics parameter tuning for medical robotics.
  - `gazebo/INTEGRATION.md` — Gazebo + ROS 2 Humble/Iron integration for clinical robotics.
  - `pybullet/INTEGRATION.md` — PyBullet rapid prototyping guide.

- **Learning Domain Directories**
  - `supervised-learning/` — Strengths, limitations, and illustrative results for supervised learning in oncology.
  - `reinforcement-learning/` — Strengths, limitations, and illustrative results for RL in surgical robotics and treatment planning.
  - `self-supervised-learning/` — Strengths, limitations, and illustrative results for SSL in medical imaging.
  - `generative-ai/` — Strengths, limitations, and illustrative results for generative AI in oncology.

- **Configuration** (`configs/`)
  - `training_config.yaml` — Comprehensive training configuration with environment, domain randomization, PPO/SAC algorithms, federated learning parameters, network architecture, safety constraints, curriculum learning, evaluation metrics, deployment, and sim-to-real transfer settings.

- **Verification** (`scripts/`)
  - `verify_installation.py` — importlib-based dependency verification with colored output, version validation, and graceful handling of missing optional packages.

- **Community Health**
  - `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1 adapted for clinical software (patient safety, data privacy, scientific integrity).
  - `CONTRIBUTING.md` — Contribution types, recency/oncology/reproducibility requirements, development workflow, benchmark data policy, safety/compliance section.
  - `SECURITY.md` — Vulnerability reporting (7-day acknowledgment / 30-day fix SLA), supported versions, deployer responsibilities.
  - `SUPPORT.md` — Getting help, upstream framework support table, regulatory disclaimer.
  - `.github/PULL_REQUEST_TEMPLATE.md` — Summary, 7 change type categories, safety/compliance checklist.
  - `.github/ISSUE_TEMPLATE/bug_report.md` — Environment, reproduction steps, expected/actual behavior.
  - `.github/ISSUE_TEMPLATE/feature_request.md` — Domain category checkboxes, clinical relevance, implementation notes.

- **CI/CD Enhancements**
  - `.github/workflows/ci.yml` — Added yamllint for YAML validation, validate-scripts job, explicit `permissions: contents read`, ruff `--output-format=github`.
  - `ruff.toml` — Standalone ruff configuration with line-length 120, per-file-ignores for all directories.

- **Root Files**
  - `posts.md` — Placeholder for external references and publications.
  - `releases.md/v0.3.0.md` — Release notes for v0.3.0.

### Changed

- `README.md` — Added badges (MIT license, v0.3.0, Python 3.10+), responsible-use notice, repository structure tree with all new directories, Core Technologies table, Unification Framework section, Multi-Organization Cooperation table, expanded dependency listing, contributing guidelines summary.
- `requirements.txt` — Expanded with grouped dependencies (Core, Deep Learning, Medical Imaging, Agentic AI, Visualization, Deployment, Testing), pinned versions with release dates and GitHub URLs.
- `pyproject.toml` — Version bumped to 0.3.0. Ruff configuration moved to standalone `ruff.toml`. Added `unification*` to setuptools package discovery.
- `CITATION.cff` — Updated to v0.3.0, added `unification-framework`, `agentic-ai`, `simulation-physics`, `cross-platform` keywords.
- `.github/workflows/ci.yml` — Enhanced with yamllint, validate-scripts job, permissions, output format.

## [0.2.0] - 2026-02-17

### Added

- **Aggregation Strategies**
  - `federated/coordinator.py` — Added FedProx (Li et al., 2020) with configurable proximal term `mu` for handling non-IID client data.
  - `federated/coordinator.py` — Added SCAFFOLD (Karimireddy et al., 2020) with server and client control variates for correcting client drift.
  - `federated/coordinator.py` — `AggregationStrategy` enum for selecting between `fedavg`, `fedprox`, and `scaffold`.
  - `federated/coordinator.py` — Automatic convergence detection with configurable window and threshold.

- **Data Harmonization**
  - `federated/data_harmonization.py` — New module for multi-site data alignment: DICOM/FHIR vocabulary mapping, unit conversion (mm/cm, cGy/Gy, F/C, mg-dL/mmol-L), z-score and min-max feature normalisation, and batch record harmonization with reporting.

- **Site Enrollment Management**
  - `federated/site_enrollment.py` — New module managing multi-site enrollment lifecycle: enrollment, data-readiness validation, compliance checks, activation, suspension, withdrawal, reactivation, quality-weighted site selection, round participation tracking, and audit trail.

- **Enhanced Digital Twins**
  - `physical_ai/digital_twin.py` — Added logistic and Gompertz tumor growth models alongside the existing exponential model, selectable via the `growth_model` parameter.
  - `physical_ai/digital_twin.py` — Added `simulate_immunotherapy_response()` with stochastic response modelling using `immuno_sensitivity`.
  - `physical_ai/digital_twin.py` — Added `simulate_combination_therapy()` for sequential multi-modality treatment simulation (chemo + radiation + immunotherapy).
  - `physical_ai/digital_twin.py` — Added `simulate_with_uncertainty()` for Monte-Carlo uncertainty quantification with configurable parameter noise.

- **Framework Detection**
  - `physical_ai/framework_detection.py` — New module that detects installed simulation frameworks (Isaac Lab, MuJoCo, Gazebo, PyBullet) and recommends a training/validation/deployment pipeline.

- **Surgical Task Definitions**
  - `physical_ai/surgical_tasks.py` — New module with standardised surgical task definitions for oncology procedures (needle biopsy, tissue resection, specimen handling, lymph node dissection, ablation) including clinical accuracy thresholds, reward structures, and domain randomization parameters.
  - `physical_ai/surgical_tasks.py` — `SurgicalTaskEvaluator` for evaluating robotic procedure performance against clinical thresholds.

- **Role-Based Access Control**
  - `privacy/access_control.py` — New module with six roles (coordinator, site admin, researcher, data engineer, auditor, patient) and thirteen granular permissions. All access decisions are logged for HIPAA/GDPR audit compliance.

- **Breach Response Protocol**
  - `privacy/breach_response.py` — New module for automated breach indicator detection, auto-escalation, and incident lifecycle management (detection, investigation, containment, resolution, regulatory reporting) compliant with HIPAA 72-hour and GDPR notification requirements.

- **FDA Submission Tracking**
  - `regulatory/fda_submission.py` — New module managing FDA submission packages for AI/ML-enabled medical devices. Supports Pre-Submission, De Novo, 510(k), PMA, and Breakthrough Device pathways with standard document checklists and readiness reporting.

- **New Tests**
  - `tests/test_data_harmonization.py` — Tests for vocabulary mapping, unit conversion, feature normalisation, and batch harmonization.
  - `tests/test_site_enrollment.py` — Tests for enrollment lifecycle, activation validation, quality-based auto-suspension, and round site selection.
  - Additional tests across existing files for FedProx, SCAFFOLD, convergence detection, growth models, combination therapy, uncertainty quantification, framework detection, surgical task evaluation, access control, and breach response.

### Changed

- `federated/coordinator.py` — Refactored to support pluggable aggregation strategies. Added `strategy`, `mu`, `convergence_window`, and `convergence_threshold` parameters. `RoundResult` now includes `elapsed_seconds` and `converged` fields.
- `federated/client.py` — Added `mu` parameter to `train_local()` for FedProx proximal term. Added `train_local_scaffold()` method. Added `get_training_history()` accessor.
- `physical_ai/digital_twin.py` — `TumorModel` gained `immuno_sensitivity`, `growth_model`, `carrying_capacity`, and `histology` fields. `PatientDigitalTwin` now supports combination therapy and uncertainty quantification.
- `examples/run_federation.py` — Added `--strategy` and `--mu` CLI arguments, site enrollment step, convergence display, and early stopping.
- `docker-compose.yml` — Added `fedprox` service profile for FedProx strategy simulation.
- `pyproject.toml` — Version bumped to 0.2.0. Development status upgraded to Beta. Added `digital-twins` and `surgical-robotics` keywords.
- `CITATION.cff` — Updated to v0.2.0 with expanded keywords and abstract.

## [0.1.0] - 2026-02-17

### Added

- **Federated Learning Framework**
  - `federated/coordinator.py` — Federation coordinator implementing FedAvg with configurable rounds, client weighting, and evaluation.
  - `federated/client.py` — Simulated hospital nodes with local training, parameter exchange, and sample counting.
  - `federated/model.py` — Numpy-based multilayer perceptron with forward/backward passes, parameter extraction/injection, and model evaluation.
  - `federated/secure_aggregation.py` — Mask-based secure aggregation protocol ensuring the coordinator only sees aggregate updates.
  - `federated/differential_privacy.py` — Gaussian mechanism with gradient clipping, calibrated noise injection, and privacy budget tracking.
  - `federated/data_ingestion.py` — Synthetic oncology data generation and IID/non-IID partitioning across sites.

- **Physical AI Integration**
  - `physical_ai/digital_twin.py` — Patient digital twin with tumor growth simulation, chemotherapy/radiation/immunotherapy response modeling, and feature vector generation for federated training.
  - `physical_ai/robotic_integration.py` — Surgical robot interface supporting procedure planning, execution simulation, and de-identified telemetry feature extraction.
  - `physical_ai/sensor_fusion.py` — Multi-modal clinical sensor fusion with real-time anomaly detection (Z-score based).
  - `physical_ai/simulation_bridge.py` — Cross-platform robot model conversion (URDF, MJCF, SDF, USD) with physics validation.

- **Privacy Infrastructure**
  - `privacy/phi_detector.py` — PHI detection for the 18 HIPAA identifiers using regex and field-name heuristics.
  - `privacy/deidentification.py` — De-identification pipeline with redact, hash, generalize, and replace strategies.
  - `privacy/consent_manager.py` — Patient consent and DUA management with audit trail.
  - `privacy/audit_logger.py` — HIPAA-compliant audit logging with integrity hashing and breach detection.
  - `privacy/dua_templates/` — Standard and multi-site Data Use Agreement templates.

- **Regulatory Compliance**
  - `regulatory/compliance_checker.py` — Automated HIPAA, GDPR, and FDA 21 CFR Part 11 compliance validation.
  - `regulatory/templates/` — IRB protocol and informed consent templates.

- **CI/CD**
  - GitHub Actions workflow with ruff linting and formatting checks across Python 3.10, 3.11, and 3.12.
  - Automated test suite execution.

- **Tests**
  - Comprehensive unit tests for all modules (`tests/`).
  - End-to-end tests verifying multi-site federation, DP integration, secure aggregation equivalence, digital twin federation, and privacy pipeline.

- **Examples & Documentation**
  - `examples/run_federation.py` — Complete federation simulation script with CLI arguments.
  - `examples/generate_synthetic_data.py` — Synthetic data generation and export to CSV.
  - `docs/federated_training_demo.ipynb` — Jupyter notebook demonstrating federated training, centralized comparison, DP impact, and digital twin integration.
  - `scripts/deploy.sh` — Docker-based deployment script.

- **Infrastructure**
  - `Dockerfile` and `docker-compose.yml` for containerized execution.
  - `environment.yml` for Conda-based reproducibility.
  - `pyproject.toml` with full project metadata and tool configuration.
  - `CITATION.cff` for academic citation.
  - MIT License.
