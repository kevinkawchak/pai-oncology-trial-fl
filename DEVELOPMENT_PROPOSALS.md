# Development Proposals for pai-oncology-trial-fl

**Version:** v0.9.2 baseline
**Date:** 2026-02-18
**Status:** Draft for senior engineering review
**Audience:** Platform architects, clinical informatics leads, regulatory engineers

---

## Overview

This document presents three future extension proposals for the pai-oncology-trial-fl platform. Each proposal follows the established development methodology used in Prompts 1-10 (see `prompts.md`): single-prompt delivery, production-quality Python modules with Enum/dataclass patterns, comprehensive test suites, progressive examples, regulatory alignment, and full ruff compliance.

The platform currently comprises:

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `federated/` | Core federated learning | FedAvg, FedProx, SCAFFOLD, secure aggregation, DP, site enrollment |
| `physical_ai/` | Physical AI integration | Digital twins, robotic integration, sensor fusion, surgical tasks |
| `privacy/` | HIPAA/GDPR compliance | PHI detection, de-identification, RBAC, breach response, DUA generator |
| `regulatory/` | Regulatory infrastructure | FDA submission tracking, IRB management, ICH-GCP, regulatory intelligence |
| `unification/` | Cross-framework bridge | Isaac-MuJoCo bridge, unified agent interface, model converter, validation |
| `digital-twins/` | Patient modeling domain | Patient physiology engine, treatment simulation, clinical integration |
| `tools/` | CLI tooling | DICOM inspector, dose calculator, site monitor, sim runner, deployment readiness |
| `clinical-analytics/` | Federated analytics | PK/PD engine, survival analysis, risk stratification, interoperability, DSMB reporting |
| `regulatory-submissions/` | Submission platform | eCTD compiler, compliance validator, document generator, submission analytics |
| `q1-2026-standards/` | Standards objectives | Model conversion pipeline, model registry, cross-platform benchmarking |

**Current scale:** 194 Python files, 97 test files, 31 example scripts, 12,400+ LOC in clinical-analytics and regulatory-submissions alone.

---

## Proposal 1: Real-Time Federated Safety Signal Detection

### Exact Prompt Text

```
Build real-time-safety/ for pai-oncology-trial-fl — a streaming safety signal detection system
that integrates with the existing federated learning, clinical-analytics, and regulatory
infrastructure to enable privacy-preserving, multi-site adverse event surveillance in oncology
clinical trials.

1. real-time-safety/ with README.md (architecture diagram in text, directory tree,
per-component descriptions with class names and algorithms, quick start code examples,
compliance alignment table covering ICH E2B(R3) / MedDRA / FDA FAERS / CIOMS / 21 CFR
Part 11 / EU ICSR, roadmap alignment notes):

* signal_detector.py (600-1000 LOC): Core signal detection engine. SignalType Enum
  (DISPROPORTIONALITY, TEMPORAL_CLUSTER, DOSE_RESPONSE, ORGAN_TOXICITY, INTERACTION,
  UNEXPECTED_AE), DetectionMethod Enum (PRR, ROR, BCPNN, MGPS, SEQUENTIAL_PROBABILITY),
  SignalStrength Enum (WEAK, MODERATE, STRONG, DEFINITE), SignalThreshold/DetectedSignal/
  SignalContext dataclasses. SafetySignalDetector class with proportional reporting ratio
  (PRR) with chi-squared test, reporting odds ratio (ROR) with confidence intervals,
  Bayesian Confidence Propagation Neural Network (BCPNN) information component (IC),
  Multi-item Gamma Poisson Shrinker (MGPS) empirical Bayes geometric mean (EBGM),
  sequential probability ratio test (SPRT) for continuous monitoring. Division-by-zero
  guards on all ratio computations. All methods return structured SignalResult with
  p-values, confidence intervals, and clinical interpretation.

* streaming_aggregator.py (500-800 LOC): Federated streaming aggregation for safety data.
  AggregationWindow Enum (HOURLY, DAILY, WEEKLY, CUMULATIVE), StreamStatus Enum.
  FederatedSafetyAggregator class implementing privacy-preserving aggregate AE counts
  across sites using secure summation (no raw patient data exchanged), sliding window
  aggregation with configurable decay, site-level contribution tracking with minimum
  site threshold (k-anonymity), latency-aware synchronization with timeout handling.
  Integration point with federated/secure_aggregation.py for mask-based aggregation of
  safety counts.

* adverse_event_classifier.py (500-800 LOC): MedDRA-based adverse event classification
  and grading. MedDRALevel Enum (SOC, HLGT, HLT, PT, LLT), CTCAEGrade Enum (0-5),
  CausalityAssessment Enum (DEFINITE, PROBABLE, POSSIBLE, UNLIKELY, UNRELATED, UNASSESSABLE),
  SeriousnessCriteria Enum (DEATH, LIFE_THREATENING, HOSPITALIZATION, DISABILITY,
  CONGENITAL_ANOMALY, MEDICALLY_SIGNIFICANT). AdverseEventClassifier with MedDRA SOC/PT
  mapping (synthetic dictionary covering 20+ oncology-relevant preferred terms), CTCAE
  v5.0 grade assignment with organ-system-specific thresholds, WHO-UMC causality
  assessment algorithm, seriousness determination per ICH E2A. 21 CFR Part 11 audit
  trail on all classification decisions.

* regulatory_reporter.py (500-800 LOC): Automated safety reporting to regulatory
  authorities. ReportType Enum (IND_SAFETY, SUSAR, DSUR, PBRER, CIOMS_I, CIOMS_II,
  FDA_MEDWATCH_3500A), ReportingTimeline Enum (IMMEDIATE_24H, EXPEDITED_7D, EXPEDITED_15D,
  PERIODIC_ANNUAL), SubmissionStatus Enum. RegulatoryReporter class generating structured
  safety reports per ICH E2B(R3) ICSR format, FDA MedWatch 3500A field mapping, EMA
  EudraVigilance ICSR submission structure, CIOMS I form generation, SUSAR 7-day and
  15-day timeline tracking with overdue detection (check OVERDUE before IMMINENT per
  v0.7.0 pattern), DSUR annual aggregate report compilation. Integration with
  regulatory-submissions/submission_orchestrator.py for workflow tracking.

* dashboard_engine.py (400-600 LOC): Safety monitoring dashboard data generation.
  DashboardPanel Enum (SIGNAL_OVERVIEW, AE_TIMELINE, ORGAN_HEATMAP, SITE_COMPARISON,
  DOSE_RESPONSE_CURVE, DSMB_SUMMARY), RefreshInterval Enum. SafetyDashboardEngine
  producing structured JSON dashboard payloads with signal trend charts (time series of
  PRR/ROR), AE incidence heatmaps by organ system and treatment arm, cross-site safety
  comparison (anonymized), cumulative toxicity curves, DSMB safety summary tables.
  SHA-256 integrity hashing on all dashboard snapshots. Integration with
  clinical-analytics/consortium_reporting.py for DSMB report enrichment.

* All output must be generated Markdown, structured data, or JSON — no external API calls.
  All clinical parameters must be bounded. MedDRA dictionary is synthetic (research use).

2. real-time-safety/examples-real-time-safety/: README.md + 6 progressive examples
   (200-400 LOC):
* 01_basic_signal_detection.py: Minimal example — synthetic AE table, PRR computation,
  signal interpretation.
* 02_streaming_safety_aggregation.py: Federated streaming — multi-site AE count
  aggregation with privacy preservation.
* 03_adverse_event_classification.py: MedDRA mapping, CTCAE grading, causality
  assessment for synthetic AE reports.
* 04_regulatory_safety_reporting.py: Generate IND safety report, SUSAR notification,
  CIOMS I form from classified AEs.
* 05_safety_dashboard.py: Dashboard payload generation with signal trends, organ
  heatmaps, cross-site comparisons.
* 06_full_safety_surveillance_pipeline.py: End-to-end combining all components with
  clinical-analytics and regulatory-submissions integration.

3. tests/test_real_time_safety/: __init__.py + one test file per component (5 files,
   150-300 LOC each, 12-22 tests per file, 70+ total). Must verify:
   - Signal detection math (PRR, ROR, BCPNN IC, EBGM) against hand-calculated values
   - Division-by-zero guards return sensible defaults
   - Streaming aggregation respects k-anonymity threshold
   - MedDRA classification covers all SOC categories
   - Regulatory timeline computation (7-day, 15-day, annual) with overdue detection
   - Dashboard integrity hashing
   - Integration test: clinical-analytics survival data -> signal detection -> regulatory report

4. Updates: ruff.toml per-file-ignores, README.md structure tree.
   Quality gates: Same as Prompts 9-10. All code passes ruff check + ruff format --check.
   RESEARCH USE ONLY in all modules. Structured logging, dataclass configs, Enum states.
   Tests pass with numpy/scipy/pytest/pyyaml only. No external services. Only Python 3.10+
   stdlib + numpy/scipy. Division guards return defaults. HMAC-SHA256 for all hashing.
   Audit log accessors return defensive copies. All clinical thresholds are configurable
   with bounded defaults.
```

### What It Accomplishes

- Deploys a five-module real-time safety signal detection system (`real-time-safety/`) that runs entirely in-process with no external service dependencies
- Implements five pharmacovigilance signal detection algorithms (PRR, ROR, BCPNN, MGPS, SPRT) with full statistical output including confidence intervals and p-values
- Provides federated streaming aggregation that enables multi-site adverse event surveillance without exchanging raw patient data, integrating with the existing `federated/secure_aggregation.py` mask-based protocol
- Delivers MedDRA-based adverse event classification with CTCAE v5.0 grading and WHO-UMC causality assessment, using a synthetic MedDRA dictionary covering 20+ oncology-relevant preferred terms
- Automates regulatory safety reporting across ICH E2B(R3) ICSR, FDA MedWatch 3500A, EMA EudraVigilance, and CIOMS I/II formats with 7-day, 15-day, and annual timeline tracking
- Generates structured dashboard payloads (JSON) for DSMB safety review including signal trend charts, organ-system heatmaps, cross-site comparisons, and cumulative toxicity curves
- Adds 6 progressive example scripts demonstrating the full safety surveillance pipeline from raw AE data through federated aggregation, signal detection, classification, regulatory reporting, and DSMB dashboard generation
- Adds 70+ tests with hand-calculated validation of signal detection statistics, k-anonymity verification, and end-to-end integration testing across clinical-analytics, regulatory-submissions, and the new safety module
- Integrates with `clinical-analytics/consortium_reporting.py` for DSMB report enrichment and `regulatory-submissions/submission_orchestrator.py` for regulatory workflow tracking

### Why It Is Error-Free

- **Security (S-01 through S-12 patterns):** No torch.load or np.load calls are needed in this module; all data is structured Python objects. All hashing uses HMAC-SHA256 with `os.urandom(32)` salts (not plain hashlib). Dashboard snapshot integrity uses SHA-256 via `hmac.new()`. No mutable audit log references are returned; all `get_audit_trail()` methods return `list()` copies.
- **Logic bugs (L-01 through L-14 patterns):** All ratio computations (PRR, ROR, EBGM) include explicit zero-denominator guards that return structured defaults (e.g., `SignalResult` with `signal_detected=False` and `reason="insufficient_data"`), not exceptions. Streaming aggregation windows are bounded by `MAX_WINDOW_SIZE` named constant. Timeline overdue detection checks `OVERDUE` before `IMMINENT` in the if/elif chain (per L-04/regulatory_tracker.py fix). All floating-point comparisons use `np.isclose()` or tolerance-based checks. No truthiness bugs: thresholds of 0 and 0.0 are handled with explicit `is None` checks.
- **Compliance (C-01 through C-35 patterns):** Every Python module includes "RESEARCH USE ONLY" in its docstring. All safety classification decisions are logged to a 21 CFR Part 11 audit trail. MedDRA dictionary is explicitly labeled as synthetic research data. All regulatory timeline references cite specific ICH/FDA guidance documents with dates.

### Key Files

| File | LOC | Description |
|------|-----|-------------|
| `real-time-safety/signal_detector.py` | 600-1000 | Core pharmacovigilance signal detection engine with PRR, ROR, BCPNN, MGPS, and SPRT algorithms |
| `real-time-safety/streaming_aggregator.py` | 500-800 | Federated privacy-preserving streaming aggregation with sliding windows and k-anonymity |
| `real-time-safety/adverse_event_classifier.py` | 500-800 | MedDRA classification, CTCAE v5.0 grading, WHO-UMC causality assessment |
| `real-time-safety/regulatory_reporter.py` | 500-800 | ICH E2B(R3) ICSR, FDA MedWatch, CIOMS I/II automated report generation |
| `real-time-safety/dashboard_engine.py` | 400-600 | DSMB dashboard payload generation with signal trends and organ heatmaps |
| `real-time-safety/README.md` | 200-300 | Architecture diagram, compliance alignment, integration documentation |
| `real-time-safety/examples-real-time-safety/01_basic_signal_detection.py` | 200-400 | Minimal PRR/ROR signal detection example |
| `real-time-safety/examples-real-time-safety/02_streaming_safety_aggregation.py` | 200-400 | Multi-site federated streaming aggregation example |
| `real-time-safety/examples-real-time-safety/03_adverse_event_classification.py` | 200-400 | MedDRA mapping and CTCAE grading example |
| `real-time-safety/examples-real-time-safety/04_regulatory_safety_reporting.py` | 200-400 | IND safety report and SUSAR notification generation |
| `real-time-safety/examples-real-time-safety/05_safety_dashboard.py` | 200-400 | Dashboard payload generation with heatmaps and trends |
| `real-time-safety/examples-real-time-safety/06_full_safety_surveillance_pipeline.py` | 200-400 | End-to-end pipeline with clinical-analytics and regulatory-submissions integration |
| `tests/test_real_time_safety/test_signal_detector.py` | 150-300 | Hand-calculated PRR/ROR/BCPNN/EBGM validation, division-by-zero guards |
| `tests/test_real_time_safety/test_streaming_aggregator.py` | 150-300 | Sliding window, k-anonymity threshold, privacy preservation tests |
| `tests/test_real_time_safety/test_adverse_event_classifier.py` | 150-300 | MedDRA mapping, CTCAE grading, causality assessment tests |
| `tests/test_real_time_safety/test_regulatory_reporter.py` | 150-300 | Timeline computation, overdue detection, report format tests |
| `tests/test_real_time_safety/test_dashboard_engine.py` | 150-300 | Dashboard payload structure, integrity hashing, integration tests |

**Estimated total:** 4,700-8,100 LOC (modules) + 1,200-2,400 LOC (examples) + 750-1,500 LOC (tests) = 6,650-12,000 LOC

---

## Proposal 2: Genomic Biomarker Integration Pipeline

### Exact Prompt Text

```
Build genomic-biomarkers/ for pai-oncology-trial-fl — a genomic biomarker integration pipeline
that extends clinical-analytics/ with variant-level genomic data processing, biomarker-driven
subgroup analysis, federated genome-wide association, and precision oncology workflows for
multi-site clinical trials.

1. genomic-biomarkers/ with README.md (architecture diagram in text, directory tree,
per-component descriptions with class names and algorithms, quick start code examples,
compliance alignment table covering ACMG/AMP variant classification / GA4GH / CLIA /
CAP / FDA companion diagnostics / ICH E18 / HIPAA genomic data, roadmap alignment notes):

* variant_processor.py (600-1000 LOC): Genomic variant processing engine. VariantType Enum
  (SNV, INDEL, CNV, FUSION, STRUCTURAL), VariantClassification Enum (PATHOGENIC,
  LIKELY_PATHOGENIC, VUS, LIKELY_BENIGN, BENIGN) per ACMG/AMP 2015 guidelines,
  ClinicalSignificance Enum (TIER_I_STRONG, TIER_I_POTENTIAL, TIER_II, TIER_III, TIER_IV)
  per AMP/ASCO/CAP 2017 somatic classification, VariantCall/GenomicPosition/
  AnnotationResult dataclasses. VariantProcessor class parsing VCF-like structured
  records (synthetic; no real VCF parser dependency), MAF-like mutation tables,
  allele frequency filtering (gnomAD-style thresholds), functional impact prediction
  scoring (SIFT/PolyPhen2 score interpretation), oncogenicity scoring per ClinGen/CGC/
  VICC 2022 somatic oncogenicity guidelines. Division-by-zero guards on allele frequency
  ratios. All genomic positions validated for chromosome bounds.

* biomarker_analyzer.py (500-800 LOC): Biomarker-driven subgroup analysis engine.
  BiomarkerType Enum (GENOMIC_MUTATION, GENE_EXPRESSION, PROTEIN_EXPRESSION, TMB, MSI,
  HRD, FUSION_GENE, COPY_NUMBER), ExpressionLevel Enum (HIGH, INTERMEDIATE, LOW,
  NEGATIVE), BiomarkerResult/SubgroupDefinition/EnrichmentResult dataclasses.
  BiomarkerAnalyzer class with tumor mutational burden (TMB) computation from variant
  counts and panel size (with zero-panel-size guard), microsatellite instability (MSI)
  scoring from repeat loci, homologous recombination deficiency (HRD) score from LOH/
  TAI/LST components, PD-L1 combined positive score (CPS) computation, biomarker-defined
  subgroup partitioning for stratified analysis, treatment-biomarker interaction testing
  via two-way ANOVA-style F-statistic. Integration with clinical-analytics/
  risk_stratification.py for biomarker-enriched risk scoring.

* federated_gwas.py (500-800 LOC): Privacy-preserving federated genome-wide association
  analysis. GWASModel Enum (ADDITIVE, DOMINANT, RECESSIVE), CorrectionMethod Enum
  (BONFERRONI, FDR_BH, GENOMIC_CONTROL), ManhattanPlotData/QQPlotData/GWASResult
  dataclasses. FederatedGWASEngine class implementing federated logistic regression for
  single-variant association (site-local sufficient statistics aggregation, no individual
  genotypes shared), minor allele frequency (MAF) filtering across sites with privacy-
  preserving frequency aggregation, multiple testing correction (Bonferroni, Benjamini-
  Hochberg FDR), genomic inflation factor (lambda_gc) computation, Manhattan plot and
  Q-Q plot data generation. Integration with federated/secure_aggregation.py for
  privacy-preserving statistic summation. All p-value computations use scipy.stats with
  explicit edge-case handling for zero-variance sites.

* precision_oncology.py (500-800 LOC): Precision oncology treatment matching engine.
  EvidenceLevel Enum (LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_R1, LEVEL_R2) per
  OncoKB evidence levels, TherapyCategory Enum (TARGETED, IMMUNOTHERAPY, CHEMOTHERAPY,
  HORMONE, COMBINATION), MatchConfidence Enum (HIGH, MODERATE, LOW, INSUFFICIENT).
  TreatmentMatch/DrugGenomicAssociation/ClinicalTrialMatch dataclasses.
  PrecisionOncologyEngine with synthetic oncology knowledge base covering 30+ gene-drug
  associations (EGFR-erlotinib, ALK-crizotinib, BRAF-vemurafenib, HER2-trastuzumab,
  BRCA-olaparib, etc.), variant-to-therapy matching with evidence level assignment,
  resistance mutation detection (e.g., EGFR T790M, ALK G1202R), clinical trial
  eligibility screening based on genomic profile, companion diagnostic requirement
  flagging per FDA-approved CDx list. All knowledge base entries are synthetic research
  data with explicit "RESEARCH USE ONLY — not for clinical decision-making" labeling.

* genomic_reporting.py (400-600 LOC): Genomic reporting and data export engine.
  ReportFormat Enum (CLINICAL_GENOMIC_REPORT, RESEARCH_SUMMARY, REGULATORY_APPENDIX,
  GA4GH_VRS), ReportSection Enum. GenomicReportGenerator producing structured genomic
  test reports with variant tables, biomarker summary, treatment recommendations (with
  evidence levels), clinical trial matches, and methodology description. GA4GH Variant
  Representation Specification (VRS) export for federated variant sharing. CDISC
  SDTM PGx domain-compatible data structures. SHA-256 integrity hashing on all
  genomic reports. Integration with regulatory-submissions/document_generator.py for
  regulatory appendix generation.

* All output must be generated Markdown or structured data — no external API calls.
  No real genomic databases or VCF parsing libraries required. Synthetic data only.

2. genomic-biomarkers/examples-genomic-biomarkers/: README.md + 6 progressive examples
   (200-400 LOC):
* 01_basic_variant_processing.py: Minimal — parse synthetic VCF records, classify
  variants by ACMG/AMP, filter by allele frequency.
* 02_biomarker_analysis.py: TMB computation, MSI scoring, PD-L1 CPS, subgroup
  definition for a synthetic cohort.
* 03_federated_gwas.py: Multi-site federated GWAS with privacy-preserving MAF
  aggregation, Manhattan plot data, multiple testing correction.
* 04_precision_oncology_matching.py: Variant-to-therapy matching, resistance detection,
  clinical trial eligibility for a synthetic patient profile.
* 05_genomic_reporting.py: Clinical genomic report generation, GA4GH VRS export,
  SDTM PGx output.
* 06_full_genomic_biomarker_pipeline.py: End-to-end combining all components with
  clinical-analytics integration (survival by biomarker subgroup) and regulatory-
  submissions integration (genomic appendix for 510(k)).

3. tests/test_genomic_biomarkers/: __init__.py + one test file per component (5 files,
   150-300 LOC each, 12-22 tests per file, 70+ total). Must verify:
   - Variant classification matches ACMG/AMP criteria for known synthetic variants
   - TMB computation handles zero panel size and zero variant count
   - Federated GWAS produces consistent results vs. simulated centralized analysis
   - Multiple testing correction reduces false positives below nominal alpha
   - Treatment matching returns correct evidence levels for known gene-drug pairs
   - Report generation includes all required sections with integrity hashing
   - Integration test: variant processing -> biomarker analysis -> survival stratification
     -> regulatory appendix

4. Updates: ruff.toml per-file-ignores, README.md structure tree.
   Quality gates: Same as Prompts 9-10. All code passes ruff check + ruff format --check.
   RESEARCH USE ONLY in all modules. Structured logging, dataclass configs, Enum states.
   Tests pass with numpy/scipy/pytest/pyyaml only. No external services. Only Python 3.10+
   stdlib + numpy/scipy. Genomic knowledge base is synthetic. Division guards return
   defaults. HMAC-SHA256 for all hashing. Audit log accessors return defensive copies.
```

### What It Accomplishes

- Deploys a five-module genomic biomarker integration pipeline (`genomic-biomarkers/`) extending the platform into precision oncology with federated genomic analysis capabilities
- Implements VCF-like and MAF-like synthetic variant processing with ACMG/AMP 2015 germline classification and AMP/ASCO/CAP 2017 somatic variant tiering, enabling structured genomic data to flow through the analytics pipeline
- Provides biomarker-driven subgroup analysis computing TMB, MSI, HRD, and PD-L1 CPS scores from synthetic data, with integration into `clinical-analytics/risk_stratification.py` for biomarker-enriched risk scoring
- Delivers federated genome-wide association analysis that computes per-site sufficient statistics and aggregates them via `federated/secure_aggregation.py`, producing Manhattan plots and Q-Q plots without sharing individual-level genotype data
- Implements a precision oncology treatment matching engine with a synthetic knowledge base covering 30+ gene-drug associations, resistance mutation detection, and clinical trial eligibility screening based on genomic profiles
- Generates structured genomic reports compatible with GA4GH Variant Representation Specification (VRS) and CDISC SDTM PGx domains, with integration into `regulatory-submissions/document_generator.py` for regulatory appendix generation
- Adds 6 progressive example scripts from basic variant processing through full end-to-end genomic biomarker pipelines with survival stratification by biomarker subgroup
- Adds 70+ tests verifying ACMG/AMP classification logic, TMB edge cases, federated GWAS consistency versus centralized baselines, and cross-module integration
- Bridges the gap between genomic profiling and federated trial analytics, enabling biomarker-stratified treatment effect estimation across multiple sites without centralizing sensitive genomic data

### Why It Is Error-Free

- **Security (S-01 through S-12 patterns):** No torch.load or np.load calls are needed; variant data is represented as Python dataclasses, not serialized model weights. All report integrity hashing uses HMAC-SHA256 with `os.urandom(32)`. No genomic data is stored in plaintext logs; patient identifiers are HMAC-hashed before any cross-site operation. All audit trail accessors return `list()` copies.
- **Logic bugs (L-01 through L-14 patterns):** TMB computation guards against zero panel size (`if panel_size_mb <= 0: return TMBResult(tmb=0.0, quality="insufficient_panel")`). Allele frequency ratios guard against zero total allele counts. Federated GWAS handles zero-variance sites by excluding them from association testing rather than producing NaN. Manhattan plot p-value log-transformation uses `max(p_value, 1e-300)` to prevent log(0). Multiple testing correction handles the edge case of zero tests. All if/elif chains for evidence levels are exhaustive with a terminal `else` clause. Floating-point p-value comparisons use tolerance-based checks.
- **Compliance (C-01 through C-35 patterns):** Every module includes "RESEARCH USE ONLY" in its docstring. The synthetic knowledge base is explicitly labeled as not suitable for clinical decision-making at the class level, in every example script header, and in the README. All variant classification decisions are audit-trailed. Genomic data handling references HIPAA genetic information provisions and GINA requirements.

### Key Files

| File | LOC | Description |
|------|-----|-------------|
| `genomic-biomarkers/variant_processor.py` | 600-1000 | VCF/MAF variant processing with ACMG/AMP and AMP/ASCO/CAP classification |
| `genomic-biomarkers/biomarker_analyzer.py` | 500-800 | TMB, MSI, HRD, PD-L1 CPS computation and biomarker-driven subgroup analysis |
| `genomic-biomarkers/federated_gwas.py` | 500-800 | Privacy-preserving federated GWAS with sufficient statistics aggregation |
| `genomic-biomarkers/precision_oncology.py` | 500-800 | Variant-to-therapy matching with 30+ synthetic gene-drug associations |
| `genomic-biomarkers/genomic_reporting.py` | 400-600 | GA4GH VRS export, CDISC SDTM PGx, clinical genomic report generation |
| `genomic-biomarkers/README.md` | 200-300 | Architecture, compliance alignment (ACMG/GA4GH/CLIA/CAP/FDA CDx/ICH E18) |
| `genomic-biomarkers/examples-genomic-biomarkers/01_basic_variant_processing.py` | 200-400 | Synthetic VCF parsing, ACMG/AMP classification, allele frequency filtering |
| `genomic-biomarkers/examples-genomic-biomarkers/02_biomarker_analysis.py` | 200-400 | TMB, MSI, PD-L1 CPS computation for synthetic cohort |
| `genomic-biomarkers/examples-genomic-biomarkers/03_federated_gwas.py` | 200-400 | Multi-site federated GWAS with Manhattan plot data |
| `genomic-biomarkers/examples-genomic-biomarkers/04_precision_oncology_matching.py` | 200-400 | Gene-drug matching, resistance detection, trial eligibility |
| `genomic-biomarkers/examples-genomic-biomarkers/05_genomic_reporting.py` | 200-400 | Clinical genomic report and GA4GH VRS export |
| `genomic-biomarkers/examples-genomic-biomarkers/06_full_genomic_biomarker_pipeline.py` | 200-400 | End-to-end with clinical-analytics survival stratification and regulatory appendix |
| `tests/test_genomic_biomarkers/test_variant_processor.py` | 150-300 | ACMG/AMP classification validation, allele frequency edge cases |
| `tests/test_genomic_biomarkers/test_biomarker_analyzer.py` | 150-300 | TMB zero-panel guard, MSI scoring, subgroup partitioning |
| `tests/test_genomic_biomarkers/test_federated_gwas.py` | 150-300 | Federated vs. centralized consistency, multiple testing correction |
| `tests/test_genomic_biomarkers/test_precision_oncology.py` | 150-300 | Gene-drug matching accuracy, resistance detection, evidence levels |
| `tests/test_genomic_biomarkers/test_genomic_reporting.py` | 150-300 | Report structure, GA4GH VRS format, integrity hashing |

**Estimated total:** 4,500-7,600 LOC (modules) + 1,200-2,400 LOC (examples) + 750-1,500 LOC (tests) = 6,450-11,500 LOC

---

## Proposal 3: Adaptive Trial Design Engine

### Exact Prompt Text

```
Build adaptive-designs/ for pai-oncology-trial-fl — a Bayesian adaptive clinical trial design
engine that integrates with the existing federated learning, clinical-analytics, and regulatory
infrastructure to support response-adaptive randomization, predictive probability of success,
interim analysis automation, and platform trial designs for multi-site oncology trials.

1. adaptive-designs/ with README.md (architecture diagram in text, directory tree,
per-component descriptions with class names and algorithms, quick start code examples,
compliance alignment table covering FDA Adaptive Design Guidance (2019) / ICH E9(R1)
estimands / EMA Reflection Paper on Adaptive Designs / ICH E20 / 21 CFR Part 11 /
CONSORT-Adaptive extension, roadmap alignment notes):

* bayesian_engine.py (600-1000 LOC): Core Bayesian inference engine for adaptive trials.
  PriorType Enum (NON_INFORMATIVE, WEAKLY_INFORMATIVE, INFORMATIVE, HISTORICAL,
  COMMENSURATE), PosteriorSummary/BayesianModel/TrialDesignParameters dataclasses.
  BayesianTrialEngine class with Beta-Binomial conjugate model for binary endpoints
  (response rate, ORR), Normal-Normal conjugate model for continuous endpoints (tumor
  shrinkage, biomarker change), Gamma-Poisson model for count/rate endpoints (AE rates,
  event counts), posterior probability computation P(treatment > control | data),
  Bayesian predictive probability of success (PPoS) via Monte Carlo simulation of
  future enrollment, historical borrowing via power prior with discount factor a0,
  effective sample size (ESS) computation for informative priors. Division-by-zero
  guards on all posterior parameter updates. All random sampling uses np.random with
  configurable seed for reproducibility.

* adaptive_randomizer.py (500-800 LOC): Response-adaptive randomization engine.
  RandomizationMethod Enum (EQUAL, BLOCK, STRATIFIED, RESPONSE_ADAPTIVE, BAYESIAN_ADAPTIVE,
  THOMPSON_SAMPLING, MINIMUM_ALLOCATION), AllocationRatio/RandomizationResult/
  StrataDefinition dataclasses. AdaptiveRandomizer class with fixed block randomization
  (permuted blocks with variable block sizes), stratified randomization by prognostic
  factors, response-adaptive randomization (RAR) with configurable tuning parameter,
  Thompson sampling for multi-arm trials (Beta posterior per arm), minimum allocation
  ratio enforcement (no arm drops below configurable floor, default 10%), allocation
  ratio capping to prevent extreme imbalance. Integration with clinical-analytics/
  risk_stratification.py for stratification factor definition. Randomization audit trail
  per 21 CFR Part 11 with sealed envelopes pattern (allocation concealment logging).

* interim_analyzer.py (500-800 LOC): Automated interim analysis engine.
  InterimType Enum (EFFICACY, FUTILITY, SAFETY, SAMPLE_SIZE_REESTIMATION, ADAPTATION),
  BoundaryType Enum (OBRIEN_FLEMING, POCOCK, HAYBITTLE_PETO, GAMMA_FAMILY, PREDICTIVE),
  StoppingDecision Enum (CONTINUE, STOP_EFFICACY, STOP_FUTILITY, STOP_SAFETY,
  MODIFY_ALLOCATION, INCREASE_SAMPLE_SIZE). InterimAnalysisEngine class with group
  sequential boundaries (O'Brien-Fleming, Pocock, Haybittle-Peto spending functions),
  alpha spending function (Lan-DeMets) for flexible interim timing, conditional power
  computation at each interim look, predictive probability of success for futility
  monitoring, sample size re-estimation based on observed effect size (promising zone
  approach), information fraction computation from observed events. Integration with
  clinical-analytics/survival_analysis.py for event-driven interim triggers. All
  boundary computations validated against known spending function values. Division
  guards on information fraction when zero events observed.

* platform_trial.py (500-800 LOC): Platform and master protocol trial design engine.
  PlatformComponent Enum (BACKBONE, EXPERIMENTAL_ARM, BIOMARKER_COHORT, SHARED_CONTROL),
  ArmStatus Enum (ENROLLING, SUSPENDED, GRADUATED_EFFICACY, GRADUATED_FUTILITY, CLOSED),
  GraduationCriteria/PlatformArm/MasterProtocol dataclasses. PlatformTrialEngine class
  supporting multi-arm multi-stage (MAMS) designs with shared control, arm addition and
  removal during trial conduct, biomarker-driven cohort assignment (integration with
  genomic-biomarkers/ if available, graceful degradation if not), arm-specific interim
  analysis with multiplicity adjustment (Dunnett or Bonferroni), operational seamlessness
  (new arms inherit infrastructure without protocol restart). Integration with
  federated/coordinator.py for multi-site arm enrollment coordination.

* design_simulator.py (400-600 LOC): Trial design simulation and operating characteristics
  engine. SimulationScenario/OperatingCharacteristics/SimulationResult dataclasses.
  DesignSimulator class with Monte Carlo simulation of complete trial trajectories
  under null and alternative hypotheses, type I error rate estimation under global null,
  power curves across a range of treatment effects, expected sample size computation
  under various scenarios, probability of early stopping (for efficacy and futility),
  comparison of fixed vs. adaptive designs under identical assumptions. Configurable
  number of simulations (default 10,000) with progress tracking. All simulations
  bounded by MAX_SIMULATIONS constant. np.random.seed for reproducibility.

* All output must be generated Markdown or structured data — no external API calls.
  All statistical computations use scipy.stats. All random sampling is seeded.

2. adaptive-designs/examples-adaptive-designs/: README.md + 6 progressive examples
   (200-400 LOC):
* 01_bayesian_response_rate.py: Minimal — Beta-Binomial model for ORR, posterior
  probability P(p_treatment > p_control | data), prior sensitivity analysis.
* 02_response_adaptive_randomization.py: RAR with Thompson sampling in a three-arm
  trial, allocation ratio evolution over time.
* 03_group_sequential_interim.py: O'Brien-Fleming boundaries for a two-arm superiority
  trial with three interim looks, alpha spending visualization.
* 04_platform_trial_design.py: MAMS platform trial with shared control, two experimental
  arms, biomarker cohort, arm graduation criteria.
* 05_design_simulation.py: Operating characteristics comparison of fixed vs. adaptive
  two-arm trial under 5 treatment effect scenarios.
* 06_full_adaptive_trial_pipeline.py: End-to-end combining all components with
  clinical-analytics survival data, federated multi-site enrollment, and regulatory-
  submissions adaptive design supplement generation.

3. tests/test_adaptive_designs/: __init__.py + one test file per component (5 files,
   150-300 LOC each, 12-22 tests per file, 70+ total). Must verify:
   - Beta-Binomial posterior matches analytical solution for known data
   - Normal-Normal posterior mean and variance match conjugate formulas
   - Thompson sampling allocation converges to superior arm over many rounds
   - O'Brien-Fleming boundaries match published tables (Jennison & Turnbull)
   - Alpha spending function cumulative spend equals nominal alpha at final analysis
   - Platform trial arm graduation triggers at correct posterior threshold
   - Simulation type I error under null is within Monte Carlo SE of nominal alpha
   - Integration test: federated enrollment -> adaptive randomization -> interim analysis
     -> regulatory adaptive design report

4. Updates: ruff.toml per-file-ignores, README.md structure tree.
   Quality gates: Same as Prompts 9-10. All code passes ruff check + ruff format --check.
   RESEARCH USE ONLY in all modules. Structured logging, dataclass configs, Enum states.
   Tests pass with numpy/scipy/pytest/pyyaml only. No external services. Only Python 3.10+
   stdlib + numpy/scipy. All random sampling seeded. Division guards return defaults.
   HMAC-SHA256 for all hashing. Audit log accessors return defensive copies. Boundary
   computations validated against published statistical tables.
```

### What It Accomplishes

- Deploys a five-module adaptive trial design engine (`adaptive-designs/`) providing Bayesian adaptive methodology infrastructure for multi-site oncology trials
- Implements Beta-Binomial, Normal-Normal, and Gamma-Poisson conjugate models with posterior probability computation, predictive probability of success (PPoS), and historical borrowing via power priors with configurable discount factors
- Provides response-adaptive randomization including Thompson sampling for multi-arm trials, with minimum allocation ratio enforcement (no arm below 10% default) and stratified randomization integrating with `clinical-analytics/risk_stratification.py`
- Delivers automated interim analysis with group sequential boundaries (O'Brien-Fleming, Pocock, Haybittle-Peto), Lan-DeMets alpha spending functions for flexible timing, conditional power, and sample size re-estimation via the promising zone approach
- Implements platform trial (MAMS) designs with shared control arms, dynamic arm addition/removal, biomarker-driven cohort assignment, and arm-specific interim analysis with multiplicity adjustment
- Provides Monte Carlo simulation of trial operating characteristics including type I error estimation, power curves, expected sample size, and comparison of fixed versus adaptive designs under matched assumptions
- Adds 6 progressive example scripts from basic Bayesian response rate modeling through full adaptive trial pipelines with federated enrollment and regulatory supplement generation
- Adds 70+ tests with analytical validation against conjugate posterior formulas, published O'Brien-Fleming boundary tables, and Monte Carlo type I error calibration
- Integrates with `clinical-analytics/survival_analysis.py` for event-driven interim triggers, `federated/coordinator.py` for multi-site enrollment coordination, and `regulatory-submissions/document_generator.py` for adaptive design regulatory supplements per FDA guidance

### Why It Is Error-Free

- **Security (S-01 through S-12 patterns):** No torch.load or np.load calls; all models are analytical conjugate updates, not serialized neural network weights. Randomization audit trails use HMAC-SHA256 with `os.urandom(32)` for allocation concealment integrity. All `get_audit_trail()` methods return `list()` copies. No patient identifiers appear in simulation logs.
- **Logic bugs (L-01 through L-14 patterns):** All posterior parameter updates guard against zero observations (`if n_observations == 0: return prior_parameters`). Thompson sampling handles zero-success arms by sampling from the prior, not dividing by zero. Information fraction computation guards against zero total events. Alpha spending function output is clamped to `[0.0, alpha]` to prevent overspending from floating-point drift. Allocation ratios are normalized with an explicit `sum(ratios)` zero check. Minimum allocation enforcement uses `max(ratio, floor)` before normalization, ensuring no arm is silently dropped. All Monte Carlo simulations use bounded `MAX_SIMULATIONS` constant. The if/elif chain in `StoppingDecision` evaluation checks `STOP_SAFETY` first (highest priority), then `STOP_EFFICACY`, then `STOP_FUTILITY`, then `MODIFY_ALLOCATION`, ensuring safety decisions are never shadowed.
- **Compliance (C-01 through C-35 patterns):** Every module includes "RESEARCH USE ONLY" in its docstring. All randomization decisions are 21 CFR Part 11 audit-trailed with timestamps and allocation concealment. Interim analysis decisions reference FDA Adaptive Design Guidance (2019) section numbers. Platform trial methodology references ICH E20 (adaptive designs in confirmatory trials). Statistical boundary computations cite Jennison & Turnbull (2000) and Lan & DeMets (1983).

### Key Files

| File | LOC | Description |
|------|-----|-------------|
| `adaptive-designs/bayesian_engine.py` | 600-1000 | Beta-Binomial, Normal-Normal, Gamma-Poisson conjugate models, PPoS, power priors |
| `adaptive-designs/adaptive_randomizer.py` | 500-800 | Response-adaptive randomization, Thompson sampling, block/stratified randomization |
| `adaptive-designs/interim_analyzer.py` | 500-800 | Group sequential boundaries, alpha spending, conditional power, sample size re-estimation |
| `adaptive-designs/platform_trial.py` | 500-800 | MAMS platform trials, shared control, arm graduation, multiplicity adjustment |
| `adaptive-designs/design_simulator.py` | 400-600 | Monte Carlo operating characteristics, type I error, power curves, fixed vs. adaptive |
| `adaptive-designs/README.md` | 200-300 | Architecture, compliance alignment (FDA/ICH E9(R1)/EMA/CONSORT-Adaptive) |
| `adaptive-designs/examples-adaptive-designs/01_bayesian_response_rate.py` | 200-400 | Beta-Binomial ORR model, posterior probability, prior sensitivity |
| `adaptive-designs/examples-adaptive-designs/02_response_adaptive_randomization.py` | 200-400 | Thompson sampling three-arm RAR with allocation ratio evolution |
| `adaptive-designs/examples-adaptive-designs/03_group_sequential_interim.py` | 200-400 | O'Brien-Fleming boundaries, alpha spending visualization |
| `adaptive-designs/examples-adaptive-designs/04_platform_trial_design.py` | 200-400 | MAMS shared control with biomarker cohort and arm graduation |
| `adaptive-designs/examples-adaptive-designs/05_design_simulation.py` | 200-400 | Fixed vs. adaptive operating characteristics comparison |
| `adaptive-designs/examples-adaptive-designs/06_full_adaptive_trial_pipeline.py` | 200-400 | End-to-end with federated enrollment, survival interim, regulatory supplement |
| `tests/test_adaptive_designs/test_bayesian_engine.py` | 150-300 | Conjugate posterior analytical validation, PPoS convergence, power prior ESS |
| `tests/test_adaptive_designs/test_adaptive_randomizer.py` | 150-300 | Thompson sampling convergence, minimum allocation enforcement, block balance |
| `tests/test_adaptive_designs/test_interim_analyzer.py` | 150-300 | O'Brien-Fleming boundary tables, alpha spending cumulative sum, conditional power |
| `tests/test_adaptive_designs/test_platform_trial.py` | 150-300 | Arm graduation thresholds, shared control allocation, multiplicity adjustment |
| `tests/test_adaptive_designs/test_design_simulator.py` | 150-300 | Type I error calibration, power monotonicity, expected sample size bounds |

**Estimated total:** 4,500-7,600 LOC (modules) + 1,200-2,400 LOC (examples) + 750-1,500 LOC (tests) = 6,450-11,500 LOC

---

## Comparative Feature Table

| Dimension | Proposal 1: Real-Time Safety | Proposal 2: Genomic Biomarkers | Proposal 3: Adaptive Designs |
|-----------|------------------------------|-------------------------------|------------------------------|
| **Estimated LOC** | 6,650-12,000 | 6,450-11,500 | 6,450-11,500 |
| **New Modules** | 5 (signal_detector, streaming_aggregator, adverse_event_classifier, regulatory_reporter, dashboard_engine) | 5 (variant_processor, biomarker_analyzer, federated_gwas, precision_oncology, genomic_reporting) | 5 (bayesian_engine, adaptive_randomizer, interim_analyzer, platform_trial, design_simulator) |
| **New Tests** | 70+ across 5 test files | 70+ across 5 test files | 70+ across 5 test files |
| **New Examples** | 6 progressive scripts | 6 progressive scripts | 6 progressive scripts |
| **Integration: federated/** | secure_aggregation.py (streaming AE counts) | secure_aggregation.py (GWAS statistic summation) | coordinator.py (multi-site enrollment) |
| **Integration: clinical-analytics/** | consortium_reporting.py (DSMB enrichment), survival_analysis.py (safety endpoints) | risk_stratification.py (biomarker-enriched scoring), survival_analysis.py (biomarker-stratified KM) | survival_analysis.py (event-driven interim), risk_stratification.py (stratified randomization) |
| **Integration: regulatory-submissions/** | submission_orchestrator.py (safety report workflow) | document_generator.py (genomic regulatory appendix) | document_generator.py (adaptive design supplement) |
| **Integration: other modules** | privacy/ (PHI in AE reports), regulatory/ (FDA FAERS alignment) | genomic-biomarkers ←→ digital-twins/ (biomarker simulation) | federated/site_enrollment.py (enrollment coordination) |
| **Primary Regulatory Standards** | ICH E2B(R3), ICH E2A, FDA FAERS, CIOMS I/II, MedDRA, 21 CFR Part 11 | ACMG/AMP 2015, AMP/ASCO/CAP 2017, GA4GH VRS, CLIA/CAP, FDA CDx, ICH E18 | FDA Adaptive Design Guidance (2019), ICH E9(R1), ICH E20, EMA Reflection Paper, CONSORT-Adaptive |
| **Clinical Domain** | Pharmacovigilance, DSMB safety monitoring | Precision oncology, molecular tumor boards | Adaptive trial methodology, biostatistics |
| **Statistical Methods** | PRR, ROR, BCPNN, MGPS, SPRT | Logistic regression (GWAS), F-statistic (interaction), chi-squared (TMB) | Beta-Binomial, Normal-Normal, Gamma-Poisson, O'Brien-Fleming, Lan-DeMets |
| **Data Standards** | MedDRA SOC/PT, CTCAE v5.0, ICH E2B(R3) ICSR | VCF, MAF, GA4GH VRS, CDISC SDTM PGx | CDISC ADaM, CONSORT-Adaptive |
| **Key Dependency** | numpy, scipy (scipy.stats.chi2, scipy.stats.norm) | numpy, scipy (scipy.stats.logistic, scipy.stats.chi2) | numpy, scipy (scipy.stats.beta, scipy.stats.norm, scipy.special) |

---

## Strategic Impact Matrix

Each cell is rated on a 5-point scale: **1** = Minimal, **2** = Low, **3** = Moderate, **4** = High, **5** = Transformative.

| Impact Dimension | Proposal 1: Real-Time Safety | Proposal 2: Genomic Biomarkers | Proposal 3: Adaptive Designs | Rationale |
|------------------|------------------------------|-------------------------------|------------------------------|-----------|
| **Patient Safety** | **5 — Transformative** | **3 — Moderate** | **4 — High** | Proposal 1 directly detects safety signals in real time, enabling earlier identification of serious adverse events. Proposal 3 supports futility stopping that prevents continued enrollment in ineffective arms. Proposal 2 improves treatment matching but does not directly monitor ongoing safety. |
| **Regulatory Readiness** | **5 — Transformative** | **4 — High** | **5 — Transformative** | Proposal 1 automates ICH E2B(R3) ICSR generation and FDA FAERS-aligned reporting, directly reducing regulatory submission burden. Proposal 3 implements FDA Adaptive Design Guidance (2019) methodology required for adaptive trial INDs. Proposal 2 aligns with FDA companion diagnostic requirements and GA4GH standards but is less directly tied to submission timelines. |
| **Scientific Rigor** | **4 — High** | **5 — Transformative** | **5 — Transformative** | Proposal 2 enables molecular-level treatment effect heterogeneity analysis via federated GWAS and biomarker-stratified outcomes, advancing precision oncology. Proposal 3 provides formally validated Bayesian adaptive methods with operating characteristics verified by simulation. Proposal 1 applies established pharmacovigilance statistics (PRR, BCPNN, MGPS) to the federated setting. |
| **Engineering Quality** | **4 — High** | **4 — High** | **4 — High** | All three proposals follow the same Enum/dataclass/structured-logging architecture, use the same quality gates (ruff, RESEARCH USE ONLY, division guards, HMAC-SHA256, defensive copies), and include 70+ tests per module. Each integrates with 2-3 existing modules, exercising the platform's composability. |
| **Market Differentiation** | **5 — Transformative** | **5 — Transformative** | **4 — High** | No existing open-source platform combines federated learning with real-time pharmacovigilance (Proposal 1) or federated GWAS for oncology trials (Proposal 2). Proposal 3 addresses adaptive designs, where commercial tools (FACTS, EAST, ADDPLAN) exist but lack federated multi-site integration. All three proposals differentiate pai-oncology-trial-fl from single-capability alternatives. |
| **Weighted Total (equal weights)** | **23 / 25** | **21 / 25** | **22 / 25** | Proposal 1 scores highest due to direct patient safety and regulatory impact. Proposal 3 follows closely due to regulatory readiness and scientific rigor. Proposal 2 is strongest on scientific differentiation and precision oncology. |

---

## Implementation Priority Recommendation

Based on the strategic impact analysis, dependency ordering, and integration complexity:

**Phase 1 — Proposal 1 (Real-Time Federated Safety Signal Detection)**
- Highest patient safety impact (transformative)
- Immediate regulatory value (ICH E2B(R3) ICSR generation)
- Lightweight integration footprint (extends existing streaming patterns)
- Unblocks DSMB monitoring infrastructure needed by Proposals 2 and 3

**Phase 2 — Proposal 3 (Adaptive Trial Design Engine)**
- Required for modern oncology trial designs (nearly all Phase II/III oncology trials use adaptive features)
- Bayesian engine provides statistical infrastructure reusable by Proposals 1 and 2
- Interim analysis automation integrates with Phase 1 safety monitoring for joint efficacy-safety decisions

**Phase 3 — Proposal 2 (Genomic Biomarker Integration Pipeline)**
- Benefits from Proposal 3's stratified randomization and subgroup analysis infrastructure
- Federated GWAS methodology can leverage Proposal 3's Bayesian multiple testing framework
- Precision oncology matching is enhanced by Proposal 1's safety signal data (resistance mutations correlating with adverse events)

Each phase follows the established single-prompt delivery methodology (see `prompts.md`) and can be delivered independently. Cross-proposal integration points are designed to be optional, using graceful degradation patterns (conditional imports with `HAS_X` flags) so that each module functions standalone.

---

## Appendix: Error Prevention Checklist

Derived from the v0.7.0 security audit (`SECURITY_AUDIT.md`) findings, every new module in all three proposals must satisfy:

| Audit Finding | Prevention Measure | Applies To |
|---------------|-------------------|------------|
| S-01 to S-05: `torch.load` unsafe deserialization | No torch.load calls in any proposal (no neural network serialization) | All |
| S-06: `np.load` without `allow_pickle=False` | No np.load calls (all data is structured Python objects) | All |
| S-07 to S-09: Weak hashing without HMAC | All hashing uses `hmac.new(key, msg, hashlib.sha256)` with `os.urandom(32)` | All |
| S-10 to S-11: Mutable audit log references | All `get_*_log()` and `get_*_trail()` return `list()` copies | All |
| S-12: Weak pseudonymization salt | All salts generated via `os.urandom(32)`, never hardcoded | All |
| L-01 to L-03: Division by zero | Explicit zero-denominator guards returning structured defaults, not exceptions | All |
| L-04, L-07: Truthiness bugs | `is None` checks for all optional numeric parameters; `< 0.0` not `<= 0.0` for validity | All |
| L-05: Dead data (hardcoded values replacing variables) | Code review checklist item: verify all dataclass fields use variable references | All |
| L-08: Missing null check on optional methods | Explicit None checks with descriptive `RuntimeError` for missing optional dependencies | All |
| L-13: Undocumented assumptions | All algorithmic assumptions documented in docstrings with citations | All |
| C-01 to C-35: Missing RESEARCH USE ONLY | Module-level docstring template enforced; pre-commit check recommended | All |
