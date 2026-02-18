# Test Suite

Comprehensive pytest-based test suite for **PAI Oncology Trial FL** achieving coverage of all Python modules.

## Testing Philosophy

- **Mock isolation:** All tests are self-contained — no GPU, no network, no filesystem dependencies
- **Deterministic RNG:** `np.random.seed(42)` autouse fixture ensures reproducible results
- **Optional dependency skip:** Modules with heavy optional dependencies (PyTorch, pydicom, Presidio) auto-skip via `load_module()` pattern using `importlib.util`
- **Hyphenated directory support:** Source modules in directories like `privacy/phi-pii-management/` are loaded via `importlib.util.spec_from_file_location`

## How to Run

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run a specific subdirectory
pytest tests/test_federated/ -v

# Run a specific test class
pytest tests/test_privacy/test_phi_detector.py::TestPHIDetection -v

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures, load_module(), mock data factories
├── README.md                      # This file
├── test_federated/                # Federated learning module tests (8 files)
│   ├── test_coordinator.py        #   Federation coordinator, aggregation, SCAFFOLD
│   ├── test_client.py             #   Federated client, local training
│   ├── test_model.py              #   OncologyMLP, forward pass, parameters
│   ├── test_differential_privacy.py  # DP mechanism, gradient clipping, budget
│   ├── test_secure_aggregation.py #   Secure aggregation, HMAC pair seeds
│   ├── test_site_enrollment.py    #   Site enrollment lifecycle, audit trail
│   ├── test_data_ingestion.py     #   Synthetic data generation, partitioning
│   └── test_data_harmonization.py #   Multi-site data harmonization
├── test_physical_ai/              # Physical AI module tests (6 files)
│   ├── test_digital_twin.py       #   Tumor growth models, trajectories
│   ├── test_sensor_fusion.py      #   Multi-modal sensor fusion
│   ├── test_robotic_integration.py #  Robotic interface, telemetry
│   ├── test_surgical_tasks.py     #   Surgical tasks, thresholds
│   ├── test_simulation_bridge.py  #   Cross-platform simulation bridge
│   └── test_framework_detection.py #  Framework detection pipeline
├── test_digital_twins/            # Digital twin module tests (3 files)
│   ├── test_clinical_integrator.py #  Clinical data integration, HMAC
│   ├── test_patient_model.py      #   Patient modeling, biomarkers
│   └── test_treatment_simulator.py #  Treatment simulation, PK
├── test_privacy/                  # Privacy module tests (5 files)
│   ├── test_phi_detector.py       #   PHI detection, 18 HIPAA identifiers
│   ├── test_deidentification.py   #   De-identification, HMAC, salt
│   ├── test_access_control.py     #   RBAC, audit log copies
│   ├── test_breach_response.py    #   Breach response, risk assessment
│   └── test_dua_generator.py      #   DUA templates, retention
├── test_regulatory/               # Regulatory module tests (4 files)
│   ├── test_fda_submission.py     #   FDA tracker, AI/ML components
│   ├── test_irb_protocol.py       #   IRB lifecycle, amendments
│   ├── test_gcp_compliance.py     #   GCP principles, scoring
│   └── test_regulatory_tracker.py #   Multi-jurisdiction, overdue ordering
├── test_tools/                    # CLI tool tests (5 files)
│   ├── test_dose_calculator.py    #   BED, EQD2, OAR constraints
│   ├── test_deployment_readiness.py # Readiness checks, scoring
│   ├── test_trial_site_monitor.py #   Site monitoring, alerts
│   ├── test_sim_job_runner.py     #   Job runner, configuration
│   └── test_dicom_inspector.py    #   DICOM inspection, PHI tags
├── test_unification/              # Unification layer tests (5 files)
│   ├── test_bridge.py             #   Isaac-MuJoCo bridge, sync
│   ├── test_converter.py          #   Model converter, formats
│   ├── test_agent_interface.py    #   Unified agent interface
│   ├── test_framework_detector.py #   Framework detection
│   └── test_validation_suite.py   #   Validation suite, reports
├── test_standards/                # Standards module tests (3 files)
│   ├── test_conversion_pipeline.py # Format conversion, tolerance
│   ├── test_benchmark_runner.py   #   Benchmark runner, metrics
│   └── test_model_validator.py    #   Model validation, metadata
├── test_agentic_ai/               # Agentic AI tests (5 files)
│   ├── test_mcp_server.py         #   MCP oncology server
│   ├── test_react_planner.py      #   ReAct treatment planner
│   ├── test_monitoring_agent.py   #   Adaptive monitoring agent
│   ├── test_simulation_orchestrator.py # Simulation orchestrator
│   └── test_safety_executor.py    #   Safety-constrained executor
├── test_examples/                 # Example script tests (6 files)
│   ├── test_federated_workflow.py #   End-to-end federated workflow
│   ├── test_digital_twin_planning.py # Twin treatment planning
│   ├── test_cross_framework_validation.py # Cross-framework validation
│   ├── test_outcome_prediction.py #   Outcome prediction pipeline
│   ├── test_physical_ai_examples.py #  Physical AI examples
│   └── test_generate_synthetic_data.py # Synthetic data generation
├── test_integration/              # Cross-module integration tests (6 files)
│   ├── test_trial_lifecycle.py    #   End-to-end trial lifecycle
│   ├── test_cross_framework.py    #   Cross-framework workflows
│   ├── test_privacy_clinical.py   #   Privacy → clinical integration
│   ├── test_domain_safety.py      #   Domain → safety integration
│   ├── test_agentic_regulatory.py #   Agentic → regulatory integration
│   └── test_federated_privacy.py  #   Federated → privacy integration
└── test_regression/               # Regression tests for v0.7.0 fixes
    └── test_security_audit_fixes.py # Guards for all 61 audit findings
```

## How to Add Tests

1. Create a new `test_*.py` file in the appropriate subdirectory
2. Use `from tests.conftest import load_module` to import modules from hyphenated directories
3. Group tests in classes: `class TestFeatureName:`
4. Use fixtures from `conftest.py` for common setup
5. Follow naming: `test_[method]_[scenario]`

## CI Integration

Tests are run in the CI pipeline for Python 3.10, 3.11, and 3.12:

```bash
pip install numpy scipy pytest pyyaml
pytest tests/ -v --tb=short
```

## Coverage Targets

- **1400+ tests** across **68 test files** in **12 subdirectories**
- Every public class and function has at least one test
- All Enum states are exercised
- Edge cases: empty inputs, None values, boundary conditions, zero values
- Regression guards for all v0.7.0 security audit findings
