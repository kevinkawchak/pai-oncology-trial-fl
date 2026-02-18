"""Shared fixtures and helpers for the pai-oncology-trial-fl test suite.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Source modules in hyphenated directories loaded via importlib.util.
NumPy RNG seeded for deterministic tests.
LICENSE: MIT.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Module loader — the permanent architectural pattern for optional heavy
# dependencies and hyphenated directory names.
# ---------------------------------------------------------------------------
_MODULE_CACHE: dict[str, Any] = {}


def load_module(name: str, relative_path: str) -> Any:
    """Load a Python module by its file-system path relative to PROJECT_ROOT.

    Uses ``importlib.util.spec_from_file_location`` so that modules inside
    hyphenated directories (e.g., ``privacy/phi-pii-management/phi_detector.py``)
    can be imported despite being invalid Python package names.

    Returns the cached module if already loaded.  Calls ``pytest.skip()`` if
    the file is not found.  Wraps ``spec.loader.exec_module()`` in a
    ``try/except ImportError`` — on failure the partial module is removed from
    ``sys.modules`` and the test is skipped.
    """
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    if name in sys.modules:
        _MODULE_CACHE[name] = sys.modules[name]
        return sys.modules[name]

    filepath = PROJECT_ROOT / relative_path
    if not filepath.exists():
        pytest.skip(f"Module file not found: {relative_path}")

    spec = importlib.util.spec_from_file_location(name, str(filepath))
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create module spec for: {relative_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        sys.modules.pop(name, None)
        pytest.skip(f"ImportError loading {relative_path}: {exc}")

    _MODULE_CACHE[name] = module
    return module


# ---------------------------------------------------------------------------
# Deterministic RNG seed (autouse)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed NumPy RNG for deterministic, reproducible tests."""
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Section: Federated Learning fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def federated_coordinator():
    """Load and return the ``federated.coordinator`` module."""
    return load_module("federated.coordinator", "federated/coordinator.py")


@pytest.fixture
def federated_client():
    """Load and return the ``federated.client`` module."""
    return load_module("federated.client", "federated/client.py")


@pytest.fixture
def federated_model():
    """Load and return the ``federated.model`` module."""
    return load_module("federated.model", "federated/model.py")


@pytest.fixture
def federated_dp():
    """Load and return the ``federated.differential_privacy`` module."""
    return load_module(
        "federated.differential_privacy",
        "federated/differential_privacy.py",
    )


@pytest.fixture
def federated_secure_agg():
    """Load and return the ``federated.secure_aggregation`` module."""
    return load_module(
        "federated.secure_aggregation",
        "federated/secure_aggregation.py",
    )


@pytest.fixture
def federated_site_enrollment():
    """Load and return the ``federated.site_enrollment`` module."""
    return load_module(
        "federated.site_enrollment",
        "federated/site_enrollment.py",
    )


@pytest.fixture
def federated_data_ingestion():
    """Load and return the ``federated.data_ingestion`` module."""
    return load_module(
        "federated.data_ingestion",
        "federated/data_ingestion.py",
    )


@pytest.fixture
def federated_data_harmonization():
    """Load and return the ``federated.data_harmonization`` module."""
    return load_module(
        "federated.data_harmonization",
        "federated/data_harmonization.py",
    )


# ---------------------------------------------------------------------------
# Section: Physical AI / Digital Twin fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def physical_ai_digital_twin():
    """Load and return the ``physical_ai.digital_twin`` module."""
    return load_module("physical_ai.digital_twin", "physical_ai/digital_twin.py")


@pytest.fixture
def physical_ai_sensor_fusion():
    """Load and return the ``physical_ai.sensor_fusion`` module."""
    return load_module("physical_ai.sensor_fusion", "physical_ai/sensor_fusion.py")


@pytest.fixture
def physical_ai_robotic():
    """Load and return the ``physical_ai.robotic_integration`` module."""
    return load_module(
        "physical_ai.robotic_integration",
        "physical_ai/robotic_integration.py",
    )


@pytest.fixture
def physical_ai_surgical():
    """Load and return the ``physical_ai.surgical_tasks`` module."""
    return load_module("physical_ai.surgical_tasks", "physical_ai/surgical_tasks.py")


@pytest.fixture
def physical_ai_sim_bridge():
    """Load and return the ``physical_ai.simulation_bridge`` module."""
    return load_module(
        "physical_ai.simulation_bridge",
        "physical_ai/simulation_bridge.py",
    )


@pytest.fixture
def physical_ai_framework_detect():
    """Load and return the ``physical_ai.framework_detection`` module."""
    return load_module(
        "physical_ai.framework_detection",
        "physical_ai/framework_detection.py",
    )


@pytest.fixture
def dt_clinical_integrator():
    """Load clinical integrator from hyphenated directory."""
    return load_module(
        "clinical_integrator",
        "digital-twins/clinical-integration/clinical_integrator.py",
    )


@pytest.fixture
def dt_patient_model():
    """Load patient model from hyphenated directory."""
    return load_module(
        "patient_model",
        "digital-twins/patient-modeling/patient_model.py",
    )


@pytest.fixture
def dt_treatment_simulator():
    """Load treatment simulator from hyphenated directory."""
    return load_module(
        "treatment_simulator",
        "digital-twins/treatment-simulation/treatment_simulator.py",
    )


# ---------------------------------------------------------------------------
# Section: Privacy fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def privacy_phi_detector():
    """Load PHI detector (top-level)."""
    return load_module("privacy.phi_detector", "privacy/phi_detector.py")


@pytest.fixture
def privacy_deidentification():
    """Load de-identification (top-level)."""
    return load_module("privacy.deidentification", "privacy/deidentification.py")


@pytest.fixture
def privacy_access_control():
    """Load access control (top-level)."""
    return load_module("privacy.access_control", "privacy/access_control.py")


@pytest.fixture
def privacy_breach_response():
    """Load breach response (top-level)."""
    return load_module("privacy.breach_response", "privacy/breach_response.py")


@pytest.fixture
def privacy_consent_manager():
    """Load consent manager (top-level)."""
    return load_module("privacy.consent_manager", "privacy/consent_manager.py")


@pytest.fixture
def privacy_audit_logger():
    """Load audit logger (top-level)."""
    return load_module("privacy.audit_logger", "privacy/audit_logger.py")


@pytest.fixture
def privacy_phi_detector_v2():
    """Load advanced PHI detector from hyphenated directory."""
    return load_module(
        "phi_detector_v2",
        "privacy/phi-pii-management/phi_detector.py",
    )


@pytest.fixture
def privacy_deident_pipeline():
    """Load de-identification pipeline from hyphenated directory."""
    return load_module(
        "deidentification_pipeline",
        "privacy/de-identification/deidentification_pipeline.py",
    )


@pytest.fixture
def privacy_access_mgr():
    """Load access control manager from hyphenated directory."""
    return load_module(
        "access_control_manager",
        "privacy/access-control/access_control_manager.py",
    )


@pytest.fixture
def privacy_breach_protocol():
    """Load breach response protocol from hyphenated directory."""
    return load_module(
        "breach_response_protocol",
        "privacy/breach-response/breach_response_protocol.py",
    )


@pytest.fixture
def privacy_dua_generator():
    """Load DUA generator from hyphenated directory."""
    return load_module(
        "dua_generator",
        "privacy/dua-templates-generator/dua_generator.py",
    )


# ---------------------------------------------------------------------------
# Section: Regulatory fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def regulatory_compliance():
    """Load compliance checker (top-level)."""
    return load_module(
        "regulatory.compliance_checker",
        "regulatory/compliance_checker.py",
    )


@pytest.fixture
def regulatory_fda():
    """Load FDA submission (top-level)."""
    return load_module("regulatory.fda_submission", "regulatory/fda_submission.py")


@pytest.fixture
def regulatory_fda_tracker():
    """Load FDA submission tracker from hyphenated directory."""
    return load_module(
        "fda_submission_tracker",
        "regulatory/fda-compliance/fda_submission_tracker.py",
    )


@pytest.fixture
def regulatory_irb():
    """Load IRB protocol manager from hyphenated directory."""
    return load_module(
        "irb_protocol_manager",
        "regulatory/irb-management/irb_protocol_manager.py",
    )


@pytest.fixture
def regulatory_gcp():
    """Load GCP compliance checker from hyphenated directory."""
    return load_module(
        "gcp_compliance_checker",
        "regulatory/ich-gcp/gcp_compliance_checker.py",
    )


@pytest.fixture
def regulatory_tracker():
    """Load regulatory tracker from hyphenated directory."""
    return load_module(
        "regulatory_tracker",
        "regulatory/regulatory-intelligence/regulatory_tracker.py",
    )


# ---------------------------------------------------------------------------
# Section: Tools fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tool_dose_calculator():
    """Load dose calculator from hyphenated directory."""
    return load_module(
        "dose_calculator",
        "tools/dose-calculator/dose_calculator.py",
    )


@pytest.fixture
def tool_deployment_readiness():
    """Load deployment readiness from hyphenated directory."""
    return load_module(
        "deployment_readiness",
        "tools/deployment-readiness/deployment_readiness.py",
    )


@pytest.fixture
def tool_trial_site_monitor():
    """Load trial site monitor from hyphenated directory."""
    return load_module(
        "trial_site_monitor",
        "tools/trial-site-monitor/trial_site_monitor.py",
    )


@pytest.fixture
def tool_sim_job_runner():
    """Load sim job runner from hyphenated directory."""
    return load_module(
        "sim_job_runner",
        "tools/sim-job-runner/sim_job_runner.py",
    )


@pytest.fixture
def tool_dicom_inspector():
    """Load DICOM inspector from hyphenated directory."""
    return load_module(
        "dicom_inspector",
        "tools/dicom-inspector/dicom_inspector.py",
    )


# ---------------------------------------------------------------------------
# Section: Unification fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def unif_bridge():
    """Load Isaac-MuJoCo bridge."""
    return load_module(
        "isaac_mujoco_bridge",
        "unification/simulation_physics/isaac_mujoco_bridge.py",
    )


@pytest.fixture
def unif_agent():
    """Load unified agent interface."""
    return load_module(
        "unified_agent_interface",
        "unification/agentic_generative_ai/unified_agent_interface.py",
    )


@pytest.fixture
def unif_converter():
    """Load model converter."""
    return load_module(
        "model_converter",
        "unification/cross_platform_tools/model_converter.py",
    )


@pytest.fixture
def unif_framework_detector():
    """Load framework detector."""
    return load_module(
        "framework_detector",
        "unification/cross_platform_tools/framework_detector.py",
    )


@pytest.fixture
def unif_validation():
    """Load validation suite."""
    return load_module(
        "validation_suite",
        "unification/cross_platform_tools/validation_suite.py",
    )


@pytest.fixture
def unif_policy():
    """Load policy exporter."""
    return load_module(
        "policy_exporter",
        "unification/cross_platform_tools/policy_exporter.py",
    )


# ---------------------------------------------------------------------------
# Section: Standards fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def standards_conversion():
    """Load conversion pipeline."""
    return load_module(
        "conversion_pipeline",
        "q1-2026-standards/objective-1-model-conversion/conversion_pipeline.py",
    )


@pytest.fixture
def standards_validator():
    """Load model validator."""
    return load_module(
        "model_validator",
        "q1-2026-standards/objective-2-model-registry/model_validator.py",
    )


@pytest.fixture
def standards_benchmark():
    """Load benchmark runner."""
    return load_module(
        "benchmark_runner",
        "q1-2026-standards/objective-3-benchmarking/benchmark_runner.py",
    )


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_patient_record() -> dict[str, Any]:
    """Generate a synthetic patient record for testing."""
    return {
        "patient_id": "PAT-TEST-001",
        "age": 62,
        "sex": "M",
        "tumor_type": "NSCLC",
        "stage": "IIIB",
        "tumor_volume_cm3": 24.5,
        "biomarkers": {"CEA": 8.2, "PD-L1": 65.0},
        "treatment_arm": "experimental",
        "site_id": "SITE-A",
    }


@pytest.fixture
def synthetic_model_weights() -> list[np.ndarray]:
    """Generate synthetic model weights for testing."""
    return [
        np.random.randn(10, 5).astype(np.float32),
        np.random.randn(5).astype(np.float32),
        np.random.randn(5, 2).astype(np.float32),
        np.random.randn(2).astype(np.float32),
    ]


@pytest.fixture
def trial_cohort_config() -> dict[str, Any]:
    """Generate a trial cohort configuration for testing."""
    return {
        "trial_id": "NCT-TEST-2026-001",
        "protocol_version": "1.0",
        "arms": ["control", "experimental"],
        "sites": ["SITE-A", "SITE-B", "SITE-C"],
        "enrollment_target": 300,
        "primary_endpoint": "overall_survival",
        "secondary_endpoints": ["progression_free_survival", "objective_response_rate"],
    }


@pytest.fixture
def sample_sensor_data() -> dict[str, np.ndarray]:
    """Generate sample multi-modal sensor data."""
    return {
        "force": np.random.randn(100, 3).astype(np.float32),
        "position": np.random.randn(100, 6).astype(np.float32),
        "vision": np.random.randn(10, 64, 64).astype(np.float32),
        "timestamps": np.linspace(0, 1.0, 100).astype(np.float64),
    }


@pytest.fixture
def sample_phi_text() -> str:
    """Return a synthetic text string containing mock PHI patterns."""
    return (
        "Patient John Doe (SSN: 123-45-6789) was admitted on 01/15/2026. "
        "MRN: MRN-12345678. Contact: (555) 123-4567, john.doe@example.com. "
        "Address: 123 Main St, Springfield, IL 62704. IP: 192.168.1.100."
    )
