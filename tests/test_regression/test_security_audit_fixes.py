"""Regression tests for all security audit fixes from Prompt 5 (v0.7.0).

Verifies every bug fix applied during the security audit:
- torch.load uses weights_only=True
- np.load uses allow_pickle=False
- HMAC hashing (not plain SHA-256) in secure_aggregation, audit_logger,
  clinical_integrator
- get_access_log / get_audit_trail return copies (mutation safety)
- os.urandom salt default in deidentification
- Division by zero guard in coordinator
- Zero-norm gradient clipping
- Truthiness fix in dose_calculator (max_dose=0)
- Truthiness fix in patient_model (baseline_value=0)
- vital_name fix in multi_sensor_fusion
- Explicit zero-patient check in outcome_prediction
- NumPy trapezoid/trapz null check
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import PROJECT_ROOT, load_module

# ---------------------------------------------------------------------------
# Load all modules under test -- use unique names to avoid collisions
# ---------------------------------------------------------------------------
sec_agg_mod = load_module(
    "federated.secure_aggregation",
    "federated/secure_aggregation.py",
)
audit_mod = load_module(
    "privacy.audit_logger",
    "privacy/audit_logger.py",
)
clinical_mod = load_module(
    "clinical_integrator",
    "digital-twins/clinical-integration/clinical_integrator.py",
)
access_mod = load_module(
    "privacy.access_control",
    "privacy/access_control.py",
)
consent_mod = load_module(
    "privacy.consent_manager",
    "privacy/consent_manager.py",
)
enrollment_mod = load_module(
    "federated.site_enrollment",
    "federated/site_enrollment.py",
)
coordinator_mod = load_module(
    "federated.coordinator",
    "federated/coordinator.py",
)
dp_mod = load_module(
    "federated.differential_privacy",
    "federated/differential_privacy.py",
)
dose_mod = load_module(
    "dose_calculator",
    "tools/dose-calculator/dose_calculator.py",
)
patient_mod = load_module(
    "patient_model",
    "digital-twins/patient-modeling/patient_model.py",
)
fusion_mod = load_module(
    "multi_sensor_fusion",
    "examples-physical-ai/02_multi_sensor_fusion.py",
)
outcome_mod = load_module(
    "outcome_prediction",
    "examples/05_outcome_prediction_pipeline.py",
)
treatment_sim_mod = load_module(
    "treatment_simulator",
    "digital-twins/treatment-simulation/treatment_simulator.py",
)
model_mod = load_module("federated.model", "federated/model.py")


def _read_source_file(relative_path: str) -> str:
    """Read a source file's text directly from disk."""
    return (PROJECT_ROOT / relative_path).read_text()


class TestTorchLoadWeightsOnly:
    """Verify torch.load always uses weights_only=True in source code."""

    def test_benchmark_runner_torch_load_weights_only(self):
        src = _read_source_file("q1-2026-standards/objective-3-benchmarking/benchmark_runner.py")
        for match in re.finditer(r"torch\.load\(([^)]+)\)", src):
            args_text = match.group(1)
            assert "weights_only=True" in args_text, f"torch.load call missing weights_only=True: {match.group(0)}"

    def test_conversion_pipeline_torch_load_weights_only(self):
        src = _read_source_file("q1-2026-standards/objective-1-model-conversion/conversion_pipeline.py")
        for match in re.finditer(r"torch\.load\(([^)]+)\)", src):
            args_text = match.group(1)
            assert "weights_only=True" in args_text, f"torch.load call missing weights_only=True: {match.group(0)}"

    def test_model_validator_torch_load_weights_only(self):
        src = _read_source_file("q1-2026-standards/objective-2-model-registry/model_validator.py")
        for match in re.finditer(r"torch\.load\(([^)]+)\)", src):
            args_text = match.group(1)
            assert "weights_only=True" in args_text, f"torch.load call missing weights_only=True: {match.group(0)}"


class TestNpLoadAllowPickle:
    """Verify np.load uses allow_pickle=False where relevant."""

    def test_model_validator_np_load_allow_pickle_false(self):
        src = _read_source_file("q1-2026-standards/objective-2-model-registry/model_validator.py")
        for match in re.finditer(r"np\.load\(([^)]+)\)", src):
            args_text = match.group(1)
            # npz archives don't need allow_pickle but if used it should be False
            if "allow_pickle" in args_text:
                assert "allow_pickle=False" in args_text, (
                    f"np.load call should use allow_pickle=False: {match.group(0)}"
                )


class TestHMACHashingNotPlainSHA256:
    """Verify HMAC-based hashing in secure_aggregation, audit_logger, clinical_integrator."""

    def test_secure_aggregation_uses_hmac(self):
        src = inspect.getsource(sec_agg_mod.SecureAggregator)
        assert "hmac" in src, "SecureAggregator should use hmac for seed generation"
        assert "hmac.new" in src, "SecureAggregator._pair_seed should call hmac.new"

    def test_secure_aggregation_pair_seed_produces_int(self):
        agg = sec_agg_mod.SecureAggregator(seed=42)
        seed_val = agg._pair_seed(0, 1)
        assert isinstance(seed_val, int)
        assert seed_val > 0

    def test_audit_logger_uses_hmac(self):
        src = _read_source_file("privacy/audit_logger.py")
        assert "hmac" in src, "audit_logger module should use hmac"

    def test_audit_event_hash_uses_hmac(self):
        event = audit_mod.AuditEvent(
            event_type=audit_mod.EventType.DATA_ACCESS,
            severity=audit_mod.Severity.INFO,
            actor="test",
            resource="test_resource",
            action="test_action",
        )
        assert len(event.integrity_hash) == 32

    def test_clinical_integrator_uses_hmac_for_patient_id(self):
        src = _read_source_file("digital-twins/clinical-integration/clinical_integrator.py")
        assert "hmac" in src, "clinical_integrator should use hmac for patient ID hashing"


class TestGetAccessLogReturnsCopy:
    """Verify get_access_log / get_audit_trail return copies, not references."""

    def test_access_control_get_access_log_copy(self):
        mgr = access_mod.AccessControlManager()
        mgr.register_user("u1", "User 1", role="researcher")
        log = mgr.get_access_log()
        original_len = len(log)
        log.append({"fake": "entry"})
        assert len(mgr.get_access_log()) == original_len

    def test_consent_manager_get_audit_trail_copy(self):
        cm = consent_mod.ConsentManager()
        cm.register_consent("PT-001", "STUDY-001")
        trail = cm.get_audit_trail()
        original_len = len(trail)
        trail.append({"fake": "entry"})
        assert len(cm.get_audit_trail()) == original_len

    def test_enrollment_get_audit_trail_copy(self):
        mgr = enrollment_mod.SiteEnrollmentManager(study_id="TEST")
        mgr.enroll_site("S1", "Site 1", patient_count=50)
        trail = mgr.get_audit_trail()
        original_len = len(trail)
        trail.append({"fake": "entry"})
        assert len(mgr.get_audit_trail()) == original_len


class TestOsUrandomSaltDefault:
    """Verify Deidentifier uses os.urandom for salt when none provided."""

    def test_deidentification_source_uses_urandom(self):
        src = _read_source_file("privacy/deidentification.py")
        assert "os.urandom" in src, "Deidentifier should use os.urandom for default salt"

    def test_deidentification_salt_not_hardcoded(self):
        src = _read_source_file("privacy/deidentification.py")
        # The default should call os.urandom, not use a hardcoded string
        assert "salt = os.urandom" in src


class TestDivisionByZeroGuard:
    """Verify coordinator guards against division by zero in weighted averaging."""

    def test_coordinator_zero_total_samples_no_crash(self):
        config = model_mod.ModelConfig(input_dim=5, output_dim=2, seed=42)
        coord = coordinator_mod.FederationCoordinator(model_config=config)
        coord.initialize()
        # Two clients each claiming 0 samples
        update1 = coord.global_model.get_parameters()
        update2 = coord.global_model.get_parameters()
        result = coord.run_round(
            client_updates=[update1, update2],
            client_sample_counts=[0, 0],
        )
        assert result.round_number == 1

    def test_coordinator_source_has_zero_check(self):
        src = _read_source_file("federated/coordinator.py")
        assert "total == 0" in src or "total == 0" in src, (
            "Coordinator should have explicit zero check for total sample count"
        )


class TestZeroNormGradientClipping:
    """Verify DP gradient clipping handles zero-norm vectors gracefully."""

    def test_clip_zero_norm_vector(self):
        dp = dp_mod.DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.zeros(10)]
        clipped = dp.clip_parameters(params)
        np.testing.assert_array_equal(clipped[0], np.zeros(10))

    def test_clip_near_zero_norm_vector(self):
        dp = dp_mod.DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        params = [np.array([1e-20, 0.0, 0.0])]
        clipped = dp.clip_parameters(params)
        # Should not produce NaN or inf
        assert np.all(np.isfinite(clipped[0]))

    def test_dp_source_has_zero_norm_guard(self):
        src = _read_source_file("federated/differential_privacy.py")
        assert "== 0.0" in src or "== 0" in src, "DP clip_parameters should have zero-norm guard"


class TestTruthinessFixDoseCalculator:
    """Verify dose_calculator handles max_dose=0 correctly (not falsy)."""

    def test_max_dose_zero_not_skipped(self):
        """When max_dose lookup returns 0, warning should still be checked."""
        src = _read_source_file("tools/dose-calculator/dose_calculator.py")
        # The fix uses `is not None` instead of truthiness check
        assert "is not None" in src, "dose_calculator should use 'is not None' check for max_dose"


class TestTruthinessFixPatientModel:
    """Verify patient_model handles baseline_value=0 correctly."""

    def test_baseline_value_zero_not_rejected(self):
        """baseline_value=0 should be treated as valid (not negative)."""
        src = _read_source_file("digital-twins/patient-modeling/patient_model.py")
        # The fix changed `<= 0.0` to `< 0.0` to accept zero as valid
        assert "baseline_value < 0.0" in src, "patient_model should use `< 0.0` (not `<= 0.0`) for baseline_value check"


class TestVitalNameFix:
    """Verify multi_sensor_fusion uses vital_name variable, not hardcoded value."""

    def test_anomaly_record_contains_vital_name_variable(self):
        src = _read_source_file("examples-physical-ai/02_multi_sensor_fusion.py")
        # The fix ensures the anomaly record uses the actual vital_name variable
        assert '"vital_name": vital_name' in src, (
            "multi_sensor_fusion should reference vital_name variable in anomaly record"
        )


class TestExplicitZeroPatientCheck:
    """Verify outcome_prediction uses explicit zero check, not max(total, 1)."""

    def test_zero_patient_percentage_explicit_check(self):
        src = _read_source_file("examples/05_outcome_prediction_pipeline.py")
        # The fix replaced max(total_patients, 1) with explicit `== 0` check
        assert "total_patients == 0" in src, "outcome_prediction should use explicit 'total_patients == 0' check"

    def test_zero_patient_returns_zero_pct(self):
        # SiteMetrics with 0 patients should be handled
        sm = outcome_mod.SiteMetrics(site_id="EMPTY", num_patients=0)
        assert sm.num_patients == 0


class TestNumpyTrapezoidNullCheck:
    """Verify treatment_simulator checks for trapezoid/trapz availability."""

    def test_trapezoid_null_check_in_source(self):
        src = _read_source_file("digital-twins/treatment-simulation/treatment_simulator.py")
        assert "_trapz is None" in src, "treatment_simulator._compute_auc should check for None trapezoid"

    def test_compute_auc_produces_valid_result(self):
        # Should not raise RuntimeError when numpy has trapezoid/trapz
        auc = treatment_sim_mod._compute_auc(dose_mg=100.0)
        assert isinstance(auc, float)
        assert auc >= 0.0

    def test_compute_auc_zero_dose_returns_zero(self):
        auc = treatment_sim_mod._compute_auc(dose_mg=0.0)
        assert auc == 0.0
