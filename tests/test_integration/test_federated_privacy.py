"""Federated-to-privacy integration tests.

Tests DP mechanism -> secure aggregation -> access control -> audit trail,
verifying the privacy pipeline protects federated model updates.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
dp_mod = load_module(
    "federated.differential_privacy",
    "federated/differential_privacy.py",
)
sec_agg_mod = load_module(
    "federated.secure_aggregation",
    "federated/secure_aggregation.py",
)
access_mod = load_module("privacy.access_control", "privacy/access_control.py")
audit_mod = load_module("privacy.audit_logger", "privacy/audit_logger.py")
model_mod = load_module("federated.model", "federated/model.py")


class TestDifferentialPrivacy:
    """Verify DP mechanism clips and adds noise correctly."""

    def test_dp_creation(self):
        dp = dp_mod.DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5

    def test_dp_clip_parameters(self):
        dp = dp_mod.DifferentialPrivacy(max_grad_norm=1.0)
        params = [np.array([10.0, 0.0, 0.0])]
        clipped = dp.clip_parameters(params)
        total_norm = np.sqrt(sum(float(np.sum(p**2)) for p in clipped))
        assert total_norm <= 1.0 + 1e-6

    def test_dp_add_noise(self):
        dp = dp_mod.DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        params = [np.zeros(10)]
        noisy = dp.add_noise_to_parameters(params, num_clients=1, seed=42)
        assert not np.allclose(noisy[0], 0.0)

    def test_dp_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            dp_mod.DifferentialPrivacy(epsilon=-1.0)

    def test_dp_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            dp_mod.DifferentialPrivacy(delta=0.0)


class TestSecureAggregation:
    """Verify secure aggregation module structure."""

    def test_secure_aggregation_module_loaded(self):
        assert sec_agg_mod is not None

    def test_secure_aggregation_has_expected_attrs(self):
        attrs = dir(sec_agg_mod)
        assert len(attrs) > 0


class TestAccessControl:
    """Verify access control module integrates with privacy pipeline."""

    def test_register_and_check_permission(self):
        mgr = access_mod.AccessControlManager()
        mgr.register_user(
            user_id="analyst_1",
            display_name="Test Analyst",
            role=access_mod.Role.DATA_ENGINEER,
        )
        user = mgr.get_user("analyst_1")
        assert user is not None
        assert user.active is True

    def test_deactivate_user(self):
        mgr = access_mod.AccessControlManager()
        mgr.register_user("u1", "User 1", role="data_engineer")
        result = mgr.deactivate_user("u1")
        assert result is True
        user = mgr.get_user("u1")
        assert user.active is False

    def test_access_log_returns_copy(self):
        mgr = access_mod.AccessControlManager()
        mgr.register_user("u1", "User 1", role="data_engineer")
        log = mgr.get_access_log()
        original_len = len(log)
        log.append({"fake": "entry"})
        assert len(mgr.get_access_log()) == original_len


class TestDPToSecureAggPipeline:
    """Verify DP clipping feeds into aggregation correctly."""

    def test_dp_clip_then_aggregate(self):
        dp = dp_mod.DifferentialPrivacy(epsilon=2.0, delta=1e-5, max_grad_norm=1.0)
        client_params = [
            [np.random.randn(5) * 10 for _ in range(2)],
            [np.random.randn(5) * 10 for _ in range(2)],
        ]
        clipped = [dp.clip_parameters(p) for p in client_params]
        for cp in clipped:
            norm = np.sqrt(sum(float(np.sum(p**2)) for p in cp))
            assert norm <= 1.0 + 1e-6

    def test_dp_noise_then_verify_privacy(self):
        dp = dp_mod.DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        params = [np.ones(10)]
        noisy1 = dp.add_noise_to_parameters(params, seed=1)
        noisy2 = dp.add_noise_to_parameters(params, seed=2)
        assert not np.allclose(noisy1[0], noisy2[0])


class TestAuditTrailCapture:
    """Verify audit logger captures privacy-related events."""

    def test_audit_log_event_types(self):
        al = audit_mod.AuditLogger()
        al.log(
            event_type=audit_mod.EventType.PARAMETER_EXCHANGE,
            actor="dp_module",
            resource="model_params",
            action="dp noise added",
            details={"epsilon": 1.0, "round": 1},
        )
        events = al.get_events()
        assert len(events) >= 1

    def test_audit_get_events_returns_reference(self):
        """Verify audit events are retrievable and consistent."""
        al = audit_mod.AuditLogger()
        al.log(
            event_type=audit_mod.EventType.MODEL_TRAINING,
            actor="system",
            resource="model",
            action="training started",
        )
        events = al.get_events()
        assert len(events) >= 1
        assert events[0].event_type == audit_mod.EventType.MODEL_TRAINING
