"""Tests for agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py.

Validates safety constraints, execution boundaries, constraint evaluation,
enum definitions, and dataclass creation.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module(
    "safety_executor",
    "agentic-ai/examples-agentic-ai/05_safety_constrained_agent_executor.py",
)


class TestSafetyEnums:
    """Verify all safety executor enums."""

    def test_safety_level_values(self):
        assert mod.SafetyLevel.NEGLIGIBLE.value == "negligible"
        assert mod.SafetyLevel.LOW.value == "low"
        assert mod.SafetyLevel.MODERATE.value == "moderate"
        assert mod.SafetyLevel.HIGH.value == "high"
        assert mod.SafetyLevel.CRITICAL.value == "critical"

    def test_executor_state_values(self):
        assert mod.ExecutorState.IDLE.value == "idle"
        assert mod.ExecutorState.PRE_CHECK.value == "pre_check"
        assert mod.ExecutorState.AWAITING_APPROVAL.value == "awaiting_approval"
        assert mod.ExecutorState.EXECUTING.value == "executing"
        assert mod.ExecutorState.POST_CHECK.value == "post_check"
        assert mod.ExecutorState.ROLLING_BACK.value == "rolling_back"
        assert mod.ExecutorState.EMERGENCY_STOP.value == "emergency_stop"
        assert mod.ExecutorState.LOCKED_OUT.value == "locked_out"

    def test_approval_status_values(self):
        assert mod.ApprovalStatus.PENDING.value == "pending"
        assert mod.ApprovalStatus.APPROVED.value == "approved"
        assert mod.ApprovalStatus.REJECTED.value == "rejected"
        assert mod.ApprovalStatus.TIMED_OUT.value == "timed_out"
        assert mod.ApprovalStatus.AUTO_APPROVED.value == "auto_approved"

    def test_violation_type_values(self):
        assert mod.ViolationType.DOSE_LIMIT_EXCEEDED.value == "dose_limit_exceeded"
        assert mod.ViolationType.FORCE_LIMIT_EXCEEDED.value == "force_limit_exceeded"
        assert mod.ViolationType.WORKSPACE_BOUNDARY.value == "workspace_boundary"
        assert mod.ViolationType.EMERGENCY_STOP.value == "emergency_stop"

    def test_incident_severity_values(self):
        assert mod.IncidentSeverity.NEAR_MISS.value == "near_miss"
        assert mod.IncidentSeverity.MINOR.value == "minor"
        assert mod.IncidentSeverity.SERIOUS.value == "serious"
        assert mod.IncidentSeverity.CRITICAL.value == "critical"

    def test_risk_category_values(self):
        assert mod.RiskCategory.ACCEPTABLE.value == "acceptable"
        assert mod.RiskCategory.ALARP.value == "as_low_as_reasonably_practicable"
        assert mod.RiskCategory.UNACCEPTABLE.value == "unacceptable"


class TestSafetyConstraint:
    """Verify SafetyConstraint dataclass and evaluation."""

    def test_safety_constraint_defaults(self):
        sc = mod.SafetyConstraint()
        assert sc.constraint_type == "pre_condition"
        assert sc.safety_level == mod.SafetyLevel.MODERATE
        assert sc.enabled is True

    def test_safety_constraint_custom(self):
        sc = mod.SafetyConstraint(
            name="dose_limit",
            description="Check dose does not exceed protocol maximum",
            safety_level=mod.SafetyLevel.CRITICAL,
            standard_reference="IEC 80601-2-77",
        )
        assert sc.name == "dose_limit"
        assert sc.safety_level == mod.SafetyLevel.CRITICAL
        assert sc.standard_reference == "IEC 80601-2-77"

    def test_safety_constraint_disabled_evaluates_true(self):
        sc = mod.SafetyConstraint(enabled=False)
        passed, msg = sc.evaluate({})
        assert passed is True
        assert "disabled" in msg.lower()

    def test_safety_constraint_unique_ids(self):
        s1 = mod.SafetyConstraint()
        s2 = mod.SafetyConstraint()
        assert s1.constraint_id != s2.constraint_id

    def test_safety_constraint_id_prefix(self):
        sc = mod.SafetyConstraint()
        assert sc.constraint_id.startswith("CON-")

    def test_safety_constraint_with_parameters(self):
        sc = mod.SafetyConstraint(parameters={"max_dose_gy": 60.0, "max_force_n": 40.0})
        assert sc.parameters["max_dose_gy"] == 60.0
        assert sc.parameters["max_force_n"] == 40.0


class TestExecutorStateTransitions:
    """Verify the set of executor states covers required lifecycle."""

    def test_executor_state_count(self):
        states = list(mod.ExecutorState)
        assert len(states) >= 8

    def test_executor_state_all_unique(self):
        values = [s.value for s in mod.ExecutorState]
        assert len(values) == len(set(values))

    def test_violation_type_all_unique(self):
        values = [v.value for v in mod.ViolationType]
        assert len(values) == len(set(values))


class TestModuleAttributes:
    """Verify the module exposes expected top-level attributes."""

    def test_has_safety_level(self):
        assert hasattr(mod, "SafetyLevel")

    def test_has_executor_state(self):
        assert hasattr(mod, "ExecutorState")

    def test_has_safety_constraint(self):
        assert hasattr(mod, "SafetyConstraint")

    def test_has_approval_status(self):
        assert hasattr(mod, "ApprovalStatus")

    def test_has_violation_type(self):
        assert hasattr(mod, "ViolationType")
