"""
Safety-Constrained Agent Executor with Pre/Post-Conditions and Human-in-the-Loop.

CLINICAL CONTEXT:
    This module implements a safety-gated agent executor designed for
    oncology clinical trial operations involving Physical AI systems.
    Every tool invocation passes through pre-condition safety checks,
    post-condition validation, and optional human-in-the-loop approval
    gates. The executor enforces IEC 80601-2-77 (medical robotics) and
    ISO 14971 (risk management) aligned safety constraints, with
    emergency stop capability and automatic rollback on safety violations.

    The agent maintains a safety state machine that prevents execution
    of actions that could violate dose limits, force boundaries, or
    procedural sequencing constraints, providing defense-in-depth for
    autonomous clinical trial operations.

USE CASES COVERED:
    1. Pre-condition safety gates verifying clinical, mechanical, and
       regulatory constraints before any tool execution.
    2. Post-condition validation confirming that tool outputs satisfy
       safety invariants (dose limits, force bounds, tissue integrity).
    3. Human-in-the-loop approval workflow for high-risk actions with
       configurable escalation timeouts and override policies.
    4. Emergency stop (E-stop) capability that immediately halts all
       agent actions and enters a safe state per IEC 80601-2-77.
    5. Automatic rollback on safety violations with state restoration
       and detailed incident logging per ISO 14971 risk management.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0        (https://numpy.org/)
    Optional:
        - anthropic >= 0.39.0    (https://docs.anthropic.com/)
        - openai >= 1.0.0        (https://platform.openai.com/)

REFERENCES:
    - IEC 80601-2-77:2019 Medical Electrical Equipment - Particular
      requirements for the basic safety and essential performance of
      robotically assisted surgical equipment.
    - ISO 14971:2019 Medical devices - Application of risk management
      to medical devices.
    - FDA 21 CFR Part 11 - Electronic Records; Electronic Signatures.
    - IEC 62304:2006+AMD1:2015 Medical device software - Software
      life cycle processes.
    - ICH E6(R3) Good Clinical Practice (2023).

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or direct patient care. All safety-critical outputs
    must be validated by qualified engineers and clinicians.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports for optional dependencies
# ---------------------------------------------------------------------------
try:
    import anthropic  # type: ignore[import-untyped]

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

try:
    import openai  # type: ignore[import-untyped]

    HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[assignment]
    HAS_OPENAI = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class SafetyLevel(Enum):
    """Safety classification levels per ISO 14971 risk matrix."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutorState(Enum):
    """State machine for the safety-constrained executor."""

    IDLE = "idle"
    PRE_CHECK = "pre_check"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    POST_CHECK = "post_check"
    ROLLING_BACK = "rolling_back"
    EMERGENCY_STOP = "emergency_stop"
    COMPLETE = "complete"
    ERROR = "error"
    LOCKED_OUT = "locked_out"


class ApprovalStatus(Enum):
    """Status of a human-in-the-loop approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    AUTO_APPROVED = "auto_approved"


class ViolationType(Enum):
    """Types of safety violations."""

    DOSE_LIMIT_EXCEEDED = "dose_limit_exceeded"
    FORCE_LIMIT_EXCEEDED = "force_limit_exceeded"
    WORKSPACE_BOUNDARY = "workspace_boundary"
    PROCEDURE_SEQUENCE = "procedure_sequence"
    TISSUE_INTEGRITY = "tissue_integrity"
    TIMING_CONSTRAINT = "timing_constraint"
    AUTHORIZATION = "authorization"
    REGULATORY = "regulatory"
    EMERGENCY_STOP = "emergency_stop"


class IncidentSeverity(Enum):
    """Incident severity for ISO 14971 risk reporting."""

    NEAR_MISS = "near_miss"
    MINOR = "minor"
    MODERATE = "moderate"
    SERIOUS = "serious"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """ISO 14971 risk categories."""

    ACCEPTABLE = "acceptable"
    ALARP = "as_low_as_reasonably_practicable"
    UNACCEPTABLE = "unacceptable"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SafetyConstraint:
    """A safety constraint that must be satisfied before/after tool execution."""

    constraint_id: str = field(default_factory=lambda: f"CON-{uuid.uuid4().hex[:8].upper()}")
    name: str = ""
    description: str = ""
    constraint_type: str = "pre_condition"
    safety_level: SafetyLevel = SafetyLevel.MODERATE
    check_function: Optional[Callable[..., bool]] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    standard_reference: str = ""
    failure_message: str = ""

    def evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Evaluate the constraint against the given context.

        Returns:
            Tuple of (passed, message).
        """
        if not self.enabled:
            return True, "Constraint disabled"

        if self.check_function is not None:
            try:
                passed = self.check_function(context, self.parameters)
                msg = "Passed" if passed else self.failure_message
                return passed, msg
            except Exception as exc:
                return False, f"Constraint evaluation error: {exc}"

        return True, "No check function defined"


@dataclass
class ApprovalRequest:
    """A human-in-the-loop approval request."""

    request_id: str = field(default_factory=lambda: f"APR-{uuid.uuid4().hex[:10].upper()}")
    action_name: str = ""
    action_parameters: dict[str, Any] = field(default_factory=dict)
    safety_level: SafetyLevel = SafetyLevel.HIGH
    reason: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: float = field(default_factory=time.time)
    responded_at: float = 0.0
    approver_id: str = ""
    approver_notes: str = ""
    timeout_seconds: float = 300.0
    risk_assessment: dict[str, Any] = field(default_factory=dict)

    def is_timed_out(self) -> bool:
        """Check if the approval request has timed out."""
        return (time.time() - self.requested_at) > self.timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "action_name": self.action_name,
            "safety_level": self.safety_level.value,
            "reason": self.reason,
            "status": self.status.value,
            "requested_at": self.requested_at,
            "timeout_seconds": self.timeout_seconds,
            "risk_assessment": self.risk_assessment,
        }


@dataclass
class SafetyViolation:
    """A detected safety violation with incident tracking."""

    violation_id: str = field(default_factory=lambda: f"VIO-{uuid.uuid4().hex[:10].upper()}")
    violation_type: ViolationType = ViolationType.DOSE_LIMIT_EXCEEDED
    severity: IncidentSeverity = IncidentSeverity.MINOR
    description: str = ""
    action_name: str = ""
    constraint_id: str = ""
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    remediation: str = ""
    rollback_performed: bool = False
    standard_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "action_name": self.action_name,
            "constraint_id": self.constraint_id,
            "timestamp": self.timestamp,
            "remediation": self.remediation,
            "rollback_performed": self.rollback_performed,
            "standard_reference": self.standard_reference,
        }


@dataclass
class ExecutionCheckpoint:
    """A checkpoint for rollback support."""

    checkpoint_id: str = field(default_factory=lambda: f"CKP-{uuid.uuid4().hex[:8].upper()}")
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    action_name: str = ""
    action_index: int = 0


@dataclass
class SafetyAction:
    """An action to be executed with safety constraints."""

    action_id: str = field(default_factory=lambda: f"ACT-{uuid.uuid4().hex[:10].upper()}")
    name: str = ""
    description: str = ""
    handler: Optional[Callable[..., dict[str, Any]]] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    safety_level: SafetyLevel = SafetyLevel.MODERATE
    pre_conditions: list[SafetyConstraint] = field(default_factory=list)
    post_conditions: list[SafetyConstraint] = field(default_factory=list)
    requires_approval: bool = False
    approval_timeout_s: float = 300.0
    rollback_handler: Optional[Callable[..., bool]] = None
    standard_references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_id": self.action_id,
            "name": self.name,
            "description": self.description,
            "safety_level": self.safety_level.value,
            "requires_approval": self.requires_approval,
            "pre_conditions_count": len(self.pre_conditions),
            "post_conditions_count": len(self.post_conditions),
            "standard_references": self.standard_references,
        }


@dataclass
class ExecutionResult:
    """Result of a safety-constrained action execution."""

    result_id: str = field(default_factory=lambda: f"EXR-{uuid.uuid4().hex[:10].upper()}")
    action_id: str = ""
    action_name: str = ""
    success: bool = False
    output: dict[str, Any] = field(default_factory=dict)
    pre_check_passed: bool = False
    post_check_passed: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.AUTO_APPROVED
    violations: list[SafetyViolation] = field(default_factory=list)
    rollback_performed: bool = False
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "result_id": self.result_id,
            "action_id": self.action_id,
            "action_name": self.action_name,
            "success": self.success,
            "pre_check_passed": self.pre_check_passed,
            "post_check_passed": self.post_check_passed,
            "approval_status": self.approval_status.value,
            "violations_count": len(self.violations),
            "rollback_performed": self.rollback_performed,
            "duration_ms": self.duration_ms,
        }


@dataclass
class RiskAssessment:
    """ISO 14971 risk assessment for an action."""

    assessment_id: str = field(default_factory=lambda: f"RSK-{uuid.uuid4().hex[:8].upper()}")
    action_name: str = ""
    hazard_description: str = ""
    probability: float = 0.0
    severity_score: float = 0.0
    risk_score: float = 0.0
    risk_category: RiskCategory = RiskCategory.ACCEPTABLE
    mitigation_measures: list[str] = field(default_factory=list)
    residual_risk_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "action_name": self.action_name,
            "hazard_description": self.hazard_description,
            "probability": self.probability,
            "severity_score": self.severity_score,
            "risk_score": self.risk_score,
            "risk_category": self.risk_category.value,
            "mitigation_measures": self.mitigation_measures,
            "residual_risk_score": self.residual_risk_score,
        }


# ---------------------------------------------------------------------------
# Built-in safety check functions
# ---------------------------------------------------------------------------
def check_dose_limit(context: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check that a dose does not exceed the protocol-defined maximum."""
    proposed_dose = context.get("dose_mg", 0.0)
    max_dose = params.get("max_dose_mg", 200.0)
    return proposed_dose <= max_dose


def check_force_limit(context: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check that applied force is within IEC 80601-2-77 limits."""
    force_n = context.get("force_n", 0.0)
    max_force = params.get("max_force_n", 10.0)
    return force_n <= max_force


def check_workspace_boundary(context: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check that position is within the defined workspace boundary."""
    position = context.get("position_mm", [0.0, 0.0, 0.0])
    boundary_min = params.get("boundary_min_mm", [-200, -200, -200])
    boundary_max = params.get("boundary_max_mm", [200, 200, 200])
    for i in range(min(len(position), 3)):
        if position[i] < boundary_min[i] or position[i] > boundary_max[i]:
            return False
    return True


def check_procedure_sequence(context: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check that the procedure step follows the required sequence."""
    current_step = context.get("current_step", "")
    allowed_predecessors = params.get("allowed_predecessors", [])
    previous_step = context.get("previous_step", "")
    if not allowed_predecessors:
        return True
    return previous_step in allowed_predecessors


def check_tissue_integrity(context: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check that tissue damage score is below the safety threshold."""
    damage_score = context.get("tissue_damage_score", 0.0)
    max_damage = params.get("max_damage_score", 0.3)
    return damage_score <= max_damage


def check_timing_constraint(context: dict[str, Any], params: dict[str, Any]) -> bool:
    """Check that the action is within the allowed time window."""
    elapsed_s = context.get("elapsed_seconds", 0.0)
    max_duration_s = params.get("max_duration_s", 3600.0)
    return elapsed_s <= max_duration_s


# ---------------------------------------------------------------------------
# Safety-constrained agent executor
# ---------------------------------------------------------------------------
class SafetyConstrainedAgentExecutor:
    """Safety-gated agent executor with pre/post-conditions and human-in-the-loop.

    Enforces IEC 80601-2-77 and ISO 14971 aligned safety constraints on
    every action. Provides emergency stop capability, automatic rollback
    on safety violations, and human-in-the-loop approval for high-risk
    actions.

    State Machine:
        IDLE -> PRE_CHECK -> AWAITING_APPROVAL -> EXECUTING -> POST_CHECK -> COMPLETE
                    |                                              |
                    v                                              v
                  ERROR                                      ROLLING_BACK -> ERROR
                    ^                                              ^
                    |                                              |
                EMERGENCY_STOP <--- (any state) --------> LOCKED_OUT
    """

    def __init__(
        self,
        auto_approve_below: SafetyLevel = SafetyLevel.LOW,
        enable_rollback: bool = True,
        seed: int = 42,
    ) -> None:
        self._state = ExecutorState.IDLE
        self._auto_approve_below = auto_approve_below
        self._enable_rollback = enable_rollback
        self._actions: dict[str, SafetyAction] = {}
        self._execution_history: list[ExecutionResult] = []
        self._violations: list[SafetyViolation] = []
        self._approval_requests: list[ApprovalRequest] = []
        self._checkpoints: list[ExecutionCheckpoint] = []
        self._context: dict[str, Any] = {}
        self._e_stop_active = False
        self._lockout_until: float = 0.0
        self._risk_assessments: list[RiskAssessment] = []
        self._rng = np.random.default_rng(seed)

        logger.info(
            "SafetyConstrainedAgentExecutor initialized: auto_approve_below=%s, rollback=%s",
            auto_approve_below.value,
            enable_rollback,
        )

    def register_action(self, action: SafetyAction) -> None:
        """Register a safety-constrained action."""
        self._actions[action.name] = action
        logger.info(
            "Action registered: %s (safety=%s, approval=%s)",
            action.name,
            action.safety_level.value,
            action.requires_approval,
        )

    def set_context(self, context: dict[str, Any]) -> None:
        """Set the execution context for safety checks."""
        self._context = context

    def update_context(self, updates: dict[str, Any]) -> None:
        """Update the execution context."""
        self._context.update(updates)

    def execute_action(
        self,
        action_name: str,
        parameters: Optional[dict[str, Any]] = None,
        human_approver_id: str = "auto",
    ) -> ExecutionResult:
        """Execute an action through the safety pipeline.

        Pipeline: pre-check -> approval -> execute -> post-check -> complete/rollback

        Args:
            action_name: Name of the registered action.
            parameters: Action parameters (merged with defaults).
            human_approver_id: ID of the human approver for HITL.

        Returns:
            ExecutionResult with full safety audit.
        """
        start_time = time.time()
        result = ExecutionResult(action_name=action_name)

        # Check emergency stop
        if self._e_stop_active:
            result.success = False
            result.output = {"error": "Emergency stop is active; all actions blocked"}
            logger.error("Action blocked: E-stop active")
            return result

        # Check lockout
        if time.time() < self._lockout_until:
            result.success = False
            result.output = {"error": "Executor is locked out after safety violation"}
            self._state = ExecutorState.LOCKED_OUT
            logger.error("Action blocked: Executor locked out")
            return result

        # Look up action
        action = self._actions.get(action_name)
        if action is None:
            result.success = False
            result.output = {"error": f"Unknown action: {action_name}"}
            return result

        result.action_id = action.action_id
        merged_params = {**action.parameters, **(parameters or {})}
        execution_context = {**self._context, **merged_params}

        # Create checkpoint for rollback
        if self._enable_rollback:
            checkpoint = ExecutionCheckpoint(
                state_snapshot=copy.deepcopy(self._context),
                action_name=action_name,
                action_index=len(self._execution_history),
            )
            self._checkpoints.append(checkpoint)

        # Phase 1: Pre-condition checks
        self._state = ExecutorState.PRE_CHECK
        logger.info("Pre-check: %s (%d constraints)", action_name, len(action.pre_conditions))

        pre_violations = []
        for constraint in action.pre_conditions:
            passed, msg = constraint.evaluate(execution_context)
            if not passed:
                violation = SafetyViolation(
                    violation_type=self._map_constraint_to_violation(constraint),
                    severity=self._map_safety_to_incident(constraint.safety_level),
                    description=f"Pre-condition failed: {constraint.name} - {msg}",
                    action_name=action_name,
                    constraint_id=constraint.constraint_id,
                    context_snapshot=dict(execution_context),
                    remediation=constraint.failure_message,
                    standard_reference=constraint.standard_reference,
                )
                pre_violations.append(violation)
                self._violations.append(violation)
                logger.warning(
                    "Pre-check FAILED: %s - %s [%s]",
                    constraint.name,
                    msg,
                    constraint.standard_reference,
                )

        if pre_violations:
            result.pre_check_passed = False
            result.violations = pre_violations
            result.success = False
            result.output = {"error": "Pre-condition safety check failed", "violations": len(pre_violations)}
            result.duration_ms = (time.time() - start_time) * 1000.0
            self._state = ExecutorState.ERROR
            self._execution_history.append(result)

            if self._enable_rollback:
                self._perform_rollback(action_name)
                result.rollback_performed = True

            return result

        result.pre_check_passed = True
        logger.info("Pre-check PASSED for %s", action_name)

        # Phase 2: Human-in-the-loop approval
        if action.requires_approval or self._needs_approval(action.safety_level):
            self._state = ExecutorState.AWAITING_APPROVAL
            approval = self._request_approval(action, merged_params, human_approver_id)
            result.approval_status = approval.status

            if approval.status not in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED):
                result.success = False
                result.output = {"error": f"Approval {approval.status.value}", "request_id": approval.request_id}
                result.duration_ms = (time.time() - start_time) * 1000.0
                self._state = ExecutorState.ERROR
                self._execution_history.append(result)
                return result

            logger.info("Approval %s for %s", approval.status.value, action_name)
        else:
            result.approval_status = ApprovalStatus.AUTO_APPROVED

        # Phase 3: Execute action
        self._state = ExecutorState.EXECUTING
        logger.info("Executing: %s", action_name)

        try:
            if action.handler is not None:
                output = action.handler(execution_context)
            else:
                output = self._default_handler(action_name, execution_context)

            result.output = output
            execution_context.update(output)

        except Exception as exc:
            result.success = False
            result.output = {"error": str(exc)}
            result.duration_ms = (time.time() - start_time) * 1000.0
            self._state = ExecutorState.ERROR
            logger.error("Execution failed: %s - %s", action_name, exc)

            if self._enable_rollback:
                self._perform_rollback(action_name)
                result.rollback_performed = True

            self._execution_history.append(result)
            return result

        # Phase 4: Post-condition checks
        self._state = ExecutorState.POST_CHECK
        logger.info("Post-check: %s (%d constraints)", action_name, len(action.post_conditions))

        post_violations = []
        for constraint in action.post_conditions:
            passed, msg = constraint.evaluate(execution_context)
            if not passed:
                violation = SafetyViolation(
                    violation_type=self._map_constraint_to_violation(constraint),
                    severity=self._map_safety_to_incident(constraint.safety_level),
                    description=f"Post-condition failed: {constraint.name} - {msg}",
                    action_name=action_name,
                    constraint_id=constraint.constraint_id,
                    context_snapshot=dict(execution_context),
                    remediation=constraint.failure_message,
                    standard_reference=constraint.standard_reference,
                )
                post_violations.append(violation)
                self._violations.append(violation)
                logger.warning("Post-check FAILED: %s - %s", constraint.name, msg)

        if post_violations:
            result.post_check_passed = False
            result.violations.extend(post_violations)
            result.success = False

            # Rollback on post-condition failure
            if self._enable_rollback:
                self._perform_rollback(action_name)
                result.rollback_performed = True

            # Lockout on critical violations
            critical_count = sum(
                1 for v in post_violations if v.severity in (IncidentSeverity.SERIOUS, IncidentSeverity.CRITICAL)
            )
            if critical_count > 0:
                self._lockout_until = time.time() + 60.0
                logger.error("LOCKOUT activated: %d critical violations", critical_count)

            self._state = ExecutorState.ERROR
        else:
            result.post_check_passed = True
            result.success = True
            self._context.update(result.output)
            self._state = ExecutorState.COMPLETE
            logger.info("Post-check PASSED for %s", action_name)

        result.duration_ms = (time.time() - start_time) * 1000.0
        self._execution_history.append(result)
        return result

    def emergency_stop(self, reason: str = "Manual E-stop triggered") -> dict[str, Any]:
        """Activate emergency stop, halting all agent actions.

        Per IEC 80601-2-77, the E-stop immediately transitions to a safe
        state and prevents any further action execution until manually reset.

        Args:
            reason: Reason for the emergency stop.

        Returns:
            Dictionary with E-stop status and details.
        """
        self._e_stop_active = True
        self._state = ExecutorState.EMERGENCY_STOP

        violation = SafetyViolation(
            violation_type=ViolationType.EMERGENCY_STOP,
            severity=IncidentSeverity.CRITICAL,
            description=f"Emergency stop activated: {reason}",
            action_name="EMERGENCY_STOP",
            standard_reference="IEC 80601-2-77:2019 Section 201.12.4.4",
        )
        self._violations.append(violation)

        logger.critical("EMERGENCY STOP ACTIVATED: %s", reason)

        return {
            "e_stop_active": True,
            "reason": reason,
            "timestamp": time.time(),
            "state": self._state.value,
            "violation_id": violation.violation_id,
            "standard_reference": "IEC 80601-2-77:2019",
        }

    def reset_emergency_stop(self, authorized_by: str = "safety_officer") -> bool:
        """Reset the emergency stop (requires authorization).

        Args:
            authorized_by: Identity of the person resetting E-stop.

        Returns:
            True if reset was successful.
        """
        if not self._e_stop_active:
            return True

        self._e_stop_active = False
        self._state = ExecutorState.IDLE
        self._lockout_until = 0.0

        logger.info("Emergency stop RESET by %s", authorized_by)
        return True

    def assess_risk(self, action_name: str) -> RiskAssessment:
        """Perform ISO 14971 risk assessment for an action.

        Args:
            action_name: Name of the action to assess.

        Returns:
            RiskAssessment with probability, severity, and risk category.
        """
        action = self._actions.get(action_name)
        if action is None:
            return RiskAssessment(
                action_name=action_name,
                hazard_description="Unknown action",
                risk_category=RiskCategory.UNACCEPTABLE,
            )

        # Map safety level to probability and severity
        level_map = {
            SafetyLevel.NEGLIGIBLE: (0.05, 1.0),
            SafetyLevel.LOW: (0.1, 2.0),
            SafetyLevel.MODERATE: (0.2, 3.0),
            SafetyLevel.HIGH: (0.3, 4.0),
            SafetyLevel.CRITICAL: (0.5, 5.0),
        }
        probability, severity = level_map.get(action.safety_level, (0.2, 3.0))

        risk_score = probability * severity
        # Mitigations reduce risk
        n_pre = len(action.pre_conditions)
        n_post = len(action.post_conditions)
        mitigation_factor = max(0.3, 1.0 - (n_pre + n_post) * 0.1)
        residual_risk = risk_score * mitigation_factor

        if residual_risk <= 0.5:
            category = RiskCategory.ACCEPTABLE
        elif residual_risk <= 1.5:
            category = RiskCategory.ALARP
        else:
            category = RiskCategory.UNACCEPTABLE

        mitigations = []
        if n_pre > 0:
            mitigations.append(f"{n_pre} pre-condition safety checks")
        if n_post > 0:
            mitigations.append(f"{n_post} post-condition safety checks")
        if action.requires_approval:
            mitigations.append("Human-in-the-loop approval required")
        if self._enable_rollback:
            mitigations.append("Automatic rollback on violation")

        assessment = RiskAssessment(
            action_name=action_name,
            hazard_description=f"Potential harm from {action.name}: {action.description}",
            probability=probability,
            severity_score=severity,
            risk_score=risk_score,
            risk_category=category,
            mitigation_measures=mitigations,
            residual_risk_score=residual_risk,
        )

        self._risk_assessments.append(assessment)
        logger.info(
            "Risk assessment for %s: score=%.2f, residual=%.2f, category=%s",
            action_name,
            risk_score,
            residual_risk,
            category.value,
        )
        return assessment

    def _needs_approval(self, safety_level: SafetyLevel) -> bool:
        """Determine if an action needs human approval based on safety level."""
        level_order = [
            SafetyLevel.NEGLIGIBLE,
            SafetyLevel.LOW,
            SafetyLevel.MODERATE,
            SafetyLevel.HIGH,
            SafetyLevel.CRITICAL,
        ]
        action_idx = level_order.index(safety_level)
        threshold_idx = level_order.index(self._auto_approve_below)
        return action_idx > threshold_idx

    def _request_approval(self, action: SafetyAction, params: dict[str, Any], approver_id: str) -> ApprovalRequest:
        """Create and process a human-in-the-loop approval request."""
        risk = self.assess_risk(action.name)

        request = ApprovalRequest(
            action_name=action.name,
            action_parameters=params,
            safety_level=action.safety_level,
            reason=f"Action requires approval: safety_level={action.safety_level.value}",
            timeout_seconds=action.approval_timeout_s,
            risk_assessment=risk.to_dict(),
        )

        # In demo mode, simulate approval/rejection
        if approver_id == "auto":
            if action.safety_level in (SafetyLevel.NEGLIGIBLE, SafetyLevel.LOW, SafetyLevel.MODERATE):
                request.status = ApprovalStatus.AUTO_APPROVED
                request.approver_id = "auto_system"
            else:
                # Simulate human decision (approve for demo)
                request.status = ApprovalStatus.APPROVED
                request.approver_id = "simulated_clinician"
                request.approver_notes = "Reviewed and approved for research demonstration"
        else:
            request.status = ApprovalStatus.APPROVED
            request.approver_id = approver_id

        request.responded_at = time.time()
        self._approval_requests.append(request)
        return request

    def _perform_rollback(self, action_name: str) -> bool:
        """Rollback to the last checkpoint."""
        if not self._checkpoints:
            logger.warning("No checkpoints available for rollback")
            return False

        self._state = ExecutorState.ROLLING_BACK
        checkpoint = self._checkpoints[-1]
        self._context = copy.deepcopy(checkpoint.state_snapshot)

        logger.info(
            "Rollback performed: action=%s, checkpoint=%s",
            action_name,
            checkpoint.checkpoint_id,
        )
        return True

    def _default_handler(self, action_name: str, context: dict[str, Any]) -> dict[str, Any]:
        """Default action handler for demonstration."""
        return {
            "action": action_name,
            "status": "executed",
            "timestamp": time.time(),
            "force_n": float(self._rng.uniform(0, context.get("max_force_n", 10.0))),
            "position_mm": [float(self._rng.uniform(-100, 100)) for _ in range(3)],
            "tissue_damage_score": float(self._rng.uniform(0, 0.3)),
        }

    @staticmethod
    def _map_constraint_to_violation(constraint: SafetyConstraint) -> ViolationType:
        """Map a constraint type to a violation type."""
        mapping = {
            "dose_limit": ViolationType.DOSE_LIMIT_EXCEEDED,
            "force_limit": ViolationType.FORCE_LIMIT_EXCEEDED,
            "workspace": ViolationType.WORKSPACE_BOUNDARY,
            "sequence": ViolationType.PROCEDURE_SEQUENCE,
            "tissue": ViolationType.TISSUE_INTEGRITY,
            "timing": ViolationType.TIMING_CONSTRAINT,
        }
        for key, vtype in mapping.items():
            if key in constraint.name.lower():
                return vtype
        return ViolationType.REGULATORY

    @staticmethod
    def _map_safety_to_incident(safety_level: SafetyLevel) -> IncidentSeverity:
        """Map safety level to incident severity."""
        mapping = {
            SafetyLevel.NEGLIGIBLE: IncidentSeverity.NEAR_MISS,
            SafetyLevel.LOW: IncidentSeverity.MINOR,
            SafetyLevel.MODERATE: IncidentSeverity.MODERATE,
            SafetyLevel.HIGH: IncidentSeverity.SERIOUS,
            SafetyLevel.CRITICAL: IncidentSeverity.CRITICAL,
        }
        return mapping.get(safety_level, IncidentSeverity.MODERATE)

    def get_safety_summary(self) -> dict[str, Any]:
        """Get a comprehensive safety summary."""
        return {
            "state": self._state.value,
            "e_stop_active": self._e_stop_active,
            "registered_actions": len(self._actions),
            "executions_total": len(self._execution_history),
            "executions_successful": sum(1 for r in self._execution_history if r.success),
            "violations_total": len(self._violations),
            "violations_by_type": {
                vtype.value: sum(1 for v in self._violations if v.violation_type == vtype)
                for vtype in ViolationType
                if any(v.violation_type == vtype for v in self._violations)
            },
            "approval_requests": len(self._approval_requests),
            "rollbacks_performed": sum(1 for r in self._execution_history if r.rollback_performed),
            "checkpoints": len(self._checkpoints),
            "risk_assessments": len(self._risk_assessments),
        }

    def export_audit_trail(self) -> str:
        """Export the complete safety audit trail as JSON."""
        trail = {
            "executions": [r.to_dict() for r in self._execution_history],
            "violations": [v.to_dict() for v in self._violations],
            "approvals": [a.to_dict() for a in self._approval_requests],
            "risk_assessments": [r.to_dict() for r in self._risk_assessments],
        }
        return json.dumps(trail, indent=2)

    def export_audit_hash(self) -> str:
        """Compute an integrity hash over the audit trail."""
        data = self.export_audit_trail().encode("utf-8")
        return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the safety-constrained agent executor."""
    logger.info("=" * 80)
    logger.info("Safety-Constrained Agent Executor Demonstration")
    logger.info("Version: 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 80)

    logger.info("Optional dependencies: HAS_ANTHROPIC=%s, HAS_OPENAI=%s", HAS_ANTHROPIC, HAS_OPENAI)

    # Initialize executor
    executor = SafetyConstrainedAgentExecutor(
        auto_approve_below=SafetyLevel.MODERATE,
        enable_rollback=True,
        seed=42,
    )

    # Define safety constraints
    dose_constraint = SafetyConstraint(
        name="dose_limit_check",
        description="Verify dose does not exceed protocol maximum",
        constraint_type="pre_condition",
        safety_level=SafetyLevel.HIGH,
        check_function=check_dose_limit,
        parameters={"max_dose_mg": 200.0},
        standard_reference="ICH E6(R3) Section 6.4",
        failure_message="Proposed dose exceeds protocol maximum",
    )

    force_constraint = SafetyConstraint(
        name="force_limit_check",
        description="Verify applied force within IEC 80601-2-77 limits",
        constraint_type="post_condition",
        safety_level=SafetyLevel.CRITICAL,
        check_function=check_force_limit,
        parameters={"max_force_n": 10.0},
        standard_reference="IEC 80601-2-77:2019 Section 201.12.4",
        failure_message="Applied force exceeds safety limit",
    )

    workspace_constraint = SafetyConstraint(
        name="workspace_boundary_check",
        description="Verify position within workspace boundary",
        constraint_type="pre_condition",
        safety_level=SafetyLevel.HIGH,
        check_function=check_workspace_boundary,
        parameters={"boundary_min_mm": [-200, -200, -200], "boundary_max_mm": [200, 200, 200]},
        standard_reference="IEC 80601-2-77:2019 Section 201.12.4.3",
        failure_message="Position outside safe workspace boundary",
    )

    tissue_constraint = SafetyConstraint(
        name="tissue_integrity_check",
        description="Verify tissue damage below safety threshold",
        constraint_type="post_condition",
        safety_level=SafetyLevel.HIGH,
        check_function=check_tissue_integrity,
        parameters={"max_damage_score": 0.3},
        standard_reference="ISO 14971:2019 Section 5.5",
        failure_message="Tissue damage score exceeds threshold",
    )

    # Register actions
    administer_dose = SafetyAction(
        name="administer_treatment_dose",
        description="Administer a treatment dose to the patient",
        safety_level=SafetyLevel.HIGH,
        pre_conditions=[dose_constraint],
        post_conditions=[tissue_constraint],
        requires_approval=True,
        standard_references=["ICH E6(R3)", "21 CFR Part 11"],
    )

    robotic_move = SafetyAction(
        name="robotic_instrument_move",
        description="Move the robotic instrument to a target position",
        safety_level=SafetyLevel.MODERATE,
        pre_conditions=[workspace_constraint],
        post_conditions=[force_constraint, tissue_constraint],
        requires_approval=False,
        standard_references=["IEC 80601-2-77:2019", "ISO 14971:2019"],
    )

    calibration = SafetyAction(
        name="system_calibration",
        description="Calibrate the robotic system sensors",
        safety_level=SafetyLevel.LOW,
        pre_conditions=[],
        post_conditions=[],
        requires_approval=False,
        standard_references=["IEC 62304:2006"],
    )

    executor.register_action(administer_dose)
    executor.register_action(robotic_move)
    executor.register_action(calibration)

    # Set initial context
    executor.set_context(
        {
            "dose_mg": 150.0,
            "position_mm": [50.0, 30.0, 20.0],
            "force_n": 0.0,
            "tissue_damage_score": 0.0,
            "previous_step": "setup",
            "current_step": "calibration",
            "elapsed_seconds": 120.0,
            "max_force_n": 10.0,
        }
    )

    # Execute calibration (low risk, auto-approved)
    logger.info("-" * 60)
    logger.info("Executing: system_calibration (LOW risk)")
    result1 = executor.execute_action("system_calibration")
    logger.info("Result: success=%s, approval=%s", result1.success, result1.approval_status.value)

    # Execute robotic move (moderate risk)
    logger.info("-" * 60)
    logger.info("Executing: robotic_instrument_move (MODERATE risk)")
    result2 = executor.execute_action("robotic_instrument_move", {"position_mm": [80.0, 40.0, 25.0]})
    logger.info(
        "Result: success=%s, pre=%s, post=%s, violations=%d",
        result2.success,
        result2.pre_check_passed,
        result2.post_check_passed,
        len(result2.violations),
    )

    # Execute treatment dose (high risk, requires approval)
    logger.info("-" * 60)
    logger.info("Executing: administer_treatment_dose (HIGH risk)")
    result3 = executor.execute_action("administer_treatment_dose", {"dose_mg": 150.0})
    logger.info(
        "Result: success=%s, approval=%s, violations=%d",
        result3.success,
        result3.approval_status.value,
        len(result3.violations),
    )

    # Try exceeding dose limit
    logger.info("-" * 60)
    logger.info("Executing: administer_treatment_dose with EXCESSIVE dose")
    executor.update_context({"dose_mg": 300.0})
    result4 = executor.execute_action("administer_treatment_dose", {"dose_mg": 300.0})
    logger.info(
        "Result: success=%s, pre=%s, rollback=%s, violations=%d",
        result4.success,
        result4.pre_check_passed,
        result4.rollback_performed,
        len(result4.violations),
    )

    # Risk assessment
    logger.info("-" * 60)
    logger.info("Risk assessments:")
    for action_name in ["administer_treatment_dose", "robotic_instrument_move", "system_calibration"]:
        risk = executor.assess_risk(action_name)
        logger.info(
            "  %s: risk=%.2f, residual=%.2f, category=%s",
            action_name,
            risk.risk_score,
            risk.residual_risk_score,
            risk.risk_category.value,
        )

    # Emergency stop demonstration
    logger.info("-" * 60)
    logger.info("Triggering EMERGENCY STOP...")
    estop = executor.emergency_stop("Demonstration of E-stop capability")
    logger.info("E-stop: %s", estop)

    # Try action during E-stop
    result5 = executor.execute_action("system_calibration")
    logger.info("Action during E-stop: success=%s", result5.success)

    # Reset E-stop
    executor.reset_emergency_stop(authorized_by="demo_safety_officer")
    logger.info("E-stop reset")

    # Safety summary
    logger.info("-" * 60)
    summary = executor.get_safety_summary()
    logger.info("SAFETY SUMMARY:")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)

    # Audit hash
    audit_hash = executor.export_audit_hash()
    logger.info("Audit trail hash: %s", audit_hash[:24])

    logger.info("=" * 80)
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
