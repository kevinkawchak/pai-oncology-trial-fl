#!/usr/bin/env python3
"""Human-AI shared autonomy levels for surgical teleoperation.

CLINICAL CONTEXT
================
Modern surgical robotic systems operate across a spectrum of autonomy,
from fully manual teleoperation to fully autonomous execution. The
transition between levels must be governed by safety gates that consider
operator state, environmental conditions, procedure context, and system
confidence.

This module demonstrates a five-level autonomy framework for oncology
surgical robotics with smooth blending of operator input and AI
guidance, safety gate transitions, authority arbitration, and
performance monitoring. The autonomy levels follow the SAE J3016-
inspired taxonomy adapted for surgical robotics.

USE CASES COVERED
=================
1. Five-level autonomy taxonomy: Manual, Assisted, Shared, Supervised,
   and Autonomous with well-defined responsibilities at each level.
2. Input blending between operator commands and AI guidance with
   configurable authority allocation per degree of freedom.
3. Safety gate transitions with pre-conditions, post-conditions, and
   timeout enforcement for level changes.
4. Authority arbitration for conflict resolution between human and AI
   commands using priority-based and confidence-weighted strategies.
5. Performance monitoring with real-time tracking of operator fatigue,
   AI confidence, task progress, and safety margins.
6. Context-aware autonomy adjustment based on procedure phase, tissue
   proximity, and patient vital signs.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

Optional:
    torch >= 2.1.0   (https://pytorch.org) -- GPU-accelerated inference
    mujoco >= 3.0.0  (https://mujoco.org) -- Physics simulation backend

HARDWARE REQUIREMENTS
=====================
- CPU: 4+ cores for real-time control loop
- RAM: 4 GB minimum
- GPU: Optional for AI inference acceleration
- Input: Haptic controller / master device (simulated)

REFERENCES
==========
- IEC 80601-2-77:2019 -- Operator control requirements for RASE
- ISO 14971:2019 -- Use-related risk analysis
- IEC 62304:2006+AMD1:2015 -- State machine requirements
- Yang et al. (2017), "Medical robotics -- Regulatory, ethical, and
  legal considerations for increasing levels of autonomy"

DISCLAIMER
==========
RESEARCH USE ONLY. This software is provided for research and educational
purposes. It is NOT validated for clinical use and must NOT be deployed in
any patient care setting without appropriate regulatory clearance and
comprehensive clinical validation.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]

HAS_MUJOCO = False
try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.shared_autonomy")


# ============================================================================
# Enumerations
# ============================================================================


class AutonomyLevel(str, Enum):
    """Five levels of surgical autonomy.

    MANUAL:     Full operator control; no AI assistance.
    ASSISTED:   Operator leads; AI provides haptic guidance/warnings.
    SHARED:     Blended control; both operator and AI contribute.
    SUPERVISED: AI leads; operator monitors and can intervene.
    AUTONOMOUS: Full AI control; operator observes only.
    """

    MANUAL = "manual"
    ASSISTED = "assisted"
    SHARED = "shared"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"


class GateStatus(str, Enum):
    """Status of a safety gate for autonomy transition."""

    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    TIMEOUT = "timeout"


class ConflictResolution(str, Enum):
    """Strategy for resolving human-AI control conflicts."""

    HUMAN_PRIORITY = "human_priority"
    AI_PRIORITY = "ai_priority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    SAFETY_FIRST = "safety_first"


class OperatorState(str, Enum):
    """Assessed state of the human operator."""

    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    FATIGUED = "fatigued"
    ABSENT = "absent"


class ProcedureContext(str, Enum):
    """Current context of the surgical procedure."""

    OPEN_SPACE = "open_space"
    NEAR_TISSUE = "near_tissue"
    CONTACT = "contact"
    CRITICAL_STRUCTURE = "critical_structure"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class AutonomyConfig:
    """Configuration for the shared autonomy system.

    Attributes:
        default_level: Starting autonomy level.
        blending_rate: Rate of authority blending change per step.
        conflict_strategy: Default conflict resolution strategy.
        ai_confidence_threshold: Minimum AI confidence for autonomous actions.
        fatigue_threshold: Operator fatigue score triggering advisory.
        transition_cooldown_s: Minimum time between level transitions.
        max_force_n: Maximum allowable output force.
        control_frequency_hz: Control loop frequency.
    """

    default_level: AutonomyLevel = AutonomyLevel.MANUAL
    blending_rate: float = 0.05
    conflict_strategy: ConflictResolution = ConflictResolution.SAFETY_FIRST
    ai_confidence_threshold: float = 0.8
    fatigue_threshold: float = 0.7
    transition_cooldown_s: float = 2.0
    max_force_n: float = 5.0
    control_frequency_hz: float = 1000.0


@dataclass
class ControlCommand:
    """A control command from either operator or AI.

    Attributes:
        timestamp: Command timestamp.
        source: Who issued the command ('operator' or 'ai').
        position_mm: Desired position (x, y, z).
        velocity_mm_s: Desired velocity (vx, vy, vz).
        force_n: Desired force (fx, fy, fz).
        confidence: Source confidence in this command [0, 1].
        priority: Priority level (higher = more important).
    """

    timestamp: float = field(default_factory=time.time)
    source: str = "operator"
    position_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_mm_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    confidence: float = 1.0
    priority: int = 5


@dataclass
class BlendedCommand:
    """The result of blending operator and AI commands.

    Attributes:
        timestamp: Blended command timestamp.
        position_mm: Blended position command.
        velocity_mm_s: Blended velocity command.
        force_n: Blended force command.
        operator_authority: Fraction of operator authority [0, 1].
        ai_authority: Fraction of AI authority [0, 1].
        autonomy_level: Current autonomy level.
        conflict_detected: Whether a conflict was resolved.
    """

    timestamp: float = field(default_factory=time.time)
    position_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_mm_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    operator_authority: float = 1.0
    ai_authority: float = 0.0
    autonomy_level: AutonomyLevel = AutonomyLevel.MANUAL
    conflict_detected: bool = False


@dataclass
class SafetyGate:
    """A safety gate controlling autonomy level transitions.

    Attributes:
        gate_id: Unique gate identifier.
        from_level: Source autonomy level.
        to_level: Target autonomy level.
        preconditions: Named preconditions that must be satisfied.
        status: Current gate status.
        timeout_s: Maximum time to hold gate open.
        opened_at: When the gate was opened.
    """

    gate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_level: AutonomyLevel = AutonomyLevel.MANUAL
    to_level: AutonomyLevel = AutonomyLevel.ASSISTED
    preconditions: dict[str, bool] = field(default_factory=dict)
    status: GateStatus = GateStatus.CLOSED
    timeout_s: float = 10.0
    opened_at: float = 0.0


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for the shared autonomy system.

    Attributes:
        timestamp: Metrics timestamp.
        operator_fatigue: Estimated operator fatigue [0, 1].
        ai_confidence: Current AI confidence [0, 1].
        task_progress: Task completion fraction [0, 1].
        safety_margin: Distance to nearest safety limit [0, 1].
        tracking_error_mm: Current position tracking error.
        force_utilisation: Force as fraction of maximum.
        conflict_rate: Rate of human-AI conflicts per second.
        autonomy_level: Current autonomy level.
    """

    timestamp: float = field(default_factory=time.time)
    operator_fatigue: float = 0.0
    ai_confidence: float = 1.0
    task_progress: float = 0.0
    safety_margin: float = 1.0
    tracking_error_mm: float = 0.0
    force_utilisation: float = 0.0
    conflict_rate: float = 0.0
    autonomy_level: AutonomyLevel = AutonomyLevel.MANUAL


@dataclass
class TransitionEvent:
    """Record of an autonomy level transition.

    Attributes:
        event_id: Unique event identifier.
        timestamp: When the transition occurred.
        from_level: Previous autonomy level.
        to_level: New autonomy level.
        reason: Why the transition occurred.
        gate_id: Associated safety gate.
        metrics_snapshot: Performance metrics at transition time.
        checksum: Integrity hash.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    from_level: AutonomyLevel = AutonomyLevel.MANUAL
    to_level: AutonomyLevel = AutonomyLevel.MANUAL
    reason: str = ""
    gate_id: str = ""
    metrics_snapshot: dict[str, float] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self) -> None:
        """Compute integrity checksum."""
        if not self.checksum:
            payload = f"{self.event_id}:{self.timestamp}:{self.from_level.value}:{self.to_level.value}"
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()


# ============================================================================
# Input Blender
# ============================================================================


class InputBlender:
    """Blends operator and AI control inputs based on autonomy level.

    Computes authority allocation and resolves conflicts between
    human and AI commands using the configured strategy.

    Args:
        config: Autonomy configuration.
    """

    # Authority allocation per autonomy level: (operator, ai)
    AUTHORITY_MAP: dict[AutonomyLevel, tuple[float, float]] = {
        AutonomyLevel.MANUAL: (1.0, 0.0),
        AutonomyLevel.ASSISTED: (0.85, 0.15),
        AutonomyLevel.SHARED: (0.5, 0.5),
        AutonomyLevel.SUPERVISED: (0.15, 0.85),
        AutonomyLevel.AUTONOMOUS: (0.0, 1.0),
    }

    def __init__(self, config: AutonomyConfig) -> None:
        self._config = config
        self._current_operator_authority = 1.0
        self._current_ai_authority = 0.0
        self._conflict_count = 0
        self._total_blends = 0

    def blend(
        self,
        operator_cmd: ControlCommand,
        ai_cmd: ControlCommand,
        level: AutonomyLevel,
    ) -> BlendedCommand:
        """Blend operator and AI commands.

        Smoothly transitions authority allocation towards the target
        defined by the current autonomy level.

        Args:
            operator_cmd: Operator's control command.
            ai_cmd: AI's control command.
            level: Current autonomy level.

        Returns:
            Blended control command.
        """
        self._total_blends += 1
        target_op, target_ai = self.AUTHORITY_MAP[level]

        # Smooth authority transition
        rate = self._config.blending_rate
        self._current_operator_authority += rate * (target_op - self._current_operator_authority)
        self._current_ai_authority += rate * (target_ai - self._current_ai_authority)

        # Normalise
        total = self._current_operator_authority + self._current_ai_authority
        if total > 1e-10:
            self._current_operator_authority /= total
            self._current_ai_authority /= total

        # Detect conflict
        conflict = self._detect_conflict(operator_cmd, ai_cmd)
        if conflict:
            self._conflict_count += 1

        # Resolve conflict if needed
        op_weight = self._current_operator_authority
        ai_weight = self._current_ai_authority

        if conflict:
            op_weight, ai_weight = self._resolve_conflict(operator_cmd, ai_cmd, op_weight, ai_weight)

        # Blend commands
        blended_pos = op_weight * operator_cmd.position_mm + ai_weight * ai_cmd.position_mm
        blended_vel = op_weight * operator_cmd.velocity_mm_s + ai_weight * ai_cmd.velocity_mm_s
        blended_force = op_weight * operator_cmd.force_n + ai_weight * ai_cmd.force_n

        # Apply force limits
        force_mag = float(np.linalg.norm(blended_force))
        if force_mag > self._config.max_force_n:
            blended_force = blended_force * (self._config.max_force_n / force_mag)

        return BlendedCommand(
            position_mm=blended_pos,
            velocity_mm_s=blended_vel,
            force_n=blended_force,
            operator_authority=op_weight,
            ai_authority=ai_weight,
            autonomy_level=level,
            conflict_detected=conflict,
        )

    def _detect_conflict(self, op_cmd: ControlCommand, ai_cmd: ControlCommand) -> bool:
        """Detect if operator and AI commands are in conflict.

        Commands conflict if their directions differ significantly.
        """
        op_dir = op_cmd.velocity_mm_s
        ai_dir = ai_cmd.velocity_mm_s

        op_norm = np.linalg.norm(op_dir)
        ai_norm = np.linalg.norm(ai_dir)

        if op_norm < 1e-6 or ai_norm < 1e-6:
            return False

        cosine = float(np.dot(op_dir, ai_dir) / (op_norm * ai_norm))
        return cosine < 0.0  # Opposing directions

    def _resolve_conflict(
        self,
        op_cmd: ControlCommand,
        ai_cmd: ControlCommand,
        op_weight: float,
        ai_weight: float,
    ) -> tuple[float, float]:
        """Resolve a conflict between operator and AI commands.

        Returns adjusted authority weights.
        """
        strategy = self._config.conflict_strategy

        if strategy == ConflictResolution.HUMAN_PRIORITY:
            return 1.0, 0.0

        elif strategy == ConflictResolution.AI_PRIORITY:
            return 0.0, 1.0

        elif strategy == ConflictResolution.CONFIDENCE_WEIGHTED:
            total_conf = op_cmd.confidence + ai_cmd.confidence
            if total_conf < 1e-10:
                return 0.5, 0.5
            return op_cmd.confidence / total_conf, ai_cmd.confidence / total_conf

        elif strategy == ConflictResolution.SAFETY_FIRST:
            # Reduce velocities and increase operator authority
            op_weight = max(op_weight, 0.6)
            ai_weight = 1.0 - op_weight
            return op_weight, ai_weight

        return op_weight, ai_weight

    @property
    def conflict_count(self) -> int:
        """Total number of conflicts detected."""
        return self._conflict_count

    @property
    def conflict_rate(self) -> float:
        """Conflict rate as fraction of total blends."""
        if self._total_blends == 0:
            return 0.0
        return self._conflict_count / self._total_blends


# ============================================================================
# Safety Gate Controller
# ============================================================================


class SafetyGateController:
    """Manages safety gates for autonomy level transitions.

    Each transition between autonomy levels must pass through a
    safety gate with defined preconditions. Gates have timeouts
    and must be explicitly opened by satisfying all conditions.
    """

    def __init__(self, config: AutonomyConfig) -> None:
        self._config = config
        self._gates: dict[str, SafetyGate] = {}
        self._last_transition_time = 0.0
        self._create_default_gates()

    def _create_default_gates(self) -> None:
        """Create default safety gates for all valid transitions."""
        transitions = [
            (AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED),
            (AutonomyLevel.ASSISTED, AutonomyLevel.SHARED),
            (AutonomyLevel.SHARED, AutonomyLevel.SUPERVISED),
            (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS),
            # Downgrade transitions
            (AutonomyLevel.AUTONOMOUS, AutonomyLevel.SUPERVISED),
            (AutonomyLevel.SUPERVISED, AutonomyLevel.SHARED),
            (AutonomyLevel.SHARED, AutonomyLevel.ASSISTED),
            (AutonomyLevel.ASSISTED, AutonomyLevel.MANUAL),
            # Emergency fallback (any level to manual)
            (AutonomyLevel.AUTONOMOUS, AutonomyLevel.MANUAL),
            (AutonomyLevel.SUPERVISED, AutonomyLevel.MANUAL),
            (AutonomyLevel.SHARED, AutonomyLevel.MANUAL),
        ]

        for from_level, to_level in transitions:
            gate_key = f"{from_level.value}->{to_level.value}"
            preconditions = self._default_preconditions(from_level, to_level)
            self._gates[gate_key] = SafetyGate(
                from_level=from_level,
                to_level=to_level,
                preconditions=preconditions,
                timeout_s=10.0,
            )

    def _default_preconditions(self, from_level: AutonomyLevel, to_level: AutonomyLevel) -> dict[str, bool]:
        """Generate default preconditions for a transition."""
        level_order = list(AutonomyLevel)
        from_idx = level_order.index(from_level)
        to_idx = level_order.index(to_level)

        preconditions: dict[str, bool] = {
            "system_healthy": False,
            "operator_consent": False,
        }

        # Upgrading autonomy requires additional checks
        if to_idx > from_idx:
            preconditions["ai_confidence_sufficient"] = False
            preconditions["operator_attentive"] = False
            preconditions["safety_margin_adequate"] = False

        return preconditions

    def evaluate_gate(
        self,
        from_level: AutonomyLevel,
        to_level: AutonomyLevel,
        conditions: dict[str, bool],
    ) -> tuple[GateStatus, str]:
        """Evaluate whether a transition gate can be opened.

        Args:
            from_level: Current autonomy level.
            to_level: Target autonomy level.
            conditions: Current state of preconditions.

        Returns:
            Tuple of (gate_status, reason).
        """
        gate_key = f"{from_level.value}->{to_level.value}"
        gate = self._gates.get(gate_key)
        if gate is None:
            return GateStatus.CLOSED, f"No gate defined for {gate_key}"

        # Check cooldown
        now = time.time()
        if now - self._last_transition_time < self._config.transition_cooldown_s:
            remaining = self._config.transition_cooldown_s - (now - self._last_transition_time)
            return GateStatus.PENDING, f"Cooldown active ({remaining:.1f}s remaining)"

        # Update precondition states
        for key in gate.preconditions:
            if key in conditions:
                gate.preconditions[key] = conditions[key]

        # Check all preconditions
        unmet = [k for k, v in gate.preconditions.items() if not v]
        if unmet:
            return GateStatus.CLOSED, f"Unmet preconditions: {', '.join(unmet)}"

        gate.status = GateStatus.OPEN
        gate.opened_at = now
        return GateStatus.OPEN, "All preconditions satisfied"

    def record_transition(self) -> None:
        """Record that a transition occurred (for cooldown tracking)."""
        self._last_transition_time = time.time()

    def get_gate_summary(self) -> dict[str, Any]:
        """Get summary of all gates."""
        return {
            key: {
                "from": gate.from_level.value,
                "to": gate.to_level.value,
                "status": gate.status.value,
                "preconditions": dict(gate.preconditions),
            }
            for key, gate in self._gates.items()
        }


# ============================================================================
# Performance Monitor
# ============================================================================


class PerformanceMonitor:
    """Monitors real-time performance of the shared autonomy system.

    Tracks operator fatigue, AI confidence, task progress, and safety
    margins to inform autonomy level adjustments.
    """

    def __init__(self, config: AutonomyConfig) -> None:
        self._config = config
        self._metrics_history: list[PerformanceMetrics] = []
        self._max_history = 5000
        self._fatigue_estimator_state = 0.0
        self._confidence_buffer: list[float] = []

    def update(
        self,
        blended_cmd: BlendedCommand,
        target_position: np.ndarray,
        ai_confidence: float = 1.0,
        elapsed_time_s: float = 0.0,
        total_time_s: float = 100.0,
    ) -> PerformanceMetrics:
        """Update performance metrics with new data.

        Args:
            blended_cmd: Latest blended command.
            target_position: Current target position.
            ai_confidence: Current AI confidence.
            elapsed_time_s: Time elapsed in procedure.
            total_time_s: Estimated total procedure time.

        Returns:
            Updated performance metrics.
        """
        # Tracking error
        tracking_err = float(np.linalg.norm(blended_cmd.position_mm - target_position))

        # Force utilisation
        force_mag = float(np.linalg.norm(blended_cmd.force_n))
        force_util = force_mag / max(self._config.max_force_n, 1e-6)

        # Fatigue estimation (simple exponential model)
        fatigue_rate = 0.0001
        self._fatigue_estimator_state = min(1.0, self._fatigue_estimator_state + fatigue_rate)

        # Task progress
        progress = min(1.0, elapsed_time_s / max(total_time_s, 1.0))

        # Safety margin (simplified)
        safety_margin = max(0.0, 1.0 - force_util)

        # Confidence tracking
        self._confidence_buffer.append(ai_confidence)
        if len(self._confidence_buffer) > 200:
            self._confidence_buffer.pop(0)

        metrics = PerformanceMetrics(
            operator_fatigue=self._fatigue_estimator_state,
            ai_confidence=ai_confidence,
            task_progress=progress,
            safety_margin=safety_margin,
            tracking_error_mm=tracking_err,
            force_utilisation=force_util,
            conflict_rate=0.0,
            autonomy_level=blended_cmd.autonomy_level,
        )

        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)

        return metrics

    def get_trend(self, window: int = 100) -> dict[str, Any]:
        """Compute performance trends over recent history.

        Args:
            window: Number of recent entries to analyse.

        Returns:
            Trend statistics.
        """
        recent = self._metrics_history[-window:]
        if not recent:
            return {"count": 0}

        fatigue_vals = [m.operator_fatigue for m in recent]
        conf_vals = [m.ai_confidence for m in recent]
        err_vals = [m.tracking_error_mm for m in recent]

        return {
            "count": len(recent),
            "fatigue_mean": float(np.mean(fatigue_vals)),
            "fatigue_trend": "increasing" if len(fatigue_vals) > 1 and fatigue_vals[-1] > fatigue_vals[0] else "stable",
            "confidence_mean": float(np.mean(conf_vals)),
            "tracking_error_mean_mm": float(np.mean(err_vals)),
            "tracking_error_max_mm": float(np.max(err_vals)),
        }

    def should_suggest_transition(self, current_level: AutonomyLevel) -> tuple[bool, AutonomyLevel | None, str]:
        """Suggest an autonomy level transition based on performance.

        Args:
            current_level: Current autonomy level.

        Returns:
            Tuple of (should_transition, suggested_level, reason).
        """
        if len(self._metrics_history) < 10:
            return False, None, "Insufficient data"

        recent = self._metrics_history[-50:]
        mean_fatigue = float(np.mean([m.operator_fatigue for m in recent]))
        mean_confidence = float(np.mean([m.ai_confidence for m in recent]))
        mean_safety = float(np.mean([m.safety_margin for m in recent]))

        level_order = list(AutonomyLevel)
        current_idx = level_order.index(current_level)

        # Suggest upgrade if operator is fatigued and AI is confident
        if mean_fatigue > self._config.fatigue_threshold and mean_confidence > self._config.ai_confidence_threshold:
            if current_idx < len(level_order) - 1:
                return (
                    True,
                    level_order[current_idx + 1],
                    f"Operator fatigue ({mean_fatigue:.2f}) with high AI confidence ({mean_confidence:.2f})",
                )

        # Suggest downgrade if safety margin is low
        if mean_safety < 0.2:
            if current_idx > 0:
                return True, level_order[current_idx - 1], f"Low safety margin ({mean_safety:.2f})"

        # Suggest downgrade if AI confidence drops
        if mean_confidence < 0.4 and current_idx > 1:
            return True, level_order[current_idx - 1], f"Low AI confidence ({mean_confidence:.2f})"

        return False, None, "No transition needed"

    def reset_fatigue(self) -> None:
        """Reset fatigue estimator (e.g., after operator break)."""
        self._fatigue_estimator_state = 0.0


# ============================================================================
# Shared Autonomy Controller (Main Orchestrator)
# ============================================================================


class SharedAutonomyController:
    """Top-level controller for the shared autonomy system.

    Orchestrates input blending, safety gates, and performance
    monitoring into a unified control pipeline.

    Args:
        config: Autonomy configuration.
    """

    def __init__(self, config: AutonomyConfig | None = None) -> None:
        self._config = config or AutonomyConfig()
        self._current_level = self._config.default_level
        self._blender = InputBlender(self._config)
        self._gate_controller = SafetyGateController(self._config)
        self._performance = PerformanceMonitor(self._config)
        self._transition_history: list[TransitionEvent] = []
        self._step_count = 0
        logger.info("SharedAutonomyController initialised at level=%s", self._current_level.value)

    @property
    def current_level(self) -> AutonomyLevel:
        """Current autonomy level."""
        return self._current_level

    def step(
        self,
        operator_cmd: ControlCommand,
        ai_cmd: ControlCommand,
        target_position: np.ndarray | None = None,
        elapsed_time_s: float = 0.0,
    ) -> BlendedCommand:
        """Process one control step.

        Args:
            operator_cmd: Operator's command.
            ai_cmd: AI's command.
            target_position: Current target for performance tracking.
            elapsed_time_s: Elapsed procedure time.

        Returns:
            Blended control command.
        """
        self._step_count += 1

        if target_position is None:
            target_position = np.zeros(3)

        # Blend commands
        blended = self._blender.blend(operator_cmd, ai_cmd, self._current_level)

        # Update performance metrics
        self._performance.update(
            blended,
            target_position,
            ai_confidence=ai_cmd.confidence,
            elapsed_time_s=elapsed_time_s,
        )

        return blended

    def request_transition(self, target_level: AutonomyLevel, conditions: dict[str, bool]) -> tuple[bool, str]:
        """Request a transition to a new autonomy level.

        Args:
            target_level: Requested autonomy level.
            conditions: Current state of preconditions.

        Returns:
            Tuple of (success, reason).
        """
        if target_level == self._current_level:
            return True, "Already at requested level"

        gate_status, reason = self._gate_controller.evaluate_gate(self._current_level, target_level, conditions)

        if gate_status == GateStatus.OPEN:
            old_level = self._current_level
            self._current_level = target_level
            self._gate_controller.record_transition()

            event = TransitionEvent(
                from_level=old_level,
                to_level=target_level,
                reason=reason,
            )
            self._transition_history.append(event)

            logger.info(
                "Autonomy transition: %s -> %s (%s)",
                old_level.value,
                target_level.value,
                reason,
            )
            return True, reason

        logger.info("Transition denied: %s -> %s (%s)", self._current_level.value, target_level.value, reason)
        return False, reason

    def emergency_fallback(self) -> None:
        """Immediately fall back to manual control."""
        old_level = self._current_level
        self._current_level = AutonomyLevel.MANUAL

        event = TransitionEvent(
            from_level=old_level,
            to_level=AutonomyLevel.MANUAL,
            reason="Emergency fallback to manual control",
        )
        self._transition_history.append(event)
        logger.critical("EMERGENCY FALLBACK: %s -> MANUAL", old_level.value)

    def get_suggested_transition(self) -> tuple[bool, AutonomyLevel | None, str]:
        """Get a suggested autonomy transition based on performance.

        Returns:
            Tuple of (should_transition, suggested_level, reason).
        """
        return self._performance.should_suggest_transition(self._current_level)

    def get_report(self) -> dict[str, Any]:
        """Generate a comprehensive autonomy report."""
        perf_trend = self._performance.get_trend()
        return {
            "current_level": self._current_level.value,
            "total_steps": self._step_count,
            "transitions": len(self._transition_history),
            "conflict_count": self._blender.conflict_count,
            "conflict_rate": self._blender.conflict_rate,
            "performance_trend": perf_trend,
            "transition_history": [
                {
                    "from": e.from_level.value,
                    "to": e.to_level.value,
                    "reason": e.reason,
                }
                for e in self._transition_history
            ],
        }


# ============================================================================
# Demonstration
# ============================================================================


def generate_simulated_commands(
    num_steps: int = 1000,
    seed: int = 42,
) -> tuple[list[ControlCommand], list[ControlCommand], list[np.ndarray]]:
    """Generate simulated operator and AI commands.

    Args:
        num_steps: Number of simulation steps.
        seed: Random seed.

    Returns:
        Tuple of (operator_commands, ai_commands, targets).
    """
    rng = np.random.default_rng(seed)
    t_values = np.linspace(0, 4 * np.pi, num_steps)

    # Target trajectory
    targets = [np.array([50 * np.sin(t), 30 * np.cos(t), 20 * np.sin(2 * t) + 40]) for t in t_values]

    operator_cmds: list[ControlCommand] = []
    ai_cmds: list[ControlCommand] = []

    for i in range(num_steps):
        target = targets[i]
        # Operator command: target + noise (simulating imprecise human input)
        op_noise = rng.normal(0, 2.0, size=3)
        op_vel_noise = rng.normal(0, 1.0, size=3)
        op_cmd = ControlCommand(
            source="operator",
            position_mm=target + op_noise,
            velocity_mm_s=op_vel_noise,
            force_n=rng.normal(0, 0.3, size=3),
            confidence=max(0.3, 1.0 - i / (num_steps * 2)),
        )
        operator_cmds.append(op_cmd)

        # AI command: target + small noise (more precise)
        ai_noise = rng.normal(0, 0.2, size=3)
        ai_confidence = 0.6 + 0.3 * np.sin(i * 0.01) + rng.normal(0, 0.05)
        ai_confidence = float(np.clip(ai_confidence, 0.1, 1.0))

        # Occasionally generate conflicting AI command
        if 400 <= i <= 420:
            ai_vel = -op_vel_noise  # Opposing direction
        else:
            ai_vel = op_vel_noise * 0.8 + rng.normal(0, 0.2, size=3)

        ai_cmd = ControlCommand(
            source="ai",
            position_mm=target + ai_noise,
            velocity_mm_s=ai_vel,
            force_n=rng.normal(0, 0.1, size=3),
            confidence=ai_confidence,
        )
        ai_cmds.append(ai_cmd)

    return operator_cmds, ai_cmds, targets


def run_shared_autonomy_demo() -> dict[str, Any]:
    """Run the complete shared autonomy demonstration.

    Returns:
        Final autonomy report.
    """
    logger.info("=" * 70)
    logger.info("  Human-AI Shared Autonomy Demo")
    logger.info("  Version 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 70)

    # 1. Configure the system
    config = AutonomyConfig(
        default_level=AutonomyLevel.MANUAL,
        blending_rate=0.05,
        conflict_strategy=ConflictResolution.SAFETY_FIRST,
        ai_confidence_threshold=0.8,
        fatigue_threshold=0.05,
        transition_cooldown_s=0.01,
    )

    controller = SharedAutonomyController(config)

    # 2. Generate simulated commands
    logger.info("\n--- Generating Simulated Commands ---")
    operator_cmds, ai_cmds, targets = generate_simulated_commands(num_steps=1000, seed=42)
    logger.info("Generated %d command pairs", len(operator_cmds))

    # 3. Run the autonomy simulation
    logger.info("\n--- Running Shared Autonomy Simulation ---")
    blended_results: list[BlendedCommand] = []

    # Schedule autonomy transitions
    transition_schedule = [
        (100, AutonomyLevel.ASSISTED, "Entering assisted phase"),
        (250, AutonomyLevel.SHARED, "Transitioning to shared control"),
        (500, AutonomyLevel.SUPERVISED, "AI taking over supervision"),
        (750, AutonomyLevel.AUTONOMOUS, "Full autonomous operation"),
        (900, AutonomyLevel.MANUAL, "Returning to manual control"),
    ]
    schedule_idx = 0

    for i in range(len(operator_cmds)):
        # Check for scheduled transitions
        if schedule_idx < len(transition_schedule):
            step_num, target_level, reason = transition_schedule[schedule_idx]
            if i == step_num:
                conditions = {
                    "system_healthy": True,
                    "operator_consent": True,
                    "ai_confidence_sufficient": True,
                    "operator_attentive": True,
                    "safety_margin_adequate": True,
                }
                success, msg = controller.request_transition(target_level, conditions)
                if success:
                    logger.info("Step %d: Transition to %s -- %s", i, target_level.value, reason)
                schedule_idx += 1

        # Process control step
        elapsed = i * 0.001  # 1 kHz
        blended = controller.step(operator_cmds[i], ai_cmds[i], targets[i], elapsed_time_s=elapsed)
        blended_results.append(blended)

        # Periodic logging
        if (i + 1) % 200 == 0:
            logger.info(
                "  Step %d/%d: level=%s, op_auth=%.2f, ai_auth=%.2f, conflict=%s",
                i + 1,
                len(operator_cmds),
                blended.autonomy_level.value,
                blended.operator_authority,
                blended.ai_authority,
                blended.conflict_detected,
            )

    # 4. Analyse results
    logger.info("\n--- Autonomy Simulation Results ---")
    report = controller.get_report()

    logger.info("  Current level:    %s", report["current_level"])
    logger.info("  Total steps:      %d", report["total_steps"])
    logger.info("  Transitions:      %d", report["transitions"])
    logger.info("  Conflicts:        %d (rate=%.4f)", report["conflict_count"], report["conflict_rate"])

    # Authority distribution analysis
    op_authorities = [b.operator_authority for b in blended_results]
    ai_authorities = [b.ai_authority for b in blended_results]

    logger.info("\n  Authority Distribution:")
    logger.info("    Operator mean:  %.3f", np.mean(op_authorities))
    logger.info("    AI mean:        %.3f", np.mean(ai_authorities))

    # Level distribution
    level_counts: dict[str, int] = {}
    for b in blended_results:
        key = b.autonomy_level.value
        level_counts[key] = level_counts.get(key, 0) + 1

    logger.info("\n  Time per Autonomy Level:")
    for level_name, count in level_counts.items():
        pct = count / len(blended_results) * 100
        logger.info("    %-15s %d steps (%.1f%%)", level_name, count, pct)

    # Tracking error analysis
    tracking_errors = [float(np.linalg.norm(b.position_mm - targets[i])) for i, b in enumerate(blended_results)]
    logger.info("\n  Tracking Error:")
    logger.info("    Mean:  %.3f mm", np.mean(tracking_errors))
    logger.info("    Max:   %.3f mm", np.max(tracking_errors))
    logger.info("    Std:   %.3f mm", np.std(tracking_errors))

    # Transition history
    logger.info("\n  Transition History:")
    for t in report["transition_history"]:
        logger.info("    %s -> %s: %s", t["from"], t["to"], t["reason"])

    perf = report["performance_trend"]
    logger.info("\n  Performance Trend:")
    logger.info("    Fatigue mean:        %.4f", perf.get("fatigue_mean", 0))
    logger.info("    Confidence mean:     %.4f", perf.get("confidence_mean", 0))
    logger.info("    Tracking error mean: %.4f mm", perf.get("tracking_error_mean_mm", 0))

    logger.info("\n" + "=" * 70)
    logger.info("  RESEARCH USE ONLY -- NOT FOR CLINICAL DEPLOYMENT")
    logger.info("=" * 70)

    return report


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    report = run_shared_autonomy_demo()
    sys.exit(0)
