#!/usr/bin/env python3
"""ROS 2 surgical deployment pattern for robotic oncology systems.

CLINICAL CONTEXT
================
Deploying surgical robotic systems in operating theatres requires a
robust middleware layer for real-time communication, lifecycle management,
and fault tolerance. ROS 2 (Robot Operating System 2) provides the
DDS-based communication infrastructure, but integrating it with
clinical safety requirements demands careful architectural design.

This module demonstrates the deployment pipeline for a surgical robotic
system built on ROS 2, including node lifecycle management, service
discovery, action servers for multi-step procedure execution, and
health monitoring. All ROS 2 interfaces are mocked for portability.

USE CASES COVERED
=================
1. ROS 2 node lifecycle management (configure, activate, deactivate,
   shutdown) for surgical subsystems.
2. Service discovery and registration for multi-node surgical systems.
3. Action server pattern for long-running procedure execution with
   feedback and cancellation support.
4. Health monitoring with heartbeat-based node status tracking.
5. Topic-based telemetry streaming with QoS configuration.
6. Parameter management for runtime reconfiguration of safety limits.
7. Graceful degradation and fault recovery for node failures.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

Optional:
    torch >= 2.1.0   (https://pytorch.org) -- GPU-accelerated inference
    mujoco >= 3.0.0  (https://mujoco.org) -- Physics simulation backend

HARDWARE REQUIREMENTS
=====================
- CPU: 4+ cores for concurrent node execution
- RAM: 4 GB minimum
- GPU: Not required
- Network: DDS-compatible network (simulated in this example)

REFERENCES
==========
- IEC 80601-2-77:2019 -- System integration for RASE
- ISO 14971:2019 -- Risk management for software-controlled systems
- IEC 62304:2006+AMD1:2015 -- Software architecture requirements
- ROS 2 Humble Hawksbill (https://docs.ros.org/en/humble/)

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
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol

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

HAS_RCLPY = False
try:
    import rclpy

    HAS_RCLPY = True
except ImportError:
    rclpy = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.ros2_deployment")


# ============================================================================
# Enumerations
# ============================================================================


class NodeLifecycleState(str, Enum):
    """ROS 2 managed node lifecycle states."""

    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    ERROR = "error"


class TransitionResult(str, Enum):
    """Result of a lifecycle transition."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class QoSProfile(str, Enum):
    """Quality of Service profiles for ROS 2 communication."""

    SENSOR_DATA = "sensor_data"
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"
    SYSTEM_DEFAULT = "system_default"


class ActionStatus(str, Enum):
    """Status of an action server goal."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    ABORTED = "aborted"


class NodeHealth(str, Enum):
    """Health status of a ROS 2 node."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProcedurePhase(str, Enum):
    """Phases of a surgical procedure execution."""

    PREPARATION = "preparation"
    POSITIONING = "positioning"
    APPROACH = "approach"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    RETRACTION = "retraction"
    COMPLETION = "completion"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class NodeDescriptor:
    """Descriptor for a ROS 2 node in the surgical system.

    Attributes:
        node_name: Unique node name.
        namespace: ROS 2 namespace.
        node_type: Functional category of the node.
        lifecycle_state: Current lifecycle state.
        health: Current health status.
        last_heartbeat: Time of last heartbeat received.
        parameters: Node parameters.
        publishers: List of published topics.
        subscribers: List of subscribed topics.
        services: List of provided services.
    """

    node_name: str = ""
    namespace: str = "/surgical"
    node_type: str = "generic"
    lifecycle_state: NodeLifecycleState = NodeLifecycleState.UNCONFIGURED
    health: NodeHealth = NodeHealth.UNKNOWN
    last_heartbeat: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    publishers: list[str] = field(default_factory=list)
    subscribers: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=list)


@dataclass
class TopicMessage:
    """A message published on a ROS 2 topic.

    Attributes:
        topic: Topic name.
        timestamp: Publication timestamp.
        data: Message payload.
        qos: Quality of service profile used.
        publisher_node: Name of the publishing node.
    """

    topic: str = ""
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    qos: QoSProfile = QoSProfile.SYSTEM_DEFAULT
    publisher_node: str = ""


@dataclass
class ServiceRequest:
    """A ROS 2 service request.

    Attributes:
        service_name: Name of the service.
        request_id: Unique request identifier.
        client_node: Requesting node name.
        parameters: Request parameters.
        timestamp: Request timestamp.
    """

    service_name: str = ""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_node: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ServiceResponse:
    """A ROS 2 service response.

    Attributes:
        request_id: Matching request identifier.
        success: Whether the service call succeeded.
        result: Response data.
        error_message: Error description if failed.
        processing_time_ms: Time taken to process the request.
    """

    request_id: str = ""
    success: bool = True
    result: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    processing_time_ms: float = 0.0


@dataclass
class ActionGoal:
    """A ROS 2 action goal for procedure execution.

    Attributes:
        goal_id: Unique goal identifier.
        action_name: Name of the action.
        parameters: Goal parameters.
        status: Current goal status.
        feedback: Accumulated feedback messages.
        result: Final result when completed.
        start_time: When execution started.
        end_time: When execution completed.
    """

    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    feedback: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class DeploymentEvent:
    """An event in the deployment audit trail.

    Attributes:
        event_id: Unique event identifier.
        timestamp: When the event occurred.
        event_type: Category of event.
        node_name: Associated node.
        description: Human-readable description.
        metadata: Additional structured data.
        checksum: Integrity hash.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    node_name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self) -> None:
        """Compute integrity checksum."""
        if not self.checksum:
            payload = f"{self.event_id}:{self.timestamp}:{self.event_type}:{self.node_name}"
            self.checksum = hashlib.sha256(payload.encode()).hexdigest()


# ============================================================================
# Mock ROS 2 Node
# ============================================================================


class MockNode:
    """Mock ROS 2 node with lifecycle management.

    Simulates a ROS 2 managed (lifecycle) node with configurable
    transition callbacks and parameter management.

    Args:
        descriptor: Node descriptor with configuration.
    """

    def __init__(self, descriptor: NodeDescriptor) -> None:
        self._descriptor = descriptor
        self._topic_buffers: dict[str, list[TopicMessage]] = {}
        self._service_handlers: dict[str, Any] = {}
        self._transition_log: list[dict[str, Any]] = []
        self._created_at = time.time()
        logger.info("Node '%s/%s' created (type=%s)", descriptor.namespace, descriptor.node_name, descriptor.node_type)

    @property
    def name(self) -> str:
        """Full node name with namespace."""
        return f"{self._descriptor.namespace}/{self._descriptor.node_name}"

    @property
    def state(self) -> NodeLifecycleState:
        """Current lifecycle state."""
        return self._descriptor.lifecycle_state

    @property
    def descriptor(self) -> NodeDescriptor:
        """Node descriptor."""
        return self._descriptor

    def configure(self) -> TransitionResult:
        """Transition: unconfigured -> inactive.

        Returns:
            Transition result.
        """
        if self._descriptor.lifecycle_state != NodeLifecycleState.UNCONFIGURED:
            logger.warning("Cannot configure node '%s' from state '%s'", self.name, self.state.value)
            return TransitionResult.FAILURE

        self._descriptor.lifecycle_state = NodeLifecycleState.INACTIVE
        self._log_transition("configure", NodeLifecycleState.UNCONFIGURED, NodeLifecycleState.INACTIVE)
        logger.info("Node '%s' configured -> INACTIVE", self.name)
        return TransitionResult.SUCCESS

    def activate(self) -> TransitionResult:
        """Transition: inactive -> active.

        Returns:
            Transition result.
        """
        if self._descriptor.lifecycle_state != NodeLifecycleState.INACTIVE:
            logger.warning("Cannot activate node '%s' from state '%s'", self.name, self.state.value)
            return TransitionResult.FAILURE

        self._descriptor.lifecycle_state = NodeLifecycleState.ACTIVE
        self._descriptor.health = NodeHealth.HEALTHY
        self._descriptor.last_heartbeat = time.time()
        self._log_transition("activate", NodeLifecycleState.INACTIVE, NodeLifecycleState.ACTIVE)
        logger.info("Node '%s' activated -> ACTIVE", self.name)
        return TransitionResult.SUCCESS

    def deactivate(self) -> TransitionResult:
        """Transition: active -> inactive.

        Returns:
            Transition result.
        """
        if self._descriptor.lifecycle_state != NodeLifecycleState.ACTIVE:
            logger.warning("Cannot deactivate node '%s' from state '%s'", self.name, self.state.value)
            return TransitionResult.FAILURE

        self._descriptor.lifecycle_state = NodeLifecycleState.INACTIVE
        self._descriptor.health = NodeHealth.UNKNOWN
        self._log_transition("deactivate", NodeLifecycleState.ACTIVE, NodeLifecycleState.INACTIVE)
        logger.info("Node '%s' deactivated -> INACTIVE", self.name)
        return TransitionResult.SUCCESS

    def shutdown(self) -> TransitionResult:
        """Transition: any -> finalized.

        Returns:
            Transition result.
        """
        old_state = self._descriptor.lifecycle_state
        self._descriptor.lifecycle_state = NodeLifecycleState.FINALIZED
        self._descriptor.health = NodeHealth.UNKNOWN
        self._log_transition("shutdown", old_state, NodeLifecycleState.FINALIZED)
        logger.info("Node '%s' shut down -> FINALIZED", self.name)
        return TransitionResult.SUCCESS

    def publish(self, topic: str, data: dict[str, Any], qos: QoSProfile = QoSProfile.SYSTEM_DEFAULT) -> bool:
        """Publish a message to a topic.

        Args:
            topic: Topic name.
            data: Message payload.
            qos: QoS profile.

        Returns:
            True if published successfully.
        """
        if self._descriptor.lifecycle_state != NodeLifecycleState.ACTIVE:
            return False

        msg = TopicMessage(
            topic=topic,
            data=data,
            qos=qos,
            publisher_node=self._descriptor.node_name,
        )

        if topic not in self._topic_buffers:
            self._topic_buffers[topic] = []
        self._topic_buffers[topic].append(msg)

        # Trim buffer
        if len(self._topic_buffers[topic]) > 1000:
            self._topic_buffers[topic] = self._topic_buffers[topic][-500:]

        return True

    def get_topic_messages(self, topic: str, count: int = 10) -> list[TopicMessage]:
        """Retrieve recent messages from a topic buffer.

        Args:
            topic: Topic name.
            count: Maximum messages to return.

        Returns:
            List of recent messages.
        """
        return self._topic_buffers.get(topic, [])[-count:]

    def register_service(self, service_name: str, handler: Any) -> None:
        """Register a service handler.

        Args:
            service_name: Service name.
            handler: Callable service handler.
        """
        self._service_handlers[service_name] = handler
        self._descriptor.services.append(service_name)
        logger.info("Node '%s' registered service '%s'", self.name, service_name)

    def call_service(self, request: ServiceRequest) -> ServiceResponse:
        """Handle a service request.

        Args:
            request: The service request.

        Returns:
            Service response.
        """
        start = time.time()
        handler = self._service_handlers.get(request.service_name)
        if handler is None:
            return ServiceResponse(
                request_id=request.request_id,
                success=False,
                error_message=f"Service '{request.service_name}' not registered",
            )

        try:
            result = handler(request.parameters)
            elapsed = (time.time() - start) * 1000
            return ServiceResponse(
                request_id=request.request_id,
                success=True,
                result=result if isinstance(result, dict) else {"value": result},
                processing_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return ServiceResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(exc),
                processing_time_ms=elapsed,
            )

    def heartbeat(self) -> None:
        """Send a heartbeat to update health status."""
        self._descriptor.last_heartbeat = time.time()
        if self._descriptor.lifecycle_state == NodeLifecycleState.ACTIVE:
            self._descriptor.health = NodeHealth.HEALTHY

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a node parameter."""
        self._descriptor.parameters[name] = value

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a node parameter."""
        return self._descriptor.parameters.get(name, default)

    def _log_transition(self, transition: str, from_state: NodeLifecycleState, to_state: NodeLifecycleState) -> None:
        """Log a state transition."""
        self._transition_log.append(
            {
                "transition": transition,
                "from": from_state.value,
                "to": to_state.value,
                "timestamp": time.time(),
            }
        )

    def get_transition_history(self) -> list[dict[str, Any]]:
        """Return the transition history."""
        return list(self._transition_log)


# ============================================================================
# Lifecycle Manager
# ============================================================================


class LifecycleManager:
    """Manages the lifecycle of multiple ROS 2 nodes.

    Provides coordinated startup, shutdown, and health monitoring
    for the surgical robotic system's node graph.

    Args:
        heartbeat_timeout_s: Time without heartbeat before marking unhealthy.
    """

    def __init__(self, heartbeat_timeout_s: float = 2.0) -> None:
        self._nodes: dict[str, MockNode] = {}
        self._heartbeat_timeout = heartbeat_timeout_s
        self._events: list[DeploymentEvent] = []
        self._startup_order: list[str] = []

    def register_node(self, node: MockNode) -> None:
        """Register a node with the lifecycle manager.

        Args:
            node: The node to register.
        """
        self._nodes[node.name] = node
        self._startup_order.append(node.name)
        self._record_event("node_registered", node.descriptor.node_name, f"Node registered: {node.name}")
        logger.info("LifecycleManager: registered node '%s'", node.name)

    def startup_all(self) -> dict[str, TransitionResult]:
        """Configure and activate all registered nodes in order.

        Returns:
            Dictionary of node_name to transition result.
        """
        results: dict[str, TransitionResult] = {}
        logger.info("Starting up %d nodes...", len(self._startup_order))

        for name in self._startup_order:
            node = self._nodes[name]

            # Configure
            config_result = node.configure()
            if config_result != TransitionResult.SUCCESS:
                results[name] = config_result
                logger.error("Failed to configure node '%s'", name)
                continue

            # Activate
            activate_result = node.activate()
            results[name] = activate_result
            if activate_result == TransitionResult.SUCCESS:
                self._record_event("node_activated", node.descriptor.node_name, f"Node activated: {name}")
            else:
                logger.error("Failed to activate node '%s'", name)

        active_count = sum(1 for r in results.values() if r == TransitionResult.SUCCESS)
        logger.info("Startup complete: %d/%d nodes active", active_count, len(self._startup_order))
        return results

    def shutdown_all(self) -> dict[str, TransitionResult]:
        """Shut down all nodes in reverse order.

        Returns:
            Dictionary of node_name to transition result.
        """
        results: dict[str, TransitionResult] = {}
        logger.info("Shutting down %d nodes...", len(self._startup_order))

        for name in reversed(self._startup_order):
            node = self._nodes[name]
            result = node.shutdown()
            results[name] = result
            self._record_event("node_shutdown", node.descriptor.node_name, f"Node shut down: {name}")

        logger.info("Shutdown complete.")
        return results

    def check_health(self) -> dict[str, NodeHealth]:
        """Check health of all registered nodes.

        Returns:
            Dictionary of node_name to health status.
        """
        now = time.time()
        health_map: dict[str, NodeHealth] = {}

        for name, node in self._nodes.items():
            desc = node.descriptor
            if desc.lifecycle_state != NodeLifecycleState.ACTIVE:
                health_map[name] = NodeHealth.UNKNOWN
            elif now - desc.last_heartbeat > self._heartbeat_timeout:
                desc.health = NodeHealth.UNHEALTHY
                health_map[name] = NodeHealth.UNHEALTHY
            else:
                health_map[name] = desc.health

        return health_map

    def get_active_nodes(self) -> list[str]:
        """Return names of all active nodes."""
        return [
            name for name, node in self._nodes.items() if node.descriptor.lifecycle_state == NodeLifecycleState.ACTIVE
        ]

    def _record_event(self, event_type: str, node_name: str, description: str) -> None:
        """Record a deployment event."""
        self._events.append(
            DeploymentEvent(
                event_type=event_type,
                node_name=node_name,
                description=description,
            )
        )

    def get_events(self) -> list[DeploymentEvent]:
        """Return all deployment events."""
        return list(self._events)

    def get_summary(self) -> dict[str, Any]:
        """Get lifecycle management summary."""
        health = self.check_health()
        return {
            "total_nodes": len(self._nodes),
            "active_nodes": len(self.get_active_nodes()),
            "health": {name: h.value for name, h in health.items()},
            "events_count": len(self._events),
        }


# ============================================================================
# Procedure Action Server
# ============================================================================


class ProcedureActionServer:
    """Action server for executing multi-step surgical procedures.

    Simulates ROS 2 action server semantics with goal acceptance,
    feedback publishing, and result delivery. Supports cancellation
    at phase boundaries.

    Args:
        action_name: Name of the action.
        node: The hosting mock node.
    """

    def __init__(self, action_name: str, node: MockNode) -> None:
        self._action_name = action_name
        self._node = node
        self._active_goal: ActionGoal | None = None
        self._goal_history: list[ActionGoal] = []
        self._cancel_requested = False
        self._phase_durations: dict[str, float] = {
            ProcedurePhase.PREPARATION.value: 0.5,
            ProcedurePhase.POSITIONING.value: 1.0,
            ProcedurePhase.APPROACH.value: 0.8,
            ProcedurePhase.EXECUTION.value: 2.0,
            ProcedurePhase.VERIFICATION.value: 0.5,
            ProcedurePhase.RETRACTION.value: 0.7,
            ProcedurePhase.COMPLETION.value: 0.3,
        }
        logger.info("ProcedureActionServer '%s' created on node '%s'", action_name, node.name)

    def accept_goal(self, goal: ActionGoal) -> bool:
        """Accept a new procedure goal.

        Args:
            goal: The goal to accept.

        Returns:
            True if the goal was accepted.
        """
        if self._active_goal is not None and self._active_goal.status == ActionStatus.EXECUTING:
            logger.warning("Cannot accept goal: another goal is executing")
            return False

        goal.status = ActionStatus.ACCEPTED
        self._active_goal = goal
        self._cancel_requested = False
        logger.info("Goal '%s' accepted for action '%s'", goal.goal_id[:8], self._action_name)
        return True

    def execute(self, goal: ActionGoal, rng: np.random.Generator | None = None) -> ActionGoal:
        """Execute a procedure goal through all phases.

        Simulates each phase with synthetic telemetry generation and
        publishes feedback at each phase transition.

        Args:
            goal: The goal to execute.
            rng: Random number generator for simulation.

        Returns:
            The goal with updated status and result.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        goal.status = ActionStatus.EXECUTING
        goal.start_time = time.time()

        procedure_type = goal.parameters.get("procedure_type", "biopsy")
        target_location = goal.parameters.get("target_location", [0.0, 0.0, 0.0])

        logger.info("Executing procedure '%s' at %s", procedure_type, target_location)

        phases = list(ProcedurePhase)
        total_score = 0.0

        for phase in phases:
            if self._cancel_requested:
                goal.status = ActionStatus.CANCELED
                goal.end_time = time.time()
                logger.info("Goal canceled at phase '%s'", phase.value)
                self._goal_history.append(goal)
                return goal

            # Simulate phase execution
            phase_result = self._execute_phase(phase, target_location, rng)
            total_score += phase_result.get("score", 0.0)

            # Publish feedback
            feedback = {
                "phase": phase.value,
                "progress": (phases.index(phase) + 1) / len(phases),
                "phase_result": phase_result,
            }
            goal.feedback.append(feedback)
            self._node.publish(
                f"{self._action_name}/feedback",
                feedback,
                QoSProfile.RELIABLE,
            )

            logger.info(
                "  Phase '%s': score=%.3f, progress=%.0f%%",
                phase.value,
                phase_result.get("score", 0),
                feedback["progress"] * 100,
            )

        # Compute final result
        avg_score = total_score / len(phases)
        success = avg_score > 0.5

        goal.status = ActionStatus.SUCCEEDED if success else ActionStatus.ABORTED
        goal.end_time = time.time()
        goal.result = {
            "success": success,
            "average_score": avg_score,
            "duration_s": goal.end_time - goal.start_time,
            "phases_completed": len(phases),
            "procedure_type": procedure_type,
        }

        self._goal_history.append(goal)
        logger.info(
            "Goal '%s' %s (score=%.3f, duration=%.3fs)",
            goal.goal_id[:8],
            goal.status.value,
            avg_score,
            goal.result["duration_s"],
        )
        return goal

    def _execute_phase(
        self,
        phase: ProcedurePhase,
        target: list[float],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Simulate execution of a single procedure phase.

        Args:
            phase: The phase to execute.
            target: Target location.
            rng: Random number generator.

        Returns:
            Phase execution results.
        """
        # Simulate position accuracy
        position_error = float(rng.exponential(1.0))
        force_applied = float(rng.uniform(0.5, 4.0))
        duration = self._phase_durations.get(phase.value, 1.0) * rng.uniform(0.8, 1.2)

        # Score based on accuracy and force compliance
        accuracy_score = max(0, 1.0 - position_error / 5.0)
        force_score = max(0, 1.0 - max(0, force_applied - 3.0) / 5.0)
        phase_score = 0.6 * accuracy_score + 0.4 * force_score

        return {
            "phase": phase.value,
            "position_error_mm": round(position_error, 3),
            "force_applied_n": round(force_applied, 3),
            "duration_s": round(float(duration), 3),
            "score": round(phase_score, 4),
        }

    def cancel(self) -> bool:
        """Request cancellation of the active goal."""
        if self._active_goal and self._active_goal.status == ActionStatus.EXECUTING:
            self._cancel_requested = True
            logger.info("Cancellation requested for goal '%s'", self._active_goal.goal_id[:8])
            return True
        return False

    def get_goal_history(self) -> list[dict[str, Any]]:
        """Return summary of all executed goals."""
        return [
            {
                "goal_id": g.goal_id,
                "status": g.status.value,
                "phases_completed": len(g.feedback),
                "duration_s": g.end_time - g.start_time if g.end_time else 0,
                "result": g.result,
            }
            for g in self._goal_history
        ]


# ============================================================================
# Service Discovery
# ============================================================================


class ServiceDirectory:
    """Service discovery and registration for the surgical system.

    Provides a centralized registry for services offered by nodes,
    enabling dynamic discovery and load balancing.
    """

    def __init__(self) -> None:
        self._services: dict[str, list[dict[str, Any]]] = {}
        self._lookup_count = 0

    def register(self, service_name: str, node_name: str, service_type: str = "generic") -> None:
        """Register a service endpoint.

        Args:
            service_name: Service name.
            node_name: Name of the providing node.
            service_type: Type classification.
        """
        if service_name not in self._services:
            self._services[service_name] = []
        self._services[service_name].append(
            {
                "node_name": node_name,
                "service_type": service_type,
                "registered_at": time.time(),
            }
        )
        logger.info("Service '%s' registered by node '%s'", service_name, node_name)

    def discover(self, service_name: str) -> list[dict[str, Any]]:
        """Discover providers for a service.

        Args:
            service_name: Service name to look up.

        Returns:
            List of service provider descriptors.
        """
        self._lookup_count += 1
        return self._services.get(service_name, [])

    def discover_by_type(self, service_type: str) -> dict[str, list[dict[str, Any]]]:
        """Discover all services of a given type.

        Args:
            service_type: Type to filter by.

        Returns:
            Dictionary of service_name to provider list.
        """
        self._lookup_count += 1
        result: dict[str, list[dict[str, Any]]] = {}
        for sname, providers in self._services.items():
            matching = [p for p in providers if p["service_type"] == service_type]
            if matching:
                result[sname] = matching
        return result

    def get_all_services(self) -> dict[str, int]:
        """Return all services with provider counts."""
        return {name: len(providers) for name, providers in self._services.items()}

    @property
    def total_lookups(self) -> int:
        """Total number of service lookups performed."""
        return self._lookup_count


# ============================================================================
# Health Monitor
# ============================================================================


class SystemHealthMonitor:
    """Monitors overall health of the surgical robotic system.

    Aggregates node health, communication quality, and system
    resource utilisation into an overall health assessment.

    Args:
        lifecycle_manager: The lifecycle manager to monitor.
        check_interval_s: Health check interval.
    """

    def __init__(self, lifecycle_manager: LifecycleManager, check_interval_s: float = 1.0) -> None:
        self._lifecycle_manager = lifecycle_manager
        self._check_interval = check_interval_s
        self._health_history: list[dict[str, Any]] = []
        self._alerts: list[dict[str, Any]] = []

    def check(self) -> dict[str, Any]:
        """Perform a system health check.

        Returns:
            Health check result.
        """
        node_health = self._lifecycle_manager.check_health()

        healthy_count = sum(1 for h in node_health.values() if h == NodeHealth.HEALTHY)
        total_count = len(node_health)
        system_health_ratio = healthy_count / max(total_count, 1)

        if system_health_ratio >= 0.9:
            overall = "healthy"
        elif system_health_ratio >= 0.5:
            overall = "degraded"
        else:
            overall = "critical"

        result = {
            "timestamp": time.time(),
            "overall_health": overall,
            "healthy_ratio": system_health_ratio,
            "node_health": {name: h.value for name, h in node_health.items()},
            "active_nodes": len(self._lifecycle_manager.get_active_nodes()),
            "total_nodes": total_count,
        }

        self._health_history.append(result)
        if len(self._health_history) > 500:
            self._health_history.pop(0)

        # Generate alerts for unhealthy nodes
        for name, health in node_health.items():
            if health == NodeHealth.UNHEALTHY:
                alert = {
                    "timestamp": time.time(),
                    "node_name": name,
                    "health": health.value,
                    "message": f"Node '{name}' is unhealthy -- heartbeat timeout",
                }
                self._alerts.append(alert)
                logger.warning("HEALTH ALERT: %s", alert["message"])

        return result

    def get_health_trend(self, window: int = 50) -> dict[str, Any]:
        """Compute health trend over recent history.

        Args:
            window: Number of recent checks to analyse.

        Returns:
            Trend statistics.
        """
        recent = self._health_history[-window:]
        if not recent:
            return {"mean_ratio": 0.0, "trend": "unknown", "checks": 0}

        ratios = [r["healthy_ratio"] for r in recent]
        mean_ratio = float(np.mean(ratios))

        if len(ratios) >= 3:
            if ratios[-1] > ratios[0]:
                trend = "improving"
            elif ratios[-1] < ratios[0]:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "mean_ratio": mean_ratio,
            "trend": trend,
            "checks": len(recent),
        }

    def get_alerts(self) -> list[dict[str, Any]]:
        """Return all generated alerts."""
        return list(self._alerts)


# ============================================================================
# Surgical Deployment Orchestrator
# ============================================================================


class SurgicalDeploymentOrchestrator:
    """Top-level orchestrator for ROS 2 surgical system deployment.

    Coordinates node creation, lifecycle management, service discovery,
    action servers, and health monitoring into a unified deployment
    workflow.
    """

    def __init__(self) -> None:
        self._lifecycle_mgr = LifecycleManager(heartbeat_timeout_s=2.0)
        self._service_dir = ServiceDirectory()
        self._health_monitor = SystemHealthMonitor(self._lifecycle_mgr)
        self._action_servers: dict[str, ProcedureActionServer] = {}
        self._nodes: dict[str, MockNode] = {}

    def create_surgical_system(self) -> dict[str, MockNode]:
        """Create the standard surgical system node graph.

        Returns:
            Dictionary of created nodes.
        """
        node_specs = [
            NodeDescriptor(
                node_name="safety_monitor",
                node_type="safety",
                publishers=["/safety/status", "/safety/events"],
                services=["get_safety_status", "emergency_stop"],
            ),
            NodeDescriptor(
                node_name="motion_controller",
                node_type="control",
                publishers=["/motion/state", "/motion/telemetry"],
                subscribers=["/motion/command"],
                services=["set_control_mode"],
            ),
            NodeDescriptor(
                node_name="sensor_fusion",
                node_type="perception",
                publishers=["/fusion/state", "/fusion/anomalies"],
                subscribers=["/sensors/force", "/sensors/camera", "/sensors/encoders"],
            ),
            NodeDescriptor(
                node_name="procedure_executor",
                node_type="planning",
                publishers=["/procedure/status"],
                services=["start_procedure", "abort_procedure"],
            ),
            NodeDescriptor(
                node_name="telemetry_recorder",
                node_type="logging",
                subscribers=["/motion/telemetry", "/fusion/state", "/safety/events"],
                services=["get_recording_status"],
            ),
        ]

        for spec in node_specs:
            node = MockNode(spec)
            self._nodes[spec.node_name] = node
            self._lifecycle_mgr.register_node(node)

            # Register services
            for service_name in spec.services:
                self._service_dir.register(service_name, node.name, spec.node_type)

        # Create action server for procedures
        proc_node = self._nodes["procedure_executor"]
        action_server = ProcedureActionServer("execute_procedure", proc_node)
        self._action_servers["execute_procedure"] = action_server

        logger.info("Surgical system created with %d nodes", len(self._nodes))
        return dict(self._nodes)

    def deploy(self) -> dict[str, Any]:
        """Deploy the surgical system (configure and activate all nodes).

        Returns:
            Deployment result summary.
        """
        logger.info("Deploying surgical system...")
        results = self._lifecycle_mgr.startup_all()

        # Send initial heartbeats
        for node in self._nodes.values():
            node.heartbeat()

        # Initial health check
        health = self._health_monitor.check()

        return {
            "startup_results": {name: r.value for name, r in results.items()},
            "system_health": health,
            "services_registered": self._service_dir.get_all_services(),
        }

    def execute_procedure(
        self,
        procedure_type: str = "biopsy",
        target_location: list[float] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute a surgical procedure through the action server.

        Args:
            procedure_type: Type of procedure.
            target_location: Target location (x, y, z).
            seed: Random seed for simulation.

        Returns:
            Procedure execution result.
        """
        if target_location is None:
            target_location = [50.0, 30.0, 20.0]

        action_server = self._action_servers.get("execute_procedure")
        if action_server is None:
            return {"success": False, "error": "No action server available"}

        goal = ActionGoal(
            action_name="execute_procedure",
            parameters={
                "procedure_type": procedure_type,
                "target_location": target_location,
                "safety_margin_mm": 5.0,
            },
        )

        if not action_server.accept_goal(goal):
            return {"success": False, "error": "Goal rejected"}

        rng = np.random.default_rng(seed)
        result = action_server.execute(goal, rng)

        return {
            "goal_id": result.goal_id,
            "status": result.status.value,
            "result": result.result,
            "feedback_count": len(result.feedback),
        }

    def run_health_check(self) -> dict[str, Any]:
        """Run a system health check.

        Returns:
            Health check result.
        """
        for node in self._nodes.values():
            if node.state == NodeLifecycleState.ACTIVE:
                node.heartbeat()
        return self._health_monitor.check()

    def teardown(self) -> dict[str, Any]:
        """Tear down the surgical system.

        Returns:
            Teardown summary.
        """
        results = self._lifecycle_mgr.shutdown_all()
        return {
            "shutdown_results": {name: r.value for name, r in results.items()},
            "total_events": len(self._lifecycle_mgr.get_events()),
        }

    def get_report(self) -> dict[str, Any]:
        """Generate deployment report."""
        return {
            "lifecycle_summary": self._lifecycle_mgr.get_summary(),
            "services": self._service_dir.get_all_services(),
            "service_lookups": self._service_dir.total_lookups,
            "health_trend": self._health_monitor.get_health_trend(),
            "alerts": self._health_monitor.get_alerts(),
            "action_history": {name: server.get_goal_history() for name, server in self._action_servers.items()},
        }


# ============================================================================
# Demonstration
# ============================================================================


def run_ros2_deployment_demo() -> dict[str, Any]:
    """Run the complete ROS 2 surgical deployment demonstration.

    Returns:
        Final deployment report.
    """
    logger.info("=" * 70)
    logger.info("  ROS 2 Surgical Deployment Pattern Demo")
    logger.info("  Version 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 70)

    orchestrator = SurgicalDeploymentOrchestrator()

    # 1. Create system
    logger.info("\n--- Creating Surgical System ---")
    nodes = orchestrator.create_surgical_system()
    logger.info("Created %d nodes", len(nodes))

    # 2. Deploy (configure + activate)
    logger.info("\n--- Deploying System ---")
    deploy_result = orchestrator.deploy()
    for name, status in deploy_result["startup_results"].items():
        logger.info("  %-40s [%s]", name, status)
    logger.info("  System health: %s", deploy_result["system_health"]["overall_health"])

    # 3. Service discovery
    logger.info("\n--- Service Discovery ---")
    services = deploy_result["services_registered"]
    for sname, count in services.items():
        logger.info("  %-30s %d provider(s)", sname, count)

    # 4. Execute procedures
    logger.info("\n--- Executing Surgical Procedures ---")
    procedures = [
        ("biopsy", [50.0, 30.0, 20.0]),
        ("resection", [45.0, 25.0, 15.0]),
        ("ablation", [60.0, 35.0, 25.0]),
    ]

    for proc_type, target in procedures:
        logger.info("\nProcedure: %s at %s", proc_type, target)
        result = orchestrator.execute_procedure(proc_type, target, seed=hash(proc_type) % 10000)
        logger.info(
            "  Result: %s (score=%.3f, duration=%.3fs)",
            result["status"],
            result["result"].get("average_score", 0),
            result["result"].get("duration_s", 0),
        )

    # 5. Health monitoring
    logger.info("\n--- System Health Check ---")
    health = orchestrator.run_health_check()
    logger.info("  Overall: %s (%.0f%% healthy)", health["overall_health"], health["healthy_ratio"] * 100)

    # 6. Generate report
    logger.info("\n--- Deployment Report ---")
    report = orchestrator.get_report()
    lifecycle = report["lifecycle_summary"]
    logger.info("  Total nodes:     %d", lifecycle["total_nodes"])
    logger.info("  Active nodes:    %d", lifecycle["active_nodes"])
    logger.info("  Events logged:   %d", lifecycle["events_count"])
    logger.info("  Service lookups: %d", report["service_lookups"])

    health_trend = report["health_trend"]
    logger.info("  Health trend:    %s (mean=%.2f)", health_trend["trend"], health_trend["mean_ratio"])

    # 7. Teardown
    logger.info("\n--- Teardown ---")
    teardown = orchestrator.teardown()
    logger.info("  Nodes shut down: %d", len(teardown["shutdown_results"]))
    logger.info("  Total events:    %d", teardown["total_events"])

    logger.info("\n" + "=" * 70)
    logger.info("  RESEARCH USE ONLY -- NOT FOR CLINICAL DEPLOYMENT")
    logger.info("=" * 70)

    return report


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    report = run_ros2_deployment_demo()
    sys.exit(0)
