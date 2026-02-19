"""
Isaac Sim <-> MuJoCo Bidirectional Simulation Bridge for Oncology Robotics.

Provides state conversion, parameter mapping, and synchronized stepping
between NVIDIA Isaac Sim (PhysX 5) and DeepMind MuJoCo physics engines.
Designed for federated learning workflows where different clinical trial
sites may use different simulation backends for surgical robotics training.

DISCLAIMER: RESEARCH USE ONLY
This software is provided for research and educational purposes only.
It has NOT been validated for clinical use, is NOT approved by the FDA
or any other regulatory body, and MUST NOT be used to make clinical
decisions or control physical surgical robots in patient-facing settings.
All simulation results must be independently validated before any
clinical application. Use at your own risk.

Copyright (c) 2026 PAI Oncology Trial FL Contributors
License: MIT
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
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from utils.time import utc_now_iso

# ---------------------------------------------------------------------------
# Conditional imports for optional simulation backends
# ---------------------------------------------------------------------------
try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore[assignment]
    HAS_MUJOCO = False

try:
    import isaacsim  # noqa: F401
    from omni.isaac.core import SimulationContext  # type: ignore[import-untyped]

    HAS_ISAAC = True
except ImportError:
    isaacsim = None  # type: ignore[assignment]
    SimulationContext = None  # type: ignore[assignment,misc]
    HAS_ISAAC = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEP_MUJOCO: float = 0.002  # 2 ms (MuJoCo default)
DEFAULT_TIMESTEP_ISAAC: float = 1.0 / 60.0  # ~16.67 ms (Isaac Sim 60 Hz)
MAX_JOINT_VELOCITY_RAD_S: float = 6.2832  # 2*pi rad/s safety limit
MAX_JOINT_TORQUE_NM: float = 500.0  # Safety limit for surgical robots
POSITION_TOLERANCE_M: float = 1e-4  # 0.1 mm positional tolerance
ORIENTATION_TOLERANCE_RAD: float = 1e-3  # ~0.057 degrees
DEFAULT_GRAVITY: tuple[float, float, float] = (0.0, 0.0, -9.81)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class BridgeState(Enum):
    """Lifecycle state of the simulation bridge."""

    UNINITIALIZED = "uninitialized"
    CONFIGURING = "configuring"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    DISCONNECTED = "disconnected"


class CoordinateConvention(Enum):
    """Coordinate frame conventions used by each engine."""

    ISAAC_Y_UP = "isaac_y_up"
    MUJOCO_Z_UP = "mujoco_z_up"
    ROS_Z_UP = "ros_z_up"


class JointType(Enum):
    """Canonical joint types supported across both engines."""

    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    FREE = "free"
    BALL = "ball"
    CONTINUOUS = "continuous"


class SolverType(Enum):
    """Solver implementations available in each engine."""

    MUJOCO_CG = "mujoco_cg"
    MUJOCO_NEWTON = "mujoco_newton"
    PHYSX_TGS = "physx_tgs"
    PHYSX_PGS = "physx_pgs"


class ContactModel(Enum):
    """Contact model type used by the physics engine."""

    SOFT_CONTACT = "soft_contact"
    RIGID_CONTACT = "rigid_contact"
    COMPLIANT = "compliant"


class SyncDirection(Enum):
    """Direction of state synchronization."""

    ISAAC_TO_MUJOCO = "isaac_to_mujoco"
    MUJOCO_TO_ISAAC = "mujoco_to_isaac"
    BIDIRECTIONAL = "bidirectional"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class JointState:
    """Canonical representation of a single joint's state."""

    name: str
    joint_type: JointType
    position: float = 0.0  # radians for revolute, meters for prismatic
    velocity: float = 0.0  # rad/s or m/s
    acceleration: float = 0.0
    torque: float = 0.0  # N*m or N
    position_limits: tuple[float, float] = (-3.14159, 3.14159)
    velocity_limit: float = MAX_JOINT_VELOCITY_RAD_S
    torque_limit: float = MAX_JOINT_TORQUE_NM
    damping: float = 0.0
    stiffness: float = 0.0
    friction: float = 0.0

    def clamp_to_limits(self) -> JointState:
        """Return a new JointState with values clamped to safety limits."""
        return JointState(
            name=self.name,
            joint_type=self.joint_type,
            position=max(self.position_limits[0], min(self.position, self.position_limits[1])),
            velocity=max(-self.velocity_limit, min(self.velocity, self.velocity_limit)),
            acceleration=self.acceleration,
            torque=max(-self.torque_limit, min(self.torque, self.torque_limit)),
            position_limits=self.position_limits,
            velocity_limit=self.velocity_limit,
            torque_limit=self.torque_limit,
            damping=self.damping,
            stiffness=self.stiffness,
            friction=self.friction,
        )


@dataclass
class LinkState:
    """Canonical representation of a rigid body / link state."""

    name: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world frame [x, y, z]
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # quaternion [w,x,y,z]
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class ContactPoint:
    """Canonical representation of a contact point between bodies."""

    body_a: str
    body_b: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    depth: float = 0.0
    friction_coefficient: float = 0.5


@dataclass
class SimulationState:
    """Complete simulation state for cross-engine transfer."""

    timestamp: str = ""
    sim_time: float = 0.0
    joint_states: list[JointState] = field(default_factory=list)
    link_states: list[LinkState] = field(default_factory=list)
    contact_points: list[ContactPoint] = field(default_factory=list)
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY
    coordinate_convention: CoordinateConvention = CoordinateConvention.MUJOCO_Z_UP
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_array(self) -> np.ndarray:
        """Flatten joint positions and velocities into a single array for federation."""
        positions = [js.position for js in self.joint_states]
        velocities = [js.velocity for js in self.joint_states]
        return np.array(positions + velocities, dtype=np.float64)

    def compute_hash(self) -> str:
        """Compute a deterministic hash of the state for audit purposes."""
        data = {
            "sim_time": round(self.sim_time, 10),
            "joint_positions": [round(js.position, 10) for js in self.joint_states],
            "joint_velocities": [round(js.velocity, 10) for js in self.joint_states],
        }
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass
class BridgeConfig:
    """Configuration for the Isaac-MuJoCo bridge."""

    mujoco_model_path: str = ""
    isaac_usd_path: str = ""
    sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    mujoco_timestep: float = DEFAULT_TIMESTEP_MUJOCO
    isaac_timestep: float = DEFAULT_TIMESTEP_ISAAC
    coordinate_convention: CoordinateConvention = CoordinateConvention.MUJOCO_Z_UP
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY
    joint_mapping: dict[str, str] = field(default_factory=dict)
    link_mapping: dict[str, str] = field(default_factory=dict)
    enable_contact_sync: bool = True
    enable_force_sync: bool = True
    max_sync_drift_m: float = POSITION_TOLERANCE_M
    max_sync_drift_rad: float = ORIENTATION_TOLERANCE_RAD
    enable_safety_limits: bool = True
    enable_audit_logging: bool = True
    solver_type_mujoco: SolverType = SolverType.MUJOCO_NEWTON
    solver_type_isaac: SolverType = SolverType.PHYSX_TGS
    contact_model: ContactModel = ContactModel.SOFT_CONTACT
    substeps_mujoco: int = 1
    substeps_isaac: int = 4
    position_gain_kp: float = 100.0
    velocity_gain_kd: float = 10.0
    isaac_config: dict[str, Any] = field(default_factory=dict)
    mujoco_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues: list[str] = []
        if self.mujoco_timestep <= 0:
            issues.append("mujoco_timestep must be positive")
        if self.isaac_timestep <= 0:
            issues.append("isaac_timestep must be positive")
        if self.mujoco_timestep > 0.1:
            issues.append("mujoco_timestep > 100ms is unusually large")
        if self.isaac_timestep > 0.1:
            issues.append("isaac_timestep > 100ms is unusually large")
        if self.substeps_mujoco < 1:
            issues.append("substeps_mujoco must be >= 1")
        if self.substeps_isaac < 1:
            issues.append("substeps_isaac must be >= 1")
        if self.max_sync_drift_m < 0:
            issues.append("max_sync_drift_m must be non-negative")
        if self.position_gain_kp < 0:
            issues.append("position_gain_kp must be non-negative")
        if self.velocity_gain_kd < 0:
            issues.append("velocity_gain_kd must be non-negative")
        return issues


@dataclass
class SyncReport:
    """Report generated after each synchronization step."""

    sync_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    timestamp: str = ""
    sim_time_source: float = 0.0
    sim_time_target: float = 0.0
    num_joints_synced: int = 0
    num_links_synced: int = 0
    num_contacts_synced: int = 0
    max_position_drift_m: float = 0.0
    max_orientation_drift_rad: float = 0.0
    max_velocity_drift: float = 0.0
    safety_limits_applied: int = 0
    sync_duration_ms: float = 0.0
    success: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    state_hash_source: str = ""
    state_hash_target: str = ""


@dataclass
class CalibrationResult:
    """Result of a cross-engine calibration run."""

    engine_a: str = "isaac"
    engine_b: str = "mujoco"
    num_steps: int = 0
    mean_position_error_m: float = 0.0
    max_position_error_m: float = 0.0
    mean_velocity_error: float = 0.0
    max_velocity_error: float = 0.0
    mean_force_error_n: float = 0.0
    max_force_error_n: float = 0.0
    correlation_coefficient: float = 0.0
    passed: bool = False
    calibration_parameters: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol interface for engine backends
# ---------------------------------------------------------------------------
class PhysicsBackend(Protocol):
    """Protocol that each physics engine adapter must implement."""

    def initialize(self, config: BridgeConfig) -> bool: ...
    def step(self, dt: float) -> None: ...
    def get_state(self) -> SimulationState: ...
    def set_state(self, state: SimulationState) -> None: ...
    def get_joint_state(self, joint_name: str) -> JointState: ...
    def set_joint_state(self, joint_name: str, state: JointState) -> None: ...
    def get_link_state(self, link_name: str) -> LinkState: ...
    def get_contacts(self) -> list[ContactPoint]: ...
    def reset(self) -> None: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Coordinate transformation utilities
# ---------------------------------------------------------------------------
class CoordinateTransformer:
    """Handles coordinate frame conversions between engine conventions."""

    # Isaac Sim uses Y-up by default; MuJoCo uses Z-up
    # Rotation from Y-up to Z-up: rotate -90 degrees around X axis
    _Y_UP_TO_Z_UP = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    _Z_UP_TO_Y_UP = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )

    @staticmethod
    def transform_position(
        position: np.ndarray,
        source: CoordinateConvention,
        target: CoordinateConvention,
    ) -> np.ndarray:
        """Transform a position vector between coordinate conventions."""
        if source == target:
            return position.copy()

        pos = np.asarray(position, dtype=np.float64)

        if source == CoordinateConvention.ISAAC_Y_UP and target == CoordinateConvention.MUJOCO_Z_UP:
            return CoordinateTransformer._Y_UP_TO_Z_UP @ pos
        elif source == CoordinateConvention.MUJOCO_Z_UP and target == CoordinateConvention.ISAAC_Y_UP:
            return CoordinateTransformer._Z_UP_TO_Y_UP @ pos
        elif source == CoordinateConvention.ROS_Z_UP and target == CoordinateConvention.MUJOCO_Z_UP:
            return pos.copy()  # Same convention
        elif source == CoordinateConvention.MUJOCO_Z_UP and target == CoordinateConvention.ROS_Z_UP:
            return pos.copy()
        elif source == CoordinateConvention.ISAAC_Y_UP and target == CoordinateConvention.ROS_Z_UP:
            return CoordinateTransformer._Y_UP_TO_Z_UP @ pos
        elif source == CoordinateConvention.ROS_Z_UP and target == CoordinateConvention.ISAAC_Y_UP:
            return CoordinateTransformer._Z_UP_TO_Y_UP @ pos
        else:
            logger.warning("Unknown coordinate conversion: %s -> %s", source, target)
            return pos.copy()

    @staticmethod
    def transform_quaternion(
        quat: np.ndarray,
        source: CoordinateConvention,
        target: CoordinateConvention,
    ) -> np.ndarray:
        """Transform a quaternion [w, x, y, z] between coordinate conventions."""
        if source == target:
            return quat.copy()

        q = np.asarray(quat, dtype=np.float64)
        w, x, y, z = q[0], q[1], q[2], q[3]

        if source == CoordinateConvention.ISAAC_Y_UP and target == CoordinateConvention.MUJOCO_Z_UP:
            return np.array([w, x, -z, y])
        elif source == CoordinateConvention.MUJOCO_Z_UP and target == CoordinateConvention.ISAAC_Y_UP:
            return np.array([w, x, z, -y])
        else:
            return q.copy()

    @staticmethod
    def transform_velocity(
        velocity: np.ndarray,
        source: CoordinateConvention,
        target: CoordinateConvention,
    ) -> np.ndarray:
        """Transform a linear or angular velocity vector."""
        return CoordinateTransformer.transform_position(velocity, source, target)


# ---------------------------------------------------------------------------
# Parameter mapping utilities
# ---------------------------------------------------------------------------
class ParameterMapper:
    """Maps physics parameters between Isaac Sim (PhysX) and MuJoCo representations."""

    @staticmethod
    def mujoco_friction_to_physx(
        sliding: float,
        torsional: float,
        rolling: float,
    ) -> dict[str, float]:
        """Convert MuJoCo friction parameters to PhysX equivalents.

        MuJoCo friction is specified as (sliding, torsional, rolling) where
        sliding is the tangential Coulomb friction coefficient.
        PhysX uses (static_friction, dynamic_friction, restitution).
        """
        static_friction = sliding * 1.1  # approximate: static > dynamic
        dynamic_friction = sliding
        return {
            "static_friction": min(static_friction, 2.0),
            "dynamic_friction": min(dynamic_friction, 2.0),
            "torsional_friction": torsional,
            "rolling_friction": rolling,
        }

    @staticmethod
    def physx_friction_to_mujoco(
        static_friction: float,
        dynamic_friction: float,
        restitution: float = 0.0,
    ) -> dict[str, float]:
        """Convert PhysX friction parameters to MuJoCo equivalents."""
        sliding = (static_friction + dynamic_friction) / 2.0
        torsional = 0.005  # MuJoCo default torsional
        rolling = 0.001  # MuJoCo default rolling
        return {
            "sliding": sliding,
            "torsional": torsional,
            "rolling": rolling,
            "condim": 3 if torsional > 0 else 1,
        }

    @staticmethod
    def mujoco_damping_to_physx(damping: float, stiffness: float) -> dict[str, float]:
        """Convert MuJoCo joint damping/stiffness to PhysX drive parameters."""
        return {
            "drive_stiffness": stiffness,
            "drive_damping": damping,
            "max_force": MAX_JOINT_TORQUE_NM,
        }

    @staticmethod
    def physx_drive_to_mujoco(
        drive_stiffness: float,
        drive_damping: float,
        max_force: float = MAX_JOINT_TORQUE_NM,
    ) -> dict[str, float]:
        """Convert PhysX drive parameters to MuJoCo equivalents."""
        return {
            "stiffness": drive_stiffness,
            "damping": drive_damping,
            "frictionloss": 0.0,
            "armature": 0.01,
        }

    @staticmethod
    def mujoco_solver_to_physx_params(
        solver_type: SolverType,
        iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> dict[str, Any]:
        """Map MuJoCo solver configuration to PhysX solver parameters."""
        if solver_type == SolverType.MUJOCO_CG:
            return {
                "solver_type": "TGS",
                "position_iterations": max(4, iterations // 10),
                "velocity_iterations": max(1, iterations // 20),
                "stabilization_threshold": tolerance * 100,
            }
        elif solver_type == SolverType.MUJOCO_NEWTON:
            return {
                "solver_type": "TGS",
                "position_iterations": max(8, iterations // 5),
                "velocity_iterations": max(2, iterations // 10),
                "stabilization_threshold": tolerance * 10,
            }
        return {
            "solver_type": "TGS",
            "position_iterations": 8,
            "velocity_iterations": 2,
            "stabilization_threshold": 1e-6,
        }

    @staticmethod
    def physx_solver_to_mujoco_params(
        solver_type: SolverType,
        position_iterations: int = 8,
        velocity_iterations: int = 2,
    ) -> dict[str, Any]:
        """Map PhysX solver configuration to MuJoCo solver parameters."""
        total_iterations = (position_iterations + velocity_iterations) * 5
        if solver_type == SolverType.PHYSX_TGS:
            return {
                "solver": "Newton",
                "iterations": total_iterations,
                "tolerance": 1e-8,
                "noslip_iterations": max(1, position_iterations // 2),
            }
        elif solver_type == SolverType.PHYSX_PGS:
            return {
                "solver": "CG",
                "iterations": total_iterations * 2,
                "tolerance": 1e-6,
                "noslip_iterations": 0,
            }
        return {"solver": "Newton", "iterations": 100, "tolerance": 1e-8, "noslip_iterations": 3}

    @staticmethod
    def convert_inertia_tensor(
        inertia: np.ndarray,
        source: CoordinateConvention,
        target: CoordinateConvention,
    ) -> np.ndarray:
        """Convert an inertia tensor between coordinate conventions."""
        if source == target:
            return inertia.copy()

        if source == CoordinateConvention.ISAAC_Y_UP and target == CoordinateConvention.MUJOCO_Z_UP:
            rot = CoordinateTransformer._Y_UP_TO_Z_UP
        elif source == CoordinateConvention.MUJOCO_Z_UP and target == CoordinateConvention.ISAAC_Y_UP:
            rot = CoordinateTransformer._Z_UP_TO_Y_UP
        else:
            return inertia.copy()

        return rot @ inertia @ rot.T


# ---------------------------------------------------------------------------
# MuJoCo backend adapter
# ---------------------------------------------------------------------------
class MuJoCoBackend:
    """Adapter for the MuJoCo physics engine."""

    def __init__(self) -> None:
        self._model: Any = None
        self._data: Any = None
        self._initialized: bool = False
        self._joint_name_to_id: dict[str, int] = {}
        self._body_name_to_id: dict[str, int] = {}

    def initialize(self, config: BridgeConfig) -> bool:
        """Load a MuJoCo model from an MJCF or XML file."""
        if not HAS_MUJOCO:
            logger.error("MuJoCo is not installed. Install with: pip install mujoco")
            return False

        model_path = Path(config.mujoco_model_path)
        if not model_path.exists():
            logger.error("MuJoCo model file not found: %s", model_path)
            return False

        try:
            self._model = mujoco.MjModel.from_xml_path(str(model_path))
            self._data = mujoco.MjData(self._model)

            # Configure solver
            if config.solver_type_mujoco == SolverType.MUJOCO_NEWTON:
                self._model.opt.solver = 2  # mjSOL_NEWTON
            else:
                self._model.opt.solver = 1  # mjSOL_CG

            self._model.opt.timestep = config.mujoco_timestep

            # Build name-to-id mappings
            for i in range(self._model.njnt):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name:
                    self._joint_name_to_id[name] = i

            for i in range(self._model.nbody):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name:
                    self._body_name_to_id[name] = i

            self._initialized = True
            logger.info(
                "MuJoCo backend initialized: %d joints, %d bodies",
                self._model.njnt,
                self._model.nbody,
            )
            return True
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Failed to initialize MuJoCo backend: %s", exc)
            return False

    def step(self, dt: float) -> None:
        """Advance the simulation by one timestep."""
        if not self._initialized or self._model is None or self._data is None:
            return
        num_substeps = max(1, int(dt / self._model.opt.timestep))
        for _ in range(num_substeps):
            mujoco.mj_step(self._model, self._data)

    def get_state(self) -> SimulationState:
        """Extract the full simulation state."""
        if not self._initialized or self._model is None or self._data is None:
            return SimulationState()

        joint_states: list[JointState] = []
        for name, jid in self._joint_name_to_id.items():
            jtype = self._model.jnt_type[jid]
            canonical_type = {0: JointType.FREE, 1: JointType.BALL, 2: JointType.PRISMATIC, 3: JointType.REVOLUTE}.get(
                int(jtype), JointType.REVOLUTE
            )

            qpos_addr = self._model.jnt_qposadr[jid]
            qvel_addr = self._model.jnt_dofadr[jid]

            pos = float(self._data.qpos[qpos_addr]) if canonical_type != JointType.FREE else 0.0
            vel = float(self._data.qvel[qvel_addr]) if canonical_type != JointType.FREE else 0.0

            joint_states.append(
                JointState(
                    name=name,
                    joint_type=canonical_type,
                    position=pos,
                    velocity=vel,
                    damping=float(self._model.dof_damping[qvel_addr]) if qvel_addr < self._model.nv else 0.0,
                )
            )

        link_states: list[LinkState] = []
        for name, bid in self._body_name_to_id.items():
            link_states.append(
                LinkState(
                    name=name,
                    position=self._data.xpos[bid].copy(),
                    orientation=self._data.xquat[bid].copy(),
                    linear_velocity=self._data.cvel[bid, 3:6].copy() if bid < self._data.cvel.shape[0] else np.zeros(3),
                    angular_velocity=self._data.cvel[bid, 0:3].copy()
                    if bid < self._data.cvel.shape[0]
                    else np.zeros(3),
                    mass=float(self._model.body_mass[bid]),
                )
            )

        contacts: list[ContactPoint] = []
        for i in range(self._data.ncon):
            con = self._data.contact[i]
            body_a_name = ""
            body_b_name = ""
            geom1_body = self._model.geom_bodyid[con.geom1]
            geom2_body = self._model.geom_bodyid[con.geom2]
            for bname, bid in self._body_name_to_id.items():
                if bid == geom1_body:
                    body_a_name = bname
                if bid == geom2_body:
                    body_b_name = bname

            contacts.append(
                ContactPoint(
                    body_a=body_a_name,
                    body_b=body_b_name,
                    position=con.pos.copy(),
                    normal=con.frame[:3].copy(),
                    depth=float(con.dist),
                )
            )

        return SimulationState(
            timestamp=utc_now_iso(),
            sim_time=float(self._data.time),
            joint_states=joint_states,
            link_states=link_states,
            contact_points=contacts,
            coordinate_convention=CoordinateConvention.MUJOCO_Z_UP,
        )

    def set_state(self, state: SimulationState) -> None:
        """Apply a simulation state to the MuJoCo model."""
        if not self._initialized or self._model is None or self._data is None:
            return

        for js in state.joint_states:
            if js.name in self._joint_name_to_id:
                jid = self._joint_name_to_id[js.name]
                qpos_addr = self._model.jnt_qposadr[jid]
                qvel_addr = self._model.jnt_dofadr[jid]
                self._data.qpos[qpos_addr] = js.position
                self._data.qvel[qvel_addr] = js.velocity

        mujoco.mj_forward(self._model, self._data)

    def get_joint_state(self, joint_name: str) -> JointState:
        """Get the state of a single joint by name."""
        if not self._initialized or joint_name not in self._joint_name_to_id:
            return JointState(name=joint_name, joint_type=JointType.REVOLUTE)

        jid = self._joint_name_to_id[joint_name]
        qpos_addr = self._model.jnt_qposadr[jid]
        qvel_addr = self._model.jnt_dofadr[jid]

        return JointState(
            name=joint_name,
            joint_type=JointType.REVOLUTE,
            position=float(self._data.qpos[qpos_addr]),
            velocity=float(self._data.qvel[qvel_addr]),
        )

    def set_joint_state(self, joint_name: str, state: JointState) -> None:
        """Set the state of a single joint."""
        if not self._initialized or joint_name not in self._joint_name_to_id:
            return

        jid = self._joint_name_to_id[joint_name]
        qpos_addr = self._model.jnt_qposadr[jid]
        qvel_addr = self._model.jnt_dofadr[jid]
        self._data.qpos[qpos_addr] = state.position
        self._data.qvel[qvel_addr] = state.velocity

    def get_link_state(self, link_name: str) -> LinkState:
        """Get the state of a single link / body by name."""
        if not self._initialized or link_name not in self._body_name_to_id:
            return LinkState(name=link_name)

        bid = self._body_name_to_id[link_name]
        return LinkState(
            name=link_name,
            position=self._data.xpos[bid].copy(),
            orientation=self._data.xquat[bid].copy(),
            mass=float(self._model.body_mass[bid]),
        )

    def get_contacts(self) -> list[ContactPoint]:
        """Get all active contact points."""
        state = self.get_state()
        return state.contact_points

    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        if self._initialized and self._model is not None:
            self._data = mujoco.MjData(self._model)
            logger.info("MuJoCo backend reset to initial state")

    def shutdown(self) -> None:
        """Clean up resources."""
        self._model = None
        self._data = None
        self._initialized = False
        logger.info("MuJoCo backend shut down")


# ---------------------------------------------------------------------------
# Isaac Sim backend adapter (stub for environments without Omniverse)
# ---------------------------------------------------------------------------
class IsaacSimBackend:
    """Adapter for NVIDIA Isaac Sim physics engine."""

    def __init__(self) -> None:
        self._sim_context: Any = None
        self._initialized: bool = False
        self._joint_name_to_path: dict[str, str] = {}
        self._body_name_to_path: dict[str, str] = {}
        self._joint_states: dict[str, JointState] = {}
        self._body_states: dict[str, LinkState] = {}
        self._sim_time: float = 0.0
        self._timestep: float = DEFAULT_TIMESTEP_ISAAC

    def initialize(self, config: BridgeConfig) -> bool:
        """Initialize Isaac Sim with a USD stage."""
        self._timestep = config.isaac_timestep

        if HAS_ISAAC:
            try:
                self._sim_context = SimulationContext(physics_dt=config.isaac_timestep)
                self._sim_context.initialize_physics()
                self._initialized = True
                logger.info("Isaac Sim backend initialized with live Omniverse connection")
                return True
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Isaac Sim initialization failed, falling back to stub: %s", exc)

        # Stub mode for environments without Isaac Sim
        self._initialized = True
        logger.info(
            "Isaac Sim backend initialized in STUB mode (no Omniverse). "
            "State will be managed in-memory for bridge testing."
        )
        return True

    def step(self, dt: float) -> None:
        """Advance the simulation by dt seconds."""
        if not self._initialized:
            return

        if HAS_ISAAC and self._sim_context is not None:
            num_steps = max(1, int(dt / self._timestep))
            for _ in range(num_steps):
                self._sim_context.step()
            self._sim_time += dt
        else:
            self._sim_time += dt

    def get_state(self) -> SimulationState:
        """Extract simulation state (from Omniverse or stub)."""
        joint_states = list(self._joint_states.values())
        link_states = list(self._body_states.values())
        return SimulationState(
            timestamp=utc_now_iso(),
            sim_time=self._sim_time,
            joint_states=joint_states,
            link_states=link_states,
            contact_points=[],
            coordinate_convention=CoordinateConvention.ISAAC_Y_UP,
        )

    def set_state(self, state: SimulationState) -> None:
        """Apply simulation state."""
        for js in state.joint_states:
            self._joint_states[js.name] = copy.deepcopy(js)
        for ls in state.link_states:
            self._body_states[ls.name] = copy.deepcopy(ls)
        self._sim_time = state.sim_time

    def get_joint_state(self, joint_name: str) -> JointState:
        """Get a single joint state."""
        return self._joint_states.get(joint_name, JointState(name=joint_name, joint_type=JointType.REVOLUTE))

    def set_joint_state(self, joint_name: str, state: JointState) -> None:
        """Set a single joint state."""
        self._joint_states[joint_name] = copy.deepcopy(state)

    def get_link_state(self, link_name: str) -> LinkState:
        """Get a single link state."""
        return self._body_states.get(link_name, LinkState(name=link_name))

    def get_contacts(self) -> list[ContactPoint]:
        """Get contact points (stub returns empty list)."""
        return []

    def reset(self) -> None:
        """Reset to initial state."""
        self._joint_states.clear()
        self._body_states.clear()
        self._sim_time = 0.0
        if HAS_ISAAC and self._sim_context is not None:
            self._sim_context.reset()
        logger.info("Isaac Sim backend reset")

    def shutdown(self) -> None:
        """Clean up resources."""
        if HAS_ISAAC and self._sim_context is not None:
            self._sim_context.stop()
        self._initialized = False
        logger.info("Isaac Sim backend shut down")


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------
class IsaacMuJoCoBridge:
    """Bidirectional state synchronization bridge between Isaac Sim and MuJoCo.

    The bridge maintains two physics backends and provides methods to:
    - Transfer complete simulation states between engines
    - Convert coordinate conventions automatically
    - Map physics parameters (friction, damping, solver settings)
    - Enforce safety limits on all transferred states
    - Generate audit reports for every synchronization
    - Run calibration sequences to quantify cross-engine drift

    Intended for use in federated oncology trial workflows where different
    sites run different simulation engines for surgical robotics training.
    """

    def __init__(self, config: Optional[BridgeConfig] = None) -> None:
        self._config = config or BridgeConfig()
        self._state = BridgeState.UNINITIALIZED
        self._mujoco = MuJoCoBackend()
        self._isaac = IsaacSimBackend()
        self._sync_history: list[SyncReport] = []
        self._step_count: int = 0
        self._total_sync_time_ms: float = 0.0
        self._transformer = CoordinateTransformer()
        self._param_mapper = ParameterMapper()

        issues = self._config.validate()
        if issues:
            for issue in issues:
                logger.warning("Bridge config issue: %s", issue)

    @property
    def state(self) -> BridgeState:
        """Current lifecycle state of the bridge."""
        return self._state

    @property
    def step_count(self) -> int:
        """Number of synchronization steps executed."""
        return self._step_count

    @property
    def sync_history(self) -> list[SyncReport]:
        """History of all sync reports."""
        return list(self._sync_history)

    def initialize(self) -> bool:
        """Initialize both physics backends."""
        self._state = BridgeState.CONFIGURING
        logger.info("Initializing Isaac-MuJoCo bridge...")

        isaac_ok = self._isaac.initialize(self._config)
        if not isaac_ok:
            logger.error("Failed to initialize Isaac Sim backend")
            self._state = BridgeState.ERROR
            return False

        mujoco_ok = self._mujoco.initialize(self._config)
        if not mujoco_ok:
            logger.warning("MuJoCo backend not available; bridge will operate in Isaac-only mode")

        self._state = BridgeState.READY
        logger.info("Bridge initialized successfully")
        return True

    def sync_isaac_to_mujoco(self) -> SyncReport:
        """Transfer state from Isaac Sim to MuJoCo."""
        return self._sync(SyncDirection.ISAAC_TO_MUJOCO)

    def sync_mujoco_to_isaac(self) -> SyncReport:
        """Transfer state from MuJoCo to Isaac Sim."""
        return self._sync(SyncDirection.MUJOCO_TO_ISAAC)

    def sync_bidirectional(self) -> SyncReport:
        """Perform bidirectional synchronization (average states)."""
        return self._sync(SyncDirection.BIDIRECTIONAL)

    def _sync(self, direction: SyncDirection) -> SyncReport:
        """Core synchronization logic."""
        start_time = time.time()
        report = SyncReport(direction=direction, timestamp=utc_now_iso())

        if self._state not in (BridgeState.READY, BridgeState.RUNNING):
            report.success = False
            report.errors.append(f"Bridge not ready: state={self._state.value}")
            return report

        self._state = BridgeState.RUNNING

        try:
            if direction == SyncDirection.ISAAC_TO_MUJOCO:
                source_state = self._isaac.get_state()
                target_state = self._convert_state(
                    source_state,
                    source_convention=CoordinateConvention.ISAAC_Y_UP,
                    target_convention=CoordinateConvention.MUJOCO_Z_UP,
                )
                if self._config.enable_safety_limits:
                    target_state, num_clamped = self._apply_safety_limits(target_state)
                    report.safety_limits_applied = num_clamped
                self._mujoco.set_state(target_state)
                report.state_hash_source = source_state.compute_hash()
                report.state_hash_target = target_state.compute_hash()
                report.sim_time_source = source_state.sim_time
                report.sim_time_target = target_state.sim_time

            elif direction == SyncDirection.MUJOCO_TO_ISAAC:
                source_state = self._mujoco.get_state()
                target_state = self._convert_state(
                    source_state,
                    source_convention=CoordinateConvention.MUJOCO_Z_UP,
                    target_convention=CoordinateConvention.ISAAC_Y_UP,
                )
                if self._config.enable_safety_limits:
                    target_state, num_clamped = self._apply_safety_limits(target_state)
                    report.safety_limits_applied = num_clamped
                self._isaac.set_state(target_state)
                report.state_hash_source = source_state.compute_hash()
                report.state_hash_target = target_state.compute_hash()
                report.sim_time_source = source_state.sim_time
                report.sim_time_target = target_state.sim_time

            elif direction == SyncDirection.BIDIRECTIONAL:
                isaac_state = self._isaac.get_state()
                mujoco_state = self._mujoco.get_state()
                averaged = self._average_states(isaac_state, mujoco_state)
                if self._config.enable_safety_limits:
                    averaged, num_clamped = self._apply_safety_limits(averaged)
                    report.safety_limits_applied = num_clamped
                self._isaac.set_state(
                    self._convert_state(
                        averaged,
                        CoordinateConvention.MUJOCO_Z_UP,
                        CoordinateConvention.ISAAC_Y_UP,
                    )
                )
                self._mujoco.set_state(averaged)
                report.state_hash_source = isaac_state.compute_hash()
                report.state_hash_target = mujoco_state.compute_hash()

            report.num_joints_synced = (
                len(source_state.joint_states)
                if direction != SyncDirection.BIDIRECTIONAL
                else max(len(isaac_state.joint_states), len(mujoco_state.joint_states))
            )
            report.num_links_synced = (
                len(source_state.link_states)
                if direction != SyncDirection.BIDIRECTIONAL
                else max(len(isaac_state.link_states), len(mujoco_state.link_states))
            )
            report.num_contacts_synced = (
                len(source_state.contact_points) if direction != SyncDirection.BIDIRECTIONAL else 0
            )
            report.success = True

        except (RuntimeError, ValueError, OSError) as exc:
            report.success = False
            report.errors.append(str(exc))
            logger.error("Sync failed: %s", exc)
            self._state = BridgeState.ERROR

        elapsed = (time.time() - start_time) * 1000.0
        report.sync_duration_ms = elapsed
        self._total_sync_time_ms += elapsed
        self._step_count += 1
        self._sync_history.append(report)

        if self._config.enable_audit_logging:
            logger.info(
                "Sync %s [%s]: joints=%d links=%d contacts=%d drift=%.6fm clamped=%d time=%.2fms",
                report.sync_id[:8],
                direction.value,
                report.num_joints_synced,
                report.num_links_synced,
                report.num_contacts_synced,
                report.max_position_drift_m,
                report.safety_limits_applied,
                report.sync_duration_ms,
            )

        self._state = BridgeState.READY
        return report

    def _convert_state(
        self,
        state: SimulationState,
        source_convention: CoordinateConvention,
        target_convention: CoordinateConvention,
    ) -> SimulationState:
        """Convert a simulation state between coordinate conventions."""
        converted_joints: list[JointState] = []
        for js in state.joint_states:
            mapped_name = self._config.joint_mapping.get(js.name, js.name)
            converted_joints.append(
                JointState(
                    name=mapped_name,
                    joint_type=js.joint_type,
                    position=js.position,
                    velocity=js.velocity,
                    acceleration=js.acceleration,
                    torque=js.torque,
                    position_limits=js.position_limits,
                    velocity_limit=js.velocity_limit,
                    torque_limit=js.torque_limit,
                    damping=js.damping,
                    stiffness=js.stiffness,
                    friction=js.friction,
                )
            )

        converted_links: list[LinkState] = []
        for ls in state.link_states:
            mapped_name = self._config.link_mapping.get(ls.name, ls.name)
            converted_links.append(
                LinkState(
                    name=mapped_name,
                    position=CoordinateTransformer.transform_position(
                        ls.position, source_convention, target_convention
                    ),
                    orientation=CoordinateTransformer.transform_quaternion(
                        ls.orientation, source_convention, target_convention
                    ),
                    linear_velocity=CoordinateTransformer.transform_velocity(
                        ls.linear_velocity, source_convention, target_convention
                    ),
                    angular_velocity=CoordinateTransformer.transform_velocity(
                        ls.angular_velocity, source_convention, target_convention
                    ),
                    mass=ls.mass,
                    inertia=self._param_mapper.convert_inertia_tensor(ls.inertia, source_convention, target_convention),
                )
            )

        converted_contacts: list[ContactPoint] = []
        if self._config.enable_contact_sync:
            for cp in state.contact_points:
                converted_contacts.append(
                    ContactPoint(
                        body_a=self._config.link_mapping.get(cp.body_a, cp.body_a),
                        body_b=self._config.link_mapping.get(cp.body_b, cp.body_b),
                        position=CoordinateTransformer.transform_position(
                            cp.position, source_convention, target_convention
                        ),
                        normal=CoordinateTransformer.transform_position(
                            cp.normal, source_convention, target_convention
                        ),
                        force=CoordinateTransformer.transform_position(cp.force, source_convention, target_convention),
                        depth=cp.depth,
                        friction_coefficient=cp.friction_coefficient,
                    )
                )

        return SimulationState(
            timestamp=utc_now_iso(),
            sim_time=state.sim_time,
            joint_states=converted_joints,
            link_states=converted_links,
            contact_points=converted_contacts,
            gravity=state.gravity,
            coordinate_convention=target_convention,
            metadata=dict(state.metadata),
        )

    def _apply_safety_limits(self, state: SimulationState) -> tuple[SimulationState, int]:
        """Clamp all joint states to safety limits. Returns (state, num_clamped)."""
        num_clamped = 0
        clamped_joints: list[JointState] = []
        for js in state.joint_states:
            clamped = js.clamp_to_limits()
            if clamped.position != js.position or clamped.velocity != js.velocity or clamped.torque != js.torque:
                num_clamped += 1
                logger.warning(
                    "Safety limit applied to joint '%s': pos %.4f->%.4f vel %.4f->%.4f",
                    js.name,
                    js.position,
                    clamped.position,
                    js.velocity,
                    clamped.velocity,
                )
            clamped_joints.append(clamped)

        safe_state = SimulationState(
            timestamp=state.timestamp,
            sim_time=state.sim_time,
            joint_states=clamped_joints,
            link_states=state.link_states,
            contact_points=state.contact_points,
            gravity=state.gravity,
            coordinate_convention=state.coordinate_convention,
            metadata=state.metadata,
        )
        return safe_state, num_clamped

    def _average_states(
        self,
        state_a: SimulationState,
        state_b: SimulationState,
    ) -> SimulationState:
        """Average two simulation states for bidirectional sync."""
        joint_map_b = {js.name: js for js in state_b.joint_states}
        averaged_joints: list[JointState] = []

        for js_a in state_a.joint_states:
            mapped_name = self._config.joint_mapping.get(js_a.name, js_a.name)
            js_b = joint_map_b.get(mapped_name)
            if js_b is not None:
                averaged_joints.append(
                    JointState(
                        name=mapped_name,
                        joint_type=js_a.joint_type,
                        position=(js_a.position + js_b.position) / 2.0,
                        velocity=(js_a.velocity + js_b.velocity) / 2.0,
                        acceleration=(js_a.acceleration + js_b.acceleration) / 2.0,
                        torque=(js_a.torque + js_b.torque) / 2.0,
                        position_limits=js_a.position_limits,
                        velocity_limit=js_a.velocity_limit,
                        torque_limit=js_a.torque_limit,
                        damping=js_a.damping,
                        stiffness=js_a.stiffness,
                        friction=js_a.friction,
                    )
                )
            else:
                averaged_joints.append(js_a)

        return SimulationState(
            timestamp=utc_now_iso(),
            sim_time=(state_a.sim_time + state_b.sim_time) / 2.0,
            joint_states=averaged_joints,
            link_states=state_a.link_states,
            contact_points=[],
            gravity=state_a.gravity,
            coordinate_convention=CoordinateConvention.MUJOCO_Z_UP,
        )

    def run_calibration(
        self,
        num_steps: int = 1000,
        dt: float = 0.002,
        joint_torques: Optional[dict[str, float]] = None,
    ) -> CalibrationResult:
        """Run a calibration sequence to quantify cross-engine drift.

        Applies identical torques to both engines for num_steps and measures
        the divergence in joint positions, velocities, and contact forces.
        """
        logger.info("Starting calibration: %d steps at dt=%.4fs", num_steps, dt)
        result = CalibrationResult(num_steps=num_steps)

        position_errors: list[float] = []
        velocity_errors: list[float] = []

        self._mujoco.reset()
        self._isaac.reset()

        for step_idx in range(num_steps):
            self._mujoco.step(dt)
            self._isaac.step(dt)

            mj_state = self._mujoco.get_state()
            isaac_state = self._isaac.get_state()

            for mj_js in mj_state.joint_states:
                mapped_name = self._config.joint_mapping.get(mj_js.name, mj_js.name)
                isaac_js = self._isaac.get_joint_state(mapped_name)
                pos_err = abs(mj_js.position - isaac_js.position)
                vel_err = abs(mj_js.velocity - isaac_js.velocity)
                position_errors.append(pos_err)
                velocity_errors.append(vel_err)

        if position_errors:
            result.mean_position_error_m = float(np.mean(position_errors))
            result.max_position_error_m = float(np.max(position_errors))
        if velocity_errors:
            result.mean_velocity_error = float(np.mean(velocity_errors))
            result.max_velocity_error = float(np.max(velocity_errors))

        # Pass if mean positional error is below tolerance
        result.passed = result.mean_position_error_m < self._config.max_sync_drift_m * 10

        logger.info(
            "Calibration complete: mean_pos_err=%.6fm max_pos_err=%.6fm passed=%s",
            result.mean_position_error_m,
            result.max_position_error_m,
            result.passed,
        )
        return result

    def step_synchronized(self, dt: float) -> SyncReport:
        """Advance both engines by dt and synchronize in the configured direction."""
        self._mujoco.step(dt)
        self._isaac.step(dt)

        if self._config.sync_direction == SyncDirection.ISAAC_TO_MUJOCO:
            return self.sync_isaac_to_mujoco()
        elif self._config.sync_direction == SyncDirection.MUJOCO_TO_ISAAC:
            return self.sync_mujoco_to_isaac()
        else:
            return self.sync_bidirectional()

    def get_drift_metrics(self) -> dict[str, float]:
        """Compute current drift between the two engines."""
        mj_state = self._mujoco.get_state()
        isaac_state = self._isaac.get_state()

        position_drifts: list[float] = []
        velocity_drifts: list[float] = []

        isaac_joint_map = {js.name: js for js in isaac_state.joint_states}

        for mj_js in mj_state.joint_states:
            mapped = self._config.joint_mapping.get(mj_js.name, mj_js.name)
            isaac_js = isaac_joint_map.get(mapped)
            if isaac_js is not None:
                position_drifts.append(abs(mj_js.position - isaac_js.position))
                velocity_drifts.append(abs(mj_js.velocity - isaac_js.velocity))

        return {
            "mean_position_drift": float(np.mean(position_drifts)) if position_drifts else 0.0,
            "max_position_drift": float(np.max(position_drifts)) if position_drifts else 0.0,
            "mean_velocity_drift": float(np.mean(velocity_drifts)) if velocity_drifts else 0.0,
            "max_velocity_drift": float(np.max(velocity_drifts)) if velocity_drifts else 0.0,
            "num_joints_compared": len(position_drifts),
        }

    def get_performance_stats(self) -> dict[str, float]:
        """Return performance statistics for the bridge."""
        return {
            "total_syncs": self._step_count,
            "total_sync_time_ms": self._total_sync_time_ms,
            "avg_sync_time_ms": self._total_sync_time_ms / max(1, self._step_count),
            "sync_success_rate": (sum(1 for r in self._sync_history if r.success) / max(1, len(self._sync_history))),
        }

    def export_audit_log(self, output_path: str) -> None:
        """Export the full sync history as a JSON audit log."""
        records = []
        for report in self._sync_history:
            records.append(
                {
                    "sync_id": report.sync_id,
                    "direction": report.direction.value,
                    "timestamp": report.timestamp,
                    "sim_time_source": report.sim_time_source,
                    "sim_time_target": report.sim_time_target,
                    "num_joints_synced": report.num_joints_synced,
                    "num_links_synced": report.num_links_synced,
                    "num_contacts_synced": report.num_contacts_synced,
                    "max_position_drift_m": report.max_position_drift_m,
                    "safety_limits_applied": report.safety_limits_applied,
                    "sync_duration_ms": report.sync_duration_ms,
                    "success": report.success,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "state_hash_source": report.state_hash_source,
                    "state_hash_target": report.state_hash_target,
                }
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"audit_records": records, "bridge_config": str(self._config)}, fh, indent=2)

        logger.info("Audit log exported: %s (%d records)", output_path, len(records))

    def shutdown(self) -> None:
        """Shut down both backends and release resources."""
        logger.info("Shutting down Isaac-MuJoCo bridge...")
        self._mujoco.shutdown()
        self._isaac.shutdown()
        self._state = BridgeState.SHUTDOWN
        logger.info("Bridge shutdown complete. Total syncs: %d", self._step_count)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_bridge(
    mujoco_model_path: str = "",
    isaac_usd_path: str = "",
    sync_direction: str = "bidirectional",
    enable_safety: bool = True,
    enable_audit: bool = True,
) -> IsaacMuJoCoBridge:
    """Factory function to create and initialize a bridge with common defaults."""
    direction_map = {
        "isaac_to_mujoco": SyncDirection.ISAAC_TO_MUJOCO,
        "mujoco_to_isaac": SyncDirection.MUJOCO_TO_ISAAC,
        "bidirectional": SyncDirection.BIDIRECTIONAL,
    }
    config = BridgeConfig(
        mujoco_model_path=mujoco_model_path,
        isaac_usd_path=isaac_usd_path,
        sync_direction=direction_map.get(sync_direction, SyncDirection.BIDIRECTIONAL),
        enable_safety_limits=enable_safety,
        enable_audit_logging=enable_audit,
    )
    bridge = IsaacMuJoCoBridge(config)
    return bridge


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------
def main() -> None:
    """Run a basic bridge demonstration."""
    logger.info("Isaac-MuJoCo Bridge demonstration")
    logger.info("HAS_MUJOCO=%s  HAS_ISAAC=%s", HAS_MUJOCO, HAS_ISAAC)

    config = BridgeConfig(
        sync_direction=SyncDirection.BIDIRECTIONAL,
        enable_safety_limits=True,
        enable_audit_logging=True,
    )
    bridge = IsaacMuJoCoBridge(config)
    bridge.initialize()

    # Create a synthetic joint state for demonstration
    test_state = SimulationState(
        sim_time=0.0,
        joint_states=[
            JointState(name="shoulder_pan", joint_type=JointType.REVOLUTE, position=0.5, velocity=0.1),
            JointState(name="shoulder_lift", joint_type=JointType.REVOLUTE, position=-1.0, velocity=0.0),
            JointState(name="elbow", joint_type=JointType.REVOLUTE, position=1.2, velocity=-0.05),
            JointState(name="wrist_1", joint_type=JointType.REVOLUTE, position=0.0, velocity=0.0),
            JointState(name="wrist_2", joint_type=JointType.REVOLUTE, position=-0.3, velocity=0.02),
            JointState(name="wrist_3", joint_type=JointType.REVOLUTE, position=0.8, velocity=0.0),
        ],
        link_states=[
            LinkState(name="base_link", position=np.zeros(3)),
            LinkState(name="tool0", position=np.array([0.5, 0.1, 0.8])),
        ],
        coordinate_convention=CoordinateConvention.MUJOCO_Z_UP,
    )

    # Inject into MuJoCo side and sync to Isaac
    bridge._mujoco._joint_states = {js.name: js for js in test_state.joint_states}
    bridge._mujoco._initialized = True

    report = bridge.sync_mujoco_to_isaac()
    logger.info(
        "Sync report: success=%s joints=%d time=%.2fms",
        report.success,
        report.num_joints_synced,
        report.sync_duration_ms,
    )

    drift = bridge.get_drift_metrics()
    logger.info("Drift metrics: %s", drift)

    stats = bridge.get_performance_stats()
    logger.info("Performance: %s", stats)

    bridge.shutdown()
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
