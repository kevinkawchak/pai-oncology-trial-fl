"""Tests for unification/simulation_physics/isaac_mujoco_bridge.py.

Covers enums, dataclasses (JointState, LinkState, ContactPoint, SimulationState,
BridgeConfig, SyncReport, CalibrationResult), CoordinateTransformer,
ParameterMapper, IsaacSimBackend, IsaacMuJoCoBridge, and factory function.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("isaac_mujoco_bridge", "unification/simulation_physics/isaac_mujoco_bridge.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestBridgeState:
    """Tests for the BridgeState enum."""

    def test_members(self):
        """BridgeState has seven members."""
        expected = {"UNINITIALIZED", "CONFIGURING", "READY", "RUNNING", "PAUSED", "ERROR", "SHUTDOWN"}
        assert set(mod.BridgeState.__members__.keys()) == expected

    def test_lifecycle_order(self):
        """Key lifecycle states have expected values."""
        assert mod.BridgeState.UNINITIALIZED.value == "uninitialized"
        assert mod.BridgeState.READY.value == "ready"
        assert mod.BridgeState.SHUTDOWN.value == "shutdown"


class TestSyncDirection:
    """Tests for the SyncDirection enum."""

    def test_members(self):
        """SyncDirection has three members."""
        expected = {"ISAAC_TO_MUJOCO", "MUJOCO_TO_ISAAC", "BIDIRECTIONAL"}
        assert set(mod.SyncDirection.__members__.keys()) == expected


class TestJointType:
    """Tests for the JointType enum."""

    def test_members(self):
        """JointType has six members."""
        assert len(mod.JointType) == 6
        assert mod.JointType.REVOLUTE.value == "revolute"
        assert mod.JointType.PRISMATIC.value == "prismatic"


class TestCoordinateConvention:
    """Tests for the CoordinateConvention enum."""

    def test_members(self):
        """CoordinateConvention has three members."""
        assert len(mod.CoordinateConvention) == 3


class TestSolverType:
    """Tests for the SolverType enum."""

    def test_members(self):
        """SolverType has four members."""
        assert len(mod.SolverType) == 4


# ---------------------------------------------------------------------------
# JointState dataclass
# ---------------------------------------------------------------------------
class TestJointState:
    """Tests for the JointState dataclass."""

    def test_defaults(self):
        """Default JointState has zero position/velocity."""
        js = mod.JointState(name="test", joint_type=mod.JointType.REVOLUTE)
        assert js.position == 0.0
        assert js.velocity == 0.0

    def test_clamp_to_limits(self):
        """clamp_to_limits constrains out-of-range values."""
        js = mod.JointState(
            name="j1",
            joint_type=mod.JointType.REVOLUTE,
            position=10.0,  # exceeds default [-pi, pi]
            velocity=100.0,  # exceeds velocity_limit
        )
        clamped = js.clamp_to_limits()
        assert clamped.position <= js.position_limits[1]
        assert clamped.velocity <= js.velocity_limit

    def test_clamp_preserves_within_limits(self):
        """clamp_to_limits preserves values within limits."""
        js = mod.JointState(
            name="j2",
            joint_type=mod.JointType.REVOLUTE,
            position=0.5,
            velocity=0.1,
        )
        clamped = js.clamp_to_limits()
        assert clamped.position == 0.5
        assert clamped.velocity == 0.1


# ---------------------------------------------------------------------------
# SimulationState dataclass
# ---------------------------------------------------------------------------
class TestSimulationState:
    """Tests for the SimulationState dataclass."""

    def test_defaults(self):
        """Default SimulationState has empty lists."""
        ss = mod.SimulationState()
        assert ss.joint_states == []
        assert ss.link_states == []
        assert ss.gravity == (0.0, 0.0, -9.81)

    def test_to_flat_array(self):
        """to_flat_array returns positions then velocities."""
        js1 = mod.JointState(name="a", joint_type=mod.JointType.REVOLUTE, position=1.0, velocity=2.0)
        js2 = mod.JointState(name="b", joint_type=mod.JointType.REVOLUTE, position=3.0, velocity=4.0)
        ss = mod.SimulationState(joint_states=[js1, js2])
        arr = ss.to_flat_array()
        np.testing.assert_array_equal(arr, [1.0, 3.0, 2.0, 4.0])

    def test_compute_hash_deterministic(self):
        """compute_hash produces same hash for same state."""
        js = mod.JointState(name="a", joint_type=mod.JointType.REVOLUTE, position=1.0, velocity=0.5)
        ss = mod.SimulationState(sim_time=0.1, joint_states=[js])
        h1 = ss.compute_hash()
        h2 = ss.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_compute_hash_differs(self):
        """Different states produce different hashes."""
        js1 = mod.JointState(name="a", joint_type=mod.JointType.REVOLUTE, position=1.0)
        js2 = mod.JointState(name="a", joint_type=mod.JointType.REVOLUTE, position=2.0)
        h1 = mod.SimulationState(joint_states=[js1]).compute_hash()
        h2 = mod.SimulationState(joint_states=[js2]).compute_hash()
        assert h1 != h2


# ---------------------------------------------------------------------------
# BridgeConfig
# ---------------------------------------------------------------------------
class TestBridgeConfig:
    """Tests for the BridgeConfig dataclass."""

    def test_defaults(self):
        """Default config is valid."""
        cfg = mod.BridgeConfig()
        assert cfg.sync_direction == mod.SyncDirection.BIDIRECTIONAL
        assert cfg.validate() == []

    def test_invalid_timestep(self):
        """Negative timestep produces validation issue."""
        cfg = mod.BridgeConfig(mujoco_timestep=-0.001)
        issues = cfg.validate()
        assert any("positive" in i for i in issues)

    def test_large_timestep_warning(self):
        """Timestep > 100ms produces a warning."""
        cfg = mod.BridgeConfig(isaac_timestep=0.2)
        issues = cfg.validate()
        assert any("unusually large" in i for i in issues)

    def test_negative_drift(self):
        """Negative max_sync_drift_m produces validation issue."""
        cfg = mod.BridgeConfig(max_sync_drift_m=-0.01)
        issues = cfg.validate()
        assert any("non-negative" in i for i in issues)


# ---------------------------------------------------------------------------
# CoordinateTransformer
# ---------------------------------------------------------------------------
class TestCoordinateTransformer:
    """Tests for the CoordinateTransformer class."""

    def test_same_convention_noop(self):
        """Same source and target returns copy of input."""
        pos = np.array([1.0, 2.0, 3.0])
        result = mod.CoordinateTransformer.transform_position(
            pos,
            mod.CoordinateConvention.MUJOCO_Z_UP,
            mod.CoordinateConvention.MUJOCO_Z_UP,
        )
        np.testing.assert_array_equal(result, pos)

    def test_y_up_to_z_up(self):
        """Y-up [0, 1, 0] maps to Z-up [0, 0, 1]."""
        pos = np.array([0.0, 1.0, 0.0])
        result = mod.CoordinateTransformer.transform_position(
            pos,
            mod.CoordinateConvention.ISAAC_Y_UP,
            mod.CoordinateConvention.MUJOCO_Z_UP,
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_roundtrip_position(self):
        """Y-up -> Z-up -> Y-up roundtrip preserves vector."""
        pos = np.array([1.0, 2.0, 3.0])
        intermediate = mod.CoordinateTransformer.transform_position(
            pos,
            mod.CoordinateConvention.ISAAC_Y_UP,
            mod.CoordinateConvention.MUJOCO_Z_UP,
        )
        result = mod.CoordinateTransformer.transform_position(
            intermediate,
            mod.CoordinateConvention.MUJOCO_Z_UP,
            mod.CoordinateConvention.ISAAC_Y_UP,
        )
        np.testing.assert_allclose(result, pos, atol=1e-10)

    def test_quaternion_same_noop(self):
        """Same convention quaternion transform is a no-op."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = mod.CoordinateTransformer.transform_quaternion(
            q,
            mod.CoordinateConvention.MUJOCO_Z_UP,
            mod.CoordinateConvention.MUJOCO_Z_UP,
        )
        np.testing.assert_array_equal(result, q)


# ---------------------------------------------------------------------------
# ParameterMapper
# ---------------------------------------------------------------------------
class TestParameterMapper:
    """Tests for the ParameterMapper class."""

    def test_mujoco_friction_to_physx(self):
        """MuJoCo friction maps to PhysX with static > dynamic."""
        result = mod.ParameterMapper.mujoco_friction_to_physx(0.5, 0.01, 0.001)
        assert result["static_friction"] > result["dynamic_friction"]
        assert result["dynamic_friction"] == 0.5

    def test_physx_friction_to_mujoco(self):
        """PhysX friction maps to MuJoCo with averaged sliding."""
        result = mod.ParameterMapper.physx_friction_to_mujoco(0.6, 0.4)
        assert result["sliding"] == pytest.approx(0.5)
        assert "condim" in result

    def test_damping_roundtrip_preserves_values(self):
        """Damping/stiffness values are preserved through conversion."""
        physx = mod.ParameterMapper.mujoco_damping_to_physx(10.0, 100.0)
        assert physx["drive_damping"] == 10.0
        assert physx["drive_stiffness"] == 100.0


# ---------------------------------------------------------------------------
# IsaacSimBackend (stub mode)
# ---------------------------------------------------------------------------
class TestIsaacSimBackend:
    """Tests for IsaacSimBackend in stub mode."""

    def test_initialize_stub(self):
        """Initialize succeeds in stub mode."""
        backend = mod.IsaacSimBackend()
        cfg = mod.BridgeConfig()
        assert backend.initialize(cfg) is True

    def test_step_advances_time(self):
        """Step advances sim_time in stub mode."""
        backend = mod.IsaacSimBackend()
        backend.initialize(mod.BridgeConfig())
        backend.step(0.01)
        state = backend.get_state()
        assert state.sim_time > 0.0

    def test_set_get_joint_state(self):
        """Joint state roundtrips through set/get."""
        backend = mod.IsaacSimBackend()
        backend.initialize(mod.BridgeConfig())
        js = mod.JointState(name="j1", joint_type=mod.JointType.REVOLUTE, position=1.5)
        backend.set_joint_state("j1", js)
        result = backend.get_joint_state("j1")
        assert result.position == 1.5

    def test_reset_clears_state(self):
        """Reset clears all joint and body states."""
        backend = mod.IsaacSimBackend()
        backend.initialize(mod.BridgeConfig())
        backend.set_joint_state("j1", mod.JointState(name="j1", joint_type=mod.JointType.REVOLUTE))
        backend.reset()
        state = backend.get_state()
        assert state.joint_states == []
        assert state.sim_time == 0.0


# ---------------------------------------------------------------------------
# IsaacMuJoCoBridge
# ---------------------------------------------------------------------------
class TestIsaacMuJoCoBridge:
    """Tests for the IsaacMuJoCoBridge class."""

    def test_initial_state(self):
        """Bridge starts UNINITIALIZED."""
        bridge = mod.IsaacMuJoCoBridge()
        assert bridge.state == mod.BridgeState.UNINITIALIZED
        assert bridge.step_count == 0

    def test_initialize(self):
        """Initialize transitions to READY."""
        bridge = mod.IsaacMuJoCoBridge()
        result = bridge.initialize()
        assert result is True
        assert bridge.state == mod.BridgeState.READY

    def test_shutdown(self):
        """Shutdown transitions to SHUTDOWN."""
        bridge = mod.IsaacMuJoCoBridge()
        bridge.initialize()
        bridge.shutdown()
        assert bridge.state == mod.BridgeState.SHUTDOWN

    def test_sync_history_empty_initially(self):
        """Sync history is empty before any syncs."""
        bridge = mod.IsaacMuJoCoBridge()
        assert bridge.sync_history == []

    def test_performance_stats(self):
        """Performance stats return expected keys."""
        bridge = mod.IsaacMuJoCoBridge()
        stats = bridge.get_performance_stats()
        assert "total_syncs" in stats
        assert "avg_sync_time_ms" in stats

    def test_create_bridge_factory(self):
        """create_bridge factory returns a bridge instance."""
        bridge = mod.create_bridge(sync_direction="bidirectional")
        assert isinstance(bridge, mod.IsaacMuJoCoBridge)
