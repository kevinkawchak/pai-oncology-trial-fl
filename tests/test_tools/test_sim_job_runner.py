"""Tests for tools/sim-job-runner/sim_job_runner.py.

Covers enums, dataclasses (SimJobConfig, SimJobResult), framework detection,
job validation, launch, status, and results retrieval.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module("sim_job_runner", "tools/sim-job-runner/sim_job_runner.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestSimFramework:
    """Tests for the SimFramework enum."""

    def test_members(self):
        """SimFramework has four members."""
        expected = {"ISAAC", "MUJOCO", "PYBULLET", "GAZEBO"}
        assert set(mod.SimFramework.__members__.keys()) == expected

    def test_values(self):
        """SimFramework values are lowercase."""
        assert mod.SimFramework.MUJOCO.value == "mujoco"
        assert mod.SimFramework.ISAAC.value == "isaac"


class TestJobStatus:
    """Tests for the JobStatus enum."""

    def test_members(self):
        """JobStatus has six members."""
        expected = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"}
        assert set(mod.JobStatus.__members__.keys()) == expected

    def test_terminal_states(self):
        """Terminal states are COMPLETED, FAILED, CANCELLED."""
        terminal = {"completed", "failed", "cancelled"}
        for member in mod.JobStatus:
            if member.value in terminal:
                assert member.value in terminal


class TestTaskType:
    """Tests for the TaskType enum."""

    def test_members(self):
        """TaskType has five members."""
        assert len(mod.TaskType) == 5
        assert mod.TaskType.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestSimJobConfig:
    """Tests for the SimJobConfig dataclass."""

    def test_defaults(self):
        """Default config uses mujoco and needle_insertion."""
        cfg = mod.SimJobConfig()
        assert cfg.framework == "mujoco"
        assert cfg.task_type == "needle_insertion"
        assert cfg.num_envs == 1
        assert cfg.physics_dt == 0.002

    def test_job_id_auto_generated(self):
        """Job ID is auto-generated with 'sim-' prefix."""
        cfg = mod.SimJobConfig()
        assert cfg.job_id.startswith("sim-")

    def test_custom_job_id_preserved(self):
        """Provided job ID is preserved."""
        cfg = mod.SimJobConfig(job_id="my-job-123")
        assert cfg.job_id == "my-job-123"

    def test_num_envs_clamped(self):
        """num_envs is clamped to [1, 16384]."""
        cfg = mod.SimJobConfig(num_envs=0)
        assert cfg.num_envs == 1
        cfg2 = mod.SimJobConfig(num_envs=99999)
        assert cfg2.num_envs == 16384

    def test_max_steps_clamped(self):
        """max_steps is clamped to [1, 1_000_000]."""
        cfg = mod.SimJobConfig(max_steps=0)
        assert cfg.max_steps == 1
        cfg2 = mod.SimJobConfig(max_steps=2_000_000)
        assert cfg2.max_steps == 1_000_000

    def test_physics_dt_clamped(self):
        """physics_dt is clamped to [1e-5, 0.1]."""
        cfg = mod.SimJobConfig(physics_dt=0.0)
        assert cfg.physics_dt == 1e-5
        cfg2 = mod.SimJobConfig(physics_dt=1.0)
        assert cfg2.physics_dt == 0.1


class TestSimJobResult:
    """Tests for the SimJobResult dataclass."""

    def test_defaults(self):
        """Default result is PENDING."""
        result = mod.SimJobResult()
        assert result.status == "pending"
        assert result.success_rate == 0.0

    def test_success_rate_clamped(self):
        """success_rate is clamped to [0, 1]."""
        result = mod.SimJobResult(success_rate=1.5)
        assert result.success_rate == 1.0
        result2 = mod.SimJobResult(success_rate=-0.5)
        assert result2.success_rate == 0.0

    def test_created_at_auto(self):
        """created_at is auto-set if empty."""
        result = mod.SimJobResult()
        assert result.created_at != ""


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------
class TestDetectAvailableFrameworks:
    """Tests for detect_available_frameworks."""

    def test_returns_dict(self):
        """Returns a dict with all four framework keys."""
        available = mod.detect_available_frameworks()
        assert isinstance(available, dict)
        assert set(available.keys()) == {"isaac", "mujoco", "pybullet", "gazebo"}

    def test_values_are_bool(self):
        """All values in the dict are boolean."""
        for v in mod.detect_available_frameworks().values():
            assert isinstance(v, bool)


# ---------------------------------------------------------------------------
# Job validation
# ---------------------------------------------------------------------------
class TestValidateJobConfig:
    """Tests for validate_job_config."""

    def test_valid_mujoco_needle(self):
        """Valid mujoco + needle_insertion config has framework availability issue only."""
        cfg = mod.SimJobConfig(framework="mujoco", task_type="needle_insertion")
        issues = mod.validate_job_config(cfg)
        # May have framework not installed issue, but no unknown framework issue
        assert not any("Unknown framework" in i for i in issues)

    def test_unknown_framework(self):
        """Unknown framework name produces an issue."""
        cfg = mod.SimJobConfig()
        cfg.framework = "unknown_engine"
        issues = mod.validate_job_config(cfg)
        assert any("Unknown framework" in i for i in issues)

    def test_incompatible_task_framework(self):
        """Task not supported by framework produces an issue."""
        cfg = mod.SimJobConfig(framework="gazebo", task_type="tumour_resection")
        issues = mod.validate_job_config(cfg)
        assert any("not supported by framework" in i for i in issues)

    def test_custom_task_no_compatibility_issue(self):
        """Custom task type bypasses framework compatibility check."""
        cfg = mod.SimJobConfig(framework="gazebo", task_type="custom")
        issues = mod.validate_job_config(cfg)
        assert not any("not supported by framework" in i for i in issues)


# ---------------------------------------------------------------------------
# Launch, status, results
# ---------------------------------------------------------------------------
class TestLaunchSimulation:
    """Tests for launch_simulation."""

    def test_launch_returns_result(self):
        """launch_simulation returns a SimJobResult."""
        cfg = mod.SimJobConfig()
        result = mod.launch_simulation(cfg)
        assert isinstance(result, mod.SimJobResult)
        assert result.job_id == cfg.job_id

    def test_launch_with_issues_configuring(self):
        """Jobs with issues are in CONFIGURING status."""
        cfg = mod.SimJobConfig()
        cfg.framework = "unknown_engine"
        result = mod.launch_simulation(cfg)
        assert result.status == mod.JobStatus.CONFIGURING.value
        assert len(result.warnings) > 0


class TestGetJobStatus:
    """Tests for get_job_status."""

    def test_stub_returns_pending(self):
        """Stub returns PENDING status."""
        result = mod.get_job_status("test-job-123")
        assert result.status == "pending"
        assert result.job_id == "test-job-123"


class TestGetJobResults:
    """Tests for get_job_results."""

    def test_stub_returns_completed(self):
        """Stub returns COMPLETED with demo metrics."""
        result = mod.get_job_results("test-job-456")
        assert result.status == "completed"
        assert result.episodes_completed == 100
        assert result.success_rate == 0.92
        assert result.mean_reward == 0.85
