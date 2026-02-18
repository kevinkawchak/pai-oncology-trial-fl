"""Tests for agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py.

Validates simulation orchestrator enums, dataclasses, job scheduling,
framework selection, and configuration creation.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_module

mod = load_module(
    "simulation_orchestrator",
    "agentic-ai/examples-agentic-ai/04_autonomous_simulation_orchestrator.py",
)


class TestOrchestratorEnums:
    """Verify all orchestrator enums."""

    def test_sim_framework_values(self):
        assert mod.SimFramework.ISAAC_SIM.value == "isaac_sim"
        assert mod.SimFramework.MUJOCO.value == "mujoco"
        assert mod.SimFramework.PYBULLET.value == "pybullet"

    def test_job_status_values(self):
        assert mod.JobStatus.QUEUED.value == "queued"
        assert mod.JobStatus.RUNNING.value == "running"
        assert mod.JobStatus.COMPLETED.value == "completed"
        assert mod.JobStatus.FAILED.value == "failed"
        assert mod.JobStatus.CANCELLED.value == "cancelled"
        assert mod.JobStatus.RETRYING.value == "retrying"

    def test_procedure_type_values(self):
        assert mod.ProcedureType.BIOPSY.value == "biopsy"
        assert mod.ProcedureType.RESECTION.value == "resection"
        assert mod.ProcedureType.ABLATION.value == "ablation"
        assert mod.ProcedureType.NEEDLE_INSERTION.value == "needle_insertion"

    def test_optimization_objective_values(self):
        assert mod.OptimizationObjective.TRAJECTORY_ACCURACY.value == "trajectory_accuracy"
        assert mod.OptimizationObjective.FORCE_COMPLIANCE.value == "force_compliance"
        assert mod.OptimizationObjective.COLLISION_AVOIDANCE.value == "collision_avoidance"
        assert mod.OptimizationObjective.TISSUE_SAFETY.value == "tissue_safety"

    def test_orchestrator_state_values(self):
        assert mod.OrchestratorState.IDLE.value == "idle"
        assert mod.OrchestratorState.PLANNING.value == "planning"
        assert mod.OrchestratorState.SUBMITTING.value == "submitting"
        assert mod.OrchestratorState.MONITORING.value == "monitoring"
        assert mod.OrchestratorState.COMPLETE.value == "complete"
        assert mod.OrchestratorState.ERROR.value == "error"


class TestSimulationConfig:
    """Verify SimulationConfig dataclass."""

    def test_simulation_config_defaults(self):
        cfg = mod.SimulationConfig()
        assert cfg.framework == mod.SimFramework.MUJOCO
        assert cfg.procedure == mod.ProcedureType.BIOPSY
        assert cfg.robot_model == "kuka_iiwa_14"
        assert cfg.time_step_s == 0.001
        assert cfg.max_duration_s == 60.0

    def test_simulation_config_custom(self):
        cfg = mod.SimulationConfig(
            framework=mod.SimFramework.PYBULLET,
            procedure=mod.ProcedureType.RESECTION,
            robot_model="da_vinci_xi",
            time_step_s=0.002,
        )
        assert cfg.framework == mod.SimFramework.PYBULLET
        assert cfg.procedure == mod.ProcedureType.RESECTION
        assert cfg.robot_model == "da_vinci_xi"

    def test_simulation_config_gravity_default(self):
        cfg = mod.SimulationConfig()
        assert cfg.gravity == [0.0, 0.0, -9.81]

    def test_simulation_config_unique_ids(self):
        c1 = mod.SimulationConfig()
        c2 = mod.SimulationConfig()
        assert c1.config_id != c2.config_id

    def test_simulation_config_id_prefix(self):
        cfg = mod.SimulationConfig()
        assert cfg.config_id.startswith("CFG-")


class TestFrameworkSelection:
    """Verify framework enumeration and selection logic."""

    def test_all_frameworks_distinct(self):
        values = [f.value for f in mod.SimFramework]
        assert len(values) == len(set(values))

    def test_framework_count(self):
        assert len(list(mod.SimFramework)) == 3

    def test_procedure_type_count(self):
        procedures = list(mod.ProcedureType)
        assert len(procedures) >= 5

    def test_optimization_objective_count(self):
        objectives = list(mod.OptimizationObjective)
        assert len(objectives) >= 5


class TestModuleAttributes:
    """Verify the module exposes expected top-level attributes."""

    def test_has_sim_framework(self):
        assert hasattr(mod, "SimFramework")

    def test_has_job_status(self):
        assert hasattr(mod, "JobStatus")

    def test_has_simulation_config(self):
        assert hasattr(mod, "SimulationConfig")

    def test_has_orchestrator_state(self):
        assert hasattr(mod, "OrchestratorState")

    def test_has_procedure_type(self):
        assert hasattr(mod, "ProcedureType")
