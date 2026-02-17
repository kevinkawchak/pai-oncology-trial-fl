"""Surgical task definitions for oncology robotic procedures.

Defines standardised task specifications for common oncology surgical
procedures including biopsy needle insertion, tissue resection, specimen
handling, and shunt placement.  Each task includes clinical accuracy
thresholds derived from surgical oncology standards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ProcedureType(str, Enum):
    """Standard oncology robotic procedure types."""

    NEEDLE_BIOPSY = "needle_biopsy"
    TISSUE_RESECTION = "tissue_resection"
    SPECIMEN_HANDLING = "specimen_handling"
    SHUNT_PLACEMENT = "shunt_placement"
    LYMPH_NODE_DISSECTION = "lymph_node_dissection"
    ABLATION = "ablation"


class SafetyLevel(str, Enum):
    """Safety classification for surgical procedures."""

    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class ClinicalThreshold:
    """Clinical accuracy and safety thresholds for a procedure.

    Attributes:
        max_position_error_mm: Maximum acceptable positioning error.
        max_force_n: Maximum allowable force.
        min_success_rate: Minimum acceptable success rate [0, 1].
        max_collision_rate: Maximum collision rate [0, 1].
        max_procedure_time_s: Maximum acceptable procedure time.
        safety_level: Overall safety classification.
    """

    max_position_error_mm: float = 2.0
    max_force_n: float = 5.0
    min_success_rate: float = 0.95
    max_collision_rate: float = 0.0
    max_procedure_time_s: float = 300.0
    safety_level: SafetyLevel = SafetyLevel.MODERATE_RISK


@dataclass
class SurgicalTaskDefinition:
    """Full definition of a surgical task for robot training.

    Includes procedure specification, clinical thresholds, environment
    parameters, and reward structure for reinforcement learning.

    Attributes:
        procedure: The type of procedure.
        description: Human-readable description.
        thresholds: Clinical accuracy and safety thresholds.
        workspace_bounds_mm: 3D workspace limits [(min, max)] per axis.
        observation_keys: Sensor observations available during task.
        reward_weights: Weights for reward components.
        domain_randomization: Parameters to randomize during training
            for sim-to-real transfer.
    """

    procedure: ProcedureType
    description: str = ""
    thresholds: ClinicalThreshold = field(default_factory=ClinicalThreshold)
    workspace_bounds_mm: list[tuple[float, float]] = field(
        default_factory=lambda: [(-100, 100), (-100, 100), (-50, 50)]
    )
    observation_keys: list[str] = field(
        default_factory=lambda: [
            "end_effector_pos",
            "end_effector_vel",
            "force_torque",
            "tumor_location",
            "tissue_deformation",
        ]
    )
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "position_accuracy": 1.0,
            "force_penalty": -2.0,
            "collision_penalty": -10.0,
            "smoothness": 0.5,
            "time_penalty": -0.1,
            "success_bonus": 5.0,
        }
    )
    domain_randomization: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "tissue_stiffness": (0.8, 1.2),
            "friction_coefficient": (0.3, 0.7),
            "tumor_position_noise_mm": (0.0, 2.0),
            "sensor_noise_std": (0.0, 0.05),
        }
    )


# ---- Standard task definitions ----

STANDARD_TASKS: dict[ProcedureType, SurgicalTaskDefinition] = {
    ProcedureType.NEEDLE_BIOPSY: SurgicalTaskDefinition(
        procedure=ProcedureType.NEEDLE_BIOPSY,
        description=(
            "Image-guided needle insertion for tissue sampling.  Requires "
            "sub-millimetre accuracy to target tumor core while avoiding "
            "critical structures."
        ),
        thresholds=ClinicalThreshold(
            max_position_error_mm=1.5,
            max_force_n=3.0,
            min_success_rate=0.97,
            max_collision_rate=0.0,
            max_procedure_time_s=120.0,
            safety_level=SafetyLevel.HIGH_RISK,
        ),
    ),
    ProcedureType.TISSUE_RESECTION: SurgicalTaskDefinition(
        procedure=ProcedureType.TISSUE_RESECTION,
        description=(
            "Robotic tissue resection with margin control.  The robot "
            "follows a pre-planned resection boundary while maintaining "
            "a safety margin around the tumor."
        ),
        thresholds=ClinicalThreshold(
            max_position_error_mm=2.0,
            max_force_n=5.0,
            min_success_rate=0.95,
            max_collision_rate=0.0,
            max_procedure_time_s=600.0,
            safety_level=SafetyLevel.CRITICAL,
        ),
    ),
    ProcedureType.SPECIMEN_HANDLING: SurgicalTaskDefinition(
        procedure=ProcedureType.SPECIMEN_HANDLING,
        description=(
            "Transfer of excised tissue specimen to pathology container.  "
            "Requires gentle handling to preserve tissue integrity for "
            "histological analysis."
        ),
        thresholds=ClinicalThreshold(
            max_position_error_mm=5.0,
            max_force_n=2.0,
            min_success_rate=0.99,
            max_collision_rate=0.01,
            max_procedure_time_s=60.0,
            safety_level=SafetyLevel.LOW_RISK,
        ),
    ),
    ProcedureType.LYMPH_NODE_DISSECTION: SurgicalTaskDefinition(
        procedure=ProcedureType.LYMPH_NODE_DISSECTION,
        description=(
            "Robotic lymph node dissection adjacent to major vessels.  "
            "Requires precise tissue identification and careful "
            "manoeuvring to avoid vascular injury."
        ),
        thresholds=ClinicalThreshold(
            max_position_error_mm=1.0,
            max_force_n=2.5,
            min_success_rate=0.96,
            max_collision_rate=0.0,
            max_procedure_time_s=900.0,
            safety_level=SafetyLevel.CRITICAL,
        ),
    ),
    ProcedureType.ABLATION: SurgicalTaskDefinition(
        procedure=ProcedureType.ABLATION,
        description=(
            "Thermal or cryogenic ablation of tumor tissue.  Requires "
            "precise probe placement and real-time temperature monitoring."
        ),
        thresholds=ClinicalThreshold(
            max_position_error_mm=2.0,
            max_force_n=4.0,
            min_success_rate=0.93,
            max_collision_rate=0.0,
            max_procedure_time_s=1200.0,
            safety_level=SafetyLevel.HIGH_RISK,
        ),
    ),
}


class SurgicalTaskEvaluator:
    """Evaluates robotic procedure performance against clinical thresholds.

    Takes telemetry from a completed procedure and checks whether it
    meets the defined clinical accuracy and safety requirements.
    """

    def __init__(self, task_def: SurgicalTaskDefinition) -> None:
        self.task_def = task_def

    def evaluate(
        self,
        position_errors_mm: list[float],
        forces_n: list[float],
        collisions: int,
        total_attempts: int,
        procedure_time_s: float,
    ) -> dict[str, bool | float]:
        """Evaluate a procedure against clinical thresholds.

        Args:
            position_errors_mm: Per-step positioning errors.
            forces_n: Per-step force readings.
            collisions: Number of collisions detected.
            total_attempts: Total procedure attempts for success rate.
            procedure_time_s: Total procedure duration.

        Returns:
            Dictionary of metric name -> (value, passed) pairs.
        """
        t = self.task_def.thresholds
        max_pos_err = float(np.max(position_errors_mm)) if position_errors_mm else 0.0
        max_force = float(np.max(forces_n)) if forces_n else 0.0
        collision_rate = collisions / max(total_attempts, 1)
        success_rate = 1.0 - collision_rate

        results: dict[str, bool | float] = {
            "max_position_error_mm": max_pos_err,
            "position_passed": max_pos_err <= t.max_position_error_mm,
            "max_force_n": max_force,
            "force_passed": max_force <= t.max_force_n,
            "collision_rate": collision_rate,
            "collision_passed": collision_rate <= t.max_collision_rate,
            "success_rate": success_rate,
            "success_passed": success_rate >= t.min_success_rate,
            "procedure_time_s": procedure_time_s,
            "time_passed": procedure_time_s <= t.max_procedure_time_s,
        }

        all_passed = all(v for k, v in results.items() if k.endswith("_passed"))
        results["clinical_ready"] = all_passed

        logger.info(
            "Procedure %s evaluation: clinical_ready=%s",
            self.task_def.procedure.value,
            all_passed,
        )
        return results
