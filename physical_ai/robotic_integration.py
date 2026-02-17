"""Surgical robotics integration for oncology trials.

Provides abstractions for surgical robot systems (e.g. da Vinci)
used in oncology procedures, enabling federated-learning-informed
surgical planning and telemetry collection across trial sites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class RobotType(str, Enum):
    """Supported surgical robot platforms."""

    DA_VINCI = "da_vinci"
    ROBOTIC_ARM = "robotic_arm"
    SPECIMEN_HANDLER = "specimen_handler"
    GENERIC = "generic"


class TaskStatus(str, Enum):
    """Execution status for a robotic task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RoboticTask:
    """Specification for a robotic procedure task.

    Attributes:
        task_id: Unique task identifier.
        task_type: Category (e.g. ``"biopsy"``, ``"resection"``).
        target_location: 3D coordinates of the surgical target.
        constraints: Safety constraints for the procedure.
        status: Current execution status.
    """

    task_id: str
    task_type: str
    target_location: np.ndarray = field(default_factory=lambda: np.zeros(3))
    constraints: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    telemetry: list[dict] = field(default_factory=list)


class SurgicalRobotInterface:
    """Interface to a surgical robot system for oncology procedures.

    This abstraction allows federated-learning models to inform
    surgical planning while collecting de-identified telemetry data
    for cross-site model improvement.

    Args:
        robot_id: Unique robot identifier at this site.
        robot_type: Type of surgical robot.
        capabilities: List of supported procedure types.
    """

    def __init__(
        self,
        robot_id: str,
        robot_type: RobotType = RobotType.GENERIC,
        capabilities: list[str] | None = None,
    ):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.capabilities = capabilities or ["biopsy", "resection", "specimen_handling"]
        self._task_log: list[RoboticTask] = []
        self._is_connected = False

    def connect(self) -> bool:
        """Establish connection to the robot system (simulated)."""
        self._is_connected = True
        logger.info("Robot %s (%s) connected.", self.robot_id, self.robot_type.value)
        return True

    def disconnect(self) -> None:
        """Disconnect from the robot system."""
        self._is_connected = False
        logger.info("Robot %s disconnected.", self.robot_id)

    def plan_procedure(
        self,
        task_type: str,
        target_location: np.ndarray,
        tumor_volume: float = 0.0,
        safety_margin_mm: float = 5.0,
    ) -> RoboticTask:
        """Plan a surgical procedure using AI-informed parameters.

        Args:
            task_type: Type of procedure to plan.
            target_location: 3D target coordinates.
            tumor_volume: Estimated tumor volume for approach planning.
            safety_margin_mm: Required safety margin around the tumor.

        Returns:
            A RoboticTask ready for execution.

        Raises:
            ValueError: If the task type is not supported.
        """
        if task_type not in self.capabilities:
            raise ValueError(f"Task '{task_type}' not supported. Available: {self.capabilities}")

        task = RoboticTask(
            task_id=f"{self.robot_id}_{task_type}_{len(self._task_log)}",
            task_type=task_type,
            target_location=np.array(target_location),
            constraints={
                "safety_margin_mm": safety_margin_mm,
                "tumor_volume_cm3": tumor_volume,
                "max_force_n": 5.0,
                "max_velocity_mm_s": 10.0,
            },
        )
        self._task_log.append(task)
        logger.info("Planned task %s at location %s", task.task_id, target_location)
        return task

    def execute_task(self, task: RoboticTask, seed: int = 0) -> RoboticTask:
        """Execute a planned robotic task (simulated).

        In simulation, generates realistic telemetry data representing
        the procedure execution.

        Args:
            task: The task to execute.
            seed: Random seed for simulation reproducibility.

        Returns:
            The task with updated status and telemetry.
        """
        if not self._is_connected:
            task.status = TaskStatus.FAILED
            logger.warning("Cannot execute — robot not connected.")
            return task

        rng = np.random.default_rng(seed)
        task.status = TaskStatus.RUNNING

        # Simulate telemetry over 10 time steps
        for step in range(10):
            position = task.target_location + rng.normal(0, 0.5, size=3)
            force = rng.uniform(0.1, task.constraints.get("max_force_n", 5.0))
            task.telemetry.append(
                {
                    "step": step,
                    "position": position.tolist(),
                    "force_n": round(float(force), 3),
                    "velocity_mm_s": round(float(rng.uniform(1, 10)), 3),
                }
            )

        success = rng.random() > 0.05  # 95% success rate in simulation
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED

        logger.info(
            "Task %s %s — %d telemetry records",
            task.task_id,
            task.status.value,
            len(task.telemetry),
        )
        return task

    def get_telemetry_features(self, task: RoboticTask) -> np.ndarray:
        """Extract de-identified feature vector from task telemetry.

        Produces a fixed-size numeric vector summarizing the procedure,
        suitable for federated model training without PHI.
        """
        if not task.telemetry:
            return np.zeros(10)

        forces = [t["force_n"] for t in task.telemetry]
        velocities = [t["velocity_mm_s"] for t in task.telemetry]
        positions = np.array([t["position"] for t in task.telemetry])

        features = [
            np.mean(forces),
            np.std(forces),
            np.max(forces),
            np.mean(velocities),
            np.std(velocities),
            np.max(velocities),
            float(np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
            float(len(task.telemetry)),
            1.0 if task.status == TaskStatus.COMPLETED else 0.0,
            task.constraints.get("tumor_volume_cm3", 0.0),
        ]
        return np.array(features, dtype=np.float64)

    def get_task_log(self) -> list[dict]:
        """Return de-identified summary of all executed tasks."""
        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "status": t.status.value,
                "telemetry_records": len(t.telemetry),
            }
            for t in self._task_log
        ]
