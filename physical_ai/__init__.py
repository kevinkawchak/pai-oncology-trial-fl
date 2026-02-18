"""Physical AI integration for oncology clinical trials.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Bridges robotic systems, digital twins, and sensor fusion
with the federated learning platform.
"""

from physical_ai.digital_twin import PatientDigitalTwin, TumorModel
from physical_ai.framework_detection import FrameworkDetector
from physical_ai.robotic_integration import (
    RoboticTask,
    SurgicalRobotInterface,
)
from physical_ai.sensor_fusion import ClinicalSensorFusion, SensorReading
from physical_ai.simulation_bridge import SimulationBridge
from physical_ai.surgical_tasks import SurgicalTaskDefinition, SurgicalTaskEvaluator

__all__ = [
    "ClinicalSensorFusion",
    "FrameworkDetector",
    "PatientDigitalTwin",
    "RoboticTask",
    "SensorReading",
    "SimulationBridge",
    "SurgicalRobotInterface",
    "SurgicalTaskDefinition",
    "SurgicalTaskEvaluator",
    "TumorModel",
]
