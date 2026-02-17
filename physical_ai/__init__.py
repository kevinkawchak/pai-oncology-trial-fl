"""Physical AI integration for oncology clinical trials.

Bridges robotic systems, digital twins, and sensor fusion
with the federated learning platform.
"""

from physical_ai.digital_twin import PatientDigitalTwin, TumorModel
from physical_ai.robotic_integration import (
    RoboticTask,
    SurgicalRobotInterface,
)
from physical_ai.sensor_fusion import ClinicalSensorFusion, SensorReading
from physical_ai.simulation_bridge import SimulationBridge

__all__ = [
    "ClinicalSensorFusion",
    "PatientDigitalTwin",
    "RoboticTask",
    "SensorReading",
    "SimulationBridge",
    "SurgicalRobotInterface",
    "TumorModel",
]
