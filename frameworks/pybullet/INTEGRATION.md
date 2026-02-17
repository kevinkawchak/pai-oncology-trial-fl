# PyBullet Integration Guide

Integration guide for connecting PyBullet to the PAI Oncology Trial FL
platform. PyBullet provides lightweight, dependency-minimal physics
simulation for rapid prototyping, unit testing, and CI/CD validation
of surgical robotics policies across oncology trial sites.

## System Requirements

| Component     | Minimum                          | Recommended            |
|---------------|----------------------------------|------------------------|
| CPU           | 4 cores (x86_64)               | 8+ cores (x86_64)     |
| RAM           | 8 GB                             | 16 GB                  |
| GPU           | Not required                     | Not required           |
| OS            | Ubuntu 20.04+ / macOS / Windows  | Ubuntu 22.04 LTS      |
| Python        | 3.8                              | 3.10 - 3.12           |
| PyBullet      | 3.2.5                            | 3.2.6                  |

PyBullet has no mandatory GPU requirement and runs on all major
operating systems -- the most portable option in the framework stack.

## Installation

```bash
pip install pybullet==3.2.6
python -c "import pybullet; print(pybullet.getPhysicsEngineParameters())"
```

For headless environments (CI servers, Docker containers):

```bash
python -c "
import pybullet as p
cid = p.connect(p.DIRECT)  # no display server needed
print(f'Connected: physics client {cid}')
p.disconnect()
"
```

```bash
pip install -e /path/to/pai-oncology-trial-fl
```

Reference: [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit)

## Lightweight Simulation Setup

### Initializing the Physics Environment

```python
import pybullet as p
import pybullet_data
import numpy as np

physics_client = p.connect(p.DIRECT)  # p.GUI for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)
p.setPhysicsEngineParameter(
    numSolverIterations=50, numSubSteps=4,
    contactBreakingThreshold=0.001, enableConeFriction=True,
)

# Load operating table and surgical arm
table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0], useFixedBase=True)
surgical_arm_id = p.loadURDF(
    "surgical_arm.urdf", basePosition=[0, -0.3, 0.7], useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
)
```

### Tumor Phantom with Soft-Body Approximation

PyBullet lacks native FEM deformable bodies. Approximate tissue
deformation with tuned contact dynamics:

```python
tumor_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.015)
tumor_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.015,
                                 rgbaColor=[0.85, 0.55, 0.55, 1.0])
tumor_id = p.createMultiBody(
    baseMass=0.02, baseCollisionShapeIndex=tumor_col,
    baseVisualShapeIndex=tumor_vis, basePosition=[0.15, 0.0, 0.74],
)
p.changeDynamics(
    tumor_id, -1,
    lateralFriction=0.6, spinningFriction=0.01, rollingFriction=0.001,
    restitution=0.05,          # Low bounce — tissue-like
    contactStiffness=3000.0,   # Pa-equivalent for liver parenchyma
    contactDamping=50.0,
)
```

### Simulation Loop with Telemetry Collection

```python
def run_procedure_simulation(arm_id, target_pos, num_steps=2400):
    """Run simulated surgical procedure and collect telemetry."""
    telemetry = []
    ee_link = p.getNumJoints(arm_id) - 1
    for step in range(num_steps):
        joint_poses = p.calculateInverseKinematics(
            arm_id, ee_link, target_pos,
            maxNumIterations=100, residualThreshold=1e-5,
        )
        for i, pos in enumerate(joint_poses):
            p.setJointMotorControl2(
                arm_id, i, p.POSITION_CONTROL,
                targetPosition=pos, force=5.0, maxVelocity=2.0,
            )
        p.stepSimulation()
        if step % 24 == 0:  # 10 Hz telemetry
            ee_state = p.getLinkState(arm_id, ee_link)
            ee_pos = np.array(ee_state[0])
            contacts = p.getContactPoints(bodyA=arm_id)
            total_force = sum(c[9] for c in contacts)
            telemetry.append({
                "step": step, "position": ee_pos.tolist(),
                "force_n": round(float(total_force), 3),
                "velocity_mm_s": round(float(np.linalg.norm(ee_pos - target_pos)) * 100, 3),
            })
    return telemetry
```

## Connecting to the Federated Learning Platform

```python
from physical_ai.simulation_bridge import SimulationBridge, RobotModel, ModelFormat
from physical_ai.robotic_integration import SurgicalRobotInterface, RobotType
from physical_ai.framework_detection import FrameworkDetector
import numpy as np

# 1. Detect frameworks
detector = FrameworkDetector()
frameworks = detector.detect()
assert frameworks["pybullet"].available, "PyBullet not found"
pipeline = detector.recommend_pipeline()
# PyBullet preferred for "rapid-prototyping" and "unit-testing"

# 2. Register the URDF model
bridge = SimulationBridge()
surgical_arm = RobotModel(
    name="surgical_arm_pybullet",
    source_format=ModelFormat.URDF,
    num_joints=7, num_links=8, mass_kg=2.3,
    joint_limits=[
        (-3.14, 3.14), (-2.0, 2.0), (-2.0, 2.0), (-3.14, 3.14),
        (-1.57, 1.57), (-1.57, 1.57), (0.0, 0.04),
    ],
    metadata={"framework": "pybullet", "version": "3.2.6"},
)
bridge.register_model(surgical_arm)

# 3. Collect de-identified telemetry for FL training
robot = SurgicalRobotInterface(
    robot_id="site_delta_pybullet_01", robot_type=RobotType.ROBOTIC_ARM,
    capabilities=["biopsy", "resection", "specimen_handling"],
)
robot.connect()
task = robot.plan_procedure(
    task_type="specimen_handling",
    target_location=np.array([20.0, 5.0, 2.0]),
    tumor_volume=0.5, safety_margin_mm=5.0,
)
completed = robot.execute_task(task, seed=99)
features = robot.get_telemetry_features(completed)
```

## Framework Comparison for Oncology Use Cases

| Criterion                  | PyBullet        | MuJoCo 3.x     | Isaac Lab 1.x     | Gazebo Harmonic  |
|----------------------------|-----------------|-----------------|--------------------|------------------|
| **Setup complexity**       | Minimal         | Low             | High               | Moderate         |
| **GPU required**           | No              | No (MJX: yes)   | Yes                | Optional         |
| **Contact fidelity**       | Moderate        | High            | High               | High             |
| **Soft-tissue support**    | Approximate     | Native          | Native (PhysX 5)   | Via plugins      |
| **Batched GPU envs**       | No              | Yes (MJX)       | Yes (thousands)    | No               |
| **ROS 2 integration**      | Manual          | Manual          | Via OmniGraph      | Native           |
| **Headless CI/CD**         | Native (DIRECT) | Native (EGL)    | Requires GPU node  | Requires X/EGL   |
| **Best oncology use case** | Prototyping     | Validation      | Large-scale RL     | HW-in-the-loop   |

### When to Use PyBullet

- **Rapid prototyping** -- test surgical task definitions and reward
  functions before committing to a heavier simulator.
- **CI/CD pipelines** -- DIRECT mode requires no GPU, starts in
  milliseconds, ideal for automated regression testing.
- **Resource-constrained sites** -- trial sites without GPU
  workstations can participate in federated training.

### When NOT to Use PyBullet

- **Final policy validation** -- contact dynamics insufficient for
  clinical-grade certification. Use MuJoCo or Isaac Lab.
- **Large-scale RL training** -- no GPU-batched environments.
- **Sensor-realistic rendering** -- use Gazebo or Isaac Sim for
  vision-based policy transfer.

## Evaluating Against Clinical Thresholds

```python
from physical_ai.surgical_tasks import STANDARD_TASKS, ProcedureType, SurgicalTaskEvaluator

task_def = STANDARD_TASKS[ProcedureType.SPECIMEN_HANDLING]
evaluator = SurgicalTaskEvaluator(task_def)
results = evaluator.evaluate(
    position_errors_mm=[1.2, 2.0, 3.5, 1.8, 2.4],
    forces_n=[0.5, 0.8, 1.2, 0.6, 0.9],
    collisions=0, total_attempts=150, procedure_time_s=45.0,
)
# Thresholds: max_pos_error=5.0mm, max_force=2.0N, success>=0.99
print(f"Clinical ready: {results['clinical_ready']}")
```

## Troubleshooting

| Symptom                             | Cause                              | Fix                                               |
|-------------------------------------|------------------------------------|----------------------------------------------------|
| `pybullet build from source failed` | Missing build dependencies         | `sudo apt install build-essential cmake`           |
| No visual output in GUI mode        | Display not available              | Use `p.connect(p.DIRECT)` for headless operation   |
| URDF load fails with mesh errors    | Mesh paths not found               | Set `p.setAdditionalSearchPath()` or use abs paths |
| IK solver returns NaN               | Target outside reachable workspace | Clamp target to workspace bounds from task def     |
| Unstable contacts with small objects| Solver iterations too low          | Increase `numSolverIterations` to 100+             |

## Source Citations

1. PyBullet Quickstart Guide:
   https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit
2. PyBullet GitHub Repository:
   https://github.com/bulletphysics/bullet3
3. Bullet Physics SDK: https://pybullet.org/wordpress/
4. Bullet3 Real-Time Simulation: https://bulletphysics.org/
5. PyBullet Gymnasium Integration:
   https://gymnasium.farama.org/environments/third_party_environments/
6. Erwin Coumans, Yunfei Bai. "PyBullet, a Python module for physics
   simulation for games, robotics and machine learning." http://pybullet.org
