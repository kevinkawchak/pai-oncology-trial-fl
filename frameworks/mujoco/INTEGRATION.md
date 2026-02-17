# MuJoCo Integration Guide

Integration guide for connecting MuJoCo (Multi-Joint dynamics with
Contact) to the PAI Oncology Trial FL platform for high-fidelity
physics validation of surgical robotics policies in oncology trials.

## System Requirements

| Component     | Minimum                | Recommended             |
|---------------|------------------------|-------------------------|
| CPU           | 8 cores (x86_64)      | 16+ cores (x86_64)     |
| RAM           | 16 GB                  | 32 GB                   |
| GPU           | Not required           | NVIDIA GPU (for MJX)    |
| CUDA          | N/A (CPU-only MuJoCo)  | 12.1+ (MJX with JAX)   |
| OS            | Ubuntu 20.04+ / macOS  | Ubuntu 22.04 LTS        |
| Python        | 3.9                    | 3.10 - 3.12             |
| MuJoCo        | 3.1.0                  | 3.2.6                   |

MuJoCo 3.x is maintained by Google DeepMind. The legacy OpenAI
`mujoco-py` wrapper is not used by this platform.

## Installation

```bash
pip install mujoco==3.2.6
python -c "import mujoco; print(mujoco.__version__)"  # verify
```

For GPU-accelerated batched simulation via MJX (optional):

```bash
pip install mujoco-mjx==3.2.6
pip install jax[cuda12]==0.4.35
```

Reference: [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
| [MJX Documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)

Install the PAI platform:

```bash
pip install -e /path/to/pai-oncology-trial-fl
```

## Version-Specific Configuration — MuJoCo 3.x MJCF

The platform's `SimulationBridge` handles URDF-to-MJCF conversion, but
native MJCF models offer finer control over contact and tendon dynamics
critical for surgical simulation.

```xml
<!-- surgical_arm_oncology.xml — 7-DOF surgical instrument -->
<mujoco model="surgical_arm_oncology">
  <compiler angle="radian" meshdir="meshes/" autolimits="true"/>
  <option timestep="0.002" iterations="50" solver="Newton"
          integrator="implicitfast" cone="elliptic">
    <flag warmstart="enable" multiccd="enable"/>
  </option>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom condim="4" friction="0.6 0.005 0.0001"
          solimp="0.95 0.99 0.001" solref="0.02 1.0"/>
    <motor ctrlrange="-1.0 1.0" ctrllimited="true"/>
  </default>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <body name="table" pos="0 0 0.7">
      <geom type="box" size="0.4 0.3 0.02" rgba="0.7 0.7 0.7 1"/>
      <body name="tumor_phantom" pos="0.15 0.0 0.04">
        <geom name="tumor" type="ellipsoid" size="0.015 0.012 0.01"
              rgba="0.85 0.55 0.55 1" mass="0.02"
              solref="0.01 0.5" solimp="0.9 0.95 0.001"/>
      </body>
    </body>
    <body name="base_link" pos="0 -0.3 0.7">
      <geom type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="joint_1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" size="0.02 0.1" rgba="0.5 0.5 0.5 1"/>
        <!-- Continue chain through joint_7 / gripper -->
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="act_1" joint="joint_1" gear="50"/>
  </actuator>
  <sensor>
    <force name="ee_force" site="end_effector"/>
    <torque name="ee_torque" site="end_effector"/>
    <touch name="gripper_touch" site="gripper_pad"/>
  </sensor>
</mujoco>
```

Reference: [MJCF XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)

### Simulation Loop

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("surgical_arm_oncology.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

for step in range(5000):  # 10 seconds at dt=0.002
    data.ctrl[:] = np.zeros(model.nu)  # replace with policy output
    mujoco.mj_step(model, data)
    ee_force = data.sensor("ee_force").data.copy()
    ee_pos = data.site("end_effector").xpos.copy()
```

## Physics Parameter Tuning for Medical Robotics

### Contact Parameter Reference

| Parameter       | Biological Tissue Range | Recommended Default | Units   |
|-----------------|------------------------|---------------------|---------|
| `solref[0]`     | 0.005 - 0.02          | 0.01                | s       |
| `solref[1]`     | 0.3 - 1.0             | 0.5                 | -       |
| `solimp[0]`     | 0.9 - 0.98            | 0.95                | -       |
| `solimp[1]`     | 0.95 - 0.999          | 0.99                | -       |
| `friction[0]`   | 0.3 - 0.8             | 0.6                 | -       |

### Runtime Domain Randomization

```python
def randomize_tissue_contact(model, rng):
    """Randomize tissue contact for sim-to-real transfer.
    Ranges calibrated to hepatic/renal parenchyma biomechanics."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tumor")
    model.geom_solref[gid, 0] = rng.uniform(0.005, 0.02)
    model.geom_solref[gid, 1] = rng.uniform(0.3, 1.0)
    model.geom_friction[gid, 0] = rng.uniform(0.3, 0.8)
```

Reference: [MuJoCo Contact Model](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)

## Connecting to the Federated Learning Platform

```python
from physical_ai.simulation_bridge import SimulationBridge, RobotModel, ModelFormat
from physical_ai.robotic_integration import SurgicalRobotInterface, RobotType
from physical_ai.framework_detection import FrameworkDetector
import numpy as np

# 1. Detect MuJoCo availability
detector = FrameworkDetector()
frameworks = detector.detect()
assert frameworks["mujoco"].available, "MuJoCo not found"
pipeline = detector.recommend_pipeline()
# MuJoCo preferred for "validation" and "physics-accuracy"

# 2. Register the MJCF model
bridge = SimulationBridge()
surgical_arm = RobotModel(
    name="surgical_arm_mujoco",
    source_format=ModelFormat.MJCF,
    num_joints=7, num_links=8, mass_kg=2.3,
    joint_limits=[
        (-3.14, 3.14), (-2.0, 2.0), (-2.0, 2.0), (-3.14, 3.14),
        (-1.57, 1.57), (-1.57, 1.57), (0.0, 0.04),
    ],
    metadata={"framework": "mujoco", "version": "3.2.6"},
)
bridge.register_model(surgical_arm)

# 3. Cross-validate with Isaac Lab model via SimulationBridge
isaac_model = bridge.get_model("da_vinci_instrument")
if isaac_model:
    converted = bridge.convert(isaac_model, ModelFormat.MJCF)
    validation = bridge.validate_conversion(isaac_model, converted)
    assert all(validation.values()), f"Conversion mismatch: {validation}"

# 4. Collect telemetry for FL training
robot = SurgicalRobotInterface(
    robot_id="site_beta_mujoco_01", robot_type=RobotType.DA_VINCI,
)
robot.connect()
task = robot.plan_procedure(
    task_type="biopsy",
    target_location=np.array([15.0, 0.0, 4.0]),
    tumor_volume=1.8, safety_margin_mm=3.0,
)
completed = robot.execute_task(task, seed=7)
features = robot.get_telemetry_features(completed)
```

## Evaluating Against Clinical Thresholds

```python
from physical_ai.surgical_tasks import STANDARD_TASKS, ProcedureType, SurgicalTaskEvaluator

task_def = STANDARD_TASKS[ProcedureType.NEEDLE_BIOPSY]
evaluator = SurgicalTaskEvaluator(task_def)
results = evaluator.evaluate(
    position_errors_mm=[0.4, 0.7, 1.1, 0.3, 0.5],
    forces_n=[1.0, 1.8, 2.5, 0.9, 1.4],
    collisions=0, total_attempts=200, procedure_time_s=85.0,
)
# Thresholds: max_position_error=1.5mm, max_force=3.0N, success>=0.97
print(f"Clinical ready: {results['clinical_ready']}")
```

## Troubleshooting

| Symptom                              | Cause                            | Fix                                              |
|--------------------------------------|----------------------------------|--------------------------------------------------|
| `FatalError: gladLoadGL failed`      | No display / headless server     | Set `os.environ["MUJOCO_GL"] = "egl"` or `osmesa`|
| Diverging simulation                 | Timestep too large for contacts  | Reduce `timestep` to 0.001 or increase iterations|
| Mesh penetration on soft contact     | `solimp` too permissive          | Tighten `solimp[0]` toward 0.98                  |
| MJX `TracerArrayConversionError`     | Non-JAX ops in traced code       | Replace NumPy calls with `jax.numpy`             |
| Sensor data all zeros                | Site not attached to body        | Verify `site` attribute in `<sensor>` elements   |

## Source Citations

1. MuJoCo Documentation (DeepMind):
   https://mujoco.readthedocs.io/en/stable/
2. MuJoCo Python Bindings:
   https://mujoco.readthedocs.io/en/stable/python.html
3. MJCF XML Reference:
   https://mujoco.readthedocs.io/en/stable/XMLreference.html
4. MuJoCo Contact Model:
   https://mujoco.readthedocs.io/en/stable/computation/index.html#contact
5. MJX GPU-Accelerated MuJoCo:
   https://mujoco.readthedocs.io/en/stable/mjx.html
6. MuJoCo GitHub Repository:
   https://github.com/google-deepmind/mujoco
7. MuJoCo Changelog:
   https://mujoco.readthedocs.io/en/stable/changelog.html
