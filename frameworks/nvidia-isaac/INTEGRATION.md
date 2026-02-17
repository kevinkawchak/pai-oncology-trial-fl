# NVIDIA Isaac Sim / Isaac Lab Integration Guide

Integration guide for connecting NVIDIA Isaac Sim and Isaac Lab to the
PAI Oncology Trial FL platform for GPU-accelerated surgical robotics
simulation in federated oncology clinical trials.

## System Requirements

| Component         | Minimum                     | Recommended                   |
|-------------------|-----------------------------|-------------------------------|
| GPU               | NVIDIA RTX 3070 (8 GB VRAM) | NVIDIA RTX 4090 (24 GB VRAM) |
| GPU Driver        | 535.129.03+                 | 550.54.15+                    |
| CUDA              | 12.1                        | 12.4                          |
| RAM               | 32 GB                       | 64 GB                         |
| OS                | Ubuntu 22.04 LTS            | Ubuntu 22.04 LTS              |
| Python            | 3.10                        | 3.10                          |
| Isaac Sim         | 4.2.0                       | 4.5.0                         |
| Isaac Lab         | 1.2.0                       | 1.4.0                         |
| Nucleus Server    | 2023.2.5+                   | 2024.1+                       |

Isaac Lab requires Isaac Sim as its underlying simulator. Refer to the
[Isaac Lab compatibility matrix](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
for validated version pairings.

## Installation

### 1. Install Isaac Sim via Omniverse Launcher

```bash
# After installing via the Launcher, verify the installation
~/.local/share/ov/pkg/isaac-sim-4.5.0/isaac-sim.sh --help
```

### 2. Install Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
git checkout v1.4.0
ln -s ~/.local/share/ov/pkg/isaac-sim-4.5.0 _isaac_sim
./isaaclab.sh --install
```

Reference: [Isaac Lab Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

### 3. Install PAI Oncology Trial FL

```bash
pip install -e /path/to/pai-oncology-trial-fl
```

## Version-Specific Configuration — Isaac Sim 4.x / Isaac Lab 1.x

Isaac Lab 1.x provides Gymnasium-compatible RL environment wrappers.
The oncology platform hooks into these to collect telemetry and feed
federated model parameters back into the simulation.

```python
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg

SURGICAL_ARM_CFG = ArticulationCfg(
    prim_path="/World/SurgicalArm",
    spawn=sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/Library/Robots/surgical_arm.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint_.*": 0.0}),
)

class OncologyResectionSceneCfg(InteractiveSceneCfg):
    """Scene with surgical arm, tumor phantom, and OR table."""
    ground = AssetBaseCfg(
        prim_path="/World/OperatingTable",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/Library/Props/or_table.usd"
        ),
    )
    robot = SURGICAL_ARM_CFG

class OncologyResectionEnvCfg(ManagerBasedRLEnvCfg):
    """RL environment for tumor resection training."""
    scene = OncologyResectionSceneCfg(num_envs=1024, env_spacing=2.5)
    sim = sim_utils.SimulationCfg(dt=1 / 120, render_interval=2)
```

Reference: [Isaac Lab RL Environments](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/index.html)

### Soft-Tissue Physics (PhysX 5)

Oncology procedures require deformable body simulation for tumor and
organ interaction:

```python
from omni.isaac.lab.sim import DeformableBodyPropertiesCfg

TUMOR_PHANTOM_PROPS = DeformableBodyPropertiesCfg(
    deformable_enabled=True,
    kinematic_enabled=False,
    youngs_modulus=3000.0,     # Pa — liver parenchyma range [1-10 kPa]
    poissons_ratio=0.45,       # Near-incompressible biological tissue
    damping_scale=0.01,
    dynamic_friction=0.4,
    vertex_velocity_damping=0.005,
)
```

Reference: [PhysX Deformable Bodies](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Deformables.html)

## Connecting to the Federated Learning Platform

The `SimulationBridge` and `SurgicalRobotInterface` in `physical_ai/`
provide the connection layer. Isaac Lab environments produce telemetry
that is de-identified and contributed to federated training rounds.

```python
from physical_ai.simulation_bridge import SimulationBridge, RobotModel, ModelFormat
from physical_ai.robotic_integration import SurgicalRobotInterface, RobotType
from physical_ai.framework_detection import FrameworkDetector
import numpy as np

# 1. Detect frameworks and confirm Isaac Lab availability
detector = FrameworkDetector()
frameworks = detector.detect()
assert frameworks["isaac_lab"].available, "Isaac Lab not found"
pipeline = detector.recommend_pipeline()
# Expected: {"training": "isaac_lab", "validation": "mujoco", ...}

# 2. Register the USD robot model with the simulation bridge
bridge = SimulationBridge()
surgical_arm = RobotModel(
    name="da_vinci_instrument",
    source_format=ModelFormat.USD,
    num_joints=7, num_links=8, mass_kg=2.3,
    joint_limits=[
        (-3.14, 3.14), (-2.0, 2.0), (-2.0, 2.0), (-3.14, 3.14),
        (-1.57, 1.57), (-1.57, 1.57), (0.0, 0.04),
    ],
    metadata={"framework": "isaac_lab", "version": "1.4.0"},
)
bridge.register_model(surgical_arm)

# 3. Connect the robotic interface for telemetry collection
robot = SurgicalRobotInterface(
    robot_id="site_alpha_isaac_01",
    robot_type=RobotType.DA_VINCI,
    capabilities=["biopsy", "resection", "lymph_node_dissection"],
)
robot.connect()

# 4. Plan and simulate a procedure — telemetry feeds FL training
task = robot.plan_procedure(
    task_type="resection",
    target_location=np.array([10.0, -5.0, 3.0]),
    tumor_volume=4.2, safety_margin_mm=5.0,
)
completed_task = robot.execute_task(task, seed=42)

# 5. Extract de-identified features for federated model update
features = robot.get_telemetry_features(completed_task)
print(f"Feature vector ({features.shape}): {features}")
```

## Oncology Surgical Robotics Evaluation

Evaluate Isaac Lab policies against clinical thresholds from
`physical_ai/surgical_tasks.py`:

```python
from physical_ai.surgical_tasks import STANDARD_TASKS, ProcedureType, SurgicalTaskEvaluator

task_def = STANDARD_TASKS[ProcedureType.TISSUE_RESECTION]
evaluator = SurgicalTaskEvaluator(task_def)
results = evaluator.evaluate(
    position_errors_mm=[0.8, 1.2, 1.5, 0.9, 1.1],
    forces_n=[2.1, 3.5, 4.8, 2.0, 3.0],
    collisions=0, total_attempts=100, procedure_time_s=450.0,
)
print(f"Clinical ready: {results['clinical_ready']}")
```

## Multi-Site Federated Aggregation

Each clinical trial site runs Isaac Lab locally. Model weights are
aggregated by the coordinator without exchanging raw telemetry:

```python
from federated.coordinator import FederatedCoordinator

coordinator = FederatedCoordinator(
    strategy="fedavg", num_rounds=50, min_clients=3,
)
```

## Troubleshooting

| Symptom                          | Cause                                | Fix                                                  |
|----------------------------------|--------------------------------------|------------------------------------------------------|
| `ModuleNotFoundError: omni`      | Isaac Sim Python env not activated   | Source `setup_python_env.sh` from Isaac Sim install   |
| GPU out-of-memory at 1024 envs   | Insufficient VRAM                    | Reduce `num_envs` or use RTX 4090 / A100             |
| USD asset fails to load          | Nucleus server not running           | Start Nucleus via Omniverse Launcher                  |
| Physics instability in soft body | Time step too large                  | Reduce `dt` to `1/240` and increase solver iterations |
| Telemetry NaN values             | Simulation divergence                | Check joint limits and damping parameters             |

## Source Citations

1. NVIDIA Isaac Sim Documentation:
   https://docs.omniverse.nvidia.com/isaacsim/latest/index.html
2. NVIDIA Isaac Lab Documentation:
   https://isaac-sim.github.io/IsaacLab/main/index.html
3. Isaac Lab Installation Guide:
   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
4. Isaac Lab RL Environment Tutorials:
   https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/index.html
5. PhysX 5 SDK — Deformable Bodies:
   https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Deformables.html
6. Omniverse Nucleus Documentation:
   https://docs.omniverse.nvidia.com/nucleus/latest/index.html
7. NVIDIA Isaac Sim System Requirements:
   https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html
