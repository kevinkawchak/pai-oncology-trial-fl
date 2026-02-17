# Challenges in Unifying Simulation Physics for Oncology Robotics

## Overview

Oncology clinical trials that incorporate surgical robotics and physical AI
require high-fidelity physics simulation for pre-operative planning, intra-
operative guidance, and post-operative analysis. Today, no single simulation
engine satisfies every requirement. Isaac Sim excels at GPU-accelerated
parallel environments and photorealistic rendering. MuJoCo offers fast,
stable contact dynamics suitable for policy optimization. Gazebo provides
tight ROS integration for hardware-in-the-loop testing. PyBullet enables
rapid prototyping with minimal dependencies.

Unifying these engines behind a common interface introduces significant
technical, organizational, and regulatory challenges.

---

## 1. Physics Engine Divergence

### 1.1 Contact Model Incompatibility

Each engine implements a fundamentally different contact model:

- **MuJoCo** uses a soft-contact convex optimization solver that represents
  contacts as complementarity constraints with impedance parameters.
- **Isaac Sim (PhysX 5)** uses a GPU-accelerated temporal Gauss-Seidel (TGS)
  solver with speculative contacts and rigid-body penetration recovery.
- **Gazebo (DART / Bullet)** supports multiple solver backends with different
  stability/performance trade-offs.
- **PyBullet** uses a sequential impulse solver with split impulse for
  penetration recovery.

The practical consequence is that identical joint torques applied to
geometrically identical robots produce measurably different trajectories
across engines. For oncology procedures where sub-millimeter accuracy
matters (e.g., stereotactic needle biopsy), this drift is clinically
significant.

### 1.2 Integrator Time-Step Sensitivity

MuJoCo defaults to a semi-implicit Euler integrator at 2 ms. Isaac Sim
often runs at 1/60 s with PhysX sub-steps. Policies trained at one
time-step frequently fail to transfer to another without re-tuning gains.

### 1.3 Friction and Damping Parameterization

Coulomb friction, viscous damping, and joint stiffness are parameterized
differently. A "friction coefficient of 0.5" in MuJoCo does not produce the
same behavior as "static friction 0.5 + dynamic friction 0.4" in PhysX.

---

## 2. Model Format Fragmentation

### 2.1 URDF Limitations

URDF is the de facto interchange format, but it lacks support for:

- Closed-loop kinematic chains (common in parallel-jaw grippers and
  some surgical instruments)
- Tendon-driven actuators (used in cable-driven surgical robots)
- Muscle and soft-tissue models (needed for organ deformation simulation)
- Site / sensor placement semantics

### 2.2 MJCF vs. USD vs. SDF

Each engine has a native format with features absent from URDF:

| Feature | MJCF | USD | SDF |
|---------|------|-----|-----|
| Tendons | Yes | No | No |
| Deformable bodies | Yes (flex) | Yes (PhysX) | Limited |
| Material shaders | Basic | Full PBR | Basic |
| Composite joints | Yes | Via articulation | Limited |
| Sensor definitions | Built-in | Schema extension | Built-in |

Lossless round-trip conversion is not possible today.

### 2.3 Mesh and Collision Geometry

Convex decomposition algorithms differ across engines. Isaac Sim uses
V-HACD via PhysX, MuJoCo uses its own convex hull pipeline, and Gazebo
relies on the DART / FCL collision library. The same STL mesh can yield
different collision hulls, changing contact behavior.

---

## 3. Tissue and Organ Simulation Fidelity

### 3.1 Soft-Body Physics

Oncology procedures interact with deformable tissue. Liver resection,
lung biopsy, and brain tumor removal all require accurate soft-body
simulation. Current capabilities vary:

- **Isaac Sim**: PhysX 5 FEM-based deformables with GPU acceleration,
  but limited tissue material models.
- **MuJoCo 3.x**: Flex bodies with particle-based deformation, suitable
  for simple tissues but not validated against clinical data.
- **SOFA Framework**: Gold standard for surgical simulation but has no
  native federated learning integration.

### 3.2 Fluid and Bleeding Simulation

Intra-operative bleeding affects visibility, tool grip, and tissue
properties. No mainstream robotics simulator handles fluid simulation
at clinically relevant fidelity within real-time constraints.

### 3.3 Patient-Specific Anatomy

Digital twins require patient-specific meshes derived from CT/MRI.
Converting DICOM volumetric data to simulation-ready meshes is a
multi-step pipeline (segmentation, surface extraction, decimation,
material assignment) that is engine-specific.

---

## 4. Sim-to-Real Transfer for Clinical Safety

### 4.1 Domain Gap

Policies trained in simulation must transfer to physical robots operating
on real patients. The domain gap is compounded when policies are trained
in one engine but validated in another:

- Engine-specific artifacts (e.g., contact jitter, penetration recovery
  behavior) can become encoded in the policy.
- Randomization ranges that work in MuJoCo may produce unstable behavior
  in Isaac Sim due to solver differences.

### 4.2 Safety Verification

Regulatory bodies (FDA, notified bodies under MDR) require evidence that
safety-critical software behaves correctly. If training occurs in Engine A
but deployment validation occurs in Engine B, the verification argument
must account for cross-engine fidelity.

### 4.3 Determinism and Reproducibility

GPU-accelerated solvers (PhysX, Warp) are not bitwise deterministic across
hardware generations. This complicates regulatory submissions that require
reproducible test results.

---

## 5. Federated Learning Across Heterogeneous Simulators

### 5.1 Policy Gradient Variance

When different federated learning sites use different simulation engines
for local training, the gradient distributions will differ even for
identical policies and identical task definitions. Naive FedAvg aggregation
amplifies this variance.

### 5.2 Reward Function Alignment

Reward functions that reference engine-specific quantities (e.g., PhysX
contact impulse vs. MuJoCo contact force) produce non-comparable reward
scales. Federated aggregation of policies trained with misaligned rewards
leads to divergence.

### 5.3 State-Space Normalization

Joint angles may be represented differently (quaternion vs. Euler, radians
vs. degrees). Observation vectors that include raw simulator state must be
normalized to a common representation before aggregation.

---

## 6. Organizational and Regulatory Challenges

### 6.1 Licensing Constraints

- Isaac Sim requires an NVIDIA Omniverse license (free for research,
  commercial terms vary).
- MuJoCo is now open-source (Apache 2.0) but was previously proprietary.
- Gazebo is open-source (Apache 2.0).
- PyBullet is open-source (zlib license).

Institutions with strict procurement policies may be unable to adopt
certain engines, forcing the federation to accommodate heterogeneity.

### 6.2 Validation Burden

Each additional engine in the federation multiplies the validation matrix.
If the trial uses N engines and M robot models, the cross-validation
effort scales as O(N x M).

### 6.3 Data Sovereignty

Some institutions cannot share simulation configurations if they encode
proprietary surgical instrument geometry. The bridge must support
federated model aggregation without requiring raw model exchange.

---

## Mitigation Strategies

1. **Canonical state representation**: Define a physics-engine-agnostic
   state vector for all joints, links, and contacts.
2. **Calibration protocol**: Require each site to run a standardized
   calibration scenario and report deviation metrics before joining
   the federation.
3. **Domain randomization budget**: Allocate a per-engine randomization
   budget that accounts for known solver biases.
4. **Dual-engine validation**: Require policies to pass acceptance tests
   on at least two engines before clinical deployment.
5. **Incremental bridging**: Start with the Isaac-MuJoCo pair (highest
   priority for GPU training + fast validation), then extend to Gazebo
   and PyBullet.
