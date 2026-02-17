# Challenges in Unifying Surgical Robotics for Oncology Trials

## Overview

Surgical robotics platforms used in oncology span a wide range of
architectures, from cable-driven systems (Intuitive da Vinci) to
direct-drive arms (Medtronic Hugo), to research platforms (Franka
Emika Panda, Universal Robots UR5e). Each platform has a proprietary
control interface, different kinematic structures, and distinct
safety architectures. Federated learning across institutions using
different robots requires unification at the control, simulation,
and data layers.

---

## 1. Kinematic and Dynamic Heterogeneity

### 1.1 Varying Degrees of Freedom

Clinical surgical robots range from 4 DOF (some needle insertion
robots) to 7+ DOF (da Vinci instruments with wrist articulation).
Policies trained on a 7-DOF system cannot directly execute on a
6-DOF system without kinematic remapping or task-space control
abstraction.

### 1.2 Workspace Differences

Each robot has a different reachable workspace, which affects
which surgical approaches are feasible. A policy trained in one
workspace may command unreachable configurations on another robot.

### 1.3 Cable-Drive vs. Direct-Drive Dynamics

Cable-driven systems (da Vinci) have nonlinear transmission
dynamics (backlash, hysteresis, cable stretch) that direct-drive
systems (Franka) do not exhibit. Policies must be robust to these
differences or explicitly model them.

### 1.4 Instrument-Specific Kinematics

Surgical instruments have their own kinematics beyond the robot arm.
Wristed instruments, staplers, and energy devices each add different
joint types and constraints at the end-effector.

---

## 2. Control Interface Fragmentation

### 2.1 Proprietary APIs

- **Intuitive Surgical**: da Vinci Research Kit (dVRK) provides
  open-source ROS interfaces, but production da Vinci Xi/SP uses
  a closed API.
- **Medtronic Hugo**: Limited research API access.
- **Franka Emika**: libfranka provides real-time control at 1 kHz.
- **Universal Robots**: URScript + RTDE for real-time data exchange.

These APIs differ in:
- Command rate (1 kHz for Franka, 125 Hz for UR, variable for dVRK)
- Command modes (position, velocity, torque, impedance)
- Safety layer integration
- Real-time vs. non-real-time execution

### 2.2 Coordinate Frame Conventions

Each manufacturer defines its own base frame, tool frame, and
joint zero positions. Converting between these is error-prone
and must be validated per-configuration.

### 2.3 Safety System Integration

Surgical robots have hardware safety systems (force limiters,
collision detection, emergency stops) that are implemented
differently on each platform. The unification layer must respect
and not circumvent these safety systems.

---

## 3. Sensor Data Standardization

### 3.1 Force/Torque Sensing

- da Vinci: Estimated from motor currents (no direct F/T sensor
  in production systems)
- Franka: Integrated joint torque sensors + end-effector F/T
- UR5e: Wrist-mounted F/T sensor
- Research platforms: External F/T sensors (ATI, OnRobot)

Sensor characteristics (noise floor, bandwidth, drift) differ
significantly. Policies trained on one sensor type may perform
poorly with another.

### 3.2 Vision Systems

Surgical vision spans:
- Stereo endoscopes (da Vinci)
- Monocular laparoscopes
- External tracking systems (Polaris, OptiTrack)
- Intra-operative ultrasound
- Fluorescence imaging (ICG, Firefly)

Unifying visual observations across these modalities is a major
perception challenge.

### 3.3 Telemetry Data Formats

Each robot produces telemetry in a different format, at different
rates, with different fields. Standardizing telemetry for
federated learning requires a canonical data schema.

---

## 4. Sim-to-Real Transfer for Surgical Procedures

### 4.1 Tissue Interaction Modeling

Oncology procedures involve cutting, grasping, and retracting
living tissue. Simulation fidelity for these interactions is
limited by:
- Inadequate tissue material models (non-linear, anisotropic,
  viscoelastic behavior)
- Lack of bleeding and fluid simulation in real-time
- Patient-specific anatomy variation

### 4.2 Tool-Tissue Interaction

The physics of electrocautery, ultrasonic dissection, and laser
ablation are poorly modeled in current robotics simulators. These
are common oncology procedures.

### 4.3 Domain Randomization Scope

Sim-to-real transfer typically uses domain randomization, but the
relevant randomization parameters for surgical environments
(tissue properties, lighting in the body cavity, smoke from
cautery) are not well characterized.

---

## 5. Regulatory and Safety Challenges

### 5.1 Software as a Medical Device (SaMD)

Any control software that influences surgical robot behavior is
likely classified as a medical device under FDA, EU MDR, and
equivalent regulations. This imposes:
- IEC 62304 software lifecycle requirements
- ISO 14971 risk management
- IEC 60601 electrical safety (for connected systems)
- FDA 510(k) or De Novo classification

### 5.2 Autonomous vs. Teleoperated Operation

Current FDA-cleared surgical robots operate under human
teleoperation. Autonomous or semi-autonomous operation requires
new regulatory pathways with additional safety evidence.

### 5.3 Liability and Responsibility

When a federated policy trained across multiple institutions
controls a surgical robot, liability in case of adverse events
becomes complex. The legal framework for multi-institutional
AI-driven surgical systems is still evolving.

---

## 6. Clinical Workflow Integration

### 6.1 Operating Room Constraints

The OR has strict constraints on:
- Sterile field boundaries
- Equipment placement and cable routing
- Network connectivity (often limited)
- Power and UPS requirements

These constraints vary by institution and affect how the
unification layer is deployed.

### 6.2 Surgeon Acceptance

Surgeons must trust the AI system. Different surgical cultures
and individual preferences affect how autonomous assistance is
received. The unification layer must support configurable
autonomy levels.

### 6.3 Training and Credentialing

Each institution has its own training requirements for new
surgical technology. A unified system must accommodate different
training curricula and proficiency assessment criteria.

---

## Mitigation Strategies

1. **Task-space abstraction**: Define policies in Cartesian task
   space rather than joint space. This enables cross-robot
   transfer via inverse kinematics.

2. **Canonical telemetry schema**: Define a standardized telemetry
   format that all robot adapters emit, enabling federated learning
   on a common observation space.

3. **Safety envelope enforcement**: Implement a robot-independent
   safety layer that enforces force/velocity limits in task space,
   with robot-specific kinematic mapping underneath.

4. **Progressive autonomy**: Start with post-operative analysis
   and pre-operative planning (lower regulatory burden), then
   progress to intra-operative assistance as evidence accumulates.

5. **Hardware-in-the-loop validation**: Require all federated
   policies to pass HIL testing on the target robot before any
   clinical use.
