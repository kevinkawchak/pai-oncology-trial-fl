# Reinforcement Learning: Results for Oncology Clinical Trials with Physical AI

> **DISCLAIMER: Illustrative data for demonstration purposes only. Published results cite
> sources. Benchmark results include hardware and seed configuration.**

## Overview

This document presents benchmark results for reinforcement learning in federated oncology
settings with Physical AI integration. RL evaluation in clinical contexts relies heavily on
offline policy evaluation (OPE) and simulation, as online deployment remains limited. All
illustrative benchmarks reflect representative architectures and evaluation methodologies.

---

## 1. Adaptive Treatment Planning Benchmarks

### Illustrative: Chemotherapy Dose Optimization for Metastatic NSCLC

Offline RL agents trained on retrospective treatment trajectories (6 cycles of
platinum-doublet chemotherapy) to optimize cumulative survival reward with toxicity
constraints.

| Algorithm        | OPE C-index | Estimated OS Gain (months) | Toxicity Rate | Sites |
|------------------|-------------|---------------------------|---------------|-------|
| Clinician Policy | 0.612       | Baseline                  | 0.34          | 6     |
| FQI (FedAvg)     | 0.647       | +1.8                      | 0.29          | 6     |
| CQL (FedAvg)     | 0.661       | +2.3                      | 0.27          | 6     |
| CQL (FedProx)    | 0.669       | +2.5                      | 0.26          | 6     |
| Decision Tx (FedAvg) | 0.654   | +2.1                      | 0.28          | 6     |
| CQL (centralized)| 0.678       | +2.8                      | 0.25          | 1     |

**Hardware**: NVIDIA A100 40GB, 2 GPUs per site. **Seeds**: 10 seeds, mean reported.
**OPE method**: Doubly robust estimator with cross-fitting. **Confidence**: 95% CI on OS
gain is +/- 1.1 months for all RL methods.

### Published: Sepsis Treatment Optimization (Transferable Methodology)

| Study                        | Algorithm | Estimated Mortality Reduction | Cohort  |
|------------------------------|-----------|-------------------------------|---------|
| Komorowski et al., Nat Med 2018 | FQI    | 1.8–3.6% absolute            | 17,083  |
| Raghu et al., MLHC 2017     | DDQN      | 2.1% absolute (OPE)          | 11,142  |

**Note**: These sepsis studies demonstrate methodology applicable to oncology treatment
optimization. Direct oncology RL deployment studies remain limited.

---

## 2. Robotic Surgical Control Benchmarks

### Illustrative: Autonomous Needle Insertion for Biopsy (Simulation)

RL agents trained in NVIDIA Isaac Sim for CT-guided percutaneous needle insertion
targeting liver lesions, evaluated on trajectory accuracy and tissue damage.

| Algorithm    | Target Error (mm) | Max Force (N) | Path Length (mm) | Success Rate | Steps |
|--------------|-------------------|---------------|------------------|--------------|-------|
| PPO          | 1.42 ± 0.38      | 2.8           | 98.3             | 0.891        | 5M    |
| SAC          | 1.21 ± 0.31      | 2.5           | 94.7             | 0.923        | 3M    |
| TD3          | 1.33 ± 0.35      | 2.6           | 96.1             | 0.908        | 4M    |
| CPO (safe)   | 1.38 ± 0.34      | 1.9           | 101.2            | 0.878        | 8M    |
| Human expert | 1.85 ± 0.72      | 3.4           | 112.6            | 0.847        | N/A   |

**Hardware**: NVIDIA A100 80GB, Isaac Sim 2023.1. **Seeds**: 5 seeds, mean ± std.
**Safety constraint (CPO)**: Max force < 2.0N at all timesteps.

### Illustrative: Tissue Retraction in Laparoscopic Surgery (Simulation)

| Algorithm       | Success Rate | Tissue Damage Score | Completion Time (s) | Sim Steps |
|-----------------|--------------|--------------------|--------------------|-----------|
| PPO             | 0.73         | 0.42               | 34.2               | 10M       |
| SAC             | 0.81         | 0.38               | 31.7               | 8M        |
| PPO + Demo      | 0.87         | 0.31               | 28.9               | 4M        |
| RLPD            | 0.84         | 0.34               | 30.1               | 3M        |
| Scripted Expert | 0.69         | 0.51               | 42.3               | N/A       |

**Hardware**: NVIDIA A100 80GB, SurRoL simulation framework. **Seeds**: 5 seeds, mean.
**PPO + Demo**: PPO with demonstration-augmented replay buffer.

---

## 3. Radiation Dose Optimization Benchmarks

### Illustrative: Adaptive Fractionation for Head and Neck Cancer

RL agents optimizing per-fraction dose based on weekly CBCT-assessed tumor response,
compared against standard 70Gy/35fx protocol.

| Algorithm     | TCP (%)  | Mean OAR Dose (Gy) | Max Parotid (Gy) | Adapted Fractions |
|---------------|----------|--------------------|--------------------|-------------------|
| Standard 2Gy  | 85.2     | 42.1               | 48.3               | 0/35              |
| DQN           | 87.8     | 39.4               | 44.7               | 8.2/35            |
| PPO           | 88.3     | 38.9               | 43.9               | 9.1/35            |
| CQL (offline) | 87.1     | 40.2               | 45.8               | 7.4/35            |
| SAC           | 88.7     | 38.6               | 43.5               | 9.6/35            |

**Hardware**: NVIDIA A100 40GB, dose calculation via Monte Carlo (GPU-accelerated).
**Seeds**: 10 seeds, mean reported. **Patient cohort**: 200 simulated patients with
heterogeneous tumor geometries.

### Illustrative: IMRT Beam Angle Optimization

| Method            | Plan Quality Score | Optimization Time (min) | OAR Violations |
|-------------------|--------------------|------------------------|----------------|
| Clinical planner  | 0.82               | 45                     | 0.12           |
| Greedy search     | 0.79               | 120                    | 0.08           |
| DQN               | 0.86               | 3.2                    | 0.06           |
| PPO               | 0.88               | 2.8                    | 0.05           |
| PPO + constraint  | 0.85               | 3.5                    | 0.02           |

**Hardware**: NVIDIA V100 32GB. **Seeds**: 5 seeds, mean. **Plan Quality Score**: Composite
metric based on target coverage, conformity index, and OAR sparing (0–1 scale).

---

## 4. Federated RL Convergence Benchmarks

### Illustrative: Federated Policy Learning Across Heterogeneous Treatment Sites

Federated CQL for chemotherapy optimization with varying site heterogeneity.

| Sites | Heterogeneity | FedAvg Rounds | FedProx Rounds | OPE C-index | Variance (σ²) |
|-------|---------------|---------------|----------------|-------------|----------------|
| 3     | Low           | 300           | 250            | 0.668       | 0.0012         |
| 3     | High          | 500           | 380            | 0.649       | 0.0031         |
| 6     | Low           | 350           | 280            | 0.672       | 0.0009         |
| 6     | High          | 650           | 470            | 0.641       | 0.0048         |
| 10    | Low           | 420           | 330            | 0.675       | 0.0008         |
| 10    | High          | 800           | 580            | 0.632       | 0.0062         |

**Heterogeneity**: Low = similar treatment protocols across sites; High = different
standard-of-care guidelines (e.g., NCCN vs ESMO vs institutional).
**Hardware**: NVIDIA A100 40GB per site. **Seeds**: 10 seeds, variance reported.

### Illustrative: Communication Efficiency in Federated RL vs Supervised FL

| Task                       | Paradigm    | Rounds to Convergence | Bytes/Round (MB) |
|----------------------------|-------------|----------------------|------------------|
| Tumor segmentation         | Supervised  | 200                  | 124              |
| Treatment optimization     | RL (CQL)    | 500                  | 89               |
| Treatment optimization     | RL (PPO)    | 750                  | 156              |
| Surgical phase recognition | Supervised  | 180                  | 98               |
| Surgical control           | RL (SAC)    | 1200                 | 201              |

---

## 5. Sim-to-Real Transfer Results

### Illustrative: Needle Insertion Sim-to-Real Gap Analysis

| Training Configuration       | Sim Success Rate | Phantom Success Rate | Gap    |
|------------------------------|------------------|---------------------|--------|
| Sim only (no DR)             | 0.923            | 0.612               | 0.311  |
| Sim + Domain Randomization   | 0.901            | 0.783               | 0.118  |
| Sim + DR + System ID         | 0.897            | 0.834               | 0.063  |
| Sim + DR + Real fine-tune (10)| 0.889           | 0.871               | 0.018  |
| Sim + DR + Real fine-tune (50)| 0.891           | 0.892               | -0.001 |

**Hardware**: NVIDIA Jetson Orin (inference), A100 (training). **Seeds**: 3 seeds, mean.
**DR**: Domain randomization of tissue stiffness (±30%), friction (±20%), lighting (±40%).
**Phantom**: Silicone liver phantom with embedded targets.

---

## 6. Safety Constraint Satisfaction

### Illustrative: Constraint Violation Rates During Training

| Algorithm         | Violations per 1000 Episodes (early) | Violations (converged) | Return |
|-------------------|--------------------------------------|----------------------|--------|
| PPO (unconstrained)| N/A                                | N/A                  | 847    |
| PPO + penalty     | 142                                  | 23                   | 791    |
| CPO               | 87                                   | 8                    | 768    |
| RCPO              | 94                                   | 11                   | 774    |
| CRL (recovery RL) | 31                                   | 3                    | 756    |
| Safe Layer        | 0                                    | 0                    | 712    |

**Context**: Robotic needle insertion, constraint = max force < 2.0N.
**Hardware**: NVIDIA A100 80GB. **Seeds**: 5 seeds, mean.
**Safe Layer**: Analytical safety filter overriding RL actions that would violate constraints.

---

## 7. Key Observations

1. **RL improves over clinical baselines in simulation**: Across treatment planning and
   surgical tasks, RL agents consistently outperform standard-of-care protocols and scripted
   controllers in simulated environments. However, these gains require prospective clinical
   validation.
2. **Offline RL narrows the gap**: CQL and Decision Transformer reduce the performance gap
   between online and offline RL to 5–10%, making batch RL from retrospective data viable
   for clinical applications.
3. **Federated RL requires 2–5x more communication rounds**: Compared to federated supervised
   learning, federated RL convergence is significantly slower, particularly under high site
   heterogeneity. Advanced aggregation strategies are needed.
4. **Sim-to-real transfer is achievable but requires investment**: Domain randomization plus
   minimal real-world fine-tuning (10–50 episodes) effectively closes the sim-to-real gap for
   Physical AI tasks on phantoms. Cadaver and in-vivo validation remains an open challenge.
5. **Safety and performance trade off**: Constrained RL methods reduce violations but sacrifice
   5–15% of return. Analytical safety filters eliminate violations entirely but with larger
   performance cost. The choice depends on the acceptable risk level for each application.

---

## References

- Komorowski, M., et al. "The Artificial Intelligence Clinician learns optimal treatment
  strategies for sepsis in intensive care." Nature Medicine 24, 1716–1720 (2018).
- Gottesman, O., et al. "Guidelines for reinforcement learning in healthcare." Nature
  Medicine 25, 16–18 (2019).
- Henderson, P., et al. "Deep Reinforcement Learning That Matters." AAAI (2018).
- Scheikl, P.M., et al. "LapGym — An Open Source Framework for Reinforcement Learning in
  Robot-Assisted Laparoscopic Surgery." ICRA (2023).
- Levine, S., et al. "Offline Reinforcement Learning: Tutorial, Review, and Perspectives
  on Open Problems." arXiv:2005.01643 (2020).
