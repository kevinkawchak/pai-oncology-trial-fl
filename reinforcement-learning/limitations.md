# Reinforcement Learning: Limitations for Oncology Clinical Trials with Physical AI

## Overview

Reinforcement learning faces severe practical barriers in clinical oncology and Physical AI
deployment. The gap between RL's theoretical promise and safe, reproducible clinical
deployment remains wide. These limitations must be addressed with engineering rigor before
RL-based systems can participate in regulated clinical workflows.

## 1. Sample Inefficiency

RL's data hunger is fundamentally at odds with the small, expensive datasets available in
oncology.

- **Episode cost in clinical settings**: Each "episode" in treatment planning RL represents
  a complete patient treatment trajectory — spanning weeks to months and costing tens of
  thousands of dollars. Online RL exploration (trying suboptimal actions to learn) is
  ethically and financially impossible. This forces reliance on offline/batch RL, which
  introduces its own limitations (see below).
- **Offline RL brittleness**: Batch Constrained Q-Learning (BCQ), Conservative Q-Learning
  (CQL), and Decision Transformer mitigate extrapolation error but require behavioral
  policies with sufficient coverage of the state-action space. In oncology, treatment data
  is heavily concentrated along a few standard-of-care protocols, leaving most of the
  action space unexplored.
- **Sparse and delayed rewards**: Treatment outcomes (overall survival, progression-free
  survival) may not be observed for years after treatment initiation. This extreme reward
  delay makes temporal credit assignment unreliable, especially with the small sample sizes
  typical of clinical trials (100–500 patients per arm).
- **Simulation sample requirements**: Even in simulation, model-free RL algorithms like PPO
  and SAC require millions of environment steps for surgical control tasks. High-fidelity
  surgical simulators run at 10–100x slower than real-time, making this computationally
  expensive.
- **Reference**: Levine et al., "Offline Reinforcement Learning: Tutorial, Review, and
  Perspectives on Open Problems," arXiv:2005.01643, 2020.

## 2. Safety Constraints

Safety is non-negotiable in oncology, and standard RL formulations provide insufficient
safety guarantees.

- **Reward hacking in clinical contexts**: RL agents optimize the specified reward function,
  which may not fully capture clinical intent. An agent optimizing tumor shrinkage may
  recommend toxic dose escalation that a clinician would never consider. Reward misspecification
  in oncology can cause direct patient harm.
- **Constraint satisfaction is not guaranteed**: Constrained RL methods (CPO, RCPO, Lagrangian
  relaxation) provide asymptotic constraint satisfaction but may violate constraints during
  training and early deployment — unacceptable in clinical settings where a single constraint
  violation (e.g., exceeding organ-at-risk dose limits) can cause irreversible injury.
- **No formal verification**: Unlike classical control systems, deep RL policies cannot
  currently be formally verified to satisfy safety properties across all possible states.
  This is a fundamental barrier to regulatory approval for autonomous Physical AI systems.
- **Distributional shift at test time**: RL policies trained in simulation or on historical
  data may encounter states outside their training distribution during deployment. Without
  robust out-of-distribution detection, the policy's actions in novel states are unpredictable.
- **Reference**: Dulac-Arnold et al., "Challenges of Real-World Reinforcement Learning,"
  ICML Workshop, 2019.

## 3. Sim-to-Real Gap

Physical AI systems require transfer from simulated training environments to real clinical
settings, and this gap is substantial.

- **Tissue modeling fidelity**: Soft tissue mechanics (viscoelasticity, anisotropy,
  inhomogeneity), bleeding dynamics, and thermal effects during electrosurgery are
  approximated by simulators but not faithfully reproduced. Policies learned on simplified
  tissue models may execute inappropriate force profiles or cutting trajectories on real
  tissue.
- **Sensor discrepancy**: Simulated RGB, depth, and force sensor outputs differ from their
  real counterparts in noise characteristics, latency, calibration, and dynamic range.
  Domain randomization helps but does not eliminate the gap, particularly for tactile and
  force feedback.
- **Environment stochasticity**: Real surgical environments include unexpected events —
  bleeding, tissue tearing, instrument malfunction, patient movement — that simulators
  typically do not model. RL policies without robust handling of these stochastic events
  may fail catastrophically.
- **Validation cost**: Validating sim-to-real transfer requires cadaver studies, animal
  models, and eventually human trials — a multi-year, multi-million-dollar process for each
  surgical task, making iteration cycles extremely long.

## 4. Reward Engineering

Designing reward functions for oncology RL is an unsolved problem that directly impacts
policy quality and safety.

- **Multi-stakeholder objectives**: Patients, oncologists, surgeons, insurers, and
  regulators have different and sometimes conflicting objectives. Encoding these as a scalar
  reward requires subjective weighting decisions that significantly affect learned policies.
- **Surrogate reward validity**: When long-term outcomes are unavailable, researchers use
  surrogate rewards (tumor response at 3 months, dose-volume histogram metrics, instrument
  tip accuracy). The validity of these surrogates as proxies for patient benefit is often
  unproven.
- **Reward non-stationarity**: Treatment guidelines, drug availability, and clinical
  endpoints change over time. A reward function calibrated to current practice may become
  inappropriate as standards of care evolve during a multi-year trial.
- **Inverse RL limitations**: Learning reward functions from expert demonstrations (inverse
  RL) assumes expert behavior is near-optimal — an assumption violated when expert clinicians
  disagree on treatment decisions or when practice varies across federated sites.

## 5. Federated RL-Specific Challenges

- **Non-stationary environments across sites**: Patient populations, treatment protocols,
  and outcome distributions differ across federated sites. Aggregating Q-functions or policy
  gradients across sites with different environmental dynamics can produce policies that are
  optimal nowhere — a failure mode worse than learning from a single site.
- **Communication overhead**: Policy gradient methods require frequent communication for
  stable learning. The communication rounds needed for federated RL convergence can be 5–10x
  higher than federated supervised learning, straining bandwidth and synchronization
  requirements across clinical sites.
- **Behavioral policy heterogeneity**: While behavioral policy diversity can help exploration,
  it also means that off-policy corrections (importance sampling ratios) must handle vastly
  different logging policies across sites, introducing high variance in value estimates.
- **Asynchronous patient trajectories**: Patient episodes at different sites proceed at
  different rates (treatment cycles may be 2, 3, or 4 weeks). Synchronizing federated RL
  updates across sites with asynchronous episode completion is an open research problem.

## 6. Reproducibility and Evaluation

- **High variance across seeds**: Deep RL algorithms exhibit notoriously high variance across
  random seeds, hyperparameters, and implementation details. Henderson et al. (2018) showed
  that standard RL benchmarks require 10+ seeds for reliable comparison — a standard rarely
  met in clinical RL papers.
- **Off-policy evaluation uncertainty**: Evaluating treatment policies without deploying them
  requires off-policy evaluation (OPE) methods (importance sampling, doubly robust estimators,
  fitted Q-evaluation). OPE estimates have wide confidence intervals and can be misleading
  when behavioral and target policies diverge significantly.
- **Lack of standardized clinical RL benchmarks**: Unlike supervised learning (which has
  BraTS, CAMELYON, TCGA), there are no widely accepted RL benchmarks for oncology treatment
  optimization, making cross-study comparison unreliable.
- **Reference**: Henderson et al., "Deep Reinforcement Learning That Matters," AAAI, 2018.

## 7. Regulatory and Ethical Barriers

- **No regulatory pathway for autonomous RL**: FDA has approved locked supervised learning
  algorithms (SaMD) but has no established pathway for continuously learning RL agents. The
  Predetermined Change Control Plan (PCCP) framework is a step toward adaptive algorithms
  but does not address fully autonomous RL.
- **Explainability deficit**: RL policies (deep Q-networks, actor-critic networks) are less
  interpretable than supervised classifiers. Clinicians cannot readily understand why an RL
  agent recommends a specific treatment action, hindering trust and adoption.
- **Liability assignment**: When an RL-recommended treatment leads to adverse outcomes, the
  chain of liability — developer, hospital, federation operator, treating physician — is
  legally undefined, creating institutional risk aversion.

## Summary

RL's limitations in oncology are not incremental engineering problems but fundamental
challenges at the intersection of sample efficiency, safety, and clinical validation. The
path to deployment requires advances in offline RL robustness, formal safety verification,
high-fidelity simulation, and regulatory frameworks for adaptive algorithms. In the near
term, RL is best positioned as a decision-support tool with mandatory human-in-the-loop
oversight rather than an autonomous agent.
