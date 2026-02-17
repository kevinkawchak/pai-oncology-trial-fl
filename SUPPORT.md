# Getting Help

This document explains how to get help with PAI Oncology Trial FL and where to
find support for the upstream frameworks the platform builds upon.

## Getting Help with PAI Oncology Trial FL

### GitHub Issues

The primary channel for bug reports, feature requests, and technical questions
is the GitHub issue tracker:

**[github.com/kevinkawchak/pai-oncology-trial-fl/issues](https://github.com/kevinkawchak/pai-oncology-trial-fl/issues)**

Before opening a new issue:

1. **Search existing issues** to check whether your question or problem has
   already been addressed.
2. **Check the documentation** in `docs/`, the `README.md`, and example scripts
   in `examples/` for usage guidance.
3. **Review the changelog** (`CHANGELOG.md`) to see if the behavior changed in
   a recent release.

When opening an issue, use the appropriate issue template (bug report or feature
request) and provide as much detail as possible, including your Python version,
operating system, and the versions of key dependencies.

### Security Issues

Do **not** use the public issue tracker for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for private reporting instructions.

### Discussions

For open-ended questions, design discussions, or ideas that are not yet concrete
enough for a feature request, use
[GitHub Discussions](https://github.com/kevinkawchak/pai-oncology-trial-fl/discussions)
if enabled, or open an issue with the `question` label.

## Upstream Framework Support

PAI Oncology Trial FL integrates with and builds upon several upstream
frameworks. For issues specific to those frameworks rather than to this
project's integration layer, please consult their official support channels:

| Framework | Domain | Documentation | Issues / Support |
|-----------|--------|---------------|------------------|
| [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac-lab) | Robotic simulation and reinforcement learning | [Isaac Lab Docs](https://isaac-sim.github.io/IsaacLab/) | [Isaac Lab GitHub Issues](https://github.com/isaac-sim/IsaacLab/issues) |
| [MuJoCo](https://mujoco.org/) | Physics simulation | [MuJoCo Docs](https://mujoco.readthedocs.io/) | [MuJoCo GitHub Issues](https://github.com/google-deepmind/mujoco/issues) |
| [Gazebo](https://gazebosim.org/) | Multi-robot simulation | [Gazebo Docs](https://gazebosim.org/docs) | [Gazebo GitHub Issues](https://github.com/gazebosim/gz-sim/issues) |
| [PyBullet](https://pybullet.org/) | Physics simulation and robotics | [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/) | [PyBullet GitHub Issues](https://github.com/bulletphysics/bullet3/issues) |
| [MONAI](https://monai.io/) | Medical image AI | [MONAI Docs](https://docs.monai.io/) | [MONAI GitHub Issues](https://github.com/Project-MONAI/MONAI/issues) |
| [Flower](https://flower.ai/) | Federated learning framework | [Flower Docs](https://flower.ai/docs/) | [Flower GitHub Issues](https://github.com/adap/flower/issues) |

When reporting an issue, please determine whether the problem originates in
PAI Oncology Trial FL's integration code (e.g., `physical_ai/simulation_bridge.py`,
`physical_ai/framework_detection.py`) or in the upstream framework itself. If
you are unsure, open an issue in this repository and we will help triage.

## Regulatory and Clinical Disclaimer

PAI Oncology Trial FL is open-source research software. It is **not** a
certified medical device and has **not** received regulatory clearance or
approval from the FDA, EMA, or any other regulatory body.

The platform provides tooling for:

- Federated model training across institutional boundaries.
- Privacy-preserving techniques (differential privacy, secure aggregation,
  de-identification).
- Regulatory compliance checking and FDA submission tracking.
- Digital twin simulation and surgical task evaluation.

**None of these tools constitute medical advice, clinical decision support, or
validated diagnostic or therapeutic instruments.** The compliance checking and
FDA submission modules are informational aids and do not replace professional
regulatory, legal, or clinical counsel.

Deployers who intend to use this software, or derivatives of it, in clinical
settings or regulatory submissions are solely responsible for:

- Obtaining all required regulatory approvals and institutional review board
  (IRB) clearances.
- Validating model outputs against clinical evidence before any patient-facing
  use.
- Ensuring compliance with HIPAA, GDPR, FDA 21 CFR Part 11, ICH-GCP, and all
  other applicable regulations.
- Engaging qualified clinical, legal, and regulatory professionals.

For more information on deployer responsibilities, see [SECURITY.md](SECURITY.md).
