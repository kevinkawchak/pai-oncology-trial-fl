# Supervised Learning: Limitations for Oncology Clinical Trials with Physical AI

## Overview

Despite its maturity and regulatory familiarity, supervised learning faces fundamental
constraints when deployed across federated oncology networks with Physical AI systems.
These limitations are not merely academic — they directly impact model safety, fairness,
and clinical utility in multi-site trial settings.

## 1. Data Labeling Burden

The cost and complexity of acquiring high-quality labels in oncology is the single largest
bottleneck for supervised learning in federated settings.

- **Expert annotation scarcity**: Pathology segmentation masks require board-certified
  pathologists who charge $200–$500/hour. A single whole-slide image (WSI) at 40x
  magnification may require 30–60 minutes of expert time for exhaustive annotation. Scaling
  this across federated sites with thousands of cases is prohibitively expensive.
- **Label ambiguity and inter-rater variability**: Tumor grading (e.g., Gleason scoring in
  prostate cancer, Nottingham grading in breast cancer) exhibits well-documented inter-rater
  disagreement rates of 20–40%. Supervised models trained on noisy labels inherit and
  sometimes amplify this disagreement, producing inconsistent predictions across sites.
- **Temporal label decay**: Treatment guidelines evolve — the AJCC staging system, RECIST
  criteria, and molecular subtype definitions change over multi-year trial periods. Labels
  generated under older criteria become stale, requiring costly relabeling campaigns.
- **Physical AI annotation complexity**: Labeling intraoperative data (surgical video frames,
  force-torque traces, robotic kinematics) requires specialized annotation tools and domain
  expertise that few centers possess. There is no equivalent of Amazon Mechanical Turk for
  surgical phase recognition or tissue-tool interaction labeling.
- **Reference**: Elmore et al., "Diagnostic Concordance Among Pathologists Interpreting
  Breast Biopsy Specimens," JAMA, 2015.

## 2. Domain Shift Across Federated Sites

Heterogeneity across clinical sites is the defining challenge of federated oncology learning,
and supervised models are particularly vulnerable to distribution shift.

- **Scanner and protocol variation**: CT reconstruction kernels, MRI field strengths (1.5T
  vs 3T), PET tracer uptake times, and staining protocols (H&E thickness, IHC antibody
  clones) create site-specific image distributions that a single supervised model struggles
  to generalize across without explicit harmonization.
- **Population shift**: Patient demographics, cancer stage distributions, comorbidity
  profiles, and treatment histories vary systematically across geographic regions. A model
  trained predominantly on early-stage NSCLC at a comprehensive cancer center will
  underperform on advanced-stage cases at a community hospital.
- **Covariate shift in Physical AI sensors**: Robotic platforms across sites may use
  different endoscope models, lighting configurations, or force sensor calibrations. These
  hardware-level distributional shifts are invisible in the label space but catastrophic in
  feature space.
- **Label shift**: The prevalence of positive cases varies across sites (e.g., a tertiary
  referral center sees a higher proportion of rare tumor subtypes). Standard supervised
  training assumes matched train-test priors, an assumption violated in federated settings.
- **Reference**: Guan & Liu, "Domain Adaptation for Medical Image Analysis: A Survey,"
  IEEE Transactions on Biomedical Engineering, 2022.

## 3. Class Imbalance

Oncology datasets exhibit severe class imbalance at multiple levels, and standard supervised
losses handle this poorly without explicit intervention.

- **Rare tumor subtypes**: Sarcomas, neuroendocrine tumors, and rare molecular subtypes
  (e.g., NTRK fusion-positive cancers) may constitute <1% of cases at any single site.
  Federated aggregation can help but does not eliminate the imbalance when the condition is
  globally rare.
- **Event-rate imbalance in survival prediction**: For endpoints like 5-year overall
  survival in early-stage cancers, the event (death) rate may be <10%, creating extreme
  imbalance for binary classification formulations.
- **Pixel-level imbalance in segmentation**: Tumor voxels typically occupy <5% of a CT
  volume. Standard cross-entropy loss overwhelmingly optimizes for background classification,
  requiring Dice loss, focal loss, or other imbalance-aware objectives.
- **Federated imbalance amplification**: If some sites have near-zero prevalence of a rare
  class, their gradient contributions actively push the global model away from detecting that
  class. FedAvg weights by dataset size, inadvertently amplifying the majority-class bias of
  large-but-imbalanced sites.

## 4. Generalizability and External Validation

Supervised models optimized on internal cross-validation routinely fail to generalize to
truly external cohorts, a critical liability for multi-site trial deployment.

- **Shortcut learning**: Supervised models exploit spurious correlations — scanner-specific
  artifacts, institutional text overlays on images, site-specific EHR coding patterns — that
  do not represent genuine clinical signal. Federated training mitigates but does not
  eliminate this when sites share systematic confounders.
- **Overfit to majority populations**: Models trained predominantly on data from patients of
  European ancestry exhibit degraded performance on underrepresented populations, a failure
  mode with direct health equity implications in oncology trials.
- **Temporal generalization failure**: Models validated on retrospective data may fail on
  prospective trial data due to changes in treatment practice, imaging technology, and
  patient selection criteria over time.
- **Reference**: Zech et al., "Variable generalization performance of a deep learning model
  to detect pneumonia in chest radiographs," Nature Medicine, 2018.

## 5. Scalability Constraints for Physical AI Systems

- **Real-time adaptation gap**: Supervised models are static post-training. Physical AI
  systems operating in dynamic intraoperative environments cannot adapt to novel tissue
  presentations or unexpected anatomical variations without retraining — a process that takes
  hours to days and cannot occur during surgery.
- **Catastrophic forgetting under continual learning**: When sites attempt to fine-tune
  a global supervised model on local data streams (e.g., new surgical cases), the model
  forgets previously learned patterns. Elastic Weight Consolidation and similar techniques
  add complexity and do not fully solve the problem.
- **Multi-task interference**: Physical AI platforms require simultaneous predictions (tissue
  type, tool position, phase recognition, margin status). Multi-task supervised learning
  introduces gradient conflicts between task heads, degrading performance on safety-critical
  tasks.

## 6. Privacy and Regulatory Constraints on Labels

- **Label leakage risk**: In federated settings, gradient updates from supervised training
  can leak information about training labels through gradient inversion attacks. For oncology,
  this means diagnosis and outcome information — among the most sensitive protected health
  information — is at risk.
- **Consent scope limitations**: Patient consent for data use in clinical trials may not
  extend to the specific labeling activities required for supervised learning (e.g., detailed
  re-annotation by external pathologists), creating IRB compliance challenges.
- **Label provenance across jurisdictions**: GDPR, HIPAA, and other regulations impose
  different requirements on how labels (which constitute derived health data) can be created,
  stored, and shared — even in a federated framework where raw data stays local.

## 7. Evaluation Limitations

- **Metric saturation**: On well-curated benchmarks, supervised models achieve near-ceiling
  AUC (>0.99), making it difficult to distinguish meaningful improvements from noise. This
  masks real-world performance gaps that only manifest under distribution shift.
- **Lack of causal evaluation**: Supervised accuracy does not imply causal understanding.
  A model predicting treatment response from imaging may be exploiting confounders (tumor
  size correlating with treatment choice) rather than learning biologically meaningful
  features.

## Summary

Supervised learning's limitations in federated oncology settings are structural, not merely
engineering challenges to be optimized away. The labeling burden, domain shift vulnerability,
class imbalance sensitivity, and generalization failures collectively argue for hybrid
approaches that combine supervised training with self-supervised pre-training, domain
adaptation, and active learning strategies.
