# Self-Supervised Learning: Limitations for Oncology Clinical Trials with Physical AI

## Overview

Despite its transformative potential for representation learning from unlabeled medical data,
self-supervised learning introduces its own set of challenges — particularly in the design
of pretext tasks for clinical domains, the computational cost of large-scale pre-training,
and the difficulty of evaluating representations without downstream task performance. These
limitations must be understood by engineering teams deploying SSL in federated oncology
settings.

## 1. Pretext Task Design for Medical Domains

The effectiveness of SSL depends critically on the pretext task aligning with downstream
clinical objectives, and this alignment is not guaranteed.

- **Domain-agnostic pretext tasks may miss clinical signal**: Standard contrastive learning
  augmentations (random crop, color jitter, Gaussian blur) were designed for natural images.
  In medical imaging, clinically relevant features — calcification patterns, subtle textural
  changes, small lesion foci — may be destroyed by aggressive augmentations. Color jitter on
  H&E pathology images can alter stain appearance in ways that remove diagnostically
  meaningful information.
- **Augmentation sensitivity**: The choice of augmentation pipeline has outsized impact on
  SSL quality. Studies show that removing a single augmentation (e.g., color distortion in
  SimCLR) can drop downstream performance by 10–15%. For medical imaging, the optimal
  augmentation set is modality-specific (CT vs MRI vs pathology vs surgical video) and
  requires empirical tuning that may not transfer across the federated network.
- **Negative pair selection**: Contrastive methods require negative pairs to be semantically
  different from the anchor. In medical datasets with high visual similarity (e.g., adjacent
  tissue patches from the same slide, consecutive CT slices), randomly sampled negatives may
  actually be semantically similar, producing false negative pairs that corrupt the
  representation space.
- **Masked modeling design choices**: For masked autoencoders (MAE), the masking ratio,
  patch size, and reconstruction target (pixel, token, feature) all affect representation
  quality. Medical images with large homogeneous regions (e.g., lung parenchyma in CT)
  make reconstruction trivial at low masking ratios, requiring higher masking ratios that
  may discard lesion regions entirely.
- **Reference**: Azizi et al., "Big Self-Supervised Models Advance Medical Image
  Classification," ICCV 2021.

## 2. Computational Cost

SSL pre-training at the scale required for medical foundation models demands computational
resources that most clinical institutions cannot provide.

- **GPU-hour requirements**: Pre-training a ViT-Large with DINO on 100 million pathology
  patches requires approximately 2,000–4,000 A100 GPU-hours (estimated from published
  configurations for UNI and Virchow). In a federated setting where each site pre-trains
  locally before aggregation, this cost is multiplied by the number of participating sites.
- **Memory pressure from medical data**: 3D medical volumes (CT, MRI) are substantially
  larger than 2D natural images. A single high-resolution CT scan occupies 200–500MB,
  requiring careful memory management for batch construction. Contrastive methods that
  maintain large memory banks or momentum queues face additional memory constraints.
- **Communication overhead for federated SSL**: Federated pre-training requires
  communicating model parameters across all sites at each aggregation round. For ViT-Large
  (307M parameters), each round transmits approximately 1.2GB per site. With hundreds of
  aggregation rounds, total communication cost can exceed 100GB per site over a pre-training
  run.
- **Energy and carbon cost**: Large-scale SSL pre-training has significant environmental
  impact. Clinical institutions operating under sustainability mandates must weigh the
  carbon cost of pre-training against the downstream label savings.
- **Diminishing returns at scale**: While scaling SSL pre-training data generally improves
  representations, the gains follow a logarithmic curve. Doubling pre-training data from
  10M to 20M samples yields substantially less improvement than doubling from 1M to 2M,
  making the cost-benefit calculus uncertain at the upper end.

## 3. Evaluation Challenges

Evaluating SSL representations without downstream task performance is an open problem that
complicates model selection and hyperparameter tuning.

- **No intrinsic quality metric**: Unlike supervised learning where training loss directly
  measures task performance, SSL training loss (contrastive loss, reconstruction loss) does
  not reliably predict downstream task performance. Low contrastive loss can correspond to
  representation collapse (all embeddings converging to a constant), and low reconstruction
  loss may indicate memorization rather than meaningful feature learning.
- **Probe-dependent evaluation**: The standard SSL evaluation protocol — linear probing or
  k-NN classification on a labeled validation set — introduces a dependency on labeled data
  that SSL was designed to avoid. The choice of probing protocol (linear vs nonlinear,
  frozen vs fine-tuned) can change the ranking of SSL methods, making comparison unreliable.
- **Task-specific evaluation gap**: SSL representations that excel on one downstream task
  (e.g., tissue classification) may underperform on another (e.g., cell segmentation). There
  is no guarantee that a single SSL checkpoint is optimal across all tasks in an oncology
  platform, requiring task-specific evaluation that is expensive to conduct.
- **Federated evaluation complexity**: In federated settings, evaluation must occur across
  sites with different downstream tasks and label distributions. A representation that
  performs well on average across sites may perform poorly at specific sites with unusual
  patient populations or imaging protocols.

## 4. Representation Collapse and Training Instability

- **Mode collapse in contrastive learning**: Without careful design, contrastive methods
  converge to trivial solutions where all representations are identical (collapse) or
  occupy only a low-dimensional subspace (dimensional collapse). While methods like BYOL,
  VICReg, and Barlow Twins address this, they add hyperparameters (exponential moving
  average coefficients, covariance regularization weights) that require tuning.
- **Batch size sensitivity**: SimCLR and MoCo performance is sensitive to batch size —
  larger batches provide more negative pairs and better gradient estimates. Federated
  sites with smaller datasets may have insufficient batch sizes for stable contrastive
  learning, creating a performance disparity across the federation.
- **Training instability with vision transformers**: ViT-based SSL exhibits training
  instability (loss spikes, gradient explosions) at large scale, requiring techniques like
  gradient clipping, learning rate warmup, and patch dropout that add engineering complexity.

## 5. Gap Between Pre-Training and Clinical Deployment

- **Fine-tuning is still required**: SSL does not eliminate the need for labeled data — it
  reduces the amount required. For rare cancers or novel biomarkers where even small labeled
  datasets do not exist, SSL representations alone cannot produce clinical predictions.
- **Frozen vs fine-tuned tradeoff**: Freezing the SSL backbone and training only the task
  head is computationally efficient but sacrifices performance. Full fine-tuning achieves
  better results but is expensive and may overwrite the general representations that made
  SSL valuable, particularly with small fine-tuning datasets.
- **Representation drift**: When the SSL model is updated via federated continual
  pre-training, downstream task heads trained on earlier representations become misaligned.
  Managing this drift requires periodic re-fine-tuning of all downstream heads — a
  coordination challenge in federated settings.
- **No causal guarantees**: SSL representations capture statistical regularities in the
  data, not causal mechanisms. Features that are predictive in the pre-training distribution
  may not be causal markers of disease, limiting their reliability under distribution shift.

## 6. Domain-Specific Technical Challenges

- **3D SSL for volumetric data**: Most SSL research targets 2D images. Extending contrastive
  learning and masked modeling to 3D medical volumes (CT, MRI) requires architectural
  modifications (3D patch embedding, volumetric augmentations) and significantly more
  compute. The optimal pretext task design for 3D data is less well understood.
- **Multi-resolution pathology**: WSIs span 4–5 magnification levels, each carrying different
  information (tissue architecture at low magnification, cellular morphology at high
  magnification). SSL methods must learn representations that capture this multi-scale
  structure, but most current approaches operate at a single resolution.
- **Temporal data for Physical AI**: SSL for surgical video and robotic sensor streams
  requires temporal pretext tasks (temporal order prediction, video pace prediction) that
  are less mature than spatial tasks for static images.

## 7. Regulatory and Interpretability Concerns

- **Opaque feature learning**: SSL features are learned without human-specified semantics,
  making them harder to interpret than supervised features optimized for clinically defined
  labels. Regulatory submissions must explain what the model has learned, which is
  challenging for SSL representations.
- **Pre-training data governance**: In federated SSL, the pre-training data corpus spans
  multiple institutions with different consent frameworks. Ensuring that pre-training on
  unlabeled data complies with each institution's IRB approval and data use agreements
  requires careful governance.
- **Reproducibility challenges**: SSL training is sensitive to hyperparameters, random
  seeds, and hardware configuration. Reproducing SSL results across federated sites with
  different GPU types, batch sizes, and data ordering is non-trivial.

## Summary

SSL's limitations in federated oncology are primarily in the engineering and evaluation
domains: designing appropriate pretext tasks for clinical data, managing computational cost
across federated sites, and evaluating representations without reliable intrinsic metrics.
These challenges are actively being addressed by the research community, but production
deployments must account for the current maturity gaps when building SSL into clinical
oncology pipelines.
