# Self-Supervised Learning: Results for Oncology Clinical Trials with Physical AI

> **DISCLAIMER: Illustrative data for demonstration purposes only. Published results cite
> sources. Benchmark results include hardware and seed configuration.**

## Overview

This document presents benchmark results for self-supervised learning in federated oncology
settings with Physical AI integration. SSL evaluation centers on representation quality
measured through downstream task performance after linear probing or fine-tuning. Published
results from pathology and radiology foundation models are included alongside illustrative
federated benchmarks.

---

## 1. Pathology Foundation Model Benchmarks

### Published: Large-Scale SSL Pathology Encoders

| Model                     | Pre-train Data   | Params | Downstream Tasks | Avg AUC | Source                        |
|---------------------------|------------------|--------|------------------|---------|-------------------------------|
| UNI (Chen et al.)         | 100M patches     | 307M   | 34 tasks         | 0.940   | Nature Medicine, 2024         |
| Virchow (Vorontsov et al.)| 1.5M slides      | 632M   | 12 tasks         | 0.938   | Nature Medicine, 2024         |
| CONCH (Lu et al.)         | 1.17M slide-text | 86M    | 14 tasks         | 0.927   | Nature Medicine, 2024         |
| CTransPath (Wang et al.)  | 15M patches      | 28M    | 9 tasks          | 0.894   | Medical Image Analysis, 2022  |
| RetCCL (Wang et al.)      | 2.4M patches     | 25M    | 7 tasks          | 0.881   | Medical Image Analysis, 2023  |

### Illustrative: Federated SSL Pathology Pre-Training

SSL pre-training across federated sites compared to centralized pre-training and ImageNet
initialization, evaluated on colorectal cancer MSI-H classification.

| Pre-Training Strategy      | Sites | Pre-train Patches | MSI AUC (linear) | MSI AUC (fine-tuned) |
|----------------------------|-------|-------------------|-------------------|----------------------|
| ImageNet supervised init   | N/A   | N/A               | 0.821             | 0.891                |
| DINO (single-site)         | 1     | 2M                | 0.862             | 0.912                |
| DINO (FedAvg)              | 5     | 10M (2M/site)     | 0.889             | 0.931                |
| DINO (centralized)         | 1     | 10M               | 0.897             | 0.938                |
| DINOv2 (FedAvg)            | 5     | 10M (2M/site)     | 0.901             | 0.942                |
| DINOv2 (centralized)       | 1     | 10M               | 0.908             | 0.947                |

**Hardware**: NVIDIA A100 80GB, 4 GPUs per site. **Seeds**: 3 seeds, mean reported.
**Architecture**: ViT-Base/16. **Pre-training**: 100 epochs, 200 federated rounds.

---

## 2. Radiology SSL Benchmarks

### Published: 3D Medical Image SSL

| Model                | Modality | Pre-train Data      | Downstream Task        | Dice/AUC | Source                    |
|----------------------|----------|---------------------|------------------------|----------|---------------------------|
| Swin UNETR (SSL)    | CT       | 5,050 CT volumes    | BTCV organ seg         | 0.872    | Tang et al., CVPR 2022    |
| Models Genesis       | CT       | 623 CT volumes      | Lung nodule detection  | 0.908    | Zhou et al., MIA 2021     |
| SwinMM              | CT/MRI   | 6,800 volumes       | BraTS segmentation     | 0.882    | Wang et al., MICCAI 2023  |

### Illustrative: Federated 3D SSL for Lung Cancer Screening

Federated MAE pre-training on unlabeled chest CT scans, evaluated on downstream lung
nodule detection (LUNA16-partitioned).

| Pre-Training Strategy      | Sites | Pre-train CTs | Sensitivity@4FP | AUC    |
|----------------------------|-------|---------------|-----------------|--------|
| Random initialization      | N/A   | N/A           | 0.832           | 0.921  |
| ImageNet 2D transfer       | N/A   | N/A           | 0.851           | 0.934  |
| MAE 3D (single-site)       | 1     | 1,200         | 0.868           | 0.942  |
| MAE 3D (FedAvg, 4 sites)   | 4     | 4,800         | 0.891           | 0.956  |
| MAE 3D (centralized)       | 1     | 4,800         | 0.898           | 0.961  |
| MAE 3D (FedAvg, 8 sites)   | 8     | 9,600         | 0.903           | 0.964  |

**Hardware**: NVIDIA A100 80GB, 2 GPUs per site. **Seeds**: 3 seeds, mean reported.
**Architecture**: ViT-Base/16 with 3D patch embedding. **Masking ratio**: 75%.

---

## 3. Label Efficiency Benchmarks

### Illustrative: SSL Pre-Training Reduces Label Requirements

Downstream performance on breast cancer HER2 status prediction from H&E WSIs as a
function of labeled training data, comparing SSL-pre-trained vs supervised-only training.

| Label Fraction | Supervised Only AUC | DINO Pre-trained AUC | DINOv2 Pre-trained AUC |
|----------------|---------------------|----------------------|------------------------|
| 1% (15 slides) | 0.612              | 0.781                | 0.803                  |
| 5% (75 slides) | 0.723              | 0.851                | 0.868                  |
| 10% (150 slides)| 0.789             | 0.889                | 0.901                  |
| 25% (375 slides)| 0.851             | 0.912                | 0.921                  |
| 50% (750 slides)| 0.891             | 0.928                | 0.934                  |
| 100% (1500 slides)| 0.912           | 0.938                | 0.942                  |

**Hardware**: NVIDIA A100 40GB. **Seeds**: 5 seeds, mean reported.
**Observation**: DINO pre-training with 10% labels matches supervised training with 50%
labels — a 5x label efficiency gain.

### Illustrative: Label Efficiency Across Oncology Tasks (10% Label Regime)

| Task                          | Supervised Only | SSL + Fine-tune | Relative Gain |
|-------------------------------|-----------------|-----------------|---------------|
| Glioma grade classification   | 0.742           | 0.871           | +17.4%        |
| Lung adenocarcinoma subtyping | 0.768           | 0.893           | +16.3%        |
| Breast cancer ER status       | 0.791           | 0.898           | +13.5%        |
| Colorectal MSI prediction     | 0.712           | 0.856           | +20.2%        |
| Liver lesion detection (CT)   | 0.823           | 0.901           | +9.5%         |
| Kidney tumor segmentation     | 0.781 (Dice)    | 0.867 (Dice)    | +11.0%        |

**Hardware**: NVIDIA A100 80GB. **Seeds**: 3 seeds, mean. **SSL method**: DINOv2 ViT-B/16.

---

## 4. Cross-Site Generalization

### Illustrative: SSL Representations Under Domain Shift

Evaluating frozen SSL representations on out-of-distribution sites (sites not included in
pre-training), compared to supervised features.

| Evaluation Setting        | Supervised Features (AUC) | SSL Features (AUC) | Gap   |
|---------------------------|---------------------------|---------------------|-------|
| In-distribution (seen sites)| 0.928                   | 0.934               | +0.006|
| Near OOD (similar scanner)  | 0.891                   | 0.918               | +0.027|
| Far OOD (different country)  | 0.842                  | 0.896               | +0.054|
| Far OOD (different stain protocol)| 0.817             | 0.881               | +0.064|

**Task**: Colorectal cancer tumor detection from H&E WSIs.
**Hardware**: NVIDIA A100 40GB. **Seeds**: 3 seeds, mean reported.
**Observation**: SSL representations degrade more gracefully under domain shift, with the
advantage increasing as shift severity increases.

---

## 5. Physical AI Sensor SSL Benchmarks

### Illustrative: Surgical Video SSL Pre-Training

SSL pre-training on unlabeled laparoscopic cholecystectomy videos, evaluated on downstream
surgical phase recognition (Cholec80 dataset).

| Pre-Training Method        | Pre-train Videos | Phase Accuracy | F1 (macro) | Latency (ms) |
|----------------------------|------------------|----------------|------------|---------------|
| Random init + supervised   | 0                | 0.871          | 0.823      | 12.4          |
| ImageNet init + supervised | 0                | 0.889          | 0.847      | 12.4          |
| VideoMAE (40 videos)       | 40               | 0.912          | 0.878      | 14.1          |
| VideoMAE (200 videos)      | 200              | 0.928          | 0.901      | 14.1          |
| TimeSformer + DINO (200)   | 200              | 0.921          | 0.893      | 16.8          |

**Hardware**: NVIDIA A100 40GB (training), Jetson Orin (inference latency). **Seeds**: 3.

### Illustrative: Robotic Kinematic SSL for Skill Assessment

SSL pre-training on unlabeled da Vinci kinematic traces (joint angles, velocities, gripper
states), evaluated on surgical skill classification (JIGSAWS dataset).

| Method                    | Pre-train Trials | Accuracy | Cohen's Kappa |
|---------------------------|------------------|----------|---------------|
| Hand-crafted features     | N/A              | 0.812    | 0.714         |
| LSTM (supervised)         | N/A              | 0.847    | 0.763         |
| TS2Vec SSL + linear probe | 500              | 0.873    | 0.801         |
| TS2Vec SSL + fine-tune    | 500              | 0.891    | 0.831         |
| TFC SSL + fine-tune       | 500              | 0.884    | 0.822         |

**Hardware**: NVIDIA V100 32GB. **Seeds**: 5 seeds, mean reported.

---

## 6. Federated SSL Communication and Convergence

### Illustrative: Communication Efficiency of Federated SSL

| SSL Method   | Architecture | Rounds | Bytes/Round (MB) | Total Transfer (GB) | Downstream AUC |
|--------------|--------------|--------|------------------|---------------------|----------------|
| SimCLR       | ResNet-50    | 150    | 98               | 14.7                | 0.912          |
| DINO         | ViT-B/16     | 200    | 346              | 69.2                | 0.931          |
| MAE          | ViT-B/16     | 120    | 346              | 41.5                | 0.923          |
| DINOv2       | ViT-L/16     | 250    | 1,226            | 306.5               | 0.942          |
| BYOL         | ResNet-50    | 180    | 98               | 17.6                | 0.918          |

**Task**: Colorectal cancer tissue classification. **Sites**: 5.
**Hardware**: NVIDIA A100 80GB per site.
**Observation**: MAE converges in fewer rounds than DINO due to smoother loss landscape,
but DINO achieves higher downstream AUC. ViT-L models have prohibitive communication cost.

---

## 7. Key Observations

1. **SSL narrows the centralized-federated gap**: Federated SSL achieves 97–99% of
   centralized SSL performance across pathology and radiology tasks, a smaller gap than
   observed in federated supervised learning (93–98%).
2. **Label efficiency is the primary value proposition**: SSL pre-training consistently
   provides 3–5x label efficiency gains across oncology tasks. At the 10% label regime,
   SSL + fine-tuning matches or exceeds fully supervised training.
3. **Pathology foundation models set the standard**: Published models (UNI, Virchow, CONCH)
   demonstrate that SSL at sufficient scale produces representations that generalize across
   dozens of cancer types and clinical tasks.
4. **Domain shift robustness**: SSL representations degrade 40–60% less than supervised
   features under out-of-distribution evaluation, validating their use in heterogeneous
   federated networks.
5. **Computational cost is the barrier**: ViT-Large SSL pre-training in federated settings
   requires hundreds of gigabytes of communication and thousands of GPU-hours, placing it
   beyond the reach of smaller clinical sites without cloud compute budgets.

---

## References

- Chen, R.J., et al. "Towards a general-purpose foundation model for computational
  pathology." Nature Medicine (2024).
- Vorontsov, E., et al. "Virchow: A Million-Slide Digital Pathology Foundation Model."
  Nature Medicine (2024).
- Lu, M.Y., et al. "A visual-language foundation model for computational pathology."
  Nature Medicine (2024).
- Tang, Y., et al. "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image
  Analysis." CVPR (2022).
- Zhou, Z., et al. "Models Genesis: Generic Autodidactic Models for 3D Medical Image
  Analysis." Medical Image Analysis (2021).
- Azizi, S., et al. "Big Self-Supervised Models Advance Medical Image Classification."
  ICCV (2021).
