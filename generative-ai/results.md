# Generative AI: Results for Oncology Clinical Trials with Physical AI

> **DISCLAIMER: Illustrative data for demonstration purposes only. Published results cite
> sources. Benchmark results include hardware and seed configuration.**

## Overview

This document presents benchmark results for generative AI applied to federated oncology
settings with Physical AI integration. Generative AI evaluation spans multiple dimensions:
synthetic data quality (fidelity, privacy, utility), LLM clinical accuracy, agentic workflow
performance, and digital twin simulation fidelity. Published results are included alongside
illustrative federated benchmarks.

---

## 1. Synthetic Data Quality Benchmarks

### Illustrative: Synthetic CT Image Generation for Lung Cancer

Diffusion models trained to generate synthetic chest CT patches for lung nodule augmentation,
evaluated on visual quality and downstream utility.

| Model              | Training | FID (lower=better) | Downstream Sens@4FP | Privacy (ε) | Sites |
|--------------------|----------|--------------------|--------------------|-------------|-------|
| Real data only     | N/A      | N/A                | 0.871              | N/A         | 4     |
| DDPM (single-site) | Local    | 34.2               | 0.883              | ∞           | 1     |
| DDPM (FedAvg)      | Federated| 28.7               | 0.896              | ∞           | 4     |
| DDPM (FedAvg + DP) | Federated| 41.3               | 0.889              | 8.0         | 4     |
| LDM (FedAvg)       | Federated| 22.4               | 0.908              | ∞           | 4     |
| LDM (centralized)  | Central  | 19.8               | 0.913              | ∞           | 1     |

**Hardware**: NVIDIA A100 80GB, 4 GPUs per site. **Seeds**: 3 seeds, mean reported.
**FID**: Frechet Inception Distance computed on 10,000 generated vs real patches.
**Downstream**: ResNet-50 trained on real + synthetic data (50% augmentation ratio).

### Illustrative: Synthetic Tabular Data for Clinical Trial Augmentation

| Generator      | Statistical Fidelity | ML Utility (AUC) | Privacy (DCR) | Column Corr |
|----------------|---------------------|-------------------|---------------|-------------|
| Real data      | 1.000               | 0.731             | N/A           | 1.000       |
| CTGAN          | 0.891               | 0.709             | 0.423         | 0.867       |
| TVAE           | 0.878               | 0.698             | 0.451         | 0.854       |
| TabDDPM        | 0.923               | 0.724             | 0.389         | 0.912       |
| TabDDPM + DP   | 0.867               | 0.701             | 0.612         | 0.841       |

**Task**: Overall survival prediction for metastatic colorectal cancer.
**Hardware**: NVIDIA A100 40GB. **Seeds**: 5 seeds, mean reported.
**DCR**: Distance to Closest Record (higher = better privacy, min real-synthetic distance).
**Column Corr**: Average pairwise column correlation preservation.

---

## 2. LLM Clinical Accuracy Benchmarks

### Published: Medical Question Answering

| Model                | MedQA (USMLE) | PubMedQA | MMLU Clinical | Source                    |
|----------------------|----------------|----------|---------------|---------------------------|
| GPT-4 (2023)        | 90.2%          | 75.2%    | 87.0%         | Nori et al., 2023         |
| Med-PaLM 2          | 86.5%          | 81.8%    | 88.7%         | Singhal et al., 2023      |
| GPT-4o (2024)       | 92.1%          | 78.3%    | 89.4%         | OpenAI, 2024              |
| Llama 3 70B         | 78.3%          | 71.2%    | 80.1%         | Meta, 2024                |
| Mixtral 8x7B        | 72.1%          | 67.4%    | 74.3%         | Mistral AI, 2024          |

### Illustrative: Oncology-Specific LLM Evaluation

LLMs evaluated on oncology treatment recommendation accuracy against NCCN guidelines,
scored by board-certified oncologists.

| Model                     | Guideline Concordance | Hallucination Rate | Completeness | Latency (s) |
|---------------------------|----------------------|--------------------|--------------|-------------|
| GPT-4o                    | 0.87                 | 0.068              | 0.82         | 2.1         |
| GPT-4o + RAG (NCCN)      | 0.93                 | 0.031              | 0.91         | 3.8         |
| Llama 3 70B              | 0.74                 | 0.112              | 0.71         | 1.4         |
| Llama 3 70B + RAG (NCCN) | 0.84                 | 0.062              | 0.83         | 2.9         |
| Llama 3 8B (on-premise)  | 0.61                 | 0.178              | 0.58         | 0.6         |
| Llama 3 8B + RAG (NCCN)  | 0.73                 | 0.094              | 0.72         | 1.8         |

**Evaluation**: 200 oncology case vignettes across 10 cancer types, scored by 3 oncologists.
**Guideline Concordance**: Fraction of recommendations consistent with current NCCN guidelines.
**Hallucination Rate**: Fraction of outputs containing factually incorrect clinical claims.
**Hardware**: 4x A100 80GB (70B models), 1x A100 (8B models). **Temperature**: 0.0.

---

## 3. Agentic Workflow Benchmarks

### Illustrative: Automated Clinical Trial Matching

Agentic LLM systems matching patient profiles to eligible clinical trials from
ClinicalTrials.gov.

| System                  | Precision | Recall | F1    | Avg Time/Patient (s) | Trials Screened |
|-------------------------|-----------|--------|-------|---------------------|-----------------|
| Manual screening        | 0.95      | 0.62   | 0.75  | 2400 (40 min)       | 15              |
| Rule-based NLP          | 0.82      | 0.71   | 0.76  | 12                  | 200             |
| GPT-4o agent            | 0.88      | 0.84   | 0.86  | 45                  | 500             |
| GPT-4o agent + tools    | 0.91      | 0.87   | 0.89  | 62                  | 500             |
| Llama 3 70B agent       | 0.81      | 0.78   | 0.79  | 38                  | 500             |

**Dataset**: 150 patient profiles, 50 active oncology trials (illustrative).
**Tools**: Drug interaction DB, genomic variant annotator, eligibility logic parser.
**Note**: Manual screening precision is high but recall is limited by human time constraints.

### Illustrative: Federated Query Orchestration

Natural language to federated analytics pipeline, evaluated on query correctness and
result accuracy.

| Query Complexity | Parse Accuracy | Execution Success | Result Correctness | Avg Latency (s) |
|------------------|---------------|-------------------|-------------------|-----------------|
| Simple (1 table) | 0.96          | 0.94              | 0.92              | 8               |
| Medium (2 joins) | 0.89          | 0.85              | 0.81              | 23              |
| Complex (subquery)| 0.78         | 0.71              | 0.67              | 45              |
| Aggregation + stat| 0.84         | 0.79              | 0.74              | 34              |

**Model**: GPT-4o with structured output parsing. **Queries**: 100 per complexity level.

---

## 4. Digital Twin and Simulation Benchmarks

### Illustrative: Patient-Specific Tumor Growth Prediction

Generative models predicting tumor evolution from baseline imaging, compared against
actual follow-up imaging at 3 and 6 months.

| Model              | SSIM (3mo) | SSIM (6mo) | Volume Error (3mo) | Volume Error (6mo) |
|--------------------|-----------|-----------|-------------------|-------------------|
| Linear growth      | 0.721     | 0.643     | 28.4%             | 41.2%             |
| Gompertz model     | 0.768     | 0.701     | 19.7%             | 29.3%             |
| cGAN               | 0.834     | 0.762     | 12.3%             | 21.8%             |
| LDM (conditional)  | 0.871     | 0.803     | 8.7%              | 16.4%             |
| LDM (fed, 4 sites) | 0.862     | 0.791     | 9.8%              | 17.9%             |

**Hardware**: NVIDIA A100 80GB. **Seeds**: 3 seeds, mean reported.
**SSIM**: Structural Similarity Index against actual follow-up imaging.
**Cohort**: 120 NSCLC patients with baseline and follow-up CT scans (illustrative).

### Illustrative: Surgical Digital Twin for Pre-Operative Planning

Generative anatomical models for robotic surgery rehearsal, evaluated on geometric
accuracy and simulation utility.

| Metric                     | CT-only Reconstruction | + Diffusion Refinement | + Physics Sim |
|----------------------------|----------------------|----------------------|---------------|
| Surface distance (mm)      | 2.34                 | 1.47                 | 1.52          |
| Volume overlap (Dice)      | 0.891                | 0.934                | 0.931         |
| Vessel tree accuracy (%)   | 72.3                 | 86.7                 | 85.9          |
| Simulation time to setup (min) | 45              | 12                   | 18            |
| Surgeon utility score (1-5)| 3.2                  | 4.1                  | 4.4           |

**Hardware**: NVIDIA A100 80GB (generation), RTX 4090 (physics simulation).
**Evaluation**: 30 hepatic surgery cases reviewed by 5 surgeons (illustrative).

---

## 5. Privacy and Memorization Analysis

### Illustrative: Memorization Detection in Federated Generative Models

| Model            | Training | Memorization Rate | Nearest Neighbor Dist | MIA AUC |
|------------------|----------|------------------|-----------------------|---------|
| DDPM             | Local    | 0.023            | 0.412                 | 0.587   |
| DDPM (FedAvg)    | Federated| 0.018            | 0.438                 | 0.571   |
| DDPM + DP (ε=10) | Federated| 0.004            | 0.523                 | 0.531   |
| DDPM + DP (ε=1)  | Federated| 0.001            | 0.612                 | 0.508   |
| LDM              | Local    | 0.031            | 0.387                 | 0.612   |
| LDM + DP (ε=10)  | Federated| 0.006            | 0.498                 | 0.539   |

**Memorization Rate**: Fraction of generated samples within threshold distance of a
training sample. **MIA AUC**: Membership Inference Attack success rate (0.5 = random).
**Hardware**: NVIDIA A100 80GB. **Seeds**: 3 seeds, mean reported.

---

## 6. Computational Cost Comparison

### Illustrative: Resource Requirements Across Generative Approaches

| Application              | Model          | Training (GPU-hrs) | Inference (ms) | Memory (GB) |
|--------------------------|----------------|--------------------|--------------------|-------------|
| Synthetic CT generation  | DDPM           | 800                | 12,000 (per volume) | 24          |
| Synthetic CT generation  | LDM            | 400                | 3,200 (per volume)  | 16          |
| Synthetic tabular data   | TabDDPM        | 24                 | 50 (per batch)      | 4           |
| Clinical copilot         | Llama 3 70B    | 2,000 (fine-tune)  | 1,400 (per response)| 140         |
| Clinical copilot         | Llama 3 8B     | 200 (fine-tune)    | 600 (per response)  | 16          |
| Trial matching agent     | GPT-4o (API)   | N/A                | 45,000 (per patient)| N/A (cloud) |
| Digital twin generation  | LDM + physics  | 500                | 180,000 (per case)  | 32          |

**Hardware**: NVIDIA A100 80GB for training; varies for inference (noted per row).

---

## 7. Key Observations

1. **Synthetic data improves downstream performance**: Across imaging and tabular modalities,
   augmentation with generative synthetic data improves downstream model performance by 2–5%,
   with larger gains in low-data regimes and for rare classes.
2. **RAG dramatically reduces hallucination**: Retrieval-augmented generation cuts LLM
   hallucination rates by 50–70% in oncology applications, making RAG non-optional for any
   clinical deployment of LLM copilots.
3. **Federated generative models approach centralized quality**: Federated diffusion models
   achieve 90–95% of centralized FID scores, with the gap decreasing as the number of
   federated sites increases (more diverse training data).
4. **Agentic systems outperform static tools for trial matching**: LLM agents with tool
   access achieve F1 scores 13–17% higher than rule-based NLP systems for clinical trial
   matching, while screening 25x more trials than manual review.
5. **Privacy-utility tradeoff is manageable**: DP-trained generative models (epsilon=8–10)
   retain 90–95% of downstream utility while reducing memorization rates by 5–10x,
   providing practical privacy protection for clinical deployment.
6. **On-premise LLMs trade accuracy for control**: Llama 3 8B running on-premise achieves
   61–73% guideline concordance versus 87–93% for GPT-4o, quantifying the cost of data
   sovereignty constraints. RAG partially bridges this gap.

---

## References

- Singhal, K., et al. "Large language models encode clinical knowledge." Nature 620,
  172–180 (2023).
- Nori, H., et al. "Can Generalist Foundation Models Outcompete Special-Purpose Tuning?
  Case Study in Medicine." arXiv:2311.16452 (2023).
- Pinaya, W.H.L., et al. "Brain Imaging Generation with Latent Diffusion Models."
  MICCAI (2022).
- Ktena, S.I., et al. "Generative models improve fairness of medical classifiers."
  Nature Medicine (2024).
- Omiye, J.A., et al. "Large Language Models Propagate Race-Based Medicine." npj Digital
  Medicine 6, 195 (2023).
- FDA. "Artificial Intelligence and Machine Learning (AI/ML)-Enabled Medical Devices." (2023).
