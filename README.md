# Active Learning with Coresets in MLPs
## Margin Sampling & Gaussian Kernel Reweighting on CIFAR-10 / CIFAR-100

**Anubhav Kumar (ZDA24B034) · Rohan Saha (ZDA24B009)**
Foundations of Machine Learning — IIT Madras Zanzibar · June 2026

---

## Project Summary

This repository contains the complete implementation for our course project on **compression-aware coreset selection** for MLPs. We study how coreset selection strategies interact with model compression (pruning + quantization) and propose a novel **compress-first ordering** — selecting coresets in the post-compression embedding space rather than the full-precision space.

### Key Finding

> Selecting coresets **after** compression (compress-first) consistently outperforms selecting **before** compression (select-first) by **+7 to +16 percentage points** across all selectors, pruning levels, and datasets. Embedding geometry collapse (Pearson r > 0.86 between silhouette score and Acc@1) is the mechanistic cause. The Gaussian advantage over plain margin sampling is fully preserved after INT8 quantization (< 0.1 pp drop).

---

## D2 Implementation Checklist

### Classical ML Methods (minimum 2-3 required)
- **SVM** — RBF kernel, C=10, gamma=scale, trained on PCA-100 features
- **Random Forest** — 200 trees, trained on PCA-100 features
- **MLP** — 3-layer fully connected, 1,740,682 parameters (see architecture below)

### MLP Compression Techniques Applied
- **Coreset Selection** — 4 strategies: random, plain margin, k-center greedy, Gaussian margin
- **Pruning** — magnitude-based unstructured (PyTorch l1_unstructured) at 0%, 10%, 30%, 50%, 70%, 90%
- **Quantization** — post-training INT8 dynamic quantization (PyTorch quantize_dynamic) applied to all margin-based selectors at all pruning levels

### Dimensionality Reduction (optional — implemented)
- **PCA-100** applied to raw 32x32x3 pixels for all classical ML experiments
- Explains >90% of pixel variance on both CIFAR-10 and CIFAR-100

---

## MLP Architecture

```
Input (3,072) → FC-512 → BatchNorm → ReLU → Dropout(0.3)
             → FC-256 → BatchNorm → ReLU → Dropout(0.3)
             → FC-128 → BatchNorm → ReLU   ← 128-dim embedding space
             → FC-Nclasses
```

**Total parameters:** 1,740,682

---

## Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Coreset budget | 5,000 samples (10% of training set) |
| MLP training epochs | 40 |
| Fine-tune epochs (post-prune) | 10 (select-first), 20 (compress-first) |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| LR scheduler | StepLR (step=15, gamma=0.1) |
| Pruning method | Magnitude-based unstructured (l1_unstructured) |
| Pruning rates | 0%, 10%, 30%, 50%, 70%, 90% |
| Quantization | INT8 dynamic (quantize_dynamic) |
| Gaussian sigma | 0.25 × median pairwise distance |
| PCA components | 100 |
| SVM C / kernel | 10 / RBF |
| RF trees | 200 |
| Random seeds | 42, 123, 456 |

---

## Coreset Selection Methods

### 1. Random Selector (baseline)
Uniform random sampling without replacement. Primary baseline.

### 2. Plain Margin Sampling (Settles baseline)
Selects the n most uncertain samples — smallest gap between top-2 softmax probabilities:
```
score(i) = p1(i) - p2(i)
```

### 3. k-Center Greedy (Sener & Savarese 2018)
Iteratively selects the point farthest from all currently selected centres, minimising the maximum distance from any unselected point to its nearest selected neighbour. Produces the tightest geometric coverage (lowest coverage radius R).

### 4. Gaussian-Kernel Margin Reweighting (proposed — novel)
Reweights margin scores by a Gaussian kernel on embedding density:
```
score(i) = margin(i) × exp(-d² / 2σ²)
```
where d = distance from sample i to its nearest cluster centre in the 128-dim embedding space, and σ = 0.25 × median pairwise distance. Upweights uncertain samples in dense embedding regions.

---

## Novel Contribution: Compress-First Ordering

### Condition A — Select-First (standard approach)
```
Select coreset on full-precision embeddings
→ Train MLP on coreset (40 epochs)
→ Prune at target sparsity
→ Fine-tune (10 epochs)
→ Evaluate
```

### Condition B — Compress-First (proposed)
```
Train MLP on full 50k samples (40 epochs)
→ Prune at target sparsity
→ Extract compressed 128-dim embeddings
→ Select coreset in compressed embedding space
→ Fine-tune on coreset (20 epochs)
→ Evaluate
```

**Why it works:** Pruning collapses the embedding geometry. Coresets selected in the full-precision space become misaligned with the compressed model's learned representation. Selecting in the compressed space ensures the coreset matches what the compressed model can learn.

---

## Experiments

| Exp | Owner | Description | Runs |
|-----|-------|-------------|------|
| EXP-1 | Anubhav | Selector comparison at full precision (0% pruning) | 24 |
| EXP-2 | Rohan | Accuracy-vs-sparsity degradation curves | 144 |
| EXP-3 | Anubhav | Embedding geometry under pruning (silhouette + coverage radius R) | 36 |
| EXP-4 | Rohan | Order-of-operations: compress-first vs select-first | 144 |
| EXP-5 | Anubhav | Classical ML robustness check (SVM, RF vs MLP) | 48 |
| EXP-6 | Rohan | Gaussian sigma ablation + 3-way comparison + quantization | 228 |
| **Total** | | | **624** |

---

## Key Results

### Classical ML vs MLP — Acc@1 (EXP-1 + EXP-5)

| Dataset | Selector | MLP | SVM | RF |
|---------|----------|-----|-----|----|
| CIFAR-10 | Random | 45.41 ± 0.37 | 44.16 ± 0.33 | 40.62 ± 0.82 |
| CIFAR-10 | k-Center | 45.33 ± 0.41 | 42.46 ± 0.49 | 36.66 ± 1.38 |
| CIFAR-10 | Gaussian | 42.91 ± 0.06 | 42.27 ± 0.93 | 36.33 ± 0.44 |
| CIFAR-10 | Margin | 34.70 ± 0.50 | 37.14 ± 0.57 | 23.23 ± 1.24 |

**Selector ranking Random > k-Center > Gaussian > Margin is identical across SVM, RF, and MLP — confirms selectors measure genuine data geometry not model artefacts.**

### Accuracy vs Sparsity (EXP-2) — CIFAR-10

| Selector | 0% | 50% | 90% | AUC |
|----------|-----|-----|-----|-----|
| Random | 45.44 ± 0.82 | 46.01 ± 0.63 | 44.60 ± 0.45 | 41.15 |
| k-Center | 45.22 ± 0.08 | 45.24 ± 0.25 | 44.07 ± 0.15 | 40.62 |
| Gaussian | 34.27 ± 0.58 | 35.39 ± 0.85 | 32.64 ± 0.93 | 31.31 |
| Margin | 34.98 ± 0.85 | 35.97 ± 0.20 | 33.33 ± 0.71 | 32.12 |

### Compress-First Advantage (EXP-4) — CIFAR-10

| Selector | p=30% | p=50% | p=70% | Mean Delta |
|----------|-------|-------|-------|------------|
| Random | +7.34% | +7.06% | +6.70% | +7.03% |
| Margin | +15.04% | +14.70% | +14.96% | +14.90% |
| k-Center | +6.99% | +7.10% | +6.33% | +6.81% |
| Gaussian | +16.05% | +15.08% | +16.27% | +15.80% |

**All values positive — compress-first wins unanimously across every selector, pruning level, and dataset.**

### Quantization Robustness (EXP-6) — CIFAR-10

| Method | p=0% Before | p=0% After | p=50% Before | p=50% After |
|--------|------------|-----------|-------------|------------|
| Plain Margin | 35.50 | 35.55 | 35.60 | 35.59 |
| Gaussian | 38.72 | **38.82** | 38.43 | **38.48** |
| Inv. Gaussian | 34.48 | 34.56 | 35.80 | 35.83 |

**Mean accuracy drop from INT8 quantization: < 0.1 pp across all methods and pruning levels.**

### Embedding Geometry (EXP-3) — Pearson r (silhouette vs Acc@1)

| Selector | Pearson r | p-value |
|----------|-----------|---------|
| Random | +0.9192 | 0.0095 |
| Margin | +0.8686 | 0.0248 |
| k-Center | +0.9490 | 0.0038 |
| Gaussian | +0.9461 | 0.0043 |

**All r > 0.86, p < 0.05 — embedding geometry collapse is the mechanistic cause of accuracy degradation under pruning.**

---

## Interaction Law

> *MLP pruning above 50% sparsity collapses the penultimate-layer embedding geometry (Pearson r > 0.86 between silhouette score and Acc@1), invalidating coresets selected in the full-precision embedding space. Compress-first ordering recovers +7.10pp accuracy over select-first at 50% sparsity on CIFAR-10, with the advantage reaching +15.80pp for Gaussian margin selection. The Gaussian advantage is robust to INT8 quantization (< 0.1pp degradation).*

---

## Repository Structure

- `notebooks/` — All 6 experiment Jupyter notebooks (run on Kaggle T4 GPU)
- `results/` — All CSV result files (raw runs + aggregated, mean ± std over 3 seeds)
- `figures/` — All experiment figures (PNG, 150 DPI)
- `reports/` — D2 preliminary report PDF (NeurIPS 2026 template)
- `README.md` — This file
- `requirements.txt` — Python dependencies

---

## Reproducibility

- Fixed random seeds: **42, 123, 456** across all experiments
- All results averaged over 3 seeds with mean ± std reported
- Crash-safe CSV logging — re-running any notebook resumes from last completed run automatically
- Coreset indices saved as `.npy` files — exact coresets per run are fully reproducible
- All notebooks run on **Kaggle T4 GPU** (16GB VRAM, PyTorch 2.10, CUDA 12.8)

---

## Datasets

- **CIFAR-10**: 50k train / 10k test / 10 classes / 5,000 samples per class
- **CIFAR-100**: 50k train / 10k test / 100 classes / 500 samples per class
- **Coreset budget**: 10% = 5,000 samples for both datasets
- Both datasets perfectly balanced — no class weighting required
- Downloaded automatically via torchvision

---

## References

1. Sener, O. & Savarese, S. (2018). *Active Learning for Convolutional Neural Networks: A Core-Set Approach.* ICLR 2018.
2. Geifman, Y. & El-Yaniv, R. (2017). *Deep Active Learning over the Long Tail.* NeurIPS 2017.
3. Bahri, Y. et al. (2022). *Explaining Neural Scaling Laws.* PNAS 2022.
4. Settles, B. (2009). *Active Learning Literature Survey.* UW-Madison TR 1648.
5. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.*

---

## Authors

| Name | Roll No | Experiments |
|------|---------|-------------|
| Anubhav Kumar | ZDA24B034 | EXP-1, EXP-3, EXP-5 |
| Rohan Saha | ZDA24B009 | EXP-2, EXP-4, EXP-6 |

**Course:** Foundations of Machine Learning — IIT Madras Zanzibar — June 2026
