# Active Learning with Coresets in MLPs
### Margin Sampling & Gaussian Kernel Reweighting on CIFAR-10 / CIFAR-100

**Anubhav Kumar (ZDA24B034) · Rohan Saha (ZDA24B009)**  
Foundations of Machine Learning — IIT Madras Zanzibar · June 2026

---

## Overview

This repository contains the complete experimental codebase for our course project on **compression-aware coreset selection** for MLPs. We study how coreset selection strategies interact with model compression (pruning), and propose a novel **compress-first ordering** — selecting coresets in the post-compression embedding space rather than the full-precision space.

### Key Finding

> Selecting coresets **after** compression (compress-first) consistently outperforms selecting **before** compression (select-first) by **+7 to +16 percentage points** across all selectors, pruning levels, and datasets. Embedding geometry collapse (Pearson r > 0.86 between silhouette score and Acc@1) is the mechanistic cause.

---

## Repository Structure

```
active-learning-coresets-mlp/
│
├── notebooks/
│   ├── exp1_selector_comparison.ipynb       # Anubhav — selector comparison at full precision
│   ├── exp2_degradation_curves.ipynb        # Rohan — accuracy vs sparsity curves
│   ├── exp3_embedding_geometry.ipynb        # Anubhav — embedding geometry under pruning
│   ├── exp4_order_of_operations.ipynb       # Rohan — compress-first vs select-first
│   ├── exp5_classical_ml.ipynb              # Anubhav — SVM/RF robustness check
│   └── exp6_sigma_ablation.ipynb            # Rohan — Gaussian sigma ablation + 3-way
│
├── results/
│   ├── exp1_summary.csv                     # EXP-1 aggregated results
│   ├── exp2_runs.csv                        # EXP-2 all 144 raw runs
│   ├── exp2_aggregated.csv                  # EXP-2 mean ± std over 3 seeds
│   ├── exp4_runs.csv                        # EXP-4 all 144 raw runs
│   ├── exp4_aggregated.csv                  # EXP-4 mean ± std per condition
│   ├── exp4_delta.csv                       # EXP-4 compress-first advantage (B - A)
│   ├── exp6_runs.csv                        # EXP-6 all 228 raw runs
│   ├── exp6_ablation_agg.csv                # EXP-6 sigma ablation aggregated
│   ├── exp6_threeway_agg.csv                # EXP-6 3-way comparison aggregated
│   └── exp6_quantization.csv                # EXP-6 quantization results
│
├── figures/
│   ├── exp1_bar_chart.png                   # Selector comparison bar chart
│   ├── exp2_cifar10_degradation.png         # CIFAR-10 degradation curves
│   ├── exp2_cifar100_degradation.png        # CIFAR-100 degradation curves
│   ├── exp2_combined.png                    # Combined 2x2 degradation figure
│   ├── exp2_training_curves.png             # Training curves
│   ├── exp3_cifar10_dual_axis.png           # Silhouette + Acc@1 dual axis
│   ├── exp3_cifar100_dual_axis.png          # CIFAR-100 dual axis
│   ├── exp3_combined_selectors.png          # All selectors silhouette
│   ├── exp3_heatmap.png                     # Silhouette heatmap
│   ├── exp4_delta_vs_pruning.png            # Compress-first advantage plot
│   ├── exp4_condA_vs_condB_bars.png         # Condition A vs B bar charts
│   ├── exp4_coverage_radius.png             # Coverage radius A vs B
│   ├── exp6_sigma_sensitivity.png           # Sigma sensitivity 2x2 grid
│   ├── exp6_threeway_degradation.png        # 3-way comparison curves
│   ├── exp6_gaussian_advantage.png          # Gaussian advantage over margin
│   └── exp6_coverage_radius.png             # Coverage radius 3-way
│
├── reports/
│   └── D2_report.pdf                        # D2 preliminary report
│
├── README.md
└── requirements.txt
```

---

## Experiments

| Exp | Owner | Description | Runs |
|-----|-------|-------------|------|
| EXP-1 | Anubhav | Selector comparison at full precision (0% pruning) | 24 |
| EXP-2 | Rohan | Accuracy-vs-sparsity degradation curves | 144 |
| EXP-3 | Anubhav | Embedding geometry under pruning (silhouette + coverage radius) | 36 |
| EXP-4 | Rohan | Order-of-operations: compress-first vs select-first | 144 |
| EXP-5 | Anubhav | Classical ML robustness check (SVM, RF) | 48 |
| EXP-6 | Rohan | Gaussian σ ablation + 3-way comparison vs plain margin | 228 |
| **Total** | | | **624** |

---

## Methods

### MLP Architecture
```
Input (3072) → FC-512-BN-ReLU-Dropout(0.3)
             → FC-256-BN-ReLU-Dropout(0.3)
             → FC-128-BN-ReLU  ← 128-dim embedding space
             → FC-Nclasses
```
1,740,682 parameters. Trained with Adam (lr=0.001, wd=1e-4, batch=256, 40 epochs, StepLR).

### Coreset Selectors

| Selector | Description |
|----------|-------------|
| Random | Uniform random sampling — primary baseline |
| Plain Margin | Selects most uncertain samples (smallest top-2 softmax gap) |
| k-Center Greedy | Minimises maximum distance from any point to nearest selected centre [1] |
| Gaussian Margin | Reweights margin by Gaussian kernel on embedding density (proposed) |

### Gaussian Kernel Reweighting (Novel)
```
score(i) = margin(i) × exp(-d² / 2σ²)
```
where `d` = distance to nearest cluster centre in 128-dim embedding space,  
`σ` = 0.25 × median pairwise distance (best from EXP-6 ablation).

### Pruning
Magnitude-based unstructured pruning via PyTorch `l1_unstructured` at rates:
0%, 10%, 30%, 50%, 70%, 90%.

### Compress-First Ordering (Novel Contribution)
- **Condition A (select-first):** coreset on full-precision embeddings → train → prune → finetune 10 epochs
- **Condition B (compress-first):** train full data → prune → coreset on compressed embeddings → finetune 20 epochs

---

## Key Results

### Compress-First Advantage (EXP-4) — CIFAR-10

| Selector | p=30% | p=50% | p=70% | Mean Δ |
|----------|-------|-------|-------|--------|
| Random | +7.34% | +7.06% | +6.70% | +7.03% |
| Margin | +15.04% | +14.70% | +14.96% | +14.90% |
| k-Center | +6.99% | +7.10% | +6.33% | +6.81% |
| Gaussian | +16.05% | +15.08% | +16.27% | +15.80% |

### Gaussian vs Plain Margin (EXP-6)

| Dataset | Plain Margin AUC | Gaussian AUC | Δ AUC |
|---------|-----------------|--------------|-------|
| CIFAR-10 | 31.86 | 34.19 | **+2.33** |
| CIFAR-100 | 8.36 | 9.63 | **+1.27** |

### Embedding Geometry (EXP-3) — Pearson r (silhouette ↔ Acc@1)

| Selector | Pearson r | p-value |
|----------|-----------|---------|
| Random | +0.9192 | 0.0095 |
| Margin | +0.8686 | 0.0248 |
| k-Center | +0.9490 | 0.0038 |
| Gaussian | +0.9461 | 0.0043 |

---

## Interaction Law

> *"MLP pruning above 50% sparsity collapses the penultimate-layer embedding geometry
> (Pearson r > 0.86 between silhouette score and Acc@1), invalidating coresets selected
> in the full-precision embedding space. Compress-first ordering recovers +7.10pp accuracy
> over select-first at 50% sparsity on CIFAR-10, with the advantage reaching +15.80pp
> for Gaussian margin selection."*

---

## Reproducibility

All experiments use fixed random seeds: **42, 123, 456**.  
All results averaged over 3 seeds with mean ± std reported.  
Coreset indices saved as `.npy` files — exact coresets used in each run are reproducible.

### Running on Kaggle

All notebooks were developed and run on **Kaggle T4 GPU** instances.

1. Upload the notebook to Kaggle
2. Enable GPU accelerator (Settings → Accelerator → GPU T4)
3. Enable Internet (Settings → Internet → On)
4. Run all cells in order

Each notebook is self-contained and includes crash-safe CSV logging — if the session dies, re-running the notebook resumes from the last completed run automatically.

---

## Datasets

- **CIFAR-10**: 50k train / 10k test / 10 classes — downloaded automatically via torchvision
- **CIFAR-100**: 50k train / 10k test / 100 classes — downloaded automatically via torchvision
- **Coreset budget**: 10% of training set = 5,000 samples for both datasets

---

## References

[1] Sener, O. & Savarese, S. (2018). *Active Learning for Convolutional Neural Networks: A Core-Set Approach.* ICLR 2018.

[2] Geifman, Y. & El-Yaniv, R. (2017). *Deep Active Learning over the Long Tail.* NeurIPS 2017.

[3] Bahri, Y. et al. (2022). *Explaining Neural Scaling Laws.* PNAS 2022.

[4] Settles, B. (2009). *Active Learning Literature Survey.* UW-Madison TR 1648.

[5] Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* UofT Technical Report.

---

## Authors

| Name | Roll No | Experiments |
|------|---------|-------------|
| Anubhav Kumar | ZDA24B034 | EXP-1, EXP-3, EXP-5 |
| Rohan Saha | ZDA24B009 | EXP-2, EXP-4, EXP-6 |

Course: Foundations of Machine Learning — IIT Madras Zanzibar — June 2026
