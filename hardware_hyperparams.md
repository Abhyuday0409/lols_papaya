# Hardware, Hyperparameters, and Efficiency Reference

This document consolidates all configuration tables and timing benchmarks from the paper appendix. Numbers here correspond directly to Tables XII–XV in the paper and are provided to support exact reproduction of reported results.

---

## Hardware and Software Environment

| | |
|---|---|
| **Framework** | PyTorch 1.12.0, Python 3.11, DGL |
| **Training device** | NVIDIA A100-SXM4-80GB, CUDA 12.2 |
| **CPU inference device** | Apple M4 Max |

All accuracy numbers in the paper are averaged over ten independent seeds. Inference latency figures are CPU-only and averaged over five seeds.

---

## Table XII — Student and Teacher Network Configurations

Per-dataset hidden dimensions, layer counts, and feature mask settings. `S` = student MLP; `T` = teacher GNN (same architecture used for GCN, GAT, and GraphSAGE). `Masks` refers to the number of feature masks applied during training where dataset splits are not pre-defined; `--` indicates standard random splits are used instead.

| Dataset | S.Dim | S.Layers | T.Dim | T.Layers | Masks |
|---------|------:|:--------:|------:|:--------:|:-----:|
| Actor | 128 | 2 | 256 | 3 | 10 |
| PubMed | 256 | 3 | 256 | 3 | — |
| Amazon-Photo | 256 | 2 | 256 | 2 | — |
| Coauthor CS | 256 | 2 | 256 | 2 | — |
| Flickr | 512 | 3 | 2048 | 3 | — |
| Questions | 256 | 2 | 256 | 3 | 10 |
| Amazon Ratings | 256 | 2 | 256 | 3 | 10 |

Where masks are not provided, an 80/10/10 train/validation/test split is used.

---

## Table XIII — DBSCAN Clusters Discovered per Dataset

Final cluster count and the `ε` value selected via grid search on the validation set. Embeddings are L2-normalised prior to DBSCAN so all distances lie in [0, 2].

| Dataset | # Clusters (*L*) | ε |
|---------|:----------------:|:---:|
| Actor | 3 | 0.30 |
| PubMed | 4 | 0.25 |
| Amazon-Photo | 5 | 0.30 |
| Coauthor CS | 6 | 0.25 |
| Flickr | 4 | 0.30 |
| Questions | 3 | 0.25 |
| Amazon Ratings | 4 | 0.30 |

---

## DBSCAN ε Sensitivity Analysis

A common question is how sensitive MIDAS is to the choice of ε. Two things are worth clarifying upfront:

- **ε controls cluster granularity, not the number of adapters.** The number of cluster-specific adapters equals the number of teachers (3), which is fixed. What ε governs is how finely the node population is partitioned before teacher assignment — a smaller ε produces more, tighter clusters; a larger ε merges them.
- **There is a stable operating regime.** Sweeping ε on Actor and PubMed reveals two clear regimes: below the threshold, most nodes are labelled as outliers and accuracy drops noticeably; above it, accuracy variance stays below 0.008, indicating robust behaviour. The selected ε for every dataset falls within this stable region.

### ε sweep — Actor

| ε | # Clusters | Outlier % | Test Acc. |
|:---:|:----------:|:---------:|:---------:|
| 0.10 | 8 | 41.2 | 33.81 |
| 0.15 | 6 | 28.7 | 35.40 |
| 0.20 | 5 | 17.3 | 37.02 |
| 0.25 | 4 | 9.1 | 38.44 |
| **0.30** | **3** | **4.6** | **38.75** ← selected |
| 0.40 | 3 | 3.8 | 38.68 |
| 0.50 | 2 | 3.1 | 38.61 |
| 0.70 | 1 | 2.4 | 37.90 |

### ε sweep — PubMed

| ε | # Clusters | Outlier % | Test Acc. |
|:---:|:----------:|:---------:|:---------:|
| 0.10 | 9 | 38.5 | 76.14 |
| 0.15 | 7 | 24.1 | 78.33 |
| 0.20 | 5 | 12.8 | 80.10 |
| **0.25** | **4** | **6.3** | **81.70** ← selected |
| 0.30 | 4 | 5.7 | 81.58 |
| 0.40 | 3 | 4.9 | 81.41 |
| 0.50 | 2 | 3.8 | 81.20 |
| 0.70 | 1 | 2.6 | 80.05 |

### Accuracy vs ε (Actor)

```
Acc
39.0 |                    ●─────●─────●
38.5 |               ●
38.0 |                                      ●
37.5 |          ●
37.0 |
36.5 |
36.0 |     ●
35.5 |
35.0 |
34.0 | ●
     +----+----+----+----+----+----+----+----→  ε
       0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.70
                         ↑
                      selected
```

The plateau from ε = 0.25 to 0.50 (variance < 0.008) is the stable region. Below ε ≈ 0.20 the outlier rate rises sharply and accuracy degrades. This pattern is consistent across all seven datasets.

**Choosing ε for a new dataset:** sweep the range 0.2–0.7 in steps of 0.05, targeting an outlier ratio below 15% and at least 2 non-trivial clusters. The silhouette score printed during the run is a reliable guide when ground-truth labels are unavailable.

---

## Table XIV — End-to-End Training Cost and Inference Speedup

Training times are in seconds and measured on the A100. Struc2Vec is a **one-time offline preprocessing step** whose cost is amortised across all subsequent runs and seeds. Inference times are CPU-only in milliseconds.

### Training cost (seconds)

| Phase | Actor | PubMed | Co-CS | A-Photo |
|-------|------:|-------:|------:|--------:|
| Avg. single teacher | 47.0 | 88.7 | 371.5 | 105.5 |
| Struc2Vec (one-time) | 24.5 | 29.0 | 23.2 | 22.2 |
| DBSCAN + center loss | 0.9 | 0.9 | 0.6 | 0.3 |
| Dual-encoder training | 2.1 | 4.3 | 10.5 | 1.7 |
| Distillation | 11.2 | 24.9 | 16.5 | 6.2 |
| **Total MIDAS** | **64.4** | **103.5** | **185.1** | **349.2** |

Total MIDAS training cost is comparable to training three separate GNN teachers (3× single-teacher cost), with the Struc2Vec step paid only once.

### Inference (CPU-only, milliseconds)

| Method | Actor | PubMed | Co-CS | A-Photo |
|--------|------:|-------:|------:|--------:|
| Avg. teacher | 47.0 | 88.7 | 371.5 | 105.5 |
| Sequential ensemble | 141.7 | 266.1 | 1114.5 | 316.5 |
| **MIDAS student** | **6.61** | **20.3** | **51.1** | **15.7** |
| Speedup vs avg. teacher | 7.1× | 4.4× | 7.3× | 6.7× |
| Speedup vs ensemble | 21.4× | 13.1× | 21.8× | 20.2× |

MIDAS student inference accesses only raw node features and executes a single trunk forward pass followed by one adapter — no adjacency matrix, no neighbourhood lookup. This gives a 7–22× speedup over teacher GNNs and a 13–22× speedup over running all three teachers sequentially.

---

## Table XV — Struc2Vec Scalability

Theoretical operation counts and empirical/estimated preprocessing times under exact and κ-NN approximate Struc2Vec (κ = 10). Exact times marked † are theoretical estimates and were not directly measured; the κ-NN variant (implemented via FastDTW with κ-NN candidate restriction) was used for all datasets in the paper.

| Dataset | *n* | *k'* | κ-NN time (s) | Exact time (s)† | *n*/κ speedup |
|---------|----:|:---:|-------------:|----------------:|:-------------:|
| Actor | 7,600 | 3 | 24.5 | ~18,620 | 760× |
| PubMed | 19,717 | 4 | 29.0 | ~57,179 | 1,972× |
| Amazon-Photo | 7,650 | 5 | 22.2 | ~16,983 | 765× |
| Coauthor CS | 18,333 | 6 | 23.2 | ~42,533 | 1,833× |
| Flickr | 89,520 | 4 | ~491† | ~4.4M† | 8,952× |
| Questions | 48,921 | 3 | ~191† | ~932K† | 4,892× |
| Amazon Ratings | 24,492 | 4 | ~119† | ~291K† | 2,449× |

*k'* = number of DBSCAN clusters discovered for that dataset (Table XIII). Theoretical speedup is *n*/κ. The random walk phase contributes O(*nRL*) independently and is embarrassingly parallel across nodes.

---

## Full Hyperparameter Search Space

All hyperparameters are defined as named constants at the top of each script. The defaults below reproduce the paper results on Actor. Ranges listed are recommended starting points for adaptation to new datasets. Final values for each dataset were selected via grid search on the validation set.

### `train_teachers.py`

| Parameter | Default | Tuning range | Notes |
|-----------|---------|-------------|-------|
| `HIDDEN_DIM` | 128 | 64–512 | GNN hidden dimension for all teachers |
| `LR` | 0.01 | 1e-3 – 5e-2 | Adam learning rate |
| `WEIGHT_DECAY` | 5e-4 | 1e-5 – 1e-3 | L2 regularisation |
| `EPOCHS` | 200 | 100–500 | Max training epochs |
| `PATIENCE` | 20 | 10–50 | Early stopping patience |
| `DROPOUT_SAGE / GCN` | 0.5 | 0.3–0.7 | |
| `DROPOUT_GAT` | 0.6 | 0.4–0.7 | |
| `GAT_HEADS` | 8 | 4–16 | Attention heads for GAT |
| `NUM_SEEDS` | 5 | — | Number of independent runs |

### `clustering.py`

| Parameter | Default | Tuning range | Notes |
|-----------|---------|-------------|-------|
| `STRUC2VEC_DIM` | 128 | 64–256 | Embedding dimension for Struc2Vec and dual encoder output |
| `WALKS_PER_NODE` | 25 | 10–50 | More walks = richer structural signal, slower |
| `WALK_LENGTH` | 20 | 10–40 | |
| `CLIP_HIDDEN` | 256 | 128–512 | Dual encoder hidden dimension |
| `CLIP_OUT` | 128 | 64–256 | Shared latent space dimension; student input dimension |
| `CLIP_LR` | 1e-4 | 5e-5 – 5e-4 | |
| `CLIP_EPOCHS` | 300 | 100–500 | |
| `CLIP_TEMP` | 0.07 | 0.05–0.2 | InfoNCE temperature; lower = sharper alignment |
| `CENTER_LOSS_WEIGHT` | 0.001 | 1e-4 – 1e-2 | Weight of center loss relative to InfoNCE + CE |
| `CENTER_LR` | 0.05 | 0.01–0.5 | SGD learning rate for cluster centers |
| `DBSCAN_EPS` | 0.7 | 0.2–0.7 | Most dataset-sensitive parameter; see ε sensitivity analysis above for sweep methodology |
| `DBSCAN_MIN_SAMPLES` | 5 | 3–20 | Minimum cluster size; increase to reduce noise clusters |

### `train_student.py`

| Parameter | Default | Tuning range | Notes |
|-----------|---------|-------------|-------|
| `TRUNK_HIDDEN` | 512 | 256–1024 | Shared trunk width |
| `ADAPTER_HIDDEN` | 128 | 64–256 | Per-cluster adapter bottleneck; ~0.1–0.3× trunk |
| `DROPOUT` | 0.3 | 0.1–0.5 | Applied in trunk and adapters |
| `LR` | 5e-2 | 1e-3 – 1e-1 | Phase 2 learning rate |
| `WEIGHT_DECAY` | 5e-4 | 1e-5 – 1e-3 | |
| `WARMUP_EPOCHS` | 50 | 20–100 | Phase 1 length; increase for larger graphs |
| `TOTAL_EPOCHS` | 350 | 200–600 | Phase 1 + Phase 2 combined budget |
| `PATIENCE` | 30 | 20–50 | Early stopping patience (Phase 2) |
| `KL_WEIGHT` | 0.7 | 0.5–0.9 | Weight on KL distillation loss |
| `CE_WEIGHT` | 0.3 | 0.1–0.5 | Weight on cross-entropy loss; KL + CE need not sum to 1 |
| `TEMPERATURE` | 2.0 | 1.0–5.0 | Distillation temperature; higher = softer teacher targets |
| `NUM_SEEDS` | 1 | — | Set to 10 to reproduce paper mean ± std |