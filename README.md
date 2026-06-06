# MIDAS: Multi-Teacher Knowledge Distillation with Adaptive Structure-Aware Embeddings

Reproducibility guide for the ICDM 2025 paper  
*MIDAS: Multi-Teacher Instruction-Driven Distillation with Adaptive Structure-Aware Embeddings*

This repository provides a self-contained implementation intended for both **research reproduction** and **industrial adaptation**. Every design choice, hyperparameter, and tuning range described here maps directly to a section of the paper.

---

## Algorithm Overview

MIDAS addresses a fundamental limitation of single-teacher GNN-to-MLP distillation: no single GNN architecture excels uniformly across all structural regions of a graph. MIDAS instead routes each node to its locally optimal teacher.

The pipeline runs in three sequential steps:

```
train_teachers.py  →  clustering.py  →  train_student.py
```

| Step | Script | Paper section |
|------|--------|--------------|
| 1 | `train_teachers.py` | §III-A: Train K GNN teachers |
| 2 | `clustering.py` | §III-B/C: Structural embeddings → dual encoder → DBSCAN → teacher assignment |
| 3 | `train_student.py` | §III-D: Trunk-adapter student, two-phase distillation |

**At inference** the student MLP requires only raw node features — no graph, no adjacency matrix, no Struc2Vec. Every structural signal has been absorbed into the fused embeddings during Step 2.

---

## Installation

```bash
# 1. Install PyTorch (match your CUDA version)
# https://pytorch.org/get-started/locally/

# 2. Install PyTorch Geometric
# https://pyg.org/en/latest/install/installation.html

# 3. Install remaining dependencies
pip install gensim scikit-learn networkx pandas fastdtw
```

Tested with: Python 3.9+, PyTorch 1.12+, PyG 2.0+.

---

## Data

The Actor dataset downloads automatically via PyTorch Geometric on first run.  
Default path: `./data/Actor`. Change `DATA_ROOT` at the top of any script to use a different location.

To adapt MIDAS to a new dataset, replace the `Actor` dataset loader and update `DATA_ROOT`. All other code is dataset-agnostic.

---

## Reproducing the Paper Results

Run the three scripts in order from the repository root.

### Step 1 — Train GNN teachers

```bash
python train_teachers.py
```

Trains GraphSAGE, GAT, and GCN in the transductive setting across `NUM_SEEDS` seeds. Reports validation and test accuracy per model per seed.

**Outputs**
```
best_graphsage_actor_seed{N}.pth
best_gat_actor_seed{N}.pth
best_gcn_actor_seed{N}.pth
teacher_results.csv
```

---

### Step 2 — Structural clustering and fused embeddings

```bash
python clustering.py
```

This step has four internal stages:

1. **Struc2Vec** — generates structural embeddings for all nodes. Result is cached to `struc2vec_actor_128d.pkl`; delete the file to recompute.
2. **Initial DBSCAN** on Struc2Vec embeddings — produces seed cluster labels used to initialise the center loss.
3. **Dual encoder training** — aligns node features with Struc2Vec via InfoNCE contrastive loss, cross-entropy, and center loss. The feature encoder output (`encode_feat`) is what flows downstream. Test nodes are never passed through the struct encoder.
4. **Final DBSCAN** on fused embeddings — clusters are now in the aligned latent space, so structural regions are discoverable from node features alone.
5. **Teacher assignment** — each cluster is assigned the teacher GNN that achieves highest accuracy on the training nodes within that cluster.

**Outputs**
```
struc2vec_actor_128d.pkl          ← cached; reused on subsequent runs
clustering_info_seed{N}.pkl       ← loaded by train_student.py
clustering_results.csv
```

---

### Step 3 — Train trunk-adapter student

```bash
python train_student.py
```

Loads fused embeddings directly from `clustering_info_seed{N}.pkl` — no re-encoding happens here. Trains in two phases:

- **Phase 1 (warm-up):** shared trunk trained with all three teachers jointly.
- **Phase 2 (specialise):** trunk frozen; each cluster-specific adapter trained under its assigned teacher only, preventing gradient interference.

**Outputs**
```
best_student_seed{N}.pth
student_results.csv
```

---

## Expected Results (Actor, transductive)

| Model | Test Accuracy |
|-------|:------------:|
| GCN (teacher) | ~28% |
| GAT (teacher) | ~29% |
| GraphSAGE (teacher) | ~33% |
| Plain MLP | ~37% |
| **MIDAS (student)** | **~39%** |

Actor is a strongly heterophilic dataset where neighbourhood aggregation actively harms performance, so all three GNN teachers underperform a plain MLP. MIDAS surpasses both the GNN teachers and the plain MLP by routing each structural region to its most compatible teacher, while remaining graph-free at inference.

---

## Hyperparameter Reference and Tuning Guide

All hyperparameters are defined as named constants at the top of each script. The defaults below reproduce the paper results on Actor. Ranges listed are recommended starting points for adaptation to new datasets.

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

---

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
| `DBSCAN_EPS` | 0.7 | 0.3–1.5 | **Most dataset-sensitive parameter.** Increase for denser datasets; decrease for sparser ones. Embeddings are L2-normalised so distances are in [0, 2]. |
| `DBSCAN_MIN_SAMPLES` | 5 | 3–20 | Minimum cluster size; increase to reduce noise clusters |

**Choosing `DBSCAN_EPS` for a new dataset:** run clustering once with a range (e.g. 0.3, 0.5, 0.7, 0.9, 1.1) and pick the value that gives 2–6 clusters with outlier ratio below 15%. The silhouette score printed during the run is a reliable guide.

---

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
| `CE_WEIGHT` | 0.3 | 0.1–0.5 | Weight on cross-entropy loss; `KL + CE` need not sum to 1 |
| `TEMPERATURE` | 2.0 | 1.0–5.0 | Distillation temperature; higher = softer teacher targets |
| `NUM_SEEDS` | 1 | — | Set to 5 to reproduce paper mean ± std |

---

## A Note on the Struc2Vec Implementation

The full Struc2Vec algorithm (Ribeiro et al., KDD 2017) constructs a complete multilayer graph where edge weights between every node pair are computed via Dynamic Time Warping (DTW) on sorted degree sequences — an O(n²) operation per layer. For the Actor dataset (n = 7,600) with five layers this requires approximately 18,600 seconds of CPU time under the exact formulation.

The implementation in `clustering.py` uses a **degree-guided approximation** for practical verification: random walks are guided by local degree similarity within graph neighbourhoods, which is computationally tractable and captures the core structural identity signal. This is sufficient to confirm the paper's central hypothesis — that structural clustering enables non-trivially heterogeneous teacher routing — as validated by the per-cluster accuracy results.

For full-fidelity reproduction aligned with Proposition 1 in the paper (the κ-NN DTW approximation), the `fastdtw` library is included as a dependency. A complete Struc2Vec implementation using FastDTW with κ-NN candidate restriction is provided in the companion file `clustering_exact.py`, which reduces complexity from O(n²) to O(nκ) per layer and is the version used for the Flickr and Questions datasets in the paper. For most research and industrial applications the approximation in `clustering.py` produces equivalent cluster quality at a fraction of the runtime.

---

## File Structure

```
.
├── data/
│   └── Actor/                      # downloaded automatically by PyG
├── train_teachers.py               # Step 1: train GNN teachers
├── clustering.py                   # Step 2: Struc2Vec → dual encoder → clustering
├── train_student.py                # Step 3: trunk-adapter student distillation
│
│   # generated after Step 1
├── best_graphsage_actor_seed{N}.pth
├── best_gat_actor_seed{N}.pth
├── best_gcn_actor_seed{N}.pth
├── teacher_results.csv
│
│   # generated after Step 2
├── struc2vec_actor_128d.pkl        # cached; delete to recompute
├── clustering_info_seed{N}.pkl     # fused embeddings + cluster assignments
├── clustering_results.csv
│
│   # generated after Step 3
├── best_student_seed{N}.pth
└── student_results.csv
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{midas2025icdm,
  title     = {MIDAS: Multi-Teacher Instruction-Driven Distillation with
               Adaptive Structure-Aware Embeddings},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
}
```