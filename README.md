# MIDAS: Multi-Teacher Knowledge Distillation with Adaptive Structure-Aware Embeddings

Reproducibility guide for the ICDM 2026 paper  
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
@inproceedings{midas2026icdm,
  title     = {MIDAS: Multi-Teacher Instruction-Driven Distillation with
               Adaptive Structure-Aware Embeddings},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
}
```