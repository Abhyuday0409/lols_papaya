"""
MIDAS: Train Trunk-Adapter Student MLP
========================================
Step 3 of the MIDAS pipeline (runs after train_teachers.py + clustering.py).

Loads pre-computed fused embeddings from clustering_info_seed{N}.pkl and
trains the trunk-adapter student via two-phase distillation:
  Phase 1: warm-up — shared trunk sees all teachers jointly.
  Phase 2: specialise — trunk frozen, each adapter sees only its
           assigned teacher (cluster-conditional supervision).

At inference: route each node to its cluster adapter.
Outliers → nearest centroid (Algorithm 1 in the paper).

Requires:
    clustering_info_seed{N}.pkl       (from clustering.py)
    best_graphsage_actor_seed{N}.pth
    best_gat_actor_seed{N}.pth
    best_gcn_actor_seed{N}.pth

Usage:
    python train_student.py

Outputs:
    best_student_seed{N}.pth
    student_results.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Actor
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRUNK_HIDDEN   = 512
ADAPTER_HIDDEN = 128
DROPOUT        = 0.3
LR             = 1e-3          # was 5e-2 — that caused the warm-up collapse
WEIGHT_DECAY   = 5e-4
WARMUP_EPOCHS  = 50
TOTAL_EPOCHS   = 350
PATIENCE       = 40
KL_WEIGHT      = 0.7
CE_WEIGHT      = 0.3
TEMPERATURE    = 2.0
# Fallback weight: how much to weight the globally best teacher signal
# on outlier/unassigned nodes in Phase 2.  Keeps all nodes receiving
# gradient even when cluster coverage is incomplete.
FALLBACK_WEIGHT = 0.3

NUM_SEEDS = 1
DATA_ROOT = "./data/Actor"
TEACHERS  = ["GraphSAGE", "GAT", "GCN"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


# ---------------------------------------------------------------------------
# Teacher GNN definitions
# ---------------------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_f, h, nc, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_f, h);  self.conv2 = SAGEConv(h, nc)
        self.dropout = dropout
    def forward(self, x, ei):
        return self.conv2(F.dropout(F.relu(self.conv1(x, ei)),
                                    p=self.dropout, training=self.training), ei)

class GAT(torch.nn.Module):
    def __init__(self, in_f, h, nc, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_f, h, heads=heads, dropout=dropout)
        self.conv2 = GATConv(h * heads, nc, heads=1, dropout=dropout)
        self.dropout = dropout
    def forward(self, x, ei):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(F.dropout(F.elu(self.conv1(x, ei)),
                                    p=self.dropout, training=self.training), ei)

class GCN(torch.nn.Module):
    def __init__(self, in_f, h, nc, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_f, h);  self.conv2 = GCNConv(h, nc)
        self.dropout = dropout
    def forward(self, x, ei):
        return self.conv2(F.dropout(F.relu(self.conv1(x, ei)),
                                    p=self.dropout, training=self.training), ei)


# ---------------------------------------------------------------------------
# Student: shared trunk + per-cluster adapters
# ---------------------------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, trunk_dim, hidden, num_classes, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(trunk_dim, hidden)
        self.bn  = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout
    def forward(self, x):
        return self.fc2(F.dropout(F.relu(self.bn(self.fc1(x))),
                                  p=self.dropout, training=self.training))


class StudentMLP(nn.Module):
    def __init__(self, in_feats, trunk_h, adapter_h, num_classes, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_feats, trunk_h), nn.BatchNorm1d(trunk_h),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(trunk_h, trunk_h), nn.BatchNorm1d(trunk_h),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.adapters = nn.ModuleDict(
            {t: Adapter(trunk_h, adapter_h, num_classes, dropout) for t in TEACHERS}
        )
    def forward(self, x, teacher=None):
        h = self.trunk(x)
        if teacher is not None:
            return self.adapters[teacher](h)
        return {t: a(h) for t, a in self.adapters.items()}
    def freeze_trunk(self):
        for p in self.trunk.parameters():
            p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------
def load_teachers(dataset, seed):
    cfg = {
        "GraphSAGE": dict(cls=GraphSAGE, h=128, dropout=0.5),
        "GAT":       dict(cls=GAT,       h=128, heads=8, dropout=0.6),
        "GCN":       dict(cls=GCN,       h=128, dropout=0.5),
    }
    models = {}
    for name, c in cfg.items():
        m = (c["cls"](dataset.num_features, c["h"], dataset.num_classes,
                      heads=c["heads"], dropout=c["dropout"])
             if name == "GAT"
             else c["cls"](dataset.num_features, c["h"], dataset.num_classes,
                           dropout=c["dropout"]))
        path = f"best_{name.lower()}_actor_seed{seed}.pth"
        assert os.path.exists(path), f"Missing: {path}"
        ckpt = torch.load(path, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        models[name] = m.to(device).eval()
    return models


@torch.no_grad()
def get_teacher_logits(models, data):
    return {n: m(data.x, data.edge_index) for n, m in models.items()}


# ---------------------------------------------------------------------------
# Build cluster masks from saved clustering info
# ---------------------------------------------------------------------------
def build_cluster_masks(cluster_stats, tv_indices, tv_clusters, num_nodes):
    """
    cluster_stats : {cid: {best_model, ...}}  — from clustering_info pkl
    tv_indices    : [N_tv] global node IDs of train+val nodes
    tv_clusters   : [N_tv] DBSCAN labels (may include -1 for outliers)

    Returns dict  : teacher → bool tensor [N] on device
    Outlier nodes (label -1) are left False in all masks;
    they are routed at inference by nearest centroid.
    """
    masks = {t: torch.zeros(num_nodes, dtype=torch.bool, device=device)
             for t in TEACHERS}
    for local_i, global_idx in enumerate(tv_indices):
        cid = tv_clusters[local_i]
        if cid == -1:
            continue   # outlier — handled by nearest centroid
        best_t = cluster_stats[cid]["best_model"]
        masks[best_t][global_idx] = True
    return masks


def build_centroids(cluster_masks, fused):
    """
    Mean fused embedding per teacher cluster.
    Used to route outlier nodes: k* = argmin_m ||z_v - c_m||  (Algorithm 1).
    """
    fused_d = fused.to(device)
    cents   = []
    for t in TEACHERS:
        m = cluster_masks[t]
        cents.append(fused_d[m].mean(dim=0) if m.sum() > 0
                     else torch.zeros(fused_d.shape[1], device=device))
    return torch.stack(cents)   # [K, CLIP_OUT]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def warmup_step(student, teacher_logits, data, train_mask, optimizer):
    student.train()
    optimizer.zero_grad()
    all_out = student(data.x)
    loss = sum(
        KL_WEIGHT * F.kl_div(
            F.log_softmax(all_out[t][train_mask] / TEMPERATURE, dim=1),
            F.softmax(teacher_logits[t][train_mask] / TEMPERATURE, dim=1),
            reduction="batchmean",
        ) * TEMPERATURE ** 2
        + CE_WEIGHT * F.cross_entropy(all_out[t][train_mask], data.y[train_mask])
        for t in TEACHERS
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def specialise_step(student, teacher_logits, cluster_masks,
                    best_teacher, data, train_mask, optimizer):
    """
    Phase 2 loss — two parts:

    Part A (cluster-routed):
        Each adapter receives gradients only from its assigned teacher
        on its assigned nodes.  This is the hard-routing specialisation.

    Part B (fallback):
        Nodes not assigned to any cluster (outliers) still receive
        gradient via the globally best teacher on all adapters, weighted
        down by FALLBACK_WEIGHT.  Without this, outlier nodes contribute
        zero gradient and the adapters see too little data.
    """
    student.train()
    optimizer.zero_grad()
    all_out = student(data.x)
    loss    = torch.tensor(0.0, device=device)

    # Part A: cluster-routed hard supervision
    all_assigned = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    for t in TEACHERS:
        mask = (train_mask & cluster_masks[t].cpu()).to(device)
        if mask.sum() == 0:
            continue
        all_assigned |= mask
        loss = loss + (
            KL_WEIGHT * F.kl_div(
                F.log_softmax(all_out[t][mask] / TEMPERATURE, dim=1),
                F.softmax(teacher_logits[t][mask] / TEMPERATURE, dim=1),
                reduction="batchmean",
            ) * TEMPERATURE ** 2
            + CE_WEIGHT * F.cross_entropy(all_out[t][mask], data.y[mask])
        )

    # Part B: fallback — outlier/unassigned train nodes via best teacher
    outlier_mask = (train_mask.to(device) & ~all_assigned)
    if outlier_mask.sum() > 0:
        for t in TEACHERS:
            loss = loss + FALLBACK_WEIGHT * (
                KL_WEIGHT * F.kl_div(
                    F.log_softmax(all_out[t][outlier_mask] / TEMPERATURE, dim=1),
                    F.softmax(teacher_logits[best_teacher][outlier_mask] / TEMPERATURE, dim=1),
                    reduction="batchmean",
                ) * TEMPERATURE ** 2
                + CE_WEIGHT * F.cross_entropy(
                    all_out[t][outlier_mask], data.y[outlier_mask])
            )

    if loss.requires_grad:
        loss.backward()
        optimizer.step()
    return loss.item()


@torch.no_grad()
def warmup_eval(student, data, mask):
    student.eval()
    all_out = student(data.x)
    stacked = torch.stack([all_out[t][mask].argmax(1) for t in TEACHERS])
    preds   = torch.mode(stacked, dim=0).values
    return (preds == data.y[mask]).float().mean().item()


@torch.no_grad()
def evaluate(student, cluster_masks, centroids, fused, data, mask):
    """
    Route each node:
      assigned cluster → that cluster's adapter
      outlier          → nearest centroid adapter  (Algorithm 1)
    """
    student.eval()
    mask_d   = mask.to(device)
    all_out  = student(data.x)
    indices  = torch.where(mask_d)[0]
    preds    = torch.zeros(len(indices), dtype=torch.long, device=device)
    fused_d  = fused.to(device)

    for i, idx in enumerate(indices):
        # Find assigned teacher from cluster masks
        t = next((t for t in TEACHERS if cluster_masks[t][idx]), None)
        if t is None:
            # Outlier: nearest centroid
            dists = torch.norm(centroids - fused_d[idx].unsqueeze(0), dim=1)
            t     = TEACHERS[dists.argmin().item()]
        preds[i] = all_out[t][idx].argmax()

    return (preds == data.y[mask_d]).float().mean().item()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    dataset = Actor(root=DATA_ROOT)
    data    = dataset[0].to(device)
    if data.train_mask.dim() > 1:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask   = data.val_mask[:, 0]
        data.test_mask  = data.test_mask[:, 0]
    if data.y.dim() > 1:
        data.y = data.y.argmax(dim=1)
    data.train_mask = data.train_mask.bool().cpu()
    data.val_mask   = data.val_mask.bool().cpu()
    data.test_mask  = data.test_mask.bool().cpu()
    return dataset, data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset, data = load_data()
    original_x    = data.x.clone()

    rows = []

    for seed in range(NUM_SEEDS):
        print(f"\n{'═'*60}")
        print(f"  Seed {seed}")
        print(f"{'═'*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # ── Load clustering info ──────────────────────────────────────────
        cfile = f"clustering_info_seed{seed}.pkl"
        assert os.path.exists(cfile), f"Missing: {cfile} — run clustering.py first"
        with open(cfile, "rb") as f:
            info = pickle.load(f)

        cluster_stats  = info["cluster_stats"]
        tv_indices     = info["train_val_indices"]     # [N_tv] global node IDs
        tv_clusters    = info["train_val_clusters"]    # [N_tv] DBSCAN labels
        best_teacher   = info["best_overall_teacher"]
        # fused_embeddings: [N, CLIP_OUT] for ALL nodes, produced by clustering.py
        fused = torch.FloatTensor(info["fused_embeddings"])  # [N, CLIP_OUT]

        print(f"\nLoaded fused embeddings: {fused.shape}  "
              f"(clusters: {len([c for c in cluster_stats if c != -1])}, "
              f"best teacher: {best_teacher})")

        # ── Build cluster masks ───────────────────────────────────────────
        cluster_masks = build_cluster_masks(
            cluster_stats, tv_indices, tv_clusters, data.num_nodes
        )
        centroids = build_centroids(cluster_masks, fused)  # [K, CLIP_OUT]

        assigned = sum(cluster_masks[t][data.train_mask].sum().item()
                       for t in TEACHERS)
        total    = data.train_mask.sum().item()
        print(f"Cluster assignments: {assigned}/{total} train nodes  "
              f"({total - assigned} outliers → nearest centroid)")

        # ── Teacher logits (use original node features + graph) ───────────
        data.x  = original_x.to(device)
        teachers     = load_teachers(dataset, seed)
        teacher_logits = get_teacher_logits(teachers, data)

        # ── Switch to fused features for student ──────────────────────────
        # Student sees graph-free structure-aware features from clustering.py.
        # Index directly: fused[node_idx] is the embedding for that node.
        data.x = fused.to(device)   # [N, CLIP_OUT]

        # ── Student ───────────────────────────────────────────────────────
        clip_out = fused.shape[1]
        student  = StudentMLP(
            in_feats   = clip_out,
            trunk_h    = TRUNK_HIDDEN,
            adapter_h  = ADAPTER_HIDDEN,
            num_classes= dataset.num_classes,
            dropout    = DROPOUT,
        ).to(device)

        # Phase 1: warm-up — trunk + all adapters, all teachers jointly
        # LR=1e-3 with cosine decay prevents the collapse seen at LR=5e-2
        print("\n── Phase 1: warm-up ──")
        opt1     = optim.Adam(student.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
        sched1   = optim.lr_scheduler.CosineAnnealingLR(
                       opt1, T_max=WARMUP_EPOCHS, eta_min=LR * 0.1)
        best_warmup_val   = 0.0
        best_warmup_state = None
        for epoch in range(1, WARMUP_EPOCHS + 1):
            loss = warmup_step(student, teacher_logits, data,
                               data.train_mask.to(device), opt1)
            sched1.step()
            if epoch % 10 == 0:
                val = warmup_eval(student, data, data.val_mask.to(device))
                if val > best_warmup_val:
                    best_warmup_val   = val
                    best_warmup_state = {k: v.clone()
                                         for k, v in student.state_dict().items()}
                print(f"  epoch {epoch:3d}  loss={loss:.4f}  val={val:.4f}  "
                      f"lr={sched1.get_last_lr()[0]:.2e}")
        # Restore best warm-up checkpoint before Phase 2
        if best_warmup_state is not None:
            student.load_state_dict(best_warmup_state)
        print(f"  Best warm-up val: {best_warmup_val:.4f}")

        # Phase 2: specialise — trunk frozen, adapters see assigned teachers
        # Outlier/unassigned nodes get FALLBACK_WEIGHT * best_teacher signal
        print("\n── Phase 2: specialise ──")
        student.freeze_trunk()
        opt2   = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()),
                            lr=LR * 0.5, weight_decay=WEIGHT_DECAY)
        sched2 = optim.lr_scheduler.CosineAnnealingLR(
                     opt2, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.01)

        best_val, best_test, no_improve = 0.0, 0.0, 0
        for epoch in range(WARMUP_EPOCHS + 1, TOTAL_EPOCHS + 1):
            loss = specialise_step(student, teacher_logits, cluster_masks,
                                   best_teacher, data, data.train_mask, opt2)
            sched2.step()
            val  = evaluate(student, cluster_masks, centroids, fused,
                            data, data.val_mask)
            test = evaluate(student, cluster_masks, centroids, fused,
                            data, data.test_mask)

            if val > best_val:
                best_val, best_test = val, test
                no_improve = 0
                torch.save(student.state_dict(), f"best_student_seed{seed}.pth")
            else:
                no_improve += 1

            if epoch % 20 == 0:
                print(f"  epoch {epoch:3d}  loss={loss:.4f}  "
                      f"val={val:.4f}  test={test:.4f}")

            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

        print(f"\n  Seed {seed}  →  val={best_val:.4f}  test={best_test:.4f}")
        rows.append({"seed": seed, "val_acc": best_val, "test_acc": best_test})

    # ── Summary ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv("student_results.csv", index=False)
    print(f"\n{'═'*60}")
    print("  MIDAS Final Results")
    print(f"{'═'*60}")
    print(df.to_string(index=False))
    print(f"\n  Test: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    print("\nSaved → student_results.csv")


if __name__ == "__main__":
    main()