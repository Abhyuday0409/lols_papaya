"""
MIDAS: Struc2Vec + Dual Encoder + Clustering
=============================================
Step 2 of the MIDAS pipeline:
  1. Generate Struc2Vec structural embeddings for all nodes (cached).
  2. Train dual encoder: align node features ↔ Struc2Vec via InfoNCE
     + center loss so clusters become compact.
     - Train encoder only on train+val nodes.
     - For test nodes: use ONLY the feature encoder (no Struc2Vec).
  3. Save fused embeddings for ALL nodes → used by train_student.py.
  4. Run DBSCAN on train+val fused embeddings.
  5. Assign best teacher per cluster.
  6. Save clustering_info_seed{N}.pkl.

Requires:
    best_graphsage_actor_seed{N}.pth
    best_gat_actor_seed{N}.pth
    best_gcn_actor_seed{N}.pth

Usage:
    python clustering.py

Outputs:
    struc2vec_actor_128d.pkl          (cached Struc2Vec, reused across seeds)
    clustering_info_seed{N}.pkl       (loaded by train_student.py)
    clustering_results.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
from torch_geometric.datasets import Actor
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Struc2Vec
STRUC2VEC_DIM  = 128
WALKS_PER_NODE = 25
WALK_LENGTH    = 20

# Dual encoder
CLIP_HIDDEN    = 256
CLIP_OUT       = 128
CLIP_LR        = 1e-4
CLIP_EPOCHS    = 300
CLIP_TEMP      = 0.07

# Center loss (applied during dual encoder training)
CENTER_LOSS_WEIGHT = 0.001
CENTER_LR          = 0.05

# DBSCAN
DBSCAN_EPS         = 0.7
DBSCAN_MIN_SAMPLES = 5

NUM_SEEDS  = 1
DATA_ROOT  = "./data/Actor"
TEACHERS   = ["GraphSAGE", "GAT", "GCN"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


# ---------------------------------------------------------------------------
# GNN Teacher definitions
# ---------------------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.conv1   = SAGEConv(in_feats, hidden)
        self.conv2   = SAGEConv(hidden, num_classes)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class GAT(torch.nn.Module):
    def __init__(self, in_feats, hidden, num_classes, heads=8, dropout=0.6):
        super().__init__()
        self.conv1   = GATConv(in_feats, hidden, heads=heads, dropout=dropout)
        self.conv2   = GATConv(hidden * heads, num_classes, heads=1, dropout=dropout)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, hidden)
        self.conv2   = GCNConv(hidden, num_classes)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# ---------------------------------------------------------------------------
# Dual encoder with center loss
# ---------------------------------------------------------------------------
class DualEncoder(nn.Module):
    """
    Feature encoder  : node features  → shared latent space
    Struct encoder   : Struc2Vec       → shared latent space
    Both are aligned via InfoNCE so that structural proximity becomes
    readable from node features alone at inference.
    Center loss applied on the feature embeddings encourages tight clusters.
    """
    def __init__(self, feat_dim, struct_dim, hidden, out_dim, num_classes):
        super().__init__()
        self.feat_enc = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, out_dim),
        )
        self.struct_enc = nn.Sequential(
            nn.Linear(struct_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, out_dim),
        )
        self.classifier = nn.Linear(out_dim, num_classes)
        self.log_scale  = nn.Parameter(torch.log(torch.tensor(1.0 / CLIP_TEMP)))

    def encode_feat(self, x):
        return F.normalize(self.feat_enc(x), dim=-1)

    def encode_struct(self, s):
        return F.normalize(self.struct_enc(s), dim=-1)

    def contrastive_loss(self, zf, zs):
        logits = self.log_scale.exp() * zf @ zs.T
        labels = torch.arange(len(zf), device=zf.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2

    def classify(self, zf):
        return self.classifier(zf)


class CenterLoss(nn.Module):
    def __init__(self, num_clusters, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_clusters, feat_dim))
    def forward(self, features, labels):
        centers_batch = self.centers[labels.long()]
        return (features - centers_batch).pow(2).sum() / features.size(0)


def train_dual_encoder(encoder, node_feats, struct_feats, labels,
                       tv_mask, initial_clusters):
    """
    Train the dual encoder on train+val nodes using:
      - InfoNCE contrastive loss  (align feature ↔ Struc2Vec)
      - Cross-entropy             (task supervision)
      - Center loss               (compact clusters in feature space)

    Center loss requires initial cluster labels from a first-pass DBSCAN
    on Struc2Vec embeddings.  Outliers (label -1) are excluded from
    center loss but still contribute to InfoNCE and CE.

    At inference: ONLY encode_feat() is called.
    Struc2Vec is never used after this function returns.

    Returns fused embeddings for ALL nodes [N, CLIP_OUT], L2-normalised.
    """
    print("\n── Dual-encoder training ──")

    nf  = node_feats.to(device)     # [N, feat_dim]
    sf  = struct_feats.to(device)   # [N, struct_dim]
    y   = labels.to(device)         # [N]
    tv  = tv_mask.to(device)        # [N] bool

    # Remap cluster labels for center loss (must be 0..n_cls-1, no -1)
    valid_mask = torch.tensor(initial_clusters != -1, device=device)  # [N_tv]
    tv_indices_all = torch.where(tv)[0]                               # actual node indices

    unique_lbl = sorted(set(initial_clusters[initial_clusters != -1]))
    lbl_map    = {old: new for new, old in enumerate(unique_lbl)}
    n_cls      = len(unique_lbl)

    remapped = np.array([lbl_map.get(l, -1) for l in initial_clusters])
    # remapped[i] = -1 for outliers, 0..n_cls-1 for cluster nodes

    use_center = n_cls >= 2
    center_loss_fn = CenterLoss(n_cls, CLIP_OUT).to(device) if use_center else None
    opt_c = optim.SGD(center_loss_fn.parameters(), lr=CENTER_LR) if use_center else None

    opt = optim.Adam(encoder.parameters(), lr=CLIP_LR, weight_decay=1e-5)

    best_acc, best_enc_state = 0.0, None

    for epoch in range(1, CLIP_EPOCHS + 1):
        encoder.train()
        opt.zero_grad()
        if opt_c:
            opt_c.zero_grad()

        zf  = encoder.encode_feat(nf[tv])    # [N_tv, CLIP_OUT]
        zs  = encoder.encode_struct(sf[tv])  # [N_tv, CLIP_OUT]

        loss = (0.3 * encoder.contrastive_loss(zf, zs) +
                0.7 * F.cross_entropy(encoder.classify(zf), y[tv]))

        # Center loss on non-outlier train+val nodes
        if use_center and valid_mask.sum() > 0:
            valid_zf  = zf[valid_mask]
            valid_lbl = torch.tensor(
                remapped[initial_clusters != -1], dtype=torch.long, device=device
            )
            c_loss = center_loss_fn(valid_zf, valid_lbl)
            loss   = loss + CENTER_LOSS_WEIGHT * c_loss

        encoder.log_scale.data.clamp_(max=np.log(100))
        loss.backward()
        opt.step()

        if use_center and opt_c:
            for p in center_loss_fn.parameters():
                if p.grad is not None:
                    p.grad.data *= CENTER_LR / CENTER_LOSS_WEIGHT
            opt_c.step()

        if epoch % 50 == 0:
            encoder.eval()
            with torch.no_grad():
                zf_v = encoder.encode_feat(nf[tv])
                acc  = (encoder.classify(zf_v).argmax(1) == y[tv]).float().mean().item()
            if acc > best_acc:
                best_acc       = acc
                best_enc_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.4f}")

    encoder.load_state_dict(best_enc_state)
    encoder.eval()
    with torch.no_grad():
        # ALL nodes: feature encoder only — no Struc2Vec
        fused_all = encoder.encode_feat(nf)   # [N, CLIP_OUT], L2-normalised
    print(f"  Best acc: {best_acc:.4f}  |  fused shape: {fused_all.shape}")
    return fused_all.cpu()


# ---------------------------------------------------------------------------
# Struc2Vec
# ---------------------------------------------------------------------------
def build_nx_graph(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().cpu().numpy().tolist())
    return G


def compute_struc2vec(G, dim=128, walks_per_node=25, walk_length=20):
    print("Generating Struc2Vec walks...")
    walks = []
    nodes = list(G.nodes())
    for i, node in enumerate(nodes):
        if i % 2000 == 0:
            print(f"  node {i}/{len(nodes)}")
        deg_v = G.degree(node)
        for _ in range(walks_per_node):
            walk = [str(node)]
            cur  = node
            for _ in range(walk_length):
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                cur = min(nbrs, key=lambda u: abs(G.degree(u) - deg_v))
                walk.append(str(cur))
            walks.append(walk)

    print("Training Word2Vec on walks...")
    w2v = Word2Vec(walks, vector_size=dim, window=5, min_count=0,
                   sg=1, workers=4, epochs=5)
    embeddings = np.stack([w2v.wv[str(n)] for n in nodes])
    print(f"Struc2Vec embeddings: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Hyperparams for exact Struc2Vec (used if switching to compute_struc2vec_exact)
# ---------------------------------------------------------------------------
S2V_MAX_LAYERS  = 6       # k* cap: number of DTW comparison layers
S2V_W2V_WINDOW  = 10      # Word2Vec context window (paper uses 10)
S2V_W2V_EPOCHS  = 5
S2V_WALK_LENGTH = 80      # longer walks critical for structural context
S2V_WALKS_NODE  = 10


def compute_struc2vec_exact(G, dim=128):
    """
    Full Struc2Vec (Ribeiro et al., KDD 2017).

    Steps:
      1. For each layer k=0..k*, compute DTW distance between sorted
         degree sequences of every node pair at hop distance k.
      2. Build a weighted k*-layer multigraph M where edge (u,v) in
         layer k has weight exp(-f_k(u,v)), f_k = DTW distance.
      3. Generate biased random walks on M: at each step either stay
         in the current layer (move to a weighted neighbour) or change
         layer (up/down with equal prob).
      4. Train Word2Vec (Skip-Gram) on the walk corpus.

    Requires: pip install fastdtw
    Runtime:  O(n^2 * k* * d_avg^2) — feasible up to n~5000 on CPU.
              For larger graphs use compute_struc2vec (approximation).
    """
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    import math

    nodes = list(G.nodes())
    n     = len(nodes)
    node2idx = {v: i for i, v in enumerate(nodes)}

    # ── Step 1: degree sequences per hop ──────────────────────────────────
    print(f"  Computing degree sequences (layers=0..{S2V_MAX_LAYERS-1}) ...")
    def ring_degrees(v, k):
        """Sorted degrees of nodes at exactly hop k from v."""
        if k == 0:
            return [G.degree(v)]
        visited, frontier = {v}, {v}
        for _ in range(k):
            nxt = {nb for u in frontier for nb in G.neighbors(u)} - visited
            visited |= nxt
            frontier = nxt
        return sorted(G.degree(u) for u in frontier) if frontier else [0]

    seqs = [[ring_degrees(v, k) for v in nodes]
            for k in range(S2V_MAX_LAYERS)]

    # ── Step 2: build multigraph edge weights ──────────────────────────────
    print("  Building multigraph weights ...")
    layer_w = []   # layer_w[k][(i,j)] = exp(-DTW_k(i,j))
    for k in range(S2V_MAX_LAYERS):
        print(f"    layer {k} ...")
        w = {}
        for i in range(n):
            for j in range(i + 1, n):
                a = np.array(seqs[k][i], dtype=float).reshape(-1, 1)
                b = np.array(seqs[k][j], dtype=float).reshape(-1, 1)
                d, _ = fastdtw(a, b, dist=euclidean)
                w[(i, j)] = math.exp(-d)
        layer_w.append(w)

    # ── Step 3: biased random walks ────────────────────────────────────────
    print("  Generating random walks ...")
    walks = []
    for start_idx, start in enumerate(nodes):
        if start_idx % 500 == 0:
            print(f"    node {start_idx}/{n}")
        for _ in range(S2V_WALKS_NODE):
            walk  = [str(start)]
            cur   = node2idx[start]
            layer = 0
            for _ in range(S2V_WALK_LENGTH):
                if np.random.rand() < 0.5 or S2V_MAX_LAYERS == 1:
                    # Within-layer: weighted move to any other node
                    candidates = list(range(n))
                    candidates.remove(cur)
                    if not candidates:
                        break
                    weights = np.array([
                        layer_w[layer].get((min(cur, j), max(cur, j)), 1e-9)
                        for j in candidates
                    ])
                    weights /= weights.sum()
                    cur = candidates[np.random.choice(len(candidates), p=weights)]
                else:
                    # Layer change
                    if layer == 0:
                        layer = min(1, S2V_MAX_LAYERS - 1)
                    elif layer == S2V_MAX_LAYERS - 1:
                        layer -= 1
                    else:
                        layer += np.random.choice([-1, 1])
                walk.append(str(nodes[cur]))
            walks.append(walk)

    # ── Step 4: Word2Vec ───────────────────────────────────────────────────
    print("  Training Word2Vec ...")
    w2v = Word2Vec(walks, vector_size=dim, window=S2V_W2V_WINDOW,
                   min_count=0, sg=1, workers=4, epochs=S2V_W2V_EPOCHS)
    embeddings = np.stack([w2v.wv[str(v)] for v in nodes])
    print(f"  Exact Struc2Vec done. Shape: {embeddings.shape}")
    return embeddings

# ---------------------------------------------------------------------------
# DBSCAN helper
# ---------------------------------------------------------------------------
def run_dbscan(embeddings, eps, min_samples):
    labels     = DBSCAN(eps=eps, min_samples=min_samples,
                        n_jobs=-1).fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    print(f"  DBSCAN → {n_clusters} clusters, {n_outliers} outliers "
          f"({n_outliers / len(labels):.1%})")
    return labels, n_clusters


# ---------------------------------------------------------------------------
# Teacher loading & evaluation
# ---------------------------------------------------------------------------
def load_teachers(dataset, seed):
    cfg = {
        "GraphSAGE": dict(cls=GraphSAGE, hidden=128, dropout=0.5),
        "GAT":       dict(cls=GAT,       hidden=128, heads=8, dropout=0.6),
        "GCN":       dict(cls=GCN,       hidden=128, dropout=0.5),
    }
    models = {}
    in_f, nc = dataset.num_features, dataset.num_classes
    for name, c in cfg.items():
        m = (c["cls"](in_f, c["hidden"], nc, heads=c["heads"], dropout=c["dropout"])
             if name == "GAT"
             else c["cls"](in_f, c["hidden"], nc, dropout=c["dropout"]))
        path = f"best_{name.lower()}_actor_seed{seed}.pth"
        assert os.path.exists(path), f"Missing: {path}"
        ckpt = torch.load(path, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        models[name] = m.to(device).eval()
    return models


@torch.no_grad()
def get_predictions(models, data):
    return {name: model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()
            for name, model in models.items()}


def assign_teachers_to_clusters(cluster_labels, true_labels, predictions):
    stats  = {}
    unique = sorted(c for c in set(cluster_labels) if c != -1)
    for cid in unique:
        mask = cluster_labels == cid
        y    = true_labels[mask]
        accs = {n: (p[mask] == y).mean() for n, p in predictions.items()}
        best = max(accs, key=accs.get)
        stats[cid] = {"best_model": best, "accuracies": accs, "size": int(mask.sum())}

    outlier_mask = cluster_labels == -1
    if outlier_mask.any():
        y    = true_labels[outlier_mask]
        accs = {n: (p[outlier_mask] == y).mean() for n, p in predictions.items()}
        best = max(accs, key=accs.get)
        stats[-1] = {"best_model": best, "accuracies": accs,
                     "size": int(outlier_mask.sum())}
    return stats


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
    return dataset, data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset, data = load_data()

    train_val_mask = (data.train_mask | data.val_mask).cpu()
    tv_indices     = torch.where(train_val_mask)[0].numpy()   # shape [N_tv]
    true_labels    = data.y.cpu().numpy()                     # shape [N]

    # ── Struc2Vec (cached) ────────────────────────────────────────────────
    s2v_cache = f"struc2vec_actor_{STRUC2VEC_DIM}d.pkl"
    if os.path.exists(s2v_cache):
        print(f"Loading cached Struc2Vec from {s2v_cache}")
        with open(s2v_cache, "rb") as f:
            s2v_emb = pickle.load(f)        # [N, STRUC2VEC_DIM]
    else:
        G       = build_nx_graph(data.edge_index, data.num_nodes)
        s2v_emb = compute_struc2vec(G, dim=STRUC2VEC_DIM,
                                    walks_per_node=WALKS_PER_NODE,
                                    walk_length=WALK_LENGTH)
        with open(s2v_cache, "wb") as f:
            pickle.dump(s2v_emb, f)
        print(f"Saved Struc2Vec → {s2v_cache}")

    rows = []

    for seed in range(NUM_SEEDS):
        print(f"\n{'═'*60}")
        print(f"  Seed {seed}")
        print(f"{'═'*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # ── Teacher predictions (for cluster→teacher assignment) ──────────
        models    = load_teachers(dataset, seed)
        preds_all = get_predictions(models, data)
        # tv_preds indexed 0..N_tv-1 (sliced to train+val)
        tv_preds  = {n: p[tv_indices] for n, p in preds_all.items()}
        tv_labels = true_labels[tv_indices]            # [N_tv]

        # ── Step 1: Initial DBSCAN on raw Struc2Vec (train+val only) ──────
        # This gives initial cluster labels to seed the center loss.
        print("\nInitial DBSCAN on Struc2Vec embeddings...")
        tv_s2v         = s2v_emb[tv_indices]           # [N_tv, STRUC2VEC_DIM]
        init_clusters, _ = run_dbscan(tv_s2v, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
        # init_clusters: [N_tv], labels -1..L-1

        # ── Step 2: Train dual encoder with center loss ───────────────────
        # Struc2Vec used for training alignment only.
        # For ALL nodes (incl. test): output is encode_feat(node_features).
        encoder = DualEncoder(
            feat_dim   = dataset.num_features,
            struct_dim = STRUC2VEC_DIM,
            hidden     = CLIP_HIDDEN,
            out_dim    = CLIP_OUT,
            num_classes= dataset.num_classes,
        ).to(device)

        struct_tensor = torch.FloatTensor(s2v_emb)     # [N, STRUC2VEC_DIM]

        fused_all = train_dual_encoder(
            encoder,
            node_feats      = data.x.cpu(),            # [N, feat_dim]
            struct_feats    = struct_tensor,            # [N, STRUC2VEC_DIM]
            labels          = data.y.cpu(),             # [N]
            tv_mask         = train_val_mask,           # [N] bool
            initial_clusters= init_clusters,            # [N_tv] for center loss
        )
        # fused_all: [N, CLIP_OUT], L2-normalised, CPU
        # test nodes produced via encode_feat only — no Struc2Vec

        # ── Step 3: Final DBSCAN on fused train+val embeddings ────────────
        print("\nFinal DBSCAN on fused embeddings...")
        tv_fused       = fused_all[tv_indices].numpy()  # [N_tv, CLIP_OUT]
        final_clusters, n_clusters = run_dbscan(
            tv_fused, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
        )
        # final_clusters: [N_tv], labels -1..L-1

        # ── Step 4: Assign best teacher per cluster ────────────────────────
        cluster_stats = assign_teachers_to_clusters(
            final_clusters, tv_labels, tv_preds
        )
        best_teacher  = max(
            {n: (tv_preds[n] == tv_labels).mean() for n in TEACHERS}.items(),
            key=lambda x: x[1]
        )[0]

        # Print summary
        print(f"\nCluster assignment (seed {seed}):")
        for cid, info in sorted(cluster_stats.items()):
            tag      = " [outliers]" if cid == -1 else ""
            accs_str = "  ".join(
                f"{n}={v:.3f}{'*' if n == info['best_model'] else ''}"
                for n, v in info["accuracies"].items()
            )
            print(f"  cluster {cid:2d}{tag}  n={info['size']:4d}  {accs_str}")
        print(f"  Best overall teacher: {best_teacher}")

        # ── Save ──────────────────────────────────────────────────────────
        # fused_all contains embeddings for ALL nodes.
        # train_student.py uses fused_all[node_idx] directly — no re-encoding.
        # tv_indices maps position in final_clusters → global node index.
        out = {
            "seed":               seed,
            "train_val_indices":  tv_indices,        # [N_tv] global node IDs
            "train_val_clusters": final_clusters,    # [N_tv] DBSCAN labels
            "cluster_stats":      cluster_stats,     # {cid: {best_model,...}}
            "best_overall_teacher": best_teacher,
            "fused_embeddings":   fused_all.numpy(), # [N, CLIP_OUT] ALL nodes
            "dbscan_eps":         DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
        }
        out_path = f"clustering_info_seed{seed}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"\nSaved → {out_path}")

        rows.append({
            "seed":        seed,
            "n_clusters":  n_clusters,
            "n_outliers":  int((final_clusters == -1).sum()),
            "best_teacher": best_teacher,
        })

    df = pd.DataFrame(rows)
    df.to_csv("clustering_results.csv", index=False)
    print(f"\nSaved → clustering_results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()