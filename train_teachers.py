"""
MIDAS: Train GNN Teacher Models on Actor Dataset
=================================================
Trains three GNN teachers (GraphSAGE, GAT, GCN) in the transductive setting.
Checkpoints are saved and used downstream by clustering.py and train_student.py.

Usage:
python train_teachers.py

Outputs:
best_graphsage_actor_seed{N}.pth
best_gat_actor_seed{N}.pth
best_gcn_actor_seed{N}.pth
teacher_results.csv
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Actor
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128
DROPOUT_SAGE = 0.5
DROPOUT_GCN = 0.5
DROPOUT_GAT = 0.6
GAT_HEADS = 8
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20
NUM_SEEDS = 1
DATA_ROOT = "./data/Actor"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
def __init__(self, in_feats, hidden, num_classes, dropout=0.5):
super().__init__()
self.conv1 = SAGEConv(in_feats, hidden)
self.conv2 = SAGEConv(hidden, num_classes)
self.dropout = dropout

def forward(self, x, edge_index):
x = F.relu(self.conv1(x, edge_index))
x = F.dropout(x, p=self.dropout, training=self.training)
return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
def __init__(self, in_feats, hidden, num_classes, heads=8, dropout=0.6):
super().__init__()
self.conv1 = GATConv(in_feats, hidden, heads=heads, dropout=dropout)
self.conv2 = GATConv(hidden * heads, num_classes, heads=1, dropout=dropout)
self.dropout = dropout

def forward(self, x, edge_index):
x = F.dropout(x, p=self.dropout, training=self.training)
x = F.elu(self.conv1(x, edge_index))
x = F.dropout(x, p=self.dropout, training=self.training)
return self.conv2(x, edge_index)


class GCN(torch.nn.Module):
def __init__(self, in_feats, hidden, num_classes, dropout=0.5):
super().__init__()
self.conv1 = GCNConv(in_feats, hidden)
self.conv2 = GCNConv(hidden, num_classes)
self.dropout = dropout

def forward(self, x, edge_index):
x = F.relu(self.conv1(x, edge_index))
x = F.dropout(x, p=self.dropout, training=self.training)
return self.conv2(x, edge_index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data():
dataset = Actor(root=DATA_ROOT)
data = dataset[0].to(device)
if data.train_mask.dim() > 1:
data.train_mask = data.train_mask[:, 0]
data.val_mask = data.val_mask[:, 0]
data.test_mask = data.test_mask[:, 0]
if data.y.dim() > 1:
data.y = data.y.argmax(dim=1)
return dataset, data


def make_model(name, in_feats, num_classes):
if name == "GraphSAGE":
return GraphSAGE(in_feats, HIDDEN_DIM, num_classes, DROPOUT_SAGE).to(device)
if name == "GAT":
return GAT(in_feats, HIDDEN_DIM, num_classes, GAT_HEADS, DROPOUT_GAT).to(device)
if name == "GCN":
return GCN(in_feats, HIDDEN_DIM, num_classes, DROPOUT_GCN).to(device)
raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def accuracy(model, data, mask):
model.eval()
pred = model(data.x, data.edge_index)[mask].argmax(dim=1)
return (pred == data.y[mask]).float().mean().item()


def train_model(name, data, in_feats, num_classes, seed):
model = make_model(name, in_feats, num_classes)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val, best_test = 0.0, 0.0
no_improve = 0

for epoch in range(1, EPOCHS + 1):
model.train()
optimizer.zero_grad()
out = model(data.x, data.edge_index)
loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
loss.backward()
optimizer.step()

val_acc = accuracy(model, data, data.val_mask)
test_acc = accuracy(model, data, data.test_mask)

if val_acc > best_val:
best_val, best_test = val_acc, test_acc
no_improve = 0
torch.save(
{"model_state_dict": model.state_dict(),
"val_acc": val_acc, "test_acc": test_acc, "seed": seed},
f"best_{name.lower()}_actor_seed{seed}.pth",
)
else:
no_improve += 1

if no_improve >= PATIENCE:
break

return best_val, best_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
dataset, data = load_data()
in_feats = dataset.num_features
num_classes = dataset.num_classes

print(f"Nodes: {data.num_nodes} | Edges: {data.num_edges} | "
f"Features: {in_feats} | Classes: {num_classes}")
print(f"Train: {data.train_mask.sum().item()} | "
f"Val: {data.val_mask.sum().item()} | "
f"Test: {data.test_mask.sum().item()}\n")

rows = []
for seed in range(NUM_SEEDS):
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed(seed)

print(f"─── Seed {seed} " + "─" * 50)
for name in ["GraphSAGE", "GAT", "GCN"]:
val_acc, test_acc = train_model(name, data, in_feats, num_classes, seed)
print(f" {name:<12} val={val_acc:.4f} test={test_acc:.4f}")
rows.append({"seed": seed, "model": name,
"val_acc": val_acc, "test_acc": test_acc})

# Summary
df = pd.DataFrame(rows)
df.to_csv("teacher_results.csv", index=False)
print("\n═══ Summary (mean ± std over seeds) ═══")
for name in ["GraphSAGE", "GAT", "GCN"]:
sub = df[df["model"] == name]
print(f" {name:<12} "
f"val={sub['val_acc'].mean():.4f}±{sub['val_acc'].std():.4f} "
f"test={sub['test_acc'].mean():.4f}±{sub['test_acc'].std():.4f}")
print("\nSaved → teacher_results.csv")


if __name__ == "__main__":
main()