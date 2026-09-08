"""
Graph Neural Network Recommendation System (LightGCN)
========================================================
Implements LightGCN (He et al., SIGIR 2020) — one of the most effective and
widely-used GNN architectures for collaborative filtering / recommendation.

Core idea:
  - Build a bipartite user-item graph from interaction data (e.g. clicks,
    purchases, ratings >= threshold).
  - Propagate embeddings over the graph using simplified graph convolution
    (no feature transformation, no nonlinearity — just neighborhood
    aggregation), which LightGCN showed works better than heavier GCNs
    for this task.
  - Combine embeddings from all propagation layers (layer combination).
  - Train with BPR (Bayesian Personalized Ranking) loss on implicit
    feedback (positive vs. sampled negative items).
  - Recommend top-K items per user via dot-product scoring.

Dependencies: only torch + numpy (no torch_geometric needed — the graph
convolution is implemented directly via sparse matrix multiplication so
this runs anywhere PyTorch runs).

    pip install torch numpy --break-system-packages
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


# ----------------------------------------------------------------------
# 1. Synthetic dataset (replace with your real user-item interactions)
# ----------------------------------------------------------------------
def generate_synthetic_data(n_users=500, n_items=300, n_interactions=8000, seed=42):
    """
    Returns a list of (user_id, item_id) positive interaction pairs.
    In practice, load this from your logs / ratings table instead.
    """
    rng = np.random.default_rng(seed)

    # give users latent "taste clusters" so the graph has real structure
    n_clusters = 8
    user_cluster = rng.integers(0, n_clusters, size=n_users)
    item_cluster = rng.integers(0, n_clusters, size=n_items)

    pairs = set()
    while len(pairs) < n_interactions:
        u = rng.integers(0, n_users)
        # 80% of the time, prefer an item from the user's own cluster
        if rng.random() < 0.8:
            candidates = np.where(item_cluster == user_cluster[u])[0]
            if len(candidates) == 0:
                candidates = np.arange(n_items)
        else:
            candidates = np.arange(n_items)
        i = rng.choice(candidates)
        pairs.add((int(u), int(i)))

    return list(pairs), n_users, n_items


def train_test_split(pairs, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n_test = int(len(pairs) * test_ratio)
    return pairs[n_test:], pairs[:n_test]


# ----------------------------------------------------------------------
# 2. Build normalized bipartite adjacency matrix
# ----------------------------------------------------------------------
def build_norm_adj(train_pairs, n_users, n_items, device):
    """
    Builds the symmetrically-normalized adjacency matrix of the bipartite
    user-item graph, as a single sparse (n_users+n_items) square matrix:

        A_norm = D^(-1/2) * A * D^(-1/2)

    where A is the bipartite adjacency [[0, R], [R^T, 0]].
    """
    n_nodes = n_users + n_items

    users = np.array([u for u, i in train_pairs])
    items = np.array([i for u, i in train_pairs]) + n_users  # offset items

    # edges in both directions (undirected graph)
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    vals = np.ones(len(rows), dtype=np.float32)

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values = torch.tensor(vals)
    A = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

    # degree normalization D^-1/2 A D^-1/2
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg = torch.clamp(deg, min=1.0)
    deg_inv_sqrt = torch.pow(deg, -0.5)

    row, col = A.indices()
    norm_vals = deg_inv_sqrt[row] * A.values() * deg_inv_sqrt[col]
    A_norm = torch.sparse_coo_tensor(A.indices(), norm_vals, (n_nodes, n_nodes)).coalesce()

    return A_norm.to(device)


# ----------------------------------------------------------------------
# 3. LightGCN model
# ----------------------------------------------------------------------
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_users + n_items, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self, A_norm):
        """
        Returns final user and item embeddings after multi-layer
        neighborhood propagation with layer combination (mean of all
        layers, including layer 0 — this is what makes LightGCN 'light':
        no weight matrices, no activations between layers).
        """
        all_emb = self.embedding.weight
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(A_norm, all_emb)
            embs.append(all_emb)

        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        user_emb, item_emb = torch.split(final_emb, [self.n_users, self.n_items])
        return user_emb, item_emb

    def forward(self, A_norm, users, pos_items, neg_items):
        user_emb, item_emb = self.propagate(A_norm)

        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]

        pos_scores = (u_e * pos_e).sum(dim=-1)
        neg_scores = (u_e * neg_e).sum(dim=-1)

        # L2 regularization on the *raw* (layer-0) embeddings, standard for BPR
        reg = (
            self.embedding(users).norm(2).pow(2)
            + self.embedding(pos_items + self.n_users).norm(2).pow(2)
            + self.embedding(neg_items + self.n_users).norm(2).pow(2)
        ) / users.shape[0]

        return pos_scores, neg_scores, reg


# ----------------------------------------------------------------------
# 4. BPR loss + negative sampling
# ----------------------------------------------------------------------
def bpr_loss(pos_scores, neg_scores, reg, reg_weight=1e-4):
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    return loss + reg_weight * reg


def sample_batch(train_pairs, user_pos_items, n_items, batch_size, device):
    idx = np.random.randint(0, len(train_pairs), size=batch_size)
    users = np.array([train_pairs[i][0] for i in idx])
    pos_items = np.array([train_pairs[i][1] for i in idx])

    neg_items = np.empty(batch_size, dtype=np.int64)
    for k, u in enumerate(users):
        while True:
            cand = np.random.randint(0, n_items)
            if cand not in user_pos_items[u]:
                neg_items[k] = cand
                break

    return (
        torch.tensor(users, dtype=torch.long, device=device),
        torch.tensor(pos_items, dtype=torch.long, device=device),
        torch.tensor(neg_items, dtype=torch.long, device=device),
    )


# ----------------------------------------------------------------------
# 5. Evaluation: Recall@K and NDCG@K
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, A_norm, user_pos_items_train, user_pos_items_test, n_items, k=10, device="cpu"):
    model.eval()
    user_emb, item_emb = model.propagate(A_norm)
    scores = user_emb @ item_emb.T  # [n_users, n_items]

    recalls, ndcgs = [], []
    for u, test_items in user_pos_items_test.items():
        if not test_items:
            continue
        u_scores = scores[u].clone()
        # mask out items the user already interacted with in training
        for seen in user_pos_items_train.get(u, []):
            u_scores[seen] = -1e9

        top_k = torch.topk(u_scores, k).indices.cpu().numpy()
        hits = np.isin(top_k, list(test_items))

        recall = hits.sum() / min(len(test_items), k)
        recalls.append(recall)

        dcg = (hits / np.log2(np.arange(2, k + 2))).sum()
        idcg = (1.0 / np.log2(np.arange(2, min(len(test_items), k) + 2))).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


# ----------------------------------------------------------------------
# 6. Training loop
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs, n_users, n_items = generate_synthetic_data()
    train_pairs, test_pairs = train_test_split(pairs)
    print(f"Users: {n_users}, Items: {n_items}, Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    user_pos_items_train = defaultdict(set)
    for u, i in train_pairs:
        user_pos_items_train[u].add(i)

    user_pos_items_test = defaultdict(set)
    for u, i in test_pairs:
        user_pos_items_test[u].add(i)

    A_norm = build_norm_adj(train_pairs, n_users, n_items, device)

    model = LightGCN(n_users, n_items, embed_dim=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 60
    batch_size = 1024
    steps_per_epoch = max(1, len(train_pairs) // batch_size)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            users, pos_items, neg_items = sample_batch(
                train_pairs, user_pos_items_train, n_items, batch_size, device
            )
            pos_scores, neg_scores, reg = model(A_norm, users, pos_items, neg_items)
            loss = bpr_loss(pos_scores, neg_scores, reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            recall, ndcg = evaluate(
                model, A_norm, user_pos_items_train, user_pos_items_test,
                n_items, k=10, device=device,
            )
            print(
                f"Epoch {epoch:3d} | loss {epoch_loss / steps_per_epoch:.4f} "
                f"| Recall@10 {recall:.4f} | NDCG@10 {ndcg:.4f}"
            )

    # ------------------------------------------------------------------
    # 7. Example: get top-10 recommendations for a single user
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.propagate(A_norm)
        example_user = 0
        scores = (user_emb[example_user] @ item_emb.T).clone()
        for seen in user_pos_items_train.get(example_user, []):
            scores[seen] = -1e9
        top10 = torch.topk(scores, 10).indices.cpu().numpy()
        print(f"\nTop-10 recommendations for user {example_user}: {top10.tolist()}")


if __name__ == "__main__":
    main()