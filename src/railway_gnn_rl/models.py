"""GNN encoder, MLP ablation encoder and the candidate-scoring policy.

Both encoders map a (batched) DispatchGraph to
    h: [total_nodes, H]  per-node embeddings
    g: [num_graphs, H]   one global embedding per graph.
The shared `DispatchPolicy` head scores every legal action k of a decision:

    z_k = [ h[cand_node_k] | h[cand_next_node_k] | edge_attr[cand_edge_k] | cand_feat_k | g ]
    score_k = MLP(z_k)            (a scalar logit; softmax over the K actions = policy)

GNN: h0 = ReLU(W_in x); then L layers h <- h + ReLU(GINEConv(h, edge_index, edge_attr)),
     GINEConv: h_v' = MLP((1 + eps) h_v + sum_{u->v} ReLU(h_u + W_e e_uv)),  eps = 0;
     g = mean over the nodes of the graph.  After L layers a node "sees" the
     state of all nodes and sections up to L hops away.
MLP ablation: h = node-wise MLP of x (no neighbour information, same depth),
     g = MLP(flatten(x) | flatten(edge_attr)) - it receives the same inputs as
     the GNN but has no message passing / relational inductive bias.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_mean_pool

from .graph_obs import EDGE_FEATURES, NODE_FEATURES, TRAIN_FEATURES


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class GNNEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList(
            GINEConv(_mlp(hidden, hidden, hidden), edge_dim=edge_dim) for _ in range(layers))

    def forward(self, batch):
        h = torch.relu(self.inp(batch.x))
        for conv in self.convs:
            h = h + torch.relu(conv(h, batch.edge_index, batch.edge_attr))
        return h, global_mean_pool(h, batch.batch)


class MLPEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 32, layers: int = 2,
                 n_nodes: int = 7, n_edges: int = 16):
        super().__init__()
        self.inp = nn.Linear(node_dim, hidden)
        self.node_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.glob = _mlp(n_nodes * node_dim + n_edges * edge_dim, hidden // 2, hidden)

    def forward(self, batch):
        h = torch.relu(self.inp(batch.x))
        for lin in self.node_layers:
            h = h + torch.relu(lin(h))
        n_graphs = int(batch.num_graphs)
        flat = torch.cat([batch.x.view(n_graphs, -1), batch.edge_attr.view(n_graphs, -1)], dim=1)
        return h, self.glob(flat)


class DispatchPolicy(nn.Module):
    def __init__(self, encoder: nn.Module, edge_dim: int, train_dim: int, hidden: int = 32):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(3 * hidden + edge_dim + train_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, batch) -> torch.Tensor:
        """Flat tensor of action logits, graph after graph ([sum_k K_k])."""
        h, g = self.encoder(batch)
        num_cands = torch.as_tensor(batch.num_cands, dtype=torch.long).view(-1)
        cand_graph = torch.repeat_interleave(torch.arange(num_cands.numel()), num_cands)
        z = torch.cat([h[batch.cand_node], h[batch.cand_next_node],
                       batch.edge_attr[batch.cand_edge], batch.cand_feat, g[cand_graph]], dim=1)
        return self.head(z).squeeze(-1)


def make_policy(kind: str, hidden: int = 32, layers: int = 2) -> DispatchPolicy:
    nd, ed, td = len(NODE_FEATURES), len(EDGE_FEATURES), len(TRAIN_FEATURES)
    if kind == "gnn":
        enc = GNNEncoder(nd, ed, hidden, layers)
    elif kind == "mlp":
        enc = MLPEncoder(nd, ed, hidden, layers)
    else:
        raise ValueError(kind)
    return DispatchPolicy(enc, ed, td, hidden)
