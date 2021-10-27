import torch
import torch.nn as nn
import torch_scatter
#from torch_geometric.nn.conv import GCNConv  # for go to source in vscode

from torch import FloatTensor, LongTensor
from typing import Tuple

AijTensor = Tuple[LongTensor, LongTensor]
NodeFTensor = FloatTensor


def generate_batch(n_node=20, n_edge=80, n_feature=100):
    """Generate one graph with COO adj matrix and node features
    Could also interpreted as a batch of graph.
    """
    x = torch.randn(size=(n_node, n_feature))
    adj = (
        torch.randint(high=n_node, size=(n_edge,)),
        torch.randint(high=n_node, size=(n_edge,)),
    )
    return x, adj


class GNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.ff = torch.nn.Linear(2 * self.in_features, self.out_features)
    
    def forward(self, x: NodeFTensor, adj: AijTensor):
        # this is wrong impl, adj must be dense
        Ai, Aj = adj
        X_i = x[Ai]
        X_j = x[Aj]
        X_ij = torch.cat([X_i, X_j], dim=1)
        E_msg = self.ff(X_ij)
        agg_xi = torch_scatter.scatter(E_msg, Ai, dim=0, reduce='sum')
        return agg_xi


class DiffPool(nn.Module):
    def __init__(self, in_features, emb_features, next_n_nodes):
        super().__init__()
        self.in_features = in_features
        self.emb_features = emb_features
        self.next_n_nodes = next_n_nodes
        self.gnn_emb = GNN(self.in_features, self.emb_features)
        self.gnn_pool = GNN(self.in_features, self.next_n_nodes)  # `next_n_node` must be fixed?

    def forward(self, x, adj):
        z = self.gnn_emb(x, adj)   # (n_nodes, emb_features)
        s = self.gnn_pool(x, adj)  # TODO batch mask
        s = torch.softmax(s, dim=1)
        next_x = s.T @ z
        next_adj = s.T @ adj @ s   # TODO: adj is soft not hard, if Sign(next_adj) it's not differenciable
        pass


# X, A = generate_random_graph()

# Ai, Aj = A

# # indexing
# X_i = X[Ai]
# X_j = X[Aj]

# for k in range(n_edge):
#     assert torch.allclose(X_i[k], X[Ai[k]])
#     assert torch.allclose(X_j[k], X[Aj[k]])

# # concate
# X_ij = torch.cat([X_i, X_j], dim=1)
# assert X_ij.shape == (n_edge, 2 * n_features)

# # edge_msg
# gnn = torch.nn.Linear(2 * n_features, 64)
# E_msg = gnn(X_ij)
# assert E_msg.shape == (n_edge, 64)

# # aggregate
# agg_xi = torch_scatter.scatter(E_msg, Ai, dim=0, reduce='sum')
# assert agg_xi.shape == (n_node, 64)

# for i in range(n_node):
#     assert torch.allclose(agg_xi[i], E_msg[Ai == i].sum(dim=0))


