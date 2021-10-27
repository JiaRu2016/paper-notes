import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, BoolTensor
from typing import Tuple, List, Union

AijTensor = Tuple[LongTensor, LongTensor]
AdjMatrixTensor = Union[BoolTensor, LongTensor, FloatTensor]
NodeFTensor = FloatTensor


def generate_batch_dense(batch_num_nodes, n_feature) -> Tuple[NodeFTensor, BoolTensor]:
    n = sum(batch_num_nodes)
    x = torch.randn(size=(n, n_feature))
    p = 0.3
    adj = torch.rand(size=(n, n)) < p
    mask = torch.block_diag(*[torch.ones(_n, _n, dtype=torch.BoolTensor) for _n in batch_num_nodes])
    return x, adj[mask]


def generate_batch_sparse(batch_num_nodes, n_feature) -> Tuple[NodeFTensor, AijTensor]:
    n_lst = batch_num_nodes
    cum_n_lst = torch.cumsum(torch.LongTensor(n_lst), dim=0).tolist()
    n_edge_lst = [int(n / 3) for n in n_lst]
    adj = (
        torch.cat([torch.randint(low=cum_n, high=cum_n + n, size=(e,)) for cum_n, n, e in zip(cum_n_lst, n_lst, n_edge_lst)]),
        torch.cat([torch.randint(low=cum_n, high=cum_n + n, size=(e,)) for cum_n, n, e in zip(cum_n_lst, n_lst, n_edge_lst)]),
    )
    x = torch.rand(size=(sum(n_lst), n_feature))
    return x, adj


class TopKPooling(nn.Module):
    def __init__(self, k: float, in_features):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.proj = nn.Linear(self.in_features, 1, bias=False)
    
    def forward(self, x, adj, batch_num_nodes):
        if batch_num_nodes is None:
            return self.__forward_record(x, adj)
        else:
            return self.__forward_batch(x, adj, batch_num_nodes)

    def __forward_batch(self, x: NodeFTensor, adj: AijTensor, batch_num_nodes: List[int]):
        # only need to implement `batched_top_k(score, n_lst, k_lst)`
        pass

    def __forward_record(self, x: NodeFTensor, adj: AijTensor):
        score = self.proj(x).view(-1)  # (n, )
        _, idx = torch.topk(score, int(self.k * x.shape[0]), dim=0)  # (n', )
        x_sliced = x[idx]
        adj_sliced = adj_filter(adj, idx)
        return x_sliced, adj_sliced

def adj_filter(adj: AijTensor, idx: LongTensor) -> AijTensor:
    Ai, Aj = adj
    mask_i = torch_isin(Ai, idx)
    mask_j = torch_isin(Aj, idx)
    mask = mask_i & mask_j
    return Ai[mask], Aj[mask]

def torch_isin(arr, idx):
    return (arr.unsqueeze(-1) == idx).any(-1)


def test():
    batch_num_nodes = [10, 10, 10]
    n_feature = 5
    x, adj = generate_batch_sparse(batch_num_nodes, n_feature)
    print(x.shape)
    print(adj[0])
    print(adj[1])
    print('=========================')

    m = TopKPooling(k=0.7, in_features=n_feature)
    with torch.no_grad():
        x_next, adj_next = m(x, adj, None)
    print(x_next.shape)
    print(adj_next[0])
    print(adj_next[1])



if __name__ == '__main__':
    test()