import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.nn as gnn
import torch_sparse
import torch_scatter
import torch_cluster
from torch_geometric.data import (
    Data as pygData, 
    Batch as pygBatch, 
    NeighborSampler, 
    DataLoader as pygDataLoader
)
from torch_geometric.nn import MessagePassing


_vscode_souce_code_explore = False
if _vscode_souce_code_explore:
    torch_sparse.SparseTensor
    torch_sparse.coalesce
    torch_sparse.transpose
    torch_sparse.spmm
    torch_sparse.spspmm

    torch_scatter.scatter_add
    torch_scatter.gather_csr
    torch_scatter.segment_csr

    torch_cluster.random_walk

    gnn.GCNConv
    gnn.InnerProductDecoder
    gnn.GAE  # no forward, but encode(), decode(), recon_loss() and test()
    gnn.VGAE # inherence GAE, add kl_loss()
    gnn.GraphConv
    gnn.global_add_pool
    gnn.Node2Vec
    gnn.MetaPath2Vec
    
    pygData
    pygBatch
    NeighborSampler
    pygDataLoader

    pyg.torch_geometric.datasets.AMiner

#####################################################################
# SECTION DataHandling
#####################################################################
# One-graph node/link-prediction
# sampling:

def one_graph_node_prediction(n_node=100, n_edge=500, f_node=7, f_edge=9, n_class=3, sampling=(2,3,4)):
    x = torch.randn(size=(n_node, f_node))
    edge_index = torch.stack([
        torch.randint(high=n_node, size=(n_edge,)),
        torch.randint(high=n_node, size=(n_edge,)),
    ])
    edge_attr = torch.randn(size=(n_edge, f_edge))
    if n_class is None:
        y = torch.randn(size=(n_node,))
    else:
        y = torch.randint(high=n_class, size=(n_node,))
    gdata = pygData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    if not sampling:
        return gdata
    else:
        k, nlayer, batch_size = sampling
        dloader = NeighborSampler(
            gdata.edge_index, 
            sizes=[k] * nlayer,
            # **kwargs to torch DataLoader
            batch_size=batch_size,   # number of nodes in last biparti-graph dst
            shuffle=True,
            drop_last=False,
        )
        return gdata, dloader
        # for bz, node_ids, bi_graph_lst in dloader:
        #     print(bz)  # == batch_size except the last if drop_last=False
        #     print(node_ids)  # LongTensor of size (n_subsample,)
        #     print('----------------------------------------------')
        #     for bi_graph in bi_graph_lst:
        #         print(bi_graph.edge_index)
        #         print('--------')
        #     break


def one_graph__CiteSeer(task='node'):
    assert task in ('node', 'link')
    ds = pyg.datasets.Planetoid(root='/home/rjia/playground/datasets/', name='CiteSeer')    
    assert len(ds) == 1
    gdata = ds[0]
    num_edge = gdata.edge_index.shape[1]
    num_node = gdata.x.shape[0]
    xdim = gdata.x.shape[1]
    y_nclass = gdata.y.max() + 1
    print(f'[CiteSeer] One grpah \n  nun_edge = {num_edge}\n  num_node = {num_node}\n  xdim = {xdim}\n  y_nclass = {y_nclass}')
    if task == 'link':
        # change node prediciton to link prediction task
        gdata.train_mask = None
        gdata.val_mask = None
        gdata.test_mask = None
        gdata = pyg.utils.train_test_split_edges(gdata)  # inplace modify
    return gdata


#####################################################################
# Multiple-graph graph-prediction
# batching: ref to `pyg.data.DataLoader` and `Collate`

def multi_graph_graph_prediction__mutagenicity(split=False):
    root = '/home/rjia/playground/datasets/'
    ds = pyg.datasets.TUDataset(root=root, name='Mutagenicity')
    ds = ds.shuffle()
    ngraph = len(ds)
    print('Mutagenicity dataset: ngraph = ', ngraph)
    if split:
        split = int(ngraph * 0.1)
        ds_train = ds[:split]
        ds_test = ds[split:]
        dloader_train = pygDataLoader(ds_train, batch_size=4)
        dloader_test = pygDataLoader(ds_test, batch_size=4)
        return ds_train, ds_test, dloader_train, dloader_test
    else:
        dloader = pygDataLoader(ds, batch_size=4)
        return ds, dloader
    # for gdata in ds:
    #     break
    # for batch in dloader:
    #     break


def multi_graph_graph_prediction__ENZYMES():
    root = '/home/rjia/playground/datasets'
    ds = pyg.datasets.TUDataset(root=root, name='ENZYMES', use_node_attr=True)
    print('ENZYMES dataset: #graph = ', len(ds))
    dloader = pygDataLoader(ds, batch_size=32, shuffle=True)
    return ds, dloader


def multi_graph_graph_prediction__FRANKENSTEIN():
    root = '/home/rjia/playground/datasets'
    ds = pyg.datasets.TUDataset(root=root, name='FRANKENSTEIN')
    print(ds)
    dloader = pygDataLoader(ds, batch_size=32, shuffle=True)
    return ds, dloader


#####################################################################
# SECTION MessagePassing
#####################################################################

class NodeEdgeConv(MessagePassing):
    """GraphConv with edge info
    """
    def __init__(self, xdim, edim, new_xdim, new_edim):
        super().__init__(aggr=None, flow='source_to_target', node_dim=0)
        self.lin_node = nn.Linear(xdim * 2 + edim, new_xdim)
        self.lin_edge = nn.Linear(xdim * 2 + edim, new_edim)

    def forward(self, x, edge_attr, edge_index):
        # propagate(edge_index, size=None, **kwargs)
        out_x, out_e = self.propagate(edge_index, None, x=x, edge_attr=edge_attr)
        print('out_x = ', out_x.shape)
        print('out_e = ', out_e.shape)
        return out_x, out_e

    def message(self, x_j, x_i, edge_attr):
        print('--------> calling message()...')
        print('x_j.shape = ', x_j.shape)
        print('x_i.shape = ', x_i.shape)
        print('edge_attr.shape = ', edge_attr.shape)
        concated_input = torch.cat([x_i, edge_attr, x_j], dim=1)
        msg_e = self.lin_edge(concated_input)
        msg_x = self.lin_node(concated_input)
        return msg_x, msg_e
    
    def aggregate(self, msg_outputs, edge_index):
        print('--------> calling aggregate()...')
        msg_x, msg_e = msg_outputs
        edge_index_i, edge_index_j = edge_index
        agg_x = torch_scatter.scatter_add(msg_x, edge_index_j, dim=0)
        agg_e = msg_e
        return agg_x, agg_e

    def update(self, agg_outputs, x, edge_attr):
        print('--------> calling update()...')
        agg_x, agg_e = agg_outputs
        return agg_x, agg_e


def test_message_passing():
    # data
    ds, dloader = multi_graph_graph_prediction__mutagenicity()
    for gdata in ds:
        break
    for batch in dloader:
        break
    
    # hparam
    hdim = 64
    nlayer = 3
    # data-related fields
    xdim = gdata.x.shape[1]
    edim = gdata.edge_attr.shape[1]

    # model (one layer)
    layer = NodeEdgeConv(xdim, edim, hdim, hdim)
    
    # forward
    use_batch = True
    if use_batch:
        x = batch.x
        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
    else:
        x = gdata.x
        edge_attr = gdata.edge_attr
        edge_index = gdata.edge_index
    
    for _ in range(nlayer):
        x, edge_attr = layer(x=x, edge_attr=edge_attr, edge_index=edge_index)
    
    # inspector
    m = layer
    print('----------- inspector ------------')
    print('m.inspector.params: ', m.inspector.params)
    print('m.__user_args__: ', m.__user_args__)
    print('m.__fused_user_args__: ', m.__fused_user_args__)
    print('m.__class__.__bases__: ', m.__class__.__bases__)