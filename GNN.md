# GNN

## surveys

*A Comprehensive Survey Of Graph Neutral Network*

*A Comprehensive Survey Of Graph Embedding*

*Deep Learning On Graphs - A Survey*

*GNN in recomendation system A Survey*

## papers

*node2vec: Scalable Feature Learning for Networks*

GCN. *Semi-supervised classificication with graph convolutional networks*

GraphSAGE. *Inductive Representation Learning on Large Graphs*

GAT. *Graph Attenion Netoworks*

HAN. *Heterogeneous Graph Attention Network*

*struc2vec: Learning Node Representations from Structural Identity*. see video

NRI

PinSAGE

SGC, simplified GCN

Fast GCN

LINE

*Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View*

*mulit-hop Attention GNN*

*Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting*

*Hypergraph Neutral Network*

### VGAE
ref:
- [towards data science - VAEs](https://towardsdatascience.com、understanding-variational-autoencoders-vaes-f70510919f73). Explains why **Variational**: 
    + (related to GAN). For **generate** meaningfull hidden z, estimate **prob distribution of z** conditional on input $p(z|x)$, instead of a single point-estimation of z
    + suppress overfit. 
- [PyG doc - GAE and VGAE](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html)
- soruce code of `pyg.nn.GAE`, `pyg.nn.VGAE`

notes:
- `reparemterize(mu, sigma)` sample $z$ from $N(0,1)$ and let `z = mu + sigma * z` where `mu`, `sigma` is output of encoder net which keeps gradient-calculating.
- `kl_loss`. details in my notebook
$$
- 2 * KL(N(\mu, \sigma^2)||N(0,1)) = ... = 1 + 2 \log(\sigma) - \mu^2 - \sigma^2
$$

pseudo-code of GAE and VGAE
```python
# gdata. One-graph-link-prediction task. eg. CiteSeer
(   
    train_pos_links, 
    train_neg_links, 
    test_pos_links, 
    test_neg_links,
) = \
    get_train_pos_neg_links(gdata.edge_index)

# layers
encoder = GCN_Encoder()
decoder = InnerProduct_Decoder()
reconstruct_loss = BCELoss()
kl_loss = lambda mu, sigma: 1 + 2 * log(sigma) - sigm**2 - mu**2

# forward
if not Variational:
    z = encoder(x, edge_index)
else:
    mu, sigma = encoder(x, edge_index)
    z = reparameterize(mu, sigma)
pred_links = decoder(z, train_pos_links, train_neg_links)
loss = reconstruct_loss(pred_links, train_pos_links, train_neg_links)
if Variational:
    loss += kl_loss(mu, sigma)
loss.backward()
```


### [DiffPool] *Hierarchical Graph Representation Learning with Differentiable Pooling* 

- Apply "pooling" layer on graph. 
- Layer inptut is $A^l$, $X^l$, use two GNN to learn assignment matrix $S^l$ and embedding features $Z^l$. Layer output $X^{l+1}$ aggregate embedding features $Z^l$ using assignment matrix $S^l$, and new adj matrix is $S^TAS$
$$
    S^l = GNN_{l,pool}(A^l, X^l) \\
    Z^l = GNN_{l,emb}(A^l, X^l) \\
    X^{l+1} = (S^l)^T Z^l \\
    A^{l+1} = (S^l)^T A^l S^l  \\
$$
- It's very like attention, $\alpha$ is the assignment matrix $S$ and $V$ is embedding $Z$, each is learned by a per-layer GNN
- disadvantage (found after trying code implemention)
    + not scalable. Adj matrix is soft thus must use dense, even if we sign() to sparse adj it's not differenciable. That's why this paper titled  "Diff"Pool
    + not scalable if "batch" a graph. Adj matrix size is `(bz * n) ^ 2`, and we also need to calculate `batch_mask` according to input/ouput number_of_nodes/clusters
    + number of nodes/cluster must be pre-specified. Just like image size must be fixed, here even if input n_nodes could be different, each layer's number of cluster must be same.


### [TopKPooling] *Graph U-Nets*, *Towards Sparse Hierarchical Graph Classiﬁers*
- TopK pooling: slicing graph, `score = x @ w`, slicing index `idx = topk(score)`, next layer x = `(x * sigmoid(score))[idx]`
- unpooling: `x^{l+1} = distribute(0, x^l, idx)`
- unet, same as pix2pix u-net, with `GCN` inplace of `Conv`
- Code impl: `torch.topk()`, `torch.isin()`
- 会不会出现：filter node 之后 Adj matrix all zero ?


## Frameworks


### Sampling graph

[dgl: Customizing neighborhood sampler](https://docs.dgl.ai/en/0.6.x/guide/minibatch-custom-sampler.html#guide-minibatch-customizing-neighborhood-sampler)

用户需实现`BlockSampler.ample_fronterior()`, 该接口输入原始图`g`和上层`seed_nodes`，返回 sampled neighbor，返回类型是一个（二分）图 `DGLHeteroGraph`。注意每一层都有各自不同的 src-dst二分图，dgl称之为 *MessageFlowGraph*, MFG, 即 model forward 每一层用的是不同的graph. 注意src中的nodes即使在原图中有连接，在当前block也不做信息传递，如果两方都不在seed中的话。

sampler对象传入DataLoader, dataloader就会产生 blocks, which is list of (biparti)graph 而非一个单一的graph

blocks传入model.forward(), model.forward中定义哪个层用哪个block

```python
sampler = MyAwesomeSampler(**kwargs)
dataloader = dgl.dataloading.NodeDataLoader(..., sampler, ...)

for blocks in dataloader:
    x = blocks[0].srcdata
    y = blocks[-1].dstdata
    yh = model(blocks, x)
    loss = loss_fn(yh, y)

# where
class MyAwesomeSampler(BlockSampler):
    def sample_frontier(self, block_id, g, seed_nodes):
        return frontier_g   # it's a (biparti)DGLHeteroGraph

class MyModel(nn.Module):
    def forward(self, blocks, x):
        # block is List[DGLBlock], DGLBlock is subclass of DGLHeteroGraph
        x = self.conv1(blocks[0], x)
        x =  self.conv1(blocks[1], x)
        return x
```


PyG is similar, ref to [NeighborSampler](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html?highlight=NeighborSampler#torch_geometric.loader.NeighborSampler)

```python
n_node = 100
n_edge = 500

edge_index = torch.stack([
    torch.randint(n_node, size=(n_edge,)),
    torch.randint(n_node, size=(n_edge,)),
])
x = torch.rand(size=(n_node, 64))
y = torch.randint(10, size=(n_node,))

gdata = pyg.data.Data(
    x=x,
    edge_index=edge_index,
    edge_attr=None,
    y=y,
)

k = 2
nlayer = 3
batch_size = 4

train_dloader = pyg.data.NeighborSampler(
    gdata.edge_index, 
    sizes=[k] * nlayer,
    # **kwargs to torch DataLoader
    batch_size=batch_size,   # number of nodes in last biparti-graph dst
    shuffle=True,
    drop_last=False,
)

for bz, node_ids, bi_graph_lst in train_dloader:
    print(bz)  # == batch_size except the last if drop_last=False
    print(node_ids)  # LongTensor of size (n_subsample,)
    print('----------------------------------------------')
    for bi_graph in bi_graph_lst:
        print(bi_graph.edge_index)
        print('--------')
    break
```


### Minibatch - PyG


### [DGL paper] *DGL: a grpah-centric, highly-performant package for graph networks*


### [PyG paper] *Fast Graph Representation Learning with PyTorch Geometric*