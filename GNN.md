# GNN

## surveys

*A Comprehensive Survey Of Graph Neutral Network*

*A Comprehensive Survey Of Graph Embedding*

*Deep Learning On Graphs - A Survey*

*GNN in recomendation system A Survey*

## papers

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

### Node2vec, Metapath2vec

TODO: papers

PyG impl:
- train:
    + batch is a pair of `(pos_rw, neg_rw)`, `pos_rw` is sampled by random walk, `neg_rw` is random picked from all nodes.
        * random_walk: `rw = random_walk(adj, starting_nodes, walk_length, p, q)`, then rolling to get walks with sequence length of `context_size`
        * metapath: ??? PyG impl: same as random_walk, ie treat hetero graph as homo graph, but constraint walking through predefined metapath.
    + model: only one trainable prameter: `nn.Embedding(n_node, emb_dim)`
    + loss: "is_context", ie. the rest nodes is the "context" of the first node. `emb_distance = InnerProduct(emb(seq[0]), emb(seq[j])) for j in 1~n_walk`, `loss = BCELoss(sigmoid(emb_distance), is_context)`
- eval: node embedding as input, node y as target, run a simple LR using sklearn.


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


### Sampling graph - DGL

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


### Sampling graph - PyG

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

- Used for many-graphs, (usually) graph-level-prediciton task.
- concate `x`, `y`, diag concate adj matrix,  ie. concate coo `edge_index`

more details ref to [扒源码-PyG](#pyg-扒源码)


### [DGL paper] *DGL: a grpah-centric, highly-performant package for graph networks*


### [PyG paper] *Fast Graph Representation Learning with PyTorch Geometric*


### PyG 扒源码

基于 1.6.3

to-solve: import error when multiple version of torch installed

[pyg source code explore](./codes/gnn/pyg_source_code.py)

#### PyG Class Hierarchy related to data handling

- `Data` <- `Batch`
- `torch_Dataset` <- `pyg_Dataset` <- `InMemoryDataset` <- `TUDataset` etc.
- `torch_DataLoader` <- `pyg_DataLoader`
- `iter(pyg_Dataset)` yield `Data`
- `iter(pyg_DataLoader)` yield `Batch`

基础数据结构 `Data`, 包含 图拓扑结构 `edge_index`, 节点特征 `x`, 边特征 `edeg_features`，以及标签 `y` 标签可以是 graph/node/edge级别的，shape随意
```pyton
Data(
    x: Tensor[N, F_node],
    edge_index: LongTensor[2, E]
    y: Tensor[Any],
    edge_attr: Tensor[E, F_edge],
    ...
)
```

对 list of `Data` 做 batching 得到 `Batch`, Batch 继承 Data, 增加`.batch`属性，`.batch: LongTensor = [0,0,0,1,1,2,2...42]` 用于记录每个节点原来属于哪个图，可被用于eg. `global_add_pool(x, batch)`，Batch对象通常使用静态构造函数`.from_data_list()`构造，这个函数实现了垂直合并`x`,`y`,对角线合并`edge_index`的逻辑。

pygDataset 继承 torch.Dataset, 其定义遵循“有许多图、而非单一个大图”这种情况。 `__len__ == num_graph`, `__getitem__ == the i-th graph`, eg. `ds[0]`为第一个graph, 迭代 pyg`Dataset` 对象得到`Data`对象

pygDataLoader 继承 torch.DataLoader, 唯一区别是 collate_fn, 若`samples`为`Data`对象，则调用`Batch.from_data_list()`

对于“单一一个大图”的情况，需要做某种 sampling 形成训练模型所需的一个"batch". 
- `NeighborSampler` 继承 torch.DataLoader，接收`edge_index`, 迭代产生三元组`(bz, node_ids, edge_index_lst)`. 详见 [Sampling Graph - PyG](#sampling-graph-pyg)
- 基于 random walk 的采样方式， eg. `Node2vec`, `Metapath2vec`, Module实现了`loader()`方法, 返回 torch_DataLoader 对象, 迭代产生二元组`(pos_rw, neg_rw)`

#### Message Passing

propogate 依次调用 message, aggregate, update。为了节省内存，可以把 message和aggregate 合并成 message_and_aggregate。

messag/aggregate/update需要的输入通过kwargs的变量名表达，被inspector解析。可以有如下几种输入
- 后缀带 `_i`, `_j` 的，由 `__lift__` 生成。eg. `x_i = x[edge_index[0]]; x_j = x[edge_index[1]]`
- 从传入 propagate() 的 kwargs 中找。这使我们可以传入任何对象到 message/aggregate/update
- 一些特殊的args: `adj_t`, `edge_index[_i|_j]`, `ptr` `size[_i|_j]` ...

```python
class MessagePassing:
    def propagate(self, edge_index, size, **kwargs):
        # msg/aggr/update_kwargs __collect__ from inspector
        if 'implement message_and_aggregate':
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            out = self.update(out, **upddate_kwargs)
        else:
            out = self.message(**msg_kwargs)
            out = self.aggregate(out, **aggr_kwargs)
            out = self.update(out, **update_kwargs)
```

#### jit

##### `torch.jit.script/trace`

refs:
- [Introduction](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [end2end example](https://pytorch.org/tutorials/advanced/cpp_export.html)
- [doc](https://pytorch.org/docs/stable/jit.html#disable-jit-for-debugging)
- [TorchScript不支持的操作](https://pytorch.org/docs/stable/jit_unsupported.html)

Summary:

- `torch.jit.trace`实际运行`example_inputs`生成IR，无法正确处理不同input走到不同control-flow的情况
- 优先使用`torch.jit.script`
    + control flow
    + for loop

```python
torch.jit.script(model)
torch.jit.trace(model, example_inputs)
```

Why / What is TorchScript?
> TorchScript is a way to create serializable and optimizable models from PyTorch code.  Any TorchScript program can be saved from a Python process and loaded in a process **where there is no Python dependency.**


##### PyG `m.jittable()` then `torch.jit.script()`

TODO: Expected a value of type 'Tensor' for argument 'edge_index' but instead found type 'Optional[Tensor]'.

`MessagePassing`实现了`jittable()`方法，它通过分析class定义，生成一个新的、jittable的class代码（这个class实际上继承原class），并new一个新的实例返回。新的class代码从一个jinja模板填空生成，可以在 `/tmp/$USER_pyg_jit/tmpxxxxx.py` 中找到生成的新class的代码，与模板对比。

`jittable()`源码：
```python
class MessagePassing(torch.nn.Module):
    def jittable(self, typing: Optional[str] = None):
        # ... get templete args ...
        with open('message_passing.jinja') as f:
            template = Template(f.read())
        jit_module_repr = template.render(  ...args...  )
        # Instantiate a class from the rendered JIT module representation.
        cls = class_from_module_repr(cls_name, jit_module_repr)
        module = cls.__new__(cls)
        module.__dict__ = self.__dict__.copy()
        module.jittable = None
        return module
```

生成的 Jiatable class 代码
```python
from torch_geometric.nn.conv.message_passing import *
from my_file import *   # 还是会用到你自己原来的代码

# 假设原来的类叫做 NodeEdgeConv
class NodeEdgeConvJittable_272c79(NodeEdgeConv):
    def __check_input__(...)
    def __lift__(...)
    def __collect__(...)

    # 依次调用 message/aggregate/update
    # 但这三个函数没在这里重新实现，会去找原始类`NodeEdgeConv`中的定义
    # 所以要确保你自己写的这三个函数也是jittable的
    def propagate(self, edge_index, x, edge_attr, size=None):
        # kwargs ...
        out = self.message(edge_attr=kwargs.edge_attr, x_j=kwargs.x_j, x_i=kwargs.x_i)
        out = self.aggregate(out, edge_index=kwargs.edge_index)
        return self.update(out, x=kwargs.x, edge_attr=kwargs.edge_attr)

    # 完全是原来的类的code抄过来，所以要自己确保forward函数是jittable的
    def forward(self, x, edge_attr, edge_index):
        out_x, out_e = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out_x, out_e
```


#### util pkgs: torch_scatter/sparse/cluster

先行知识：稀疏矩阵存储

- COO: edge_index_i/j
- CSR: (rowptr, col, value)

torch_scatter

GroupBy-Agg functions:
- `scatter(x, index)`  reuduce: sum,mean,min,max,mul,std,logsumexp,softmax,logsoftmax
- `segment_csr(x, rowptr, reduce='sum')` segment_sum/mean/min/max_csr
- `segment_coo(x, edge_index_i, reduce='sum')` segment_sum/mean/min/max_coo

Gather functions, used in `__lift__`
- `gather_csr`
- `gather_coo`

torch_sparse: 定义 `SparseTensor`, `SparseStorage` 以及一系列稀疏矩阵算子: Coalesce, Transpose, spmm, spspmm

torch_cluster: 一些 graph cluster 算法的高效实现， eg. random_walk sampling in node2vec


### Quiver：based on PyG

[设计简介](https://github.com/quiver-team/torch-quiver/blob/main/docs/Introduction_cn.md)

解决的问题：Sampling one-large graph 特别慢，是整个训练pipeline的性能瓶颈。包括两个部分：
1. 采样nodes `node_id, edge_indexs = next(NeighborSampler())`
2. slicing node features `x[node_id]`

1_ 采样：
- 现有的方式
    + 在CPU采样：本身性能低，另外随着进程数变多cpu占用也相应变多，这就不scalable
    + 在GPU采样：显存大小限制 （Q: 一亿个边内存占用也才 1.4 G，这个得是billion级别#edge才会成为问题）
- 解决方案：CUDA特性 **Unified Virtual Address, UVA**, (since CUDA 4.0). 简单的说就是 host memroy 和 device memory在一个统一的地址空间，对程序员透明
    + TODO: 它和 `cudaMemoryManaged` 什么关系？

2_ Feature Collection (Slicing x)
- 现有方式的问题：GPU上显存显然不够，CPU上问题同上，不scalable
- 解决方案：cache hot data in GPU
    + 重要假设：nodes在edges中的出现服从幂律分布
    + 先行知识：访存速度 GPU > GPU p2p with NVlink > pined host memory
    + Cache stategy: 
        + 单卡 20% hot data to GPU
        + 多卡 根据NVLink拓扑结构确定 replica or p2p
    + 好处：随着GPU数量增加，总GPU显存也在增加，有NVLink情况下可获得**超线性加速比**


