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

*DGL: a grpah-centric, highly-performant package for graph networks*

