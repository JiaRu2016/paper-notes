# paper-notes
ML, DL paper reading notes


## optimization / training diagram

*Training ImageNet in 1 Hour* by Facebook

*Visualizing the Loss Landscape of Neural Nets*

CyclicalLR *Cyclical Learning Rates for Training Neural Networks*

*Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*

*CurriculumLearning* 2009_ICML


## distributed DL

hogwild! *Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent*

Parameter Server. *Scaling Distributed Machine Learning with the Parameter Server*, *Communication Efﬁcient Distributed Machine Learning with the Parameter Server*


## DL Framework and CS-related

BP impl. CSE599W

*PyTorch Distributed- Experiences on Accelerating Data Parallel Training* 看视频更好，overlapping compute and communication 原理和 pytorch 实现细节

GPipe *Efficient Training of Giant Neural Networks using Pipeline Parallelism* 以及pytorch实现 *torchgpipe: On-the-ﬂy Pipeline Parallelism for Training Giant Models*

*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*

混合精度训练 *mixed precision training*

为什么砍了计算量推理性能还是不变？可能跟访存有关 *Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*

## NLP / seq models

[seq2seq machine translation] `code/nlp/seq2seq.py`

- padding. `BucketIterator` minimize total num of padding, by batching similar seq_len records together
- decoder is step-by-step, use *teacher_forcing* with a prob for some token

实现这个最初是因为 自己实现GNN遇到“同一个batch node个数不同”的问题，想参考一下NLP中类似问题如何处理，这里用了padding的方法。回到GNN的问题，一般就两种思路：
1. flatten, ie. batch small grpah to one large unconnected garph. 
    - This works for the most common case that graph struct is invarient, and is `torch_geometric`'s way
    - But if we need to change graph struct through layers, eg. GraphPooling, should tackle to not generate edge accross originaly different graphs, maybe use something like batch_mask? yes! Ref to torch_geometric impl of  `dense_diff_pool` (use `mask: BoolTensor (bz, max_num_nodes)`) and `topk` (use `batch: LongTensor [0,0,1,1,1,2,2...9]`)
2. padding && pass `padding_mask`
    - `LSTM` do not consider anything like padding_mask, we can consider this in loss function `CrossEntropyLoss(igore_index=PADDING_IDX)`
    - (TODO) `TransformerEncoder` accept padding mask: `src_key_padding_mask`


## RL

### awesome

- DQN. *Playing Atari with Deep Reinforcement Learning*    
- DQN + target Q net. *Human-level control through deep reinforcement learning*      
- Policy Gradient. *Policy Gradient Methods for RL with Function Approximation* 主要是推导得到了 Policy gradient Themorm    
- A3C. *Asynchronous Methods for Deep Reinforcement Learning*   
- Duel Q network. *Dueling Network Architectures for Deep Reinforcement Learning*    
- *high dimentional continguous control using GAE*

### not so owesome

*Action Branching Architectures for Deep Reinforcement Learning* 解决action是多维的问题，这个方法正常人都能想出来。 based on dualing q network. 

### interesting applications

suphx *Suphx: Mastering Mahjong with Deep Reinforcement Learning*   麻将AI by MSRA. 看视频就ok

- Encode tiles as 4 * 34 matrix
- model: 5 models combine with descition flow。 Models are if_吃/碰/杠 and 丢牌模型.
    + mdoel input: D * 34 * 1, D contains private tiles, open hands, history, manaully features etc.
    + model output: 34 * 1 for discard model; scaler for chi/pong/kong model.
    + 3 * 1 Conv with 256 channels repeat 50x skip connected.
- training: supervised learning using human players action as label, then self-play with the trained models as policy.
- trick 1: Oracle guiding. use full infomation as teacher 常见套路
- trick 2: global reward predictor as critic. 考虑风险偏好下的reward?
- trick 3: online "finetune" to priavate tiles at hand


DouZero *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*  斗地主

- Encode card as 4 * 15 matrix
- state: history moves encode to (T, 4, 15) tensor
- action: leagal movement encode to (4, 15) matrix
- model: for each leagal movement ie. action, `Concate(a, LSTM(s)) | MLP(6,512) | scaler`, output is state action value `Q(s,a)`
- use MC with DNN
    + why MC? long horizon and sparse reward
    + why not DQN? large and variable action space, `max_a Q(s,a)` is computational expensive
    + why not policy gradient? inifinte action space. While action as feature can generalize eg. `3KKK` to `3JJJ`


滴滴打车派单算法 *Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach*

- define one day as one episode, 
- state defined as `s = (grid, time)`,  offline learn `V(s)`, thus we know `Q(s,a) = r + V(s')`
- oneline dispatch: solve `a = argmax_a Q(s, a)` with KM algriom. 一种确定性算法，输入二分图及其权重，输出一个使权值和最大的匹配方案，这里权重就是offline估计出来的`Q(s,a)`


## GNN

### surveys

*A Comprehensive Survey Of Graph Neutral Network*

*A Comprehensive Survey Of Graph Embedding*

*Deep Learning On Graphs - A Survey*

*GNN in recomendation system A Survey*

### papers

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


[DiffPool] *Hierarchical Graph Representation Learning with Differentiable Pooling* 

- Apply "pooling" layer on graph. 
- Layer inptut is $A^l$, $X^l$, Assignment Matrix $S^l = GNN_{l,pool}(A^l, X^l)$, embedding features $Z^l = GNN_{l,emb}(A^l, X^l)$. Layer output $X^{l+1} = (S^l)^T Z^l$, $A^{l+1} = (S^l)^T A^l S^l$
- It's very like attention, $\alpha$ is the assignment matrix $S$ and $V$ is embedding $Z$, each is learned by a per-layer GNN
- disadvantage (found after trying code implemention)
    + not scalable. Adj matrix is soft thus must use dense, even if we sign() to sparse adj it's not differenciable. That's why this paper titled  "Diff"Pool
    + not scalable if "batch" a graph. Adj matrix size is `(bz * n) ^ 2`, and we also need to calculate `batch_mask` according to input/ouput number_of_nodes/clusters
    + number of nodes/cluster must be pre-specified. Just like image size must be fixed, here even if input n_nodes could be different, each layer's number of cluster must be same.


[TopKPooling] *Graph U-Nets*, *Towards Sparse Hierarchical Graph Classiﬁers*
- TopK pooling: slicing graph, `score = x @ w`, slicing index `idx = topk(score)`, next layer x = `(x * sigmoid(score))[idx]`
- unpooling: `x^{l+1} = distribute(0, x^l, idx)`
- unet, same as pix2pix u-net, with `GCN` inplace of `Conv`
- Code impl: `torch.topk()`, `torch.isin()`
- 会不会出现：filter node 之后 Adj matrix all zero ?


### frameworks

*DGL: a grpah-centric, highly-performant package for graph networks*


## unsupervised / GAN

GAN *Generative Adversarial Nets*. Define the min-max math problem and give training algriothm. Theoretical proof optimial D `=p_data / (p_data + p_g)` and G `p_G = p_data`. See code `gan.py`


## tree


## quant

AAAI_2021, 8 papers