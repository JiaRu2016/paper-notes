## CV

### [ViT] *An image is worth 16x16 words: transformer for image recognition at scale*

主要思想
- 将Transformer直接运用于图片，完全抛弃CNN
- 如果直接把pix作为token扔进去，sequence长度太长，也不Scalable
- 用超大规模的数据预训练，再再下游任务微调，即题目中的 at scale

Transfromer 与 CNN 的比较
- inductive bias of CNN: **locality** and **translation equivariance** 贯穿于CNN-arch始终
- Transformer没有这个先验/regularization, 所以需要更大规模的数据学出来
- 实验结果证明了上述推断：在超大数据集上ViT才能打败ResNet
- Hybrid Architecture: CNN first, flatten the spatial dims as transformer input sequence

其他
- fintuneing at highter resolution images:
    + 这里的做法: patch_size不变，effective sequence length 变长， position embedding 2D差值
    + My Opinion: ~~可以压缩到原始的resolution更 make sence, 因为图片更大而patch_size不变，相对的 receptive field 就变了~~
        - if pix level prediction task, patch should be fixed 16x16
- **linear-few-shot-eval**
    + take a subset of samples, use the frozen reprezentation as input x, solving a regularized Least Square regression problem, then take 5 samples for each class for eval (5-shot eval)
    + for fast one-the-fly eval where fine tuning is costly

实验结果图
- Metrics
    1. 数据集大小 ~ Top1Acc
    2. 大数据预训练 number_of_samples ~ **few-shot-eval** Top1Acc
    3. pretraining FLOPS ~ Transfer acc
    4. 另外关注：随着数据集/模型大小的增长perfromance是否饱和？都没有
- Inspect
    1. 学到的LinearEmbedding层参数, RGB embedding filters (first 28 principal components)


### [MAE] *Masked Autoencoders Are Scalable Vision Learners*

自监督预训练、在下游任务精调，超过有监督

model:
- input -> mask (remove 75% patchs) + pos_embedding -> Encoder -> latent
- concate(latent, MASK) + pos_embedding -> Deocder -> reconstruct_loss

note:
- 进入Encoder的只有 visiable patch, 为了减少计算量和内存占用，使得Encoder可以更大
- lightweight decoder, decoder only used in pre-training and discarded when finetune
- Linear probing: fixed fature map, do not tune


### *MLP-Mixer: An all-MLP Architecture for Vision*

只看图就行了

- 意义：证明 CNN 和 Transformer 都不是必需的, MLP也足以达到相同的performence
- 分别在 patch 和 channels 两个维度上使用MLP
    + 在 patch 维度上的 MLP 代替了 self_attention
    + 在 channels 维度上的 MLP 就是 positional_feed_forward
- 2 trainable modules: `MLP_1(patch_size -> patch_size)`, `MLP_2(c -> c)`


### *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*

- 要解决的问题：high resolution image, too many patches, thus very long sequence length for transformer
- 解决方式：我觉得就是个缝合怪，这里面所有的要素之前都有类似想法
    + hierarchical: 
        * attention scrore 只看附近的。 Masked/sparse attention
        * 逐层merge扩大感受野，类似于CNN。 merged patch 使得sequence_length变短，所以channels每次要翻倍
        * 相邻patch之间没有信息传递怎么办？shifted window
    + todo: 工程实现 efficient self-attention in shifted window partitioning


### [PoolFormer/MetaFormer] *MetaFormer is Actually What You Need for Vision*

token-mixer是什么不重要，重要的transformer这个结构：`TokenMixer -> (add and norm) -> ChannelMLP -> (add and norm)`
`Tokenmixer`可以是 `Attention`, `SptialMLP`, even `Pooling`