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
    + My Opinion: 可以压缩到原始的resolution更 make sence, 因为图片更大而patch_size不变，相对的 receptive field 就变了
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


