## MultiModal

### CLIP

- in-batch negative sample. TODO how to parallelize it ?
- very large batch 不是简单的暴力出奇迹，而是这种图文pair数据是从互联网爬下来的，带有很大噪声，必须非常大data_size和batch_size
- model arch: `mat = OuterProduct(TextTransformer(text_input), ImgTransformer(img_input)); loss(mat, DIAG_IS_ONE)`
- 作为大规模预训练模型，在下游任务上微调超过有监督
