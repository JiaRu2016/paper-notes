## NLP and seq models

### BERT

`code/nlp/bert/` doing...

亿点实现细节
- 数据集
    + 数据集大小：完全没有想象中那么多，BookCorpors压缩后1.1G 解压后4.6G. txt 和 int 都可以完全放进内存
    + Vocab构建速度：立等可取，纯python无优化
- Model Parallel. base and large 实现了pytorch模型并行。[关于效率问题的讨论](https://github.com/huggingface/transformers/issues/10151#issuecomment-778574713) Summarize: pytorch model parallel 就是打不满GPU, pipeline又太麻烦。
- model struct: 

```python
def forward(self, batch):
    is_next, sx, sy, msk, seg = batch
    # sx, sy, msk, seg (seq_len, bz); is_next (bz,)
    src = self.emb(sx) + self.seg_emb(seg) + self.__pos_emb(sx.shape[0])
    mem = self.encoder(src, src_key_padding_mask=self.__key_padding_mask(sx))
    # next sentence prediction
    nsp_yh = self.nsp_module(mem[0, :, :]).view(-1)  # (bz, )
    nsp_loss = self.nsp_loss_fn(nsp_yh, is_next.float())
    # masked language model
    mlm_yh = self.mlm_module(mem)  # (seq_len, bz, VOCAB_SIZE)
    mlm_loss_all = self.mlm_loss_fn(mlm_yh.flatten(0,1), sy.flatten(0,1))
    mlm_loss = torch.mean(mlm_loss_all[msk.flatten(0,1)])
```


### *Attention is all you need*  

`code/nlp/seq2seq_tfm.py` 

- src -> Encoder -> memroy -> +tgt decoder -> yh && **shifted_y** -> loss && backward
- evaluation: predict yh step-by-step

### seq2seq machine translation 

`code/nlp/seq2seq_lstm.py`

- padding. `BucketIterator` minimize total num of padding, by batching similar seq_len records together
- decoder is step-by-step, use *teacher_forcing* with a prob for some token

实现这个最初是因为 自己实现GNN遇到“同一个batch node个数不同”的问题，想参考一下NLP中类似问题如何处理，这里用了padding的方法。回到GNN的问题，一般就两种思路：
1. flatten, ie. batch small grpah to one large unconnected garph. 
    - This works for the most common case that graph struct is invarient, and is `torch_geometric`'s way
    - But if we need to change graph struct through layers, eg. GraphPooling, should tackle to not generate edge accross originaly different graphs, maybe use something like batch_mask? yes! Ref to torch_geometric impl of  `dense_diff_pool` (use `mask: BoolTensor (bz, max_num_nodes)`) and `topk` (use `batch: LongTensor [0,0,1,1,1,2,2...9]`)
2. padding && pass `padding_mask`
    - `LSTM` do not consider anything like padding_mask, we can consider this in loss function `CrossEntropyLoss(igore_index=PADDING_IDX)`
    - (TODO) `TransformerEncoder` accept padding mask: `src_key_padding_mask`

