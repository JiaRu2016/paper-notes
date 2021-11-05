## NLP and seq models

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

