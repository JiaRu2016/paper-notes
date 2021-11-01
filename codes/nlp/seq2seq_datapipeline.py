from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator


class data_pipelie():
    def __init__(self, bz):
        self.SRC_VOCAB_SIZE = 0
        self.TGT_VOCAB_SIZE = 0
        self.SRC_PADDING_IDX = 1
        self.TGT_PADDING_IDX = 1
        self.TGT_SOS_IDX = 2
        self.TGT_EOS_IDX = 3
        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None
        self.__initalize(bz)
        
    def __initalize(self, bz: int):
        src_field = Field(init_token='<sos>', eos_token='<eos>', lower=True)
        tgt_field = Field(init_token='<sos>', eos_token='<eos>', lower=True)

        train_ds, valid_ds, test_ds = Multi30k.splits(
            exts = ('.de', '.en'), 
            fields = (src_field, tgt_field),
            root='/home/rjia/playground/datasets/'
        )
        # train_ds.__class__
        # isinstance(train_ds, torch.utils.data.Dataset)  # True
        # a = train_ds[0]
        # a.__class__
        # a.src
        # a.trg
        
        # MAX_SEQ_LEN
        src_max_len = []
        tgt_max_len = []
        for e in train_ds.examples:
            src_max_len.append(len(e.src))
            tgt_max_len.append(len(e.trg))
        self.SRC_MAX_SEQ_LEN = max(src_max_len) + 2
        self.TGT_MAX_SEQ_LEN = max(tgt_max_len) + 2
        
        # build vocab
        src_field.build_vocab(train_ds, min_freq=10)
        tgt_field.build_vocab(train_ds, min_freq=10)

        # for tk in [src_field.unk_token, src_field.pad_token, src_field.init_token, src_field.eos_token]:
        #     print(tk, src_field.vocab.stoi[tk])

        self.SRC_VOCAB_SIZE = len(src_field.vocab)
        self.TGT_VOCAB_SIZE = len(tgt_field.vocab)
        self.SRC_PADDING_IDX = src_field.vocab.stoi[src_field.pad_token]
        self.TGT_PADDING_IDX = tgt_field.vocab.stoi[tgt_field.pad_token]
        self.TGT_SOS_IDX = tgt_field.vocab.stoi[tgt_field.init_token]
        self.TGT_EOS_IDX = tgt_field.vocab.stoi[tgt_field.eos_token]

        self.train_iter, self.valid_iter, self.test_iter = BucketIterator.splits(
            (train_ds, valid_ds, test_ds),
            batch_size=bz
        )
    
    def __repr__(self) -> str:
        src = f'SRC vocab_size {self.SRC_VOCAB_SIZE} max_len {self.SRC_MAX_SEQ_LEN}'
        tgt = f'TGT vocab_size {self.TGT_VOCAB_SIZE} max_len {self.TGT_MAX_SEQ_LEN}'
        return f'<Seq2Seq_datapipeline> {src}, {tgt}'



if __name__ == '__main__':
    data = data_pipelie(128)
    print(data)
    batch = next(iter(data.train_iter))
    x, y = batch.src, batch.trg
    print(x)
    print(y)