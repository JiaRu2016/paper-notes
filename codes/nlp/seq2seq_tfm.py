import torch
import torch.nn as nn
from torch import LongTensor, FloatTensor, BoolTensor
import numpy as np
from seq2seq_datapipeline import data_pipelie


class Hparam():
    def __init__(self):
        # Transformer Model
        self.d_model = 512
        self.nlayer = 6
        self.nhead = 8
        self.dropout = 0.1
        # training
        self.bz = 32
        self.num_step = 10_0000
        self.optimizer = (torch.optim.Adam, {'lr': 1e-4, 'betas': (0.9, 0.98), 'eps': 1e-9})
        #self.lr = ( 1/sqrt(d_model) ) * min( 1/sqrt(step),  step*4000^(-1.5) )
        self.label_smoothing = 0.1
        # data-related
        self.SRC_VOCAB_SIZE = None
        self.TGT_VOCAB_SIZE = None
        self.SRC_PADDING_IDX = None
        self.TGT_PADDING_IDX = None
        self.TGT_SOS_IDX = None
        self.TGT_EOS_IDX = None
        self.SRC_MAX_SEQ_LEN = None
        self.TGT_MAX_SEQ_LEN = None
    
    def __repr__(self):
        return f'd{self.d_model}_l{self.nlayer}_h{self.nhead}__vocab_{self.SRC_VOCAB_SIZE}_{self.TGT_VOCAB_SIZE}'
    
    def set_data_related_fields(self, data: data_pipelie):
        self.SRC_VOCAB_SIZE = data.SRC_VOCAB_SIZE
        self.TGT_VOCAB_SIZE = data.TGT_VOCAB_SIZE
        self.SRC_PADDING_IDX = data.SRC_PADDING_IDX
        self.TGT_PADDING_IDX = data.TGT_PADDING_IDX
        self.TGT_SOS_IDX = data.TGT_SOS_IDX
        self.SRC_MAX_SEQ_LEN = data.SRC_MAX_SEQ_LEN
        self.TGT_MAX_SEQ_LEN = data.TGT_MAX_SEQ_LEN
        self.TGT_EOS_IDX = data.TGT_EOS_IDX


class Seq2SeqTransformer(nn.Module):
    def __init__(self, hp: Hparam) -> None:
        super().__init__()
        # saved values
        self.d_model = hp.d_model
        self.SRC_PADDING_IDX = hp.SRC_PADDING_IDX
        self.TGT_PADDING_IDX = hp.TGT_PADDING_IDX
        self.TGT_SOS_IDX = hp.TGT_SOS_IDX
        # saved values - for predict()
        self.TGT_EOS_IDX = hp.TGT_EOS_IDX
        self.TGT_MAX_SEQ_LEN = hp.TGT_MAX_SEQ_LEN
        self.SRC_MAX_SEQ_LEN = hp.SRC_MAX_SEQ_LEN

        # layers
        self.src_emb = nn.Embedding(hp.SRC_VOCAB_SIZE, hp.d_model)
        self.tgt_emb = nn.Embedding(hp.TGT_VOCAB_SIZE, hp.d_model)
        self.tfm = nn.Transformer(hp.d_model, hp.nhead, hp.nlayer, hp.nlayer, hp.d_model*4, hp.dropout)
        self.output_yh = nn.Linear(hp.d_model, hp.TGT_VOCAB_SIZE)  # (tgt_seq_len, bz, d_model -> TGT_VOCAB_SIZE)
        self.register_buffer('src_pe', self.__generate_position_encoding_table(hp.SRC_MAX_SEQ_LEN))
        self.register_buffer('tgt_pe', self.__generate_position_encoding_table(hp.TGT_MAX_SEQ_LEN))
        self.register_buffer('tgt_mask', self.__no_forward_looking_mask(self.TGT_MAX_SEQ_LEN))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=hp.TGT_PADDING_IDX, label_smoothing=hp.label_smoothing)

    def forward(self, x: LongTensor, y: LongTensor):
        src_seq_len, tgt_seq_len = x.shape[0], y.shape[0] - 1
        sos_y = y[:-1]
        y_eos = y[1:]
        src = self.src_emb(x) + self.src_pe[:src_seq_len][:, None, :]
        tgt = self.tgt_emb(sos_y) + self.tgt_pe[:tgt_seq_len][:, None, :]
        out = self.tfm(
            src, tgt, 
            tgt_mask=self.tgt_mask[:tgt_seq_len, :tgt_seq_len], 
            src_key_padding_mask=self.__padding_mask(x, self.SRC_PADDING_IDX), 
            tgt_key_padding_mask=self.__padding_mask(sos_y, self.TGT_PADDING_IDX),
        )
        # out: (tgt_seq_len, bz, d_model)
        yh = self.output_yh(out)  # (tgt_seq_len, bz, TGT_VOCAB_SIZE)
        loss = self.loss_fn(yh.flatten(0,1), y_eos.flatten(0,1))
        return loss, yh.argmax(-1)

    def __padding_mask(self, a: LongTensor, PADDING_IDX: int) -> BoolTensor:
        # a: (seq_len, bz)
        # `xx_key_padding_mask` require shape (bz, seq_len)
        return (a == PADDING_IDX).t()

    def __no_forward_looking_mask(self, seq_len: int) -> BoolTensor:
        # [i,i] is False, ie. DO NOT mask t itself at time step t. Otherwise decoder self_attn nan
        return torch.triu(torch.ones(size=(seq_len, seq_len)).bool(), diagonal=1)
    
    def __generate_position_encoding_table(self, max_seq_len):
        assert self.d_model % 2 == 0
        out = torch.empty(size=(max_seq_len, self.d_model))
        pos = torch.arange(max_seq_len)
        for i in range(int(self.d_model / 2)):
            out[:, 2 * i] = torch.sin(pos / 1e4 ** (2 * i / self.d_model))
            out[:, 2 * i + 1] = torch.cos(pos / 1e4 ** (2 * i / self.d_model))
        return out

    def predict(self, x):
        self.eval()
        if x.shape[0] > self.SRC_MAX_SEQ_LEN:
            x = x[:self.SRC_MAX_SEQ_LEN]
        src_seq_len, bz = x.shape
        yh = x.new_full(size=(self.TGT_MAX_SEQ_LEN, bz), fill_value=self.TGT_SOS_IDX)
        pe = self.tgt_pe[:, None, :]
        with torch.no_grad():
            for t in range(self.TGT_MAX_SEQ_LEN - 1):
                src = self.src_emb(x) + self.src_pe[:src_seq_len][:, None, :]
                tgt = self.tgt_emb(yh) + pe
                mem = self.tfm.encoder(src, src_key_padding_mask=self.__padding_mask(x, self.SRC_PADDING_IDX))
                out = self.tfm.decoder(tgt, mem, tgt_mask=self.tgt_mask)
                # out: (seq_len, bz, d_model)
                out = self.output_yh(out)  # (seq_len, bz, TGT_VOCAB_SIZE)
                out = out.argmax(-1)  # (seq_len, bz)
                yh[t+1] = out[t]
            return yh


def main():
    hp = Hparam()
    data = data_pipelie(hp.bz)
    hp.set_data_related_fields(data)

    device = torch.device(1)
    model = Seq2SeqTransformer(hp).to(device)
    OptKls, opt_kwargs = hp.optimizer
    opt = OptKls(model.parameters(), **opt_kwargs)

    ep, step = 0, 0
    while True:
        ep += 1
        for batch in data.train_iter:
            step += 1

            x = batch.src
            y = batch.trg
            x, y = x.to(device), y.to(device)
            model.train()
            opt.zero_grad()
            loss, yh = model(x, y)
            loss.backward()
            opt.step()

            if step % 10 == 0:
                print(f'ep {ep} step {step} | loss {loss.item()}')
            
            if step % 100 == 0:
                yh = model.predict(x)
                x = x[:, 0].tolist()
                y = y[:, 0].tolist()
                yh = yh[:, 0].tolist()
                acc = np.mean([y_ == yh_ for y_, yh_ in zip(y, yh) if y_ != data.TGT_PADDING_IDX])
                print(f'acc: {acc}\n{x}\n{yh}\n{y}')
            
            if step > hp.num_step:
                return


if __name__ == '__main__':
    main()