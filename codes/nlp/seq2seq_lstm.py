"""
https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
"""

import torch
import torch.nn as nn
import random
from seq2seq_datapipeline import data_pipelie


class Hparam():
    def __init__(self):
        self.bz = 128
        self.edim = 512
        self.hdim = 32
        self.nlayer = 2
        self.dropout = 0.5
        self.force_teacher_prob = 0.3
        self.num_epoch = 50
        self.lr = 0.7
        self.SRC_VOCAB_SIZE = None
        self.TGT_VOCAB_SIZE = None
        self.SRC_PADDING_IDX = None
        self.TGT_PADDING_IDX = None
        self.TGT_SOS_IDX = None
    
    def __repr__(self):
        return f'bz{self.bz}_e{self.edim}_h{self.hdim}_nlayer{self.nlayer}__vocab_{self.SRC_VOCAB_SIZE}_{self.TGT_VOCAB_SIZE}'
    
    def set_data_related_fields(self, data: data_pipelie):
        self.SRC_VOCAB_SIZE = data.SRC_VOCAB_SIZE
        self.TGT_VOCAB_SIZE = data.TGT_VOCAB_SIZE
        self.SRC_PADDING_IDX = data.SRC_PADDING_IDX
        self.TGT_PADDING_IDX = data.TGT_PADDING_IDX
        self.TGT_SOS_IDX = data.TGT_SOS_IDX


class Encoder(nn.Module):
    def __init__(self, hp: Hparam) -> None:
        super().__init__()
        self.emb = nn.Embedding(hp.SRC_VOCAB_SIZE, hp.edim, padding_idx=hp.SRC_PADDING_IDX)
        self.lstm = nn.LSTM(hp.edim, hp.hdim, hp.nlayer, batch_first=False)
    
    def forward(self, x):
        # x:  (seq_len, bz), LongTensor
        x_embedded = self.emb(x)   # (seq_len, bz, edim)
        h_L, (h_T, c_T) = self.lstm(x_embedded)  # h_T, c_T: (num_layers, bz, h)
        return h_T, c_T


class OneStepDecoder(nn.Module):
    def __init__(self, hp: Hparam) -> None:
        super().__init__()
        self.emb = nn.Embedding(hp.TGT_VOCAB_SIZE, hp.edim, padding_idx=hp.TGT_PADDING_IDX)
        self.lstm = nn.LSTM(hp.edim, hp.hdim, hp.nlayer, batch_first=False)
        self.ff = nn.Linear(hp.hdim, hp.TGT_VOCAB_SIZE)
    
    def forward(self, ipt, h, c):
        ipt_embedded = self.emb(ipt)  # (bz,) -> (bz, edim)
        h_L, (h_T, c_T) = self.lstm(ipt_embedded.unsqueeze(0), (h, c))
        # h_T, c_T: (num_layers, bz, hdim)
        yh_t = self.ff(h_T[-1])  # (bz, hdim -> VOCAB_SIZE)
        return yh_t, (h_T, c_T)


class Seq2seq(nn.Module):
    def __init__(self, hp: Hparam) -> None:
        super().__init__()
        self.encoder = Encoder(hp)
        self.one_step_decoder = OneStepDecoder(hp)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=hp.TGT_PADDING_IDX)
        self.hp = hp

    def forward(self, x, y):
        # x, y: (seq_len, bz), Long
        h, c = self.encoder(x)  # h, c: (num_layers, bz, h)

        seq_len, bz = y.shape
        yh = torch.zeros(size=(seq_len, bz, self.hp.TGT_VOCAB_SIZE))
        ipt = torch.full(size=(bz,), fill_value=self.hp.TGT_SOS_IDX)
        for t in range(seq_len):
            yh_t, (h, c) = self.one_step_decoder(ipt, h, c)
            # yh_t: (bz, VOCAB_SIZE)
            yh[t].copy_(yh_t)
            if random.random() < self.hp.force_teacher_prob:
                ipt = y[t]
            else:
                ipt = yh_t.argmax(dim=1)  # (bz,)

        return self.loss_fn(yh.flatten(0,1), y.flatten(0,1))


def main():
    hp = Hparam()
    data = data_pipelie(hp.bz)
    hp.set_data_related_fields(data)

    model = Seq2seq(hp)
    opt = torch.optim.SGD(model.parameters(), lr=hp.lr)

    for ep in range(hp.num_epoch):
        for step, batch in enumerate(data.train_iter):
            x = batch.src
            y = batch.trg
            model.train()
            opt.zero_grad()
            loss = model(x, y)
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f'ep {ep} step {step} | loss {loss.item()}')


if __name__ == '__main__':
    main()