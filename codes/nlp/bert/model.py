import torch
import torch.nn as nn
import pprint
from MP_TransformerEncoder import MP_TransformerEncoder


class Hparam():
    def __init__(self, name='tiny'):
        # data
        self.bz = 64
        self.VOCAB_SIZE = 3000  # for unittest work. should set later
        self.PAD = 1  # for unittest work. sould set later
        self.max_len = 50  # single sentence
        # Transformer Model
        assert name in ('tiny', 'base', 'large')
        if name == 'tiny':
            self.d_model = 512
            self.nlayer = 6
            self.nhead = 8
        elif name == 'base':
            self.d_model = 768
            self.nlayer = 12
            self.nhead = 12
        elif name == 'large':
            self.d_model = 1024
            self.nlayer = 24
            self.nhead = 16
        self.dropout = 0.1
        self.activation = 'gelu'
        self.name = name
        # optimizer
        self.num_step = 10_0000
        self.optimizer = (
            torch.optim.Adam, 
            {'lr': 1e-4, 'betas': (0.9, 0.999), 'weight_decay': 0.01}
        )
    
    def set_data_related_fields(self, data):
        self.VOCAB_SIZE = data.vocab.VOCAB_SIZE
        self.PAD = data.vocab.PAD


def generate_example_batch(hp: Hparam):
    s0_len = 10
    s1_len = 20
    assert s0_len <= hp.max_len and s1_len <= hp.max_len
    seq_len = s0_len + s1_len + 3
    MSK = 5
    
    sy = torch.randint(hp.VOCAB_SIZE, size=(seq_len, hp.bz))
    msk = torch.rand(size=(seq_len, hp.bz)) < 0.15
    sx = sy.masked_fill(msk, MSK)
    is_next = torch.rand(size=(hp.bz,)) < 0.5
    seg = torch.cat([torch.zeros(size=(s0_len+2, hp.bz)), torch.zeros(size=(s1_len+1, hp.bz))]).long()
    
    for ts in [is_next, sx, sy, msk, seg]:
        print(ts.shape, ts.dtype)
    return is_next, sx, sy, msk, seg


class BertModel(nn.Module):
    def __init__(self, hp: Hparam) -> None:
        super().__init__()
        self.hp = hp
        self.device = None
        # layers
        self.emb = nn.Embedding(hp.VOCAB_SIZE, hp.d_model)
        self.seg_emb = nn.Embedding(2, hp.d_model)
        self.pos_embedding_table = nn.Parameter(torch.randn(size=(hp.max_len * 2 + 3, hp.d_model)))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hp.d_model, 
                nhead=hp.nhead, 
                dim_feedforward=hp.d_model*4, 
                dropout=hp.dropout,
                activation=hp.activation,
            ),
            num_layers=hp.nlayer,
            norm=None,
        )
        self.nsp_module = nn.Sequential(
            nn.Linear(hp.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.nsp_loss_fn = nn.BCELoss()
        self.mlm_module = nn.Sequential(
            nn.Linear(hp.d_model, hp.d_model),
            nn.ReLU(),
            nn.Linear(hp.d_model, hp.VOCAB_SIZE),
        )
        self.mlm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.__num_parameters()
    
    def __num_parameters(self):
        """tiny: 22 M, base: 103 M, large: 327 M
        """
        total = self.__num_param(self)
        sub_modules = {
            name: self.__num_param(mod) for name, mod in self.named_children()
        }
        print(f'BERT_{self.hp.name}  #param: {total} M')
        pprint.pprint(sub_modules)
    
    @staticmethod
    def __num_param(mod):
        return sum(p.numel() for p in mod.parameters()) / 1e6

    # def __pos_emb(self, s0_len: int, s1_len: int):
    #     """too expensive for batch
    #     """
    #     return torch.cat([
    #         self.pos_embedding_table[:s0_len+2], 
    #         self.pos_embedding_table[:s1_len+1]
    #     ], dim=0)[:, None, :]
    
    def __pos_emb(self, seq_len):
        return self.pos_embedding_table[:seq_len, None, :]

    def __key_padding_mask(self, x):
        """`Transformer(..., src_key_padding_mask=)` need input shape (N, src_len)
        """
        return (x == self.hp.PAD).t()
    
    def forward(self, batch):
        # sx, sy (seq_len, bz)
        batch = tuple(x.to(self.device) for x in batch)
        is_next, sx, sy, msk, seg = batch
        mem = self.forward_encoder(sx, seg)
        loss = self.forward_nsp_mlm(mem, is_next, sy, msk)
        return loss

    def forward_encoder(self, sx, seg):
        # src, mem (seq_len, bz, dmodel)
        src = self.emb(sx) + self.seg_emb(seg) + self.__pos_emb(sx.shape[0])
        mem = self.encoder(src, src_key_padding_mask=self.__key_padding_mask(sx))
        return mem
    
    def forward_nsp_mlm(self, mem, is_next, sy, msk):
        # next sentence prediction
        nsp_yh = self.nsp_module(mem[0, :, :]).view(-1)  # (bz, )
        nsp_loss = self.nsp_loss_fn(nsp_yh, is_next.float())
        # masked language model
        mlm_yh = self.mlm_module(mem)  # (seq_len, bz, VOCAB_SIZE)
        mlm_loss_all = self.mlm_loss_fn(mlm_yh.flatten(0,1), sy.flatten(0,1))
        mlm_loss = torch.mean(mlm_loss_all[msk.flatten(0,1)])
        return mlm_loss + nsp_loss

    def set_and_mvto_device(self, device: torch.device):
        self.device = device
        self.to(device)
        return self


class MP_BertModel(BertModel):
    def __init__(self, hp: Hparam) -> None:
        super().__init__(hp)
        self.encoder = MP_TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hp.d_model, 
                nhead=hp.nhead, 
                dim_feedforward=hp.d_model*4, 
                dropout=hp.dropout,
                activation=hp.activation,
            ),
            num_layers=hp.nlayer,
            norm=None,
        )
        self.device_lst = None
        self.dev0 = None
        self.devN = None
    
    def set_and_mvto_device(self, device_lst):
        self.device_lst = device_lst
        self.dev0 = device_lst[0]
        self.devN = device_lst[-1]
        self.emb.to(self.dev0)
        self.seg_emb.to(self.dev0)
        # NOTE for nn.Parameter(), use `param.data = param.data.to(device)`
        self.pos_embedding_table.data = self.pos_embedding_table.data.to(self.dev0)
        self.encoder.set_and_mvto_devices(self.device_lst)
        self.nsp_module.to(self.devN)
        self.mlm_module.to(self.devN)
        return self
    
    def forward(self, batch):
        is_next, sx, sy, msk, seg = batch
        sx, seg = sx.to(self.dev0), seg.to(self.dev0)
        mem = self.forward_encoder(sx, seg)
        mem.to(self.devN)
        is_next, sy, msk = is_next.to(self.devN), sy.to(self.devN), msk.to(self.devN)
        loss = self.forward_nsp_mlm(mem, is_next, sy, msk)
        return loss


def test():
    model_name = 'base'
    hp = Hparam(model_name)

    if model_name == 'large':
        dev = list(map(torch.device, [0,1,2,3]))
        model = MP_BertModel(hp).set_and_mvto_device(dev)
    elif model_name == 'base':
        dev = list(map(torch.device, [0,1]))
        model = MP_BertModel(hp).set_and_mvto_device(dev)
    else:
        dev = torch.device(0)
        model = BertModel(hp).set_and_mvto_device(dev)
    
    # forward backward
    for _ in range(50):
        batch = generate_example_batch(hp)
        loss = model(batch)
        loss.backward()

if __name__ == '__main__':
    test()
    # CUDA_VISIBLE_DEVICES=5,6,7,8 python model.py
