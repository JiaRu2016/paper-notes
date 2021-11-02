"""
Model parallel version of `nn.TransformerEncoder`
copy and modify source code in torch/nn/modules/transformer.py
"""
import torch
from torch import Tensor
from typing import Optional, List
import torch.nn as nn
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MP_TransformerEncoder(nn.Module):
    r"""
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MP_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # related to device
        self.nlayer = num_layers
        self.n_device = 0
        self.device_lst = []
        self.device_mapping = []  # ilayer: device_idx

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i, mod in enumerate(self.layers):
            dev = self.device_mapping[i]
            output = output.to(dev)
            if mask is not None:
                mask = mask.to(dev)
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.to(dev)
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = output.to(self.device_mapping[-1])
            output = self.norm(output)

        return output

    def set_and_mvto_devices(self, device_lst: List[torch.device]):
        self.device_lst = device_lst
        self.n_device = len(device_lst)
        self.device_mapping = _gen_layer_device_mapping(self.nlayer, self.n_device)
        # mv layers to device
        for i, mod in enumerate(self.layers):
            dev = self.device_mapping[i]
            mod.to(dev)
        if self.norm is not None:
            self.norm.to(self.device_mapping[-1])
        print(f'set devices and mv layers to devices, mapping: {self.device_mapping}')


def _gen_layer_device_mapping(nlayer, ndevice):
    """layer-device arrangement: for mod, arrange to head devices. 
    In BERT base
        - embedding #param 8 M (d * VOCAB_SIZE)
        - TransformerEncoder #param 85 M  (10 d^2)
        - nsp and mlm #param 9 M (2 d^2)
    """
    div, mod = divmod(nlayer, ndevice)
    nlayer_per_device = [div + 1 for _ in range(mod)] + [div for _ in range(ndevice - mod)]
    ll = [[i] * n for i, n in enumerate(nlayer_per_device)]
    return _flatten_lst(ll)[:nlayer]


def _flatten_lst(lst: List[List]):
    return [a for sub_lst in lst for a in sub_lst]


def unittest():
    seq_len, bz, d_model = 10, 32, 512
    nlayer = 10
    # expected mapping: [0000 111 222]
    device_lst = [torch.device(0), torch.device(1), torch.device(2)]  # 3 device
    x = torch.randn(size=(seq_len, bz, d_model))
    m = MP_TransformerEncoder(
        nn.TransformerEncoderLayer(d_model, 8, d_model*4),
        num_layers=nlayer,
        norm=None,
    )
    m.set_and_mvto_devices(device_lst)
    out = m(x)
    out.sum().backward()


if __name__ == '__main__':
    unittest()
    # CUDA_VISIBLE_DEVICES=6,7,8 python MP_TransformerEncoder.py
