import torch
import torch.nn as nn
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision
import PIL
import os
import numpy as np
from typing import Tuple
from pix2pix_model import UNet, Images2Scaler


# kaggle datasets download -d vikramtiwari/pix2pix-dataset
ROOT_DATASET_DIR = '/home/rjia/datasets/pix2pix/'
VALID_DS_NAME = ['cityscapes', 'edges2shoes', 'facades', 'maps']
IMG_SIZE_MAPPING = {
    'cityscapes': 256,
    'edges2shoes': 256,
    'facades': 256,
    'maps': 600,
}


class Hparam:
    def __init__(self, ds_name: str, **kwargs) -> None:
        self.ds_name = ds_name
        self.trimed_ds_name = ds_name.replace('_rev', '')
        assert self.trimed_ds_name in VALID_DS_NAME
        self.bz = 16
        self.num_epoch = 100
        self.lr = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.l1 = 10
        self.__set_kwargs(**kwargs)
    
    def __set_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)


##################################################################
# SECTION data
##################################################################

class pix2pixDataset(Dataset):
    def __init__(self, ds_name: str, is_train: bool):
        super().__init__()
        if ds_name.endswith('_rev'):
            self.is_reverse = True
            self.trimed_ds_name = ds_name.replace('_rev', '')
        else:
            self.is_reverse = False
            self.trimed_ds_name = ds_name
        assert self.trimed_ds_name in VALID_DS_NAME
        self.ds_name = ds_name
        self.IMG_SIZE = IMG_SIZE_MAPPING[self.trimed_ds_name]
        self.is_train = is_train
        self.ds_dir = os.path.join(
            ROOT_DATASET_DIR, 
            self.trimed_ds_name, 
            self.trimed_ds_name, 
            'train' if self.is_train else 'val'
        )
        self.filename_lst = os.listdir(self.ds_dir)
        self.length = len(self.filename_lst)

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> Tuple[FloatTensor, FloatTensor]:
        path = os.path.join(self.ds_dir, self.filename_lst[index])
        img_pil = PIL.Image.open(path)
        # TODO: maybe image preprocess use torch on GPU? cpu usage is very high if multi-task
        img_array = np.array(img_pil)   # (H, W, channel)
        img_array = np.moveaxis(img_array, source=2, destination=0) #  (channel, H, W)
        img_array = (img_array - 255 / 2) / 255 * 2  # [0, 255] -> [-1, 1]
        assert img_array.shape == (3, self.IMG_SIZE, 2 * self.IMG_SIZE)
        first = torch.FloatTensor(img_array[..., :self.IMG_SIZE])
        second = torch.FloatTensor(img_array[..., self.IMG_SIZE:])
        if self.is_reverse:
            return second, first
        else:
            return first, second
        

def get_dataloader(ds_name, is_train, bz):
    ds = pix2pixDataset(ds_name, is_train)
    dloader = DataLoader(ds, batch_size=bz, shuffle=is_train, drop_last=True)
    return dloader


##################################################################
# SECTION model
##################################################################

class Generator(nn.Module):
    def __init__(self, hp: Hparam):
        super().__init__()
        self._gen = UNet(hp)  # TODO
        #self._gen = EncoderDecoder(hp)  # TODO
    
    def forward(self, x):
        return self._gen(x)

class Discriminator(nn.Module):
    def __init__(self, hp: Hparam):
        super().__init__()
        self._disc = Images2Scaler(hp) # TODO
    
    def forward(self, x, y):
        return self._disc(x, y)

##################################################################
# SECTION train
##################################################################

def train(rank, hp):
    # data
    train_dloader = get_dataloader(hp.ds_name, is_train=True, bz=hp.bz)
    eval_dloader = get_dataloader(hp.ds_name, is_train=False, bz=4)
    # model
    device = torch.device(rank)
    gen = Generator(hp).to(device)
    disc = Discriminator(hp).to(device)
    gen_opt = Adam(gen.parameters(), lr=hp.lr, betas=(hp.beta1, hp.beta2))
    disc_opt = Adam(disc.parameters(), lr=hp.lr, betas=(hp.beta1, hp.beta2))
    l1_loss_fn = nn.L1Loss()
    bce_loss_fn = nn.BCELoss()
    ONES = torch.ones((hp.bz,)).to(device)
    ZEROS = torch.zeros((hp.bz,)).to(device)
    # tblog
    tblogger = SummaryWriter(f'./tblog/pix2pix_{int(time.perf_counter())}__{rank}_ds={hp.ds_name}_l1={hp.l1}')
    # evaluate_thread
    eval_thread = evaluate(eval_dloader, device, l1_loss_fn, tblogger)
    eval_thread.send(None)

    global_step = 0
    for ep in range(hp.num_epoch):
        for x, y_real in train_dloader:
            global_step += 1
            # print(x.shape)  # (bz, 3, IMG_SIZE, IMG_SIZE)
            # print(y_real.shape)  # (bz, 3, IMG_SIZE, IMG_SIZE)
            x, y_real = x.to(device), y_real.to(device)

            y_fake = gen(x)
            # train disc
            disc_opt.zero_grad()
            disc_loss = bce_loss_fn(disc(x, y_fake.detach()), ZEROS) + bce_loss_fn(disc(x, y_real), ONES)
            disc_loss.backward()
            disc_opt.step()
            # train gen
            gen_opt.zero_grad()
            gen_loss = bce_loss_fn(disc(x, y_fake), ONES)
            if hp.l1 > 0:
                l1_loss = l1_loss_fn(y_fake, y_real)
                gen_loss +=  hp.l1 * l1_loss
            gen_loss.backward()
            gen_opt.step()

            if global_step % 10 == 0:
                print(f'rank {rank} | global_step {global_step} | disc_loss {disc_loss.item()} | gen_loss {gen_loss.item()}')
                # scaler
                tblogger.add_scalar('loss/disc/train', disc_loss)
                tblogger.add_scalar('loss/gen/train', gen_loss)
                if hp.l1 > 0:
                    tblogger.add_scalar('loss/l1/train', l1_loss)
                # image
                img_display = x_yreal_yfake_to_one_image(x, y_real, y_fake)
                tblogger.add_image(f'y_fake/train', img_display, global_step=global_step)
                # eval_thread
                eval_thread.send((global_step, gen))
            

def evaluate(eval_dloader: DataLoader, device: torch.device, l1_loss_fn, tblogger: SummaryWriter):
    x, y_real = next(iter(eval_dloader))
    x, y_real = x.to(device), y_real.to(device)
    while True:
        step, gen_model = yield
        with torch.no_grad():
            y_fake = gen_model(x)
            l1_loss = l1_loss_fn(y_fake, y_real).cpu()
        img_display = x_yreal_yfake_to_one_image(x, y_real, y_fake)
        tblogger.add_image(f'y_fake/eval', img_display, global_step=step)
        tblogger.add_scalar(f'loss/l1/eval', l1_loss, global_step=step)


def x_yreal_yfake_to_one_image(x, y_real, y_fake):
    with torch.no_grad():
        img_display = torch.cat([x, y_real, y_fake], dim=3)
        img_display = torch.cat([img for img in img_display[:4]], dim=1)
        img_display = (img_display / 2 + 0.5) * 255  # [-1, 1] -> [0, 255]
    return img_display



if __name__ == '__main__':
    hparam_lst_chunk = [
        [
            Hparam('cityscapes', l1=100),
            Hparam('cityscapes', l1=10),
            Hparam('cityscapes_rev', l1=100),
            Hparam('cityscapes_rev', l1=10),
        ],
        [
            Hparam('facades', l1=100),
            Hparam('facades', l1=10),
            Hparam('facades_rev', l1=100),
            Hparam('facades_rev', l1=10),
        ],
        [
            Hparam('edges2shoes', l1=100),
            Hparam('edges2shoes', l1=10),
            Hparam('edges2shoes_rev', l1=100),
            Hparam('edges2shoes_rev', l1=10),
        ]
    ]

    import multiprocessing as mp
    
    for hparam_lst in hparam_lst_chunk:
        print('============== next chunk')

        process_lst = []
        for rank, hp in enumerate(hparam_lst):
            p = mp.Process(target=train, args=(rank, hp))
            p.start()
            process_lst.append(p)
            print(f'submitted task {rank}')
        
        for p in process_lst:
            p.join()
        