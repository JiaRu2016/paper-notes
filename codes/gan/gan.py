import torch
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time


# need to download to local. see download_dataset.py
TORCH_DATASET_DIR = '/home/rjia/datasets/torch_datasets/'

class Hparam:
    def __init__(self):
        self.bz = 32
        self.FLATTEN_IMG_DIM = 784
        self.noise_dim = 64
        self.gen_hidden_dim = 256
        self.disc_hidden_dim = 128
        self.num_epoch = 50
        self.disc_n_step = 1
        self.lr = 3e-4


def get_real_img_dataloader(bz):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    ds_train = torchvision.datasets.MNIST(TORCH_DATASET_DIR, train=True, transform=transform)
    ds_test = torchvision.datasets.MNIST(TORCH_DATASET_DIR, train=False, transform=transform)
    ds = ConcatDataset([ds_train, ds_test])
    def __collate_fn(samples):
        # samples: (img_Tensor, label_int) where img_Tensor is (1,28,28)
        samples = [tp[0].squeeze(0) for tp in samples]
        a = torch.stack(samples)
        return a.flatten(1,2)
    return DataLoader(ds, batch_size=bz, shuffle=True, collate_fn=__collate_fn, drop_last=True)


def noise_iterator(bz, noise_dim):
    while True:
        yield torch.randn((bz, noise_dim))


class Discriminator(nn.Module):
    def __init__(self, flatten_img_dim, hidden_dim):
        super().__init__()
        self.flatten_img_dim = flatten_img_dim
        self.hidden_dim = hidden_dim
        self._disc = nn.Sequential(
            nn.Linear(self.flatten_img_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._disc(x).view(-1)


class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, flatten_img_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.flatten_img_dim = flatten_img_dim
        self._gen = nn.Sequential(
            nn.Linear(self.noise_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, self.flatten_img_dim),
            nn.Tanh()
        )
    
    def forward(self, z: Tensor) -> Tensor:
        return self._gen(z)


def train():
    hp = Hparam()
    # data
    real_img_dataloader = get_real_img_dataloader(hp.bz)
    noise_iter = noise_iterator(hp.bz, hp.noise_dim)
    # model, opt and loss_fn
    device = torch.device(0)
    gen = Generator(hp.noise_dim, hp.gen_hidden_dim, hp.FLATTEN_IMG_DIM).to(device)
    disc = Discriminator(hp.FLATTEN_IMG_DIM, hp.disc_hidden_dim).to(device)
    opt_gen = Adam(gen.parameters(), lr=hp.lr)
    opt_disc = Adam(disc.parameters(), lr=hp.lr)
    bce_loss_fn = nn.BCELoss()
    ONES = torch.ones((hp.bz,)).to(device)
    ZEROS = torch.zeros((hp.bz,)).to(device)
    # logs
    tblogger = SummaryWriter(log_dir=f'./tblog/{int(time.perf_counter())}')

    global_step = 0
    for epoch in range(hp.num_epoch):
        for x_real in real_img_dataloader:
            global_step += 1

            x_real = x_real.to(device)
            z = next(noise_iter).to(device)
            opt_disc.zero_grad()
            x_fake = gen(z).detach()
            loss = bce_loss_fn(disc(x_real), ONES) + bce_loss_fn(disc(x_fake), ZEROS)
            loss.backward()   
            opt_disc.step()
            disc_loss_float = loss.cpu().item()

            z = next(noise_iter).to(device)
            opt_gen.zero_grad()
            x_fake = gen(z)
            loss = bce_loss_fn(disc(x_fake), ONES)
            loss.backward()
            opt_gen.step()
            gen_loss_float = loss.cpu().item()

            if global_step % 100 == 0:
                print(f'global_step {global_step} | disc_loss {disc_loss_float} | gen_loss {gen_loss_float}')

            if global_step % 1_0000 == 0:
                tblogger.add_scalar('disc_loss', disc_loss_float, global_step=global_step)
                tblogger.add_scalar('gan_loss', gen_loss_float, global_step=global_step)
                tblogger.add_images('fake_image', x_fake.cpu().view(-1, 1, 28, 28), global_step=global_step)
                tblogger.add_images('real_image', x_real.cpu().view(-1, 1, 28, 28), global_step=global_step)


if __name__ == '__main__':
    train()
    