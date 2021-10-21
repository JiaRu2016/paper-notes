import torch
import torch.nn as nn
from torch.optim import Adam
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time


# hparam tunning experience:
# - Adam beta1 = 0.5 
# - d, noise_dim should be small

class Hparam:
    IMG_SIZE = 25
    LABEL_VOCAB_SIZE = 10

    def __init__(self):
        self.bz = 32
        self.noise_dim = 50
        self.d = 16
        self.label_embedding_dim = 16
        self.disc_n_step = 1
        self.lr = 2e-4
        self.beta1 = 0.5


def get_real_img_dataloader(bz, img_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.CenterCrop(size=(img_size, img_size)), 
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    p = '/home/rjia/datasets/torch_datasets/'
    ds_train = torchvision.datasets.MNIST(p, train=True, transform=transform)
    ds_test = torchvision.datasets.MNIST(p, train=False, transform=transform)
    ds = ConcatDataset([ds_train, ds_test])
    return DataLoader(ds, batch_size=bz, shuffle=True, drop_last=True)


def noise_iterator(bz, noise_dim):
    while True:
        yield torch.randn((bz, noise_dim, 1, 1))

def image_label_iterator(bz, img_size):
    dloader = get_real_img_dataloader(bz, img_size)
    it = iter(dloader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(dloader)


class Discriminator(nn.Module):
    def __init__(self, hp: Hparam):
        super().__init__()
        self.d = hp.d
        self.IMG_SIZE = hp.IMG_SIZE
        self.emb_dim = hp.IMG_SIZE ** 2
        self.LABEL_VOCAB_SIZE = hp.LABEL_VOCAB_SIZE
        self.emb = nn.Embedding(self.LABEL_VOCAB_SIZE, self.emb_dim)
        self._disc = nn.Sequential(
            # (bz, 1+1, 25, 25)
            nn.Conv2d(2, 1 * self.d, 4, 1),
            nn.BatchNorm2d(self.d),
            nn.LeakyReLU(0.2, True),
            # (bz, d, 22, 22)
            nn.Conv2d(1 * self.d, 2 * self.d, 4, 2),
            nn.BatchNorm2d(2 * self.d),
            nn.LeakyReLU(0.2, True),
            # (bz, 2d, 10, 10)
            nn.Conv2d(2 * self.d, 4 * self.d, 4, 2),
            nn.BatchNorm2d(4 * self.d),
            nn.LeakyReLU(0.2, True),
            # (bz, 4d, 4, 4)
            nn.Conv2d(4 * self.d, 1, 4, 2),
            # (bz, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x: FloatTensor, label: LongTensor) -> FloatTensor:
        emb_channel = self.emb(label).view((-1, 1, self.IMG_SIZE, self.IMG_SIZE))
        x = torch.cat([x, emb_channel], dim=1)
        return self._disc(x).view(-1)


class Generator(nn.Module):
    def __init__(self, hp: Hparam):
        super().__init__()
        self.noise_dim = hp.noise_dim
        self.d = hp.d
        self.label_embedding_dim = hp.label_embedding_dim
        self.LABEL_VOCAB_SIZE = hp.LABEL_VOCAB_SIZE
        self.emb = nn.Embedding(self.LABEL_VOCAB_SIZE, self.label_embedding_dim)
        self._gen = nn.Sequential(
            # (bz, noise_dim, 1, 1)
            nn.ConvTranspose2d(self.noise_dim + self.label_embedding_dim, 4 * self.d, 4, 2),
            nn.BatchNorm2d(4 * self.d),
            nn.ReLU(True),
            # (bz, 4d, 4, 4)
            nn.ConvTranspose2d(4 * self.d, 2 * self.d, 4, 2),
            nn.BatchNorm2d(2 * self.d),
            nn.ReLU(True),
            # (bz, 2d, 10, 10)
            nn.ConvTranspose2d(2 * self.d, 1 * self.d, 4, 2),
            nn.BatchNorm2d(1 * self.d),
            nn.ReLU(True),
            # (bz, d, 22, 22)
            nn.ConvTranspose2d(1 * self.d, 1, 4, 1),
            # (bz, d, 25, 25)
            nn.Tanh(),
        )
    
    def forward(self, z: FloatTensor, label: LongTensor) -> FloatTensor:
        z_and_label = torch.cat([z, self.emb(label).view(-1, self.label_embedding_dim, 1, 1)], dim=1)
        return self._gen(z_and_label)


def train():
    hp = Hparam()
    # data
    image_label_iter = image_label_iterator(hp.bz, hp.IMG_SIZE)
    noise_iter = noise_iterator(hp.bz, hp.noise_dim)
    # model, opt and loss_fn
    device = torch.device(0)
    gen = Generator(hp).to(device)
    disc = Discriminator(hp).to(device)
    opt_gen = Adam(gen.parameters(), lr=hp.lr, betas=(hp.beta1, 0.999))
    opt_disc = Adam(disc.parameters(), lr=hp.lr, betas=(hp.beta1, 0.999))
    bce_loss_fn = nn.BCELoss()
    ONES = torch.ones((hp.bz,)).to(device)
    ZEROS = torch.zeros((hp.bz,)).to(device)
    # logs
    tblogger = SummaryWriter(log_dir=f'./tblog/{int(time.perf_counter())}')
    # eval thread
    eval_thread = evaluate(hp, tblogger, device)
    eval_thread.send(None)

    step = 0
    while True:
        step += 1

        for k in range(hp.disc_n_step):
            x_real, label = next(image_label_iter)
            # print(x_real.shape)  # (bz, 1, 25, 25)
            # print(label.shape)   # (bz,)
            # return
            x_real = x_real.to(device)
            label = label.to(device)
            z = next(noise_iter).to(device)
            opt_disc.zero_grad()
            x_fake = gen(z, label).detach()
            loss = bce_loss_fn(disc(x_real, label), ONES) + bce_loss_fn(disc(x_fake, label), ZEROS)
            loss.backward()   
            opt_disc.step()
            disc_loss_float = loss.cpu().item()

        z = next(noise_iter).to(device)
        opt_gen.zero_grad()
        x_fake = gen(z, label)
        loss = bce_loss_fn(disc(x_fake, label), ONES)
        loss.backward()
        opt_gen.step()
        gen_loss_float = loss.cpu().item()

        if step % 100 == 0:
            print(f'step {step} | disc_loss {disc_loss_float} | gen_loss {gen_loss_float}')

        if step % 500 == 0:
            tblogger.add_scalar('disc_loss', disc_loss_float, global_step=step)
            tblogger.add_scalar('gan_loss', gen_loss_float, global_step=step)
            print(f'step {step} | evaluate...')
            eval_thread.send((step, gen))


def evaluate(hp: Hparam, tblogger: SummaryWriter, device: torch.device):
    fixed_noise = next(noise_iterator(bz=50, noise_dim=hp.noise_dim))
    labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).repeat_interleave(5)
    fixed_noise, labels = fixed_noise.to(device), labels.to(device)
    while True:
        step, gen_model = yield
        with torch.no_grad():
            fake = gen_model(fixed_noise, labels)
            fake = fake.cpu().view(-1, 1, Hparam.IMG_SIZE, Hparam.IMG_SIZE)
        tblogger.add_images('fake', fake, global_step=step)


if __name__ == '__main__':
    train()
    