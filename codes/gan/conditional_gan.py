import torch
import torch.nn as nn
from torch.optim import Adam
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time


class Hparam:
    def __init__(self):
        self.bz = 32
        self.FLATTEN_IMG_DIM = 784
        self.IMG_HW = 28
        self.noise_dim = 64
        self.gen_hidden_dim = 256
        self.disc_hidden_dim = 128
        self.label_embedding_dim = 16
        self.LABEL_VOCAB_SIZE = 10
        self.disc_n_step = 1
        self.lr = 3e-4


def get_real_img_dataloader(bz):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    p = '/home/rjia/playground/datasets_torch'
    ds_train = torchvision.datasets.MNIST(p, train=True, transform=transform)
    ds_test = torchvision.datasets.MNIST(p, train=False, transform=transform)
    ds = ConcatDataset([ds_train, ds_test])
    def __collate_fn(samples):
        # samples: (img_Tensor, label_int) where img_Tensor is (1,28,28)
        image = torch.stack([tp[0].squeeze(0) for tp in samples]).flatten(1,2)
        label = torch.LongTensor([tp[1] for tp in samples])
        return image, label
    return DataLoader(ds, batch_size=bz, shuffle=True, collate_fn=__collate_fn, drop_last=True)


def noise_iterator(bz, noise_dim):
    while True:
        yield torch.randn((bz, noise_dim))

def image_label_iterator(bz):
    dloader = get_real_img_dataloader(bz)
    it = iter(dloader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(dloader)


class Discriminator(nn.Module):
    def __init__(self, hp: Hparam):
        super().__init__()
        self.FLATTEN_IMG_DIM = hp.FLATTEN_IMG_DIM
        self.hidden_dim = hp.disc_hidden_dim
        self.label_embedding_dim = hp.label_embedding_dim
        self.LABEL_VOCAB_SIZE = hp.LABEL_VOCAB_SIZE
        self.emb = nn.Embedding(self.LABEL_VOCAB_SIZE, self.label_embedding_dim)
        self._disc = nn.Sequential(
            nn.Linear(self.FLATTEN_IMG_DIM + self.label_embedding_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: FloatTensor, label: LongTensor) -> FloatTensor:
        x = torch.cat([x, self.emb(label)], dim=1)
        return self._disc(x).view(-1)


class Generator(nn.Module):
    def __init__(self, hp: Hparam):
        super().__init__()
        self.noise_dim = hp.noise_dim
        self.hidden_dim = hp.gen_hidden_dim
        self.FLATTEN_IMG_DIM = hp.FLATTEN_IMG_DIM
        self.IMG_HW = hp.IMG_HW
        self.label_embedding_dim = hp.label_embedding_dim
        self.LABEL_VOCAB_SIZE = hp.LABEL_VOCAB_SIZE
        self.emb = nn.Embedding(self.LABEL_VOCAB_SIZE, self.label_embedding_dim)
        self._gen = nn.Sequential(
            nn.Linear(self.noise_dim + self.label_embedding_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, self.FLATTEN_IMG_DIM),
            nn.Tanh()
        )
    
    def forward(self, z: FloatTensor, label: LongTensor) -> FloatTensor:
        z_and_label = torch.cat([z, self.emb(label)], dim=1)
        return self._gen(z_and_label)
    
    @torch.no_grad()
    def predict(self, z: FloatTensor, label: LongTensor) -> FloatTensor:
        fake = self(z, label)
        return fake.view(-1, 1, self.IMG_HW, self.IMG_HW)


def train():
    hp = Hparam()
    # data
    image_label_iter = image_label_iterator(hp.bz)
    noise_iter = noise_iterator(hp.bz, hp.noise_dim)
    # model, opt and loss_fn
    device = torch.device(0)
    gen = Generator(hp).to(device)
    disc = Discriminator(hp).to(device)
    opt_gen = Adam(gen.parameters(), lr=hp.lr)
    opt_disc = Adam(disc.parameters(), lr=hp.lr)
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
            # print(x_real.shape)  # (bz, 784)
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

        if step % 1_0000 == 0:
            tblogger.add_scalar('disc_loss', disc_loss_float, global_step=step)
            tblogger.add_scalar('gan_loss', gen_loss_float, global_step=step)
            eval_thread.send((step, gen))


def evaluate(hp: Hparam, tblogger: SummaryWriter, device: torch.device):
    fixed_noise = next(noise_iterator(bz=50, noise_dim=hp.noise_dim))
    labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).repeat_interleave(5)
    fixed_noise, labels = fixed_noise.to(device), labels.to(device)
    while True:
        step, gen_model = yield
        fake = gen_model.predict(fixed_noise, labels)
        tblogger.add_images('fake', fake, global_step=step)


if __name__ == '__main__':
    train()
    