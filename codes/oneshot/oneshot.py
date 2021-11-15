import torch
import torch.nn as nn
import torchvision
import numpy as np
from typing import Dict, List, Tuple
import tqdm

ImgTensor = torch.FloatTensor
DictDS = Dict[int, List[ImgTensor]]


class Hparam():
    def __init__(self) -> None:
        # omniglot dataset: 1 * 105 * 105 img, with 20 samples for each class
        self.IMG_HW_DIM = 105
        self.IMG_N_CHANNEL = 1
        self.N_SAMPLE_PER_CLASS = 20
        self.bz = 32
        self.optimizer = (torch.optim.Adam, {'lr': 1e-3})
        self.pair_or_triple = 'pair'
        self.num_step = 10000
        self.eval_nway = 20
        self.eval_bz = 32
        self.eval_num_step = 10


################################################################################
# SECTION data
################################################################################

class DataPipline:
    def __init__(self, hp: Hparam) -> None:
        self.hp = hp
        self.ds_train = self.__get_ds(is_train=True)
        self.ds_eval = self.__get_ds(is_train=False)
        self.d_train = self.__ds_to_dict(self.ds_train)
        self.d_eval = self.__ds_to_dict(self.ds_eval)

    def __get_ds(self, is_train):
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        ds = torchvision.datasets.Omniglot(
            root='/home/rjia/playground/datasets_torch', 
            background=is_train, 
            transform=trans, 
            target_transform=None, 
            download=True,
        )
        return ds

    def __ds_to_dict(self, ds):
        d = {}
        for img, label in tqdm.tqdm(ds, desc='ds_to_dict'):
            if label not in d:
                d[label] = []
            d[label].append(img)
        return d
    
    def __sample_one__pair(self) -> Tuple[ImgTensor, ImgTensor, bool]:
        n_class = len(self.d_train)
        is_same = np.random.rand() > 0.5
        c1, c2 = np.random.randint(n_class, size=(2,))
        i1, i2 = np.random.randint(self.hp.N_SAMPLE_PER_CLASS, size=(2,))
        img1 = self.d_train[c1][i1]
        if is_same:
            img2 = self.d_train[c1][i2]
        else:
            img2 = self.d_train[c2][i2]
        return img1, img2, is_same
    
    def __sample_one__triple(self) -> Tuple[ImgTensor, ImgTensor, ImgTensor]:
        n_class = len(self.d_train)
        c, c_neg = np.random.randint(n_class, size=(2,))
        i, i_pos, i_neg = np.random.randint(self.hp.N_SAMPLE_PER_CLASS, size=(3,))
        img_anchor = self.d_train[c][i]
        img_pos = self.d_train[c][i_pos]
        img_neg = self.d_train[c_neg][i_neg]
        return img_anchor, img_pos, img_neg

    def __collate__pair(self, samples):
        return (
            torch.stack([tp[0] for tp in samples]),
            torch.stack([tp[1] for tp in samples]),
            torch.BoolTensor([tp[2] for tp in samples]),
        )
    
    def __collate__triple(self, samples):
        return (
            torch.stack([tp[0] for tp in samples]),
            torch.stack([tp[1] for tp in samples]),
            torch.stack([tp[2] for tp in samples]),
        )
    
    def train_batch_generator(self):
        collate_fn = {'pair': self.__collate__pair, 'triple': self.__collate__triple}[self.hp.pair_or_triple]
        sample_fn =  {'pair': self.__sample_one__pair, 'triple': self.__sample_one__triple}[self.hp.pair_or_triple]
        while True:
            yield collate_fn([sample_fn() for _ in range(self.hp.bz)])

    def eval_batch_generator(self):
        """one-shot-N-way batch
        yield (img_query, img_classes, class_idx)
        img_query: (bz, 1, C, H, W)
        img_classes: (bz, nway, C, H, W)
        class_idx: (bz,)
        """
        d = self.d_eval
        nclass = len(d)
        nway = self.hp.eval_nway
        nshot = self.hp.N_SAMPLE_PER_CLASS
        EvalSample = Tuple[ImgTensor, ImgTensor, int]
        EvalBatch = Tuple[ImgTensor, ImgTensor, torch.LongTensor]

        def _sample_fn() -> EvalSample:
            c_arr = np.random.choice(nclass, size=(nway,), replace=False)  # arr[70, 80, 42, ... 900]
            i_arr = np.random.randint(nshot, size=(nway,))   # arr[5, 1, 4, ... 19]
            cls_idx = np.random.randint(nway)  # 2
            c = c_arr[cls_idx]  # 42
            i = np.random.randint(nshot)
            img_q = d[c][i].unsqueeze(0)  # (1, C, H, W)
            img_cls = torch.stack([d[_c][_i] for _c, _i in zip(c_arr, i_arr)])  # (nway, C, H, W)
            return img_q, img_cls, cls_idx
        
        def _collate_fn(samples: List[EvalSample]) -> EvalBatch:
            return (
                torch.stack([tp[0] for tp in samples]),
                torch.stack([tp[1] for tp in samples]),
                torch.LongTensor([tp[2] for tp in samples]),
            )

        while True:
            yield _collate_fn([_sample_fn() for _ in range(self.hp.eval_bz)])


################################################################################
# SECTION model
################################################################################

class SiameseNet(nn.Module):
    def __init__(self, hp: Hparam):
        super().__init__()
        self.hp = hp
        # layers
        self.conv1 = nn.Conv2d(1, 8, 3, 2, False)  # 105*105 -> 52*52
        self.conv2 = nn.Conv2d(8, 64, 3, 2, False)  # -> 25*25
        self.conv3 = nn.Conv2d(64, 128, 3, 2, False)  # -> 12*12
        self.conv4 = nn.Conv2d(128, 128, 3, 2, False)  # -> 5*5
        self.fc1 = nn.Linear(3200, 512)
        self.fc2 = nn.Linear(512, 128)
        self.a = torch.relu
        self.distance_score = nn.Linear(128, 1, bias=False)
        self.loss_fn_pair = nn.BCEWithLogitsLoss()
        self.loss_fn_triple = nn.TripletMarginLoss()
    
    def forward(self, *inputs):
        if self.hp.pair_or_triple == 'pair':
            img1, img2, is_same = inputs
            return self.forward_pair(img1, img2, is_same)
        else:
            img, img_pos, img_neg = inputs
            return self.forward_tirple(img, img_pos, img_neg)

    def forward_pair(self, img1, img2, is_same):
        # img1, img2: (bz, 1, 105, 105) FloatTensor
        # is_same: (bz,) BoolTensor
        h1 = self.__extract_feature(img1)  # (bz, 128)
        h2 = self.__extract_feature(img2)  # (bz, 128)
        d = torch.abs(h1 - h2)
        score = self.distance_score(d).view(-1)  # (bz,)
        loss = self.loss_fn_pair(score, is_same.float())
        return loss
    
    def forward_tirple(self, img, img_pos, img_neg):
        h = self.__extract_feature(img)  # (bz, 128)
        h_pos = self.__extract_feature(img_pos)  # (bz, 128)
        h_neg = self.__extract_feature(img_neg)  # (bz, 128)
        loss = self.loss_fn_triple(h, h_pos, h_neg)
        return loss


    def __extract_feature(self, img):
        x = self.a(self.conv1(img))
        x = self.a(self.conv2(x))
        x = self.a(self.conv3(x))
        x = self.a(self.conv4(x))
        x = x.flatten(1, -1)
        x = self.a(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
    
    @torch.no_grad()
    def one_shot_n_way_eval(self, img_query, img_classes, class_idx):
        C, H, W = self.hp.IMG_N_CHANNEL, self.hp.IMG_HW_DIM, self.hp.IMG_HW_DIM
        bz, nway = class_idx.shape[0], img_classes.shape[1]
        assert img_query.shape == (bz, 1, C, H, W)
        assert img_classes.shape == (bz, nway, C, H, W)
        assert class_idx.shape == (bz,)
        self.eval()
        h_query = self.__extract_feature(img_query.expand(-1, nway, -1, -1, -1).flatten(0,1))
        h_classes = self.__extract_feature(img_classes.flatten(0,1))
        d = torch.abs(h_query - h_classes)  # (bz * nway, )
        score = self.distance_score(d)      # (bz * nway, )
        pred_class_idx = score.view(bz, nway).argmax(dim=1)  # (bz,)
        return pred_class_idx == class_idx


def example_batch_train(hp: Hparam):
    bz, C, H, W = hp.bz, hp.IMG_N_CHANNEL, hp.IMG_HW_DIM, hp.IMG_HW_DIM
    while True:
        img1 = torch.randn(size=(bz, C, H, W))
        img2 = torch.randn(size=(bz, C, H, W))
        is_same = torch.randint(0, 1, size=(bz,)) > 0.5
        yield img1, img2, is_same


def example_batch_oneshotnway_eval(hp: Hparam):
    bz, C, H, W = hp.bz, hp.IMG_N_CHANNEL, hp.IMG_HW_DIM, hp.IMG_HW_DIM
    nway = hp.eval_nway
    while True:
        img_query = torch.randn(size=(bz, 1, C, H, W))
        img_classes = torch.randn(size=(bz, nway, C, H, W))
        class_idx = torch.randint(nway, size=(bz,))
        yield img_query, img_classes, class_idx


################################################################################
# SECTION train
################################################################################

def train():
    hp = Hparam()
    # data
    data = DataPipline(hp)
    batch_iter_train = iter(data.train_batch_generator())
    #batch_iter_train = iter(example_batch_train(hp))
    
    # model and optimizer
    model = SiameseNet(hp)
    OptKls, opt_kwargs = hp.optimizer
    optimizer = OptKls(model.parameters(), **opt_kwargs)

    # eval thread
    eval_thread = evaluate(data, hp)
    eval_thread.send(None)
    
    for step in range(hp.num_step):
        batch = next(batch_iter_train)
        # forward backward
        optimizer.zero_grad()
        model.train()
        loss = model(*batch)
        loss.backward()
        optimizer.step()
        
        # log
        if step % 10  == 0:
            print(f'[train] step {step} | loss {loss.item()}')
        
        if step % 100 == 0:
            eval_thread.send((model, step))


def evaluate(data: DataPipline, hp: Hparam):
    batch_iter_eval = iter(data.eval_batch_generator())
    #batch_iter_eval = iter(example_batch_oneshotnway_eval(hp))
    while True:
        model, step = yield
        __evaluate(model, batch_iter_eval, step, hp)


def __evaluate(model: SiameseNet, batch_iter_eval, step, hp: Hparam):
    acc = []
    for _ in range(hp.eval_num_step):
        batch = next(batch_iter_eval)
        acc_batch = model.one_shot_n_way_eval(*batch)
        acc.append(acc_batch)
    acc = torch.cat(acc).float().mean().item()
    print(f'[eval] step {step} | one-shot-n-way Accuracy {acc}')


if __name__ == '__main__':
    train()