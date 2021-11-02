import torch
from model import Hparam, BertModel, MP_BertModel
from data import data_pipeline


def construct_optimizer(model: BertModel, hp: Hparam):
    Kls, kwargs = hp.optimizer
    return Kls(model.parameters(), **kwargs)


def construct_model(hp, dev):
    ModelKls = BertModel if isinstance(dev, torch.device) else MP_BertModel
    return ModelKls(hp).set_and_mvto_device(dev)


def get_devices(name):
    if name == 'tiny':
        return torch.device(0)
    elif name == 'base':
        return list(map(torch.device, [0,1]))
    elif name == 'large':
        return list(map(torch.device, [0,1,2,3]))


def train():
    # hp and data
    hp_name = 'base'
    hp = Hparam(hp_name)
    data = data_pipeline(hp)
    hp.set_data_related_fields(data)
    it = iter(data.train_batch_generator)
    
    # model
    device = get_devices(hp.name)
    model = construct_model(hp, device)
    optimizer = construct_optimizer(model, hp)

    print('start training ...')
    for step in range(hp.num_step):
        batch = next(it)
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f'step {step} | loss {loss.item()}')


if __name__ == '__main__':
    train()
    # CUDA_VISIBLE_DEVICES=5,6,7,8 python train.py