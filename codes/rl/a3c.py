import torch
import torch.nn as nn
from torch import Tensor
import multiprocessing as py_builtin_mp
import torch.multiprocessing as mp
import numpy as np
import gym
from typing import Dict, Tuple, Optional, Union
import ut

BatchTensor = Tensor      # Tensor with first dim being batch_size
ScalerTensor = Tensor     # Scaler Tenser, typically loss or general backward-able
BatchArray = np.ndarray   # array stands for "batch"
SampleArray = np.ndarray  # array stands for "one sample", not batch
MpList = 'py_builtin_mp.managers.ListProxy'
MpDict = 'py_builtin_mp.managers.DictProxy'


class Hparam:
    def __init__(self, **kwargs) -> None:
        ########### optimization ##########
        self.hogwild_n_process = 4
        self.batch_episodes = 16  # num of episodes to play for generating a batch
        self.gamma = 0.95
        self.lr = 0.01
        self.num_step = 1_0000
        self.entropy_penalty = None
        ########### model struct ##########
        self.policy_net_hidden_dim = 32
        self.value_net_hidden_dim = 32
        ########################################
        self.seed = 0
        self.gpu_id = None
        self.__set_slot(**kwargs)

    def __set_slot(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class NetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = None

    def set_and_to_device(self, device: torch.device):
        self.device = device
        self.to(device)
        return self


class PolicyNet(NetBase):
    def __init__(self, state_dim, n_action, hidden_dim, entropy_penalty):
        super().__init__()
        self.STATE_DIM = state_dim
        self.N_ACTION = n_action
        self.linear1 = nn.Linear(self.STATE_DIM, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.N_ACTION)
        self.softmax = nn.Softmax(dim=1)
        self.entropy_penalty = entropy_penalty
    
    def __forward(self, state: BatchTensor) -> BatchTensor:
        """return prob, shape = (bz, N_ACTION)
        """
        return self.softmax(self.linear2(self.activation(self.linear1(state))))
    
    @staticmethod
    def __sum_action_entropy(prob: BatchTensor) -> ScalerTensor:
        return -(prob * torch.log(prob)).sum()
        
    def forward(self, state: BatchTensor, action: BatchTensor, advantage: BatchTensor) -> ScalerTensor:
        """reuturn loss that can be .backward()-ed directly
        state: (bz, STATE_DIM)
        action: (bz,)
        value: (bz,)
        """
        prob = self.__forward(state)  # (bz, state_dim) -> (bz, action_dim)
        prob = prob.gather(dim=1, index=action.view(-1,1)).view(-1)  # prob[action] -> (bz,)
        log_prob = torch.log(prob)
        loss =  (- log_prob * advantage).sum()
        if self.entropy_penalty is not None:
            loss += -self.__sum_action_entropy(prob) * self.entropy_penalty
        return loss
    
    def act(self, state_array: SampleArray) -> int:
        """return a sampled action, state_array
        """
        # array to batchsize-one tensor
        assert isinstance(state_array, np.ndarray) and state_array.shape == (self.STATE_DIM,)
        state = torch.from_numpy(state_array).float().view(1,-1).to(self.device)
        # __forwrad() get probs
        self.eval()
        with torch.no_grad():
            prob = self.__forward(state)  # (1, action_dim)
        # sample action
        prob_array = prob.view(-1).cpu().numpy()
        action = np.random.choice(self.N_ACTION, p=prob_array)
        return action


class ValueNet(NetBase):
    def __init__(self, state_dim, n_action, hidden_dim):
        super().__init__()
        self.STATE_DIM = state_dim
        self.N_ACTION = n_action
        self.linear1 = nn.Linear(self.STATE_DIM, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.N_ACTION)
        
    def forward(self, state: BatchTensor, action: BatchTensor) -> BatchTensor:
        """return Q(s, a)[action]. shape = (bz,)
        """
        q = self.linear2(self.activation(self.linear1(state)))  # (bz, state_dim) -> (bz, action_dim)
        v = q.gather(dim=1, index=action.view(-1,1)).view(-1)   # Q(s,a)[a] (bz, action_dim) -> (bz,)
        return v


def play_and_get_experience(policy_net: PolicyNet, env: gym.Env, n_episode: int, gamma: float):
    batch = {
        'state': [],
        'action': [],
        'value': [],
    }
    batch_total_return = []  # for on-the-fly evaluation

    for ep in range(n_episode):
        trajectory = {
            'state': [],
            'action': [],
            'reward': [],
            'value': [],   # future cumulative discounted ret, G_t0 = \sum_t \gamma^(t-t0) * r_t
        }
        
        s = env.reset()
        total_return = 0.0
        is_done = False
        while not is_done:
            a = policy_net.act(s)
            new_s, r, is_done, _ = env.step(a)
            trajectory['state'].append(s)
            trajectory['action'].append(a)
            trajectory['reward'].append(r)
            total_return += r
            s = new_s
        
        # calcuate Vt
        reward_lst = trajectory['reward']
        trajectory['value'] = [__discounted_total_reward(reward_lst[t:], gamma) for t in range(len(reward_lst))]

        batch['state'].append(np.stack(trajectory['state'], axis=0))
        batch['action'].append(np.array(trajectory['action']))
        batch['value'].append(np.array(trajectory['value']))
        batch_total_return.append(total_return)
        
    batch['state'] = np.concatenate(batch['state'])
    batch['action'] = np.concatenate(batch['action'])
    batch['value'] = np.concatenate(batch['value'])
    avg_total_return = np.mean(batch_total_return)
    
    return batch, avg_total_return


def __discounted_total_reward(reward_lst, gamma):
    return sum([r * gamma ** t for t, r in enumerate(reward_lst)])


def get_advantage(value_net: ValueNet, batch: Dict[str, BatchTensor]) -> BatchTensor:
    state = batch['state']
    action = batch['action']
    mc_empirical_return = batch['value']  # new policy, MC empirical return ie. realized Gt.
    value_net.eval()
    with torch.no_grad():
        baseline_value = value_net(state, action)   # current policy value (predicted)
    advantage = (mc_empirical_return - baseline_value)
    return advantage


def update_policy_net(policy_net: PolicyNet, optimizer, batch):
    policy_net.train()
    optimizer.zero_grad()
    policy_net(batch['state'], batch['action'], batch['advantage']).backward()
    optimizer.step()

def update_value_net(value_net: ValueNet, optimizer, batch):
    mc_empirical_return = batch['value']
    value_net.train()
    optimizer.zero_grad()
    predicted_v = value_net(batch['state'], batch['action'])
    ((predicted_v - mc_empirical_return) ** 2).sum().backward()
    optimizer.step()

def get_tensor_batch(batch, device):
    d = {'action': torch.LongTensor}
    return {
        k: d.get(k, torch.FloatTensor)(v).to(device) for k, v in batch.items()
    }
    # return {
    #     'state': torch.FloatTensor(batch['state']).to(device),
    #     'action': torch.LongTensor(batch['action']).to(device),
    #     'advantage': torch.FloatTensor(batch['advantage']).to(device),
    #     'value': torch.FloatTensor(batch['value']).to(device),
    # }

def __train(
    rank: int, 
    hp: Hparam, 
    policy_net: PolicyNet, 
    value_net: ValueNet, 
    avg_total_reward_history: Union[MpList, list],
) -> None:
    # local seed
    local_seed = hp.seed * 1_0000 + rank
    # local env / data generator
    env = gym.make('CartPole-v1')
    env.seed(local_seed)
    # local optimizer. NOTE: model.shared_memory() shares parameter while grad is not shared, thus use local optimizer
    torch.manual_seed(local_seed)
    value_optimizer = torch.optim.RMSprop(value_net.parameters(), lr=hp.lr)
    policy_optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=hp.lr)
    # device
    device = torch.device(hp.gpu_id if hp.gpu_id is not None else 'cpu')

    # train loop and evalutate on-the-fly
    tm = ut.Timer(disabled=True)
    local_num_step = hp.num_step // hp.hogwild_n_process
    for step in range(local_num_step):
        # get batch data
        tm.enter('play_and_get_experience')
        array_batch, avg_total_return = play_and_get_experience(policy_net, env, n_episode=hp.batch_episodes, gamma=hp.gamma)
        
        # eval
        tm.enter('print')
        avg_total_reward_history.append(avg_total_return)
        print(f'rank {rank} step {step} | avg_total_return {avg_total_return}')
        if avg_total_return >= 500: 
            print(f'rank {rank} step {step} | train succeed.')
            return

        # one step model update
        tm.enter('update_net')
        batch = get_tensor_batch(array_batch, device)
        batch['advantage'] = get_advantage(value_net, batch)
        update_policy_net(policy_net, policy_optimizer, batch)
        update_value_net(value_net, value_optimizer, batch)

        # penalty\lr scheduler
        if hp.entropy_penalty is not None:
            if step % 10 == 0 and step > 0:
                policy_net.entropy_penalty /= 2

    tm.summary()
    

def train_hogwild(hp: Hparam, result_dict: Optional[MpDict] = None):
    # get env data DIMS
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    n_action = env.action_space.n
    del env

    # shared model
    device = torch.device(hp.gpu_id if hp.gpu_id is not None else 'cpu')
    value_net = ValueNet(state_dim, n_action, hp.value_net_hidden_dim)
    policy_net = PolicyNet(state_dim, n_action, hp.policy_net_hidden_dim, hp.entropy_penalty)
    value_net.set_and_to_device(device)
    policy_net.set_and_to_device(device)
    value_net.share_memory()
    policy_net.share_memory()

    if hp.hogwild_n_process == 1:
        # degenerated case, for debug
        avg_total_reward_history = []
        __train(0, hp, policy_net, value_net, avg_total_reward_history)
    else:
        # hogwild!
        ctx = py_builtin_mp.get_context('fork') if hp.gpu_id is None else mp.get_context('spawn')
        avg_total_reward_history = ctx.Manager().list()
        processes = []
        for rank in range(hp.hogwild_n_process):
            p = ctx.Process(target=__train, args=(rank, hp, policy_net, value_net, avg_total_reward_history))
            p.start()
            processes.append(p)
        [p.join() for p in processes]
        avg_total_reward_history = [r for r in avg_total_reward_history]

    # save result
    if result_dict is not None:
        result_dict[hp.seed] = avg_total_reward_history


if __name__ == '__main__':
    # hp = Hparam(hogwild_n_process=2, num_step=30, gpu_id=0)
    # train_hogwild(hp)

    n_run = 20
    num_step = 300
    use_gpu = True

    ctx = py_builtin_mp.get_context()
    result_dict = ctx.Manager().dict()

    process_lst = []
    for i in range(n_run):
        hp = Hparam(num_step=num_step, seed=i, gpu_id=i % 10 if use_gpu else None)
        p = ctx.Process(target=train_hogwild, args=(hp, result_dict))
        p.start()
        process_lst.append(p)
    
    for p in process_lst:
        p.join()
    
    def save_results(result_dict):
        import json
        import os
        result_dict = {k: v for k, v in result_dict.items()}   # ProxyDict to dict
        filepath = f'{__file__.strip(".py")}_useGPU={use_gpu}.json'
        with open(filepath, 'w') as f:
            json.dump(result_dict, f)
    
    save_results(result_dict)
