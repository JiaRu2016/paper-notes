import gym
import numpy as np
import multiprocessing as py_builtin_mp
import torch
import torch.nn as nn
from torch import Tensor
from collections import deque
from typing import Optional, Tuple

BatchTensor = Tensor      # Tensor with first dim being batch_size
ScalerTensor = Tensor     # Scaler Tenser, typically loss or general backward-able
SampleArray = np.ndarray  # array stands for "one sample", not batch

# batch = state, action, reward, new_state, is_done
_Batch = Tuple[
    torch.FloatTensor,
    torch.LongTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.BoolTensor,
]


class Hparam:
    def __init__(self, **kwargs) -> None:
        ########### optimization ##########
        self.batch_size = 256
        self.capacity = 1_0000
        self.gamma = 0.999
        self.eps=0.3      # explore ratio
        self.lr = 0.01
        self.num_step = 1_0000
        ########### model struct ##########
        self.hidden_dim = 4
        ########################################
        self.max_episodes = 5000
        self.seed = 0
        self.__set_slot(**kwargs)

    def __set_slot(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dq = deque([], capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, new_state: Optional[np.ndarray], is_done: bool):
        self.dq.append((state, action, reward, new_state, is_done))
        
    def sample(self, batch_size: int) -> _Batch:
        indexs = np.random.randint(len(self.dq), size=batch_size)
        state, action, reward, new_state, is_done = [], [], [], [], []
        for i in indexs:
            s, a, r, ss, d = self.dq[i]
            state.append(s)
            action.append(a)
            reward.append(r)
            new_state.append(ss)
            is_done.append(d)
        
        return (
            torch.FloatTensor(np.stack(state, axis=0)),
            torch.LongTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.stack(new_state, axis=0)),
            torch.BoolTensor(np.array(is_done)),
        )
        

def random_play_and_init_buffer(env: gym.Env, buffer: ReplayBuffer):
    t = 0
    while True:
        s, is_done = env.reset(), False
        while not is_done:
            a = np.random.choice(env.action_space.n)
            ss, r, is_done, _ = env.step(a)
            buffer.push(s, a, r, ss, is_done)
            t += 1
            if t >= buffer.capacity:
                return


class QNet(nn.Module):
    def __init__(self, state_dim, n_action, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.n_action = n_action
        self.hidden_dim = hidden_dim
        self.module = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),   # NOTE: USE SMALL MODEL!
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_action),
        )
    
    def forward(self, x: BatchTensor):
        """return predicted Q(s,a)
        """
        return self.module(x)
    
    def predicted_q(self, state: BatchTensor, action: BatchTensor):
        """for train (backward && update weights)
        """
        return self(state).gather(dim=1, index=action.view(-1,1)).view(-1)

    def td_target(self, new_state: BatchTensor, reward: BatchTensor, is_done: BatchTensor, gamma):
        """for TD_target 
        TD_target = r + gamma * max_a Q(new_state, a)
        TD_target = r  ie. Q(new_state, *) = 0  if is_done
        """
        with torch.no_grad():
            q_values = self(new_state)                # (bz, state_dim) -> (bz, action_dim)
            max_q_value = q_values.max(dim=1).values  # (bz,)
            mask = 1 - is_done.float()
            td_target = reward + mask * gamma * max_q_value
        return td_target
        
    def greedy_act(self, s: SampleArray):
        s = torch.from_numpy(s).float().view(1, -1)
        with torch.no_grad():
            a = torch.argmax(self(s), dim=1).view(-1)
        return a.item()

    def epsilon_greedy_act(self, state: SampleArray, eps):
        """interact with env to generate samples
        """
        if np.random.rand() < eps:
            return np.random.choice(self.n_action)
        else:
            return self.greedy_act(state)


def update_qnet(qnet_target: QNet, qnet: QNet, optimizer, loss_fn, batch: _Batch, hp):
    state, action, reward, new_state, is_done = batch
    predicted_q = qnet.predicted_q(state, action)
    td_target = qnet_target.td_target(new_state, reward, is_done, hp.gamma)
    optimizer.zero_grad()
    loss_fn(predicted_q, td_target).backward()
    [param.grad.data.clamp_(-1, 1) for param in qnet.parameters()]
    optimizer.step()


def evaluate(qnet_target: QNet, env: gym.Env):
    s, is_done, total_reward = env.reset(), False, 0.0
    while not is_done:
        a = qnet_target.greedy_act(s)
        s, r, is_done, _ = env.step(a)
        total_reward += r
    return total_reward


def train(hp: Hparam, result_dict):
    # data
    env = gym.make('CartPole-v1')
    env.seed(hp.seed)
    n_action = env.action_space.n
    state_dim = env.observation_space.shape[0]

    buffer = ReplayBuffer(hp.capacity)
    random_play_and_init_buffer(env, buffer)

    # model and optimizer
    torch.manual_seed(hp.seed)
    qnet = QNet(state_dim=state_dim, n_action=n_action, hidden_dim=hp.hidden_dim)
    qnet_target = QNet(state_dim=state_dim, n_action=n_action, hidden_dim=hp.hidden_dim)
    qnet_target.load_state_dict(qnet.state_dict())
    qnet_target.eval()
    optimizer = torch.optim.RMSprop(qnet.parameters(), lr=hp.lr)
    loss_fn = nn.SmoothL1Loss()

    # train and evaluate
    train_total_reward_history = []
    eval_avg_total_reward_history = []

    for ep in range(hp.max_episodes):
        s, is_done, total_reward = env.reset(), False, 0.0
        while not is_done:
            # act and save samples from env to buffer
            a = qnet.epsilon_greedy_act(s, eps=hp.eps)
            ss, r, is_done, _ = env.step(a)
            total_reward += r
            buffer.push(s, a, r, ss, is_done)                
            s = ss
            
            # update qnet
            batch = buffer.sample(hp.batch_size)
            update_qnet(qnet_target, qnet, optimizer, loss_fn, batch, hp)

        train_total_reward_history.append(total_reward)
                        
        # sync to qnet_target
        if ep % 10 == 0:
            qnet_target.load_state_dict(qnet.state_dict())
        
        # evaluate
        if ep % 10 == 0:
            total_reward = [evaluate(qnet_target, env) for _ in range(10)]
            avg_total_reward = np.mean(total_reward)
            std_total_reward = np.std(total_reward)
            eval_avg_total_reward_history.append(avg_total_reward)
            print('ep {} | eval total_reward: {:.2f} +- {:.2f}'.format(
                ep, avg_total_reward, std_total_reward))
            
            # if (avg_total_reward >= 499.0 and std_total_reward < 1e-4):
            #     print('train succeed.')
            #     break
        
    print('train hit max_episodes.')
    result_dict[hp.seed] = eval_avg_total_reward_history
    #return qnet_target, eval_avg_total_reward_history



if __name__ == '__main__':

    n_run = 20
    max_episodes = 5000

    ctx = py_builtin_mp.get_context()
    result_dict = ctx.Manager().dict()
    process_lst = []
    for i in range(n_run):
        hp = Hparam(max_episodes=max_episodes, seed=i)
        p = ctx.Process(target=train, args=(hp, result_dict))
        p.start()
        process_lst.append(p)
    for p in process_lst:
        p.join()
    
    def save_results(result_dict):
        import json
        import os
        result_dict = {k: v for k, v in result_dict.items()}   # ProxyDict to dict
        filepath = f'{__file__.strip(".py")}.json'
        with open(filepath, 'w') as f:
            json.dump(result_dict, f)
    
    save_results(result_dict)
    
    # hp = Hparam()
    # train(hp, {})