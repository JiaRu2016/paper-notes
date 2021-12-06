# RL

## David Silver Course

### 08_Model-based RL

- "model" is estimating the env: modeling $\eta$: $P(s_{t+1}|s_t,a_t)$, $R(s_t, a_t)$
- Model-based RL: Learn model from experience, and **plan** value function and/or policy from simulated experience
- Dyna: **learn and plan** ... 
    + Dyna-Q altrithom: 在平行世界里(in previsous state s)用model代替env,再多更新几次Q(s,a)
- MC tree search
    + from current root
    + only search sub-paths of the tree: search and evaluate dynamically, only go deeper to current best (greedy search)


## awesome

- DQN. *Playing Atari with Deep Reinforcement Learning*    
- DQN + target Q net. *Human-level control through deep reinforcement learning*      
- Policy Gradient. *Policy Gradient Methods for RL with Function Approximation* 主要是推导得到了 Policy gradient Themorm    
- A3C. *Asynchronous Methods for Deep Reinforcement Learning*   
- Duel Q network. *Dueling Network Architectures for Deep Reinforcement Learning*    
- *high dimentional continguous control using GAE*

## not so owesome

*Action Branching Architectures for Deep Reinforcement Learning* 解决action是多维的问题，这个方法正常人都能想出来。 based on dualing q network. 

## interesting applications

### 麻将AI *Suphx: Mastering Mahjong with Deep Reinforcement Learning*   

by MSRA. 看视频就ok

- Encode tiles as 4 * 34 matrix
- model: 5 models combine with descition flow。 Models are if_吃/碰/杠 and 丢牌模型.
    + mdoel input: D * 34 * 1, D contains private tiles, open hands, history, manaully features etc.
    + model output: 34 * 1 for discard model; scaler for chi/pong/kong model.
    + 3 * 1 Conv with 256 channels repeat 50x skip connected.
- training: supervised learning using human players action as label, then self-play with the trained models as policy.
- trick 1: Oracle guiding. use full infomation as teacher 常见套路
- trick 2: global reward predictor as critic. 考虑风险偏好下的reward?
- trick 3: online "finetune" to priavate tiles at hand


### 斗地主AI *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*  

- Encode card as 4 * 15 matrix
- state: history moves encode to (T, 4, 15) tensor
- action: leagal movement encode to (4, 15) matrix
- model: for each leagal movement ie. action, `Concate(a, LSTM(s)) | MLP(6,512) | scaler`, output is state action value `Q(s,a)`
- use MC with DNN
    + why MC? long horizon and sparse reward
    + why not DQN? large and variable action space, `max_a Q(s,a)` is computational expensive
    + why not policy gradient? inifinte action space. While action as feature can generalize eg. `3KKK` to `3JJJ`


### 滴滴打车派单算法 *Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach*

- define one day as one episode, 
- state defined as `s = (grid, time)`,  offline learn `V(s)`, thus we know `Q(s,a) = r + V(s')`
- oneline dispatch: solve `a = argmax_a Q(s, a)` with KM algriom. 一种确定性算法，输入二分图及其权重，输出一个使权值和最大的匹配方案，这里权重就是offline估计出来的`Q(s,a)`
