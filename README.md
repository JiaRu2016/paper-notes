# paper-notes
ML, DL paper reading notes


## optimization / training diagram

## DL Framework, distributed DL, and CS-related

## NLP / seq models

## RL

### awesome

### not so owesome

### interesting applications

suphx *Suphx: Mastering Mahjong with Deep Reinforcement Learning*   麻将AI by MSRA. 看视频就ok

- Encode tiles as 4 * 34 matrix
- model: 5 models combine with descition flow。 Models are if_吃/碰/杠 and 丢牌模型.
    + mdoel input: D * 34 * 1, D contains private tiles, open hands, history, manaully features etc.
    + model output: 34 * 1 for discard model; scaler for chi/pong/kong model.
    + 3 * 1 Conv with 256 channels repeat 50x skip connected.
- training: supervised learning using human players action as label, then self-play with the trained models as policy.
- trick 1: Oracle guiding. use full infomation as teacher 常见套路
- trick 2: global reward predictor as critic. 考虑风险偏好下的reward?
- trick 3: online "finetune" to priavate tiles at hand


DouZero *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*  斗地主

- Encode card as 4 * 15 matrix
- state: history moves encode to (T, 4, 15) tensor
- action: leagal movement encode to (4, 15) matrix
- model: for each leagal movement ie. action, `Concate(a, LSTM(s)) | MLP(6,512) | scaler`, output is state action value `Q(s,a)`
- use MC with DNN
    + why MC? long horizon and sparse reward
    + why not DQN? large and variable action space, `max_a Q(s,a)` is computational expensive
    + why not policy gradient? inifinte action space. While action as feature can generalize eg. `3KKK` to `3JJJ`


## GNN / spatio-temperal


## tree


## quant

AAAI_2021, 8 papers