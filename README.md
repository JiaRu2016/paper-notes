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

- model: 5 models combine with descition flow。 Models are if_吃/碰/杠 and 丢牌模型. modeling tiles as 4 * 34 matrix
    + mdoel input: D * 34 * 1, D contains private tiles, open hands, history, manaully features etc.
    + model output: 34 * 1 for discard model; scaler for chi/pong/kong model.
    + 3 * 1 Conv with 256 channels repeat 50x skip connected.
- training: supervised learning using human players action as label, then self-play with the trained models as policy.
- trick 1: Oracle guiding. use full infomation as teacher 常见套路
- trick 2: global reward predictor as critic. 考虑风险偏好下的reward?
- trick 3: online "finetune" to priavate tiles at hand



## GNN / spatio-temperal


## tree


## quant

AAAI_2021, 8 papers