# training


### *Scaling Laws for Neural Language Models*, OpenAI

Notations:
- D: data size
- N: model size (#of trainable parameters)
- C: compution
- perfromance measured by loss. 注意是用比较原始的Loss 而非更复杂的corr, AUC等统计量metrics, 这个很make sense。
- 有个疑问：N和C是强相关的，怎么分离开？尝试相同 #trainable params 但不同 Flops eg. self_attention vs mlp ?

几个主要结论：

- **performance depends strongly on scale, weekly on model shape(depth/width or arch)** performance只跟 N,D,C 有关，跟结构无关
- 两个倍数关系. 
    + loss ~ (N, D)：**smooth power laws**. power law意思是 N, D 要翻几倍，loss才会线性的降低几个点
    + N ~ D: **university of overfitting**. 这个讲的是模型大小和需要的数据大小之间的关系 $N^0.74/D$ ie. model_size x8 需要 data_size x5, (todo)这个是跑很多实验得出的结论？
- **University of training** 根据前几个epoch/step预测后面最终loss能降到多少
- **Transfer improves with test performance** transfer to different distribution incurs constant penalty. (todo)根据trian->validation的metrics下降，估计train->test上的metrcis下降？
- Sample efficiency
    + **Sample efficiency** 表现为大模型收敛更快：相同data大模型loss更低
    + **convergence is inefficient**
    + 这俩讲的是同一个事：大模型对样本的利用率更高，这里的利用率指的是多一个样本or多一次step能涨多少点。所以当 computation budget 有限时，应该用 大模型+少steps 而非 小模型+convergence
    + 金融数据场景: NLP的场景可以认为数据量是无限的，而金融数据的场景数据有限，所以会观测到大模型达到plateau快但overfit也快，小模型收敛慢但如果不小心多训了几个epoch它也不会overfit太多，比较稳。最终各自收敛到什么水平，没有robust的结论，因为金融数据噪音太大了，本身epochs之间、seed之间的波动性就盖过了不同model/training_method之间的区别
- **Optimal batch size** gradient noise scale 相同团队的另一篇文章 *An Empirical Model of Large-Batch Training* TODO


