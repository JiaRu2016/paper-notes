# paper-notes
ML, DL paper reading notes


## optimization / training diagram

*Training ImageNet in 1 Hour* by Facebook

*Visualizing the Loss Landscape of Neural Nets*

CyclicalLR *Cyclical Learning Rates for Training Neural Networks*

*Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*

*CurriculumLearning* 2009_ICML


## distributed DL

hogwild! *Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent*

Parameter Server. *Scaling Distributed Machine Learning with the Parameter Server*, *Communication Efﬁcient Distributed Machine Learning with the Parameter Server*


## DL Framework and CS-related

BP impl. CSE599W

*PyTorch Distributed- Experiences on Accelerating Data Parallel Training* 看视频更好，overlapping compute and communication 原理和 pytorch 实现细节

GPipe *Efficient Training of Giant Neural Networks using Pipeline Parallelism* 以及pytorch实现 *torchgpipe: On-the-ﬂy Pipeline Parallelism for Training Giant Models*

*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*

混合精度训练 *mixed precision training*

为什么砍了计算量推理性能还是不变？可能跟访存有关 *Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*

## NLP / seq models

move to [NLP_and_seq_model](./NLP_and_seq_model.md)

## CV

go to [CV](./CV.md)

## RL

move to [RL.md](./RL.md)

## GNN

mv to [GNN.md](./GNN.md)


## unsupervised / GAN

GAN *Generative Adversarial Nets*. Define the min-max math problem and give training algriothm. Theoretical proof optimial D `=p_data / (p_data + p_g)` and G `p_G = p_data`. See code `gan.py`


## FewShot

`code/onetshot/oneshot.py`. code according to https://www.youtube.com/playlist?list=PLvOO0btloRnuGl5OJM37a8c6auebn-rH2

- Learn "is two image same class" instead of "class of an image".
- Model arch: distance of two image's hidden feature representation `d = abs(h1 - h2)` where `h_1or2 = ConvNet(img_1or2)`
- Loss: binary classification is_same, or margin-triplete-loss `max(0, margin + d(x, x_pos) - d(x, x_neg)`
- evaluate: one-shot-n-way evaluation. hardmax the calss with hightest score.


## tree


## quant

AAAI_2021, 8 papers