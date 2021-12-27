# Knowledge Graph

### Conv2D KG embedding

https://arxiv.org/abs/1707.01476


- link prediction task
- Dataset: WordNet18RR (One big connected graph)

- score function:
$$
f(vec(conv2d(\bar e_s, \bar r_r ) W)) e_o
$$

like to answer question: "[head] 's [relation] is ___"

```python
# input: (s, r, o)
# forward
e_s_ = Embedding(s).reshape(m, n)
r_r_ = Embedding(r).reshape(m, n)
e_o = Embedding(o)
score = sigmoid(Linear(Conv2D(Concat[e_s_, r_r_]) e_o)
# loss
BCELoss(score, ONE_ZERO_TARGET)  # 1 if (s, r, o) exist else 0. 
    # code: use all train nodes as negtive sample
```