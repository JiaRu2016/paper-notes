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


### A Three-Way Model for Collective Learning on Multi-Relational Data

**Factorization** method of repr learning

$$
X[i,j,k] = \text{1 if triple exists else 0}
$$
$$
\min_{A, R_k} \sum_k ||X - A R_k A^T||_F^2 + regularize(A, R_k) \\
$$

so $A \in R^{N, F}$ is entity embedding, $R_k \in R^{F,F}$ (flatten) is relation k embedding


###  Emebedding entites and relations for learning and inference in knowledge basees

general framework of KG embedding:
```
S = ScoreFunction(Emb(e1), Emb(e2), Emb(r))
loss = max(Spos - Sneg + 1, 0)  # eg. Margin Ranked Loss
```

1. compare different score functions, two basic component Linear and Bilinear.
(Bilinear is weighted FM, Bilinear-diag ie. DistMult)
2. mine logical rules. TODO


### TransE: Add

head + relation(label) should be similar to tail

$$
[\gamma + d(h + l, t) - d(h^\prime + l, t^\prime)]_+
$$

### ComplEX

$$
P(Y_{so} == 1) = \sigma(X_{so})
$$
where $Y_{so}$ is paritally observed.

SVD: $X = UV^T$, where $U$ and $V$ is embedding of entites as subject and object. **We need a joint embedding of subject and object**

if $X$ is symetric, exist decomosition $EWE^T$. But X is not. So consider complex number:

$$
X = Re(E W \bar E^T)
$$

since Y is partially observed, to enable the model to be learnable, ie. generalized to unobserved links, need some regularity: since y is binary, we impose $E W \bar E^T$ has low *sign rank* K (K is hparam).

definition of sign rank: min rank of a real matrix whose sign == Y
$$
rank_\pm(Y) = \min_A \{rank(A) | sign(A) == Y\}
$$