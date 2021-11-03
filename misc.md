## Misc


### `xxNorm`: batchNorm, LayerNorm

BatchNorm group by channel C and calcuate pooled stats, ie. outer most bz dim and inner H/W/L dim
- `BatchNorm2d` input (N, C, H, W), calculate stat for each C, ie (N, H, W) as bz dimention
- `BatchNorm1d` input (N, C, L) or (N, C), calcuate state for each C, ie (N,) or (N, L) as bz dimention. (N, C) can be viewed as (N, C, 1)

LayerNorm. Take Transformer source code as example, input (seq_len, bz, d_model), `LayerNorm(normalized_shape=d_model)`, for each time_step and sample (t, i), normalize features. Thus padding do not go into non-padding-postion features

```python
############################################################
# BatchNorm2d
C, H, W = 3, 20, 20
bz = 32

img = torch.rand(bz, C, H, W)  # (bz, C, h, w)
bn2d = nn.BatchNorm2d(num_features=C, affine=False)
normed_img = bn2d(img)

# print(bn2d.running_mean)  # (C,)
# print(bn2d.running_var)   # (C,)

# # if affine == True:
# for w in bn2d.parameters():
#     print(w.shape)  # (3,)

for c in range(C):
    cube = img[:, c, :, :]
    mean, var = cube.mean(), cube.var(False)
    a = (cube - mean) / torch.sqrt(var + 1e-05)
    b = normed_img[:, c, :, :]
    assert torch.allclose(a, b, 1e-5, 1e-6, True)

############################################################
# LayerNorm
d_model = 512
seq_len = 10
bz = 32
x = torch.rand(seq_len, bz, d_model)
#print(x.shape)  # (seq_len, bz, d_model)

ln = nn.LayerNorm(normalized_shape=(d_model,), elementwise_affine=True)
out = ln(x)
#print(out.shape)  # (seq_len, bz, d_model)

# check
for t in range(seq_len):
    for i in range(bz):
        vec = x[t, i]
        mean, var = torch.mean(vec), torch.var(vec, False)
        a = (vec - mean) / torch.sqrt(var + 1e-05)
        b = out[t, i]
        assert torch.allclose(a, b, 1e-5, 1e-6, True)

# for w in ln.parameters():
#     print(w.shape)   # (d_model, )
```