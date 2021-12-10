## DL invasion to other fields


### 预测蛋白质折叠结构 AlphaFold

领域知识：

- 氨基酸一维结构唯一确定其三维结构，所以建模为“输入基酸序列，预测三维坐标”这样一个问题
    + 氨基酸有21种，氨基酸序列长度大概在几百，是transformer能处理的长度
- multial suqence alignment, MSA. 找类似于“协整”关系，有这样关系的两个氨基酸在空间中位置更近
    + AlphaFold1 作为特征放进channels中
    + AlphaFold2 直接融进模型，通过 row/col-wise attention 学到


AlphaFold1: Two Step

- input features `(L, L, d)`, Conv blocks, output pairwise distance and torsion distribution prediciton `(L, L, c)`
- 有了两两距离和旋转角度矩阵，求三维坐标。这是一个纯数学问题，用梯度下降这种数值方法求解

AlphaFold2: End-to-end and use Transforemer instead of CNN

model
- inpout sequence `(r,)` 构造两种特征 1. MSA from database search `(s,r,c)` 2. pair `(r,r,c)`
- [ `(MSA, pair)` -> Evoformer x48 -> `(MSA, pair)` -> Structure module x8 -> 3D structure] recycling 3 times
- Evoformer: details ref to figure 3 and supplementary
    + MSA: row/col-wise gated attention, add pair as attention bias, then feed-forward MLP c -> 4c -> c
    + outer product mean added to `pair`
    + triangle multiplicative update 类似于对偶图上做卷积操作，motivation是预测出的 distance matrix 必须满足三角不等式
- Structure module: todo
