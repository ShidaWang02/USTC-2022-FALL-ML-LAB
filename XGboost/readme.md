# LAB3 

XGBoost

By Yanwu Gu 2022.10.26

## 1. 实验原理

由于本次实验任务量大，故给大家提供了较为完整的理论推导，仔细阅读本节能够有助于你完成实验。

### XGBoost

XGBoost 是由多个基模型组成的一个加法模型，假设第 $k$ 个基本模型是 $f_k (x)$, 那么前 $t$ 个模型组成的模型的输出为

$$
\hat y_i^{(t)}=\sum^t_{k=1}f_k (x_i )=\hat y_i^{(t-1)}+f_t (x_i)
$$

其中 $x_i$ 为第表示第 $i$ 个训练样本，$y_i$ 表示第 $i$ 个样本的真实标签;  $\hat y_i^{(t)}$ 表示前 $t$ 个模型对第 $i$ 个样本的标签最终预测值。

在学习第 $t$ 个基模型时，XGBoost 要优化的目标函数为:

$$
\begin{split}
Obj^{(t)} &= \sum_{i=1}^n loss(y_i,\hat y_i^{(t)})+\sum_{k=1}^t penalty(f_k)\\
&=\sum_{i=1}^n loss(y_i,\hat y_i^{(t-1)}+f_t(x_i))+\sum_{k=1}^t penalty(f_k)\\
&=\sum_{i=1}^n loss(y_i,\hat y_i^{(t-1)}+f_t(x_i))+ penalty(f_t)+constant\\
\end{split}
$$

其中 $n$ 表示训练样本的数量, $penalty(f_k)$ 表示对第 $k$ 个模型的复杂度的惩罚项,  $loss(y_i,\hat y_i^{(t)})$ 表示损失函数,

例如二分类问题的 

$$
loss(y_i, \hat y_i^{(t)} )=−y_i\cdot \log⁡ p(\hat y_i^{(t)}=1|x_i)−(1−y_i)\log⁡ (1-p(\hat y_i^{(t)}=1|x_i))
$$

回归问题

$$
loss(y_i, \hat y_i^{(t)} )=(y_i- \hat y_i^{(t)} )^2
$$

将 $loss(y_i,y_i^{(t-1)}+f_t(x_i))$ 在 $y_i^{(t-1)}$ 处泰勒展开可得

$$
loss(y_i,y_i^{(t-1) }+f_t (x_i))≈loss(y_i,y_i^{(t-1)} )+g_i f_t (x_i )+\frac12 h_i f_t^2 (x_i)
$$

其中 $g_i=\frac{\partial\ loss(y_i,y_i^{(t-1)})}{\partial\  y_i^{(t-1) } }$, $h_i=\frac{\partial^2 loss(y_i,y_i^{(t-1)} )}{\partial \ (y _i^{(t-1)} )^2 }\\ $，即 $g_i$ 为一阶导数，$h_i$ 为二阶导数。

此时的优化目标变为

$$
Obj^{(t)}=∑_{i=1}^n[loss(y_i,y_i^{(t-1)} )+g_i f_t (x_i )+\frac12 h_i f_t^2 (x_i)]+penalty(f_t ) +constant
$$

去掉常数项 $loss(y_i,y_i^{(t-1) })$ (学习第 $t$ 个模型时候， $loss(y_i,y_i^{(t-1) })$ 也是一个固定值) 和 constant，可得目标函数为

$$
Obj^{(t)}=\sum_{i=1}^n[g_i f_t (x_i )+\frac12 h_i f_t^2 (x_i)]+penalty(f_t )
$$

### 决策树（回归树）

本实验中，我们以决策树（回归树）为基，因此还需要写出决策树的算法。

假设决策树有 $T$ 个叶子节点，每个叶子节点对应有一个权重。决策树模型就是将输入 $x_i$ 映射到某个叶子节点，决策树模型的输出就是这个叶子节点的权重，即 $f(x_i )=w_{q(x_i )}$ ，$w$ 是一个要学的 $T$ 维的向量其中 $q(x_i)$ 表示把输入 $x_i$ 映射到的叶子节点的索引。例如：$q(x_i )=3$，那么模型输出第三个叶子节点的权重，即 $f(x_i )=w_3$。

我们对于某一棵决策树，他的惩罚为

$$
penalty(f)=\gamma\cdot T+\frac12\lambda\cdot\|w\|^2
$$

其中 $\gamma,\lambda$ 为我们可调整的超参数，$T$ 为叶子数，$w$ 为权重向量. 由于显示问题，$\|w\|$ 实际上为 $w$ 的范数，且 $\|w\|^2=\sum_{i=1}^{dim}w_i^2$

我们将分配到第 $j$ 个叶子节点的样本用 $I_j$ 表示，即 $I_j=\{i|q(x_i )=j\} (1≤j≤T)$。

综上，我们在树结构确定（你可以自行确定）时，可以进行如下优化：

$$
\begin{split}
Obj^{(t)}&=\sum_{i=1}^n[g_i f_t (x_i )+\frac12 h_i f_t^2 (x_i)]+penalty(f_t )\\
&= \sum_{i=1}^n[g_iw_{q(x_i )} +\frac12  h_i w_{q(x_i )}^2]+\gamma \cdot T+\frac12 \lambda \cdot\|w\|^2\\
&= \sum_{j=1}^T[(\sum_{i\in I_j}g_i )⋅𝑤_𝑗+\frac12\cdot(\sum_{i\in I_j}h_i+\lambda )\cdot w_j^2 ]+\gamma\cdot T
\end{split}
$$

简单起见，我们简记 $G_j=\sum_{i\in I_j}g_i , H_j=\sum_{i\in I_j}h_i $

$$
Obj^{(t)}=\sum_{j=1}^T[G_jw_j+\frac12(H_j+\lambda)w_j^2]+\gamma T
$$


 **在上述推导之后，你应该可以推导出最优的权重（对 $w_j$ 优化），请在你的报告中写出这个权重的表达式，同时需要写出这棵决策树的得分 Obj.** 

### 构造过程

对于每一棵决策树，即每一个基的训练，我们可以按照以下步骤划分结点

1. 从根节点开始递归划分，初始情况下，所有的训练样本 $x_i$ 都分配给根节点。

2. 根据划分前后的收益划分结点，收益为

   $$
   Gain = Obj_P-Obj_L-Obj_R
   $$
   
   其中 $Obj_P$ 为父结点的得分，$Obj_L,Obj_R$ 为左右孩子的得分.

3. 选择最大增益进行划分

选择最大增益的过程如下：

1. 选出所有可以用来划分的特征集合 $\mathcal F$；
2. For feature in $\mathcal F$:
3. ​        ​        将节点分配到的样本的特征 feature 提取出来并升序排列，记作 sorted_f_value_list；
4. ​        ​        For f_value in sorted_f_value_list ：
5. ​​        ​        ​        ​        在特征 feature 上按照 f_value 为临界点将样本划分为左右两个集合；
6. ​        ​        ​        ​        计算划分后的增益；
7. 返回最大的增益所对应的 feature 和 f_value。 

### 停止策略

对于如何决定一个节点是否还需要继续划分，我们提供下列策略，你可以选择一个或多个，或自行设定合理的策略

- 划分后增益小于某个阈值则停止划分；
- 划分后树的深度大于某个阈值停止划分；
- 该节点分配到的样本数目小于某个阈值停止分化。

对于整个算法如何终止，我们提供下列策略，你可以选择一个或多个，或自行设定合理的策略

- 学习 $M$ 个颗决策树后停下来；
- 当在验证集上的均方误差小于某个阈值时停下来；
- 当验证集出现过拟合时停下来。

### 评价指标

你可以在实验中以下列指标来验证你的算法效果和不同参数对于结果的影响

- $RMSE=\sqrt{\frac1m\sum_{i=1}^m(y_{test}^{(i)}-\hat y_{test}^{(i)})^2}\\ $，越小越好，

- $R^2=1-\frac{ \sum_{i=1}^m(y_{test}^{(i)}-\hat y_{test}^{(i)})^2}{\sum_{i=1}^m(\bar y_{test}-\hat y_{test}^{(i)})^2}=1-\frac{MSE(\hat y_{test},y_{test})}{Var(y_{test})}\\ $，越大越好

- 运行时间

仍然，这些标准不会作为评分的标准

## 2. 实验数据

在 `train.data` 文件中，有 7154 条 41 维的数据，其中前 40 列为 feature，最后一列为 label.

## 3. 任务及要求

### 3.1 任务

1. 完成决策树（回归树）的算法
2. 完成 XGBoost 的算法
3. 书写你的报告

报告应当含有以下内容和其他你觉得必要的内容

1. 有关文档中提出问题的解答
2. 有关于你**两个**停止策略的选择，你关于树的选择，你关于超参数的选择
3. 实验结果的展示（最佳的模型）
4. 不同参数的比较
5. 训练过程 loss 的可视化

一些提示

1. 理论上 XGBoost 应该是一个框架，不应与基的选择有关系，但本实验中不做要求，但应将决策树算法和 XGBoost 算法做明显的区分

2. 本实验可以不做数据预处理

3. 你应当选择合适的数据结构来存储你关于决策树的参数

4. 给出一个回归树的结构示例，仅作参考
   ```python
   class RegressionTree(object):
       def __init__(self,): # 初始化回归树
       def _get_best_split(self,): # 获得最佳feature和split
       def _get_split_score(self,): # 获取某一划分的目标函数值  
       def _choose_split_point(self,): # 获取最佳的划分点
       def fit(self,):# 训练一棵回归树 
       def _predict(self,): # 预测一个样本 
       def predict(self,): # 预测多条样本
   ```

5. 由于 XGBoost 效果很好，对于树的各种属性的选择，在比较经济的情况下也能得到不错的结果，节约时间和内存的开销

###  3.2 要求

- 禁止使用`` sklearn`` 或者其他的机器学习库，你只被允许使用`numpy`, `pandas`, `matplotlib`, 和 [Standard Library](https://gitee.com/link?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Findex.html), 你需要从头开始编写这个项目。
- 你可以和其他同学讨论，但是你不可以剽窃代码，我们会用自动系统来确定你的程序的相似性，一旦被发现，你们两个都会得到这个项目的零分。

## 4. 提交

* 报告推荐格式

    1. 实验目的（可选）
    2. 实验原理（若不重要可以简要说明）
    3. 实验步骤（从读取数据、模型训练、使用xx的参数，xx的模型，得到了多少组的结果，总之就是你在每块代码做了什么事情）
    4. 实验结果（对输出进行总结、比较、可视化）
    5. 实验分析（分析结果出现的原因、分析原因）

* 提交 .zip 文件，包含以下内容（请直接对这两个文件打包）

  --main.ipynb

  --Report.pdf

* 请命名你的文件为 `LAB3_PBXXXXXXXX_中文名.zip`, 例如 `LAB3_PB19061297_顾言午`, **对于错误命名的文件，我们将不会计算分数，请注意，这次实验开始我们会严格这方面的规定**

* 请发邮件至 [ml_2022_fall@163.com](mailto:ml_2022_fall@163.com) 附带您的文件，在截止日期之前

* **截止日期: 2022.11.20 23:59:59** 