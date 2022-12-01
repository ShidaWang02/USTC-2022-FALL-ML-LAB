# LAB4

Density Peak Clustering

By HanLei 2022.11.21

## 1. 实验原理
聚类相关知识详情请回顾课件第九章《聚类》，本次聚类实验主要实现的是《**Clustering by fast search and find of density peaks**》一文中的算法（以下简称**DPC**）
- By Alex Rodriguez and Alessandro Laio
- Published on SCIENCE, 2014
- [https://sites.psu.edu/mcnl/files/2017/03/9-2dhti48.pdf](https://sites.psu.edu/mcnl/files/2017/03/9-2dhti48.pdf)
- $\color{red}{ 每位同学务必仔细阅读原论文}$

### 算法思想
集成了 k-means 和 DBSCAN 两种算法的思想
- 聚类中心周围密度较低，中心密度较高
- 聚类中心与其它密度更高的点之间通常都距离较远


### 算法流程
1. Hyperparameter: a distance threshold $d_c$
2. For each data point $i$, compute two quantities:
	- Local density: $\rho_i = \sum_{j}\chi(d_{ij}-d_c)$,where $\chi(x)=1$ if $x<0$ and $\chi(x)=0$ otherwise
	- Distance from points of higher density: $\delta_i=\mathop {min}\limits_{j:\rho_j>\rho_i} d_{ij}$
	&nbsp;&nbsp;&nbsp; <font size=1>· For the point with highest density, take $\delta_i=\mathop {max}\limits_{j} d_{ij}$</font> 
3. Identify the cluster centers and out-of-distribution (OOD) points
- Cluster centers: with both high $\rho_i$ and $\delta_i$
- OOD points: with high $\delta_i$ but low $\rho_i$
- Draw a decision graph, and make decisions manually

## 2. 实验数据

本次实验采用 3 个 2D 数据集（方便可视化）
- Datasets/D31.txt
- Datasets/R15.txt
- Datasets/Aggregation.txt

数据格式
- 每个文件都是普通的 txt 文件，包含一个数据集
- 每个文件中，每一行表示一条数据样例，以空格分隔

注意事项
- 允许对不同的数据集设置不同的超参数

## 3. 任务及要求

### 3.1 任务
#### 3.1.1 实验简介
本次实验的总体流程是完成 DPC 算法的代码实现，并在给定数据集上进行可视化实验。具体来说，同学们需要实现以下步骤
1. 读取数据集，（如有必要）对数据进行预处理
2. 实现 DPC 算法，计算数据点的 $\delta_i$ 和 $\rho_i$
3. 画出$\color{red}{决策图}$，选择样本中心和异常点
4. 确定分簇结果，计算$\color{red}{评价指标}$，画出$\color{red}{可视化图}$

助教除大致浏览代码外，以以下输出为评价标准：
- 可视化的决策图
- 可视化的聚类结果图
- 计算出的评价指标值 （DBI）
- 输出只要在**合理范围**内即可，不作严格要求

实验结果需要算法代码和实验报告
- 助教将通过可视化结果和代码来确定算法实现的正确性
- 助教将阅读实验报告来检验同学对实验和算法的理解


#### 3.1.2 评价指标
- 本次实验采用 Davis-Bouldin Index (DBI) 作为评价指标

- 建议$\color{red}{统一调用}$ sklearn.metrics. davies_bouldin_score 进行计算

#### 3.1.3 数据可视化
- 本次实验需要画两个二维散点图：决策图和聚类结果图
- 可视化库推荐 pyplot (也可自行选择别的工具，此处只做教程)
- 代码片段演示
```python
# 产生测试数据
import matplotlib.pyplot as plt
import numpy as np

x1 = np.arange(1,10) 
x2 = x1**2
fig = plt.figure() 
ax1 = fig.add_subplot(111)
# 设置每个样本点的颜色（用于聚类结果展示）
colors = ['r','y','g','b','r','y','g','b','r']
# 设置标题
ax1.set_title('Scatter Plot') 
# 设置X轴标签
plt.xlabel('X') 
# 设置Y轴标签
plt.ylabel('Y') 
# 画散点图
ax1.scatter(x1, x2, c=colors, marker='o') 
# 显示所画的图
plt.show() 
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/bdd1047ff8b34e3689111ebb1ecb3646.png)

### 3.2 要求

- 禁止使用`` sklearn`` 或者其他的机器学习库，你只被允许使用`numpy`, `pandas`, `matplotlib`, 和 [Standard Library](https://gitee.com/link?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Findex.html), 你需要从头开始编写这个项目。
- 你可以和其他同学讨论，但是你不可以剽窃代码，我们会用自动系统来确定你的程序的相似性，一旦被发现，你们两个都会得到这个项目的零分。

## 4. 提交
- 实验报告可参考[关于LAB2的一些反馈](https://gitee.com/Sqrti/ml_2022_f/issues/I5YJU8)
- 报告推荐格式

  1. 实验目的（可选）
  2. 实验原理（若不重要可以简要说明）
  3. 实验步骤（从读取数据、模型训练、使用xx的参数，xx的模型，得到了多少组的结果，总之就是你在每块代码做了什么事情）
  4. 实验结果（对输出进行总结、比较、可视化）
  5. 实验分析（分析结果出现的原因、分析原因）

- 提交 .zip 文件，包含以下内容（请直接对这两个文件打包）

  --main.ipynb

  --Report.pdf

- 请命名你的文件为 `LAB3_PBXXXXXXXX_中文名.zip`, **对于错误命名的文件，我们将不会计算分数**

- 请发邮件至 [ml_2022_fall@163.com](mailto:ml_2022_fall@163.com) 附带您的文件，在截止日期之前

- **截止日期: 2022.12.11 23:59:59** 
- 对于迟交的作业，请参考 [这里](https://gitee.com/Sqrti/ml_2022_f#%E4%B8%80%E5%85%B3%E4%BA%8E%E8%AF%BE%E7%A8%8B)