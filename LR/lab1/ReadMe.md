# Machine Learning Lab1

Logistic Regression

>王世炟 PB20151796

本次实验是基于[Loan Data Set | Kaggle](https://www.kaggle.com/datasets/burak3ergun/loan-data-set)数据集进行的逻辑回归模型代码编写与训练。

数据存储在`loan.csv`中。

实验代码分为两部分，\
`Logistic.py` 中是逻辑回归的模型代码，包含`fit()`、`predict()`、`ShowLoss()`等方法 \
`Loan.ipynb`中包含`Load Dataset`、`Data Cleaning`、`Encode`、`Data Process`、`Train`、`Test`、`Cross Validation`等模块，分别实现了以上功能。

运行时打开`Loan.ipynb`从头开始运行即可，程序会自动打印迭代轮数，画出损失函数曲线，并进行5次5折交叉验证，输出平均准确率。