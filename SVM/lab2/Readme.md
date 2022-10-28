# Machine Learning Lab2

SVM

By Yanwu Gu 2022.10.12

## 1. Theory of Supported Vector Machine

You can refer to the ppt or the Chap 6 of the textbook.

## 2. Data

In order to simplify the lab, we give the function `generate_data(dim, num)` for you to freely generate the data. The data was linearly separable, but added some mistakes intentionally. Features, labels and rate of mislabel will be given by the function respectively.

You do not need to modify the function`generate_data(dim, num)`.

## 3. Tasks, Tips and Requirements

### 3.1 Tasks

You are required to complete the class `SVM1` and `SVM2` using different methods to find the solution of the supported vector machine. More specificly, since the key of solving SVM is to solve the quadratic programming problem (6.6) in your textbook, you just need to use **two** methods to solve (6.6). The remaining part like predict can be the same. 

After finishing the SVM class, you need to test the efficiency of your code. The comparison must include

1. The accuracy,
2. The time of culculation (trainning),

If possible, you can use `sklearn` to compare with your code, feel free to be beaten by it. 

### 3.2 Tips

There are some tips for the lab:

1. We do not recommend you to use existing function to solve the **quadratic programming** problem directly, which will be penalised. Of course, if you cannot complete two methods from scratch, you can use library function.
2. We recommend you to use proper dims to make sure your result reliable, and different dims or numbers of examples will let your report rich in content. But do not let it verbose.
3. Since our data is based on linear kernel, you do not need to try other kernels. But you can try soft margin or regularization to improve the ability of your model. Remember it's not the key point of this lab. 
4. Remember to add your **mislabel rate**, which is generate by the function `generate_data` for us.

###  3.3 Requirements

- **Do not** use sklearn or other machine learning library, you are only permitted with numpy, pandas, matplotlib, and [Standard Library](https://gitee.com/link?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Findex.html), you are required to **write this project from scratch.**
- You are allowed to discuss with other students, but you are **not allowed to plagiarize the code**, we will use automatic system to determine the similarity of your programs, once detected, both of you will get **zero** mark for this project.

## 4. Submission

* Report

  * The method you use, and briefly talk about its principle
  * The result of your methods
  * The comparison of your methods

* Submit a .zip file with following contents

  --main.ipynb

  --Report.pdf

* Please name your file as `LAB2_PBXXXXXXXX.zip`, **for wrongly named file, we will not count the mark**

* Sent an email to [ml_2022_fall@163.com](mailto:ml_2022_fall@163.com) with your zip file before deadline

* **Deadline: 2022.10.30 23:59:59** 

* For late submission, please refer to [this](https://gitee.com/Sqrti/ml_2022_f#一关于课程)

# 机器学习 Lab2

支持向量机

By Yanwu Gu 2022.10.12

## 1. 支持向量机的理论

你可以参考演示文档或者书本第六章的相关内容。

## 2. Data

为了简化实验，我们为你给出了函数 `generate_data(dim, num)` 去自由地生成数据. 这个数据是线性可分的，但是故意在标签值加上了一些错误。特征、标签以及错标率会由函数依次给出。

你不需要去更改`generate_data(dim, num)`.

## 3. 任务，提示及要求

### 3.1 任务

你需要去完成类 `SVM1` 和 `SVM2` ，并且使用不同的算法去寻找支持向量机的解。 更具体地说，因为解决支持向量机的关键在于解决书本上的二次规划问题（6.6），你只需要使用两种不同的方法去解决（6.6）。剩下的部分，比如预测，内容可以相同。

在完成了类方法的部分之后，你需要测试你代码的效率。比较应当包含以下内容：

1. 正确率，
2. 计算（训练）的时间消耗。

如果可能的话，你可以使用 `sklearn` 与你的代码比较。如果比不过它，也是没事的。

### 3.2 提示

这里有一些实验的提示：

1. 我们不推荐你使用已有的库函数去**直接**解决二次规划问题，这是会被扣除一部分分数的。当然，如果你无法使用两种方法去解决，你也可以使用库函数。
2. 我们推荐你使用合适的维度去训练、测试，这会使你的结果更加可靠。同时，不同的维度和样本数也会使你的报告内容更丰富。但是不要让他过于冗杂。
3. 因为我们的数据是基于线性核生成的，你不需要尝试其他的核函数。但是你可以使用软间隔或者正则化等方法来提升你模型的能力。切记，这不是本实验的核心内容。
4. 记得添加你的**错标率**，它会由函数 `generate_data` 生成。

###  3.3 要求

- 禁止使用`` sklearn`` 或者其他的机器学习库，你只被允许使用`numpy`, `pandas`, `matplotlib`, 和 [Standard Library](https://gitee.com/link?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Findex.html), 你需要从头开始编写这个项目。
- 你可以和其他同学讨论，但是你不可以剽窃代码，我们会用自动系统来确定你的程序的相似性，一旦被发现，你们两个都会得到这个项目的零分。

## 4. 提交

* 报告

  * 你使用的理论，简要讨论它的原理，
  * 你方法的结果，
  * 你方法之间的比较。

* 提交 .zip 文件，包含以下内容

  --main.ipynb

  --Report.pdf

* 请命名你的文件为 `LAB2_PBXXXXXXXX.zip`, **对于错误命名的文件，我们将不会计算分数**

* 请发邮件至 [ml_2022_fall@163.com](mailto:ml_2022_fall@163.com) 附带您的文件，在截止日期之前

* **截止日期: 2022.10.30 23:59:59** 

* 对于迟交的作业，请点击 [这里](