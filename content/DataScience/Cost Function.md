---
title: 
draft: false
tags:
  - model
  - regression
  - logistic
---
# Cost Function

在机器学习中，成本函数（Cost Function），也叫损失函数（Loss Function），用于衡量模型的预测结果与实际结果之间的差异。成本函数的值越小，表示模型的预测越准确。通过最小化成本函数，我们可以训练模型，使其预测更加准确。

# 均方误差（Mean Squared Error, MSE）

## 概念
均方误差（MSE）是用于回归问题的常用成本函数。它通过计算预测值和实际值之间的差的平方，然后取平均值来衡量模型的性能。

## 公式
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 
$$

其中：
- $N$ 是样本数量。
- $y_i$ 是第 $i$ 个样本的实际值。
- $\hat{y}_i$ 是第 $i$ 个样本的预测值。

### 示例代码
```python
import numpy as np

# 实际值和预测值
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 1.9, 3.2, 4.0, 5.1])

# 计算MSE
mse = np.mean((y_true - y_pred) ** 2)
print(f"MSE: {mse}")
```

## 对数损失（Log Loss）

### 概念
对数损失（Log Loss）是用于分类问题的常用成本函数。它通过计算预测概率和实际标签之间的对数差异来衡量模型的性能。对数损失越小，表示模型的预测概率越接近实际分类。

### 公式
对于二分类问题，公式如下：
$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

其中：
- $N$ 是样本数量。
- $y_i$ 是第 $i$ 个样本的实际标签（0 或 1）。
- $p_i$ 是模型对第 $i$ 个样本预测为 1 的概率。

### 示例代码
```python
from sklearn.metrics import log_loss

# 实际标签和预测概率
y_true = [0, 0, 1, 1]
y_pred_proba = [0.1, 0.4, 0.35, 0.8]

# 计算对数损失
logloss = log_loss(y_true, y_pred_proba)
print(f"Log Loss: {logloss}")
```

## MSE 与 Log Loss 的比较

### 何时使用 MSE
- 适用于回归问题（预测连续值）。
- 简单直观，通过平方差来衡量预测误差。

### 何时使用 Log Loss
- 适用于分类问题（预测类别）。
- 考虑预测概率，惩罚错误分类的概率更高。

### 两者的不同
- **MSE**：对误差进行平方，强调大误差。
- **Log Loss**：对预测概率进行对数计算，惩罚错误预测的概率更高，特别适用于分类问题。

### 图示解释

#### Log Loss 图示
![[LogLoss.png]]
#### Log Loss 与 MSE 的比较图示
![[MSE.png]]


- **成本函数**：用于衡量模型预测与实际结果的差异，通过最小化成本函数来训练模型。
- **MSE**：用于回归问题，计算预测值与实际值之间的平方差的平均值。
- **Log Loss**：用于分类问题，计算预测概率与实际标签之间的对数差异。
