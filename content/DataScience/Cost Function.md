---
title: 
draft: false
tags:
  - model
  - regression
  - logistic
---


>[!Cost Function]
>在机器学习中，成本函数（Cost Function），也叫损失函数（Loss Function），用于衡量模型的预测结果与实际结果之间的差异。成本函数的值越小，表示模型的预测越准确。通过最小化成本函数，我们可以训练模型，使其预测更加准确。

# 回归问题的成本函数

## 1. 均方误差（Mean Squared Error, MSE）

### 概念
均方误差（MSE）是用于回归问题的常用成本函数。它通过计算预测值和实际值之间的差的平方，然后取平均值来衡量模型的性能。

### 公式
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

## 2. 绝对误差和（Sum of Absolute Differences, SAD）

### 概念
绝对误差和（SAD）计算的是预测值与实际值之间差异的绝对值和。

### 公式
$$
\text{SAD} = \sum_{i=1}^{N} |y_i - \hat{y}_i| 
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

# 计算SAD
sad = np.sum(np.abs(y_true - y_pred))
print(f"SAD: {sad}")
```

## 3. 平方差和（Sum of Squared Differences, SSD）

### 概念
平方差和（SSD）计算的是预测值与实际值之间差异的平方和。与均方误差不同，SSD 不取平均值。

### 公式
$$
\text{SSD} = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 
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

# 计算SSD
ssd = np.sum((y_true - y_pred) ** 2)
print(f"SSD: {ssd}")
```

# 分类问题的成本函数

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

# 成本函数的比较和选择

## 回归问题
- **均方误差（MSE）**：适用于数据误差分布较均匀的情况，对较大误差更敏感。
- **绝对误差和（SAD）**：适用于存在极端值的情况，对所有误差平等对待。
- **平方差和（SSD）**：与MSE类似，但不需要计算平均值，适用于某些特定应用场景。

## 分类问题
- **对数损失（Log Loss）**：适用于分类问题，特别是需要考虑预测概率的场景，对错误预测的概率惩罚较高。

# 图示解释

## Log Loss 图示
![[LogLoss.png]]

## Log Loss 与 MSE 的比较图示
![[MSE.png]]


# 总结

- **成本函数**：用于衡量模型预测与实际结果的差异，通过最小化成本函数来训练模型。
- **MSE、SAD、SSD**：用于回归问题的不同成本函数，选择时需考虑数据特性和应用场景。
- **Log Loss**：用于分类问题，计算预测概率与实际标签之间的对数差异。
