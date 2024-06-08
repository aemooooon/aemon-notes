---
title: 
draft: false
tags:
  - model
date: 2024-06-09
---
![[FeatureScaling.png]]
*Using a standardised dataset allowed the model to converge much faster*
# Normalisation

## Min-Max 

>[!Definition]
>Normalisation is a data preprocessing technique used to scale features to a specified range, typically 0 to 1. This process ensures that different features contribute equally to the model and helps in faster and more stable learning. Normalisation is achieved by transforming the original values using the following formula:

$$
x_{\text{norm}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
$$
where $x$ is the original value, $x_{\text{min}}$​ is the minimum value of the feature, and $x_{\text{max}}$​ is the maximum value of the feature.

**代码示例**：
```python
from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
print("Min-Max Normalized Data:\n", normalized_data)
```

## MaxAbs 

**定义**：
将每个特征的绝对最大值缩放为 1。

**公式**：
$$
x_{\text{maxabs}} = \frac{x}{\max(|x|)}
$$

**代码示例**：
```python
from sklearn.preprocessing import MaxAbsScaler

data = [[1, -1, 2], [2, 0, 0], [0, 1, -1]]
scaler = MaxAbsScaler()
maxabs_data = scaler.fit_transform(data)
print("MaxAbs Normalized Data:\n", maxabs_data)
```

# Standardisation

## Z-score 标准化

>[!Definition]
>Standardisation is another data preprocessing technique used to scale features by removing the mean and scaling to unit variance. This process centres the data around zero with a standard deviation of one, which helps in stabilising the learning process for machine learning models. Standardisation is performed using the following formula:


$$
x_{\text{std}} = \frac{x - \mu}{\sigma}
$$

where $s$ is the original value, $\mu$ is the mean of the feature, and $\sigma$ is the standard deviation of the feature.

**代码示例**：
```python
from sklearn.preprocessing import StandardScaler

data = [[1, 2], [2, 3], [4, 5]]
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
print("Standardized Data:\n", standardized_data)
```

## Robust 标准化

**定义**：
使用中位数和四分位数范围进行缩放，对异常值具有鲁棒性。

**公式**：
$$
x_{\text{robust}} = \frac{x - \text{median}}{\text{IQR}}
$$

**代码示例**：
```python
from sklearn.preprocessing import RobustScaler

data = [[1, 2], [2, 3], [4, 5], [10, 20]]
scaler = RobustScaler()
robust_data = scaler.fit_transform(data)
print("Robust Scaled Data:\n", robust_data)
```

# 其他常用的归一化和标准化方法

## L1 归一化

**定义**：
将每个样本缩放到其绝对值之和为 1。

**公式**：
$$
x_{\text{L1}} = \frac{x}{\sum |x|}
$$

**代码示例**：
```python
from sklearn.preprocessing import Normalizer

data = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
scaler = Normalizer(norm='l1')
l1_normalized_data = scaler.fit_transform(data)
print("L1 Normalized Data:\n", l1_normalized_data)
```

## L2 归一化

**定义**：
将每个样本缩放到其欧几里得范数为 1。

**公式**：
$$
x_{\text{L2}} = \frac{x}{\sqrt{\sum x^2}}
$$

**代码示例**：
```python
from sklearn.preprocessing import Normalizer

data = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
scaler = Normalizer(norm='l2')
l2_normalized_data = scaler.fit_transform(data)
print("L2 Normalized Data:\n", l2_normalized_data)
```

# 为什么需要缩放数据？

1. **加快模型训练速度**：缩放后的数据可以使模型参数的初始值更接近最优值，从而加快训练速度。
2. **提高模型性能**：缩放数据有助于模型更好地理解和利用不同特征的重要性，尤其是在处理多个特征时。
3. **防止梯度爆炸或消失**：在梯度下降过程中，特征缩放可以帮助避免梯度爆炸或消失问题，提高训练稳定性。

# 实际应用中的代码示例

## 示例数据
```python
import numpy as np
import pandas as pd

# 创建示例数据
data = pd.DataFrame({
    'height': [150, 160, 170, 180, 190],
    'weight': [65, 72, 78, 85, 90]
})

print("Original Data:\n", data)
```

## 归一化
```python
from sklearn.preprocessing import MinMaxScaler

# 使用 MinMaxScaler 进行归一化
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

print("Min-Max Normalized Data:\n", pd.DataFrame(normalized_data, columns=['height', 'weight']))
```

## 标准化
```python
from sklearn.preprocessing import StandardScaler

# 使用 StandardScaler 进行标准化
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

print("Standardized Data:\n", pd.DataFrame(standardized_data, columns=['height', 'weight']))
```

## Robust 标准化
```python
from sklearn.preprocessing import RobustScaler

# 使用 RobustScaler 进行标准化
scaler = RobustScaler()
robust_data = scaler.fit_transform(data)

print("Robust Scaled Data:\n", pd.DataFrame(robust_data, columns=['height', 'weight']))
```

# 何时需要缩放数据？

- **梯度下降法**：对于需要使用梯度下降法的模型（如神经网络），缩放数据通常是必要的。
- **特征量纲不一致**：当数据集中的特征量纲不一致时，缩放数据可以帮助模型更好地学习。
- **对缩放敏感的模型**：例如 k 近邻（KNN）、支持向量机（SVM）和主成分分析（PCA）等模型，对特征缩放较为敏感。
