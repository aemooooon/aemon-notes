---
title: 
draft: false
tags:
  - stats
  - correlation
  - residual
date: 2024-06-01
---

相关性（Correlation）是统计学中用来衡量两个变量之间关系的一种指标。它描述了两个变量在多大程度上以及以何种方式共同变化。相关性可以帮助我们理解变量之间的依赖性和预测关系。

## 相关系数

相关性的常用指标是相关系数，通常取值在 -1 和 1 之间：

- **1** 表示完全正相关：当一个变量增加时，另一个变量也以线性方式增加。
- **0** 表示无相关性：两个变量之间没有线性关系。
- **-1** 表示完全负相关：当一个变量增加时，另一个变量以线性方式减少。

需要注意的是，相关系数仅衡量线性关系，如果两个变量之间存在非线性关系，相关系数可能无法准确反映其依赖性。

## 常用的相关系数

### 1. 皮尔逊相关系数（Pearson Correlation Coefficient）

- 计算线性关系的强度和方向。
- 公式：

$$
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
$$

- 要求：变量之间是线性关系，并且数据是连续的和正态分布的。

### 2. 斯皮尔曼秩相关系数（Spearman's Rank Correlation Coefficient）

- 计算两个变量的秩序关系，不要求线性关系。
- 公式：

$$
r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

其中 $d_i$ 是两个变量的秩之差，$n$ 是样本数量。

### 3. 肯德尔秩相关系数（Kendall's Tau Coefficient）

- 衡量两个变量秩序关系的一致性。
- 公式：

$$
\tau = \frac{(C - D)}{\sqrt{(C + D + T)(C + D + U)}}
$$

其中 $C$ 是一致对，$D$ 是不一致对，$T$ 和 $U$ 是重复值的对数。

## 相关系数矩阵

假设我们有两个变量 $x$ 和 $y$，使用 `np.corrcoef(x, y)` 计算相关系数矩阵，得到： 
$$ \begin{bmatrix} 1 & r \\ r & 1 \\ \end{bmatrix} $$ 其中： 
- 第一个元素 $(0, 0)$ 是 $x$ 与 $x$ 的相关系数，自相关性为 1。 
- 第二个元素 $(0, 1)$ 是 $x$ 与 $y$ 的相关系数，即我们通常关心的相关系数 $r$。 
- 第三个元素 $(1, 0)$ 是 $y$ 与 $x$ 的相关系数，与 $(0, 1)$ 相同，即 $r$。 
- 第四个元素 $(1, 1)$ 是 $y$ 与 $y$ 的相关系数，自相关性为 1。

## 计算步骤

以皮尔逊相关系数为例，计算步骤如下：

1. 计算每个变量的均值 $\bar{x}$ 和 $\bar{y}$。
2. 计算每个数据点的偏差 $(x_i - \bar{x})$ 和 $(y_i - \bar{y})$。
3. 计算偏差的乘积和 $\sum (x_i - \bar{x})(y_i - \bar{y})$。
4. 计算每个变量的偏差平方和 $\sum (x_i - \bar{x})^2$ 和 $\sum (y_i - \bar{y})^2$。
5. 代入公式计算相关系数 $r$。

## 示例

假设有以下数据：

| x  | y  |
|----|----|
| 1  | 2  |
| 2  | 3  |
| 3  | 5  |
| 4  | 4  |
| 5  | 6  |

计算皮尔逊相关系数：

1. 计算均值：
   $$
   \bar{x} = \frac{1 + 2 + 3 + 4 + 5}{5} = 3
   $$
   $$
   \bar{y} = \frac{2 + 3 + 5 + 4 + 6}{5} = 4
   $$

2. 计算偏差：
   $$
   (x_i - \bar{x}) = [-2, -1, 0, 1, 2]
   $$
   $$
   (y_i - \bar{y}) = [-2, -1, 1, 0, 2]
   $$

3. 计算偏差乘积和：
   $$
   \sum (x_i - \bar{x})(y_i - \bar{y}) = (-2 \cdot -2) + (-1 \cdot -1) + (0 \cdot 1) + (1 \cdot 0) + (2 \cdot 2) = 4 + 1 + 0 + 0 + 4 = 9
   $$

4. 计算 $x$ 的偏差平方和：
   $$
   \sum (x_i - \bar{x})^2 = (-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 4 + 1 + 0 + 1 + 4 = 10
   $$

5. 计算 $y$ 的偏差平方和：
   $$
   \sum (y_i - \bar{y})^2 = (-2)^2 + (-1)^2 + 1^2 + 0^2 + 2^2 = 4 + 1 + 1 + 0 + 4 = 10
   $$

6. 计算相关系数：
   $$
   r = \frac{9}{\sqrt{10 \cdot 10}} = \frac{9}{10} = 0.9
   $$

## Python 代码示例

```python
import numpy as np

# 数据点
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 4, 6])

# 计算皮尔逊相关系数
correlation = np.corrcoef(x, y)[0, 1]

print(f'Pearson correlation coefficient: {correlation}')](<import numpy as np

# 数据点
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 4, 6])

# 计算相关系数矩阵
# np.corrcoef 返回一个相关系数矩阵，其中 [0, 0] 和 [1, 1] 是自相关系数，值为 1
# [0, 1] 和 [1, 0] 是 x 和 y 之间的相关系数
corr_matrix = np.corrcoef(x, y)

# 打印相关系数矩阵以便查看
print("Correlation matrix:")
print(corr_matrix)

# 提取 x 和 y 之间的相关系数
# 相关系数矩阵的 [0, 1] 位置上的值是 x 和 y 之间的皮尔逊相关系数
correlation = corr_matrix[0, 1]

print(f'Pearson correlation coefficient: {correlation}')>)
```


## 残差（Residual）

残差是实际值与预测值之间的差异，表示模型预测误差。对于每一个数据点 $i$：

$$ \text{残差}_i = y_i - \hat{y}_i $$

其中：
- $y_i$ 是第 $i$ 个数据点的实际值。
- $\hat{y}_i$ 是第 $i$ 个数据点的预测值。

## 均方误差（Mean Squared Error, MSE）

均方误差是残差的平方的平均值，用于衡量模型的预测精度。它更重视大误差，因为误差被平方了。

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$

其中：
- $n$ 是数据点的数量。
- $y_i$ 是第 $i$ 个数据点的实际值。
- $\hat{y}_i$ 是第 $i$ 个数据点的预测值。

## 平均绝对误差（Mean Absolute Error, MAE）

平均绝对误差是残差绝对值的平均值，用于衡量模型的平均预测误差。它对大误差的敏感度较低，因为误差没有被平方。

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| $$

其中：
- $n$ 是数据点的数量。
- $y_i$ 是第 $i$ 个数据点的实际值。
- $\hat{y}_i$ 是第 $i$ 个数据点的预测值。

## 均方根误差（Root Mean Squared Error, RMSE）

均方根误差是均方误差（MSE）的平方根，用于衡量预测值与实际值之间的差异。RMSE 和 MSE 一样，对大误差比较敏感，因为误差被平方了。RMSE 的单位和原始数据的单位相同，使得其更容易解释。

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} $$

其中：
- $n$ 是数据点的数量。
- $y_i$ 是第 $i$ 个数据点的实际值。
- $\hat{y}_i$ 是第 $i$ 个数据点的预测值。

## 示例

假设有以下数据点的实际值和预测值：

| 实际值 $y$ | 预测值 $\hat{y}$ |
|------------|------------------|
| 3          | 2.5              |
| -0.5       | 0.0              |
| 2          | 2.1              |
| 7          | 8.1              |

我们可以计算残差、MSE、MAE 和 RMSE：

1. **残差**：
   $$
   \text{残差}_1 = 3 - 2.5 = 0.5
   $$
   $$
   \text{残差}_2 = -0.5 - 0.0 = -0.5
   $$
   $$
   \text{残差}_3 = 2 - 2.1 = -0.1
   $$
   $$
   \text{残差}_4 = 7 - 8.1 = -1.1
   $$

2. **MSE**：
   $$
   \text{MSE} = \frac{1}{4} \left( (0.5)^2 + (-0.5)^2 + (-0.1)^2 + (-1.1)^2 \right) = \frac{1}{4} (0.25 + 0.25 + 0.01 + 1.21) = \frac{1.72}{4} = 0.43
   $$

3. **MAE**：
   $$
   \text{MAE} = \frac{1}{4} \left( |0.5| + |-0.5| + |-0.1| + |-1.1| \right) = \frac{1}{4} (0.5 + 0.5 + 0.1 + 1.1) = \frac{2.2}{4} = 0.55
   $$

4. **RMSE**：
   $$
   \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{0.43} \approx 0.656
   $$

## Python 代码示例

```python
import numpy as np

# 实际值和预测值
y = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2.1, 8.1])

# 计算残差
residuals = y - y_pred
print("Residuals:", residuals)

# 计算MSE
mse = np.mean((y - y_pred) ** 2)
print("Mean Squared Error (MSE):", mse)

# 计算MAE
mae = np.mean(np.abs(y - y_pred))
print("Mean Absolute Error (MAE):", mae)

# 计算RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Residuals: [ 0.5 -0.5 -0.1 -1.1]
# Mean Squared Error (MSE): 0.43
# Mean Absolute Error (MAE): 0.55
# Root Mean Squared Error (RMSE): 0.6557438524302003
```

