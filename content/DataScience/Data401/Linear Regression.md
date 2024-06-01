---
title: Linear Regression
draft: false
tags:
  - regression
---
# Linear Regression

线性回归是一种基本的回归分析方法，用于描述两个变量之间的线性关系。简单线性回归模型的形式为：

$$ y = \beta_0 + \beta_1 x $$

其中：
- $y$ 是因变量（响应变量）。
- $x$ 是自变量（预测变量）。
- $\beta_0$ 是截距（即当 $x = 0$ 时 $y$ 的预测值）。
- $\beta_1$ 是斜率（即 $x$ 每增加一个单位时 $y$ 的变化量）。

斜率 $\beta_1$ 的计算公式如下：

$$
\beta_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

其中：
- $n$ 是数据点的数量。
- $x_i$和 $y_i$ 是第 $i$ 个数据点的自变量和因变量的值。
- $\bar{x}$ 是自变量 $x$ 的平均值。
- $\bar{y}$ 是因变量 $y$ 的平均值。

### 解释

- 分子部分 $\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$ 是协方差，表示 $x$ 和 $y$ 之间的联合变异程度。
- 分母部分 $\sum_{i=1}^n (x_i - \bar{x})^2$ 是 \( x \) 的方差，表示 $x$ 的变异程度。

### 计算步骤

1. 计算 $x$ 和 $y$ 的平均值，记为 $\bar{x}$ 和 $\bar{y}$。
2. 计算每个数据点与平均值的差值 $(x_i - \bar{x})$ 和 $(y_i - \bar{y})$。
3. 计算差值的乘积和 $\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$。
4. 计算 $x$ 的差值平方和 $\sum_{i=1}^n (x_i - \bar{x})^2$。
6. 用乘积和除以平方和得到斜率 $\beta_1$。

### 示例

假设有以下数据点：
- \( (1, 2) \)
- \( (2, 3) \)
- \( (3, 5) \)
- \( (4, 4) \)
- \( (5, 6) \)

我们可以使用上述公式计算斜率 $\beta_1$：

1. 计算平均值：
   $$
   \bar{x} = \frac{1 + 2 + 3 + 4 + 5}{5} = 3
   $$
   $$
   \bar{y} = \frac{2 + 3 + 5 + 4 + 6}{5} = 4
   $$

2. 计算差值：
   $$
   (x_i - \bar{x}) = [-2, -1, 0, 1, 2]
   $$
   $$
   (y_i - \bar{y}) = [-2, -1, 1, 0, 2]
   $$

3. 计算差值乘积和：
   $$
   \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) = (-2 \cdot -2) + (-1 \cdot -1) + (0 \cdot 1) + (1 \cdot 0) + (2 \cdot 2) = 4 + 1 + 0 + 0 + 4 = 9
   $$

4. 计算 \( x \) 的差值平方和：
   $$
   \sum_{i=1}^n (x_i - \bar{x})^2 = (-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2 = 4 + 1 + 0 + 1 + 4 = 10
   $$

5. 计算斜率：
   $$
   \beta_1 = \frac{9}{10} = 0.9
   $$

因此，斜率 $\beta_1$ 为 0.9。

### Python 代码示例

我们也可以用 Python 代码来验证计算过程：

```python
import numpy as np

# 数据点
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 4, 6])

# 计算平均值
x_mean = np.mean(x)
y_mean = np.mean(y)

# 计算斜率
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
beta_1 = numerator / denominator

print(f'Slope (beta_1): {beta_1}')
```
