---
title: 
draft: false
tags:
  - regression
date: 2024-06-08
---
## Goodness of Fit: R²


>[!R² (R-squared) Coefficient of Determination 决定系数]
>是衡量回归模型拟合优度（Goodness of Fit）的统计指标。它表示自变量解释的因变量总变异的比例。R² 值在 0 到 1 之间，值越接近 1，说明模型解释的变异越多，拟合效果越好。

### 公式

$$
R² = 1 - \frac{SS_{res}}{SS_{tot}}
$$

其中：
- $SS_{res}$ 是残差平方和（Sum of Squares of Residuals），计算公式为：
  $$
  SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
  其中 $y_i$ 是实际值，$\hat{y}_i$ 是预测值。
- $SS_{tot}$ 是总平方和（Total Sum of Squares），计算公式为：
  $$
  SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2
  $$
  其中 $\bar{y}$ 是因变量的均值。

### 解释

- **R² = 1**：表示模型能完美解释因变量的变异，所有数据点都在回归线上。
- **R² = 0**：表示模型不能解释因变量的任何变异，模型没有预测能力。
- **0 < R² < 1**：表示模型能解释部分因变量的变异，值越接近 1，模型的解释能力越强。

### 示例代码

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 生成样本数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 拟合线性回归模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 计算 R²
r2 = r2_score(y, y_pred)
print(f"R²: {r2}")
```

We know that cost functions can be used to assess how well a model fits the data on which it's trained. Linear regression models have a special related measure called R² (_R-squared_). R² is a value between 0 and 1 that tells us how well a linear regression model fits the data. When people talk about correlations being strong, they often mean that the R² value was large.

The reality is somewhere in between. Our model could predict temperature to some degree (so it's better than R² = 0), but points varied from this prediction somewhat (so it's less than R²=1).

R² is only half the story.

R² values are widely accepted, but aren't a perfect measure we can use in isolation. They suffer four limitations:

- Because of how R² is calculated, the more samples we have, the higher the R². This can lead us to thinking that one model is better than another (identical) model, simply because R² values were calculated using different amounts of data.
- R² values don't tell us how well a model will work with new, previously unseen data. Statisticians overcome this by calculating a supplementary measure, called a _p-value_, which we won't cover here. In machine learning, we often explicitly test our model on another dataset instead.
- R² values don't tell us the direction of the relationship. For example, an R² value of 0.8 doesn't tell us whether the line is sloped upwards or downwards. It also doesn’t tell us how sloped the line is.

It's also worth keeping in mind that there’s no universal criteria for what makes an R² value "good enough." For example, in most of physics, correlations that aren't very close to 1 are unlikely to be considered useful, but when modeling complex systems, R² values as low as 0.3 might be considered to be excellent.

### 其他衡量拟合优度的指标

#### 调整 R²（Adjusted R²）

$$
\text{Adjusted } R² = 1 - \frac{(1 - R²) \cdot (n - 1)}{n - k - 1}
$$

其中：
- $n$ 是样本数量。
- $k$ 是自变量数量。

#### 均方误差（Mean Squared Error, MSE）

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

#### 均方根误差（Root Mean Squared Error, RMSE）

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$


#### **Coefficient of Determination (usually known as [[R Squared]] or R<sup>2</sup>)**: 
A relative metric in which the higher the value, the better the fit of the model. In essence, this metric represents how much of the variance between predicted and actual label values the model is able to explain.