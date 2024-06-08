---
title: 
draft: false
tags:
  - regression
  - stats
---
> [!Definition]
> 线性回归是一种基本的回归分析方法，用于描述两个变量之间的线性关系。简单线性回归模型的形式为：

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

# 解释

- 分子部分 $\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$ 是协方差，表示 $x$ 和 $y$ 之间的联合变异程度。
- 分母部分 $\sum_{i=1}^n (x_i - \bar{x})^2$ 是 \( x \) 的方差，表示 $x$ 的变异程度。

# 计算步骤

1. 计算 $x$ 和 $y$ 的平均值，记为 $\bar{x}$ 和 $\bar{y}$。
2. 计算每个数据点与平均值的差值 $(x_i - \bar{x})$ 和 $(y_i - \bar{y})$。
3. 计算差值的乘积和 $\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$。
4. 计算 $x$ 的差值平方和 $\sum_{i=1}^n (x_i - \bar{x})^2$。
6. 用乘积和除以平方和得到斜率 $\beta_1$。

# 示例

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

# Python 代码示例

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

# Residual

![[residual.png]]
Most commonly, we fit a model by minimising the residual sum of squares. This means that the cost function is calculated like so:

1. Calculate the difference between the actual and predicted values (as previously) for each data point.
2. Square these values.
3. Sum (or average) these squared values.

This squaring step means that not all points contribute evenly to the line: outliers—which are points that don't fall in the expected pattern—have disproportionately larger error, which can influence the position of the line.

## Strengths of regression

Regression techniques have many strengths that more complex models don't.

### Predictable and easy to interpret

Regressions are easy to interpret because they describe simple mathematical equations, which we can often graph. More complex models are often referred to as _black box_ solutions, because it's difficult to understand how they make predictions or how they'll behave with certain inputs.

### Easy to extrapolate

Regressions make it easy to extrapolate; to make predictions for values outside the range of our dataset. For example, it's simple to estimate in our previous example that a nine year-old dog will have a temperature of 40.5°C. You should always apply caution to extrapolation: this model would predict that a 90 year-old would have a temperature nearly hot enough to boil water.

### Optimal fitting is usually guaranteed

Most machine learning models use gradient descent to fit models, which involves tuning the gradient descent algorithm and provides no guarantee that an optimal solution will be found. By contrast, linear regression that uses the sum of squares as a cost function doesn't actually need an iterative gradient-descent procedure. Instead, clever mathematics can be used to calculate the optimal location for the line to be placed. The mathematics are outside the scope of this module, but it's useful to know that (so long as the sample size isn't too large) linear regression doesn't need special attention to be paid to the fitting process, and the optimal solution is guaranteed.

# Multiple Linear Regression

>[!Definition]
>Multiple Linear Regression (MLR) is a statistical technique that models the relationship between a dependent variable and two or more independent variables by fitting a linear equation to observed data. The model takes the form:

$$ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p + \epsilon $$

Where:
- $y$ is the dependent variable,
- $x_1, x_2, \ldots, x_p$ are the independent variables,
- $\beta_0$ is the y-intercept,
- $\beta_1, \beta_2, \ldots, \beta_p$ are the coefficients of the independent variables,
- $\epsilon$ is the error term.

**Example:**
Consider predicting the price of a house based on its size (in square feet), number of bedrooms, and age. The multiple linear regression model would be:

$$ \text{Price} = \beta_0 + \beta_1 \cdot \text{Size} + \beta_2 \cdot \text{Bedrooms} + \beta_3 \cdot \text{Age} + \epsilon $$

**Calculation in R:**

```r
# Sample data
house_data <- data.frame(
  Price = c(200000, 250000, 300000, 150000, 180000),
  Size = c(1500, 2000, 2500, 1200, 1600),
  Bedrooms = c(3, 4, 4, 2, 3),
  Age = c(10, 15, 20, 5, 8)
)

# Multiple Linear Regression Model
model <- lm(Price ~ Size + Bedrooms + Age, data = house_data)
summary(model)
```

# Overfitting

>[!Definition]
>Overfitting occurs when a statistical model describes random error or noise in the data rather than the underlying relationship. An overfitted model performs well on training data but poorly on new, unseen data.

**Causes of Overfitting:**
- **Complex Models:** Too many parameters relative to the number of observations.
- **Noise in Data:** Fitting the noise instead of the actual data pattern.

**Signs of Overfitting:**
- **High Variance:** Model predictions vary widely for different training data sets.
- **Poor Generalization:** Good performance on training data but poor performance on validation/test data.

**Example:**
If we fit a polynomial regression model of a very high degree to a small dataset, it may perfectly fit the training data but fail to predict new data accurately.

**Visualization of Overfitting:**
1. **Training Data Fit:**
   - The model fits the training data perfectly, capturing all fluctuations and noise.
2. **Test Data Performance:**
   - The model performs poorly on test data, as it fails to generalize from the training data.

**Preventing Overfitting:**
- **Cross-Validation:** Use techniques like k-fold cross-validation to ensure the model generalizes well to unseen data.
- **Simpler Models:** Prefer simpler models that capture the underlying trend without fitting the noise.
- **Regularization:** Techniques like Lasso or Ridge regression add penalties to model complexity.

**Example in R:**

```r
# Sample data
set.seed(123)
x <- 1:10
y <- x + rnorm(10)

# Overfitting with a high-degree polynomial
overfit_model <- lm(y ~ poly(x, 10))
summary(overfit_model)

# Plotting the fit
plot(x, y, main = "Overfitting Example")
lines(x, predict(overfit_model, data.frame(x=x)), col = "red")
```
