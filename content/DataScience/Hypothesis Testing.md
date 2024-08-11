---
title: 
draft: false
tags: 
date: 2024-06-18
---
# 假设检验（Hypothesis Testing）

## 什么是假设检验（What is Hypothesis Testing）

假设检验是一种统计方法，用于通过样本数据来检验关于总体参数的假设。假设检验帮助我们决定是否可以根据样本数据拒绝总体假设。

### 相关概念（Related Concepts）

1. **零假设（Null Hypothesis, $H_0$）**：零假设是一个陈述，表示没有效应或差异。例如，假设总体均值等于某个值。
2. **备择假设（Alternative Hypothesis, $H_a$）**：备择假设是一个陈述，表示存在效应或差异。例如，假设总体均值不等于某个值。
3. **显著性水平（Significance Level, $\alpha$）**：表示我们有多大概率会犯第一类错误（即错误地拒绝零假设）。常见的显著性水平有0.05和0.01。
4. **检验统计量（Test Statistic）**：从样本数据计算得出的值，用于检验假设。
5. **p值（p-value）**：表示在零假设为真时，检验统计量等于或极端于观测值的概率。如果p值小于显著性水平$\alpha$，我们拒绝零假设。
6. **拒绝域（Rejection Region）**：表示检验统计量落入的区域，如果统计量落入该区域，我们就拒绝零假设。
7. **自由度（Degrees of Freedom, DoF）**：在某些统计检验中，自由度是用来确定检验统计量的分布形状的参数。
8. **错误类型（Types of Errors）**：
   - **第一类错误（Type I Error）**：拒绝了真实的零假设。
   - **第二类错误（Type II Error）**：未能拒绝虚假的零假设。

### 错误类型（Types of Errors）

- **Type I Error**：A type I error occurs when the true null hypothesis is rejected. For example, concluding that the mean age at which children start walking is different from 12 months when in fact it is not.
- **Type II Error**：A type II error occurs when a false null hypothesis is not rejected. For example, failing to reject the null hypothesis that the proportion of businesses considering becoming a customer of the bank is equal to 20% when in fact the proportion is less than 20%.

## 假设检验的流程（Steps in Hypothesis Testing）

1. **陈述假设（State the Hypotheses）**：
   - 零假设 $H_0$：表示没有效应或差异。例如，$H_0: \mu = 7$
   - 备择假设 $H_a$：表示存在效应或差异。例如，$H_a: \mu \neq 7$

2. **选择显著性水平（Choose the Significance Level, $\alpha$）**：通常为0.05或0.01。

3. **选择适当的检验方法（Select the Appropriate Test）**：根据样本大小和数据类型选择合适的检验方法。

4. **计算检验统计量（Calculate the Test Statistic）**：从样本数据中计算检验统计量。

5. **确定临界值或计算p值（Determine the Critical Value or Calculate the p-value）**：根据选择的显著性水平和检验方法查找临界值或计算p值。

6. **作出决策（Make a Decision）**：
   - 如果检验统计量超过临界值或p值小于显著性水平$\alpha$，拒绝零假设。
   - 否则，不拒绝零假设。

## 常用假设检验方法（Common Hypothesis Tests）

### 针对单个均值的t检验（One-Sample t-Test）

- **条件**：样本来自正态分布，总体标准差未知。
- **假设**：
  - $H_0$：$\mu = \mu_0$
  - $H_a$：$\mu \neq \mu_0$
- **检验统计量**：
  $$ t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} $$
- **Python代码示例**：

```python
import scipy.stats as stats

# 样本数据
data = [12.1, 11.8, 12.3, 12.0, 12.5, 11.7, 12.2, 12.3, 12.1, 12.2]

# 假设的总体均值
mu_0 = 12

# t检验
t_stat, p_value = stats.ttest_1samp(data, mu_0)

# 显示结果
print("t-statistic:", t_stat)
print("p-value:", p_value)
```

### 针对两个独立样本均值的t检验（Independent Two-Sample t-Test）

- **条件**：两个独立样本，总体标准差未知。
- **假设**：
  - $H_0$：$\mu_1 = \mu_2$
  - $H_a$：$\mu_1 \neq \mu_2$
- **检验统计量**：
  $$ t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$
- **Python代码示例**：

```python
# 样本数据
data1 = [12.1, 11.8, 12.3, 12.0, 12.5]
data2 = [11.7, 12.2, 12.3, 12.1, 12.2]

# t检验
t_stat, p_value = stats.ttest_ind(data1, data2)

# 显示结果
print("t-statistic:", t_stat)
print("p-value:", p_value)
```

### 针对配对样本均值的t检验（Paired Sample t-Test）

- **条件**：配对样本，总体标准差未知。
- **假设**：
  - $H_0$：$\mu_d = 0$
  - $H_a$：$\mu_d \neq 0$
- **检验统计量**：
  $$ t = \frac{\bar{d}}{s_d / \sqrt{n}} $$
- **Python代码示例**：

```python
# 配对样本数据
before = [12.1, 11.8, 12.3, 12.0, 12.5]
after = [11.7, 12.2, 12.3, 12.1, 12.2]

# t检验
t_stat, p_value = stats.ttest_rel(before, after)

# 显示结果
print("t-statistic:", t_stat)
print("p-value:", p_value)
```

### 针对单个比例的z检验（One-Sample z-Test for Proportions）

- **条件**：样本量大，总体比例已知。
- **假设**：
  - $H_0$：$p = p_0$
  - $H_a$：$p \neq p_0$
- **检验统计量**：
  $$ z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1 - p_0)}{n}}} $$
- **Python代码示例**：

```python
from statsmodels.stats.proportion import proportions_ztest

# 样本数据
count = 30  # 成功次数
nobs = 50  # 总样本量
p0 = 0.5  # 假设的总体比例

# z检验
z_stat, p_value = proportions_ztest(count, nobs, value=p0)

# 显示结果
print("z-statistic:", z_stat)
print("p-value:", p_value)
```

### 针对两个独立比例的z检验（Two-Sample z-Test for Proportions）

- **条件**：两个独立样本，总体比例已知。
- **假设**：
  - $H_0$：$p_1 = p_2$
  - $H_a$：$p_1 \neq p_2$
- **检验统计量**：
  $$ z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p}) \left( \frac{1}{n_1} + \frac{1}{n_2} \right)}} $$
  其中，$\hat{p}$是合并比例：
  $$ \hat{p} = \frac{x_1 + x_2}{n_1 + n_2} $$
- **Python代码示例**：

```python
# 样本数据
count = np.array([30, 35])  # 成功次数
nobs = np.array([50, 50])  # 总样本量

# z检验
z_stat, p_value = proportions_ztest(count, nobs)

# 显示结果
print("z-statistic:", z_stat)
print("p-value:", p_value)
```

### 线性回归检验（Linear Regression Test）

- **条件**：连续自变量和因变量。
- **假设**：
  - $H_0$：$\beta_i = 0$（自变量 $X_i$ 对因变量无显著影响）
  - $H_a$：$\beta_i \neq 0$（自变量 $X_i$ 对因变量有显著影响）
- **Python代码示例**：

```python
import statsmodels.api as sm
import pandas as pd

# 样本数据
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 3, 5, 7, 11]
})

# 自变量和因变量
X = data['X']
Y = data['Y']

# 添加常数项
X = sm.add_constant(X)

# 线性回归
model = sm.OLS(Y, X).fit()

# 显示结果
print(model.summary())
```
