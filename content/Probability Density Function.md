---
title: 
draft: false
tags:
  - stats
date: 2024-06-06
---
### 概率密度函数（Probability Density Function, PDF）

**定义：**
概率密度函数是一个函数，用于描述连续型随机变量在某一特定取值范围内出现的可能性。PDF的积分表示随机变量在某个区间内出现的概率。概率密度函数是描述连续型随机变量分布的重要工具，它在统计学、信号处理、金融工程等领域有广泛应用。通过PDF，我们可以计算随机变量在特定区间内出现的概率，理解数据的分布特性。

### 关键属性

1. **非负性：**
   对于任意的$x$，$f(x) \geq 0$。

2. **归一化：**
   概率密度函数在整个定义域上的积分等于1，即
   $$ \int_{-\infty}^{\infty} f(x) \, dx = 1 $$

3. **概率计算：**
   连续型随机变量$X$在区间$[a, b]$上的概率是PDF在该区间上的积分，即
   $$ P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx $$

### 常见的概率密度函数

1. **正态分布（Normal Distribution）**
   $$ f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) $$
   其中，$\mu$是均值，$\sigma$是标准差。

2. **均匀分布（Uniform Distribution）**
   $$
   f(x) = 
   \begin{cases} 
   \frac{1}{b - a} & \text{if } a \leq x \leq b \\
   0 & \text{otherwise}
   \end{cases}
   $$

3. **指数分布（Exponential Distribution）**
   $$
   f(x) = 
   \begin{cases} 
   \lambda \exp(-\lambda x) & \text{if } x \geq 0 \\
   0 & \text{if } x < 0
   \end{cases}
   $$
   其中，$\lambda$是率参数。

### 示例

#### 正态分布的概率密度函数

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义参数
mu, sigma = 0, 1  # 均值和标准差

# 生成数据
x = np.linspace(-5, 5, 1000)
y = norm.pdf(x, mu, sigma)

# 绘制图形
plt.plot(x, y, label='Normal Distribution ($\mu=0$, $\sigma=1$)')
plt.title('Probability Density Function of Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
```

#### 均匀分布及PDF, CDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# 定义均匀分布的参数
a = 0  # 最小值
b = 10  # 最大值
loc = a
scale = b - a

# 生成均匀分布的样本数据
data = uniform.rvs(loc=loc, scale=scale, size=1000)

# 计算PDF和CDF
x = np.linspace(a, b, 1000)
pdf = uniform.pdf(x, loc=loc, scale=scale)
cdf = uniform.cdf(x, loc=loc, scale=scale)

# 绘制PDF和CDF
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# 绘制PDF
ax[0].plot(x, pdf, 'r-', lw=2, label='PDF')
ax[0].hist(data, bins=30, density=True, alpha=0.5, color='blue', edgecolor='black')
ax[0].set_title('Uniform Distribution - PDF')
ax[0].set_xlabel('x')
ax[0].set_ylabel('Density')
ax[0].legend()

# 绘制CDF
ax[1].plot(x, cdf, 'r-', lw=2, label='CDF')
ax[1].set_title('Uniform Distribution - CDF')
ax[1].set_xlabel('x')
ax[1].set_ylabel('Cumulative Probability')
ax[1].legend()

plt.tight_layout()
plt.show()
```

#### 指数分布的概率密度函数

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# 定义参数
lambda_param = 1  # 率参数

# 生成数据
x = np.linspace(0, 5, 1000)
y = expon.pdf(x, scale=1/lambda_param)

# 绘制图形
plt.plot(x, y, label='Exponential Distribution ($\lambda=1$)')
plt.title('Probability Density Function of Exponential Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
```
