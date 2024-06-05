---
title: 
draft: false
tags:
  - stats
---

### 累积分布函数（Cumulative Distribution Function, CDF）

**定义：**
累积分布函数（CDF）是一个函数，用于描述随机变量$X$小于或等于某个特定值$x$的概率。对于一个随机变量$X$，其累积分布函数$F(x)$定义为：

$$ F(x) = P(X \leq x) $$

### 关键属性

1. **非负性：**
   对于任意的$x$，$F(x) \geq 0$。

2. **单调不减：**
   如果$x_1 \leq x_2$，那么$F(x_1) \leq F(x_2)$。

3. **归一化：**
   当$x$趋向于负无穷时，$F(x) \to 0$；当$x$趋向于正无穷时，$F(x) \to 1$。

4. **右连续：**
   $F(x)$是右连续的，即对于任意的$x$，$ \lim_{\epsilon \to 0^+} F(x + \epsilon) = F(x) $。

### 关系

- **与概率密度函数（PDF）的关系：**
  对于连续型随机变量$X$，其累积分布函数$F(x)$是概率密度函数$f(x)$的积分：

  $$ F(x) = \int_{-\infty}^{x} f(t) \, dt $$

- **概率计算：**
  对于区间$[a, b]$上的概率，可以通过CDF计算：

  $$ P(a \leq X \leq b) = F(b) - F(a) $$

### 常见分布的CDF

1. **正态分布（Normal Distribution）：**
   正态分布的CDF通常用累积分布函数$\Phi(x)$表示，没有简单的解析表达式，通常通过数值积分或查表获得。

   ```python
   from scipy.stats import norm
   import matplotlib.pyplot as plt
   import numpy as np

   x = np.linspace(-5, 5, 1000)
   y = norm.cdf(x, 0, 1)  # 均值为0，标准差为1的正态分布

   plt.plot(x, y, label='Normal Distribution CDF ($\mu=0$, $\sigma=1$)')
   plt.title('Cumulative Distribution Function of Normal Distribution')
   plt.xlabel('x')
   plt.ylabel('CDF')
   plt.legend()
   plt.show()
   ```

2. **均匀分布（Uniform Distribution）：**

   对于区间$[a, b]$上的均匀分布，CDF为：

   $$ F(x) =
   \begin{cases}
   0 & \text{if } x < a \\
   \frac{x - a}{b - a} & \text{if } a \leq x \leq b \\
   1 & \text{if } x > b
   \end{cases} $$

3. **指数分布（Exponential Distribution）：**

   对于参数为$\lambda$的指数分布，CDF为：

   $$ F(x) =
   \begin{cases}
   1 - \exp(-\lambda x) & \text{if } x \geq 0 \\
   0 & \text{if } x < 0
   \end{cases} $$

   ```python
   from scipy.stats import expon
   import matplotlib.pyplot as plt
   import numpy as np

   lambda_param = 1
   x = np.linspace(0, 5, 1000)
   y = expon.cdf(x, scale=1/lambda_param)

   plt.plot(x, y, label='Exponential Distribution CDF ($\lambda=1$)')
   plt.title('Cumulative Distribution Function of Exponential Distribution')
   plt.xlabel('x')
   plt.ylabel('CDF')
   plt.legend()
   plt.show()
   ```

### 应用

- **统计学和数据分析：** CDF用于计算特定区间的概率，进行统计推断。
- **可靠性工程：** 通过CDF分析产品寿命和故障率。
- **金融工程：** CDF用于风险评估和金融工具定价，如期权定价中的二项模型和布莱克-斯科尔斯模型。

### 总结

累积分布函数（CDF）是描述随机变量分布的重要工具，通过它可以计算随机变量在特定区间内出现的概率，理解数据的分布特性。与概率密度函数（PDF）不同，CDF提供了随机变量累计概率的全貌，是概率论和统计学中的基本概念。