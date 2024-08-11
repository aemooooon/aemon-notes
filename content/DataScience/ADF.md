---
title: 
draft: false
tags: 
date: 2024-07-02
---
### Augmented Dickey-Fuller 

The Augmented Dickey-Fuller (ADF) test (增强型Dickey-Fuller检验) is a statistical method used to check whether a time series data set has a unit root, i.e., whether the data is stationary. In time series analysis, stationarity (平稳性) is a fundamental assumption for many models (e.g., ARIMA). A time series is considered stationary if its statistical properties (mean and variance) do not change over time.

### 1. Principle of the ADF Test

The ADF test checks for a unit root (单位根) in the following regression model:
$$
\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \delta \Delta Y_{t-1} + \epsilon_t
$$

Where $\Delta Y_t$ represents the first difference (一阶差分) of the time series $Y_t$, $\alpha$ and $\beta$ are constants, $\gamma$ is the coefficient being tested, $\delta \Delta Y_{t-1}$ is the lagged difference term, and $\epsilon_t$ is the white noise error term.

### 2. Hypotheses of the ADF Test

- **Null Hypothesis ($H_0$) 原假设**: The time series has a unit root, i.e., it is non-stationary. 时间序列具有单位根，即时间序列是非平稳的。
- **Alternative Hypothesis ($H_1$) 备择假设**: The time series does not have a unit root, i.e., it is stationary. 时间序列没有单位根，即时间序列是平稳的。

### 3. Steps of the ADF Test 

1. **Calculate the ADF Statistic 计算ADF统计量**: Perform regression analysis on the time series data and compute the ADF statistic.
2. **Compute the $p$-value 计算$p$值**: Calculate the $p$-value using the ADF statistic and predefined critical values.
3. **Determine the Result 判断结果**:
   - If the $p$-value is less than the significance level (e.g., 0.05), reject the null hypothesis and conclude that the time series is stationary. 如果$p$值小于显著性水平（如0.05），拒绝原假设，认为时间序列是平稳的。
   - If the $p$-value is greater than the significance level, do not reject the null hypothesis and conclude that the time series is non-stationary. 如果$p$值大于显著性水平，不拒绝原假设，认为时间序列是非平稳的。

### 4. Example: ADF Test on `price_act` 示例：对`price_act`进行ADF检验

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load data 加载数据
df = pd.read_csv('../data/cleaned_data_hourly.csv', parse_dates=['datetime'], index_col='datetime')

# Select the target variable for prediction 选择预测目标变量
target = 'price_act'

# Plot the original time series 绘制原始时间序列图
plt.figure(figsize=(12, 6))
plt.plot(df[target], label='Original')
plt.title('Original Time Series')
plt.xlabel('DateTime')
plt.ylabel(target)
plt.legend()
plt.show()

# Dickey-Fuller test function Dickey-Fuller平稳性检验函数
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    return result[1]

# ADF test for the original data 对原始数据进行ADF检验
print("ADF Test for Original Data:")
p_value = adf_test(df[target])
if p_value > 0.05:
    print("Data is not stationary and requires differencing. 数据非平稳，需要处理。")
else:
    print("Data is stationary. 数据平稳。")
```

#### 5. Test Results 

Based on the ADF test results, the $p$-value of the original data is much less than 0.05, indicating that we reject the null hypothesis. Thus, the data is statistically stationary. 根据ADF检验结果，原始数据的$p$值远小于0.05，表明我们可以拒绝原假设，即数据在统计意义上是平稳的。

#### 6. Why is Differencing Important? 为什么差分处理很重要？

Even though the original data is statistically stationary, visually observed significant fluctuations and peaks might affect the model's fit. Differencing removes trends and seasonality, making the mean and variance more stable, thereby improving the model's predictive performance. 尽管原始数据在统计意义上是平稳的，但视觉上观察到的显著波动和峰值可能会影响模型的拟合效果。通过差分处理，可以去除数据中的趋势和季节性成分，使数据的均值和方差更加稳定，从而提高模型的预测性能。

#### Summary 

The Augmented Dickey-Fuller (ADF) test (增强型Dickey-Fuller检验) is a crucial statistical tool for checking the stationarity of time series data. By applying the ADF test, we can determine whether the data needs further processing (e.g., differencing) to be suitable for time series modeling and forecasting. In Chinese, the Augmented Dickey-Fuller (ADF) test is known as 增强型Dickey-Fuller检验. 

Augmented Dickey-Fuller (ADF) test 是一种用于时间序列分析的统计检验方法，用于检测时间序列数据是否具有单位根，即数据是否是平稳的。在时间序列分析中，平稳性是许多时间序列分析和建模方法（如ARIMA模型）的基本假设。通过ADF检验，可以判断时间序列数据是否需要进一步处理（如差分），以便更好地进行时间序列建模和预测。

### What is a Unit Root? 什么是单位根？

**Unit Root 单位根** is an important concept in time series analysis. It refers to a situation where a time series model has a characteristic equation root equal to 1. Time series with a unit root exhibit the following characteristics:

单位根是时间序列分析中的一个重要概念。它指的是时间序列模型中的特征方程的根等于1的情况。具有单位根的时间序列会表现出以下特点：

1. **Non-stationarity 非平稳性**: The statistical properties (mean, variance) of the time series change over time. 时间序列的统计特性（如均值、方差）会随时间发生变化。
2. **Random Walk 随机游走**: Time series with a unit root typically exhibit a random walk, meaning the current value is the previous value plus a random error term. 具有单位根的时间序列通常表现为随机游走，即当前值是前一期值加上一个随机误差项。
3. **Long Memory Effect 长记忆效应**: Time series with a unit root show long memory effects, meaning past shocks have a long-term impact on future values. 具有单位根的时间序列会显示出长记忆效应，意味着过去的冲击会对未来产生长期的影响。

### Mathematical Description 数学描述

Consider a simple autoregressive model AR(1):  
考虑一个简单的自回归模型 AR(1):

$Y_t = \rho Y_{t-1} + \epsilon_t$

where $\epsilon_t$ is the white noise error term.  
其中，$\epsilon_t$ 是白噪声误差项。

- If $|\rho| < 1$, the time series is stationary, with constant mean and variance, and shocks are temporary. 如果 $|\rho| < 1$，时间序列是平稳的，均值和方差恒定，冲击的影响是暂时的。
- If $\rho = 1$, the time series has a unit root, exhibiting a random walk where shocks have permanent effects. 如果 $\rho = 1$，时间序列具有单位根，表现为随机游走，冲击的影响是永久的。
- If $|\rho| > 1$, the time series is explosive, with mean and variance increasing over time. 如果 $|\rho| > 1$，时间序列是爆炸性的，均值和方差随时间增加。

### Impact of Unit Roots on Time Series 单位根对时间序列的影响

Time series with a unit root are non-stationary, which poses a problem for many time series models (e.g., ARIMA) that assume stationarity. 具有单位根的时间序列是非平稳的，这对许多时间序列模型（如ARIMA）的应用是一个问题，因为这些模型假设输入数据是平稳的。

#### Summary

**Unit Root 单位根** is a critical concept in time series analysis, indicating the non-stationarity of a time series. The Dickey-Fuller test and the Augmented Dickey-Fuller (ADF) test are standard methods for unit root testing. By applying these tests, we can determine whether further processing (e.g., differencing) is required to make the time series suitable for modeling and forecasting. 在时间序列分析中，**单位根**是一个重要概念，表示时间序列的非平稳性。Dickey-Fuller检验和增强型Dickey-Fuller（ADF）检验是常用的单位根检验方法。通过这些检验可以判断时间序列是否需要进一步处理（如差分），以便更好地进行时间序列建模和预测。