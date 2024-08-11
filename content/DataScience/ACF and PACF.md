---
title: 
draft: false
tags: 
date: 2024-06-26
---

![[content/Images/acfandpacf.png]]

#### 自相关函数（ACF）

- **周期性和季节性**：
  - 在你的ACF图中，可以看到一个明显的波形图，显示出大约24个时滞的周期性。这意味着你的数据大约每24个时滞（即每24小时）出现一个重复的模式。这通常表示每日的季节性周期性。

- **滞后期的相关性**：
  - 图中滞后期k处的自相关系数显示了数据与其自身不同滞后期的相关性。例如，滞后期为1时的自相关系数接近1，表明当前值与前一个时刻的值高度相关。
  
- **衰减模式**：
  - 在ACF图中，自相关系数随着滞后期的增加呈现波动并逐渐衰减，这说明存在长期趋势和季节性。

#### 偏自相关函数（PACF）

- **直接相关性**：
  - PACF图显示的是在去除中间滞后期影响后的直接相关性。你可以看到在滞后期为1时的偏自相关系数非常高，表明第一个滞后期的直接影响非常显著。

- **滞后期的直接影响**：
  - 你可以从图中看到，在第一个滞后期后，偏自相关系数迅速下降并在滞后期5-6处再次出现几个显著的峰值。这些信息可以帮助你确定ARIMA模型的AR部分的阶数。

### 滞后期

- **滞后期（Lag）**：
  - 滞后期是指当前时间点和之前某个时间点之间的间隔。例如，如果我们当前时间点是t，滞后期为k表示我们与t-k时间点之间的关系。
  
- **k处**：
  - k处表示在滞后期为k的位置。例如，如果滞后期k为3，这意味着我们在查看当前时间点与3个时间点前的相关性。

### 结合ACF和PACF图确定ARIMA模型的参数

1. **自回归阶数（p）**：
   - 从PACF图中可以看出，第一个滞后期（lag 1）显著，之后逐渐衰减。这通常表示AR部分的阶数为1。

2. **差分阶数（d）**：
   - 如果数据存在明显的趋势，则需要进行差分处理。在ACF和PACF图中，差分阶数一般通过试验确定。在你的数据中，可能需要进行1阶差分（d=1）以去除趋势。

3. **移动平均阶数（q）**：
   - 从ACF图中可以看出，在滞后期24（每天的周期）出现明显的相关性峰值，之后逐渐衰减。这表明MA部分的阶数可能为1或更高。

### 示例代码

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 加载数据
df = pd.read_csv('data/final_data_hourly_cleaned.csv', parse_dates=['datetime'], index_col='datetime')

# 提取目标变量
data = df['price_act']

# 绘制ACF和PACF图
plt.figure(figsize=(12, 6))

plt.subplot(211)
plot_acf(data, lags=50, ax=plt.gca())
plt.title('Autocorrelation')

plt.subplot(212)
plot_pacf(data, lags=50, ax=plt.gca())
plt.title('Partial Autocorrelation')

plt.tight_layout()
plt.show()

# 选择ARIMA模型参数
p = 1  # 从PACF图中得到
d = 1  # 通过试验确定
q = 1  # 从ACF图中得到

# 拟合ARIMA模型
model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

# 预测
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
predictions = model_fit.forecast(steps=len(test))

# 计算RMSE
rmse = mean_squared_error(test, predictions, squared=False)
print(f'RMSE: {rmse}')

# 绘制实际值与预测值
plt.figure(figsize=(14, 7))
plt.plot(data.index, data, label='Actual Price')
plt.plot(test.index, predictions, label='Predicted Price', color='red')
plt.title('ARIMA Model - Actual vs Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price (¥)')
plt.legend()
plt.show()
```

通过ACF和PACF图，你可以确定ARIMA模型的最佳参数，并使用这些参数进行时间序列预测。