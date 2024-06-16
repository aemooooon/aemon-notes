---
title: 
draft: false
tags:
  - stats
date: 2024-06-02
---


# 数据科学基础术语

## Population (总体)
总体是要收集数据的源头。它包含了所有可能的观测或数据点。
- **中文:** 总体
- **英文:** Population

## Sample (样本)
样本是总体的一部分，用于进行统计分析。
- **中文:** 样本
- **英文:** Sample

## Variable (变量)
变量是可以测量或计数的任何数据项。
- **中文:** 变量
- **英文:** Variable

## Quantitative Analysis (定量分析)
定量分析是通过模式和数据可视化来收集和解释数据。
- **中文:** 定量分析
- **英文:** Quantitative Analysis

## Qualitative Analysis (定性分析)
定性分析是从非数据形式的媒体中生成一般信息的过程。
- **中文:** 定性分析
- **英文:** Qualitative Analysis

## Descriptive Statistics (描述性统计)
描述性统计是对总体的特征进行描述。
- **中文:** 描述性统计
- **英文:** Descriptive Statistics

## Inferential Statistics (推断性统计)
推断性统计是对总体进行预测。
- **中文:** 推断性统计
- **英文:** Inferential Statistics

## Central Tendency (集中趋势)
集中趋势是对数据集中心位置的测量，包括平均值、中位数和众数。
- **中文:** 集中趋势
- **英文:** Central Tendency

## Measures of the Spread (离散度的测量)
离散度的测量包括范围、方差和标准差。
- **中文:** 离散度的测量
- **英文:** Measures of the Spread

### Range (极差)
极差是数据集中每个值之间的距离。
- **中文:** 极差
- **英文:** Range

### Variance (方差)
方差是变量与其期望值之间的距离。
- **中文:** 方差
- **英文:** Variance

### Standard Deviation (标准差)
标准差是数据集从均值分散的程度。
- **中文:** 标准差
- **英文:** Standard Deviation

## Bias (偏差)
偏差是指估计量的期望值与真值之间的差异。
- **中文:** 偏差
- **英文:** Bias

## Mean (平均值)
平均值是数据集中所有值的平均值。
- **中文:** 平均值
- **英文:** Mean

## Median (中位数)
中位数是将数据集按升序排列后位于中间的数值。
- **中文:** 中位数
- **英文:** Median

## Mode (众数)
众数是数据集中出现次数最多的数值。
- **中文:** 众数
- **英文:** Mode

## Percentiles (百分位数)
百分位数是将数据集分成100个相等的部分，每个部分包含1%的数据。
- **中文:** 百分位数
- **英文:** Percentiles

## 变量类型
不同类型的变量和它们的定义。

### Nominal Variables (名义变量)
名义变量是分类变量，没有顺序或等级关系。
- **中文:** 名义变量
- **英文:** Nominal Variables

### Ordinal Variables (有序变量)
有序变量是具有顺序或等级关系的分类变量。
- **中文:** 有序变量
- **英文:** Ordinal Variables

### Continuous Variables (连续变量)
连续变量是可以取无限多个值的变量。
- **中文:** 连续变量
- **英文:** Continuous Variables

### Discrete Variables (离散变量)
离散变量是只能取有限个值的变量。
- **中文:** 离散变量
- **英文:** Discrete Variables

## 计算示例和实现

### R 代码示例
```r
# 平均值
X <- c(1, 2, 3, 4, 5)
mean_value <- mean(X)
mean_value

# 中位数
median_value <- median(X)
median_value

# 众数
mode_value <- as.numeric(names(sort(table(X), decreasing=TRUE)[1]))
mode_value

# 方差
variance_value <- var(X)
variance_value

# 标准差
standard_deviation <- sd(X)
standard_deviation

# 百分位数
quartiles <- quantile(X)
quartiles
```
### Python 代码示例

```python
import numpy as np
from scipy import stats

# 数据集
X = [1, 2, 3, 4, 5]

# 平均值
mean_value = np.mean(X)
print("Mean:", mean_value)

# 中位数
median_value = np.median(X)
print("Median:", median_value)

# 众数
mode_value = stats.mode(X)
print("Mode:", mode_value.mode[0])

# 方差
variance_value = np.var(X, ddof=1)
print("Variance:", variance_value)

# 标准差
standard_deviation = np.std(X, ddof=1)
print("Standard Deviation:", standard_deviation)

# 百分位数
percentiles = np.percentile(X, [25, 50, 75])
print("Percentiles (25th, 50th, 75th):", percentiles)
```

# 方差和标准差

## 方差 (Variance)
方差表示数据集中的每个数据点与均值（平均值）之间的平均平方差。它用来衡量数据的分散程度。方差的公式如下：

对于一组数据 $X = \{x_1, x_2, \ldots, x_n\}$，其均值（平均值）为 $\mu$，方差 $\sigma^2$ 计算公式为：

$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n}(x_i - \mu)^2
$$

其中：
- $x_i$ 是数据集中的第 $i$ 个数据点
- $\mu$ 是数据集的均值
- $n$ 是数据点的数量

方差的值越大，表示数据点离均值越远，数据分布越分散；方差越小，表示数据点离均值越近，数据分布越集中。

## 标准差 (Standard Deviation)
标准差是方差的平方根，表示数据的分散程度。标准差的公式如下：

$$
\sigma = \sqrt{\sigma^2}
$$

其中 $\sigma^2$ 是方差。

标准差与方差一样，用于描述数据的分散程度，但标准差与数据的单位相同，因而更容易理解和解释。

## 示例

假设有一组数据：$X = \{1, 2, 3, 4, 5\}$

1. 计算均值：
$$
\mu = \frac{1 + 2 + 3 + 4 + 5}{5} = 3
$$

2. 计算每个数据点与均值的差的平方：
$$
(1 - 3)^2 = 4
$$
$$
(2 - 3)^2 = 1
$$
$$
(3 - 3)^2 = 0
$$
$$
(4 - 3)^2 = 1
$$
$$
(5 - 3)^2 = 4
$$

3. 计算方差：
$$
\sigma^2 = \frac{4 + 1 + 0 + 1 + 4}{5} = 2
$$

4. 计算标准差：
$$
\sigma = \sqrt{2} \approx 1.41
$$

所以，这组数据的方差为 2，标准差约为 1.41。

---

