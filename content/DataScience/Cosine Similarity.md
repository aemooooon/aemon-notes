---
title: 
draft: false
tags: 
date:
---
## 概念
**余弦相似度（Cosine Similarity）** 是一种衡量两个向量之间相似度的指标，常用于文本分析、推荐系统、聚类分析等领域。它通过计算两个向量的夹角余弦值来判断它们的相似程度：

- 余弦相似度值范围：$[-1, 1]$
  - 值为 **1** 表示完全相似（向量同方向）
  - 值为 **0** 表示不相似（向量垂直，即夹角90°）
  - 值为 **-1** 表示完全相反（向量方向相反）

### 使用场景
- **文本相似性**：衡量两段文本的内容相似度。
- **推荐系统**：在用户-物品推荐中，通过计算特征向量的相似性，推荐相似的物品或用户。
- **聚类分析**：用于识别相似性较高的聚类簇。

## 数学计算方法
给定两个向量 $A$ 和 $B$，余弦相似度的公式为：

$$
\text{cosine\_similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
$$

其中：
- $A \cdot B$ 表示向量的点积：
  $$
  A \cdot B = \sum_{i=1}^n A_i \times B_i
  $$
- $\|A\|$ 和 $\|B\|$ 分别是向量的模（长度），计算方法为：
  $$
  \|A\| = \sqrt{\sum_{i=1}^n A_i^2}, \quad \|B\| = \sqrt{\sum_{i=1}^n B_i^2}
  $$

### 举例
假设向量 $A = [1, 2, 3]$ 和 $B = [4, 5, 6]$，计算余弦相似度的步骤如下：

1. **计算点积**：
   $$
   A \cdot B = (1 \times 4) + (2 \times 5) + (3 \times 6) = 32
   $$

2. **计算模**：
   $$
   \|A\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14}
   $$
   $$
   \|B\| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{77}
   $$

3. **计算余弦相似度**：
   $$
   \text{cosine\_similarity} = \frac{32}{\sqrt{14} \cdot \sqrt{77}} \approx 0.9756
   $$

## Python 计算方法
使用 `numpy` 库可以简单快速地计算余弦相似度：

```python
# Aemon Wang
# Email: aemooooon@gmail.com

import numpy as np

# 定义两个向量 A 和 B
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# 计算点积 A · B
dot_product = np.dot(A, B)

# 计算向量 A 和 B 的模
norm_A = np.linalg.norm(A)
norm_B = np.linalg.norm(B)

# 计算余弦相似度
cosine_similarity = dot_product / (norm_A * norm_B)

print("Cosine Similarity:", cosine_similarity)
```

### 解释
1. `np.dot(A, B)` 计算两个向量的点积。
2. `np.linalg.norm(A)` 和 `np.linalg.norm(B)` 分别计算向量 A 和 B 的模。
3. 将点积除以模的乘积即可得到余弦相似度。

### 输出
运行上述代码将输出：
```
Cosine Similarity: 0.974631846197