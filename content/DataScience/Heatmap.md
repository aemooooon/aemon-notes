---
title: 
draft: false
tags:
  - viz
---

#  Heatmap(热力图)

> [!Definition]
>热力图是一种数据可视化技术，通过颜色的变化来表示数据值的大小。它通常用于表示二维数据矩阵的值分布情况。

## Significance and Use
1. **数据分析**：用于识别数据中的模式、趋势和异常值。
2. **数据可视化**：通过颜色的变化直观展示数据的分布情况，有助于理解复杂的数据集。
3. **应用领域**：广泛应用于统计分析、生物信息学、金融分析、市场研究等领域。

## Data Types and Variables
### Data Types
热力图可以接收以下数据类型：
1. **二维数组或矩阵**：如 NumPy 数组 (numpy.ndarray) 或 Python 列表的列表 (list of lists)。
2. **Pandas DataFrame**：用于处理带有行列标签的数据。

### Number of Variables
- 热力图通常用于二维数据，因此需要两个维度的数据，每个维度可以有任意数量的变量。
- 行和列的数量可以根据具体数据集的大小变化。

### Variable Types
- **数值型变量**：热力图主要用于表示数值型数据，要求数据中的每个元素都是数值（整数或浮点数）。
- **分类变量**：行和列的标签可以是分类变量，用于标识每个数值对应的类别或组。

### Example Data Structures
1. **二维数组**：
```python
import numpy as np
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
```

2. **Pandas DataFrame**：
```python
import pandas as pd
data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})
```

## Python Code Example
使用 Seaborn 和 Matplotlib 生成热力图：
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 生成随机数据
data = np.random.rand(10, 12)

# 使用 Seaborn 生成热力图
sns.heatmap(data, annot=True, fmt=".1f", cmap='coolwarm')

# 显示图像
plt.show()
```

### 使用 Pandas DataFrame 生成热力图
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建 DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [10, 9, 8, 7, 6],
        'D': [6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

# 使用 Seaborn 生成热力图
sns.heatmap(df, annot=True, fmt="d", cmap='viridis')

# 显示图像
plt.show()
```

## Output Explanation
1. **数据矩阵**：输入数据可以是 NumPy 数组或 Pandas DataFrame，表示数据的二维矩阵。
2. **颜色映射**：数据值通过颜色映射表示，不同的颜色表示不同的数据范围。
3. **注释**：可以选择在热力图上显示每个数据值的具体数值。

热力图是一种强大的数据可视化工具，通过颜色的变化直观展示数据分布情况，帮助识别数据中的模式和异常值，非常适合用于数据分析和可视化。
