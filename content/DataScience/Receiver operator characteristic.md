---
title: 
draft: false
tags: 
date: 2024-06-09
---

Receiver operator characteristic curves are a powerful way to assess and fine-tune trained classification models.

We can assess our classification models in terms of the kinds of mistakes that they make, such as false negatives and false positives. This can give insight into the kinds of mistakes a model makes, but doesn't necessarily provide deep information on how the model could perform if slight adjustments were made to its decision criteria. Here, we'll discuss receiver operator characteristic (ROC) curves, which build on the idea of a confusion matrix but provide us with deeper information that lets us improve our models to a greater degree.

If the true positive rate is very high, but the false positive rate is also very high, then the model is biased;

ROC 曲线展示了分类模型在所有可能的决策阈值下，真阳性率和假阳性率之间的权衡。

### 真阳性率 (True Positive Rate, TPR)
$$
\text{TPR} = \frac{TP}{TP + FN}
$$
真阳性率表示实际为正类的样本中被正确预测为正类的比例，也称为召回率 (Recall)。

### 假阳性率 (False Positive Rate, FPR)
$$
\text{FPR} = \frac{FP}{FP + TN}
$$
假阳性率表示实际为负类的样本中被错误预测为正类的比例。

## Area Under the Curve (AUC)

AUC 是 ROC 曲线下面积，是评估模型性能的一个标量值。AUC 的取值范围为 0 到 1，AUC 值越大，模型性能越好。

- **AUC = 0.5** 表示模型的表现与随机猜测一样。
- **AUC = 1.0** 表示模型能够完美地将正类和负类区分开。

## 示例代码

以下是使用 Python 生成 ROC 曲线和计算 AUC 的代码示例：

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 生成一个不平衡数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测概率
y_probs = model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# 计算 AUC
roc_auc = roc_auc_score(y_test, y_probs)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
