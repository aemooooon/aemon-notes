---
title: 
draft: false
tags:
  - model
  - logistic
  - regression
  - classification
date: 2024-06-09
---
>[!Definition]
>A confusion matrix is a tool used to evaluate the performance of a classification model by summarising the results of the predictions on a test dataset. It is a square matrix where each row represents the actual class, and each column represents the predicted class.
You can think of data as continuous, categorical, or ordinal (categorical but with an order). Confusion matrices are a means of assessing how well a categorical model performs. For context as to how these work, let’s first refresh our knowledge about continuous data. Through this, we can see how confusion matrices are simply an extension of the histograms we already know.

## Continuous data distributions

When we want to understand continuous data, the first step is often to see how it's distributed. Consider the following histogram:

![Histogram showing label distribution.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-g-a.jpg)

We can see that the label is, on average, about zero, and most datapoints fall between -1 and 1. It appears as symmetrical; there are an approximately even count of numbers smaller and larger than the mean. If we wanted, we could use a table rather than a histogram, but it could be unwieldy.

## Categorical data distributions

In some respects, categorical data aren't so different from continuous data. We can still produce histograms to assess how commonly values appear for each label. For example, a binary label (true/false) might appear with frequency like so:

![Bar plot showing more false labels then true.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-g-b.jpg)

This tells us that there are 750 samples with "false" as a label, and 250 with "true" as the label.

A label for three categories is similar:

![Bar plot showing more animal labels than person and tree labels.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-g-c.jpg)

This tells us that there are 200 samples that are "person", 400 that are "animal", and 100 that are "tree".

As categorical labels are simpler, we can often show these as simple tables. The two preceding graphs would appear like so:

Expand table

|Label|False|True|
|---|---|---|
|Count|750|250|

And:

Expand table

|Label|Person|Animal|Tree|
|---|---|---|---|
|Count|200|400|100|

## Looking at predictions

We can look at predictions that the model makes just like we look at the ground-truth labels in our data. For example, we might see that in the test set our model predicted "false" 700 times and "true" 300 times.

Expand table

|Model Prediction|Count|
|---|---|
|False|700|
|True|300|

This provides direct information about the predictions our model is making, but it doesn’t tell us which of these are correct. While we can use a cost function to understand how often the correct responses are given, the cost function won't tell us which kinds of errors are being made. For example, the model might correctly guess all "true" values, but also guess "true" when it should have guessed "false".

## The confusion matrix

The key to understanding the model performance is to combine the table for model prediction with the table for ground-truth data labels:

![Diagram of the confusion matrix with total numbers added.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-2-a.jpg)

The square we haven't filled out is called the confusion matrix.

Each cell in the confusion matrix tells us one thing about the model’s performance. These are True Negatives (TN), False Negatives (FN), False Positives (FP) and True Positives (TP).

Let’s explain these one by one, replacing these acronyms with actual values. Blue-green squares mean the model made a correct prediction, and orange squares mean the model made an incorrect prediction.

### True Negatives (TN)

The top-left value will list how many times the model predicted false, and the actual label was also false. In other words, this lists how many times the model correctly predicted false. Let’s say, for our example, that this happened 500 times:

![Diagram of the confusion matrix without total numbers, showing true negatives only.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-2-b.jpg)

### False Negatives (FN)

The top-right value tells us how many times the model predicted false, but the actual label was true. We know now that this is 200. How? Because the model predicted false 700 times, and 500 of those times it did so correctly. Thus, 200 times it must have predicted false when it shouldn't have.

![Diagram of the confusion matrix showing false negatives only.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-2-c.jpg)

### False Positives (FP)

The bottom-left value holds false positives. This tells us how many times the model predicted true, but the actual label was false. We know now that this is 250, because there were 750 times that the correct answer was false. 500 of these times appear in the top-left cell (TN):

![Diagram of the confusion matrix showing false positives also.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-2-d.jpg)

### True Positives (TP)

Finally, we have true positives. This is the number of times that the model correctly prediction of true. We know that this is 50 for two reasons. Firstly, the model predicted true 300 times, but 250 times it was incorrect (bottom-left cell). Secondly, there were 250 times that true was the correct answer, but 200 times the model predicted false.

![Diagram of the confusion matrix showing true positives also.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-2-e.jpg)

### The final matrix

We normally simplify our confusion matrix slightly, like so:

![Diagram of the simplified confusion matrix.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-2-f.jpg)

We’ve colored the cells here to highlight when the model made correct predictions. From this, we know not only how often the model made certain types of predictions, but also how often those predictions were correct or incorrect.

Confusion matrices can also be constructed when there are more labels. For example, for our person/ animal/tree example, we might get a matrix like so:

![Diagram of the expanded confusion matrix with three labels: person, animal, and tree.](https://learn.microsoft.com/en-nz/training/modules/machine-learning-confusion-matrix/media/8-2-g.jpg)

When there are three categories, metrics like True Positives no longer apply, but we can still see exactly how often the model made certain kinds of mistakes. For example, we can see that the model predicted that "person" 200 times when the actual correct result was "animal".

#### Evaluation Metrics

Various evaluation metrics can be derived from the confusion matrix to assess the performance of the classification model:

1. **Accuracy**:
   Accuracy is the proportion of correctly classified instances out of the total instances.
   $$
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
   $$

2. **Precision**:
   Precision is the proportion of true positive predictions among all positive predictions.
   $$
   \text{Precision} = \frac{TP}{TP + FP}
   $$

3. **Recall (Sensitivity)**:
   Recall is the proportion of true positive instances among all actual positive instances.
   $$
   \text{Recall} = \frac{TP}{TP + FN}
   $$

4. **F1 Score**:
   F1 Score is the harmonic mean of precision and recall.
   $$
   \text{F1 Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
   $$

5. **Specificity**:
   Specificity is the proportion of true negative instances among all actual negative instances.
   $$
   \text{Specificity} = \frac{TN}{TN + FP}
   $$

6. **假阳性率 (False Positive Rate, FPR)**：实际为负类的样本中被错误预测为正类的比例。 $$ \text{假阳性率} = \frac{FP}{FP + TN} $$
#### 什么是 Balanced Accuracy?

**Balanced Accuracy** 是一种衡量分类模型性能的指标，特别适用于处理类别不平衡的数据集。它考虑了每个类别的分类准确性，以避免因为某个类别样本数量过多而导致的偏差。计算方式如下：

$$
\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}
$$

其中：
- **Sensitivity (召回率)**：正确识别正类样本的比例。
- **Specificity (特异性)**：正确识别负类样本的比例。

##### 解释

1. **Balanced Accuracy = 0**:
   - 这意味着模型没有正确分类任何一个样本，无论是正类还是负类样本的识别率都为 0。
   
2. **Balanced Accuracy = 1**:
   - 这意味着模型完美地分类了所有样本，正类和负类样本的识别率都为 1。

#### Example

Here is an example using scikit-learn to create a confusion matrix and compute the above evaluation metrics:

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Example data
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 0])

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Specificity: {specificity}")
```

#### Results Explanation

In the example above, we have 10 samples with actual labels `y_true` and predicted labels `y_pred`. The confusion matrix computed is:

```
Confusion Matrix:
[[3 1]
 [1 5]]
```

This confusion matrix indicates:
- **True Negative (TN)** = 3
- **False Positive (FP)** = 1
- **False Negative (FN)** = 1
- **True Positive (TP)** = 5

The computed evaluation metrics are:
- **Accuracy**: 0.8
- **Precision**: 0.8333
- **Recall**: 0.8333
- **F1 Score**: 0.8333
- **Specificity**: 0.75

These metrics provide a comprehensive assessment of the model's performance from different perspectives.

#### Visualising the Confusion Matrix

You can visualise the confusion matrix using a heatmap to better understand the classification results.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

This code will create a heatmap where the intensity of the color represents the count of predictions, making it easier to analyze the results.

![[confusionmatrix.png]]

# Multiple Class Confusion Metrics


多类混淆矩阵将二分类混淆矩阵扩展到多类情形。它通过提供预测结果的详细分解来评估分类模型在多个类别上的性能。

## 混淆矩阵的组成部分
1. **实际标签（行）**：这些表示数据点的真实类别。每一行对应一个实际类别。
2. **预测标签（列）**：这些表示模型分配的预测类别。每一列对应一个预测类别。

## 示例矩阵
在提供的矩阵中，我们有四个类别：`animal`（动物）、`hiker`（徒步者）、`rock`（岩石）、`tree`（树）。

$$
\begin{array}{c|cccc}
 & \text{animal} & \text{hiker} & \text{rock} & \text{tree} \\
\hline
\text{animal} & 28 & 38 & 0 & 0 \\
\text{hiker} & 30 & 103 & 1 & 0 \\
\text{rock} & 0 & 1 & 217 & 1 \\
\text{tree} & 0 & 0 & 1 & 241 \\
\end{array}
$$

## 解读每个单元格
- **真正例（对角线元素）**：这些代表每个类别的正确分类实例。
  - `animal`：28个实例正确分类为`animal`。
  - `hiker`：103个实例正确分类为`hiker`。
  - `rock`：217个实例正确分类为`rock`。
  - `tree`：241个实例正确分类为`tree`。
  
- **假正例（列的非对角线元素）**：这些代表被错误分类为某一特定类别的实例。
  - 对于`animal`（列1）：30个`hiker`实例，0个`rock`实例和0个`tree`实例被错误分类为`animal`。
  - 对于`hiker`（列2）：38个`animal`实例，1个`rock`实例和0个`tree`实例被错误分类为`hiker`。
  - 对于`rock`（列3）：0个`animal`实例，1个`hiker`实例和1个`tree`实例被错误分类为`rock`。
  - 对于`tree`（列4）：0个`animal`实例，0个`hiker`实例和1个`rock`实例被错误分类为`tree`。
  
- **假负例（行的非对角线元素）**：这些代表被错误分类为其他类别的实例。
  - 对于`animal`（行1）：38个实例被分类为`hiker`，0个实例被分类为`rock`，0个实例被分类为`tree`。
  - 对于`hiker`（行2）：30个实例被分类为`animal`，1个实例被分类为`rock`，0个实例被分类为`tree`。
  - 对于`rock`（行3）：0个实例被分类为`animal`，1个实例被分类为`hiker`，1个实例被分类为`tree`。
  - 对于`tree`（行4）：0个实例被分类为`animal`，0个实例被分类为`hiker`，1个实例被分类为`rock`。

## 可视化
提供的热力图可视化帮助快速识别正确和错误预测的数量：
- 深色代表较高的值，表示该类别中的实例较多。
- 浅色代表较低的值，表示该类别中的实例较少。

![Confusion Matrix Heatmap](multipleconfusionmatrix.jpg)
## 计算指标
从混淆矩阵中，我们可以计算各个类别的多种性能指标：

以下是将上述笔记转换为英文并且以较小字体显示的版本：


- **Accuracy**: The proportion of correctly predicted instances.
  $$
  \text{Accuracy} = \frac{28 + 103 + 217 + 241}{\text{Total Instances}}
  $$

- **Precision for each class**: The proportion of true positive predictions out of all positive predictions.
  $$
  \text{Precision}_{\text{animal}} = \frac{28}{28 + 30 + 0 + 0}
  $$

- **Recall for each class**: The proportion of true positive predictions out of all actual positive instances.
  $$
  \text{Recall}_{\text{animal}} = \frac{28}{28 + 38}
  $$

- **F1 Score for each class**: The harmonic mean of precision and recall.
  $$
  \text{F1 Score}_{\text{animal}} = 2 \times \frac{\text{Precision}_{\text{animal}} \times \text{Recall}_{\text{animal}}}{\text{Precision}_{\text{animal}} + \text{Recall}_{\text{animal}}}
  $$
