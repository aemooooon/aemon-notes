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

#### Structure of the Confusion Matrix

The basic structure of a confusion matrix is as follows:

|               | Predicted: Positive | Predicted: Negative |
|---------------|----------------------|----------------------|
| Actual: Positive | True Positive (TP)     | False Negative (FN)    |
| Actual: Negative | False Positive (FP)    | True Negative (TN)     |

- **True Positive (TP)**: The model correctly predicts the positive class.
- **False Negative (FN)**: The model incorrectly predicts the negative class for a positive instance.
- **False Positive (FP)**: The model incorrectly predicts the positive class for a negative instance.
- **True Negative (TN)**: The model correctly predicts the negative class.

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

