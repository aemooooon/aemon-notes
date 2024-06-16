---
title: 
draft: false
tags: 
date: 2024-06-15
---
 >[!Youden's Index]
 >also known as Youden's J statistic, is a metric used to evaluate the effectiveness of a diagnostic test. It helps determine the optimal threshold value for binary classification systems by balancing sensitivity and specificity.

### Formula

$$
J = \text{Sensitivity} + \text{Specificity} - 1
$$

where:
- **Sensitivity (True Positive Rate, TPR)**: The proportion of actual positives correctly identified by the test.
- **Specificity (True Negative Rate, TNR)**: The proportion of actual negatives correctly identified by the test.

### Interpretation

- **Range**: Youden's Index ranges from 0 to 1.
  - A value of 0 indicates that the test is no better than random chance.
  - A value of 1 indicates a perfect test with no false positives or false negatives.

- **Optimal Threshold**: The threshold that maximizes Youden's Index is considered the optimal threshold, achieving the best balance between sensitivity and specificity.

## Application in Logistic Regression

We can use Youden's Index to find the optimal threshold for a logistic regression model by evaluating different thresholds and selecting the one with the highest index. Here's an example to illustrate this process.

### Example: Finding the Optimal Threshold in Logistic Regression

Let's start with a simple logistic regression model and compute the ROC curve, AUC, and optimal threshold using [[Youden's Index]].

#### Step 1: Load and Split Data

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Step 2: Train the [[Logistic Regression]] Model

```python
# Train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
```

#### Step 3: Predict Probabilities and Compute [[ROC]] Curve

```python
# Predict probabilities for the test set
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
```

#### Step 4: Calculate [[Youden's Index]] and Find the Optimal Threshold

```python
# Calculate Youden's Index for each threshold
youden_index = tpr - fpr
optimal_threshold = thresholds[np.argmax(youden_index)]

print(f'Best Threshold: {optimal_threshold}')
print(f'ROC AUC: {roc_auc}')
```

#### Step 5: Plot the [[ROC]] Curve and Optimal Threshold

```python
# Plot ROC curve
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.scatter(fpr[np.argmax(youden_index)], tpr[np.argmax(youden_index)], marker='o', color='black', label=f'Best Threshold = {optimal_threshold:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```


