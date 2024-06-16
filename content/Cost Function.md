---
title: 
draft: false
tags:
  - model
  - regression
  - logistic
date: 2024-06-09
---

# Definition

>[!Cost Function]
>A loss function, also known as a cost function or objective function, is a mathematical function that quantifies the difference between the predicted values and the actual values in a machine learning model. The goal of training a model is to minimize this difference, thereby improving the model's accuracy. 
>
>在机器学习中，成本函数（Cost Function），也叫损失函数（Loss Function），用于衡量模型的预测结果与实际结果之间的差异。成本函数的值越小，表示模型的预测越准确。通过最小化成本函数，我们可以训练模型，使其预测更加准确。

# Purpose

The primary purpose of a loss function is to guide the training process of a machine learning model. By minimizing the loss, the model learns to make predictions that are closer to the actual outcomes.

# Type
## 1. Mean Squared Error (MSE)

### Definition
Mean Squared Error (MSE) measures the average of the squares of the errors, which are the differences between the predicted and actual values.

### Formula
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Explanation
- **Range**: \([0, \infty)\). The lower the MSE, the better the model's performance.
- **Usage**: Commonly used in regression tasks.
- **Models**: 
  - Linear Regression
  - Polynomial Regression
  - Neural Networks (for regression)

### Considerations
- Sensitive to outliers due to the squaring of errors.
- Use when larger errors should be penalized more heavily.

### Example
```python
# Aemon Wang
# aemooooon@gmail.com

from sklearn.metrics import mean_squared_error
import numpy as np

# Actual values
y_true = np.array([3.0, -0.5, 2.0, 7.0])

# Predicted values
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)
```

## 2. Mean Absolute Error (MAE)

### Definition
Mean Absolute Error (MAE) measures the average of the absolute differences between the predicted and actual values.

### Formula
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

### Explanation
- **Range**: \([0, \infty)\). The lower the MAE, the better the model's performance.
- **Usage**: Used in regression tasks.
- **Models**: 
  - Linear Regression
  - Polynomial Regression
  - Neural Networks (for regression)

### Considerations
- Less sensitive to outliers compared to MSE.
- Use when all errors should be penalized equally.

### Example
```python
# Aemon Wang
# aemooooon@gmail.com

from sklearn.metrics import mean_absolute_error
import numpy as np

# Actual values
y_true = np.array([3.0, -0.5, 2.0, 7.0])

# Predicted values
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error:", mae)
```

## 3. Huber Loss

### Definition
Huber Loss is a combination of MSE and MAE that is less sensitive to outliers in data than MSE.

### Formula
$$
L_\delta = \begin{cases}
\frac{1}{2}(y_i - \hat{y}_i)^2 & \text{for } |y_i - \hat{y}_i| \leq \delta \\
\delta |y_i - \hat{y}_i| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

### Explanation
- **Range**: \([0, \infty)\). The lower the Huber Loss, the better the model's performance.
- **Usage**: Used in regression tasks where robustness to outliers is needed.
- **Models**: 
  - Linear Regression
  - Polynomial Regression
  - Neural Networks (for regression)

### Considerations
- Combines the advantages of MSE (for small errors) and MAE (for large errors).
- Use when you need a balance between MSE and MAE.

### Example
```python
# Aemon Wang
# aemooooon@gmail.com

import numpy as np
from sklearn.metrics import mean_squared_error

def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta
    squared_loss = 0.5 * np.square(residual)
    linear_loss = delta * residual - 0.5 * np.square(delta)
    return np.where(condition, squared_loss, linear_loss).mean()

# Actual values
y_true = np.array([3.0, -0.5, 2.0, 7.0])

# Predicted values
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Calculate Huber Loss
hl = huber_loss(y_true, y_pred)
print("Huber Loss:", hl)
```

## 4. Cross-Entropy Loss (Log Loss)

### Definition
Cross-Entropy Loss measures the performance of a classification model whose output is a probability value between 0 and 1.

### Formula
For binary classification:
$$
L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

For multiclass classification:
$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{k} y_{ij} \log(\hat{y}_{ij})
$$

![[content/Images/LogLoss.png]]

## Log Loss vs MSE
![[content/Images/MSE.png]]
### Explanation
- **Range**: \([0, \infty)\). The lower the cross-entropy loss, the better the model's performance.
- **Usage**: Used in classification tasks.
- **Models**: 
  - Logistic Regression
  - Neural Networks (for classification)
  - Multiclass classifiers

### Considerations
- Penalizes wrong classifications more severely.
- Use when the model outputs probabilities.

### Example
```python
# Aemon Wang
# aemooooon@gmail.com

from sklearn.metrics import log_loss
import numpy as np

# Actual values
y_true = np.array([1, 0, 0, 1])

# Predicted probabilities
y_pred = np.array([0.9, 0.1, 0.2, 0.8])

# Calculate Cross-Entropy Loss
cross_entropy = log_loss(y_true, y_pred)
print("Cross-Entropy Loss:", cross_entropy)
```

## 5. Hinge Loss

### Definition
Hinge Loss is used for training classifiers, primarily for Support Vector Machines (SVMs).

### Formula
$$
L = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \hat{y}_i)
$$

### Explanation
- **Range**: \([0, \infty)\). The lower the hinge loss, the better the model's performance.
- **Usage**: Used in binary classification tasks.
- **Models**: 
  - Support Vector Machines (SVM)

### Considerations
- Only considers the misclassified points and the points within the margin.
- Use when training SVM classifiers.

### Example
```python
# Aemon Wang
# aemooooon@gmail.com

import numpy as np

def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Actual values (labels are -1 or 1)
y_true = np.array([1, -1, 1, -1])

# Predicted values
y_pred = np.array([0.8, -0.9, 0.7, -0.6])

# Calculate Hinge Loss
hl = hinge_loss(y_true, y_pred)
print("Hinge Loss:", hl)
```

## 6. Kullback-Leibler Divergence (KL Divergence)

### Definition
KL Divergence measures how one probability distribution diverges from a second, expected probability distribution.

### Formula
$$
D_{KL}(P \parallel Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
$$

### Explanation
- **Range**: \([0, \infty)\). The lower the KL divergence, the closer the distributions.
- **Usage**: Used in probabilistic models and variational autoencoders.
- **Models**: 
  - Variational Autoencoders (VAE)
  - Probabilistic models

### Considerations
- Not symmetric, meaning \(D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)\).
- Use when comparing probability distributions.

### Example
```python
# Aemon Wang
# aemooooon@gmail.com

import numpy as np
from scipy.special import rel_entr

# True distribution
P = np.array([0.1, 0.4, 0.5])

# Approximated distribution
Q = np.array([0.2, 0.3, 0.5])

# Calculate KL Divergence
kl_divergence = np.sum(rel_entr(P, Q))
print("KL Divergence:", kl_divergence)
```

