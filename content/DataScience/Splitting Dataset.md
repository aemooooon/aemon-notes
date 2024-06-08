---
title: 
draft: false
tags:
  - model
date: 2024-06-09
---
## Split Train and Test Dataset

### What is Splitting Train and Test Dataset?

Splitting a dataset into training and test sets is a fundamental technique in machine learning used to evaluate the performance and generalization ability of a model. 

- **Training Set**: The portion of the dataset used to train the model. It helps the model learn the underlying patterns in the data.
- **Test Set**: The portion of the dataset used to evaluate the model's performance. It helps assess how well the model generalizes to new, unseen data.

### Why Do We Split the Dataset?

1. **Evaluate Model Performance**: To objectively assess how well the model will perform on new, unseen data.
2. **Prevent Overfitting**: To ensure the model is not just memorizing the training data but learning the underlying patterns.
3. **Model Validation**: To fine-tune hyperparameters and select the best model through techniques like cross-validation.

### How to Split the Dataset?

#### Basic Split

The simplest way to split the dataset is to divide it into two parts: a training set and a test set. This can be done using various libraries such as scikit-learn.

```python
from sklearn.model_selection import train_test_split

# Example dataset
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set:", X_train, y_train)
print("Test set:", X_test, y_test)
```

#### Cross-Validation

For a more robust evaluation, cross-validation can be used. This involves splitting the data into multiple training and test sets to ensure the model's performance is consistent.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Example dataset
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a model
model = LinearRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())
```

### Industry Best Practices

1. **Random State**: Always set a random state to ensure reproducibility.
   ```python
   train_test_split(X, y, test_size=0.2, random_state=42)
   ```

2. **Stratified Split**: When dealing with imbalanced datasets, use stratified splitting to ensure the training and test sets have a similar distribution of classes.
   ```python
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5)
   ```

3. **Data Leakage**: Ensure that the test set is not used during the training process to avoid data leakage, which can lead to overly optimistic performance estimates.

4. **Scaling Data**: Apply scaling to both training and test sets but fit the scaler only on the training data to prevent data leakage.
   ```python
   from sklearn.preprocessing import StandardScaler

   # Fit the scaler on the training data
   scaler = StandardScaler().fit(X_train)
   
   # Transform the training and test data
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

5. **Large Datasets**: For very large datasets, a simple train-test split might suffice, but ensure the test set is representative of the overall data.

6. **Time Series Data**: For time series data, use techniques like time-based splitting to maintain the temporal order of the data.
   ```python
   train_size = int(len(data) * 0.8)
   train, test = data[:train_size], data[train_size:]
   ```

### Example: Train/Test Split and Model Evaluation

Here is a complete example demonstrating how to split a dataset, train a model, and evaluate its performance using MSE:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate example data
data = pd.DataFrame({
    'height': np.random.randint(150, 200, 100),
    'rescues_last_year': np.random.randint(50, 70, 100)
})

# Split the data
X = data[['height']]
y = data['rescues_last_year']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on training and test set
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("MSE on training set:", mse_train)
print("MSE on test set:", mse_test)
```
