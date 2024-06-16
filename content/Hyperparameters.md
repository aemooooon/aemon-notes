---
title: 
draft: false
tags: 
date: 2024-06-13
---
>[!Hyperparameters 超参数]
>In machine learning, hyperparameters are values that are set before the learning process begins and control the behavior of the training algorithm. They are different from parameters, which are learned from the training data.

> - **Parameters**: Values learned from the data during training (e.g., weights in a neural network).
> - **Hyperparameters**: Values specified by the user to guide the training process (e.g., learning rate, number of epochs).

## Types of Hyperparameters

### Model Hyperparameters
These define the structure and complexity of the model.

- **Number of Layers and Neurons**: In neural networks, the number of hidden layers and the number of neurons in each layer.
- **Kernel Type**: In support vector machines (SVM), the type of kernel used (e.g., linear, polynomial, RBF).

### Training Hyperparameters
These control the training process.

- **Learning Rate**: Controls the step size in the optimization process. A higher learning rate can speed up training but may cause the model to converge to a suboptimal solution.
- **Batch Size**: The number of training examples used in one iteration to update the model parameters.
- **Number of Epochs**: The number of complete passes through the training dataset.
- **Regularization Parameters**: Parameters that prevent overfitting by penalizing large weights (e.g., L1, L2 regularization).

## Common Hyperparameters in Different Algorithms

### Linear Regression
- **Regularization Strength**: Controls the amount of regularization applied to the model (L1, L2).

### Decision Trees
- **Max Depth**: The maximum depth of the tree.
- **Min Samples Split**: The minimum number of samples required to split an internal node.

### Random Forest
- **Number of Trees**: The number of trees in the forest.
- **Max Features**: The maximum number of features considered for splitting a node.

### Neural Networks
- **Learning Rate**: The step size for updating the weights.
- **Batch Size**: The number of samples per gradient update.
- **Number of Epochs**: The number of times the entire training dataset is passed forward and backward through the neural network.
- **Dropout Rate**: The fraction of the input units to drop to prevent overfitting.

## Hyperparameter Tuning

Hyperparameter tuning is the process of finding the optimal hyperparameters for a model. Common methods include:

### Grid Search
- An exhaustive search over a specified parameter grid.
- Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
```

### Random Search
- Samples a fixed number of hyperparameter settings from a specified distribution.
- Example:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': [None, 10, 20, 30]
}

random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_dist, n_iter=50, cv=5)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
```

### Bayesian Optimization
- Uses Bayesian techniques to model the performance of the hyperparameters and selects new hyperparameters based on previous results.

### Cross-Validation
- Used to assess the performance of the model with different hyperparameters by dividing the data into training and validation sets multiple times.
