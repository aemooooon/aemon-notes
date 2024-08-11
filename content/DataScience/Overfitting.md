---
title: 
draft: false
tags:
  - model
date: 2024-06-09
---
>[!Definition]
>Overfitting occurs when a statistical model describes random error or noise in the data rather than the underlying relationship. An overfitted model performs well on training data but poorly on new, unseen data.

**Causes of Overfitting:**
- **Complex Models:** Too many parameters relative to the number of observations.
- **Noise in Data:** Fitting the noise instead of the actual data pattern.

**Signs of Overfitting:**
- **High Variance:** Model predictions vary widely for different training data sets.
- **Poor Generalization:** Good performance on training data but poor performance on validation/test data.

**Example:**
If we fit a polynomial regression model of a very high degree to a small dataset, it may perfectly fit the training data but fail to predict new data accurately.

**Visualization of Overfitting:**
1. **Training Data Fit:**
   - The model fits the training data perfectly, capturing all fluctuations and noise.
2. **Test Data Performance:**
   - The model performs poorly on test data, as it fails to generalize from the training data.

**Preventing Overfitting:**
- **Cross-Validation:** Use techniques like k-fold cross-validation to ensure the model generalizes well to unseen data.
- **Simpler Models:** Prefer simpler models that capture the underlying trend without fitting the noise.
- **Regularization:** Techniques like Lasso or Ridge regression add penalties to model complexity.

**Example in R:**

```r
# Sample data
set.seed(123)
x <- 1:10
y <- x + rnorm(10)

# Overfitting with a high-degree polynomial
overfit_model <- lm(y ~ poly(x, 10))
summary(overfit_model)

# Plotting the fit
plot(x, y, main = "Overfitting Example")
lines(x, predict(overfit_model, data.frame(x=x)), col = "red")
```

