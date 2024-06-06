---
title: 
draft: false
tags:
  - stats
---
# Problem: Analyzing Dice Rolls

>[!Problem]
>Let's consider a problem where we roll a six-sided die multiple times and analyze the results. Each roll of the die produces one of six possible outcomes: 1, 2, 3, 4, 5, or 6. We will calculate the mean, mode, median, several quantiles, maximum value, minimum value, and identify outliers. We will also visualize the results using a bar plot and introduce the concept of the probability mass function (PMF). Finally, we will explain how to calculate probabilities using PMF with Python code.

# Generating Example Dataset

Let's generate a dataset representing the outcomes of rolling a six-sided die 100 times:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generating dice rolls data (100 rolls)
np.random.seed(42)  # for reproducibility
rolls = pd.Series(np.random.randint(1, 7, size=100))

# Custom function to calculate mode
def calculate_mode(data):
    values, counts = np.unique(data, return_counts=True)
    max_count_index = np.argmax(counts)
    return values[max_count_index]

# Calculating mean, mode, and median
mean_roll = rolls.mean()
mode_roll = calculate_mode(rolls)
median_roll = rolls.median()

# Calculating quantiles
quantiles = rolls.quantile([0.25, 0.5, 0.75])
q1 = quantiles[0.25]
q3 = quantiles[0.75]

# Calculating maximum and minimum
max_roll = rolls.max()
min_roll = rolls.min()

# Calculating interquartile range (IQR) and identifying outliers
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = rolls[(rolls < lower_bound) | (rolls > upper_bound)]

print(f"Mean: {mean_roll}")
print(f"Mode: {mode_roll}")
print(f"Median: {median_roll}")
print(f"Quantiles: \n{quantiles}")
print(f"Max: {max_roll}")
print(f"Min: {min_roll}")
print(f"Lower Bound for Outliers: {lower_bound}")
print(f"Upper Bound for Outliers: {upper_bound}")
print(f"Outliers: \n{outliers}")
```

## Data Analysis

1. **Mean, Mode, Median, Quantiles, Maximum, Minimum, Outliers:**
   - **Mean:** Average of the dice rolls.
   - **Mode:** The most frequently occurring roll.
   - **Median:** The middle roll when sorted.
   - **Quantiles:** Values that divide the dataset into equal-sized intervals.
   - **Maximum:** The highest roll.
   - **Minimum:** The lowest roll.
   - **Outliers:** Rolls that lie outside the typical range defined by 1.5 times the interquartile range (IQR).
# Plotting Bar Plot

```python
# Plotting Bar Plot
plt.figure(figsize=(10, 5))

plt.bar(rolls.value_counts().index, rolls.value_counts().values, color='blue', edgecolor='black')
plt.title('Bar Plot of Dice Rolls')
plt.xlabel('Dice Roll')
plt.ylabel('Frequency')

plt.show()
```

**Bar Plot:**
   - **x-axis:** Represents the outcomes of the dice rolls (1, 2, 3, 4, 5, 6).
   - **y-axis:** Represents the frequency (number of occurrences) of each outcome.
   - **Meaning:** The bar plot shows how frequently each outcome appears in the dataset.

# Introducing the Probability Mass Function (PMF)

The probability mass function (PMF) describes the probability of a discrete random variable taking on a particular value. PMF provides the probability of each possible outcome of the dice rolls.

## Calculating PMF and Interval Probabilities

```python
# Calculating the PMF
pmf = rolls.value_counts(normalize=True).sort_index()
print(f"PMF: \n{pmf}")

# Visualizing PMF
plt.figure(figsize=(10, 5))
plt.bar(pmf.index, pmf.values, color='blue', edgecolor='black')
plt.title('Probability Mass Function (PMF) of Dice Rolls')
plt.xlabel('Dice Roll')
plt.ylabel('Probability')

plt.show()

# Calculating the probability of a specific outcome (e.g., roll = 4)
prob_of_4 = pmf[4]
print(f"Probability of roll 4: {prob_of_4}")

# Calculating the probability of outcomes in a specific range (e.g., roll between 3 and 5)
prob_3_to_5 = pmf.loc[3:5].sum()
print(f"Probability of rolls between 3 and 5: {prob_3_to_5}")
```

**PMF Bar Plot:**
   - **x-axis:** Represents the outcomes of the dice rolls (1, 2, 3, 4, 5, 6).
   - **y-axis:** Represents the probability of each outcome.
   - **Meaning:** The PMF bar plot shows the probability of each dice roll outcome.
   




