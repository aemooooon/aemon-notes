---
title: 
draft: false
tags:
  - stats
---
# Calculating Mean, Mode, Median, Quantiles, Maximum, Minimum, and Outliers

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, mode

# Given grades data
grades = pd.Series([50.0, 50.0, 47.0, 97.0, 49.0, 3.0, 53.0, 42.0, 26.0, 74.0,
                    82.0, 62.0, 37.0, 15.0, 70.0, 27.0, 36.0, 35.0, 48.0, 52.0,
                    63.0, 64.0])

# Calculating mean, mode, and median
mean_grade = grades.mean()
mode_grade = mode(grades).mode[0]
median_grade = grades.median()

# Calculating quantiles
quantiles = grades.quantile([0.25, 0.5, 0.75])
q1 = quantiles[0.25]
q3 = quantiles[0.75]

# Calculating maximum and minimum
max_grade = grades.max()
min_grade = grades.min()

# Calculating interquartile range (IQR) and identifying outliers
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = grades[(grades < lower_bound) | (grades > upper_bound)]

print(f"Mean: {mean_grade}")
print(f"Mode: {mode_grade}")
print(f"Median: {median_grade}")
print(f"Quantiles: \n{quantiles}")
print(f"Max: {max_grade}")
print(f"Min: {min_grade}")
print(f"Lower Bound for Outliers: {lower_bound}")
print(f"Upper Bound for Outliers: {upper_bound}")
print(f"Outliers: \n{outliers}")
```

#### Plotting Histogram, Density Plot, and Box Plot

```python
# Plotting Histogram
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(grades, bins=10, edgecolor='black')
plt.title('Histogram of Grades')
plt.xlabel('Grade')
plt.ylabel('Frequency')

# Plotting Density Plot
plt.subplot(1, 3, 2)
sns.kdeplot(grades, shade=True)
plt.title('Density Plot of Grades')
plt.xlabel('Grade')
plt.ylabel('Density')

# Plotting Box Plot
plt.subplot(1, 3, 3)
sns.boxplot(grades)
plt.title('Box Plot of Grades')
plt.xlabel('Grade')

plt.tight_layout()
plt.show()
```

### Explanation of Plots

1. **Histogram:**
   - **x-axis:** Represents the grade values.
   - **y-axis:** Represents the frequency (number of occurrences) of each grade range.
   - **Meaning:** The histogram shows how frequently each grade range appears in the dataset.

2. **Density Plot:**
   - **x-axis:** Represents the grade values.
   - **y-axis:** Represents the probability density.
   - **Meaning:** The density plot is a smoothed version of the histogram. It shows the distribution of grades and helps identify the areas where grades are more concentrated.

3. **Box Plot:**
   - **x-axis:** Represents the grade values.
   - **y-axis:** Not applicable here as it’s a vertical plot.
   - **Meaning:** The box plot shows the five-number summary of the data: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It also highlights any potential outliers.

### Introducing the Probability Density Function (PDF)

The [[Probability Density Function]] (PDF) describes the likelihood of a continuous random variable to take on a particular value. For a given interval, the area under the curve of the PDF over that interval represents the probability of the variable falling within the interval.

#### Calculating Probability Density and Interval Probabilities

```python
# Calculate the probability density function
kde = gaussian_kde(grades)
density_at_37 = kde(37)[0]
print(f"Density at 37: {density_at_37}")

# Calculate probability between 36 and 38
prob_36_to_38 = kde.integrate_box_1d(36, 38)
print(f"Probability between 36 and 38: {prob_36_to_38:.4f}")
```

### Explanation of PDF and Interval Probability

- **Probability Density at a Point (e.g., 37):** 
  - **Meaning:** The density value at x = 37 indicates how dense the data is around the score of 37. It is not a probability but a density value.
  - **Example:** Density at 37: 0.014779821322454553 means the data is relatively dense around 37.

- **Probability over an Interval (e.g., 36 to 38):**
  - **Meaning:** The probability of grades falling between 36 and 38 is the area under the PDF curve between these two points.
  - **Example:** Probability between 36 and 38: 0.0296 means there is a 2.96% chance that a randomly selected grade from this dataset will fall within this range.

 