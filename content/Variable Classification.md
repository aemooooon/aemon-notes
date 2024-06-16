---
title: 
draft: false
tags:
  - stats
date: 2024-06-06
---

# Categorical Variables (Qualitative Variables)

>[!Definition]
>Categorical variables describe categories or types, typically discrete, and do not contain quantitative information.

**Subcategories:**

1. **Nominal Variables**
   - **Definition:** Nominal variables represent unordered categories. There is no inherent order between categories.
   - **Examples:** Gender (Male, Female), Blood Type (A, B, AB, O), Country Name (China, USA, UK).

2. **Ordinal Variables**
   - **Definition:** Ordinal variables represent ordered categories. There is a clear order between categories, but the intervals between them may not be equal.
   - **Examples:** Education Level (Elementary, High School, College, University), Satisfaction (Very Dissatisfied, Dissatisfied, Neutral, Satisfied, Very Satisfied).

**Python Code Examples:**

```python
import pandas as pd

# Nominal Variable Example
gender = pd.Series(['Male', 'Female', 'Female', 'Male', 'Male'])
print("Nominal Variable (Gender):")
print(gender)

# Ordinal Variable Example
education_level = pd.Categorical(['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
                                 categories=['High School', 'Bachelor', 'Master', 'PhD'],
                                 ordered=True)
print("\nOrdinal Variable (Education Level):")
print(education_level)
```

# Numerical Variables (Quantitative Variables)

**Definition:**
Numerical variables represent quantities and measurements and can be subjected to arithmetic operations.

**Subcategories:**

1. **Continuous Variables**
   - **Definition:** Continuous variables can take an infinite number of values, usually real numbers. Any value between two values is possible.
   - **Examples:** Height (175.5 cm), Weight (70.2 kg), Temperature (36.6°C).

2. **Discrete Variables**
   - **Definition:** Discrete variables can take a finite or countably infinite number of discrete values, usually integers.
   - **Examples:** Number of people (1, 2, 3), Number of events (one, two).

**Python Code Examples:**

```python
import numpy as np

# Continuous Variable Example
heights = np.random.uniform(150, 200, size=1000)  # Generate 1000 continuous values between 150 and 200
print("Continuous Variable (Heights):", heights[:10])

# Discrete Variable Example
dice_rolls = np.random.randint(1, 7, size=1000)  # Generate 1000 discrete integer values between 1 and 6
print("Discrete Variable (Dice Rolls):", dice_rolls[:10])
```

# Classification by Measurement Scale

**Definition:**
Variables can be further classified based on their measurement scale into Nominal Scale, Ordinal Scale, Interval Scale, and Ratio Scale.

1. **Nominal Scale**
   - **Definition:** Represents categories or names with no order relationship.
   - **Examples:** Colors (Red, Blue, Green), Country Name (China, USA, UK).

2. **Ordinal Scale**
   - **Definition:** Represents ordered categories but does not determine the interval between orders.
   - **Examples:** Ranking (First, Second, Third), Satisfaction (Very Dissatisfied, Dissatisfied, Neutral, Satisfied, Very Satisfied).

3. **Interval Scale**
   - **Definition:** Represents ordered numerical values, allowing for comparison of intervals, but has no true zero point.
   - **Examples:** Temperature (0°C, 25°C, 100°C), Years (1990, 2000, 2010).

4. **Ratio Scale**
   - **Definition:** Has all properties of an interval scale and also has a true zero point, allowing for arithmetic operations.
   - **Examples:** Height (0 cm, 175 cm), Weight (0 kg, 70 kg), Time (0 hours, 2 hours, 3.5 hours).

**Python Code Examples:**

```python
import pandas as pd
import numpy as np

# Nominal Scale Example
colors = pd.Series(['Red', 'Blue', 'Green', 'Blue', 'Red'])
print("Nominal Scale (Colors):")
print(colors)

# Ordinal Scale Example
satisfaction = pd.Categorical(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied'],
                              categories=['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
                              ordered=True)
print("\nOrdinal Scale (Satisfaction):")
print(satisfaction)

# Interval Scale Example
temperatures = pd.Series([0, 25, 50, 75, 100])
print("\nInterval Scale (Temperatures in Celsius):")
print(temperatures)

# Ratio Scale Example
weights = pd.Series([0, 55, 65, 70, 75])
print("\nRatio Scale (Weights in kg):")
print(weights)
```

# Crossovers and Relationships

1. **Categorical Variables and Measurement Scales:**
   - Nominal variables typically use the Nominal Scale, e.g., Gender, Colors.
   - Ordinal variables typically use the Ordinal Scale, e.g., Education Level and satisfaction.

2. **Numerical Variables and Measurement Scales:**
   - Continuous and discrete variables can use the Interval Scale or Ratio Scale.
     - Interval Scale: Temperature (Celsius, Fahrenheit), Years.
     - Ratio Scale: Height, Weight, Time, Distance.
