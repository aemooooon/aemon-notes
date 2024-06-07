---
title: 
draft: false
tags:
  - stats
---
>[!Definition]
>One-hot vectors are a common way to represent categorical data as numerical data in machine learning. In a one-hot encoding, each category is represented as a binary vector, where only one element is `1` (hot), and all others are `0`.

# Why Use One-Hot Encoding?
- **Categorical Data**: Many machine learning algorithms require numerical input, so categorical data must be converted to numerical form.
- **Avoid Ordinal Relationships**: One-hot encoding prevents algorithms from assuming a natural ordering between categories.

# How One-Hot Encoding Works
Suppose you have a list of categories: `["cat", "dog", "fish"]`.

1. **Unique Categories**: Identify all unique categories.
2. **Binary Vector**: Create a binary vector for each category where only the index corresponding to the category is `1`.

For example:
- "cat" -> [1, 0, 0]
- "dog" -> [0, 1, 0]
- "fish" -> [0, 0, 1]

# One-Hot Encoding in Python

#### Using Pandas
Pandas has a built-in method for one-hot encoding called `get_dummies`.

```python
import pandas as pd

# Example DataFrame
data = {'Animal': ['cat', 'dog', 'fish', 'cat', 'fish']}
df = pd.DataFrame(data)

# One-hot encoding using get_dummies
one_hot_encoded_df = pd.get_dummies(df, columns=['Animal'])

print(one_hot_encoded_df)
```

Output:
```
   Animal_cat  Animal_dog  Animal_fish
0           1           0            0
1           0           1            0
2           0           0            1
3           1           0            0
4           0           0            1
```

## Using Scikit-Learn
Scikit-learn provides a `OneHotEncoder` for this purpose.

```python
from sklearn.preprocessing import OneHotEncoder

# Example data
data = [['cat'], ['dog'], ['fish'], ['cat'], ['fish']]

# Create the encoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the data
one_hot_encoded = encoder.fit_transform(data)

print(one_hot_encoded)
```

Output:
```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]]
```

## Using TensorFlow
If you are working with TensorFlow, you can use its built-in functionality to create one-hot encodings.

```python
import tensorflow as tf

# Example data
categories = tf.constant(['cat', 'dog', 'fish', 'cat', 'fish'])

# Integer encode the categories
category_indices = tf.factorize(categories)[0]

# One-hot encode the indices
one_hot_encoded = tf.one_hot(category_indices, depth=3)

print(one_hot_encoded)
```

Output:
```
tf.Tensor(
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]], shape=(5, 3), dtype=float32)
```

