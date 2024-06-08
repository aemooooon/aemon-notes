---
title: 
draft: false
tags:
  - topicmodel
date: 2024-06-05
---

>[!Collection Frequency]
>词频是指一个词在整个语料库中出现的总次数。它反映了该词在文本中的总体重要性。

>[!Document Frequency]
>文档频率是指一个词在多少个不同文档中出现。它反映了该词的普遍性和分布情况。

Understanding the concepts of Collection Frequency (CF) and Document Frequency (DF) and their application in text processing can help us perform text analysis and topic modeling more effectively. By setting appropriate CF and DF thresholds, we can optimize model performance and improve the reliability and interpretability of results. Choosing the right thresholds based on different datasets and application scenarios is a crucial practice step.
# Examples

Suppose we have the following five documents:

```
#0 : a, b, c, d, e, c
#1 : a, b, e, f
#2 : c, d, c
#3 : a, e, f, g
#4 : a, b, g
```

**Collection Frequency (CF):**
- a: 4 times
- b: 3 times
- c: 4 times
- d: 2 times
- e: 3 times
- f: 2 times
- g: 2 times

**Document Frequency (DF):**
- a: 4 documents (#0, #1, #3, #4)
- b: 3 documents (#0, #1, #4)
- c: 2 documents (#0, #2)
- d: 2 documents (#0, #2)
- e: 3 documents (#0, #1, #3)
- f: 2 documents (#1, #3)
- g: 2 documents (#3, #4)

# Functions

**Collection Frequency (CF)**
- **Noise Reduction:** Removing low-frequency words can reduce data noise and improve model stability and performance.
- **Efficiency Improvement:** By reducing the size of the vocabulary, model training and inference speed can be improved.

**Document Frequency (DF)**
- **Generalization:** Removing words with low document frequency can improve the model's generalization ability, as these words usually contribute little to most documents.
- **Topic Extraction:** In topic modeling, document frequency helps extract common themes that appear in multiple documents.

# Best Practices

**Strategies for Choosing CF and DF Thresholds**

1. **Based on Dataset Size:**
   - For larger datasets, choose higher CF and DF thresholds to reduce noise and improve computational efficiency.
   - For smaller datasets, choose lower thresholds to ensure enough information is retained for analysis.

2. **Based on Text Type:**
   - **News Articles:** Since news articles typically have higher word repetition, higher CF and DF thresholds can be set.
   - **Social Media:** Since social media texts are more casual and diverse, lower thresholds can be set to capture more variability.

3. **Model Requirements:**
   - **Topic Modeling:** Usually requires retaining words with higher DF to ensure theme coherence and interpretability.
   - **Classification Tasks:** Adjust CF and DF thresholds based on the specific needs of the classification task to optimize model performance.

**Example:**

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "a b c d e c",
    "a b e f",
    "c d c",
    "a e f g",
    "a b g"
]

# Set thresholds for CF and DF
min_cf = 3
min_df = 3

# Use CountVectorizer for vocabulary pruning
vectorizer = CountVectorizer(min_df=min_df)
X = vectorizer.fit_transform(documents)

# Display the vocabulary
print(vectorizer.get_feature_names_out())
```

The above code will remove words with document frequency less than 3, retaining only the words that meet the threshold criteria.
