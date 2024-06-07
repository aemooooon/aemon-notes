---
title: 
draft: false
tags:
  - topicmodel
  - classification
  - textmining
---
# Bag of Words (BoW)

> [!Definition]
> Bag of Words (BoW) is a text representation method that describes the occurrence of words within a document, ignoring grammar and word order but considering multiplicity. BoW is used in natural language processing (NLP) for tasks like text classification, clustering, and information retrieval, where the presence or frequency of words matters more than their order.
## Python Code Example

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = ["This is a sample document.", "This document is another example.", "And this is a third document."]

# Create the Count vectorizer
vectorizer = CountVectorizer()

# Transform the documents into BoW vectors
bow_matrix = vectorizer.fit_transform(documents)

# Display the BoW matrix
print(bow_matrix.toarray())
print(vectorizer.get_feature_names_out())
```

# TF-IDF

> [!Definition]
> TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). TF-IDF is used in information retrieval and text mining to weigh the significance of words in documents, helping in tasks like search engine ranking, document clustering, and text classification.

1. **信息检索**：用来衡量文档中词的重要性，以便于搜索引擎在检索结果中对文档进行排序。
2. **文本挖掘**：在文本分类、聚类、主题建模等任务中，用作特征提取的一种方法。
3. **推荐系统**：帮助推荐系统理解文本内容，从而提高推荐的准确性。
## Formula
- **词频（TF，Term Frequency）**:
	一个词在文档中出现的次数，公式为
 $$
TF(t, d) = \frac{f(t, d)}{N} 
$$
其中，$f(t,d)$ 表示词 $t$ 在文档 $𝑑$ 中出现的次数，$N$ 是文档中的总词数。
- **逆文档频率（IDF，Inverse Document Frequency）**：
	逆文档频率是一个词在整个文档集合中的普遍重要性，公式为
$$
IDF(t, D) = \log \left( \frac{N}{1 + df(t, D)} \right)
$$
其中，$N$ 是文档集合中的文档总数，$𝑑𝑓(𝑡,𝐷)$ 是包含词 $t$ 的文档数。

将二者结合起来，得到TF-IDF值：
$$
TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)
$$
## Python Code Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents 
documents = ["This is a sample document.", "This document is another example.", "And this is a third document."]

# Create the TF-IDF vectorizer 
vectorizer = TfidfVectorizer() 

# Transform the documents into TF-IDF vectors 
tfidf_matrix = vectorizer.fit_transform(documents) 

# Display the TF-IDF matrix 
print(tfidf_matrix.toarray()) 
print(vectorizer.get_feature_names_out())
```

# Word Embeddings

> [!Definition]
Word Embeddings are dense vector representations of words that capture their meanings, semantic relationships, and context within a corpus. They map words to high-dimensional vectors of real numbers. Word embeddings are used in NLP tasks like machine translation, sentiment analysis, and information retrieval. They provide context-aware representations of words and capture semantic relationships between words, which helps in understanding the meaning and context of words more effectively than traditional methods like BoW or TF-IDF.
### Python Code Example

Using Gensim's Word2Vec:
```python
from gensim.models import Word2Vec

# Sample sentences
sentences = [["this", "is", "a", "sample", "document"],
             ["this", "document", "is", "another", "example"],
             ["and", "this", "is", "a", "third", "document"]]

# Train the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector for a specific word
vector = model.wv['document']
print(vector)
```

Using pre-trained Word Embeddings (e.g., GloVe)
```python
import gensim.downloader as api

# Load pre-trained GloVe model
model = api.load("glove-wiki-gigaword-100")

# Get the vector for a specific word
vector = model['document']
print(vector)
```

Visualizing Word Embeddings with PCA
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Words to visualize
words = ['this', 'is', 'a', 'sample', 'document', 'example', 'third']

# Get vectors for these words
word_vectors = [model[word] for word in words]

# Reduce dimensions using PCA
pca = PCA(n_components=2)
result = pca.fit_transform(word_vectors)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(result[:, 0], result[:, 1])

# Annotate points with words
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()
```

The output consists of:

- A vector representation for each word, capturing its semantic meaning and context.
- Visual representation using PCA to reduce dimensions for easier understanding of word relationships.

# BM25

> [!Definition]
> BM25 (Best Matching 25) is a probabilistic retrieval function used in information retrieval to rank documents based on their relevance to a given search query. Significance and Use BM25 is widely used in search engines and information retrieval systems due to its effectiveness in ranking documents by relevance. It considers term frequency, inverse document frequency, document length, and other parameters to provide a more refined scoring mechanism compared to simpler models like TF-IDF. 
> 
## Formula
BM25 uses a complex formula that includes term frequency, inverse document frequency, document length, and other parameters to score documents. The basic formula is: $$ \text{BM25}(q, d) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, d) \cdot (k_1 + 1)}{f(q_i, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})} $$ Where: 
- $f(q_i, d)$ is the frequency of term $q_i$ in document $d$ 
- $|d|$ is the length of document $d$ 
- $\text{avgdl}$ is the average document length in the corpus 
- $k_1$ and $b$ are free parameters, usually set to 1.5 and 0.75 respectively 
## Python Code Example 

Using the `rank_bm25` library: 
```python
import rank_bm25 
from rank_bm25 import BM25Okapi 

# Sample documents 
documents = ["This is a sample document.", "This document is another example.", "And this is a third document."] 

# Preprocess documents (tokenization) 
tokenized_corpus = [doc.split() for doc in documents] 

# Create the BM25 model 
bm25 = BM25Okapi(tokenized_corpus) 

# Query 
query = "sample document".split() 

# Get BM25 scores 
scores = bm25.get_scores(query) 
print(scores) 

# Rank documents based on the query 
top_n = bm25.get_top_n(query, documents, n=3) 
print(top_n)
```

# Topic Modeling

> [!Definition]
Topic Modeling is a type of statistical model used to discover abstract topics within a collection of documents, identifying patterns of word co-occurrence.

## Significance and Use
1. **文档分类**：用于将文档分类到不同的主题中。
2. **组织文档集合**：帮助理解和组织大规模的文档集合。
3. **揭示主题结构**：发现文档中的隐藏主题结构，有助于信息检索和文本分析。

## Common Techniques

### 1. Latent Dirichlet Allocation (LDA)

#### Formula
LDA is based on the idea that documents are mixtures of topics, and topics are mixtures of words. The generative process for LDA is:
1. For each document $d$ in the corpus $D$:
    - Choose a distribution over topics $\theta_d \sim \text{Dir}(\alpha)$
    - For each word $w_{dn}$ in document $d$:
        - Choose a topic $z_{dn} \sim \text{Multinomial}(\theta_d)$
        - Choose a word $w_{dn}$ from $p(w_{dn} | z_{dn}, \beta)$, a multinomial probability conditioned on the topic $z_{dn}$

#### Python Code Example
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = ["This is a sample document.", "This document is another example.", "And this is a third document."]

# Create the Count vectorizer
vectorizer = CountVectorizer()

# Transform the documents into BoW vectors
bow_matrix = vectorizer.fit_transform(documents)

# Create the LDA model
lda = LatentDirichletAllocation(n_components=2, random_state=42)

# Fit the LDA model
lda.fit(bow_matrix)

# Display the top words per topic
terms = vectorizer.get_feature_names_out()
for index, topic in enumerate(lda.components_):
    print(f"Topic {index}:")
    print([terms[i] for i in topic.argsort()[-10:]])
```

### 2. Non-negative Matrix Factorization (NMF)

#### Formula

NMF is a group of algorithms in multivariate analysis and linear algebra where a matrix $V$ is factorized into (usually) two matrices $W$ and $H$, with the property that all three matrices have no negative elements.

#### Python Code Example
```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = ["This is a sample document.", "This document is another example.", "And this is a third document."]

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the documents into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(documents)

# Create the NMF model
nmf = NMF(n_components=2, random_state=42)

# Fit the NMF model
nmf.fit(tfidf_matrix)

# Display the top words per topic
terms = vectorizer.get_feature_names_out()
for index, topic in enumerate(nmf.components_):
    print(f"Topic {index}:")
    print([terms[i] for i in topic.argsort()[-10:]])
```

## Output Explanation

1. **Latent Dirichlet Allocation (LDA)**:
    
    - Topics identified in the corpus, each represented by a list of the most relevant words.
    - Each topic's relevance is determined by the importance of the words in the corpus.
2. **Non-negative Matrix Factorization (NMF)**:
    
    - Similar to LDA, NMF also identifies topics in the corpus, but it uses a different mathematical approach.
    - Each topic is represented by a set of words with the highest weights in the factorized matrices.

Topic modeling provides a way to automatically discover the hidden thematic structure in a large corpus of text, making it useful for text analysis, information retrieval, and document classification.
