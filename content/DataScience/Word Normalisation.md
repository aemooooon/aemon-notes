---
title: 
draft: false
tags:
  - classification
  - topicmodel
---
> [!important]
> **Word normalisation** is a crucial preprocessing step in Natural Language Processing (NLP) that aims to reduce words to a standard form, thereby improving the performance and accuracy of various text analysis tasks. This note covers different techniques for word normalization, including their concepts, applications, advantages, and disadvantages.

# 1. Stemming

>[!Definition]
>**Stemming** is the process of reducing inflected or derived words to their base or root form, known as a "stem." This is usually done by removing suffixes. The resulting stem may not be a valid word. 

**Common Algorithms:**
- **Porter Stemmer:** Uses a set of rules to iteratively strip suffixes.
- **Snowball Stemmer:** An improvement over the Porter Stemmer with more rules.
- **Lancaster Stemmer:** An aggressive stemmer that often results in shorter stems.

## Applications

- **Information Retrieval:** Improves recall by matching similar terms.
- **Text Mining:** Reduces dimensionality of text data.
- **Sentiment Analysis:** Helps in normalizing words to capture sentiments.

## Code Example

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

porter = PorterStemmer()
snowball = SnowballStemmer(language='english')
lancaster = LancasterStemmer()

words = ['running', 'runner', 'ran', 'runs']
print([porter.stem(word) for word in words])
print([snowball.stem(word) for word in words])
print([lancaster.stem(word) for word in words])
```

## Advantages

- **Speed:** Computationally efficient.
- **Simplicity:** Easy to implement.

## Disadvantages

- **Accuracy:** Can produce non-words and inaccurate stems.
- **Lack of Context:** Does not consider the context or meaning of words.

# 2. Lemmatization

> [!Definition]
> **Lemmatization** reduces words to their base or dictionary form (lemma) by considering the context and morphological analysis. It ensures that the root word is a valid word.

**Common Algorithms:**
- **WordNet Lemmatizer:** Uses the WordNet lexical database.
- **SpaCy Lemmatizer:** A modern lemmatizer integrated into SpaCy.

## Applications

- **Text Classification:** Enhances feature consistency.
- **Machine Translation:** Improves translation accuracy.
- **Named Entity Recognition (NER):** Helps in identifying entities accurately.

## Code Example

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
words = ['running', 'better', 'happily']
print([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words])
print([lemmatizer.lemmatize(word, pos=wordnet.ADJ) for word in words])
```

## Advantages

- **Accuracy:** Produces valid words.
- **Context-Aware:** Considers the part of speech.

## Disadvantages

- **Complexity:** Requires more computational resources.
- **Dependency:** Relies on external lexical databases.

# 3. Text Normalization

> [!Definition]
> **Text normalization** involves standardizing text by converting it to a consistent format. This includes lowercasing, removing punctuation, and handling contractions.

## Applications

- **Preprocessing:** Essential for most NLP tasks.
- **Data Cleaning:** Improves data quality.
- **Chatbots:** Helps in understanding user input better.

## Code Example

```python
import re

def normalize(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

sentence = "Running, 123 running! Happily, running."
print(normalize(sentence))
```

## Advantages

- **Simplicity:** Easy to implement.
- **Effectiveness:** Improves data consistency.

## Disadvantages

- **Context Loss:** Removes potentially useful information.
- **Over-Simplification:** May not handle all edge cases.

# 4. Synonym Replacement

>[!Definition]
>**Synonym replacement** involves replacing words with their synonyms to standardize vocabulary and improve text analysis.

**Tools:**
- **NLTK WordNet:** For finding synonyms.
- **SpaCy:** Can also be used for synonym replacement with its lexical database.

## Applications

- **Text Augmentation:** Enhances dataset diversity.
- **Semantic Analysis:** Improves understanding of text.
- **Document Clustering:** Helps in grouping similar documents.

## Code Example

```python
from nltk.corpus import wordnet

def synonym_replacement(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()
    return word

sentence = "He is quickly running towards the finish line."
words = sentence.split()
replaced_sentence = ' '.join([synonym_replacement(word) for word in words])
print(replaced_sentence)
```

## Advantages

- **Semantic Preservation:** Maintains the meaning of the text.
- **Variability:** Introduces lexical diversity.

## Disadvantages

- **Complexity:** Requires a comprehensive thesaurus.
- **Context Sensitivity:** May not always find suitable replacements.

# 5. Phrase Detection


>[!Definition]
>**Phrase detection** involves identifying and merging common phrases in text to reduce the number of tokens and improve semantic representation.

**Tools:**
- **Gensim Phrases:** For detecting phrases in text.
- **NLTK Collocations:** For finding common word pairs.

## Applications

- **N-gram Models:** Enhances the quality of features.
- **Topic Modeling:** Improves topic coherence.
- **Machine Translation:** Helps in translating phrases accurately.

## Code Example

```python
import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser

sentences = [
    ['he', 'is', 'running', 'towards', 'the', 'finish', 'line'],
    ['this', 'is', 'a', 'test', 'sentence']
]

phrases = Phrases(sentences, min_count=1, threshold=1)
bigram = Phraser(phrases)
bigram_sentence = bigram[sentences[0]]
print(bigram_sentence)
```

## Advantages

- **Improved Context:** Captures multi-word expressions.
- **Enhanced Accuracy:** Reduces ambiguity.

## Disadvantages

- **Computational Cost:** Requires additional processing.
- **Data Dependency:** Dependent on the quality of input data.

# 6. Abbreviation Expansion

>[!Definition]
>**Abbreviation expansion** involves converting abbreviations into their full form to improve understanding and analysis.

**Tools:**
- **Custom Dictionaries:** For specific domain abbreviations.
- **Named Entity Recognition (NER) Systems:** For identifying abbreviations and expanding them.

## Applications

- **Medical Text Analysis:** Expanding medical abbreviations for better analysis.
- **Legal Document Processing:** Expanding legal abbreviations.

## Code Example

```python
abbreviations = {
    "US": "United States",
    "AI": "artificial intelligence",
    "ML": "machine learning"
}

def expand_abbreviations(text):
    words = text.split()
    expanded_words = [abbreviations.get(word, word) for word in words]
    return ' '.join(expanded_words)

sentence = "US is a leader in AI and ML."
print(expand_abbreviations(sentence))
```

## Advantages

- **Clarity:** Improves readability and understanding.
- **Context-Awareness:** Helps in better analysis of text.

## Disadvantages

- **Maintenance:** Requires up-to-date abbreviation dictionaries.
- **Context Sensitivity:** Some abbreviations might have multiple expansions.
