# Table of Contents
 1. [About SAQE (Sense-Aware Query Expansion)](#about-saqe-sense-aware-query-expansion)
 2. [Getting Started](#getting-started)
 3. [Code Examples](#code-examples)
    - [Use Case #1: Expand original query with synonyms](#use-case-1-expand-original-query-with-synonyms)
    - [Use Case #2: Expand original query with synonyms and Hyponyms](#use-case-2-expand-original-query-with-synonyms-and-hyponyms)
    - [Use Case #3: Expand original query with synonyms, hyponyms, and noun phrases from query term definitions](#use-case-3-expand-original-query-with-synonyms-hyponyms-and-noun-phrases-from-query-term-definitions)

# About SAQE (Sense-Aware Query Expansion)
`saqe` is an out-of-the-box, corpus-agnostic query expansion tool for lexical retrieval systems. It uses 
[WordNet](https://wordnet.princeton.edu/) as its knowledge base. For word-sense disambiguation, it uses a user-inputted 
language model or [SimCSE](https://github.com/princeton-nlp/SimCSE) by default to encode query text and 
sense-specific WordNet textual content into embeddings, and compute the semantic similarity between them. Furthermore, 
it leverages `NLTK`, `spaCy`, and `TextBlob` to optimize query term tokenization (the least number of 
non-overlapping, WordNet-meaningful terms).

# Getting Started
In a Python 3.8 virtual environment, install the `saqe` package and download its required artifacts.
```shell
python setup.py install
python -m textblob.download_corpora
python -m spacy download en_core_web_lg
```

# Code Examples
## Use Case #1: Expand original query with synonyms
```python
import json
from saqe import SAQE


original_query = "the foreign policy of the United States"
query_expander = SAQE()

print('ORIGINAL QUERY:')
print(original_query)

expansion_terms = query_expander.expand(original_query)
print('ORIGINAL QUERY EXPANDED WITH SYNONYMS')
print(f"{original_query} {expansion_terms['as_a_string']}")
print('SYNONYMS ORGANIZED BY QUERY TERMS')
print(json.dumps(expansion_terms['by_term'], indent=4))
```

## Output
```text
ORIGINAL QUERY:
the foreign policy of the United States
```

```text
ORIGINAL QUERY EXPANDED WITH SYNONYMS
the foreign policy of the United States U.S. U.S. government US Government United States government
```

```text
SYNONYMS ORGANIZED BY QUERY TERMS
{
    "United States": {
        "synonyms": [
            "U.S.",
            "U.S. government",
            "US Government",
            "United States government"
        ]
    }
}
```

## Use Case #2: Expand original query with synonyms and Hyponyms
```python
import json
from saqe import SAQE


original_query = "the foreign policy of the United States"
query_expander = SAQE(enable_hyponyms=True)

print('ORIGINAL QUERY:')
print(original_query)

expansion_terms = query_expander.expand(original_query)
print('ORIGINAL QUERY EXPANDED WITH SYNONYMS AND HYPONYMS')
print(f"{original_query} {expansion_terms['as_a_string']}")
print('SYNONYMS AND HYPONYMS ORGANIZED BY QUERY TERMS')
print(json.dumps(expansion_terms['by_term'], indent=4))
```

## Output
```text
ORIGINAL QUERY:
the foreign policy of the United States
```

```text
ORIGINAL QUERY EXPANDED WITH SYNONYMS AND HYPONYMS
the foreign policy of the United States U.S. U.S. government US Government United States government brinkmanship imperialism intervention isolationism monroe doctrine neutralism nonaggression nonintervention regionalism trade policy truman doctrine
```

```text
SYNONYMS AND HYPONYMS ORGANIZED BY QUERY TERMS
{
    "United States": {
        "synonyms": [
            "U.S.",
            "U.S. government",
            "US Government",
            "United States government"
        ]
    },
    "foreign policy": {
        "hyponyms": [
            "brinkmanship",
            "imperialism",
            "intervention",
            "isolationism",
            "monroe doctrine",
            "neutralism",
            "nonaggression",
            "nonintervention",
            "regionalism",
            "trade policy",
            "truman doctrine"
        ]
    }
}
```

## Use Case #3: Expand original query with synonyms, hyponyms, and noun phrases from query term definitions
```python
import json
from saqe import SAQE


original_query = "the foreign policy of the United States"
query_expander = SAQE(enable_hyponyms=True, enable_noun_phrases_from_definition=True)

print('ORIGINAL QUERY:')
print(original_query)

expansion_terms = query_expander.expand(original_query)
print('ORIGINAL QUERY EXPANDED WITH SYNONYMS, HYPONYMS, AND NOUN PHRASES FROM TERM DEFINITIONS')
print(f"{original_query} {expansion_terms['as_a_string']}")
print('SYNONYMS, HYPONYMS, AND NOUN PHRASES FROM TERM DEFINITIONS ORGANIZED BY QUERY TERMS')
print(json.dumps(expansion_terms['by_term'], indent=4))
```

# Output
```text
ORIGINAL QUERY:
the foreign policy of the United States
```

```text
ORIGINAL QUERY EXPANDED WITH SYNONYMS, HYPONYMS, AND NOUN PHRASES FROM TERM DEFINITIONS
the foreign policy of the United States U.S. U.S. government US Government United States United States government branches brinkmanship executive federal government imperialism international relations intervention isolationism judicial branches monroe doctrine neutralism nonaggression nonintervention policy regionalism trade policy truman doctrine
```

```text
SYNONYMS, HYPONYMS, AND NOUN PHRASES FROM TERM DEFINITIONS ORGANIZED BY QUERY TERMS
{
    "United States": {
        "synonyms": [
            "U.S.",
            "U.S. government",
            "US Government",
            "United States government"
        ],
        "noun_phrases_from_definition": [
            "United States",
            "branches",
            "executive",
            "federal government",
            "judicial branches"
        ]
    },
    "foreign policy": {
        "noun_phrases_from_definition": [
            "international relations",
            "policy"
        ],
        "hyponyms": [
            "brinkmanship",
            "imperialism",
            "intervention",
            "isolationism",
            "monroe doctrine",
            "neutralism",
            "nonaggression",
            "nonintervention",
            "regionalism",
            "trade policy",
            "truman doctrine"
        ]
    }
}
```
