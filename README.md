# About SAQE (Sense-Aware Query Expansion)
An out-of-the-box, corpus-agnostic query expansion tool for lexical retrieval systems.

This tool uses `WordNet` as its knowledge base. For word-sense disambiguation, it leverages `SimCSE` to encode query 
text and sense-specific WordNet textual content into embeddings, and compute semantic similarity between them. 
Furthermore, it leverages `NLTK`, `spaCy`, and `TextBlob` to optimize query term tokenization (the least number of 
non-overlapping, WordNet-meaningful terms).

# Getting Started
In a Python 3.8 virtual environment, install the `saqe` package and download its required artifacts.
```shell
python setup.py install
python3 -m textblob.download_corpora
python3 -m spacy download en_core_web_lg
```

# Code Examples
## Use Case #1: Expand original query with synonyms
```python
import json
from saqe import SAQE


original_query = "The foreign policy of the United States in West Africa."
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
"The foreign policy of the United States in West Africa."
```

```text
ORIGINAL QUERY EXPANDED WITH SYNONYMS
The foreign policy of the United States in West Africa. America U.S. U.S.A. US USA United States of America the States
```

```text
SYNONYMS ORGANIZED BY QUERY TERMS
{
    "United States": {
        "synonyms": [
            "America",
            "U.S.",
            "U.S.A.",
            "US",
            "USA",
            "United States of America",
            "the States"
        ]
    }
}
```

## Use Case #2: Expand original query with synonyms and noun phrases from query term definitions
```python
import json
from saqe import SAQE


original_query = "The foreign policy of the United States in West Africa."
query_expander = SAQE(enable_noun_phrases_from_definition=True)

print('ORIGINAL QUERY:')
print(original_query)

expansion_terms = query_expander.expand(original_query)
print('ORIGINAL QUERY EXPANDED WITH SYNONYMS AND NOUN PHRASES FROM TERM DEFINITIONS')
print(f"{original_query} {expansion_terms['as_a_string']}")
print('SYNONYMS AND NOUN PHRASES FROM TERM DEFINITIONS ORGANIZED BY QUERY TERMS')
print(json.dumps(expansion_terms['by_term'], indent=4))

```

## Output
```text
ORIGINAL QUERY:
"The foreign policy of the United States in West Africa."
```

```text
ORIGINAL QUERY EXPANDED WITH SYNONYMS AND NOUN PHRASES FROM TERM DEFINITIONS
The foreign policy of the United States in West Africa. international relations policy America U.S. U.S.A. US USA United States of America the States 48 conterminous states 50 states Alaska Hawaiian Islands North America North American republic Pacific Ocean alaska america conterminous states independence north american republic northwest North America pacific ocean Guinea Gulf Sahara Desert africa area guinea sahara western Africa
```

```text
SYNONYMS AND NOUN PHRASES FROM TERM DEFINITIONS ORGANIZED BY QUERY TERMS
{
    "foreign policy": {
        "noun_phrases_from_definition": [
            "international relations",
            "policy"
        ]
    },
    "United States": {
        "synonyms": [
            "America",
            "U.S.",
            "U.S.A.",
            "US",
            "USA",
            "United States of America",
            "the States"
        ],
        "noun_phrases_from_definition": [
            "48 conterminous states",
            "50 states",
            "Alaska",
            "Hawaiian Islands",
            "North America",
            "North American republic",
            "Pacific Ocean",
            "alaska",
            "america",
            "conterminous states",
            "independence",
            "north american republic",
            "northwest North America",
            "pacific ocean"
        ]
    },
    "West Africa": {
        "noun_phrases_from_definition": [
            "Guinea",
            "Gulf",
            "Sahara Desert",
            "africa",
            "area",
            "guinea",
            "sahara",
            "western Africa"
        ]
    }
}
```

## Use Case #3: Expand original query with synonyms, noun phrases from query term definitions, and hyponyms
```python
import json
from saqe import SAQE


original_query = "The foreign policy of the United States in West Africa."
query_expander = SAQE(enable_noun_phrases_from_definition=True, enable_hyponyms=True)

print('ORIGINAL QUERY:')
print(original_query)

expansion_terms = query_expander.expand(original_query)
print('ORIGINAL QUERY EXPANDED WITH SYNONYMS, NOUN PHRASES FROM TERM DEFINITIONS, AND HYPONYMS')
print(f"{original_query} {expansion_terms['as_a_string']}")
print('SYNONYMS, NOUN PHRASES FROM TERM DEFINITIONS, AND HYPONYMS ORGANIZED BY QUERY TERMS')
print(json.dumps(expansion_terms['by_term'], indent=4))
```

# Output
```text
ORIGINAL QUERY:
"The foreign policy of the United States in West Africa."
```

```text
ORIGINAL QUERY EXPANDED WITH SYNONYMS, NOUN PHRASES FROM TERM DEFINITIONS, AND HYPONYMS
The foreign policy of the United States in West Africa. international relations policy brinkmanship imperialism intervention isolationism monroe doctrine neutralism nonaggression nonintervention regionalism trade policy truman doctrine America U.S. U.S.A. US USA United States of America the States 48 conterminous states 50 states Alaska Hawaiian Islands North America North American republic Pacific Ocean alaska america conterminous states independence north american republic northwest North America pacific ocean Guinea Gulf Sahara Desert africa area guinea sahara western Africa
```

```text
SYNONYMS, NOUN PHRASES FROM TERM DEFINITIONS, AND HYPONYMS ORGANIZED BY QUERY TERMS
{
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
    },
    "United States": {
        "synonyms": [
            "America",
            "U.S.",
            "U.S.A.",
            "US",
            "USA",
            "United States of America",
            "the States"
        ],
        "noun_phrases_from_definition": [
            "48 conterminous states",
            "50 states",
            "Alaska",
            "Hawaiian Islands",
            "North America",
            "North American republic",
            "Pacific Ocean",
            "alaska",
            "america",
            "conterminous states",
            "independence",
            "north american republic",
            "northwest North America",
            "pacific ocean"
        ]
    },
    "West Africa": {
        "noun_phrases_from_definition": [
            "Guinea",
            "Gulf",
            "Sahara Desert",
            "africa",
            "area",
            "guinea",
            "sahara",
            "western Africa"
        ]
    }
}
```
