# Table of Contents
 1. [About SAQE (Sense-Aware Query Expansion)](#about-saqe-sense-aware-query-expansion)
 2. [Getting Started](#getting-started)
 3. [Quick Example](#quick-example)

# About SAQE (Sense-Aware Query Expansion)
`saqe` is an off-the-shelf, corpus-agnostic [query expansion](https://en.wikipedia.org/wiki/Query_expansion) tool 
for information retrieval systems. It uses [WordNet](https://wordnet.princeton.edu/) as its knowledge base. For 
word-sense disambiguation, it computes semantic similarity between query embeddings and Wordnet term embeddings. 
The embeddings are produced using a user-inputted language model (or `sentence-transformers/all-MiniLM-L12-v2` 
by default). Finally, it leverages `NLTK`, `spaCy`, and `TextBlob` to optimize query term tokenization (the 
least number of non-overlapping, WordNet-meaningful terms).

# Getting Started
In a Python 3.10 virtual environment, install the `saqe` package and download its required artifacts.
```shell
pip install ./
python -m textblob.download_corpora
python -m spacy download en_core_web_lg
```

# Quick Example
For a demonstration, let's consider the queries `The foreign policy of the United States.` and `The geography of the 
United States.`. The phrase `United States` occurs in both. 

```python
import json

from saqe.expanders import SAQE


query_expander = SAQE(
    text_encoder_name="sentence-transformers/all-MiniLM-L12-v2",
    enable_hyponyms=True,
    enable_noun_phrases_from_definition=True
)

# QUERY #1
expansion_terms = query_expander.expand("The foreign policy of the United States.")
print(json.dumps(expansion_terms, indent=4))

# QUERY #2
expansion_terms = query_expander.expand("The geography of the United States.")
print(json.dumps(expansion_terms, indent=4))
```

Note in the outputs below that `saqe` manages to expand on each occurrence of `United States` in the sense that best 
fits the context of its parent query, in the sense of `the government of the United States` and in the sense of 
`United States, the place` respectively.

#### QUERY #1: `The foreign policy of the United States.`
```json
{
    "United States": {
        "synonyms": [
            "U.S.",
            "U.S._government",
            "US_Government",
            "United_States_government"
        ],
        "hyponyms": [],
        "noun_phrases_from_definition": [
            "United States",
            "branches",
            "executive",
            "federal government",
            "judicial branches"
        ]
    },
    "foreign policy": {
        "synonyms": [],
        "hyponyms": [
            "brinkmanship",
            "imperialism",
            "intervention",
            "isolationism",
            "manifest destiny",
            "monroe doctrine",
            "neutralism",
            "nonaggression",
            "nonintervention",
            "open-door policy",
            "regionalism",
            "trade policy",
            "truman doctrine"
        ],
        "noun_phrases_from_definition": [
            "international relations",
            "policy"
        ]
    }
}
```
#### QUERY #2: `The geography of the United States.`
```json
{
    "geography": {
        "synonyms": [
            "geographics"
        ],
        "hyponyms": [
            "economic geography",
            "physical geography",
            "topography",
            "topology"
        ],
        "noun_phrases_from_definition": [
            "climate",
            "earth's surface",
            "people's responses",
            "soil",
            "study",
            "topography",
            "vegetation"
        ]
    },
    "United States": {
        "synonyms": [
            "America",
            "U.S.",
            "U.S.A.",
            "US",
            "USA",
            "United_States_of_America",
            "the_States"
        ],
        "hyponyms": [],
        "noun_phrases_from_definition": [
            "48 conterminous states",
            "50 states",
            "Alaska",
            "Hawaiian Islands",
            "North America",
            "North American republic",
            "Pacific Ocean",
            "conterminous states",
            "independence",
            "northwest North America"
        ]
    }
}
```
