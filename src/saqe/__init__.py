import re
from typing import List, Dict, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from numpy import argmax
from sklearn.metrics.pairwise import cosine_similarity
from simcse import SimCSE
import spacy
from textblob import TextBlob
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NounPhraseExtractor(object):
    """
    A class for extracting noun phrases from text.
    """
    def __init__(self):
        self._spacy_parser = spacy.load("en_core_web_lg")

    def __call__(self, text: str) -> List[str]:
        """
        Extracts noun phrases from text.
        Args:
            text: [REQUIRED] The text from which to extract noun phrases.

        Returns:
            A list of noun phrases extracted from the input text.
        """
        raw_noun_phrases = TextBlob(text).noun_phrases + [i.text for i in self._spacy_parser(text).noun_chunks]
        # filter out uncased phrases
        raw_noun_phrases = [i for i in raw_noun_phrases if i in text]
        # strip off leading stopwords and deduplicate the list of phrases
        noun_phrases = list(set([self._strip_off_leading_stop_words(noun_phrase) for noun_phrase in raw_noun_phrases]))
        # sort phrases
        noun_phrases.sort()

        return noun_phrases

    @staticmethod
    def _strip_off_leading_stop_words(noun_phrase: str) -> str:
        """
        Strips off leading stopwords from a noun phrase.
        Args:
            noun_phrase: [REQUIRED] the noun phrase to process

        Returns:
            The noun phrase free of any leading stopwords.
        """
        split_noun_phrase = noun_phrase.split()
        while split_noun_phrase[0].lower() in stopwords.words():
            split_noun_phrase = split_noun_phrase[1:]
        return " ".join(split_noun_phrase)


class TextSimilarity(object):
    """
    A class for computing semantic similarity between a piece of text (query) and a list of texts (corpus).
    """
    def __init__(self, model_name: str):
        self._model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': DEVICE})

    def similarity(self, query: str, corpus: List[str]) -> List[float]:
        """
        Computes semantic similarity between a piece of text (query) and a list of texts (corpus).
        Args:
            query: [REQUIRED] the text to compare against all texts in the corpus.
            corpus: [REQUIRED] the texts against which the query text will be compared.

        Returns:
            List of computed cosine similarity values. They are index matched with the corpus texts.
        """
        corpus_embeddings = self._model.embed_documents(corpus)
        query_embedding = [self._model.embed_query(query)]
        return cosine_similarity(query_embedding, corpus_embeddings)


class SAQE(object):
    """
    A class for expanding keyword queries by leveraging semantic senses and the synonymic relations between words
    Args:
        text_encoder: [OPTIONAL] The pre-trained SimCSE sentence encoding model.
        enable_noun_phrases_from_definition: [OPTIONAL] Whether to include noun phrases extracted from the definitions
        of query terms as expansion terms.
        enable_hyponyms: [OPTIONAL] Whether to include hyponyms as expansion terms.
    """
    def __init__(
            self,
            text_encoder: str = None,
            enable_hyponyms: bool = False,
            enable_noun_phrases_from_definition: bool = False
    ):
        self._sense_matcher = SimCSE("princeton-nlp/sup-simcse-roberta-large", device=DEVICE) if text_encoder is None \
            else TextSimilarity(model_name=text_encoder)
        self._lemmatizer = WordNetLemmatizer()
        self._noun_phrase_extractor = NounPhraseExtractor()

        self._enable_noun_phrases_from_definition = enable_noun_phrases_from_definition
        self._enable_hyponyms = enable_hyponyms

    def expand(self, query: str) -> Dict:
        """
        Carries out sense-aware query expansion.
        Args:
            query: [REQUIRED] The keyword query to expand.

        Returns:
            Text sequence containing the original query concatenated with the expansion keywords.
        """
        # tokenization of query keywords and phrases
        (tokenized_query_terms, wordnet_formatted_tokenized_query_terms) = self._tokenize_terms(query=query)

        expansion = dict()
        for keyword_index, query_term in enumerate(wordnet_formatted_tokenized_query_terms):
            # filter out stop words
            if query_term in stopwords.words():
                continue

            # apply WordNet lemmatization
            query_term = self._lemmatizer.lemmatize(query_term)

            # find the best set of synonyms via word-sense disambiguation
            if re.search('\w', query_term):
                expansion_terms_for_keyword = self._expand_by_sense(keyword=query_term, query=query)
                if expansion_terms_for_keyword:
                    expansion[tokenized_query_terms[keyword_index]] = expansion_terms_for_keyword

        return {
            "as_a_string": self._as_a_string(expansion_dict=expansion),
            "by_term": expansion
        }

    def _expand_by_sense(self, keyword: str, query: str) -> Dict[str, List[str]]:
        """
        Applies word-sense disambiguation to the keyword in the context of the query, and expand the keyword based on
        that sense.
        Args:
            keyword: [REQUIRED] The keyword to disambiguate and expand.
            query: [REQUIRED] The query in the context of which to apply word-sense disambiguation.

        Returns:
            Dictionary mapping lists of expansion terms to their expansion type (synonyms, noun_phrases_from_definition,
            or hyponyms)
        """
        expansion_terms = dict()
        sense_representations = list()
        validated_synonyms_by_sense = list()

        definitions = list()
        hyponyms_by_sense = list()
        for sense in wordnet.synsets(keyword):
            definitions.append(sense.definition())
            hyponyms_by_sense.append([re.sub("_", " ", i.name().split(".")[0]) for i in sense.hyponyms()])
        raw_synonyms_by_sense = wordnet.synonyms(keyword)
        for ndx, definition in enumerate(definitions):
            synonyms = [re.sub("_", " ", s) for s in raw_synonyms_by_sense[ndx] if s != ""]
            sense_representations.append(" ".join([definition, " ".join(synonyms)]).strip())
            if not sense_representations[-1]:
                continue
            validated_synonyms_by_sense.append(dict())
            if len(synonyms) > 0:
                validated_synonyms_by_sense[-1]["synonyms"] = synonyms

            if self._enable_noun_phrases_from_definition:
                if definition:
                    noun_phrases_from_definition = \
                        [i.strip() for i in list(set(self._noun_phrase_extractor(definition))) if i.strip() != ""]
                    if noun_phrases_from_definition:
                        validated_synonyms_by_sense[-1]["noun_phrases_from_definition"] = noun_phrases_from_definition

            if self._enable_hyponyms:
                hyponyms = hyponyms_by_sense[ndx]
                if len(hyponyms) > 0:
                    validated_synonyms_by_sense[-1]["hyponyms"] = hyponyms

            if not validated_synonyms_by_sense[-1]:
                del sense_representations[-1]

        # retain the list of expansion terms corresponding to the sense that is most semantically similar to the
        # keyword in the context of the query
        if len(sense_representations) > 0:
            expansion_terms = validated_synonyms_by_sense[
                argmax(self._sense_matcher.similarity(query, sense_representations)) if len(sense_representations) > 1
                else 0
            ]
            [expansion_terms[k].sort() for k in expansion_terms]
        return expansion_terms

    def _tokenize_terms(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Tokenizes the query text to a list of non-overlapping words and noun phrases.
        Args:
            query: [REQUIRED] The query text to tokenize.

        Returns:
            A pair of lists of token query terms. The terms in the second list are WordNet-formatted.
        """
        # indiscriminate tokenization of query keywords and noun phrases
        tokenized_query_keywords = wordpunct_tokenize(query)
        tokenized_query_phrases = [i for i in self._noun_phrase_extractor(text=query) if re.search('[- ]', i)]

        # only retain non overlapping single-word terms
        validated_query_keywords = list()
        for keyword in tokenized_query_keywords:
            is_part_of_a_phrase = False
            for phrase in tokenized_query_phrases:
                if self._is_in_wordnet(phrase):
                    if keyword in wordpunct_tokenize(phrase) or keyword in phrase.split():
                        is_part_of_a_phrase = True
                        break
            if not is_part_of_a_phrase:
                validated_query_keywords.append(keyword)

        # apply WordNet formatting
        wordnet_formatted_validated_query_keywords = [re.sub("[\- ]", "_", i) for i in validated_query_keywords]
        wordnet_formatted_tokenized_query_phrases = [re.sub("[\- ]", "_", i) for i in tokenized_query_phrases]

        return validated_query_keywords + tokenized_query_phrases, \
            wordnet_formatted_validated_query_keywords + wordnet_formatted_tokenized_query_phrases

    @staticmethod
    def _is_in_wordnet(term: str):
        """
        Determines whether the term exists in WordNet.
        """
        return len(wordnet.synsets(re.sub("[- ]", "_", term))) > 0

    @staticmethod
    def _as_a_string(expansion_dict: Dict[str, Dict[str, List[str]]]) -> str:
        """
        Formats the expansion terms as s string sequence.
        """
        expansion_terms = list()
        for term in expansion_dict:
            expansion_term_lists = list(expansion_dict[term].values())
            for expansion_term_list in expansion_term_lists:
                expansion_terms += expansion_term_list
        expansion_terms.sort()
        return " ".join(expansion_terms)
