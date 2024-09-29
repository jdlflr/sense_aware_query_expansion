from collections import defaultdict
from copy import deepcopy
import re
from typing import List, Dict, Tuple
import json

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

from .parsers import NounPhraseExtractor
from .sense_disambiguators import SenseDisambiguator


nltk.download('punkt_tab')


class SAQE(object):
    """
    A class for expanding keyword queries by leveraging semantic senses and the synonymic relations between words
    Args:
        text_encoder_name: [OPTIONAL] Path to the pre-trained language model to use for encoding.
        enable_noun_phrases_from_definition: [OPTIONAL] Whether to include noun phrases extracted from the definitions
        of query terms as expansion terms.
        enable_hyponyms: [OPTIONAL] Whether to include hyponyms as expansion terms.
    """
    def __init__(
            self,
            text_encoder_name: str = "sentence-transformers/all-MiniLM-L12-v2",
            enable_hyponyms: bool = False,
            enable_noun_phrases_from_definition: bool = False
    ):
        self._sense_disambiguator = SenseDisambiguator(text_encoder_name=text_encoder_name)
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
        return expansion

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
        senses = self._senses(keyword=keyword)
        if senses:
            selected_sense_index = 0
            if senses and len(senses) > 1:
                sense_representations = [
                    self._sense_representation(i) if self._sense_representation(i) else "-" for i in senses
                ]
                selected_sense_index = self._sense_disambiguator.similarity(anchor_text=query, senses=sense_representations)
            expansion_terms["synonyms"] = senses[selected_sense_index]["synonyms"]
            if self._enable_hyponyms:
                expansion_terms["hyponyms"] = senses[selected_sense_index]["hyponyms"]
            if self._enable_noun_phrases_from_definition:
                expansion_terms["noun_phrases_from_definition"] = \
                    senses[selected_sense_index]["noun_phrases_from_definition"]
        return expansion_terms

    def _senses(self, keyword: str) -> List[Dict[str, List[str]]]:
        """
        For each available sense of the keyword, collects synonyms, hyponyms, and their corresponding definitions from
        WordNet.
        Args:
            keyword: The term to collect potential expansion terms for.

        Returns:
            List of dictionaries, each corresponding to a sense of the inputted keyword in the format:
            {
                "direct_definition": [<definition of the sense>],
                "synonyms": [synonym_1, synonym_2, ...],
                "hyponyms": [hyponym_1, hyponym_2, ...],
                "definitions": [hyponym_definition_1, hyponym_definition_2, ...],
                "noun_phrases_from_definition": [noun_phrase_1, noun_phrase_2, ...]
            }
        """
        senses = list()
        for sense_index, sense in enumerate(wordnet.synsets(keyword)):
            sense_dict = defaultdict(list)
            sense_dict["direct_definition"].append(sense.definition().strip())
            sense_dict["synonyms"] += wordnet.synonyms(keyword)[sense_index]
            sense_dict["unprocessed"] += list(set([i.name() for i in sense.hyponyms()]))
            sense_dict["hyponyms"] += list(set([self._word_from_sysnet_name(i.name()) for i in sense.hyponyms()]))
            while sense_dict["unprocessed"]:
                unprocessed = deepcopy(sense_dict["unprocessed"])
                for item in unprocessed:
                    sense_dict["unprocessed"] = list(
                        set(
                            sense_dict["unprocessed"] + [i.name() for i in wordnet.synset(item).hyponyms()]
                        ).difference(set(sense_dict["processed"]))
                    )
                    sense_dict["unprocessed"].remove(item)
                    sense_dict["processed"].append(item)
                    sense_dict["hyponyms"] = list(
                        set(
                            sense_dict["hyponyms"] +
                            [self._word_from_sysnet_name(i.name()) for i in wordnet.synset(item).hyponyms()]
                        )
                    )
            del sense_dict["unprocessed"]
            for key in sense_dict:
                sense_dict[key].sort()
            sense_dict["definitions"] += list(
                set(
                    [
                        wordnet.synset(i).definition().strip() for i in sense_dict["processed"] if
                        wordnet.synset(i).definition().strip()
                    ]
                )
            )
            del sense_dict["processed"]
            if sense_dict["direct_definition"]:
                sense_dict["noun_phrases_from_definition"] += self._noun_phrase_extractor(
                    sense_dict["direct_definition"][0]
                )
            senses.append(sense_dict)
        return senses

    @staticmethod
    def _word_from_sysnet_name(name: str) -> str:
        """
        Extracts the word from the WordNet name.
        Args:
            name: the WordNet name.

        Returns:
            The term corresponding to the inputted WordNet name.
        """
        return name.split(".")[0].replace("_", " ")

    @staticmethod
    def _sense_representation(syn_hyp_def: Dict[str, List[str]]) -> str:
        """
        Concatenates the definition of the sense with the synonyms, hyponyms, and their corresponding definitions.
        Args:
            syn_hyp_def: dictionary structure as an element in the list outputted by self._senses

        Returns:
            String consisting of the definition of the sense, the synonyms, the hyponyms, and their corresponding
            definitions.
        """
        output = " ".join(syn_hyp_def["direct_definition"]).strip()
        output += f' {" ".join(syn_hyp_def["synonyms"] + syn_hyp_def["hyponyms"])}'
        output = output.strip()
        output += f' {" ".join(syn_hyp_def["definitions"])}'
        return output.strip()

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
        wordnet_formatted_validated_query_keywords = [re.sub("[- ]", "_", i) for i in validated_query_keywords]
        wordnet_formatted_tokenized_query_phrases = [re.sub("[- ]", "_", i) for i in tokenized_query_phrases]

        return validated_query_keywords + tokenized_query_phrases, \
            wordnet_formatted_validated_query_keywords + wordnet_formatted_tokenized_query_phrases

    @staticmethod
    def _is_in_wordnet(term: str):
        """
        Determines whether the term exists in WordNet.
        """
        return len(wordnet.synsets(re.sub("[- ]", "_", term))) > 0
