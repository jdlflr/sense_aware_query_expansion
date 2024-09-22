from typing import List

from nltk.corpus import stopwords
import spacy
from textblob import TextBlob


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
        stripped_noun_phrases = list()
        for noun_phrase in raw_noun_phrases:
            stripped_noun_phrase = self._strip_off_leading_stop_words(noun_phrase)
            if stripped_noun_phrase:
                stripped_noun_phrases.append(stripped_noun_phrase)
        noun_phrases = list(set(stripped_noun_phrases))
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
        try:
            while split_noun_phrase[0].lower() in stopwords.words():
                split_noun_phrase = split_noun_phrase[1:]
        except IndexError:
            split_noun_phrase = list()
        return " ".join(split_noun_phrase).strip()
