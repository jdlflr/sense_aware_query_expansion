import unittest

from src.saqe.expanders import SAQE
from src.saqe.parsers import NounPhraseExtractor
from src.saqe.sense_disambiguators import SenseDisambiguator


class SAQETestCase(unittest.TestCase):
    def test_expand(self):
        input_to_expected_output = \
            {
                "The foreign policy of the United States.": {
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
                },
                "The geography of the United States.": {
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
            }
        s = SAQE(enable_hyponyms=True, enable_noun_phrases_from_definition=True)
        for i, o in input_to_expected_output.items():
            self.assertEqual(o, s.expand(i))


class SenseDisambiguatorTestCase(unittest.TestCase):
    def test_similarity(self):
        inputs_and_expected_outputs = [({"anchor_text": "positive", "senses": ["subpar", "superb"]}, 1)]
        ts = SenseDisambiguator(text_encoder_name="sentence-transformers/all-MiniLM-L12-v2")
        for (i, o) in inputs_and_expected_outputs:
            self.assertEqual(o, ts.similarity(**i))


class NounPhraseExtractorTestCase(unittest.TestCase):
    def test(self):
        input_to_expected_output = {"The United States of America": ['America', 'United States']}
        npe = NounPhraseExtractor()
        for i, o in input_to_expected_output.items():
            self.assertEqual(o, npe(i))
