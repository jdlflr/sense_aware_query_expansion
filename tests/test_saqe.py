import unittest
import numpy

from src.saqe import *


class SAQETestCase(unittest.TestCase):
    def test_expand(self):
        input_to_expected_output = {
            "the foreign policy of the United States": {
                "as_a_string": 'U.S. U.S. government US Government United States United States government branches '
                               'brinkmanship executive federal government imperialism international relations '
                               'intervention isolationism judicial branches monroe doctrine neutralism nonaggression '
                               'nonintervention policy regionalism trade policy truman doctrine',
                "by_term": {
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
                    }
                }
            }
        }
        s = SAQE(enable_hyponyms=True, enable_noun_phrases_from_definition=True)
        for i, o in input_to_expected_output.items():
            self.assertEqual(o, s.expand(i))


class TextSimilarityTestCase(unittest.TestCase):
    def test_similarity(self):
        inputs_and_expected_outputs = [({"query": "positive", "corpus": ["subpar", "superb"]}, 1)]
        ts = TextSimilarity(model_name="prajjwal1/bert-tiny")
        for (i, o) in inputs_and_expected_outputs:
            self.assertEqual(o, numpy.argmax(ts.similarity(**i)))


class NounPhraseExtractorTestCase(unittest.TestCase):
    def test(self):
        input_to_expected_output = {"The United States of America": ['America', 'United States']}
        npe = NounPhraseExtractor()
        for i, o in input_to_expected_output.items():
            self.assertEqual(o, npe(i))
