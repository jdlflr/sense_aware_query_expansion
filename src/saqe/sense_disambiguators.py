from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from numpy import argmax
from sklearn.metrics.pairwise import cosine_similarity
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SenseDisambiguator(object):
    """
    A class for computing semantic similarity between a piece of text (query) and a list of texts (corpus).
    """
    def __init__(self, text_encoder_name: str):
        self._model = HuggingFaceEmbeddings(model_name=text_encoder_name, model_kwargs={'device': DEVICE})

    def similarity(self, anchor_text: str, senses: List[str]) -> int:
        """
        Computes semantic similarity between a piece of text (query) and a list of texts (corpus).
        Args:
            anchor_text: [REQUIRED] the text to compare against all texts in the corpus.
            senses: [REQUIRED] the texts against which the query text will be compared.

        Returns:
            The index of the entry in 'senses' most similar to 'anchor_text'.
        """
        corpus_embeddings = self._model.embed_documents(senses)
        query_embedding = [self._model.embed_query(anchor_text)]
        similarity_scores = cosine_similarity(query_embedding, corpus_embeddings)
        return argmax(similarity_scores)
