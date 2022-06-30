from abc import ABC, abstractmethod

import fasttext as ft
import numpy
import spacy
from gensim.models import KeyedVectors


class AbstractEmbeddingModel(ABC):
    """
    Abstract class for embedding models.
    """

    def __init__(self):
        self._name = 'AbstractEmbeddingModel'
        self.model = None

    @property
    def name(self):
        return self._name

    @abstractmethod
    def get_embedding(self, text: str) -> numpy.ndarray:
        pass


class BERTEmbedding(AbstractEmbeddingModel):
    """
    Class for embedding models using BERT.
    """

    def __init__(self, model):
        super().__init__()
        self._name = f'{model}'
        self.model = spacy.load(model, disable=["ner", "textcat", "parser"])

    def get_embedding(self, text: str) -> numpy.ndarray:
        """
        Returns the embedding of the text.
        :param text:
        :return:
        """
        return self.model(text).vector


class FastTextEmbedding(AbstractEmbeddingModel):
    """
    Class for embedding models using FastText model.
    """

    def __init__(self, path, model='fastText'):
        super().__init__()
        self._name = f'{model}'
        self.model = ft.load_model(path)

    def get_embedding(self, text: str) -> numpy.ndarray:
        """
        Returns the embedding of the text.
        :param text:
        :return:
        """
        return self.model.get_sentence_vector(text)


class W2VEmbedding(AbstractEmbeddingModel):
    """
    Class for embedding models using Word2Vec model.
    """

    def __init__(self, path, model='W2V-Unk'):
        super().__init__()
        self._name = f'{model}'
        self.model = KeyedVectors.load_word2vec_format(path)

    def get_embedding(self, text: str) -> numpy.ndarray:
        """
        Returns the embedding of the text.
        :param text:
        :return:
        """
        embeddings = [self.model.get_vector(x) for x in text.split(' ') if x in self.model]
        return numpy.mean(embeddings, axis=0)
