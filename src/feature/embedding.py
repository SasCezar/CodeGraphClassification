import re
from abc import ABC, abstractmethod

import fasttext as ft
import numpy
import numpy as np
import spacy
import torch
from gensim.models import KeyedVectors
from transformers import BertModel, BertTokenizer


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

    def __init__(self, path: str, model: str = 'fastText'):
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

    def __init__(self, path: str, model: str = 'W2V-Unk'):
        super().__init__()
        self._name = f'{model}'
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)

    def get_embedding(self, text: str) -> numpy.ndarray:
        """
        Returns the embedding of the text.
        :param text:
        :return:
        """
        embeddings = []
        if not text:
            embeddings.append(np.zeros(self.model.vector_size))
        for word in str(text).split():
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                embeddings.append(np.zeros(self.model.vector_size))
        return numpy.mean(embeddings, axis=0)


class SplitW2VEmbedding(W2VEmbedding):
    def get_embedding(self, text: str) -> numpy.ndarray:
        """
        Returns the embedding of the text.
        :param text:
        :return:
        """
        embeddings = []
        if not text:
            embeddings.append(np.zeros(self.model.vector_size))
        for word in self.split_camel(str(text)):
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                embeddings.append(np.zeros(self.model.vector_size))
        return numpy.mean(embeddings, axis=0)

    def split_camel(self, name: str):
        return re.sub(
            '([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)
        ).split()


class HuggingFaceEmbedding(AbstractEmbeddingModel):
    """
    Class for embedding models using HuggingFace.
    """

    def __init__(self, name, model):
        super().__init__()
        self._name = f'{name}'
        do_lower_case = True
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=do_lower_case)

    def get_embedding(self, text: str) -> numpy.ndarray:
        """
        Returns the embedding of the text.
        :param text:
        :return:
        """
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states.mean(1).detach().numpy()[0]
