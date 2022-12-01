from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict

import numpy as np
from multiset import Multiset
from sklearn.metrics.pairwise import cosine_similarity

from annotation.filtering import Filtering
from feature.embedding import AbstractEmbeddingModel


class Annotation(ABC):
    def __init__(self, filtering: Filtering):
        self.filter = filtering

    @abstractmethod
    def annotate(self, name, content):
        pass


class KeywordAnnotation(Annotation):
    def __init__(self, keywords: Dict[str, Multiset], weights: Dict[str, Dict[str, float]],
                 mapping: Dict[str, int], filtering: Filtering):
        super().__init__(filtering)
        self.keywords = keywords
        self.weights = weights
        self.mapping = mapping
        self.n = len(self.mapping)

    def annotate(self, name, content):
        node_labels = np.zeros(self.n)
        for label, kw in self.keywords.items():
            intersection = list(kw.intersection(Multiset(content.split())))
            intersection = Counter(intersection)
            node_labels[self.mapping[label]] = sum(
                [intersection[k] * self.weights[label][k] for k in intersection.keys()])

        norm = np.sum(node_labels)
        node_vec = node_labels / norm if norm > 0 else np.zeros(self.n)

        if self.filter:
            node_vec = self.filter.filter(node_vec)

        return node_vec


class SemanticSimilarityAnnotation(Annotation):
    def __init__(self, embedding: AbstractEmbeddingModel, filtering: Filtering, mapping: Dict[str, str]):
        super().__init__(filtering)
        self.embedding = embedding
        self.mapping = mapping
        self.n = len(self.mapping)

    def annotate(self, name, content):
        name_vec = self.embedding.get_embedding(name)
        node_labels = np.zeros(self.n)
        for label in self.mapping:
            label_vec = self.embedding.get_embedding(label)
            sim = cosine_similarity(label_vec, name_vec)
            node_labels[self.mapping[label]] = sim

        if self.filter:
            node_labels = self.filter.filter(node_labels)

        return node_labels
