import json
from abc import ABC, abstractmethod
from collections import Counter
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from multiset import Multiset
from sklearn.metrics.pairwise import cosine_similarity

from feature.embedding import AbstractEmbeddingModel


class Annotation(ABC):
    @abstractmethod
    def annotate(self, name, content):
        pass

    @staticmethod
    def load_keywords(keywords_files):
        keywords = {}
        weights = {}
        label_mapping = {}
        for keywords_file in keywords_files:
            label = keywords_file.stem
            label_mapping[label] = len(label_mapping)
            df = pd.read_csv(keywords_file)
            keywords[label] = Multiset(df['keyword'].tolist())
            weights[label] = dict(zip(df['keyword'].tolist(), df['tfidf'].tolist()))

        return keywords, label_mapping, weights


class KeywordAnnotation(Annotation):
    def __init__(self, keywords_files, annotations_path):
        self.keywords, self.weights, self.mapping = self.load_keywords(keywords_files)
        self.n = len(self.mapping)
        self.save_label_map(annotations_path)

    def save_label_map(self, annotations_path):
        out_path = Path(join(annotations_path, f"label_mapping.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'wt') as outf:
            json.dump(self.mapping, outf, ensure_ascii=False, indent=4)

    def annotate(self, name, content):
        node_labels = np.zeros(self.n)
        for label, kw in self.keywords.items():
            intersection = list(kw.intersection(Multiset(content.split())))
            intersection = Counter(intersection)
            node_labels[self.mapping[label]] = sum(
                [intersection[k] * self.weights[label][k] for k in intersection.keys()])

        norm = np.sum(node_labels)
        node_vec = node_labels / norm if norm > 0 else np.zeros(self.n)

        return node_vec


class SemanticSimilarityAnnotation(Annotation):
    def __init__(self, keywords_files, embedding: AbstractEmbeddingModel):
        self.embedding = embedding
        _, _, self.mapping = self.load_keywords(keywords_files)
        self.n = len(self.mapping)

    def annotate(self, name, content):
        name_vec = self.embedding.get_embedding(name)
        node_labels = np.zeros(self.n)
        for label in self.mapping:
            label_vec = self.embedding.get_embedding(label)
            sim = cosine_similarity(label_vec, name_vec)
            node_labels[self.mapping[label]] = sim

        return node_labels
