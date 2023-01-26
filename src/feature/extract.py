import json
import os
import re
from abc import abstractmethod, ABC
# import sourcy
from typing import Iterable

import igraph
import numpy as np
import pandas as pd
from more_itertools import flatten

from data.graph import ArcanGraphLoader
from feature.embedding import AbstractEmbeddingModel
from utils import check_dir


class FeatureExtraction(ABC):
    """
    Abstract method for extracting features.
    """

    def __init__(self, model: AbstractEmbeddingModel, graph_path: str = None, out_path: str = None,
                 stopwords: Iterable = None):
        """
        :param model: Embedding model.
        :param graph_path: Path to the graph directory.
        :param out_path: Path to the output directory.
        :param stopwords: List of stopwords.
        """
        self.nlp = model
        self.method = 'AbstractFE'
        self.graph_path = graph_path
        self.out_path = out_path
        if not stopwords:
            stopwords = set()
        self.stopwords = stopwords
        self.clone = True

    @abstractmethod
    def get_embeddings(self, project: str, graph: igraph.Graph):
        """
        Returns the embeddings of files in the graph.
        :param project:
        :param graph:
        :return:
        """
        raise NotImplemented()

    @staticmethod
    def split_camel(name: str):
        return re.sub(
            '([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)
        ).split()

    @staticmethod
    def save_features(features, path: str, file: str):
        """
        Saves the features in the path as a csv file.
        The first column is the file name and the other columns are the embedding.
        :param features: Features to save.
        :param path:
        :param file:
        :return:
        """
        out = os.path.join(path, file)
        df = pd.DataFrame(features, columns=['name', 'cleaned', 'embedding'])
        df['embedding'] = df['embedding'].apply(lambda x: x.tolist())
        df = pd.concat([df, df['embedding'].apply(pd.Series)], axis=1)
        df.drop(columns=['embedding', 'cleaned'], inplace=True)
        df.to_csv(out, sep=' ', index=False, header=False)

    def extract(self, project_name: str, sha: str = None, num: str = None, clean_graph: bool = False):
        """
        Extracts the features of the project.
        :param project_name: Name of the project.
        :param sha: SHA of the project version.
        :param num: Number of the project version in the git history.
        :param clean_graph: Whether to clean the graph.
        :return:
        """
        graph_file = f"dependency-graph-{num}_{sha}.graphml"
        features_name = f"dependency-graph-{num}_{sha}.vec"

        graph = ArcanGraphLoader(clean=clean_graph).load(os.path.join(self.graph_path, project_name, graph_file))
        features_out = os.path.join(self.out_path, "embedding", self.method, self.nlp.name, project_name)
        features = self.get_embeddings(project_name, graph)
        check_dir(features_out)

        self.save_features(features, features_out, features_name)


class NameFeatureExtraction(FeatureExtraction):
    """
    Extracts the features using the package name and class name as representation for the document.
    """

    def __init__(self, model: AbstractEmbeddingModel, graph_path: str = None, out_path: str = None,
                 stopwords: str = None):
        super().__init__(model, graph_path, out_path, stopwords)
        self.method = 'name'
        self.clone = False

    def get_embeddings(self, project: str, graph: igraph.Graph):
        """
        Returns the embeddings of files in the project.
        :param project: Name of the project
        :param graph: Graph of the project
        :return:
        """
        for node in graph.vs:
            name = node['name']
            name, clean = self.name_to_sentence(name)

            if not clean:
                clean = node['name']

            embedding = self.nlp.get_embedding(clean)
            yield name, clean, embedding

    def name_to_sentence(self, name: str):
        tokens = name.split(".")
        tokens = tokens[3:] if len(tokens) > 3 else ""
        clean = []

        for token in tokens:
            clean.extend(self.split_camel(token))

        return name, " ".join(clean).lower()


class IdentifiersFeatureExtraction(FeatureExtraction):
    """
    Extracts the features using the identifiers from the source code as representation for the document.
    """

    def __init__(self, model: AbstractEmbeddingModel, graph_path: str = None, out_path: str = None,
                 repo_path: str = None, preprocess: bool = True, stopwords: Iterable = None):
        super().__init__(model, graph_path, out_path, stopwords)
        self.scp = None  # sourcy.load("java")
        self.preprocess = preprocess
        self.repositories = repo_path
        self.method = 'identifiers'

    def get_embeddings(self, project: str, graph: igraph.Graph):
        """
        Returns the embeddings of files in the project.
        :param project: Name of the project
        :param graph: Graph of the project
        :return:
        """
        for node in graph.vs:
            path = os.path.join(self.repositories, project, node['filePathRelative'])

            if not os.path.isfile(path):
                continue

            identifiers = self.get_identifiers(path)
            text = " ".join(identifiers)
            embedding = self.nlp.get_embedding(text)

            yield node['filePathRelative'], path, embedding

    @staticmethod
    def read_file(filename: str):
        """
        Reads the file and returns the text.
        :param filename:
        :return:
        """
        with open(filename, "rt", encoding="utf8") as inf:
            text = inf.read()

        return text

    def get_identifiers(self, path: str):
        """
        Returns the source code identifiers from the file.
        :param path:
        :return:
        """
        text = self.read_file(path)
        doc = self.scp(text)
        ids = [self.split_camel(x.token) for x in doc.identifiers]
        ids = [x.lower() for x in flatten(ids)
               if x.lower() not in self.stopwords and len(x) > 1]

        return ids


class MethodFeatureExtraction(FeatureExtraction):
    """
    Extracts the features using the identifiers from the source code as representation for the document.
    """

    def __init__(self, model: AbstractEmbeddingModel, graph_path: str = None, out_path: str = None,
                 repo_path: str = None, methods_path: str = None, preprocess: bool = True, stopwords: Iterable = None):
        super().__init__(model, graph_path, out_path, stopwords)
        self.preprocess = preprocess
        self.repositories = repo_path
        self.method = 'methods'
        self.methods = {}
        self.methods_path = methods_path
        self.clone = False

    def extract(self, project_name: str, sha: str = None, num: str = None, clean_graph: bool = False):
        """
        Extracts the features of the project.
        :param project_name: Name of the project.
        :param sha: SHA of the project version.
        :param num: Number of the project version in the git history.
        :param clean_graph: Whether to clean the graph.
        :return:
        """

        graph_file = f"dependency-graph-{num}_{sha}.graphml"
        features_name = f"dependency-graph-{num}_{sha}.vec"

        graph = ArcanGraphLoader(clean=clean_graph).load(os.path.join(self.graph_path, project_name, graph_file))
        features_out = os.path.join(self.out_path, "embedding", self.method, self.nlp.name, project_name)
        self.methods = self.load_methods(os.path.join(self.methods_path, f'{project_name}.json'), num, sha)
        features = self.get_embeddings(project_name, graph)
        check_dir(features_out)

        self.save_features(features, features_out, features_name)

    def get_embeddings(self, project: str, graph: igraph.Graph):
        """
        Returns the embeddings of files in the project.
        :param project: Name of the project
        :param graph: Graph of the project
        :return:
        """
        for node in graph.vs:

            if node['filePathRelative'] not in self.methods:
                continue

            methods = self.methods[node['filePathRelative']]
            embeddings = []
            for method in methods:
                embeddings.append((method['body']))

            embedding = np.mean(embeddings, axis=0)
            yield node['filePathRelative'], '', embedding

    @staticmethod
    def load_methods(file, num, sha):
        with open(file, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if obj['num'] == num and obj['sha'] == sha:
                    return obj['content']
