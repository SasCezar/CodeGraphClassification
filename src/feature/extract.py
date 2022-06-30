import os
import re
from abc import abstractmethod, ABC

# import sourcy
from more_itertools import flatten

from data.graph import ArcanGraphLoader
from feature.embedding import AbstractEmbeddingModel
from utils import check_dir


class FeatureExtraction(ABC):
    """
    Abstract method for extracting features.
    """

    def __init__(self, model: AbstractEmbeddingModel, graph_path=None, out_path=None, stopwords=None):
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

    @abstractmethod
    def get_embeddings(self, project, graph):
        """
        Returns the embeddings of files in the graph.
        :param project:
        :param graph:
        :return:
        """
        raise NotImplemented()

    @staticmethod
    def split_camel(name):
        return re.sub(
            '([A-Z][a-z]+)|_', r' \1', re.sub('([A-Z]+)', r' \1', name)
        ).split()

    @staticmethod
    def save_features(features, path, file):
        """
        Saves the features in the path.
        :param features:
        :param path:
        :param file:
        :return:
        """
        out = os.path.join(path, file)
        with open(out, "wt", encoding="utf8") as outf:
            for name, cleaned, embedding in features:
                if not isinstance(embedding, list):
                    embedding = embedding.tolist()
                rep = " ".join(str(x) for x in embedding)
                line = name + " " + rep + "\n"
                outf.write(line)

    def extract(self, project_name, sha=None, num=None, clean_graph=False):
        """
        Extracts the features of the project.
        :param project_name:
        :param sha:
        :param num:
        :param clean_graph:
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

    def __init__(self, model: AbstractEmbeddingModel, graph_path=None, out_path=None, stopwords=None):
        super().__init__(model, graph_path, out_path, stopwords)
        self.method = 'name'

    def get_embeddings(self, project, graph):
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

    def name_to_sentence(self, name):
        tokens = name.split(".")[2:]
        clean = []

        for token in tokens:
            clean.extend(self.split_camel(token))

        return name, " ".join(clean).lower()


class IdentifiersFeatureExtraction(FeatureExtraction):
    """
    Extracts the features using the identifiers from the source code as representation for the document.
    """

    def __init__(self, model: AbstractEmbeddingModel, graph_path=None, out_path=None,
                 repo_path=None, preprocess=True, stopwords=None):
        super().__init__(model, graph_path, out_path, stopwords)
        self.scp = None  # sourcy.load("java")
        self.preprocess = preprocess
        self.repositories = repo_path
        self.method = 'identifiers'

    def get_embeddings(self, project, graph):
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
    def read_file(filename):
        """
        Reads the file and returns the text.
        :param filename:
        :return:
        """
        with open(filename, "rt", encoding="utf8") as inf:
            text = inf.read()

        return text

    def get_identifiers(self, path):
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
