import json
import os
import re
from abc import abstractmethod, ABC
from typing import Iterable, List, Union

import igraph
import sourcy
from more_itertools import flatten

from data.graph import ArcanGraphLoader


class ContentExtraction(ABC):
    """
    Abstract method for extracting features.
    """

    def __init__(self, graph_path: str = None,
                 stopwords: Iterable = None):
        """
        :param model: Embedding model.
        :param graph_path: Path to the graph directory.
        :param out_path: Path to the output directory.
        :param stopwords: List of stopwords.
        """

        self.method = 'AbstractFE'
        self.graph_path = graph_path
        if not stopwords:
            stopwords = set()
        self.stopwords = stopwords
        self.clone = True

    @abstractmethod
    def get_content(self, project: str, graph: igraph.Graph):
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

    def extract(self, project_name: str, sha: str = None, num: str = None, clean_graph: bool = False) -> str:
        """
        Extracts the features of the project.
        :param project_name: Name of the project.
        :param sha: SHA of the project version.
        :param num: Number of the project version in the git history.
        :param clean_graph: Whether to clean the graph.
        :return:
        """
        graph_file = f"dependency-graph-{num}_{sha}.graphml"

        graph = ArcanGraphLoader(clean=clean_graph).load(os.path.join(self.graph_path, project_name, graph_file))
        content = self.get_content(project_name, graph)
        return content


class NameContentExtraction(ContentExtraction):
    """
    Extracts the features using the package name and class name as representation for the document.
    """

    def __init__(self, graph_path: str = None,
                 stopwords: str = None):
        super().__init__(graph_path, stopwords)
        self.method = 'name'
        self.clone = False

    def get_content(self, project: str, graph: igraph.Graph):
        """
        Returns the embeddings of files in the project.
        :param project: Name of the project
        :param graph: Graph of the project
        :return:
        """
        for node in graph.vs:
            name = node['name']
            clean = self.name_to_sentence(name)

            yield node['filePathRelative'], clean

    def name_to_sentence(self, name: str):
        tokens = name.split(".")

        tokens = tokens[3:] if len(tokens) > 3 else []

        clean = []
        for token in tokens:
            clean.extend(self.split_camel(token))

        return " ".join(clean).lower()


class IdentifiersContentExtraction(ContentExtraction):
    """
    Extracts the features using the identifiers from the source code as representation for the document.
    """

    def __init__(self, graph_path: str = None,
                 repo_path: str = None, stopwords: Iterable = None):
        super().__init__(graph_path, stopwords)
        self.scp = sourcy.load("java")
        self.repositories = repo_path
        self.method = 'identifiers'

    def get_content(self, project: str, graph: igraph.Graph):
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
            yield node['filePathRelative'], text

    @staticmethod
    def read_file(filename: str):
        """
        Reads the file and returns the text.
        :param filename:
        :return:
        """
        try:
            with open(filename, "rt", encoding="utf8") as inf:
                text = inf.read()
        except UnicodeDecodeError:
            with open(filename, "rt", encoding="latin1") as inf:
                text = inf.read()
        except Exception:
            return ""

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


class CommentsContentExtraction(ContentExtraction):
    """
    Extracts the features using the identifiers from the source code as representation for the document.
    """

    def __init__(self, graph_path: str = None,
                 repo_path: str = None, stopwords: Iterable = None):
        super().__init__(graph_path, stopwords)
        self.scp = sourcy.load("java")
        self.repositories = repo_path
        self.method = 'comments'

    def get_content(self, project: str, graph: igraph.Graph):
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
            yield node['filePathRelative'], text

    @staticmethod
    def read_file(filename: str):
        """
        Reads the file and returns the text.
        :param filename:
        :return:
        """
        try:
            with open(filename, "rt", encoding="utf8") as inf:
                text = inf.read()
        except UnicodeDecodeError:
            with open(filename, "rt", encoding="latin1") as inf:
                text = inf.read()
        except Exception:
            return ""

        return text

    def get_identifiers(self, path: str):
        """
        Returns the source code identifiers from the file.
        :param path:
        :return:
        """
        text = self.read_file(path)
        doc = self.scp(text)
        ids = [x.token for x in doc.comments]
        ids = [x.lower() for x in ids if "license" not in x.lower() and "copyright" not in x.lower()]
        ids = [x.lower() for x in flatten(ids)
               if x.lower() not in self.stopwords and len(x) > 1]

        return ids


class JSONContentExtraction(ContentExtraction):
    def get_content(self, project: str, graph: igraph.Graph):
        pass

    def __init__(self, content_path: str = None):
        super().__init__(content_path, None)
        self.content_path = content_path
        self.clone = False

    def extract(self, project_name: str, sha: str = None, num: str = None, clean_graph: bool = False) -> str:
        """
        Extracts the content of the project.
        """
        project = f"{project_name}.json"
        with open(os.path.join(self.content_path, project), "r") as f:
            for line in f:
                obj = json.loads(line)
                if obj['sha'] == sha:
                    return obj['content']

        raise ValueError(f"Could not find {sha} in {project_name}")

    @staticmethod
    def read_file(filename: str):
        """
        Reads the file and returns the text.
        :param filename:
        :return:
        """
        try:
            with open(filename, "rt", encoding="utf8") as inf:
                text = inf.read()
        except UnicodeDecodeError:
            with open(filename, "rt", encoding="latin1") as inf:
                text = inf.read()
        except Exception:
            return ""

        return text


class PreComputedEmbeddingLoader(ContentExtraction):
    def __init__(self, content_path: str = None):
        super().__init__(content_path, None)
        self.content_path = content_path
        self.clone = False

    def extract(self, project_name: str, sha: str = None, num: str = None, clean_graph: bool = False) -> Union[
        str, List[float]]:
        """
        Extracts the content of the project.
        """
        vec_file = f"dependency-graph-{num}_{sha}.vec"
        with open(os.path.join(self.content_path, project_name, vec_file), "r") as f:
            for line in f:
                node, vec = line.split(maxsplit=1)
                yield node, vec

        raise ValueError(f"Could not find {sha} in {project_name}")

    def get_content(self, project: str, graph: igraph.Graph):
        pass
