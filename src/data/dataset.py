import os
from os.path import join
from typing import Callable, Optional

import numpy
import torch
import torch_geometric
from pandas import DataFrame
from tqdm import tqdm

from data.graph import ArcanGraphLoader


class GitRankingDataset(torch_geometric.data.Dataset):
    def __init__(self, graph_dir: str, feature_dir: str, out_dir: str,
                 projects: DataFrame, embedding_size: int,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        """
        :param graph_dir: directory containing the graphml files
        :param feature_dir: directory containing the feature files
        :param out_dir: directory where the processed files will be stored
        :param projects: dataframe containing the projects to be considered
        :param embedding_size: size of the embeddings
        :param transform: transformations to be applied on the graph data
        :param pre_transform: transformations to be applied before saving the data
        :param pre_filter: filter to be applied before saving the data
        """

        self.graph_dir = graph_dir
        self.feature_dir = feature_dir
        self.projects = projects
        self.size = embedding_size
        self.proj_mapping = dict(zip(projects["name"], projects["label"]))
        self.label_id = dict(zip(projects["label"], projects["labels_id"]))
        self.projects_ver_sha = list(zip(self.projects['name'], self.projects['version'], self.projects['sha']))
        super(GitRankingDataset, self).__init__(out_dir, transform, pre_transform, pre_filter)

    def download(self):
        pass

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return self.raw_file_names

    @property
    def raw_file_names(self):
        graphs = [join(self.graph_dir, project,
                       f"dependency-graph-{version}_{sha}.graphml") for project, version, sha in self.projects_ver_sha]
        features = [join(self.feature_dir, project,
                         f"dependency-graph-{version}_{sha}.vec") for project, version, sha in self.projects_ver_sha]

        return graphs + features

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return self.processed_file_names

    @property
    def processed_file_names(self):
        return [join(self.processed_dir, f"{project}-{version}_{sha}.vec") for project, version, sha in self.projects_ver_sha]

    def process(self):
        """
        Process the dataset by aligning the graph nodes with the node features and save the data as
        a torch_geometric.data.Data object.
        """
        for project, version, sha in tqdm(self.projects_ver_sha):
            graph_path = join(self.graph_dir, project, f"dependency-graph-{version}_{sha}.graphml")
            graph = ArcanGraphLoader().load(graph_path)
            edge_index = self._get_edge_index(graph)
            feature_path = join(self.feature_dir, project, f"dependency-graph-{version}_{sha}.vec")

            features = self._load_node_features(feature_path)

            embeddings = self.align(graph, features)

            assert len(graph.vs) == embeddings.shape[0], print(f"{len(graph.vs)} - {embeddings.shape[0]}")

            label = self.proj_mapping[project]
            label_id = self.label_id[label]
            data = torch_geometric.data.Data(x=embeddings,
                                             edge_index=edge_index,
                                             name=project,
                                             version=version,
                                             y=label_id,
                                             label_text=label)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"{project}-{version}_{sha}.pt"))

    def _load_node_features(self, path):
        """
        Load the node features from the file.
        The file contains one feature vector per line. The first element of the vector is the node name and the
        remaining elements are the feature values as in the Word2Vec format.
        """
        features = {}
        with open(path, "rt", encoding="utf8") as inf:
            for line in inf:
                name_features = line.split(" ")
                name = " ".join(name_features[:-self.size])
                embedding = name_features[-self.size:]
                assert len(embedding) == self.size, print(len(embedding), self.size)
                features[name] = embedding

        return features

    @staticmethod
    def _get_edge_index(graph):
        """
        Converts the graph edges to a torch_geometric compatible format.
        The format is a 2xN tensor, where N is the number of edges.
        The first row contains the source nodes, the second row the destination nodes.
        """
        source_vertices = []
        target_vertices = []
        for edge in graph.es:
            source_vertices.append(edge.source)
            target_vertices.append(edge.target)

        return torch.tensor([source_vertices, target_vertices], dtype=torch.long)

    def len(self):
        """
        Return the number of data in the dataset.
        """
        return len(self.projects_ver_sha)

    def get(self, idx):
        """
        Get the data object at index idx.
        """
        project, version, sha = self.projects_ver_sha[idx]
        data = torch.load(os.path.join(self.processed_dir, f"{project}-{version}_{sha}.pt"))
        return data

    @property
    def num_classes(self):
        """
        Returns the number of classes in the dataset.
        """
        return len(self.label_id)

    def align(self, graph, features):
        """
        Align the graph nodes with the node features.
        Nodes that are not present in the features are assigned a random embedding.
        """
        embeddings = []

        for i, node in enumerate(graph.vs):
            name = node['filePathRelative']
            if name in features:
                vector = features[name]
            else:
                vector = numpy.random.random(self.size)

            embeddings.append(vector)

        return torch.tensor(numpy.array(embeddings, dtype=float), dtype=torch.float)
