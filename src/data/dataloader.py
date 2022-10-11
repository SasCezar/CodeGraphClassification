import glob
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy
import torch
import torch_geometric
from more_itertools import flatten

from graph import ArcanGraphLoader


class ArcanDependenciesDataset(torch_geometric.data.Dataset):
    def __init__(self, root, projects, size, transform=None, pre_transform=None, pre_filter=None):
        self.projects = projects["names"].tolist()
        self.size = size
        self.versions = self.load_projects_versions(root)
        self.proj_mapping = dict(zip(projects["names"], projects["labels"]))
        self.label_id = dict(zip(projects["labels"], projects["labels_id"]))
        super(ArcanDependenciesDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return self.raw_file_names

    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, f"{x}.graphml") for x in flatten(list(self.versions.values()))]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return self.processed_file_names

    @property
    def processed_file_names(self):
        return [os.path.join(self.processed_dir, f"{x}.pt") for x in flatten(list(self.versions.values()))]

    def process(self):
        for i, raw_path in enumerate(self.raw_paths):
            graph = ArcanGraphLoader().load(raw_path)
            edge_index = self._get_edge_index(graph)
            feature_path = raw_path.replace(".graphml", ".vec")

            file = os.path.basename(os.path.basename(raw_path))
            project_w_version = os.path.splitext(file)[0]

            splitted = project_w_version.split("-")
            project = splitted[0] if len(splitted) == 3 else "-".join(splitted[:-2])
            version = splitted[-1]

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

            torch.save(data, os.path.join(self.processed_dir, file.replace(".graphml", ".pt")))

    def _load_node_features(self, path):
        features = {}
        with open(path, "rt", encoding="utf8") as inf:
            for line in inf:
                name_features = line.split(" ")
                name = " ".join(name_features[:-self.size])
                embedding = name_features[-self.size:]
                assert len(embedding) == self.size, print(len(embedding), len(self.size))
                features[name] = embedding

        return features

    @staticmethod
    def _get_edge_index(graph):
        source_vertices = []
        target_vertices = []
        for edge in graph.es:
            source_vertices.append(edge.source)
            target_vertices.append(edge.target)

        return torch.tensor([source_vertices, target_vertices], dtype=torch.long)

    def len(self):
        return len(self.projects)

    def get(self, idx):
        project = self.projects[idx]
        example = random.choice(self.versions[project])
        data = torch.load(os.path.join(self.processed_dir, f"{example}.pt"))
        return data

    @property
    def num_classes(self):
        return len(self.label_id)

    def align(self, graph, features):
        embeddings = []

        for i, node in enumerate(graph.vs):
            name = node['filePathRelative']
            if name in features:
                vector = features[name]
            else:
                vector = numpy.random.random(self.size)

            embeddings.append(vector)

        return torch.tensor(numpy.array(embeddings, dtype=float), dtype=torch.float)

    @staticmethod
    def get_filename(string):
        return Path(string).stem

    @staticmethod
    def get_proj_name(string):
        splitted = string.split("-")
        name = "-".join(splitted[:-2])
        return name

    def load_projects_versions(self, root):
        graph_files = {self.get_filename(x) for x in glob.glob(os.path.join(root, "raw", f"*.graphml"))}
        vec_files = {self.get_filename(x) for x in glob.glob(os.path.join(root, "raw", f"*.vec"))}
        intersection = graph_files & vec_files

        files = defaultdict(list)
        for file in intersection:
            proj = self.get_proj_name(file)
            if proj in self.projects:
                files[proj].append(file)

        return files


class ArcanDependenciesDatasetV2(torch_geometric.data.Dataset):
    def __init__(self, root, projects, size, transform=None, pre_transform=None, pre_filter=None):
        self.projects = projects["names"].tolist()
        self.size = size
        self.versions = self.load_projects_versions(root)
        self.proj_mapping = dict(zip(projects["names"], projects["labels"]))
        self.label_id = dict(zip(projects["labels"], projects["labels_id"]))
        super(ArcanDependenciesDatasetV2, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return self.raw_file_names

    @property
    def raw_file_names(self):
        return [os.path.join(self.raw_dir, f"{x}.graphml") for x in flatten(list(self.versions.values()))]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return self.processed_file_names

    @property
    def processed_file_names(self):
        return [os.path.join(self.processed_dir, f"{x}.pt") for x in flatten(list(self.versions.values()))]

    def process(self):
        for i, raw_path in enumerate(self.raw_paths):
            graph = ArcanGraphLoader().load(raw_path)
            edge_index = self._get_edge_index(graph)
            feature_path = raw_path.replace(".graphml", ".vec")

            file = os.path.basename(os.path.basename(raw_path))
            project_w_version = os.path.splitext(file)[0]

            splitted = project_w_version.split("-")
            project = splitted[0] if len(splitted) == 3 else "-".join(splitted[:-2])
            version = splitted[-1]

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

            torch.save(data, os.path.join(self.processed_dir, file.replace(".graphml", ".pt")))

    def _load_node_features(self, path):
        features = {}
        with open(path, "rt", encoding="utf8") as inf:
            for line in inf:
                name_features = line.split(" ")
                name = " ".join(name_features[:-self.size])
                embedding = name_features[-self.size:]
                assert len(embedding) == self.size, print(len(embedding), len(self.size))
                features[name] = embedding

        return features

    @staticmethod
    def _get_edge_index(graph):
        source_vertices = []
        target_vertices = []
        for edge in graph.es:
            source_vertices.append(edge.source)
            target_vertices.append(edge.target)

        return torch.tensor([source_vertices, target_vertices], dtype=torch.long)

    def len(self):
        return len(self.projects)

    def get(self, idx):
        project = self.projects[idx]
        example = random.choice(self.versions[project])
        data = torch.load(os.path.join(self.processed_dir, f"{example}.pt"))
        return data

    @property
    def num_classes(self):
        return len(self.label_id)

    def align(self, graph, features):
        embeddings = []

        for i, node in enumerate(graph.vs):
            name = node['filePathRelative']
            if name in features:
                vector = features[name]
            else:
                vector = numpy.random.random(self.size)

            embeddings.append(vector)

        return torch.tensor(numpy.array(embeddings, dtype=float), dtype=torch.float)

    @staticmethod
    def get_filename(string):
        return Path(string).stem

    @staticmethod
    def get_proj_name(string):
        splitted = string.split("-")
        name = "-".join(splitted[:-2])
        return name

    def load_projects_versions(self, root):
        graph_files = {self.get_filename(x) for x in glob.glob(os.path.join(root, "raw", f"*.graphml"))}
        vec_files = {self.get_filename(x) for x in glob.glob(os.path.join(root, "raw", f"*.vec"))}
        intersection = graph_files & vec_files

        files = defaultdict(list)
        for file in intersection:
            proj = self.get_proj_name(file)
            if proj in self.projects:
                files[proj].append(file)

        return files
