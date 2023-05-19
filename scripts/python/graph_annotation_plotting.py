import json

import igraph
import numpy as np
import pandas
from hydra import initialize, compose

from data.graph import ArcanGraphLoader


def projects_level_labels(annotations_path):
    df = pandas.read_csv(annotations_path)
    df.drop('node', axis=1, inplace=True)
    agg = df.mean(axis=0)
    agg = np.array(agg) / np.linalg.norm(agg)

    return agg


def random_node(graph):
    """
    Return a random node from the graph with an indegree between 20 and 50
    """
    while True:
        n = np.random.randint(0, len(graph.vs))
        if graph.vs[n]['labelV'] != 'container' and 10 < igraph.Graph.indegree(graph, n) < 20:
            return n


def annotate_subgraph(subgraph, node_annotations, top_k_labels, label_map):
    for n in subgraph.vs:
        tot = 0
        if node_annotations[n['filePathRelative']]['unannotated']:
            n['Unannotated'] = 1
            continue
        for l in top_k_labels:
            n[label_map[str(l)]] = node_annotations[n['filePathRelative']]['distribution'][l]
            tot += node_annotations[n['filePathRelative']]['distribution'][l]
        n['Other'] = 1 - tot
    return subgraph


def load_node_annotations(project_annotation_path):
    with open(project_annotation_path.replace('.csv', '.json'), 'rt') as inf:
        return json.load(inf)


def best_node(graph, node_annotations, top_label):
    best_node = None
    best_score = 0
    for n in graph.vs:
        if n['filePathRelative'] in node_annotations:
            score = node_annotations[n['filePathRelative']][top_label]
            if best_score < score < 0.75:
                best_score = score
                best_node = n.index
    print(best_score)
    return best_node


def export_for_r():
    with initialize(version_base=None, config_path="../../src/conf/"):
        cfg = compose(config_name='annotation.yaml', overrides=["local=default"])

    project = "Weka"
    project_graph_path = f"{cfg.base_path}/data/interim/arcanOutput/Waikato|weka-3.8/dependency-graph-903_04804ccd6dff03534cbf3f2a71a35c73eef24fe8.graphml"
    project_annotation_path = f"{cfg.base_path}/data/processed/annotations/kl/name/Waikato|weka-3.8-903-04804ccd6dff03534cbf3f2a71a35c73eef24fe8.csv"
    label_mapping_path = f"{cfg.base_path}/processed/annotations/name/label_mapping.json"

    k = 5

    project_labels = projects_level_labels(project_annotation_path)

    top_k = sorted(range(len(project_labels)), key=lambda i: -project_labels[i])[:k]
    node_annotations = load_node_annotations(project_annotation_path)
    graph = ArcanGraphLoader(clean=True).load(project_graph_path)
    initial_node = random_node(graph)
    neighbor = graph.neighborhood([initial_node], 1)[0][0]
    subgraph_vertices = graph.neighborhood([initial_node, neighbor], 1)[0]
    subgraph = igraph.Graph.subgraph(graph, subgraph_vertices)

    with open(label_mapping_path, 'rt') as inf:
        label_map = json.load(inf)

    print(initial_node)
    subgraph = annotate_subgraph(subgraph, node_annotations, top_k, label_map)
    igraph.Graph.write_gml(subgraph, f"{project}_node_{initial_node}_subgraph.graphml")


if __name__ == '__main__':
    export_for_r()
