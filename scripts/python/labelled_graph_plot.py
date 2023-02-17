import glob
import json
import warnings
from os.path import join
from pathlib import Path

import hydra
import igraph
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from data.graph import ArcanGraphLoader
from utils import get_versions

warnings.filterwarnings('ignore')


def projects_level_labels(annotations):
    res = []
    for node in annotations:
        if not annotations[node]['unannotated'] and np.linalg.norm(annotations[node]['distribution']):
            res.append(annotations[node]['distribution'])

    agg = np.mean(np.array(res), axis=0) / np.linalg.norm(res)

    return agg


def annotate_nodes(graph, node_annotations, top_labels, label_map, k):
    grouped = 0
    tot = 0
    group_map = dict()
    for n in graph.vs:
        if n['labelV'] != 'container':
            tot += 1
            n['plot_weight'] = 0
            out_edges = graph.incident(n, mode="out")
            assert len(out_edges) < 2
            if not out_edges:
                continue
            out_edge = out_edges[0]
            target_vertex = graph.vs[graph.es[out_edge].target]
            if target_vertex['labelV'] == 'container':
                if target_vertex.index not in group_map:
                    group_map[target_vertex.index] = len(group_map)

                g_id = group_map[target_vertex.index]
                n['group'] = str(g_id)
                # graph.vs[target_vertex.index]['group'] = str(g_id)
                # n['p_label'] = str(graph.vs[target_vertex.index]['name'])
                grouped += 1

        try:
            if node_annotations[n['filePathRelative']]['unannotated'] or not np.linalg.norm(
                    node_annotations[n['filePathRelative']]['distribution']):
                n['topic'] = str(-1)
                continue
        except:
            n['topic'] = str(-1)
            continue

        sorted_labels = np.argsort(node_annotations[n['filePathRelative']]['distribution'])[::-1]

        y_labels = []
        for i in sorted_labels[:]:
            y_labels.append(label_map[i])
        annotated = False
        for i in range(k):
            if y_labels[i] in top_labels:
                n['topic'] = str(y_labels[i]).title()
                annotated = True
                continue

        if not annotated:
            n['topic'] = "Other"

    return graph


def load_node_annotations(project_annotation_path):
    with open(project_annotation_path, 'rt') as inf:
        return json.load(inf)


def clean_edges(graph):
    remove = []
    for edge in graph.es:
        if edge['labelE'] != 'belongsTo':
            remove.append(edge)

    graph.delete_edges(remove)

    return graph


def clean_nodes(graph):
    delete = [x.index for x in graph.vs if x['labelV'] not in {'container', 'unit'}]
    graph.delete_vertices(delete)

    return graph


def load_package_annotations(path, project):
    with open(path, 'rt') as inf:
        for line in inf:
            if project in line:
                annot = json.loads(line)
                return annot['packages']


def add_root(graph):
    root_nodes = [n for n in graph.vs if not graph.incident(n, mode="out")]
    n_id = len(graph.vs)
    graph.add_vertex(n_id)
    graph.vs[n_id]['labelV'] = 'container'
    graph.add_edges([(n_id, n) for n in root_nodes])

    return graph


def annotate_package(graph, package_annotations, top_labels, label_map, k):
    total = 0
    annotated = 0
    csv_rows = []
    for n in graph.vs:
        if n['labelV'] == 'container':
            total += 1
            if n['name'] in package_annotations:
                # typ = 'Code'
                if 'test' in n['name']:
                    continue
                    # typ = 'Test'

                annotated += 1
                annot = package_annotations[n['name']]['all_distribution']
                sorted_labels = np.argsort(annot)[::-1]

                n['num_files'] = len([1 for x in graph.incident(n, mode="in") if graph.vs[x]['labelV'] == 'unit'])

                y_labels = []
                for i in sorted_labels[:]:
                    y_labels.append(label_map[i])

                annotated = False
                for i in range(k):
                    if y_labels[i] in top_labels:
                        n['package_label'] = str(y_labels[i]).title()
                        annotated = True
                        break
                if not annotated:
                    n['package_label'] = str("Other")

                csv_rows.append((n['name'], n['num_files'], n['package_label']))

    return graph, csv_rows


@hydra.main(config_path="../../src/conf", config_name="annotation", version_base="1.3")
def annotate_graphml(cfg: DictConfig):
    # project = "Waikato|weka-3.8"
    # project = "apache|zookeeper"
    projects = glob.glob(join(cfg.annotations_path, '*.json'))
    projects = [Path(n).name.split('-')[0] for n in projects]
    Path(cfg.labelled_graph_path).mkdir(parents=True, exist_ok=True)
    skipped = 0
    # projects = ['Waikato|weka-3.8', 'apache|zookeeper']
    for project in tqdm(projects):
        try:
            num, sha = get_versions(project, cfg.arcan_graphs)[-1]
            graph_path = f"dependency-graph-{num}_{sha}.graphml"

            annot_path = join(cfg.annotations_path, f"{project}-{num}-{sha}.json")
            node_annotations = load_node_annotations(annot_path)

            project_labels = projects_level_labels(node_annotations)
            project_labels = sorted(range(len(project_labels)), key=lambda i: -project_labels[i])
            graph = ArcanGraphLoader(clean=False).load(join(cfg.arcan_graphs, project, graph_path))
            graph = clean_edges(graph)
            graph = clean_nodes(graph)
            # graph = add_root(graph)
            label_mapping = join(cfg.annotations_dir, f"label_mapping.json")
            with open(label_mapping, 'rt') as outf:
                label_map = json.load(outf)
            label_map = {label_map[l]: l for l in label_map}
            package_annotations = load_package_annotations(join(cfg.package_labels_path, 'annotations.json'), project)
            for k in [3, 5, 10]:
                for t in [1, k]:
                    top_k = [label_map[i] for i in project_labels[:k]]
                    graph = annotate_nodes(graph, node_annotations, top_k, label_map, t)
                    graph, csv_rows = annotate_package(graph, package_annotations, top_k, label_map, t)
                    igraph.Graph.write_gml(graph, join(cfg.labelled_graph_path, f'{project}_top_{k}_assign_{t}.gml'))
                    df = pd.DataFrame(csv_rows, columns=['name', 'weight', 'label'])
                    df.to_csv(join(cfg.labelled_graph_path, f'{project}_top_{k}_assign_{t}.csv'))
                    # df.to_csv(f'{project}_top_{k}_assign_{t}.csv')
        except IndexError as e:
            skipped += 1
            continue
        except TypeError as e:
            print(project)
            skipped += 1
            continue

    print('Skipped', skipped)


if __name__ == '__main__':
    annotate_graphml()
