from os.path import join
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from hydra import initialize, compose
from loguru import logger
from tqdm import tqdm

from data.graph import ArcanGraphLoader


def load_arcan_graphs_path(cfg) -> List[Path]:
    arcan_path_project = Path(join(cfg.arcan_out, 'arcanOutput/'))
    projects = list(arcan_path_project.glob('**/*.graphml'))

    return projects


def load_labels(cfg) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    df = pd.read_csv(join(cfg.raw_data, 'classification_dataset_lang.csv'))
    df.drop(df[df['language'].str.upper() != cfg.language.upper()].index, inplace=True)
    df['name'] = df['full_name'].apply(lambda x: x.replace('/', '|'))
    label_mapping = df.set_index('name')['labels'].to_dict()
    level_mapping = df.set_index('name')['levels'].to_dict()
    return label_mapping, level_mapping


def load_arcan_graphs(arcan_graphs_paths: List[Path]) -> Tuple[List[Tuple[str, str, str, int, int]], int]:
    loader = ArcanGraphLoader()
    graphs = []
    skipped = 0
    for project_path in tqdm(arcan_graphs_paths):
        try:
            project = loader.load(str(project_path))
            nodes = len(project.vs)
            edges = len(project.es)
            name = project_path.parent.name
            ver, sha = project_path.stem.replace('dependency-graph-', '').split('_')
            graphs.append((name, ver, sha, nodes, edges))
        except Exception as _:
            skipped += 1

    return graphs, skipped


def create_dataframe(arcan_graphs, label_map, level_map) -> pd.DataFrame:
    names, labels, levels, version, shas, nodes, edges = [], [], [], [], [], [], []
    for name, ver, sha, num_nodes, num_edges in arcan_graphs:
        proj_label = eval(label_map[name])
        proj_levels = eval(level_map[name])
        names.append(name)
        labels.append(proj_label)
        levels.append(proj_levels)
        version.append(ver)
        shas.append(sha)
        nodes.append(num_nodes)
        edges.append(num_edges)

    df = pd.DataFrame({'name': names, 'label': labels, 'level': levels, 'version': version,
                       'sha': shas, 'nodes': nodes, 'edges': edges})
    return df


def main():
    with initialize(version_base=None, config_path="../../src/conf/"):
        cfg = compose(config_name='extract_features.yaml', overrides=["local=default"])

    arcan_graphs_paths = load_arcan_graphs_path(cfg)
    arcan_graphs, skipped = load_arcan_graphs(arcan_graphs_paths)

    logger.info(f"Skipped {skipped} projects")

    project_labels, project_levels = load_labels(cfg)

    df = create_dataframe(arcan_graphs, project_labels, project_levels)
    df.to_csv(join(cfg.raw_data, 'dataset_with_graphs.csv'), index=False)

    df.to_csv(join(cfg.raw_data, 'dataset_with_graphs.csv'), index=False)
    df.to_json(join(cfg.raw_data, 'dataset_with_graphs.jsonl'), orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
