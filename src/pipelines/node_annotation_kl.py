import json
from collections import Counter
from os.path import join
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import pandas
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from multiset import Multiset
from numpy import exp
from omegaconf import DictConfig
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from utils import git_clone, get_versions, git_checkout


def compute_node_labels(project_content: Dict[str, str], keywords: Dict[str, Multiset],
                        weights: Dict[str, Dict[str, float]], label_mapping: Dict[str, int]) -> Dict[str, list]:
    """
    Computes a weighted product of the keywords of the labels of a project.
    :param project_content: The content of the project.
    :param keywords: The keywords of the labels.
    :param weights: The weights of the labels.
    :param label_mapping: The mapping of the labels.
    :return: The weighted product of the keywords of the labels of a project.
    """
    node_labels = {}
    skipped = 0
    unannoteted = {}
    uniform_vec = np.ones(len(label_mapping)) / len(label_mapping)
    for node, content in project_content.items():
        node_labels[node] = np.zeros(len(label_mapping))
        for label, kw in keywords.items():
            intersection = list(kw.intersection(Multiset(content.split())))
            intersection = Counter(intersection)
            node_labels[node][label_mapping[label]] = sum(
                [intersection[k] * weights[label][k] for k in intersection.keys()])
        # node_labels[node] = list(node_labels[node] / np.linalg.norm(node_labels[node]))
        norm = np.sum(node_labels[node])
        node_vec = node_labels[node] / norm if norm > 0 else uniform_vec
        if norm == 0:
            node_vec = np.zeros(len(label_mapping))
        # print(1 - exp(-entropy(node_labels[node], uniform_vec)))
        if jensenshannon(node_labels[node], uniform_vec) <= 0.5:
            node_vec = np.zeros(len(label_mapping))
            skipped += 1
            unannoteted[node] = 1

        node_labels[node] = node_vec.tolist()
    print("total", len(project_content))
    print("skipped", skipped)
    print("percentage", skipped / len(project_content))

    return node_labels, unannoteted


@hydra.main(config_path="../conf", config_name="keyword_extraction", version_base="1.2")
def node_annotation(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """
    content_extractor = instantiate(cfg.content)

    projects = pd.read_csv(cfg.dataset)

    projects = projects[projects['language'].str.upper() == cfg.language.upper()]
    projects = projects['full_name']

    projects = ['Waikato/weka-3.8', 'SonarSource/sonar-java', 'GenomicParisCentre/eoulsan']
    # projects = ['PolaricServer/aprsd']
    # projects = ['SonarSource/sonar-java'] #, 'SonarSource/sonar-java', 'GenomicParisCentre/eoulsan']

    logger.info(f"Extracting features for {len(projects)} projects")

    keywords_path = Path("/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/keywords/yake/similarity")

    keywords_files = sorted(list(keywords_path.glob("*.csv")))
    keywords = {}
    weights = {}
    label_mapping = {}
    inverse_map = {}
    for keywords_file in keywords_files:
        label = keywords_file.stem
        inverse_map[len(label_mapping)] = label
        label_mapping[label] = len(label_mapping)
        df = pandas.read_csv(keywords_file)
        keywords[label] = Multiset(df['keyword'].tolist())
        weights[label] = dict(zip(df['keyword'].tolist(), df['tfidf'].tolist()))

    out_path = join(cfg.annotations_path, content_extractor.method, f"label_mapping.json")
    with open(out_path, 'wt') as outf:
        json.dump(inverse_map, outf, ensure_ascii=False, indent=4)

    for project in projects:
        logger.info(f"Extracting features for {project}")

        project_name = project.replace('/', '|')
        if content_extractor.clone:
            project_url = f'https://github.com/{project}'
            git_clone(project_url, project_name, cfg.repositories_path)
        versions = [get_versions(project_name, cfg.arcan_graphs)[-1]]

        logger.info(f"Found {len(versions)} versions for project {project}")
        for num, sha in versions:
            try:
                if content_extractor.clone:
                    git_checkout(join(cfg.repositories_path, project_name), sha)
                project_content = dict(content_extractor.extract(project_name, sha, num))
                labels, unannoteted = compute_node_labels(project_content, keywords, weights, label_mapping)
                out_path = join(cfg.annotations_path, 'kl', content_extractor.method, f"{project_name}-{num}-{sha}.json")
                unannoteted_out_path = join(cfg.annotations_path, 'kl', content_extractor.method, f"unnanotated_{project_name}-{num}-{sha}.json")
                df_out_path = join(cfg.annotations_path, 'kl', content_extractor.method,
                                   f"{project_name}-{num}-{sha}.csv")
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump(labels, f, ensure_ascii=False)
                with open(unannoteted_out_path, 'w') as f:
                    json.dump(unannoteted, f, ensure_ascii=False)
                df = pandas.DataFrame.from_dict(labels, orient='index', columns=[i for i in range(len(inverse_map))])
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'node'}, inplace=True)
                df.to_csv(df_out_path, index=False, header=True)
            except Exception as e:
                logger.error(f"Failed to extract features for {project} {num} {sha}")
                logger.error(f"{e}")
                continue


if __name__ == '__main__':
    node_annotation()
