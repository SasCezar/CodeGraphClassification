import json
import os
from ast import literal_eval
from os.path import join
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from numpy.linalg import norm
from omegaconf import DictConfig
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from utils import get_versions


def projects_level_labels(annotations_path):
    node_labels = []
    with open(annotations_path, 'rt') as inf:
        obj = json.load(inf)

    unannotated_count = 0
    for node, labels in obj.items():
        if not labels['unannotated'] and norm(labels['distribution']):
            node_labels.append(labels['distribution'])
        else:
            unannotated_count += 1
    agg = np.array(node_labels).mean(axis=0)
    agg = np.array(agg) / norm(agg)

    return agg, unannotated_count, len(obj)


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.3")
def annotate_project(cfg: DictConfig):
    projects = pd.read_csv(cfg.dataset)

    projects = projects[projects['language'].str.upper() == cfg.language.upper()]
    proj_labels = [literal_eval(x) for x in projects['labels']]

    projects = projects['full_name']

    logger.info(f"Extracting features for {len(projects)} projects")

    label_mapping = join(cfg.annotations_dir, f"label_mapping.json")
    with open(label_mapping, 'rt') as outf:
        label_map = json.load(outf)
    label_map = {v: k for k, v in label_map.items()}

    Path(cfg.project_labels_path).mkdir(parents=True, exist_ok=True)
    with open(join(cfg.project_labels_path, "annotations.json"), 'wt') as outf:
        for project, labels in tqdm(zip(projects, proj_labels)):
            try:
                project_name = project.replace('/', '|')
                num, sha = get_versions(project_name, cfg.arcan_graphs)[-1]

                project_annotation_path = join(cfg.annotations_path, f"{project_name}-{num}-{sha}.json")

                project_labels, unannotated_count, total_nodes = projects_level_labels(project_annotation_path)
                n = len(project_labels)
                uniform_vec = np.ones(n) / n
                jsd = jensenshannon(project_labels, uniform_vec)

                sorted_labels = np.argsort(project_labels)[::-1]

                y_labels = {}
                for i in sorted_labels[:]:
                    y_labels[label_map[i]] = project_labels[i]

                json_labels = json.dumps({"project": project_name, "true_labels": labels,
                                          "predicted_labels": list(y_labels.keys()), "predicted_prob": y_labels,
                                          "jsd": jsd, "distribution": project_labels.tolist(),
                                          "unannotated": unannotated_count,
                                          "total_nodes": total_nodes}, ensure_ascii=False)
                outf.write(json_labels + os.linesep)
            except IndexError as e:
                continue
            except Exception as e:
                continue


if __name__ == '__main__':
    annotate_project()
