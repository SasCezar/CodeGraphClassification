import json
import os
from os.path import join
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from utils import get_versions


def projects_level_labels(annotations_path):
    node_labels = []
    with open(annotations_path, 'rt') as inf:
        obj = json.load(inf)

    for node, labels in obj.items():
        if not labels['unannotated']:
            node_labels.append(labels['distribution'])
    agg = np.array(node_labels).mean(axis=0)
    agg = np.array(agg) / np.linalg.norm(agg)

    return agg


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def annotate_project(cfg: DictConfig):
    projects = pd.read_csv(cfg.dataset)

    projects = projects[projects['language'].str.upper() == cfg.language.upper()]
    projects = projects['full_name']

    logger.info(f"Extracting features for {len(projects)} projects")

    label_mapping = join(cfg.annotations_path, f"label_mapping.json")
    with open(label_mapping, 'rt') as outf:
        label_map = json.load(outf)
    label_map = {v: k for k, v in label_map.items()}

    Path(cfg.project_labels_dir).mkdir(parents=True, exist_ok=True)
    with open(join(cfg.project_labels_dir, "annotations.json"), 'wt') as outf:
        for project in projects:
            try:
                logger.info(f"Extracting features for {project}")

                project_name = project.replace('/', '|')
                num, sha = get_versions(project_name, cfg.arcan_graphs)[-1]

                project_annotation_path = join(cfg.annotations_path, f"{project_name}-{num}-{sha}.json")

                k = 5

                project_labels = projects_level_labels(project_annotation_path)

                sorted_labels = np.argsort(project_labels)[::-1]
                labels = {label_map[i]: project_labels[i] for i in sorted_labels[:k]}
                json_labels = json.dumps({"project": project_name, "labels": labels}, ensure_ascii=False)
                outf.write(json_labels + os.linesep)

            except Exception as e:
                logger.error(e)
                continue


if __name__ == '__main__':
    annotate_project()
