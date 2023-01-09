import json
import os
import traceback
from ast import literal_eval
from os.path import join
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from utils import get_versions


def projects_level_labels(annotations_path):
    node_labels = []
    with open(annotations_path, 'rt') as inf:
        obj = json.load(inf)

    unannotated_count = 0
    for node, labels in obj.items():
        if not labels['unannotated']:
            node_labels.append(labels['distribution'])
        else:
            unannotated_count += 1
    agg = np.array(node_labels).mean(axis=0)
    agg = np.array(agg) / np.linalg.norm(agg)

    return agg, unannotated_count, unannotated_count/len(obj)


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def annotate_project(cfg: DictConfig):
    projects = pd.read_csv(cfg.dataset)

    projects = projects[projects['language'].str.upper() == cfg.language.upper()]
    proj_labels = [literal_eval(x) for x in projects['labels']]

    projects = projects['full_name']

    logger.info(f"Extracting features for {len(projects)} projects")

    label_mapping = join(cfg.annotations_path, f"label_mapping.json")
    with open(label_mapping, 'rt') as outf:
        label_map = json.load(outf)
    label_map = {v: k for k, v in label_map.items()}

    Path(cfg.project_labels_dir).mkdir(parents=True, exist_ok=True)
    num_unnanotated = []
    percent_unnanotated = []
    pred_labels = []
    true_labels = []
    with open(join(cfg.project_labels_dir, "annotations.json"), 'wt') as outf:
        for project, labels in tqdm(zip(projects, proj_labels)):
            try:
                project_name = project.replace('/', '|')
                num, sha = get_versions(project_name, cfg.arcan_graphs)[-1]

                project_annotation_path = join(cfg.annotations_path, f"{project_name}-{num}-{sha}.json")

                k = 3

                project_labels, unannotated_count, unannotated_percent = projects_level_labels(project_annotation_path)
                num_unnanotated.append(unannotated_count)
                percent_unnanotated.append(unannotated_percent)

                sorted_labels = np.argsort(project_labels)[::-1]
                #y_labels = {label_map[i]: project_labels[i] for i in sorted_labels[:k]}
                y_labels = {}
                for i in sorted_labels[:k]:
                    y_labels[label_map[i]] = project_labels[i]
                pred_labels.append(y_labels)
                true_labels.append(labels)
                json_labels = json.dumps({"project": project_name, "true_labels": labels,
                                          "predicted_labels": list(y_labels.keys()), "predicted_prob": y_labels},
                                         ensure_ascii=False)
                outf.write(json_labels + os.linesep)
            except IndexError as e:
                #traceback.print_exc()
                continue
            except Exception as e:
                #traceback.print_exc()
                #logger.error(e)
                continue

        logger.info(f"Average number of unannotated nodes: {np.mean(num_unnanotated)}")
        logger.info(f"Median number of unannotated nodes: {np.median(num_unnanotated)}")
        logger.info(f"Max number of unannotated nodes: {np.max(num_unnanotated)}")
        logger.info(f"Std number of unannotated nodes: {np.std(num_unnanotated)}")
        logger.info(f"95 percentile number of unannotated nodes: {np.percentile(num_unnanotated, 95)}")

        logger.info(f"Average percentage of unannotated nodes: {np.mean(percent_unnanotated)}")
        logger.info(f"Median percentage of unannotated nodes: {np.median(percent_unnanotated)}")
        logger.info(f"Max percentage of unannotated nodes: {np.max(percent_unnanotated)}")
        logger.info(f"Std percentage of unannotated nodes: {np.std(percent_unnanotated)}")
        logger.info(f"95 percentile percentage of unannotated nodes: {np.percentile(percent_unnanotated, 95)}")

        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(true_labels)
        y_pred = mlb.transform(pred_labels)
        recall = recall_score(y_true, y_pred, average='samples')
        logger.info(f"Recall: {recall}")


if __name__ == '__main__':
    annotate_project()
