import csv
import json
import os
from collections import defaultdict
from os.path import join

import hydra
from omegaconf import DictConfig
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def parse_settings(settings):
    config = settings.split('/')
    if len(config) == 5:
        config.append('0')

    return config[-3:]


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def metrics(cfg: DictConfig):
    """
    Computes the JSD distribution for the projects in the dataset.
    :param cfg:
    :return:
    """

    settings = parse_settings(cfg.settings)

    annotation_path = join(cfg.project_labels_dir, "annotations.json")
    annotations = []
    with open(annotation_path, 'rt') as f:
        for line in f:
            annotations.append(json.loads(line))

    skip_header = False
    if os.path.exists(join(cfg.stats_dir, 'project_annotation_metrics.csv')):
        skip_header = True
    os.makedirs(cfg.stats_dir, exist_ok=True)

    depth = [3, 5, 10]

    true_labels = []
    pred_labels = defaultdict(list)
    for project in annotations:
        true_labels.append(project['true_labels'])
        for i in depth:
            pred_labels[i].append(project['predicted_labels'][:i])

    with open(join(cfg.stats_dir, 'project_annotation_metrics.csv'), 'at') as f:
        writer = csv.writer(f)
        header = ['transformation', 'filtering', 'threshold', 'metric', 'at', 'value']
        writer.writerow(header) if not skip_header else None

        for i in depth:
            mlb = MultiLabelBinarizer()
            y_true = mlb.fit_transform(true_labels)
            y_pred = mlb.transform(pred_labels[i])
            precision = precision_score(y_true, y_pred, average='samples')
            recall = recall_score(y_true, y_pred, average='samples')
            writer.writerow(settings + ['precision', i, precision])
            writer.writerow(settings + ['recall', i, recall])


if __name__ == '__main__':
    metrics()
