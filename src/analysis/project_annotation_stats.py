import csv
import json
import os
from os.path import join

import hydra
from omegaconf import DictConfig

from utils import parse_settings


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def stats(cfg: DictConfig):
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

    filename = 'similarity_project_annotation_stats.csv'
    skip_header = False
    if os.path.exists(join(cfg.stats_dir, filename)):
        skip_header = True
    os.makedirs(cfg.stats_dir, exist_ok=True)
    with open(join(cfg.stats_dir, filename), 'at') as f:
        writer = csv.writer(f)
        header = ['project', 'content', 'annotation', 'algorithm', 'transformation', 'filtering', 'threshold',
                  'jsd', "nodes", "unannotated", "percent_unannotated"]
        writer.writerow(header) if not skip_header else None
        for project in annotations:
            jsd = project['jsd']
            num_nodes = project['total_nodes']
            num_unannotated = project['unannotated']
            percent_unannotated = num_unannotated / num_nodes
            writer.writerow([project['project']] + settings + [jsd, num_nodes, num_unannotated, percent_unannotated])

    print(len(annotations))


if __name__ == '__main__':
    stats()
