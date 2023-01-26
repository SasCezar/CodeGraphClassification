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
    annotation_path = join(cfg.package_labels_dir, "annotations.json")
    annotations = []
    with open(annotation_path, 'rt') as f:
        for line in f:
            annotations.append(json.loads(line))

    filename = 'similarity_package_annotation_stats.csv'
    skip_header = False
    if os.path.exists(join(cfg.stats_dir, filename)):
        skip_header = True
    os.makedirs(cfg.stats_dir, exist_ok=True)
    with open(join(cfg.stats_dir, filename), 'at') as f:
        writer = csv.writer(f)
        header = ['project', 'content', 'annotation', 'algorithm', 'transformation', 'filtering', 'threshold',
                  'package', 'clean', 'jsd', 'cohesion', 'unannotated']

        writer.writerow(header) if not skip_header else None
        for project in annotations:
            for package in project['packages']:
                all_row = [package, '1']
                clean_row = [package, '0']
                all_row.append(project['packages'][package]['jsd_all'])
                clean_row.append(project['packages'][package]['jsd_clean'])

                all_row.append(project['packages'][package]['all_package_cohesion'])
                clean_row.append(project['packages'][package]['clean_package_cohesion'])

                all_row.append(project['packages'][package]['percent_unannotated'])
                clean_row.append(project['packages'][package]['percent_unannotated'])

                writer.writerow([project['project']] + settings + all_row)
                writer.writerow([project['project']] + settings + clean_row)

    print(len(annotations))


if __name__ == '__main__':
    stats()
