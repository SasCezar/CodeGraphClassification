import csv
import json
import os
from os.path import join
from pathlib import Path

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
    annotation_path = join(cfg.package_labels_path, "annotations.json")

    filename = 'similarity_package_annotation_stats.csv'
    skip_header = False
    if os.path.exists(join(cfg.stats_dir, filename)):
        skip_header = True
    out_path = Path(join(cfg.stats_dir))
    out_path.mkdir(exist_ok=True, parents=True)
    rows = 0
    with open(join(out_path, filename), 'at') as f:
        writer = csv.writer(f)
        header = ['project', 'annotation', 'content', 'algorithm', 'transformation', 'filtering', 'threshold',
                  'package', 'jsd', 'cohesion', 'cohesion_all', 'num_nodes', 'unannotated_nodes', 'unannotated']

        writer.writerow(header) if not skip_header else None
        with open(annotation_path, 'rt') as f:
            for line in f:
                project = json.loads(line)
                rows += 1
                for package in project['packages']:
                    row = [package]
                    row.append(project['packages'][package]['jsd_clean'])

                    row.append(project['packages'][package]['clean_package_cohesion'])
                    row.append(project['packages'][package]['all_package_cohesion'])

                    row.append(project['packages'][package]['num_nodes'])
                    row.append(project['packages'][package]['percent_unannotated'])
                    row.append(project['packages'][package]['unannotated'])

                    writer.writerow([project['project']] + settings + row)

    print(rows)


if __name__ == '__main__':
    stats()
