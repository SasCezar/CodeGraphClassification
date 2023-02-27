from os.path import join
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def stats(cfg: DictConfig):
    """
    Computes the JSD distribution for the projects in the dataset.
    :param cfg:
    :return:
    """

    filename = 'similarity_package_annotation_stats.csv'

    path = Path(join(cfg.stats_dir, filename))

    df = pd.read_csv(path)
    df.


if __name__ == '__main__':
    stats()
