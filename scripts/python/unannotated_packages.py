from os.path import join
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from pandas import DataFrame


@hydra.main(config_path="../../src/conf", config_name="annotation", version_base="1.2")
def stats(cfg: DictConfig):
    """
    Computes the JSD distribution for the projects in the dataset.
    :param cfg:
    :return:
    """

    filename = 'similarity_package_annotation_stats.csv'

    path = Path(join(cfg.stats_dir, filename))

    df: DataFrame = pd.read_csv(path)
    # project,annotation,content,algorithm,transformation,filtering,threshold,package,jsd,cohesion,num_nodes,unannotated_nodes,unannotated
    df = df.groupby(['project', 'annotation', 'content', 'algorithm', 'transformation', 'filtering', 'threshold'])[
        'unannotated'].mean()
    df = df.reset_index()
    df.to_csv('unannotated_avg.csv', index=False)


if __name__ == '__main__':
    stats()
