from os.path import join
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


def aggregate_project_stats(cfg):
    filename = 'similarity_project_annotation_stats.csv'
    stats_file = Path(join(cfg.stats_dir, filename))

    df = pd.read_csv(stats_file)
    df = df.groupby(
        ['annotation', 'content', 'algorithm', 'transformation', 'filtering', 'threshold', 'percent_unannotated']) \
        .aggregate({'jsd': ['mean', 'std'], 'percent_unannotated': ['mean', 'std']})

    print(df.head(10))


def aggregate_project_metric(cfg):
    filename = 'similarity_project_annotation_metrics.csv'
    stats_file = Path(join(cfg.stats_dir, filename))

    df = pd.read_csv(stats_file)

    return df


def aggregate_package_stats(cfg):
    filename = 'similarity_package_annotation_stats.csv'
    stats_file = Path(join(cfg.stats_dir, filename))

    df = pd.read_csv(stats_file)
    df = df.groupby(['annotation', 'content', 'algorithm', 'transformation', 'filtering', 'threshold', 'clean']) \
        .aggregate({'jsd': ['mean', 'std'], 'cohesion': ['mean', 'std'], 'unannotated': ['mean', 'std']})
    df = df.round(2)

    jsd = df['jsd']['mean'].astype(str) + '$\pm$' + df['jsd']['std'].astype(str)
    cohesion = df['cohesion']['mean'].astype(str) + '$\pm$' + df['cohesion']['std'].astype(str)
    unannotated = df['unannotated']['mean'].astype(str) + '$\pm$' + df['unannotated']['std'].astype(str)
    df.drop('jsd', inplace=True, axis=1)
    df.drop('cohesion', inplace=True, axis=1)
    df.drop('unannotated', inplace=True, axis=1)
    df['jsd'] = jsd
    df['cohesion'] = cohesion
    df['unannotated'] = unannotated
    # df.drop('level_1', inplace=True)
    print(df.unstack(-1).transpose().reset_index().to_latex(float_format="{:0.2f}".format,
                                                            index=False,
                                                            escape=False,
                                                            sparsify=True,
                                                            multirow=True,
                                                            multicolumn=True,
                                                            multicolumn_format='c',
                                                            position='htbp'))


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def aggregate_res(cfg: DictConfig):
    """
    Computes the JSD distribution for the projects in the dataset.
    :param cfg:
    :return:
    """

    # project_stats = aggregate_project_stats(cfg)
    # project_metric = aggregate_project_metric(cfg)
    package_stats = aggregate_package_stats(cfg)


if __name__ == '__main__':
    aggregate_res()
