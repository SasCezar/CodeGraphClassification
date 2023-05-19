from glob import glob
from os.path import join
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from utils import git_clone, get_versions, git_checkout


@hydra.main(config_path="../conf", config_name="main", version_base="1.2")
def repo2vec(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """
    embedding = instantiate(cfg.embedding.cls)

    extraction = instantiate(cfg.extraction.cls, model=embedding)

    projects = pd.read_csv(cfg.dataset)

    projects = projects[projects['language'].str.upper() == cfg.language.upper()]
    projects = projects['full_name']

    # projects = ['pac4j/vertx-pac4j']

    logger.info(f"Extracting features for {len(projects)} projects")
    for project in projects:
        logger.info(f"Extracting features for {project}")

        project_name = project.replace('/', '|')
        if extraction.clone:
            project_url = f'https://github.com/{project}'
            git_clone(project_url, project_name, cfg.repositories_path)
        versions = get_versions(project_name, cfg.arcan_graphs)

        logger.info(f"Found {len(versions)} versions for project {project}")
        for num, sha in versions:
            try:
                if extraction.clone:
                    git_checkout(join(cfg.repositories_path, project_name), sha)
                extraction.extract(project_name, sha=sha, num=num)
            except Exception as e:
                logger.error(f"Failed to extract features for {project} {num} {sha}")
                logger.error(f"{e}")
                continue


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def run(cfg: DictConfig):
    """
    Extracts graph from a project including the git history (augmented data).
    :param cfg:
    :return:
    """

    """
    Load all projects json from content_dir
    """
    projects = glob(join(cfg.content_dir, '*.json'))

    Path(join(cfg.repo2vec_out, 'repo2vec')).mkdir(parents=True, exist_ok=True)
    if cfg.num_workers > 1:
        logger.info(f"Using {cfg.num_workers} workers")
        from multiprocessing import Pool
        with Pool(cfg.num_workers) as p:
            p.starmap(repo2vec, zip([cfg] * len(projects), projects))
    else:
        for i, project in enumerate(projects):
            logger.info(f"Extracting features for {project} - Progress: {(i + 1) / len(projects) * 100:.2f}%")
            repo2vec(cfg, project)


if __name__ == '__main__':
    run()
