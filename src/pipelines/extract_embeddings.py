import traceback
from os.path import join

import hydra
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from feature.embedding import AbstractEmbeddingModel
from feature.extract import FeatureExtraction
from utils import git_clone, get_versions, git_checkout


@hydra.main(config_path="../conf", config_name="main", version_base="1.2")
def extract_embeddings(cfg: DictConfig):
    """
    :param cfg:
    :return:
    """
    embedding: AbstractEmbeddingModel = instantiate(cfg.embedding.cls)

    extraction: FeatureExtraction = instantiate(cfg.extraction.cls, model=embedding)

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
        try:
            versions = [get_versions(project_name, cfg.arcan_graphs)[-1]]
        except:
            logger.error(f"Failed to extract features for {project} {num} {sha}")
            logger.error(f"{e}")
            continue

        logger.info(f"Found {len(versions)} versions for project {project}")
        for num, sha in versions:
            try:
                if extraction.clone:
                    git_checkout(join(cfg.repositories_path, project_name), sha)
                extraction.extract(project_name, sha=sha, num=num)
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Failed to extract features for {project} {num} {sha}")
                logger.error(f"{e}")
                continue


if __name__ == '__main__':
    extract_embeddings()
