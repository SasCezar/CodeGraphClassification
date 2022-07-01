import hydra
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from utils import git_clone, get_versions


@hydra.main(config_path="../conf", config_name="extract_features", version_base="1.2")
def extract_embeddings(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """
    embedding = instantiate(cfg.embedding)

    extraction = instantiate(cfg.extraction, model=embedding)

    projects = pd.read_csv(cfg.dataset)
    projects = projects[projects['language'].str.lower() == cfg.language]
    projects = projects['full_name']

    projects = ['activej/activej']

    logger.info(f"Extracting features for {len(projects)} projects")
    for project in projects:
        logger.info(f"Extracting features for {project}")

        project_url = 'https://github.com/' + project
        project_name = project.replace('/', '|')
        git_clone(project_url, project_name, cfg.repositories_path)
        versions = get_versions(project_name, cfg.arcan_out)

        logger.info(f"Found {len(versions)} versions for project {project}")
        for num, sha in versions:
            try:
                extraction.extract(project_name, sha=sha, num=num)
            except Exception as e:
                logger.error(f"Failed to extract features for {project} {num} {sha}")
                logger.error(f"{e}")
                continue


if __name__ == '__main__':
    extract_embeddings()