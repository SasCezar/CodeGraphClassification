from os.path import join
from subprocess import call

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="extract_features", version_base="1.2")
def extract_graph(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """

    projects = pd.read_csv(cfg.dataset)
    if 'language' not in projects:
        projects['language'] = cfg.language * len(projects)

    projects = projects[projects['language'].str.upper() == cfg.language]
    languages = projects['language'].str.upper()
    projects = projects['full_name']

    logger.info(f"Extracting graphs for {len(projects)} projects")

    for i, (project, language) in enumerate(zip(projects, languages)):
        logger.info(f"Extracting features for {project} - Progress: {i / len(projects) * 100:.2f}%")
        try:
            command = [cfg.arcan_script, project, project.replace('/', '|'),
                       language, cfg.arcan_path, cfg.repository_path, cfg.arcan_out, join(cfg.logs_path, 'arcan')]
            logger.info(f"Running command: {' '.join(command)}")
            call(command)
        except Exception as e:
            logger.error(f"Failed to extract graph for {project}")
            logger.error(f"{e}")
            continue


if __name__ == '__main__':
    extract_graph()
