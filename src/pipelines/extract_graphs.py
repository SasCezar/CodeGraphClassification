import shutil
from os.path import join, exists
from pathlib import Path
from shlex import quote
from subprocess import call

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


def check_status(path):
    """
    Checks if the project has already been processed.
    :param path:
    :return:
    """
    return exists(path)


def run_arcan(cfg, project, language) -> None:
    """
    Runs the script to extract the graphs using Arcan. It also checks if the project has already been processed,
     and if so, it skips it.
    :param cfg:
    :param project:
    :param language:
    :return:
    """
    check_path = join(cfg.arcan_graphs, project.replace('/', '|'), '.completed')
    completed = check_status(check_path)
    try:
        if completed:
            logger.info(f"Skipping {project} as it has already been processed")
            return

        command = [cfg.arcan_script]

        args = [project, quote(project.replace('/', '|')),
                language, cfg.arcan_path, cfg.repository_path, cfg.arcan_out, join(cfg.logs_path, 'arcan')]

        command.extend(args)

        logger.info(f"Running command: {' '.join(command)}")

        call(" ".join(command), shell=True)

        if not completed:
            with open(check_path, 'wt') as outf:
                logger.info(f"Creating file {outf.name}")

        logger.info(f"Finished to extract graph for {project}")

    except Exception as e:
        logger.error(f"Failed to extract graph for {project}")
        logger.error(f"{e}")

    finally:
        if not completed:
            logger.info(f"Cleaning up {project} repository")
            repo_path = join(cfg.repository_path, project.replace('/', '|'))
            shutil.rmtree(repo_path, ignore_errors=True)
        return


@hydra.main(config_path="../conf", config_name="extract_features", version_base="1.2")
def extract_graph(cfg: DictConfig):
    """
    Extracts graph from a project including the git history (augmented data).
    :param cfg:
    :return:
    """
    projects = pd.read_csv(cfg.dataset)
    if 'language' not in projects:
        projects['language'] = [cfg.language] * len(projects)

    projects['language'] = projects['language'].str.upper()
    projects = projects[projects['language'] == cfg.language.upper()]
    languages = projects['language']
    projects = projects['full_name']

    logger.info(f"Extracting graphs for {len(projects)} projects")
    # # projects = ['pac4j/vertx-pac4j']
    # projects = ['StuyPulse/Rafael']
    # languages = ['JAVA']
    Path(join(cfg.logs_path, 'arcan')).mkdir(parents=True, exist_ok=True)
    if cfg.num_workers > 1:
        logger.info(f"Using {cfg.num_workers} workers")
        from multiprocessing import Pool
        with Pool(cfg.num_workers) as p:
            p.starmap(run_arcan, zip([cfg] * len(projects), projects, languages))
    else:
        for i, (project, language) in enumerate(zip(projects, languages)):
            logger.info(f"Extracting features for {project} - Progress: {(i + 1) / len(projects) * 100:.2f}%")
            run_arcan(cfg, project, language)


if __name__ == '__main__':
    extract_graph()
