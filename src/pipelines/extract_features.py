import glob
from os.path import join, basename
from typing import Tuple, List

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import git_clone


def get_versions(project, arcan_out) -> List[Tuple[str, str]]:
    files = [basename(x) for x in glob.glob(join(arcan_out, project, "*"))]
    res = []
    for file in files:
        num, sha = file.replace('.graphml', '').split("-")[-1].split("_")
        res.append((num, sha))
    return res


@hydra.main(config_path="../conf", config_name="extract_features", version_base="1.2")
def extract_features(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """
    embedding = instantiate(cfg.embedding)

    extraction = instantiate(cfg.extraction, model=embedding)

    projects = pd.read_csv(cfg.dataset)
    projects = projects[projects['language'] == cfg.language]
    projects = projects['full_name']

    projects = ['activej/activej']
    for project in projects:
        project_url = 'https://github.com/' + project
        project_name = project.replace('/', '-')
        git_clone(project_url, project_name, cfg.repositories_path)
        versions = get_versions(project_name, cfg.arcan_out)
        print(versions)
        for num, sha in versions:
            extraction.extract(project_name, sha=sha, num=num)


if __name__ == '__main__':
    extract_features()
