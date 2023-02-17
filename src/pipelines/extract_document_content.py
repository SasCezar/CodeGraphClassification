import json
import os
import traceback
from os.path import join

import hydra
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from feature.content import ContentExtraction
from utils import git_clone, get_versions, git_checkout


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def extract_content(cfg: DictConfig):
    """
    :param cfg:
    :return:
    """

    projects = list(set(pd.read_csv(cfg.project_path)['name'].tolist()))
    content_extractor: ContentExtraction = instantiate(cfg.content.cls)

    os.makedirs(cfg.content_dir, exist_ok=True)
    for project_name in projects:
        project = project_name.replace('|', '/')
        if content_extractor.clone:
            project_url = f'https://github.com/{project}'
            git_clone(project_url, project_name, cfg.repositories_path)

        versions = [get_versions(project_name, cfg.arcan_graphs)[-1]]
        logger.info(f"Found {len(versions)} versions for {project_name}")

        with open(f"{cfg.content_dir}/{project_name}.json", 'wt') as f:
            for num, sha in versions:
                try:
                    if content_extractor.clone:
                        git_checkout(join(cfg.repositories_path, project_name), sha)
                    content = list(content_extractor.extract(project_name, sha, num))
                    res = {x[0]: x[1] for x in content if x[1]}
                    content = {"project": project, "num": num, "sha": sha, "content": res}
                    row = json.dumps(content, ensure_ascii=False)
                    f.write(row + os.linesep)
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error in {project_name} {sha} {num}: {e}")
                    continue


if __name__ == '__main__':
    extract_content()
