import json
import traceback
from os.path import join
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from feature.content import JSONContentExtraction
from utils import git_clone, get_versions, git_checkout


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def node_annotation(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """
    content_extractor = JSONContentExtraction(cfg.content_dir)

    projects = pd.read_csv(cfg.dataset)

    projects = projects[projects['language'].str.upper() == cfg.language.upper()]
    projects = projects['full_name']

    logger.info(f"Extracting features for {len(projects)} projects")

    annotation = instantiate(cfg.node_annotation.cls) if cfg.node_annotation.cls else None
    transformation = instantiate(cfg.transformation.cls) if cfg.transformation.cls else None
    filtering = instantiate(cfg.filtering.cls) if cfg.filtering.cls else None

    for project in tqdm(projects):

        project_name = project.replace('/', '|')
        if content_extractor.clone:
            project_url = f'https://github.com/{project}'
            git_clone(project_url, project_name, cfg.repositories_path)

        try:
            num, sha = get_versions(project_name, cfg.arcan_graphs)[-1]
        except IndexError:
            logger.warning(f"Could not find a version for {project}")
            continue

        try:
            pname = f"{project_name}-{num}-{sha}"

            if content_extractor.clone:
                git_checkout(join(cfg.repositories_path, project_name), sha)

            project_content = dict(content_extractor.extract(project_name, sha, num))
            labels = compute_node_labels(project_content, annotation, transformation, filtering)

            out_path = Path(join(cfg.annotations_path, f"{pname}.json"))
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, 'w') as f:
                json.dump(labels, f, ensure_ascii=False)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Failed to extract features for {project} {num} {sha}")
            logger.error(f"{e}")
            continue


def compute_node_labels(contents, annotation, transform, filtering):
    node_labels = {}
    for node, content in contents.items():
        vec = annotation.annotate(node, content)
        unannotated = 0

        if filtering:
            unannotated = filtering.filter(vec)

        if transform:
            vec = transform.transform(vec)


        node_labels[node] = {'distribution': [float(x) for x in vec], "unannotated": unannotated}

    return node_labels


if __name__ == '__main__':
    node_annotation()
