import glob
import json
import traceback
from collections import defaultdict
from os.path import join
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


def load_lf_annotations(ensemble_functions):
    res = defaultdict(lambda: defaultdict(list))
    for lf in ensemble_functions:
        project_annotations = glob.glob(f'{lf}/*.json')
        for project_path in project_annotations:
            project = project_path.replace(lf, '').strip('/')
            with open(project_path, 'rt') as inf:
                file_annotations = json.load(inf)
            for node in file_annotations:
                res[project][node].append(file_annotations[node]['distribution'])

    return res


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def node_ensemble(cfg: DictConfig):
    """
    Extracts features from a project including the git history (augmented data).
    :param cfg:
    :return:
    """

    lf_functions = cfg.ensemble.labelling_functions
    transformation = instantiate(cfg.transformation.cls) if cfg.transformation.cls else None
    filtering = instantiate(cfg.filtering.cls) if cfg.filtering.cls else None

    annotations = load_lf_annotations(lf_functions)
    projects = annotations.keys()
    for project in tqdm(projects):
        try:
            labels = compute_node_labels(annotations[project], transformation, filtering)
            out_path = Path(join(cfg.annotations_dir, 'ensemble', 'best', project))
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, 'w') as f:
                json.dump(labels, f, ensure_ascii=False)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Failed to extract features for {project}")
            logger.error(f"{e}")
            continue


def compute_node_labels(annotations, transform, filtering):
    node_labels = {}
    for node in annotations:
        vec = np.max(annotations[node], axis=0)
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        unannotated = 0

        if filtering:
            unannotated = filtering.filter(vec)

        if transform:
            vec = transform.transform(vec)

        node_labels[node] = {'distribution': list(vec), "unannotated": unannotated}

    return node_labels


if __name__ == '__main__':
    node_ensemble()
