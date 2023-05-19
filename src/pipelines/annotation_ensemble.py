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
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for lf in ensemble_functions:
        project_annotations = glob.glob(f'{lf}/*.json')
        for project_path in project_annotations:
            project = project_path.replace(lf, '').strip('/')
            with open(project_path, 'rt') as inf:
                file_annotations = json.load(inf)
            # for node in file_annotations:
            #    res[project][node].append(file_annotations[node]['distribution'])
            for node in file_annotations:
                res[project][node][lf] = {'distribution': file_annotations[node]['distribution'],
                                          'unannotated': file_annotations[node]['unannotated']}


def load_project_annot(lf_functions, project):
    res = defaultdict(lambda: defaultdict(dict))
    for lf in lf_functions:
        annot_path = Path(join(lf, project))
        if annot_path.exists():
            with open(annot_path, 'rt') as inf:
                file_annotations = json.load(inf)
            for node in file_annotations:
                res[node][lf] = {'distribution': file_annotations[node]['distribution'],
                                 'unannotated': file_annotations[node]['unannotated']}

    return res


def get_projects(lf_functions):
    res = set()
    for lf in lf_functions:
        projs = [project_path.replace(lf, '').strip('/') for project_path in glob.glob(f'{lf}/*.json')]
        res.update(projs)
    return res


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.2")
def node_ensemble(cfg: DictConfig):
    """
    :param cfg:
    :return:
    """

    lf_functions = cfg.ensemble.labelling_functions
    ensemble = instantiate(cfg.ensemble.cls)

    projects = get_projects(lf_functions)
    for project in tqdm(projects):
        try:
            annotations = load_project_annot(lf_functions, project)
            labels = compute_node_labels(annotations, ensemble)
            out_path = Path(
                join(cfg.annotations_dir, 'ensemble', *cfg.ensemble.name.split('/'), 'none', 'none', project))
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, 'w') as f:
                json.dump(labels, f, ensure_ascii=False)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Failed to extract features for {project}")
            logger.error(f"{e}")
            continue


def compute_node_labels(annotations, ensemble):
    node_labels = {}
    for node in annotations:
        vec, unannotated = ensemble.annotate(annotations[node])
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm

        node_labels[node] = {'distribution': list(vec), "unannotated": unannotated}

    return node_labels


if __name__ == '__main__':
    node_ensemble()
