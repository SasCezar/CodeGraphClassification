import json
import os
import traceback
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from os.path import join
from pathlib import Path

import hydra
import numpy
import numpy as np
import pandas as pd
from loguru import logger
from numpy.linalg import norm
from omegaconf import DictConfig
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from data.graph import ArcanGraphLoader
from utils import get_versions


def load_annotations(annotations_path):
    with open(annotations_path, 'rt') as inf:
        obj = json.load(inf)

    return obj


def load_package_file(graph, annotations):
    nodes_id = [x for x in graph.vs if x['filePathRelative'] in annotations if
                graph.vs[x.index]['filePathRelative'] != '.']
    package_files = defaultdict(list)
    for node in nodes_id:
        packages = [graph.vs[x]['name'] for x in graph.neighbors(node.index) if
                    graph.vs[x]['labelV'] == 'container']
        for package in packages:
            package_files[package].append(graph.vs[node.index]['filePathRelative'])

    return package_files


def annotate(annotations, package_files_map):
    package_annotations = defaultdict(lambda: defaultdict(list))
    skipped = 0
    n = 267
    skipped_packages = 0
    for package in package_files_map:
        all_package_annotations = []
        clean_package_annotations = []
        num_unannotated = 0
        total = 0
        for file in package_files_map[package]:
            if file not in annotations:
                skipped += 1
                continue
            all_package_annotations.append(annotations[file]['distribution'])
            total += 1
            if not annotations[file]['unannotated'] and norm(annotations[file]['distribution']):
                clean_package_annotations.append(annotations[file]['distribution'])
            else:
                num_unannotated += 1

        if total:
            mean_all = np.array(all_package_annotations).mean(axis=0) if all_package_annotations else np.zeros(n)
            mean_clean = np.array(clean_package_annotations).mean(axis=0) if clean_package_annotations else np.zeros(n)

            package_annotations[package]['clean_nodes'] = all_package_annotations
            package_annotations[package]['all_nodes'] = clean_package_annotations
            package_annotations[package]['clean_distribution'] = list(mean_clean)
            package_annotations[package]['all_distribution'] = list(mean_all)
            package_annotations[package]['num_unannotated'] = num_unannotated
            package_annotations[package]['percent_unannotated'] = num_unannotated / total
        else:
            skipped_packages += 1

    if skipped_packages == len(package_files_map):
        print('Error')

    return package_annotations


def pairwise_jsd(annot):
    pairs = list(combinations(annot, 2))
    x = []
    y = []
    scores = []
    for a, b in pairs:
        x.append(a)
        y.append(b)
        if len(x) > 500:
            scores.extend(list(jensenshannon(x, y, axis=1)))
            x = []
            y = []

    if x:
        scores.extend(list(jensenshannon(x, y, axis=1)))

    scores = [x if np.isfinite(x) else -1 for x in scores]

    return scores


def annotation_cohesion(package_annotations):
    res = {}
    for package in package_annotations:

        all_annot = package_annotations[package]['all_nodes']
        clean_annot = package_annotations[package]['clean_nodes']
        all_mean = -1
        if len(all_annot) > 2:
            all_jsd_dist = pairwise_jsd(all_annot)
            all_mean = np.ma.masked_invalid(all_jsd_dist).mean()
            if type(all_mean) == numpy.ma.core.MaskedConstant:
                all_mean = -1

        clean_mean = -1
        if len(clean_annot) > 2:
            clean_jsd_dist = pairwise_jsd(clean_annot)
            clean_mean = np.ma.masked_invalid(clean_jsd_dist).mean()
            if type(clean_mean) == numpy.ma.core.MaskedConstant:
                clean_mean = -1

        res[package] = {'clean_package_cohesion': clean_mean,
                        'all_package_cohesion': all_mean}

    return res


def package_jsd(package_annotations):
    test = deepcopy(package_annotations)
    n = len(test.popitem()[1]['all_distribution'])
    uniform_dist = np.ones(n) / n
    res = {}
    clean_package = []
    all_package = []
    for package in package_annotations:
        clean_annot = package_annotations[package]['clean_distribution']
        clean_package.append(clean_annot)
        all_annot = package_annotations[package]['all_distribution']
        all_package.append(all_annot)

    jsd_clean = jensenshannon(clean_package, [uniform_dist] * len(clean_package), axis=1)
    jsd_all = jensenshannon(all_package, [uniform_dist] * len(all_package), axis=1)
    assert len(jsd_all) == len(all_package)

    for i, package in enumerate(package_annotations):
        clean_score = jsd_clean[i]
        all_score = jsd_all[i]
        if not numpy.isfinite(clean_score):
            clean_score = -1
        if not numpy.isfinite(all_score):
            clean_score = -1
        res[package] = {"jsd_clean": clean_score, "jsd_all": all_score}

    return res


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.3")
def annotate_package(cfg: DictConfig):
    projects = pd.read_csv(cfg.dataset)

    projects = projects[projects['language'].str.upper() == cfg.language.upper()]

    projects = projects['full_name']
    skipped = 0

    logger.info(f"Extracting features for {len(projects)} projects")

    Path(cfg.package_labels_path).mkdir(parents=True, exist_ok=True)
    with open(join(cfg.package_labels_path, "annotations.json"), 'wt') as outf:
        for project in tqdm(projects):
            try:
                project_name = project.replace('/', '|')
                num, sha = get_versions(project_name, cfg.arcan_graphs)[-1]

                annotations_path = join(cfg.annotations_path, f"{project_name}-{num}-{sha}.json")
                annotations = load_annotations(annotations_path)
                if not annotations:
                    logger.warning(f'Skipped project: {project_name} - {num} - {sha}')
                    skipped += 1
                    continue

                graph = ArcanGraphLoader().load(
                    join(cfg.arcan_graphs, project_name, f"dependency-graph-{num}_{sha}.graphml"))
                package_files_map = load_package_file(graph, annotations)
                package_annotations = annotate(annotations, package_files_map)

                # res = {'project': project_name, 'num': num, 'sha': sha, 'packages': package_annotation}
                assert len(package_annotations)
                cohesion = annotation_cohesion(package_annotations)
                jsd = package_jsd(package_annotations)

                pack = {}
                for package in cohesion:
                    pack[package] = cohesion[package]
                    pack[package].update(jsd[package])
                    pack[package]['percent_unannotated'] = package_annotations[package]['percent_unannotated']
                    pack[package]['clean_distribution'] = package_annotations[package]['clean_distribution']
                    pack[package]['all_distribution'] = package_annotations[package]['all_distribution']

                res = {'project': project_name, 'num': num, 'sha': sha, 'packages': pack}

                line = json.dumps(res, ensure_ascii=False)
                outf.write(line + os.linesep)

            except IndexError as e:
                continue
            except Exception as e:
                print(traceback.format_exc())
                continue

    print('Skipped', skipped)


if __name__ == '__main__':
    annotate_package()
