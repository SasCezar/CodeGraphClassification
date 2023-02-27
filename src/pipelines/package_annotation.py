import json
import os
import traceback
from collections import defaultdict
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
    skipped_packages = 0
    for package in package_files_map:
        clean_nodes_dist = []
        num_unannotated = 0
        total = 0
        for file in package_files_map[package]:
            if file not in annotations:
                skipped += 1
                continue
            total += 1

            if not annotations[file]['unannotated'] and norm(annotations[file]['distribution']):
                clean_nodes_dist.append(annotations[file]['distribution'])
            else:
                num_unannotated += 1

        if total and clean_nodes_dist:
            mean_clean = np.array(clean_nodes_dist).mean(axis=0)
            package_annotations[package]['clean_nodes'] = clean_nodes_dist
            package_annotations[package]['clean_distribution'] = list(mean_clean)
            package_annotations[package]['total'] = total
            package_annotations[package]['num_unannotated'] = num_unannotated
            package_annotations[package]['percent_unannotated'] = num_unannotated / total
            package_annotations[package]['unannotated'] = 0
        else:
            package_annotations[package]['unannotated'] = 1
            skipped_packages += 1

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

    scores = [x for x in scores if np.isfinite(x)]

    return scores


def annotation_cohesion(package_annotations):
    res = {}
    for package in package_annotations:
        if package_annotations[package]['unannotated']:
            res[package] = {'clean_package_cohesion': 0,
                            'all_package_cohesion': 0}
            continue

        clean_annot = package_annotations[package]['clean_nodes']

        if len(clean_annot) > 2:
            clean_jsd = pairwise_jsd(clean_annot)
            clean_mean = np.mean(clean_jsd)
            all_mean = sum(clean_jsd) / package_annotations[package]['total']
        else:
            clean_mean = 1 / package_annotations[package]['total']
            all_mean = 1 / package_annotations[package]['total']

        if all_mean > 1:
            print('Mean', all_mean)
            print('Annot', clean_annot)
            print('Total', package_annotations[package]['total'])

        if not np.isfinite(clean_mean):
            clean_mean = 0
        if not np.isfinite(all_mean):
            all_mean = 0

        res[package] = {'clean_package_cohesion': clean_mean,
                        'all_package_cohesion': all_mean}

    return res


def package_jsd(package_annotations):
    n = 267
    uniform_dist = np.ones(n) / n
    res = {}
    clean_dist = []
    clean_packages = {}

    for package in package_annotations:
        if not package_annotations[package]['unannotated']:
            clean_annot = package_annotations[package]['clean_distribution']
            clean_packages[package] = len(clean_dist)
            clean_dist.append(clean_annot)

    if clean_dist:
        uniform = [uniform_dist] * len(clean_dist)
        jsd_clean = jensenshannon(clean_dist, uniform, axis=1)
        assert len(jsd_clean) == len(clean_dist)

    for package in package_annotations:
        if package in clean_packages:
            clean_score = jsd_clean[clean_packages[package]]
        else:
            clean_score = 0
        if not numpy.isfinite(clean_score):
            clean_score = -1
        res[package] = {"jsd_clean": clean_score}

    return res


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.3")
def package_annotation(cfg: DictConfig):
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
                jsd = package_jsd(package_annotations) if package_annotations else defaultdict(lambda: {"jsd_clean": 0})

                pack = defaultdict(lambda: defaultdict(object))
                for package in cohesion:
                    pack[package].update(cohesion[package])
                    pack[package].update(jsd[package])
                    pack[package]['percent_unannotated'] = package_annotations[package]['percent_unannotated']
                    pack[package]['clean_distribution'] = package_annotations[package]['clean_distribution']
                    pack[package]['num_nodes'] = package_annotations[package]['total']
                    pack[package]['unannotated'] = package_annotations[package]['unannotated']

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
    package_annotation()
