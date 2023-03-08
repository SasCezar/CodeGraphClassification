import glob
import json
from collections import defaultdict
from itertools import combinations
from os.path import join
from pathlib import Path

import hydra
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from utils import parse_settings


def load_annot(files):
    annot = defaultdict(dict)
    projects = set()
    true = {}
    for file_path, filename in files:
        with open(file_path, 'rt') as inf:
            for line in inf:
                proj = json.loads(line)
                pred = proj['predicted_labels'][0]
                annot[filename][proj['project']] = pred
                projects.add(proj['project'])
                labels = proj['true_labels']

                if proj['project'] not in true:
                    true[proj['project']] = labels

    return projects, annot, true


def load_files(path):
    def get_lf(base, path):
        return path.replace(base, '').replace(Path(path).name, '')

    files = glob.glob(path + '/**/*.json', recursive=True)
    filenames = [x for x in files if
                 'single_label' not in x and 'soft_label' not in x and
                 'ensemble' not in x]  # and ('w2v-so' in x or 'keyword' in x)]

    files = []
    lfs = set()
    for file in filenames:
        if 'label_mapping' in file:
            continue
        lf = get_lf(path, file)
        project = Path(file).name.replace('.json', '').rsplit('-', maxsplit=2)[0]
        files.append((file, lf, project))
        lfs.add(lf)

    return lfs, files


def load_mapping(annotations_dir):
    with open(join(annotations_dir, 'label_mapping.json'), 'rt') as inf:
        return json.load(inf)


def load_annot_all(files):
    annot = defaultdict(dict)
    for file_path, lf_name, name in tqdm(files):
        with open(file_path, 'rt') as inf:
            proj = json.load(inf)
            for package in proj:
                try:
                    del proj[package]['distribution']
                except TypeError as e:
                    print(proj)
            annot[lf_name][name] = proj

    return annot


@hydra.main(config_path="../conf", config_name="annotation", version_base="1.3")
def pairwise_intersections(cfg: DictConfig):
    lf, files = load_files(cfg.annotations_dir)

    pairs = list(combinations(lf, 2))
    print(len(pairs))

    lf_annot = load_annot_all(files)
    res = Parallel(n_jobs=10, prefer="processes")(delayed(lf_intersections)(a, b, lf_annot) for a, b in tqdm(pairs))
    conflicts = pd.concat(res)

    conflicts.to_csv('node_annot_intersection.csv', index=False)


def node_intersection(proj_a, proj_b):
    all_nodes = set(proj_a.keys()).union(proj_b.keys())
    # common_nodes = set(proj_a.keys()).intersection(proj_b.keys())
    annotated = 0

    for file in all_nodes:
        if bool(file in proj_a and proj_a[file]['unannotated']) \
                ^ bool(file in proj_b and proj_b[file]['unannotated']):
            annotated += 1

    return annotated / len(all_nodes) * 100


def lf_intersections(a, b, lf_annot):
    intersections = pd.DataFrame(
        columns=['x_annotation', 'x_content', 'x_algorithm', 'x_transformation', 'x_filtering', 'x_threshold',
                 'y_annotation', 'y_content', 'y_algorithm', 'y_transformation', 'y_filtering', 'y_threshold',
                 'project', 'intersection_percent'])

    projs = set(lf_annot[a].keys()).union(lf_annot[b].keys())
    sett_a = parse_settings(a.strip('/'))
    sett_b = parse_settings(b.strip('/'))

    for p in tqdm(projs, position=0, leave=True):
        intersection_percent = 0
        if p in lf_annot[a] and p in lf_annot[b]:
            a_annot = lf_annot[a][p]
            b_annot = lf_annot[b][p]  # join(b[0], f'{p}.json')
            if len(a_annot) or len(b_annot):
                intersection_percent = node_intersection(a_annot, b_annot)

        row = sett_a + sett_b + [p, intersection_percent]
        intersections.loc[len(intersections)] = row

    return intersections


if __name__ == '__main__':
    pairwise_intersections()
